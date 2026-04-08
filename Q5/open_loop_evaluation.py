"""
离线开环评估脚本（LeRobot + Diffusion Policy）
================================================

一、脚本目标
------------
本脚本用于“纯离线”地评估一个 LeRobot Diffusion Policy：
1) 从 LeRobot 数据集中读取指定 episode 的观测和真值动作；
2) 逐帧将观测送入策略，得到该帧预测动作；
3) 计算整条 episode 上的整体 MSE；
4) 画出每个关节（动作维度）“GT vs 预测”的对比曲线图。

注意：这里是“开环（open-loop）评估”，即只回放离线数据并推理，不与真实机器人或仿真环境交互。

二、为什么脚本这样实现
----------------------
1) 为什么使用 `select_action`：
   在本仓库的 diffusion/modeling_diffusion.py 中，DiffusionPolicy 对外推理接口是
   `select_action(batch)`，并且它内部维护观测/动作队列（含 n_obs_steps、n_action_steps 逻辑）。
   因此离线评估时应调用 `select_action`，而不是虚构或替换成其他接口。

2) 为什么外部显式构建 preprocessor / postprocessor：
   在本仓库官方脚本（如 scripts/lerobot_eval.py、scripts/lerobot_train.py）中，推理链路是：
   observation -> preprocessor -> policy.select_action -> postprocessor。
   即归一化/反归一化属于 processor pipeline，而不是在本脚本里手写私有逻辑。
   这样做可以最大程度与训练/官方评测流程一致，避免输入输出尺度不一致。

3) 为什么保留 `preprocess_observation`：
   这是“进入官方 preprocessor 之前”的基础格式整理层，职责仅为：
   - 保证 numpy 转 torch 前为 contiguous，避免负 stride 等问题；
   - 将 HWC 图像转为 CHW（与视觉模型常规输入一致）；
   - uint8 图像先缩放到 [0,1] 浮点范围，保持数值稳定。
   真正的规范化（比如 min-max / mean-std）仍交给官方 NormalizerProcessorStep。

三、如何从 LeRobot 源码核验 API 正确性（建议流程）
------------------------------------------------
你可以按下面的“源码定位路径”逐条验证本脚本的关键 API 没有编造：

1) 推理接口核验：
   - 文件：diffusion/modeling_diffusion.py
   - 关键词：`class DiffusionPolicy`、`def select_action`
   - 结论：官方推理入口是 `select_action`。

2) Processor 核验：
   - 文件：diffusion/processor_diffusion.py
   - 关键词：`make_diffusion_pre_post_processors`
   - 结论：官方定义了 NormalizerProcessorStep 与 UnnormalizerProcessorStep 的处理顺序。

3) 官方脚本链路核验：
   - 文件：scripts/lerobot_eval.py
   - 关键词：`observation = preprocessor(observation)`、`action = policy.select_action(...)`、`action = postprocessor(action)`
   - 结论：评测流程是“前处理 -> 模型 -> 后处理”。

4) 训练阶段 stats 来源核验：
   - 文件：scripts/lerobot_train.py
   - 关键词：`dataset.meta.stats`、`make_pre_post_processors`
   - 结论：归一化参数来源于数据集统计量 dataset.meta.stats。

四、示例命令
------------
python open_loop_evaluation.py \
  --repo-id PengTianxiang/20260319First \
  --policy-path PengTianxiang/20260321diffusion_run7_060000 \
  --episode 21 \
  --output-png outputs/ep21_open_loop.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import ACTION


def preprocess_observation(observation: dict[str, torch.Tensor | np.ndarray]) -> dict[str, torch.Tensor]:
    """
    对单帧观测做“轻量且必要”的张量化与形状整理。

    这里不做策略语义层面的归一化（那是官方 preprocessor 的职责），
    只做基础数据工程工作，确保后续 pipeline 稳定：
    - numpy -> torch；
    - HWC -> CHW（仅针对看起来像图像的 3 维张量）；
    - uint8 -> float32，并缩放至 [0,1]；
    - 补 batch 维，适配后续策略接口的 batched 输入约定。
    """
    processed: dict[str, torch.Tensor] = {}
    for key, value in observation.items():
        # 1) 保持输入兼容：既支持 dataset 返回 torch.Tensor，也支持 numpy.ndarray。
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            # np.ascontiguousarray 的目的是避免负 stride / 非连续内存导致 torch.from_numpy 报错。
            tensor = torch.from_numpy(np.ascontiguousarray(value))

        # 2) 图像维度转换：
        #    当张量为 3 维且最后一维像通道数（1/3/4）时，按 HWC -> CHW 处理。
        #    这与 reference2 的 HWC -> CHW 思路一致。
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)

        # 3) dtype 统一：
        #    - 若是 uint8（典型图像），先转 float 并除以 255；
        #    - 其他类型直接转 float，保持与模型推理常用 dtype 一致。
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        else:
            tensor = tensor.float()

        # 4) contiguous + batch 维：
        #    官方 processor/policy 都按 batched 样式处理，这里统一加上 batch 维 (B=1)。
        processed[key] = tensor.contiguous().unsqueeze(0)
    return processed


def plot_joint_curves(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    episode_index: int,
    mse: float,
    output_png: Path,
    action_names: list[str] | None = None,
) -> None:
    """
    绘制“每个动作维度一张子图”的 GT / 预测对比曲线。

    图像风格尽量贴合你给出的目标示意：
    - GT：灰黑实线；
    - Prediction：红色虚线；
    - 标题包含 episode 与整体 MSE；
    - 横轴统一 Time Step，纵轴为对应关节（动作维度）值。
    """
    action_dim = gt_actions.shape[1]
    # 若数据集里拿不到动作维度名字，则回退为 Dim 0/1/2...
    if action_names is None or len(action_names) != action_dim:
        action_names = [f"Dim {i}" for i in range(action_dim)]

    # 每个维度一行子图，便于排查“哪一维误差大”。
    fig, axes = plt.subplots(action_dim, 1, figsize=(16, max(4 * action_dim, 6)), sharex=True)
    if action_dim == 1:
        axes = [axes]

    fig.suptitle(f"Open Loop Eval - Ep {episode_index} (Diffusion Policy) | MSE: {mse:.6f}", fontsize=18)
    x = np.arange(gt_actions.shape[0])
    for i, ax in enumerate(axes):
        ax.plot(x, gt_actions[:, i], color="#404040", linewidth=2.0, label="Ground Truth")
        ax.plot(x, pred_actions[:, i], color="red", linestyle="--", linewidth=1.8, label="Prediction")
        ax.set_ylabel(action_names[i])
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time Step")

    # 输出目录不存在时自动创建，保证脚本可直接运行。
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)


def evaluate_episode(
    dataset: LeRobotDataset,
    policy,
    preprocessor,
    postprocessor,
    episode_index: int,
    max_steps: int | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    在单个 episode 上执行逐帧开环评估。

    评估流程：
    1) 从 dataset.meta.episodes 读取该 episode 的绝对索引区间；
    2) 对每一帧取输入观测，经过 preprocess_observation + 官方 preprocessor；
    3) 调用 policy.select_action 推理，再经官方 postprocessor 反归一化；
    4) 收集 GT 动作与预测动作并对齐；
    5) 计算整体 MSE。
    """
    # 安全检查：显式要求官方推理入口 select_action，避免错误接口导致静默逻辑偏差。
    if not hasattr(policy, "select_action"):
        raise RuntimeError("未找到官方推理接口 select_action，当前脚本不使用 get_action。")

    # LeRobot 数据集在 meta.episodes 中维护每个 episode 的帧区间 [from, to)。
    episode_meta = dataset.meta.episodes[episode_index]
    start_idx = int(episode_meta["dataset_from_index"])
    end_idx = int(episode_meta["dataset_to_index"])
    episode_length = end_idx - start_idx
    actual_steps = episode_length if max_steps is None else min(max_steps, episode_length)

    print(f"开始评估 Episode {episode_index}，总帧数 {episode_length}，实际评估 {actual_steps} 帧")
    # 每条 episode 开始前重置策略内部队列，避免上条 episode 的时序缓存污染当前结果。
    policy.reset()
    policy.eval()

    gt_actions: list[np.ndarray] = []
    pred_actions: list[np.ndarray] = []

    # 从 policy 配置读取输入特征键，避免手写 key 与模型定义不一致。
    # 这是“从模型配置反推输入结构”的关键步骤，可减少 API 误用。
    input_feature_keys = list(policy.config.input_features.keys())
    has_output_postprocessor = postprocessor is not None

    with torch.inference_mode():
        for abs_idx in range(start_idx, start_idx + actual_steps):
            # dataset[abs_idx] 返回的是该帧完整字段（含 observation.* / action / 元信息等）。
            frame = dataset[abs_idx]

            # 只抽取模型真正需要的输入键，避免把无关键喂进策略。
            obs = {k: frame[k] for k in input_feature_keys if k in frame}
            # 基础预处理：HWC->CHW、补 batch、float 化等。
            obs = preprocess_observation(obs)
            # 官方 preprocessor：设备迁移 + normalize + rename 等。
            obs = preprocessor(obs) if preprocessor is not None else obs

            # 官方推理接口：DiffusionPolicy.select_action
            pred_action = policy.select_action(obs)
            # 官方 postprocessor：典型包含 unnormalize，输出到 CPU。
            if has_output_postprocessor:
                pred_action = postprocessor(pred_action)
            pred_np = pred_action.squeeze(0).detach().cpu().numpy().astype(np.float32)

            # GT 直接取数据集 action 字段。
            gt_tensor = frame[ACTION]
            gt_np = gt_tensor.detach().cpu().numpy().astype(np.float32)

            gt_actions.append(gt_np.reshape(-1))
            pred_actions.append(pred_np.reshape(-1))

    gt_arr = np.stack(gt_actions, axis=0)
    pred_arr = np.stack(pred_actions, axis=0)

    if gt_arr.shape != pred_arr.shape:
        raise RuntimeError(f"GT 与预测形状不一致：{gt_arr.shape} vs {pred_arr.shape}")

    # 整体 MSE：对“时间维 + 关节维”做全局平均。
    mse = float(np.mean((gt_arr - pred_arr) ** 2))
    return gt_arr, pred_arr, mse


def main() -> None:
    """
    主流程：
    - 解析参数；
    - 加载数据集与策略；
    - 构建官方 pre/post processor；
    - 执行单条 episode 评估；
    - 输出 MSE 与 PNG 图。
    """
    parser = argparse.ArgumentParser(description="LeRobot Diffusion Policy 纯离线开环评估")
    parser.add_argument("--repo-id", type=str, required=True, help="数据集 repo_id，例如 lerobot/pusht")
    parser.add_argument("--root", type=str, default=None, help="数据集本地根目录")
    parser.add_argument("--policy-path", type=str, required=True, help="训练输出的 policy checkpoint 路径")
    parser.add_argument("--episode", type=int, required=True, help="要评估的 episode 编号")
    parser.add_argument("--max-steps", type=int, default=None, help="最大评估帧数，默认整条 episode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-png", type=str, default="outputs/open_loop_eval.png")
    args = parser.parse_args()

    # 数据集加载：使用 LeRobotDataset 官方类，repo-id/root 与官方脚本一致。
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    if args.episode < 0 or args.episode >= dataset.num_episodes:
        raise ValueError(f"episode 越界：{args.episode}，数据集共有 {dataset.num_episodes} 条 episode")

    # 模型配置加载：
    # - 使用 PreTrainedConfig.from_pretrained 统一支持本地路径/Hub ID；
    # - 再将 pretrained_path 写回配置，供 make_policy / processor 构建时读取。
    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = args.device
    # 使用官方工厂创建 policy，ds_meta 来自 dataset.meta，保证特征定义一致。
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)

    # 构建 pre/post processor：
    # - dataset_stats 来自 dataset.meta.stats（与训练脚本保持一致）；
    # - preprocessor_overrides 显式覆盖 device 与 normalizer 参数；
    # - postprocessor_overrides 显式设置 unnormalizer 参数。
    # 这样做的核心目的是：让离线评估与训练/官方评测的归一化路径对齐。
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, {}),
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )

    gt_arr, pred_arr, mse = evaluate_episode(
        dataset=dataset,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        episode_index=args.episode,
        max_steps=args.max_steps,
    )

    action_names = None
    # 若数据集特征里带动作名，则用于子图纵轴标签，提升可读性。
    if ACTION in dataset.features and isinstance(dataset.features[ACTION], dict):
        action_names = dataset.features[ACTION].get("names")

    output_png = Path(args.output_png)
    plot_joint_curves(
        gt_actions=gt_arr,
        pred_actions=pred_arr,
        episode_index=args.episode,
        mse=mse,
        output_png=output_png,
        action_names=action_names,
    )

    print(f"评估完成，整体 MSE: {mse:.6f}")
    print(f"图片已保存到: {output_png.resolve()}")


if __name__ == "__main__":
    main()

