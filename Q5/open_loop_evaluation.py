"""
python open_loop_evaluation.py \
  --repo-id PengTianxiang/20260319First \
  --policy-path PengTianxiang/20260321diffusion_run7_060000\
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
    processed: dict[str, torch.Tensor] = {}
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(value))

        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)

        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        else:
            tensor = tensor.float()

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
    action_dim = gt_actions.shape[1]
    if action_names is None or len(action_names) != action_dim:
        action_names = [f"Dim {i}" for i in range(action_dim)]

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
    if not hasattr(policy, "select_action"):
        raise RuntimeError("未找到官方推理接口 select_action，当前脚本不使用 get_action。")

    episode_meta = dataset.meta.episodes[episode_index]
    start_idx = int(episode_meta["dataset_from_index"])
    end_idx = int(episode_meta["dataset_to_index"])
    episode_length = end_idx - start_idx
    actual_steps = episode_length if max_steps is None else min(max_steps, episode_length)

    print(f"开始评估 Episode {episode_index}，总帧数 {episode_length}，实际评估 {actual_steps} 帧")
    policy.reset()
    policy.eval()

    gt_actions: list[np.ndarray] = []
    pred_actions: list[np.ndarray] = []

    input_feature_keys = list(policy.config.input_features.keys())
    has_output_postprocessor = postprocessor is not None

    with torch.inference_mode():
        for abs_idx in range(start_idx, start_idx + actual_steps):
            frame = dataset[abs_idx]

            obs = {k: frame[k] for k in input_feature_keys if k in frame}
            obs = preprocess_observation(obs)
            obs = preprocessor(obs) if preprocessor is not None else obs

            pred_action = policy.select_action(obs)
            if has_output_postprocessor:
                pred_action = postprocessor(pred_action)
            pred_np = pred_action.squeeze(0).detach().cpu().numpy().astype(np.float32)

            gt_tensor = frame[ACTION]
            gt_np = gt_tensor.detach().cpu().numpy().astype(np.float32)

            gt_actions.append(gt_np.reshape(-1))
            pred_actions.append(pred_np.reshape(-1))

    gt_arr = np.stack(gt_actions, axis=0)
    pred_arr = np.stack(pred_actions, axis=0)

    if gt_arr.shape != pred_arr.shape:
        raise RuntimeError(f"GT 与预测形状不一致：{gt_arr.shape} vs {pred_arr.shape}")

    mse = float(np.mean((gt_arr - pred_arr) ** 2))
    return gt_arr, pred_arr, mse


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobot Diffusion Policy 纯离线开环评估")
    parser.add_argument("--repo-id", type=str, required=True, help="数据集 repo_id，例如 lerobot/pusht")
    parser.add_argument("--root", type=str, default=None, help="数据集本地根目录")
    parser.add_argument("--policy-path", type=str, required=True, help="训练输出的 policy checkpoint 路径")
    parser.add_argument("--episode", type=int, required=True, help="要评估的 episode 编号")
    parser.add_argument("--max-steps", type=int, default=None, help="最大评估帧数，默认整条 episode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-png", type=str, default="outputs/open_loop_eval.png")
    args = parser.parse_args()

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    if args.episode < 0 or args.episode >= dataset.num_episodes:
        raise ValueError(f"episode 越界：{args.episode}，数据集共有 {dataset.num_episodes} 条 episode")

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = args.device
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)

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

