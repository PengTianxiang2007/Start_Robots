import os
import sys
import traceback
import ast
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_DIR = Path("/root/autodl-tmp/vla_resources/SimplerEnv/ManiSkill2_real2sim/data")
DEFAULT_ENV_ID = "widowx_put_eggplant_in_basket"


class Reporter:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def pass_(self, title: str, detail: str) -> None:
        print(f"[PASS] {title}: {detail}")

    def warn(self, title: str, detail: str) -> None:
        self.warnings += 1
        print(f"[WARN] {title}: {detail}")

    def fail(self, title: str, detail: str) -> None:
        self.failures += 1
        print(f"[FAIL] {title}: {detail}")

    def summary(self) -> int:
        print("\n===== SimplerEnv Probe Summary =====")
        print(f"failures={self.failures}")
        print(f"warnings={self.warnings}")
        return 1 if self.failures else 0


def parse_environments_from_init(init_file: Path) -> list[str]:
    source = init_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(init_file))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "ENVIRONMENTS":
            continue
        if not isinstance(node.value, ast.List):
            continue
        values = []
        for item in node.value.elts:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                values.append(item.value)
        return values
    return []


def resolve_simpler_env_dir() -> Path | None:
    # 1) 用户显式指定优先
    env_override = os.environ.get("SIMPLER_ENV_DIR")
    if env_override:
        p = Path(env_override).expanduser().resolve()
        if (p / "simpler_env" / "__init__.py").exists():
            return p

    # 2) 常见候选目录自动探测
    candidates = [
        PROJECT_ROOT / "SimplerEnv",
        PROJECT_ROOT.parent / "SimplerEnv",
        Path.cwd() / "SimplerEnv",
        Path.cwd().parent / "SimplerEnv",
        Path("/root/autodl-tmp/vla_resources/SimplerEnv"),
        Path("/root/autodl-tmp/Start_Robots/SimplerEnv"),
    ]
    for p in candidates:
        p = p.expanduser().resolve()
        if (p / "simpler_env" / "__init__.py").exists():
            return p
    return None


def main() -> int:
    reporter = Reporter()
    os.environ["DISPLAY"] = ""
    os.environ["MS2_REAL2SIM_ASSET_DIR"] = str(DEFAULT_ASSET_DIR)
    runtime_probe = os.environ.get("ENABLE_RUNTIME_PROBE", "0") == "1"

    simpler_env_dir = resolve_simpler_env_dir()
    if simpler_env_dir is None:
        reporter.fail(
            "simpler_env 源码",
            "未找到有效的 SimplerEnv 目录。请设置环境变量 SIMPLER_ENV_DIR，"
            "例如: export SIMPLER_ENV_DIR=/root/autodl-tmp/vla_resources/SimplerEnv",
        )
        return reporter.summary()
    maniskill2_real2sim_dir = simpler_env_dir / "ManiSkill2_real2sim"
    reporter.pass_("simpler_env 目录", str(simpler_env_dir))

    if not DEFAULT_ASSET_DIR.exists():
        if runtime_probe:
            reporter.fail(
                "资产目录",
                f"默认资产目录不存在: {DEFAULT_ASSET_DIR}. "
                "这会导致真实仿真阶段失败。",
            )
        else:
            reporter.warn(
                "资产目录",
                f"默认资产目录不存在: {DEFAULT_ASSET_DIR}。"
                "当前是无独显兼容模式，仅做静态检查，不阻断执行。",
            )
    else:
        reporter.pass_("资产目录", f"检测到资产目录: {DEFAULT_ASSET_DIR}")

    simpler_env_init = simpler_env_dir / "simpler_env" / "__init__.py"
    if not simpler_env_init.exists():
        reporter.fail("simpler_env 源码", f"未找到 {simpler_env_init}")
    else:
        envs = parse_environments_from_init(simpler_env_init)
        if len(envs) == 0:
            reporter.fail("ENVIRONMENTS 解析", "未能从 simpler_env/__init__.py 解析出 ENVIRONMENTS")
        else:
            reporter.pass_("ENVIRONMENTS 解析", f"共解析到 {len(envs)} 个任务")
            if DEFAULT_ENV_ID in envs:
                reporter.pass_("env_id 合法性", f"{DEFAULT_ENV_ID!r} 在 ENVIRONMENTS 中")
            else:
                reporter.fail("env_id 合法性", f"{DEFAULT_ENV_ID!r} 不在 ENVIRONMENTS 中")

    for extra_path in [str(simpler_env_dir), str(maniskill2_real2sim_dir)]:
        if extra_path not in sys.path:
            sys.path.insert(0, extra_path)

    if not runtime_probe:
        reporter.warn(
            "运行时探测已跳过",
            "当前为无独显兼容模式。仅进行了静态检查；"
            "如需继续检测 env.make/reset/step，请设置 ENABLE_RUNTIME_PROBE=1。",
        )
        return reporter.summary()

    reporter.warn("运行时探测已开启", "ENABLE_RUNTIME_PROBE=1，开始执行需要图形/渲染后端的检测。")

    try:
        import simpler_env  # type: ignore
        from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict  # type: ignore
        reporter.pass_("simpler_env 导入", "导入成功")
    except Exception as exc:
        reporter.fail(
            "simpler_env 导入",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        return reporter.summary()

    try:
        env = simpler_env.make(DEFAULT_ENV_ID)
        reporter.pass_("环境构建", f"simpler_env.make({DEFAULT_ENV_ID!r}) 成功")
    except Exception as exc:
        reporter.fail(
            "环境构建",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        return reporter.summary()

    try:
        obs, info = env.reset()
        reporter.pass_("环境 reset", f"reset 成功，info keys={list(info.keys()) if isinstance(info, dict) else type(info)}")
    except Exception as exc:
        reporter.fail(
            "环境 reset",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}\n"
            "如果这里报 EGL / Vulkan / GLX / render / sapien 相关错误，通常是无图形界面下的离屏渲染环境没有配置好。",
        )
        try:
            env.close()
        except Exception:
            pass
        return reporter.summary()

    try:
        instruction = env.get_language_instruction()
        image = get_image_from_maniskill2_obs_dict(env, obs)
        if not isinstance(instruction, str):
            raise TypeError(f"instruction 应为 str，实际得到 {type(instruction)}")
        if not isinstance(image, np.ndarray):
            raise TypeError(f"观测图像应为 np.ndarray，实际得到 {type(image)}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"观测图像必须是 HxWx3，实际得到 shape={image.shape}")
        if image.dtype != np.uint8:
            raise TypeError(f"观测图像 dtype 应为 uint8，实际得到 {image.dtype}")
        reporter.pass_("语言指令", instruction)
        reporter.pass_("相机观测", f"shape={image.shape}, dtype={image.dtype}")
    except Exception as exc:
        reporter.fail(
            "观测检查",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )

    try:
        action = np.zeros(7, dtype=np.float32)
        next_obs, reward, done, truncated, step_info = env.step(action)
        _ = get_image_from_maniskill2_obs_dict(env, next_obs)
        reporter.pass_(
            "环境 step",
            f"使用 7 维零动作成功 step 一次，reward={reward}, done={done}, truncated={truncated}, "
            f"info keys={list(step_info.keys()) if isinstance(step_info, dict) else type(step_info)}",
        )
    except Exception as exc:
        reporter.fail(
            "环境 step",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}\n"
            "如果这里失败而 reset 成功，往往说明动作格式、控制器或渲染后的观测回收阶段存在问题。",
        )

    try:
        env.close()
        reporter.pass_("环境关闭", "env.close() 成功")
    except Exception as exc:
        reporter.fail(
            "环境关闭",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )

    return reporter.summary()


if __name__ == "__main__":
    raise SystemExit(main())
