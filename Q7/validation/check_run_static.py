import ast
import importlib.util
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = PROJECT_ROOT / "run.py"
if not RUN_PY.exists():
    RUN_PY = PROJECT_ROOT / "Sim2.py"
SIMPLER_ENV_DIR = PROJECT_ROOT / "SimplerEnv"
MANISKILL2_REAL2SIM_DIR = SIMPLER_ENV_DIR / "ManiSkill2_real2sim"


class Reporter:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, title: str, detail: str) -> None:
        print(f"[PASS] {title}: {detail}")

    def warn(self, title: str, detail: str) -> None:
        self.warnings += 1
        print(f"[WARN] {title}: {detail}")

    def fail(self, title: str, detail: str) -> None:
        self.failures += 1
        print(f"[FAIL] {title}: {detail}")

    def summary(self) -> int:
        print("\n===== Static Check Summary =====")
        print(f"failures={self.failures}")
        print(f"warnings={self.warnings}")
        return 1 if self.failures else 0


def get_top_level_string_assignments(tree: ast.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            result[target.id] = node.value.value
    return result


def find_display_assignment_line(tree: ast.Module) -> int | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not isinstance(target.value, ast.Attribute):
                continue
            if not isinstance(target.value.value, ast.Name):
                continue
            if target.value.value.id != "os" or target.value.attr != "environ":
                continue
            key = None
            if isinstance(target.slice, ast.Constant):
                key = target.slice.value
            elif hasattr(ast, "Index") and isinstance(target.slice, ast.Index) and isinstance(target.slice.value, ast.Constant):
                key = target.slice.value.value
            if key == "DISPLAY":
                return node.lineno
    return None


def find_import_line(tree: ast.Module, module_name: str) -> int | None:
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_name:
                    return node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.module == module_name:
                return node.lineno
    return None


def find_call_line(tree: ast.Module, func_name: str) -> int | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
            return node.lineno
        if isinstance(node.func, ast.Name) and node.func.id == func_name:
            return node.lineno
    return None


def main() -> int:
    reporter = Reporter()

    if not RUN_PY.exists():
        reporter.fail("目标脚本存在性", f"未找到 {PROJECT_ROOT / 'run.py'} 或 {PROJECT_ROOT / 'Sim2.py'}")
        return reporter.summary()

    source = RUN_PY.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(RUN_PY))
        reporter.ok("Python 语法", f"{RUN_PY.name} 可以被 AST 正常解析")
    except SyntaxError as exc:
        reporter.fail("Python 语法", f"{exc.msg} at line {exc.lineno}, column {exc.offset}")
        return reporter.summary()

    assignments = get_top_level_string_assignments(tree)
    model_path = assignments.get("model_path")
    env_id = assignments.get("env_id")

    if model_path:
        if Path(model_path).exists():
            reporter.ok("模型路径", f"模型目录存在: {model_path}")
        else:
            reporter.fail(
                "模型路径",
                f"{RUN_PY.name} 中的 model_path={model_path!r} 不存在。"
                " 如果你在 AutoDL 上运行，这通常意味着模型尚未下载到指定目录。",
            )
    else:
        reporter.fail("模型路径", f"未能在 {RUN_PY.name} 顶层找到字符串变量 model_path")

    asset_dir = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not isinstance(target.value, ast.Attribute):
                continue
            if not isinstance(target.value.value, ast.Name):
                continue
            if target.value.value.id != "os" or target.value.attr != "environ":
                continue
            key = None
            if isinstance(target.slice, ast.Constant):
                key = target.slice.value
            elif hasattr(ast, "Index") and isinstance(target.slice, ast.Index) and isinstance(target.slice.value, ast.Constant):
                key = target.slice.value.value
            if key == "MS2_REAL2SIM_ASSET_DIR" and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                asset_dir = node.value.value
    if asset_dir:
        if Path(asset_dir).exists():
            reporter.ok("Benchmark 资产路径", f"资产目录存在: {asset_dir}")
        else:
            reporter.fail(
                "Benchmark 资产路径",
                f"MS2_REAL2SIM_ASSET_DIR={asset_dir!r} 不存在。"
                " SimplerEnv 初始化时很可能会直接失败。",
            )
    else:
        reporter.fail("Benchmark 资产路径", "未找到 MS2_REAL2SIM_ASSET_DIR 的字符串赋值")

    display_line = find_display_assignment_line(tree)
    simpler_import_line = find_import_line(tree, "simpler_env")
    if display_line is None:
        reporter.fail("Headless 配置", "未找到 os.environ['DISPLAY'] 的设置；无图形界面服务器可能报错")
    elif simpler_import_line is not None and display_line < simpler_import_line:
        reporter.ok("Headless 配置", f"DISPLAY 在导入 simpler_env 之前设置，行号 {display_line} < {simpler_import_line}")
    else:
        reporter.warn(
            "Headless 配置",
            "找到了 DISPLAY 设置，但位置不早于 simpler_env 导入；这在某些环境下仍可能过晚。",
        )

    if "predict_action(" in source and 'unnorm_key="bridge_orig"' in source:
        reporter.ok("OpenVLA 反归一化", "predict_action() 显式使用 bridge_orig")
    else:
        reporter.fail("OpenVLA 反归一化", "未发现 predict_action(..., unnorm_key='bridge_orig')，BridgeV2/WidowX 反归一化可能不正确")

    if "raw_action.shape[0] != 7" in source:
        reporter.ok("动作维度保护", "存在 7 维动作检查")
    else:
        reporter.warn("动作维度保护", "未发现显式 7 维动作检查；若模型返回异常形状，错误会更难定位")

    if ".resize((224, 224))" in source:
        reporter.ok("图像尺寸", "存在 224x224 的 policy 输入缩放")
    else:
        reporter.warn("图像尺寸", "未发现 224x224 缩放；可能偏离仓库中 widowx_bridge 的参考做法")

    if "imageio.mimsave" in source:
        if importlib.util.find_spec("imageio_ffmpeg") is not None:
            reporter.ok("视频导出依赖", "检测到 imageio_ffmpeg，可直接写 mp4")
        else:
            reporter.fail("视频导出依赖", "SAVE_VIDEO=True 但未检测到 imageio_ffmpeg；导出 mp4 可能失败")
    else:
        reporter.warn("视频导出依赖", f"{RUN_PY.name} 当前未检测到 imageio.mimsave 调用")

    for extra_path in [str(SIMPLER_ENV_DIR), str(MANISKILL2_REAL2SIM_DIR)]:
        if extra_path not in sys.path:
            sys.path.insert(0, extra_path)

    try:
        import simpler_env  # type: ignore

        reporter.ok("simpler_env 导入", "可以成功导入 simpler_env")
        if env_id is not None:
            if env_id in simpler_env.ENVIRONMENTS:
                reporter.ok("env_id 合法性", f"{env_id!r} 在 simpler_env.ENVIRONMENTS 中")
            else:
                reporter.fail("env_id 合法性", f"{env_id!r} 不在 simpler_env.ENVIRONMENTS 中")
    except Exception as exc:
        reporter.fail(
            "simpler_env 导入",
            f"导入 simpler_env 失败: {type(exc).__name__}: {exc}. "
            "这通常意味着 Python 路径、依赖包或本地安装不完整。",
        )

    if find_call_line(tree, "make") is not None and "simpler_env.make(env_id)" in source:
        reporter.ok("Benchmark API", f"{RUN_PY.name} 使用 simpler_env.make(env_id) 构建预封装环境")
    else:
        reporter.fail("Benchmark API", "未检测到 simpler_env.make(env_id) 这种当前仓库支持的调用方式")

    return reporter.summary()


if __name__ == "__main__":
    raise SystemExit(main())
