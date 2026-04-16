import argparse
import traceback

import numpy as np
import tokenizers
import torch
import transformers
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


EXPECTED_TRANSFORMERS_VERSION = "4.40.1"
EXPECTED_TOKENIZERS_VERSION = "0.19.1"
DEFAULT_MODEL_PATH = "/root/autodl-tmp/vla_resources/models/openvla-7b"


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
        print("\n===== OpenVLA Compat Summary =====")
        print(f"failures={self.failures}")
        print(f"warnings={self.warnings}")
        return 1 if self.failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-probe", action="store_true", help="执行一次最小 predict_action 推理探测")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="OpenVLA 模型目录")
    args = parser.parse_args()

    reporter = Reporter()

    tf_ver = transformers.__version__
    tk_ver = tokenizers.__version__
    if tf_ver == EXPECTED_TRANSFORMERS_VERSION and tk_ver == EXPECTED_TOKENIZERS_VERSION:
        reporter.ok("依赖版本", f"transformers=={tf_ver}, tokenizers=={tk_ver}")
    else:
        reporter.fail(
            "依赖版本",
            f"期望 transformers=={EXPECTED_TRANSFORMERS_VERSION}, tokenizers=={EXPECTED_TOKENIZERS_VERSION}; "
            f"实际 transformers=={tf_ver}, tokenizers=={tk_ver}",
        )
        reporter.warn("修复建议", "pip install -U transformers==4.40.1 tokenizers==0.19.1 sentencepiece")

    try:
        import sentencepiece  # type: ignore  # noqa: F401

        reporter.ok("sentencepiece", "可导入")
    except Exception:
        reporter.warn("sentencepiece", "未安装；slow tokenizer 不可用，将回退 fast tokenizer")

    if not args.runtime_probe:
        reporter.warn("运行时探测已跳过", "添加 --runtime-probe 可进一步检测 predict_action 是否会触发 277/276 错误")
        return reporter.summary()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    try:
        try:
            processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
        except Exception:
            processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "padding_side"):
            processor.tokenizer.padding_side = "left"
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path,
            attn_implementation="eager",
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        model.eval()
        reporter.ok("模型加载", f"device={device}, dtype={model_dtype}")

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        prompt = "In: What action should the robot take to put eggplant into yellow basket?\nOut:"
        inputs = processor(prompt, Image.fromarray(image).convert("RGB")).to(device)
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and torch.is_floating_point(v):
                inputs[k] = v.to(dtype=model_dtype)

        with torch.inference_mode():
            _ = model.predict_action(
                **inputs,
                unnorm_key="bridge_orig",
                do_sample=False,
                use_cache=False,
            )
        reporter.ok("predict_action", "最小推理探测通过")
    except Exception as exc:
        reporter.fail(
            "predict_action",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )

    return reporter.summary()


if __name__ == "__main__":
    raise SystemExit(main())
