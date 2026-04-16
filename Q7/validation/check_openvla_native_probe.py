import argparse
import traceback

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


DEFAULT_MODEL_PATH = "/root/autodl-tmp/vla_resources/models/openvla-7b"


def print_env_info() -> None:
    print("===== Runtime Info =====")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    print("device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device 0:", torch.cuda.get_device_name(0))
    print("========================")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cuda:0", choices=["cuda:0", "cpu"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    print_env_info()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[FAIL] 请求使用 cuda:0，但当前 torch.cuda.is_available()=False")
        return 1

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]
    if args.device == "cpu" and model_dtype != torch.float32:
        print("[WARN] CPU 模式下将 dtype 强制改为 float32")
        model_dtype = torch.float32

    try:
        print(f"[INFO] loading processor from {args.model_path}")
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    except Exception as exc:
        print(f"[WARN] slow tokenizer 初始化失败，回退 fast tokenizer: {type(exc).__name__}: {exc}")
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)

    try:
        print(f"[INFO] loading model on {args.device} with dtype={model_dtype}")
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_path,
            torch_dtype=model_dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(args.device)
        model.eval()
    except Exception as exc:
        print(f"[FAIL] 模型加载失败: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return 1

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    prompt = "In: What action should the robot take to put eggplant into yellow basket?\nOut:"
    inputs = processor(prompt, Image.fromarray(image).convert("RGB")).to(args.device)
    for k, v in list(inputs.items()):
        if torch.is_tensor(v) and torch.is_floating_point(v):
            inputs[k] = v.to(dtype=model_dtype)

    if "attention_mask" in inputs:
        inputs.pop("attention_mask")
        print("[INFO] dropped attention_mask for probe")

    try:
        with torch.inference_mode():
            out = model.predict_action(
                **inputs,
                unnorm_key="bridge_orig",
                do_sample=False,
                use_cache=False,
            )
        shape = np.asarray(out).shape
        print(f"[PASS] predict_action 成功, output_shape={shape}")
        return 0
    except Exception as exc:
        print(f"[FAIL] predict_action 抛出 Python 异常: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
