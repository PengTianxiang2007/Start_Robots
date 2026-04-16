import os
import sys
import imageio
import numpy as np
import torch
from transforms3d.euler import euler2axangle
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

def configure_headless_render_env():
    os.environ["DISPLAY"] = ""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    os.environ.setdefault("MUJOCO_GL", "egl")
    if "VK_ICD_FILENAMES" not in os.environ:
        vk_icd_candidates = [
            "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "/etc/vulkan/icd.d/nvidia_icd.json",
        ]
        for p in vk_icd_candidates:
            if os.path.isfile(p):
                os.environ["VK_ICD_FILENAMES"] = p
                break


configure_headless_render_env()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SIMPLER_ENV_DIR = os.environ.get("SIMPLER_ENV_DIR", os.path.join(CURRENT_DIR, "SimplerEnv"))
if not os.path.isdir(SIMPLER_ENV_DIR):
    parent_candidate = os.path.join(os.path.dirname(CURRENT_DIR), "SimplerEnv")
    if os.path.isdir(parent_candidate):
        SIMPLER_ENV_DIR = parent_candidate
if not os.path.isdir(SIMPLER_ENV_DIR):
    start_robots_candidate = "/root/autodl-tmp/Start_Robots/SimplerEnv"
    if os.path.isdir(start_robots_candidate):
        SIMPLER_ENV_DIR = start_robots_candidate
MANISKILL2_REAL2SIM_DIR = os.path.join(SIMPLER_ENV_DIR, "ManiSkill2_real2sim")
for extra_path in [SIMPLER_ENV_DIR, MANISKILL2_REAL2SIM_DIR]:
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


def save_video_with_fallback(frames, output_stem):
    mp4_path = output_stem + ".mp4"
    gif_path = output_stem + ".gif"
    errors = []

    try:
        imageio.mimsave(mp4_path, frames, fps=5)
        return mp4_path, None, errors
    except Exception as exc:
        errors.append(
            "mp4 保存失败: "
            f"{type(exc).__name__}: {exc}. "
            "建议安装 `imageio[ffmpeg]` 或 `imageio[pyav]`。"
        )

    try:
        imageio.mimsave(gif_path, frames, duration=0.2)
        return gif_path, "mp4_backend_missing", errors
    except Exception as exc:
        errors.append(
            "gif 保存也失败: "
            f"{type(exc).__name__}: {exc}."
        )
        return None, "all_video_backends_failed", errors

"""
导入模型
"""
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model_path = "/root/autodl-tmp/vla_resources/models/openvla-7b"
model_path = os.environ.get("OPENVLA_MODEL_PATH", model_path)
attn_impl = os.environ.get("OPENVLA_ATTN_IMPL", "eager")
if DEVICE.startswith("cuda"):
    dtype_name = os.environ.get("OPENVLA_DTYPE", "float16").lower()
    if dtype_name == "bfloat16":
        model_dtype = torch.bfloat16
    elif dtype_name == "float32":
        model_dtype = torch.float32
    else:
        model_dtype = torch.float16
else:
    model_dtype = torch.float32
try:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
except Exception as exc:
    print(
        "[WARN] use_fast=False 初始化失败，自动回退 use_fast=True。"
        f" 具体报错: {type(exc).__name__}: {exc}"
    )
    print(
        "[INFO] 若要继续使用 slow tokenizer，可安装: pip install sentencepiece"
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
if hasattr(processor, "tokenizer"):
    if hasattr(processor.tokenizer, "legacy"):
        processor.tokenizer.legacy = True
    if hasattr(processor.tokenizer, "padding_side"):
        processor.tokenizer.padding_side = "left"
try:
    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        attn_implementation=attn_impl,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE)
except Exception as exc:
    print("flash_attention_2 load failed, fallback to default attention:", exc)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE)
vla.eval()

"""
导入benchmark
"""
_asset_dir_override = os.environ.get("MS2_REAL2SIM_ASSET_DIR")
os.environ["MS2_REAL2SIM_ASSET_DIR"] = "/root/autodl-tmp/vla_resources/SimplerEnv/ManiSkill2_real2sim/data"
if _asset_dir_override:
    os.environ["MS2_REAL2SIM_ASSET_DIR"] = _asset_dir_override
env_id = os.environ.get("SIMPLER_ENV_ID", "widowx_put_eggplant_in_basket")

# 这里不能继续向 simpler_env.make() 传入 obs_mode / control_mode / render_mode，
# 因为当前仓库里的 simpler_env.make(task_name) 真实签名只接受任务名，
# 并在内部构建对应的预封装 SIMPLER benchmark 环境。
try:
    env = simpler_env.make(env_id)
except Exception as exc:
    msg = str(exc)
    if "SapienRenderer" in msg or "rendering device" in msg or "GLFW" in msg or "X11" in msg:
        print(
            "[Headless 渲染诊断] simpler_env.make() 失败，疑似 Vulkan/EGL 渲染设备不可用。\n"
            "已在脚本中设置 DISPLAY='', PYOPENGL_PLATFORM=egl, EGL_PLATFORM=surfaceless。\n"
            "请在云服务器继续检查：\n"
            "1) nvidia-smi 是否可见 GPU；\n"
            "2) /usr/share/vulkan/icd.d/nvidia_icd.json 是否存在；\n"
            "3) 容器是否挂载了 Vulkan/NVIDIA 图形驱动（不仅是 CUDA 计算驱动）。"
        )
    raise

"""
开始循环
"""
NUM_EPISODES = 10
MAX_EPISODE_STEPS = 120
SAVE_VIDEO = True
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs", "openvla_rollouts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

success_list = []

for episode_id in range(NUM_EPISODES):
    obs, info = env.reset()
    instruction = env.get_language_instruction()
    done, truncated = False, False
    step = 0
    frames = []

    print(f"\n===== Episode {episode_id} =====")
    print("reset info:", info)
    print("instruction:", instruction)

    while not (done or truncated) and step < MAX_EPISODE_STEPS:
        image = get_image_from_maniskill2_obs_dict(env, obs)
        frames.append(image)
        image_for_policy = np.asarray(
            Image.fromarray(image).resize((224, 224)),
            dtype=np.uint8,
        )

        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = processor(prompt, Image.fromarray(image_for_policy).convert("RGB")).to(DEVICE)
        if isinstance(inputs, dict) and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)

        with torch.inference_mode():
            raw_action = vla.predict_action(
                **inputs,
                unnorm_key="bridge_orig",
                do_sample=False,
                    use_cache=False,
            )

        if isinstance(raw_action, torch.Tensor):
            raw_action = raw_action.detach().cpu().numpy()
        raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if raw_action.shape[0] != 7:
            raise ValueError(f"OpenVLA action dimension should be 7, but got shape {raw_action.shape}")

        world_vector = np.asarray(raw_action[:3], dtype=np.float32)
        rotation_delta = np.asarray(raw_action[3:6], dtype=np.float64)
        open_gripper = np.asarray(raw_action[6:7], dtype=np.float32)

        roll, pitch, yaw = rotation_delta
        rotation_axis, rotation_angle = euler2axangle(roll, pitch, yaw)
        rot_axangle = np.asarray(rotation_axis * rotation_angle, dtype=np.float32)

        gripper = np.asarray(2.0 * (open_gripper > 0.5) - 1.0, dtype=np.float32)
        action = np.concatenate([world_vector, rot_axangle, gripper], axis=0).astype(np.float32)

        obs, reward, done, truncated, info = env.step(action)
        frames.append(get_image_from_maniskill2_obs_dict(env, obs))
        step += 1

        new_instruction = env.get_language_instruction()
        if new_instruction != instruction:
            instruction = new_instruction
            print("new instruction:", instruction)

        print(
            f"step={step:03d}, reward={reward:.4f}, done={done}, truncated={truncated}"
        )

    if step >= MAX_EPISODE_STEPS and not (done or truncated):
        truncated = True
        print(f"episode {episode_id} reached max steps: {MAX_EPISODE_STEPS}")

    success = bool(done)
    success_list.append(success)
    episode_stats = info.get("episode_stats", {})
    print(f"episode {episode_id} success: {success}")
    print("episode stats:", episode_stats)

    if SAVE_VIDEO and len(frames) > 0:
        output_stem = os.path.join(
            OUTPUT_DIR,
            f"{env_id}_episode_{episode_id:03d}_success_{int(success)}",
        )
        saved_path, fallback_reason, save_errors = save_video_with_fallback(frames, output_stem)
        if saved_path is None:
            print("video save skipped due to backend error:")
            for err in save_errors:
                print("-", err)
        else:
            if fallback_reason is None:
                print("saved video to:", saved_path)
            else:
                print("saved gif fallback to:", saved_path)
                for err in save_errors:
                    print("-", err)

env.close()

success_rate = float(np.mean(success_list)) if len(success_list) > 0 else 0.0
print("\n===== Summary =====")
print("env_id:", env_id)
print("num_episodes:", NUM_EPISODES)
print("num_success:", int(np.sum(success_list)))
print("success_rate:", success_rate)

