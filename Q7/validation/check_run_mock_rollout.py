import os
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "openvla_rollouts"


class Reporter:
    def __init__(self) -> None:
        self.failures = 0

    def pass_(self, title: str, detail: str) -> None:
        print(f"[PASS] {title}: {detail}")

    def fail(self, title: str, detail: str) -> None:
        self.failures += 1
        print(f"[FAIL] {title}: {detail}")

    def summary(self) -> int:
        print("\n===== Mock Rollout Summary =====")
        print(f"failures={self.failures}")
        return 1 if self.failures else 0


class MockBatch(dict):
    def to(self, device, dtype=None):
        self["device"] = device
        self["dtype"] = dtype
        return self


class MockProcessor:
    def __call__(self, prompt, image):
        if not isinstance(prompt, str):
            raise TypeError(f"processor prompt 必须是 str，实际得到 {type(prompt)}")
        if not isinstance(image, Image.Image):
            raise TypeError(f"processor image 必须是 PIL.Image.Image，实际得到 {type(image)}")
        return MockBatch({"prompt": prompt, "image_size": image.size})


class MockVLA:
    def predict_action(self, **kwargs):
        if "unnorm_key" not in kwargs:
            raise KeyError("predict_action 缺少 unnorm_key")
        if kwargs["unnorm_key"] != "bridge_orig":
            raise ValueError(f"unnorm_key 应为 bridge_orig，实际得到 {kwargs['unnorm_key']!r}")
        image_size = kwargs.get("image_size")
        if image_size != (224, 224):
            raise ValueError(f"policy 输入图像尺寸应为 (224, 224)，实际得到 {image_size}")
        return np.array([0.01, -0.02, 0.03, 0.1, -0.05, 0.02, 1.0], dtype=np.float32)


class MockEnv:
    robot_uid = "widowx"

    def __init__(self) -> None:
        self.step_count = 0
        self.closed = False

    def reset(self):
        self.step_count = 0
        return self._build_obs(), {"mock_reset": True}

    def get_language_instruction(self):
        return "put the eggplant in the basket"

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (7,):
            raise ValueError(f"env.step(action) 期望 7 维动作，实际得到 {action.shape}")
        if action.dtype != np.float32:
            raise TypeError(f"env.step(action) 期望 float32，实际得到 {action.dtype}")
        self.step_count += 1
        done = self.step_count >= 3
        truncated = False
        reward = 1.0 if done else 0.0
        info = {"episode_stats": {"mock_steps": self.step_count, "mock_success": done}}
        return self._build_obs(), reward, done, truncated, info

    def close(self):
        self.closed = True

    def _build_obs(self):
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        rgb[..., 1] = 127
        return {"image": {"3rd_view_camera": {"rgb": rgb}}}


def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    return obs["image"]["3rd_view_camera"]["rgb"]


def simple_euler_to_axangle(roll: float, pitch: float, yaw: float):
    # 这里不追求与 transforms3d 数值完全一致，只验证 run.py 的维度与数据流是否正确。
    vec = np.asarray([roll, pitch, yaw], dtype=np.float64)
    angle = float(np.linalg.norm(vec))
    if angle < 1e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64), 0.0
    axis = vec / angle
    return axis, angle


def main() -> int:
    reporter = Reporter()
    os.environ["DISPLAY"] = ""

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = MockProcessor()
    vla = MockVLA()
    env = MockEnv()

    num_episodes = 2
    max_episode_steps = 5
    save_video = True

    success_list = []
    saved_videos = []

    try:
        for episode_id in range(num_episodes):
            obs, info = env.reset()
            instruction = env.get_language_instruction()
            done, truncated = False, False
            step = 0
            frames = []

            if not isinstance(info, dict):
                raise TypeError(f"env.reset() 第二返回值应为 dict，实际得到 {type(info)}")
            if not isinstance(instruction, str):
                raise TypeError(f"env.get_language_instruction() 应返回 str，实际得到 {type(instruction)}")

            while not (done or truncated) and step < max_episode_steps:
                image = get_image_from_maniskill2_obs_dict(env, obs)
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(f"观测图像必须是 HxWx3 uint8，实际得到 shape={image.shape}, dtype={image.dtype}")

                frames.append(image)
                image_for_policy = np.asarray(Image.fromarray(image).resize((224, 224)), dtype=np.uint8)

                prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
                inputs = processor(prompt, Image.fromarray(image_for_policy).convert("RGB")).to(
                    device,
                    dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                )

                with torch.inference_mode():
                    raw_action = vla.predict_action(
                        **inputs,
                        unnorm_key="bridge_orig",
                        do_sample=False,
                    )

                raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
                if raw_action.shape[0] != 7:
                    raise ValueError(f"OpenVLA action dimension should be 7, but got shape {raw_action.shape}")

                world_vector = np.asarray(raw_action[:3], dtype=np.float32)
                rotation_delta = np.asarray(raw_action[3:6], dtype=np.float64)
                open_gripper = np.asarray(raw_action[6:7], dtype=np.float32)

                roll, pitch, yaw = rotation_delta
                rotation_axis, rotation_angle = simple_euler_to_axangle(roll, pitch, yaw)
                rot_axangle = np.asarray(rotation_axis * rotation_angle, dtype=np.float32)
                gripper = np.asarray(2.0 * (open_gripper > 0.5) - 1.0, dtype=np.float32)
                action = np.concatenate([world_vector, rot_axangle, gripper], axis=0).astype(np.float32)

                obs, reward, done, truncated, info = env.step(action)
                frames.append(get_image_from_maniskill2_obs_dict(env, obs))
                step += 1

                if not isinstance(reward, (int, float, np.floating)):
                    raise TypeError(f"reward 应为标量数值，实际得到 {type(reward)}")
                if not isinstance(info, dict):
                    raise TypeError(f"info 应为 dict，实际得到 {type(info)}")

            success = bool(done)
            success_list.append(success)

            if save_video and len(frames) > 0:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                video_path = OUTPUT_DIR / f"mock_episode_{episode_id:03d}_success_{int(success)}.mp4"
                import imageio

                imageio.mimsave(video_path, frames, fps=5)
                saved_videos.append(video_path)

        env.close()
        if not env.closed:
            raise RuntimeError("env.close() 没有被执行")

        reporter.pass_("rollout 流程", f"成功执行 {num_episodes} 个 mock episode")
        reporter.pass_("动作维度", "action 经后处理后可以稳定形成 7 维 float32 向量")
        reporter.pass_("视频导出", f"成功写出 {len(saved_videos)} 个 mock 视频到 {OUTPUT_DIR}")
        reporter.pass_("汇总统计", f"success_rate={float(np.mean(success_list)):.4f}")

    except Exception as exc:
        reporter.fail(
            "mock rollout",
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )

    return reporter.summary()


if __name__ == "__main__":
    raise SystemExit(main())
