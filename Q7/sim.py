import os
import imageio
import numpy as np
import torch
import gymnasium as gym
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# =====================================================================
# ⚠️ 必须修改区：请根据你的实际环境替换以下变量
# =====================================================================
ENV_ID = "BridgeV2-PickAndPlace-v0"  # 替换为你实际 pip install 后注册的环境名
INSTRUCTION = "pick up the spoon and put it in the bowl" # 必须与数据集指令风格一致
OBS_IMAGE_KEY = "image"              # 图像在 obs 字典中的键名 (可能是 'pixels', 'agentview_image' 等)
UNNORM_KEY = "bridge_orig"           # 反归一化所用的数据集统计参数标识
NUM_EPISODES = 10                    # 外层循环：评估的回合数
MAX_STEPS = 200                      # 内层循环：每回合的最大物理步数
# =====================================================================

def main():
    print("--- 1. Initialization ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1.1 加载 Processor 和 VLA 模型 (严谨的 HuggingFace API)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        attn_implementation="flash_attention_2", 
        torch_dtype=torch.bfloat16,              
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to(device)
    print("✅ VLA Model Loaded.")

    # 1.2 初始化 Gymnasium 环境
    env = gym.make(ENV_ID, render_mode="rgb_array")
    print(f"✅ Environment '{ENV_ID}' Initialized. Action Space: {env.action_space}")
    
    # 构建 Prompt
    prompt = f"In: What action should the robot take to {INSTRUCTION}?\nOut:"
    
    # 用于统计的变量
    success_count = 0
    os.makedirs("simulation_videos", exist_ok=True) # 创建存视频的文件夹

    print("\n--- 2. Starting Evaluation Loop ---")
    
    # 使用 inference_mode 彻底切断梯度，保护 AutoDL 的显存
    with torch.inference_mode():
        
        # 【外层循环】：迭代测试不同的随机初始状态 (Episodes)
        for episode in range(NUM_EPISODES):
            print(f"\n▶️ Episode {episode + 1}/{NUM_EPISODES} Started...")
            
            # 环境重置，传入 seed 保证实验可复现性（也可以不传 seed 让其全随机）
            obs, info = env.reset(seed=42 + episode)
            
            step = 0
            done = False
            frames = [] # 存储当前回合的视频帧
            
            # 【内层循环】：控制周期迭代 (Steps)
            while not done and step < MAX_STEPS:
                
                # --- A. 感知 (Perception & Preprocessing) ---
                if OBS_IMAGE_KEY not in obs:
                    raise KeyError(f"当前 obs 不包含 '{OBS_IMAGE_KEY}'。可用键有: {obs.keys()}")
                
                raw_img_array = obs[OBS_IMAGE_KEY]
                
                # 防御性编程：确保通道在最后 (H, W, C)
                if raw_img_array.shape[0] == 3: 
                    raw_img_array = np.transpose(raw_img_array, (1, 2, 0))
                
                # 收集用于录像的帧 (确保是 uint8)
                frames.append(raw_img_array.astype(np.uint8))
                
                # 转为模型需要的 PIL 格式
                pil_image = Image.fromarray(raw_img_array)
                # 修改后：显式指定关键字参数，并要求返回 PyTorch Tensors
                inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device, dtype=torch.bfloat16)

                # --- B. 决策 (Inference) ---
                # predict_action 是 OpenVLA 特有 API，内部会自动反归一化
                action = vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
                
                # 确保输出格式为 numpy 数组 (Gymnasium 的要求)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()

                # --- C. 执行 (Execution) ---
                # 标准 Gymnasium v0.26+ API
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # 状态更新 (将世界推向下一帧)
                obs = next_obs
                done = terminated or truncated
                step += 1
                
                if step % 20 == 0:
                    print(f"   Step {step}/{MAX_STEPS} executed.")

            # --- 回合结束，结算处理 ---
            if terminated:
                print(f"✅ Episode {episode + 1} SUCCESS! (Finished in {step} steps)")
                success_count += 1
            else:
                print(f"❌ Episode {episode + 1} FAILED/TRUNCATED. (Reached {step} steps)")

            # 保存本回合的录像
            video_path = f"simulation_videos/episode_{episode + 1}.mp4"
            imageio.mimsave(video_path, frames, fps=10) # 假设 10Hz 的控制频率
            
    # 【最终报告】
    print("\n=========================================")
    print(f"📊 Final Report: Success Rate: {success_count}/{NUM_EPISODES} ({(success_count/NUM_EPISODES)*100:.2f}%)")
    print(f"📁 Videos saved in './simulation_videos/'")
    print("=========================================")

if __name__ == "__main__":
    main()