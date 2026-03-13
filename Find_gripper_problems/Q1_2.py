import time
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# 1. 基础配置
cfg = SO101FollowerConfig(
    port = "/dev/ttyACM0",
    use_degrees = True,
    # 稍微放宽单次最大移动限制，让它动得干脆点
    max_relative_target = 30.0 
)
arm = SO101Follower(cfg)

# 2. 安全连接（不再污染底层 EEPROM）
print("🔌 正在连接... (已关闭自动校准同步)")
arm.connect(calibrate = False)

# 目标位置
home_deg = {
    "shoulder_pan.pos": 0,
    "shoulder_lift.pos": 0,
    "elbow_flex.pos": 0,
    "wrist_flex.pos": 0,
    "wrist_roll.pos": 0,
    "gripper.pos": 40
}

print("🚀 指令持续下发中，请观察夹爪动作...")

# ==========================================
# 🛑 最核心的区别：控制循环！
# 循环 100 次，每次休息 0.02 秒，总共给舵机 2 秒钟的时间去移动
# ==========================================
for i in range(100):
    arm.send_action(home_deg)
    time.sleep(0.02) 

print("✅ 控制指令下发完毕。")

# 获取最终状态
obs = arm.get_observation()
gripper_angle = obs["gripper.pos"]
print(f"📍 当前 Gripper 最终的角度为: {gripper_angle:.2f}")

# 安全退出保护
print("⚠️ 扭矩将在 3 秒后释放，机械臂会变软，请用手扶住它！")
for i in range(3, 0, -1):
    print(f"倒计时 {i}...")
    time.sleep(1)

arm.disconnect()
print("👋 已安全断开连接。")