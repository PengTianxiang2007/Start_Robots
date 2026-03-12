from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

cfg = SO101FollowerConfig(
    port = "/dev/ttyACM0",
    use_degrees = True,
    max_relative_target = 15.0
)
arm = SO101Follower(cfg)
arm.connect(calibrate = False)

home_deg = {
    "shoulder_pan.pos": 0,
    "shoulder_lift.pos": 0,
    "elbow_flex.pos": 0,
    "wrist_flex.pos": 0,
    "wrist_roll.pos": 0,
    "gripper.pos": 40
}

print("Moving to home position...")
arm.send_action(home_deg)
# 获取当前机器人的所有观测数据（状态）
obs = arm.get_observation()

# 从返回的字典中提取 gripper 的位置
gripper_angle = obs["gripper.pos"]

print(f"当前 Gripper 的角度为: {gripper_angle}")
#obs = arm.get_observation()
#assert within_tol(obs, home_deg, pos_tol_deg=2.0)
#print("Home position reached successfully.")
