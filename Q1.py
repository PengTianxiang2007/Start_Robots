from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

cfg = SO101FollowerConfig(
    port = "/dev/ttyACM0",
    use_degrees = True,
    max_relative_target = 15.0
)
arm = SO101Follower(cfg)
arm.connect(calibrate = True)

home_deg = {
    "shoulder_pan.pos": 0,
    "shoulder_lift.pos": 0,
    "elbow_flex.pos": 0,
    "wrist_flex.pos": 0,
    "wrist_roll.pos": 0,
    "gripper.pos": 0
}

print("Moving to home position...")
arm.send_action(home_deg)

#obs = arm.get_observation()
#assert within_tol(obs, home_deg, pos_tol_deg=2.0)
#print("Home position reached successfully.")
