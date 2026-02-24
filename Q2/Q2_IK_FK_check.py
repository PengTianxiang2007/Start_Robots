import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from Q2_Joint import Joint
from Q2_FK import caculate_matrixes

"""进行正运动学计算"""
inputs = input("Please input the angle")
input_angle = list(map(float, inputs.split()))
target_position = caculate_matrixes(inputs, False)

"""进行逆运动学计算"""
shoulder_pan = Joint(0, 0, 0.0624)
shoulder_lift = Joint(-90, 0.035, 0)
elbow_flex = Joint(0, 0.116, 0)
wrist_flex = Joint(0, 0.135, 0)
wrist_roll = Joint(-90, 0, 0.061)
Joints = [shoulder_pan.assemble_joint(), shoulder_lift.assemble_joint(), elbow_flex.assemble_joint(),
            wrist_flex.assemble_joint(), wrist_roll.assemble_joint()]
robot = rtb.DHRobot(Joints)

"""反向处理求解数据"""
q0 = np.radians([0, -90, 0, 0, 0])
solution = robot.ikine_LM(target_position,q0 = q0)
angle_solution = np.degrees(solution.q)
angle_solution[1] += 90
angle_solution = np.round(angle_solution, 4)
bias = angle_solution - input_angle
print(f"The input angle is {input_angle} \n The solution is {angle_solution}")
print(f"偏差为 {bias}")