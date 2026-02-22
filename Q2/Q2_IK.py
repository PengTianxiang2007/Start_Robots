import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from Q2_Joint import Joint

"""输入并预处理输入角度"""
inputs = input("Please input the angle")
joint_angle = list(map(float, inputs.split()))
joint_angle[1] -= 90                            
joint_angle = np.radians(joint_angle)

shoulder_pan = Joint(0, 0, 0.0624, print_require=False)
shoulder_lift = Joint(-90, 0.035, 0, print_require=False)
elbow_flex = Joint(0, 0.116, 0, print_require=False)
wrist_flex = Joint(0, 0.135, 0, print_require=False)
wrist_roll = Joint(-90, 0, 0.061, print_require=False)

"""输入末位置,并转换为4*4矩阵"""
position = input("Please input the final positon,6 parameters")
final_position = list(map(float, position.split()))
target_position = SE3.Trans(final_position[0:3]) * SE3.RPY(final_position[3:6], order='xyz',unit='deg')

"""调用robticstoolbox库,实例化机器人进行逆运动学计算"""
Joints = [shoulder_pan.assemble_joint(), shoulder_lift.assemble_joint(), elbow_flex.assemble_joint(),
            wrist_flex.assemble_joint(), wrist_roll.assemble_joint()]
robot = rtb.DHRobot(Joints)

solution = robot.ikine_LM(target_position, q0=joint_angle, mask=[1,1,1,1,1,0]) #仅五个关节，忽略最后一个方向的约束
print(f"求解结果:{solution.q}")
print(f"偏差:{solution.residual}") #评估偏差