import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from Q2 import Joint

inputs = input("Please input the angle")

joint_angle = list(map(float, inputs.split()))
joint_angle[1] -= 90                              #预处理输入数据

shoulder_pan = Joint(0, 0, 0.0624, joint_angle[0])
shoulder_lift = Joint(-90, 0.035, 0, joint_angle[1])
elbow_flex = Joint(0, 0.116, 0, joint_angle[2])
wrist_flex = Joint(0, 0.135, 0, joint_angle[3])
wrist_roll = Joint(-90, 0, 0.061, joint_angle[4] )

"""计算并打印变换矩阵"""
shoulder_pan_trans = shoulder_pan.trans_matrix()
shoulder_lift_trans = shoulder_lift.trans_matrix()
elbow_flex_trans = elbow_flex.trans_matrix()
wrist_flex_trans = wrist_flex.trans_matrix()
wrist_roll_trans = wrist_roll.trans_matrix()

"""输入末位置,并转换为4*4矩阵"""
position = input("Please input the final positon,6 parameters")
final_position = list(map(float, position.split()))
target_position = SE3.Trans(final_position[0:3]) * SE3.RPY(final_position[3:6], order='xyz')

"""调用robticstoolbox库,实例化机器人进行逆运动学计算"""
Joints = [shoulder_pan.assemble_joint(), shoulder_lift.assemble_joint(), elbow_flex.assemble_joint(),
            wrist_flex.assemble_joint(), wrist_roll.assemble_joint()]
robot = rtb.DHRobot(Joints)

position = robot.ikine_LM(target_position, joint_angle)
print(position.q)
print(position.residual) #评估偏差