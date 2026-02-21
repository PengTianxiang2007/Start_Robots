import torch
from torch import nn
import numpy as np
from Q2 import Joint
from spatialmath import SE3

"""定义前向传播函数"""
def frontward(Joint_matrix, joint_angle):
    for angle in joint_angle:
        angle.requires_grad_(True)

    shoulder_pan = Joint(0, 0, 0.0624, joint_angle[0])
    shoulder_lift = Joint(-90, 0.035, 0, joint_angle[1])
    elbow_flex = Joint(0, 0.116, 0, joint_angle[2])
    wrist_flex = Joint(0, 0.135, 0, joint_angle[3])
    wrist_roll = Joint(-90, 0, 0.061, joint_angle[4] )

    shoulder_pan_trans = shoulder_pan.trans_matrix()
    shoulder_lift_trans = shoulder_lift.trans_matrix()
    elbow_flex_trans = elbow_flex.trans_matrix()
    wrist_flex_trans = wrist_flex.trans_matrix()
    wrist_roll_trans = wrist_roll.trans_matrix()

    final_matrix = shoulder_pan_trans@shoulder_lift_trans@elbow_flex_trans@wrist_flex_trans@wrist_roll_trans

#定义损失函数
def loss(current_position, target_position):
    result_matrix = (target_position - current_position)*(target_position - current_position)*0.5
    return torch.sum(result_matrix)


#current_position:各个关节当下角度  target_position：目标位置
def gradient_decent(joint_angle, target_position, lr, epoches):

    for epoch in epoches:
        current_position = frontward(joint_angle)
        punish = loss(current_position, target_position)
        gradient_matrix = punish.backward()
        joint_angle = joint_angle - lr*gradient_matrix
    return joint_angle
