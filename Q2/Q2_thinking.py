import torch
from torch import nn
import numpy as np
from Q2_Joint import Joint_torch

"""定义前向传播函数"""
def frontward(joint_angle): #需传入张量
    joint_angle.requires_grad_(True)

    shoulder_pan = Joint_torch(0, 0, 0.0624, joint_angle[0])
    shoulder_lift = Joint_torch(-90, 0.035, 0, joint_angle[1])
    elbow_flex = Joint_torch(0, 0.116, 0, joint_angle[2])
    wrist_flex = Joint_torch(0, 0.135, 0, joint_angle[3])
    wrist_roll = Joint_torch(-90, 0, 0.061, joint_angle[4] )

    shoulder_pan_trans = shoulder_pan.trans_matrix()
    shoulder_lift_trans = shoulder_lift.trans_matrix()
    elbow_flex_trans = elbow_flex.trans_matrix()
    wrist_flex_trans = wrist_flex.trans_matrix()
    wrist_roll_trans = wrist_roll.trans_matrix()

    final_matrix = shoulder_pan_trans@shoulder_lift_trans@elbow_flex_trans@wrist_flex_trans@wrist_roll_trans
    return final_matrix

#current_position:各个关节当下角度  target_position：目标位置
def gradient_decent(joint_angle, target_position,epoches ,lr, loss):
    optimzer = torch.optim.SGD([joint_angle], lr = lr)
    for epoch in range(epoches):
        optimzer.zero_grad()
        current_position = frontward(joint_angle)
        punish = loss(current_position, target_position)
        punish.backward()
        optimzer.step()
    return joint_angle

