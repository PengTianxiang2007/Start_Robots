import torch
from torch import nn
import numpy as np
from Q2_Joint import Joint_torch
from Q2_Joint import matrix_to_pose6d

"""定义前向传播函数"""
def frontward(joint_angle): #需传入张量

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
    final_matrix = matrix_to_pose6d(final_matrix)
    return final_matrix

#current_position:各个关节当下角度  target_position：目标位置
def gradient_decent(joint_angle, target_position,epoches ,lr, loss):
    optimzer = torch.optim.Adam([joint_angle], lr = lr)
    for epoch in range(epoches):
        optimzer.zero_grad()
        current_position = frontward(joint_angle)
        punish = loss(current_position, target_position)
        punish.backward()
        optimzer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{epoches}], Loss: {punish.item():.6f}")
        if punish.item() < 1e-4:
            print ("Already")
            break
    return joint_angle

inputs = input("Please input the angles")
joint_angle = torch.tensor(list(map(float, inputs.split())), requires_grad=True)
target_position = torch.tensor(list(map(float, input("Please input the target position").split())))
loss = torch.nn.MSELoss()
output = gradient_decent(joint_angle, target_position, epoches=84000, lr=0.01, loss=loss)
print(f"目标为:{target_position} \n 求解结果为{output}")