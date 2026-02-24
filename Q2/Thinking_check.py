import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from Q2_Joint import Joint
from Q2_Joint import caculate_matrixes
import torch
from torch import nn
from Q2_Joint import Joint_torch
from Q2_Joint import matrix_to_pose6d
from Q2_Joint import matrix_to_pose6d_np

"""进行逆运动学计算"""
#进行逆运动学计算，输入当前位置和目标张量
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
def gradient_decent( target_position,epoches ,lr, loss,joint_angle=[0,0,0,0,0]):
    optimzer = torch.optim.Adam([joint_angle], lr = lr)
    for epoch in range(epoches):
        optimzer.zero_grad()
        current_position = frontward(joint_angle)
        # 1. 分离位置(前3位)和姿态(后3位)
        pos_current, ori_current = current_position[:3], current_position[3:]
        pos_target, ori_target = target_position[:3], target_position[3:]
        
        # 2. 对位置赋予极高权重 (比如乘以100)，强迫模型优先对齐XYZ坐标
        loss_pos = loss(pos_current, pos_target) * 100.0  
        loss_ori = loss(ori_current, ori_target)
        punish = loss_pos + loss_ori
        punish.backward()
        optimzer.step()

        if (epoch + 1) % 4000 == 0:
            print(f"Epoch [{epoch+1}/{epoches}], Loss: {punish.item():.6f}")
        if punish.item() < 1e-2:
            print ("Already")
            break
    return joint_angle

if __name__ == "__main__":
    """进行正运动学计算"""
    inputs = input("Please input the angle  ")
    target_matrix = caculate_matrixes(inputs, False)
    target_position_np = matrix_to_pose6d_np(target_matrix)
    
    # 将 NumPy 生成的位姿转为 Torch Tensor，供逆解作为目标使用
    target_position = torch.tensor(target_position_np, dtype=torch.float32)
    #print(f"\n[由正运动学计算生成的目标位姿 6D Pose]:\n{target_position}\n")

    """进行逆运动学计算"""
    inputs_ik = input("Please input the angles  ")
    joint_angle = torch.tensor(list(map(float, inputs_ik.split())), requires_grad=True, dtype=torch.float32)

    loss = torch.nn.MSELoss()

    output = gradient_decent(target_position, epoches=84000, lr=0.01, loss=loss, joint_angle = joint_angle)
    
    print(f"\n目标为:{inputs} \n 求解结果为{output}")