import numpy as np
import roboticstoolbox as rtb
import torch
from torch import nn

class Joint:
    def __init__(self, Link_twist, Link_length, Link_offset, Joint_angle=0,print_require = False):
        self.Link_twist = np.radians(Link_twist)
        self.Link_length = Link_length
        self.Link_offset = Link_offset
        self.Joint_angle = np.radians(Joint_angle)
        self.print_require = print_require

    def trans_matrix(self):
        matrix = np.array([
            [np.cos(self.Joint_angle), -np.sin(self.Joint_angle), 0, self.Link_length],
            [np.sin(self.Joint_angle)*np.cos(self.Link_twist), np.cos(self.Joint_angle)*np.cos(self.Link_twist), -np.sin(self.Link_twist), -self.Link_offset*np.sin(self.Link_twist)],
            [np.sin(self.Joint_angle)*np.sin(self.Link_twist), np.cos(self.Joint_angle)*np.sin(self.Link_twist), np.cos(self.Link_twist), self.Link_offset*np.cos(self.Link_twist)],
            [0, 0, 0, 1]
        ])
        if self.print_require ==True:
            print(matrix)
        return matrix
    
    def assemble_joint(self):
        return rtb.RevoluteMDH(alpha=self.Link_twist, a=self.Link_length, d=self.Link_offset)
    
class Joint_torch:
    def __init__(self, Link_twist, Link_length, Link_offset, Joint_angle=0,print_require = False):
        self.Link_twist = torch.deg2rad(torch.tensor(float(Link_twist)))
        self.Link_length = torch.tensor(float(Link_length))
        self.Link_offset = torch.tensor(float(Link_offset))
        self.Joint_angle = torch.deg2rad(Joint_angle)
        self.print_require = print_require

    def trans_matrix(self):
        zero = torch.tensor(0.0)
        one = torch.tensor(1.0)
        row1 = torch.stack([torch.cos(self.Joint_angle), -torch.sin(self.Joint_angle), zero, self.Link_length])
        row2 = torch.stack([torch.sin(self.Joint_angle)*torch.cos(self.Link_twist), torch.cos(self.Joint_angle)*torch.cos(self.Link_twist), -torch.sin(self.Link_twist), -self.Link_offset*torch.sin(self.Link_twist)])
        row3 = torch.stack([torch.sin(self.Joint_angle)*torch.sin(self.Link_twist), torch.cos(self.Joint_angle)*torch.sin(self.Link_twist), torch.cos(self.Link_twist), self.Link_offset*torch.cos(self.Link_twist)])
        row4 = torch.stack([zero, zero, zero, one])
        matrix = torch.stack([row1, row2, row3, row4])
        
        if self.print_require ==True:
            print(matrix)
        return matrix
    
    def assemble_joint(self):
        return rtb.RevoluteMDH(alpha=self.Link_twist, a=self.Link_length, d=self.Link_offset)
def matrix_to_pose6d(matrix):
    # 1. 提取平移向量 (x, y, z)
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    
    # 2. 提取旋转矩阵元素
    r11, r21, r31 = matrix[0, 0], matrix[1, 0], matrix[2, 0]
    r32, r33 = matrix[2, 1], matrix[2, 2]
    
    # 3. 计算欧拉角 (假设使用 ZYX 顺序，即 Roll-Pitch-Yaw)
    # 注意：这里全部使用 torch 的方法，保证梯度可以反向传播
    pitch = torch.atan2(-r31, torch.sqrt(r11**2 + r21**2))
    yaw = torch.atan2(r21, r11)
    roll = torch.atan2(r32, r33)
    
    # 将其组合成一个长度为 6 的张量返回
    return torch.stack([x, y, z, roll, pitch, yaw])

def matrix_to_pose6d_np(matrix):
    x, y, z = matrix[0, 3], matrix[1, 3], matrix[2, 3]
    r11, r21, r31 = matrix[0, 0], matrix[1, 0], matrix[2, 0]
    r32, r33 = matrix[2, 1], matrix[2, 2]
    pitch = np.arctan2(-r31, np.sqrt(r11**2 + r21**2))
    yaw = np.arctan2(r21, r11)
    roll = np.arctan2(r32, r33)
    return np.array([x, y, z, roll, pitch, yaw])

def caculate_matrixes(inputs, print_require=False):
    angles = list(map(float, inputs.split()))
    shoulder_pan = Joint(0, 0, 0.0624, angles[0])
    shoulder_lift = Joint(-90, 0.035, 0, angles[1])
    elbow_flex = Joint(0, 0.116, 0, angles[2])
    wrist_flex = Joint(0, 0.135, 0, angles[3])
    wrist_roll = Joint(-90, 0, 0.061, angles[4])
    return shoulder_pan.trans_matrix() @ shoulder_lift.trans_matrix() @ elbow_flex.trans_matrix() @ wrist_flex.trans_matrix() @ wrist_roll.trans_matrix()