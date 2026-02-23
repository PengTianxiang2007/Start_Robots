import numpy as np
import roboticstoolbox as rtb
import torch
from torch import nn

class Joint:
    def __init__(self, Link_twist, Link_length, Link_offset, Joint_angle=0,print_require = True):
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
    def __init__(self, Link_twist, Link_length, Link_offset, Joint_angle=0,print_require = True):
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