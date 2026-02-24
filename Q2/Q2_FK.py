import numpy as np
import roboticstoolbox as rtb
from Q2_Joint import Joint 

"""定义计算函数"""
def caculate_matrixes(inputs, print_require = True):
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

    """计算并输出总变换矩阵"""
    final_matrix = shoulder_pan_trans@shoulder_lift_trans@elbow_flex_trans@wrist_flex_trans@wrist_roll_trans
    if print_require == True:
        print(f"The final result is \n{np.round(final_matrix, 4)}")
    return final_matrix

#inputs = input("Please input the angle")

#output_matirx = caculate_matrixes(inputs)
    