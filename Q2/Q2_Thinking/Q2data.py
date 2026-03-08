import numpy as np
from preparations import Joint
from preparations import matrix_to_pose6d_np
import pandas as pd
import random
import time

"""输入各个关节的角度,返回4*4矩阵"""
def caculate_matrixes(inputs, print_require = False): #inputs:输入的东西
    joint_angle = list(map(float, inputs.split()))
    joint_angle[1] -= 90                              #预处理输入数据

    """配置机械臂状态"""
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

"""生成随机数据"""
def generate_fk_dataset_matrix(num_samples=1000000, filename="fk_dataset_matrix.csv"):
    """
    随机生成关节角度并计算末端 4x4 位姿矩阵,提取前12个有效元素保存为CSV
    """
    # 1. 定义各个关节的物理限位 (单位: 度 Degree)
    # ⚠️ 请根据你实际机器人的机械限位修改这些数值
    joint_limits = [
        (-180.0, 180.0),  # Joint 0: shoulder_pan
        (-90.0, 90.0),    # Joint 1: shoulder_lift
        (-150.0, 150.0),  # Joint 2: elbow_flex
        (-180.0, 180.0),  # Joint 3: wrist_flex
        (-180.0, 180.0)   # Joint 4: wrist_roll
    ]
    
    dataset = []
    print(f"开始生成 {num_samples} 组 4x4 矩阵运动学数据...")
    start_time = time.time()
    

    for i in range(num_samples):  #通过for循环迭代
        # 2. 在限位范围内随机采样 5 个关节角度
        angles = [random.uniform(low, high) for low, high in joint_limits]  #得到随机的五个关节数据
        
        # 将角度格式化为字符串输入
        inputs_str = " ".join(map(str, angles))          #改变格式，契合caculate_matrix函数
        
        # 3. 调用正运动学求解，此时返回的是 4x4 的 numpy array
        T_matrix = caculate_matrixes(inputs_str, print_require=False)     #接收每一组的最终变换矩阵
        
        # 4. 提取矩阵的前 3 行 (共12个元素) 并展平为 1 维列表
        # T_matrix[:3, :] 会切片保留:
        # [[r11, r12, r13, tx],
        #  [r21, r22, r23, ty],
        #  [r31, r32, r33, tz]]
        matrix_flat = T_matrix[:3, :].flatten().tolist() #tolist:将numpy数组转换为列表
        
        # 5. 合并两个列表
        row_data = angles + matrix_flat
        dataset.append(row_data)     #每一组数据作为大的列表中的一项
        
        # 简单的进度打印
        if (i + 1) % 10000 == 0:
            print(f"已生成 {i + 1} / {num_samples} 组数据...")

    # 6. 使用 Pandas 导出为 CSV
    # 定义 CSV 的表头：5个关节角 + 9个旋转矩阵元素 + 3个平移元素
    columns = [
        'theta1', 'theta2', 'theta3', 'theta4', 'theta5', 
        'r11', 'r12', 'r13', 'tx', 
        'r21', 'r22', 'r23', 'ty', 
        'r31', 'r32', 'r33', 'tz'
    ]
    df = pd.DataFrame(dataset, columns=columns)  #创造dataframe对象
    df.to_csv(filename, index=False)
    
    end_time = time.time()
    print(f"数据生成完毕！共耗时 {end_time - start_time:.2f} 秒。")
    print(f"文件已保存为当前目录下的: {filename}")

# 在文件末尾执行生成逻辑
if __name__ == "__main__":
    # 生成 100000 组数据
    generate_fk_dataset_matrix(num_samples=1000000)