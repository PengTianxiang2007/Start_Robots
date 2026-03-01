import numpy as np
from scipy.spatial.transform import Rotation as R



"""
用xtrinsics_head/hand     接收该函数返回值即可    
"""

def create_extrinsic_matrix(x: float, y: float, z: float, 
                            roll: float, pitch: float, yaw: float, 
                            degrees: bool = True) -> np.ndarray:
    """
    根据平移(x, y, z)和欧拉角旋转(roll, pitch, yaw)生成 4x4 变换矩阵。
    
    参数:
        x, y, z: 在 X, Y, Z 轴上的平移距离（通常单位为米）。
        roll: 绕 X 轴的旋转角（翻滚角）。
        pitch: 绕 Y 轴的旋转角（俯仰角）。
        yaw: 绕 Z 轴的旋转角（偏航角）。
        degrees: 角度输入是否为角度制（默认为 True，如果传弧度请设为 False）。
        
    返回:
        4x4 的齐次变换矩阵 (np.ndarray, dtype=float)
    """
    # 1. 初始化 4x4 单位矩阵
    transform_matrix = np.eye(4, dtype=float)
    
    # 2. 计算 3x3 旋转矩阵
    # 'xyz' 表示依次绕 X, Y, Z 轴进行旋转（外旋）
    rotation_matrix = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees).as_matrix()
    
    # 3. 将旋转和平移填入 4x4 矩阵
    transform_matrix[:3, :3] = rotation_matrix  # 左上角 3x3 赋值为旋转
    transform_matrix[:3, 3] = [x, y, z]         # 右上角 3x1 赋值为平移
    
    return transform_matrix