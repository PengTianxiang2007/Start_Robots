import numpy as np
from scipy.spatial.transform import Rotation as R

def create_extrinsic_matrix(x: float, y: float, z: float, 
                            roll: float, pitch: float, yaw: float, 
                            degrees: bool = True) -> np.ndarray:

    transform_matrix = np.eye(4, dtype=float)

    #输入角度，并且创建旋转对象，转换为矩阵
    rotation_matrix = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees).as_matrix()
    
    transform_matrix[:3, :3] = rotation_matrix  # 左上角 3x3 赋值为旋转
    transform_matrix[:3, 3] = [x, y, z]         # 右上角 3x1 赋值为平移
    
    return transform_matrix