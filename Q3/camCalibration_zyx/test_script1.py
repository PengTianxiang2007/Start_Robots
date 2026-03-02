"""
测试:除读取图片得到矩阵外的其他代码的有效性
"""

import cv2
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os

"""由4*4变换矩阵切割为3*3旋转矩阵与3*1平移矩阵,并添加在分别的列表后"""
def split_matrixes(big_matrix, rot_matrix_list, trs_matrix_list):
    rot_matrix_list.append(big_matrix[:3, :3])
    trs_matrix_list.append(big_matrix[:3, 3:4])

"由3*3旋转矩阵和3*1平移矩阵拼接4*4变换矩阵"
def splice_matrix(rot_matrix, trs_matrix):
    output = np.eye(4)
    output[:3, :3] = rot_matrix
    output[:3, 3] = trs_matrix.reshape(3)
    return output

"""读取json文件中base->gripper矩阵,并切割添加在相应列表中"""
def get_bg_matrixes(path, rot_matrix_list, trs_matrix_list, inverse_matrix=False):
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:  
            line = line.strip()
            data = json.loads(line)  #将该行转换为json对象
            matrix = np.array(data["matrix4x4"])
            if inverse_matrix==True:
                matrix = np.linalg.inv(matrix)
            split_matrixes(matrix, rot_matrix_list, trs_matrix_list)
    return rot_matrix_list, trs_matrix_list

"""读取json文件中相机参数"""
def get_camera_parameters(path):
    with open(path, "r", encoding="utf-8") as f:
        dic = json.load(f)
        camera_matrix = np.array(dic["camera_matrix"], dtype=np.float64)
        distCoeffs = np.array(dic["dist_coeffs"], dtype=np.float64)
    return camera_matrix, distCoeffs

"""得到objectPoints参数"""
def get_objectPoints(cols=11, rows=8, square_size=0.02):
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    board_size = (cols, rows)
    return objp, board_size

def get_cb_matrixes(path, rotation_list, translation_list):
    with open(path, "r", encoding="utf-8") as f:
        dic = json.load(f)
        detections = dic["detections"]
        for img in detections:
            r_mat = np.array(img["rotation_matrix"], dtype=np.float64)
            t_vec = np.array(img["tvec_m"], dtype=np.float64).reshape(3, 1)
            rotation_list.append(r_mat)
            translation_list.append(t_vec)

def get_extrinsics_hand_matrix():
    
    rot_bt = [] #3*3旋转矩阵  base->gripper
    trs_bt = [] #3*1平移矩阵  camera->board
    rot_cb = []
    trs_cb = []

    """获得 base->gripper 相关矩阵"""
    current_path = Path(__file__).parent
    captures_path = current_path / "data_20251224_handineye" / "captures.json"
    get_bg_matrixes(captures_path, rot_bt, trs_bt, inverse_matrix=False)

    """得到 camera 参数矩阵"""
    camera_data_path = current_path / "data_20251224_handineye" / "target2cam_result.json"
    camera_matrix, distCoeffs = get_camera_parameters(camera_data_path)

    """得到 camera->board 相关矩阵"""
    get_cb_matrixes(camera_data_path, rot_cb, trs_cb)

    """计算,返回目标矩阵"""
    final_rotation_matrix, final_translation_matrix = cv2.calibrateHandEye(rot_bt, trs_bt, rot_cb, trs_cb)
    output = splice_matrix(final_rotation_matrix, final_translation_matrix)
    return output

def get_extrinsics_head_matrix():
    
    rot_bt = [] #3*3旋转矩阵  base->gripper
    trs_bt = [] #3*1平移矩阵  camera->board
    rot_cb = []
    trs_cb = []

    """获得 base->gripper 相关矩阵"""
    current_path = Path(__file__).parent
    captures_path = current_path / "data_20251229_handtoeye" / "captures.json"
    get_bg_matrixes(captures_path, rot_bt, trs_bt, inverse_matrix=True)

    """得到 camera 参数矩阵"""
    camera_data_path = current_path / "data_20251229_handtoeye" / "target2cam_result.json"
    camera_matrix, distCoeffs = get_camera_parameters(camera_data_path)

    """得到 camera->board 相关矩阵"""
    get_cb_matrixes(camera_data_path, rot_cb, trs_cb)

    """计算,返回目标矩阵"""
    final_rotation_matrix, final_translation_matrix = cv2.calibrateHandEye(rot_bt, trs_bt, rot_cb, trs_cb)
    output = splice_matrix(final_rotation_matrix, final_translation_matrix)
    return output