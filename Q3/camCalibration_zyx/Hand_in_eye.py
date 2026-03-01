"""
流程:
    1.编写函数,读取每个图片的基座-相机变换矩阵,并添加在列表后
    2.编写函数,读取相机相关数据
    3.得到每个图片相机-标定板变换矩阵,并添加在列表后
        3.1图像预处理:读取图像转换为灰度图   cv2.cvtColor
        3.2角点检测:图像中寻找标定板内角点   cv2.findCheeseboardCorners   cv2.cornerSubPix
        3.3位姿求解                        cv2.solvePnP
        3.4切割得到两个矩阵,添加在两个列表   cv2.Rodrigues
    4.带入求解器,得到目标矩阵
"""

"""图像处理:
    使用cv2读取,作为一个numpy数组
"""

import cv2
import pandas as pd
import numpy as np
import json
from pathlib import Path



def split_matrixes(big_matrix, rot_matrix_list, trs_matrix_list):
    rot_matrix_list.append(big_matrix[:3, :3])
    trs_matrix_list.append(big_matrix[:3, 3:4])

def get_bg_matrixes(path, rot_matrix_list, trs_matrix_list):
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:  
            line = line.strip()
            data = json.loads(line)  #将该行转换为json对象
            matrix = np.array(data["matrix4x4"])
            split_matrixes(matrix, rot_matrix_list, trs_matrix_list)
    return rot_matrix_list, trs_matrix_list

def get_camera_parameters(path):
    with open(path, "r", encoding="utf-8") as f:
        dic = json.load(f)
        camera_matrix = np.array(dic["camera_matrix"], dtype=np.float64)
        distCoeffs = np.array(dic["dist_coeffs"], dtype=np.float64)
    return camera_matrix, distCoeffs

def get_objectPoints(cols=10, rows=7, square_size=0.02):
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp

def get_single_cb_matrixes(image, camera_matrix, distCoeffs, criteria,board_size=(10,7), zeroZone=(-1, -1), winSize=(11, 11)):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flag, corners = cv2.findChessboardCorners(gray_image, board_size, None)
    if flag:
        corners_fixed = cv2.cornerSubPix(gray_image, corners, winSize, zeroZone, criteria)
    else:
        print("ERROR in cv2.findChessboardCorners")
        return
    objp = get_objectPoints()
    flag_PnP, rotation_matirx, translation_matrix = cv2.solvePnP(objp, corners_fixed, camera_matrix, distCoeffs)
    rotation_matirx, _ = cv2.Rodrigues(rotation_matirx)
    return rotation_matirx, translation_matrix   




rot_bt = [] #3*3旋转矩阵  base->gripper
trs_bt = [] #3*1平移矩阵  camera->board
rot_cb = []
trs_cb = []

"""get base->gripper matrixes"""
current_dir = Path(__file__).parent
file_path = current_dir / "data_20251224_handineye" / "captures.json"
get_bg_matrixes(file_path, rot_bt, trs_bt)
rot_bt = np.array(rot_bt)
trs_bt = np.array(trs_bt)

"""get camera parameters"""
camera_data_path = current_dir / "data_20251224_handineye" / "target2cam_result.json"
camera_matrix, distCoeffs = get_camera_parameters(camera_data_path)

