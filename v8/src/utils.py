import numpy as np
import glob
import os
from pathlib import Path

def load_lidar_poses(poses_file):
    """加载激光雷达位姿"""
    poses = {}
    try:
        with open(poses_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    timestamp = float(parts[0])
                    position = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
                    quaternion = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float64)
                    poses[timestamp] = (position, quaternion)
    except Exception as e:
        print(f"加载位姿文件失败: {e}")
    return poses

def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵"""
    x, y, z, w = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)
    return R

def get_transform_matrix(position, quaternion):
    """获取变换矩阵"""
    R = quaternion_to_rotation_matrix(quaternion)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = position
    return T

def find_closest_timestamp(target_time, timestamps):
    """找到最接近的时间戳"""
    timestamps = np.array(list(timestamps))
    idx = np.argmin(np.abs(timestamps - target_time))
    return timestamps[idx]

def extract_timestamp_from_filename(filename):
    """从文件名提取时间戳"""
    return float(Path(filename).stem)

def get_image_pcd_pairs(image_folder, pcd_folder):
    """获取匹配的图像和点云文件对"""
    image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    pcd_files = sorted(glob.glob(os.path.join(pcd_folder, "*.pcd")))
    
    # 确保数量一致
    min_len = min(len(image_files), len(pcd_files))
    image_files = image_files[:min_len]
    pcd_files = pcd_files[:min_len]
    
    return image_files, pcd_files