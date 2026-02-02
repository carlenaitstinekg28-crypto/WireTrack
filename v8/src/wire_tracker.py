import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

class WireTracker:
    """导线跟踪器类（支持强度存储与高级几何过滤）"""
    
    def __init__(self, config):
        self.config = config
        wire_config = config.get('wire_tracking', {})
        self.dist_thresh = wire_config.get('wire_point_dist_thresh', 0.2)
        
        # 读取过滤参数
        filter_config = config.get('filter', {})
        self.min_z = filter_config.get('min_z', -100.0)
        self.max_z = filter_config.get('max_z', 100.0)
        self.min_linearity = filter_config.get('min_linearity', 0.6)
        self.filter_radius = filter_config.get('radius', 0.5)
        self.min_intensity = filter_config.get('min_intensity', 0)
        
        # 点云存储：初始化为 (0, 4) 形状，存储 [x, y, z, intensity]
        self.wire_points_world = np.empty((0, 4), dtype=np.float64)
        self.wire_points_timestamps = []
        self.kdtree = None
        
        self.min_curve_points = wire_config.get('min_curve_points', 20)
        self.curve_model = None
    
    def add_points(self, new_points_world, timestamp, current_pose_matrix, intensities=None):
        """添加新的导线点 (支持强度)"""
        if len(new_points_world) == 0:
            return np.array([])
        
        # 1. 组合 XYZ 和 Intensity
        if intensities is not None and len(intensities) == len(new_points_world):
            # 将强度 reshape 为 (N, 1) 并拼接到 XYZ 后面
            new_points_with_feat = np.hstack([new_points_world, intensities.reshape(-1, 1)])
        else:
            # 如果没有强度，默认填 0
            new_points_with_feat = np.hstack([new_points_world, np.zeros((len(new_points_world), 1))])
            
        # 2. 空间滤波 (仅使用 XYZ 坐标构建 KDTree)
        filtered_points_feat = self._spatial_filter(new_points_with_feat)
        
        if len(filtered_points_feat) == 0:
            return np.array([])
        
        # 3. 添加到历史点云
        self.wire_points_world = np.vstack([self.wire_points_world, filtered_points_feat])
        self.wire_points_timestamps.extend([timestamp] * len(filtered_points_feat))
        
        # 4. 重新构建KD树 (仅使用 XYZ)
        if len(self.wire_points_world) > 0:
            self.kdtree = KDTree(self.wire_points_world[:, :3])
            
        return filtered_points_feat[:, :3] # 返回 XYZ 用于显示
    
    def _spatial_filter(self, new_points_with_feat):
        """空间滤波：基于距离阈值过滤新点"""
        if len(self.wire_points_world) == 0:
            return new_points_with_feat
        
        # 提取新点的 XYZ 用于查询
        new_points_xyz = new_points_with_feat[:, :3]
        
        # KDTree 查询
        distances, _ = self.kdtree.query(new_points_xyz, k=1)
        
        # 保留距离小于阈值的点
        mask = distances < self.dist_thresh
        
        if np.any(mask):
            return new_points_with_feat[mask]
        else:
            return np.array([])
    
    def apply_geometry_correction(self):
        """
        [新增] 几何矫正：利用电力线物理约束修复由外参误差导致的漂移弯曲
        约束1: XY平面投影应为直线 (通过PCA强行拉直)
        约束2: 垂直方向应为平滑曲线 (通过二次多项式拟合平滑Z轴)
        """
        if len(self.wire_points_world) < 10:
            return np.array([])
            
        print(f"\n[几何矫正] 正在执行模型化修复 (输入点数: {len(self.wire_points_world)})...")
        
        # 提取坐标
        points = self.wire_points_world[:, :3]
        feats = self.wire_points_world[:, 3:] # 强度等其他特征
        
        # --- 步骤 1: XY平面“拉直” (消除水平漂移) ---
        # 计算 XY 重心
        xy = points[:, :2]
        mean_xy = np.mean(xy, axis=0)
        centered_xy = xy - mean_xy
        
        # PCA (主成分分析) 找主方向
        # 计算协方差矩阵并进行SVD分解
        cov = np.cov(centered_xy.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # 特征值最大的对应的特征向量即为导线走向 (主方向)
        # eigh 返回的是升序，所以取最后一个
        primary_direction = eigenvectors[:, 1] 
        
        # 将所有点投影到这条主直线上
        # 投影公式: projection = (dot(point, dir)) * dir
        projected_scalars = np.dot(centered_xy, primary_direction)
        rectified_xy = mean_xy + np.outer(projected_scalars, primary_direction)
        
        # 计算矫正产生的位移量（用于评估效果）
        drift_error = np.linalg.norm(xy - rectified_xy, axis=1).mean()
        print(f"  -> XY平面: 已将弯曲点云投影至主轴，平均修正漂移: {drift_error:.4f} m")
        
        # --- 步骤 2: Z轴“平滑” (拟合悬链线/抛物线) ---
        # 自变量 d: 点在主轴上的距离 (也就是上面的 projected_scalars)
        # 因变量 z: 点的原始 Z 坐标
        d = projected_scalars
        z = points[:, 2]
        
        # 使用 RANSAC 思想或者简单的二次多项式拟合 Z = a*d^2 + b*d + c
        # 这里用简单的最小二乘法 (np.polyfit)，如果噪点多可用 RANSAC
        try:
            # 拟合抛物线 (deg=2)
            coeffs = np.polyfit(d, z, 2) 
            poly_func = np.poly1d(coeffs)
            
            # 计算新的平滑 Z 值
            smooth_z = poly_func(d)
            
            z_correction = np.abs(z - smooth_z).mean()
            print(f"  -> Z 轴: 已拟合抛物线模型，平均平滑抖动: {z_correction:.4f} m")
            
            # --- 步骤 3: 组合修正后的坐标 ---
            rectified_points = np.hstack([
                rectified_xy, 
                smooth_z.reshape(-1, 1)
            ])
            
            # 更新内部存储的数据
            self.wire_points_world = np.hstack([rectified_points, feats])
            
            # 重建 KDTree 以便后续如果还需要搜索
            self.kdtree = KDTree(self.wire_points_world[:, :3])
            
            # 返回被移除的点(这里其实没有移除，只是移动了位置，返回空即可)
            # 或者返回修正前后的距离作为“被修改的量”用于显示
            return rectified_points
            
        except Exception as e:
            print(f"  [!] Z轴拟合失败，仅应用XY矫正。错误: {e}")
            self.wire_points_world[:, :2] = rectified_xy
            self.kdtree = KDTree(self.wire_points_world[:, :3])
            return self.wire_points_world[:, :3]

    def apply_advanced_filter(self):
        """
        逻辑：
        1. 只要不满足 Z轴范围 或 强度阈值 -> 剔除
        2. 满足1后，如果 线性度不够 或 方向与参考向量偏差过大 -> 剔除
        3. 收集所有被剔除的点用于可视化演示
        """
        count_start = len(self.wire_points_world)
        if count_start < 10:
            return np.array([])

        print(f"\n[高级过滤调试] 初始点数: {count_start}")
        
        # 用于收集所有被移除的点 (List of ndarray)
        removed_points_list = []
        
        # 读取配置
        ref_vec = np.array(self.config.get('filter.reference_vector', [1.0, 0.0, 0.0]))
        max_angle = self.config.get('filter.max_angle_deviation', 30.0)
        use_dir_filter = self.config.get('filter.use_direction_filter', False)
        
        # -----------------------------------------------------
        # 1. 基础过滤 (Z轴高度 + 强度)
        # -----------------------------------------------------
        z_vals = self.wire_points_world[:, 2]
        int_vals = self.wire_points_world[:, 3]
        
        # 找出保留的 Mask
        z_mask = (z_vals >= self.min_z) & (z_vals <= self.max_z)
        i_mask = (int_vals >= self.min_intensity)
        basic_mask = z_mask & i_mask
        
        # 收集第一阶段被剔除的点
        removed_part_1 = self.wire_points_world[~basic_mask]
        if len(removed_part_1) > 0:
            removed_points_list.append(removed_part_1[:, :3]) # 只取XYZ
            print(f"  -> [基础过滤] 剔除 {len(removed_part_1)} 个点 (高度或强度不达标)")
            
        # 剩下的候选点
        points_to_check = self.wire_points_world[basic_mask]
        
        if len(points_to_check) < 10:
            print(f"  [!!!] 警告: 基础过滤后点数过少 ({len(points_to_check)})，跳过几何过滤。")
            self.wire_points_world = points_to_check
            if removed_points_list:
                return np.vstack(removed_points_list)
            return np.array([])

        # -----------------------------------------------------
        # 2. 几何过滤 (线性度 + 方向)
        # -----------------------------------------------------
        print(f"  -> [几何计算] 对剩余 {len(points_to_check)} 个点进行 PCA 分析...")

        # 构建 Open3D 对象加速搜索
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_to_check[:, :3])
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        points_np = np.asarray(pcd.points)
        
        keep_indices = []    # 存放保留的索引
        removed_indices = [] # 存放剔除的索引
        
        ref_vec_norm = ref_vec / np.linalg.norm(ref_vec)
        
        # 遍历每个点
        for i in range(len(points_np)):
            # 搜索邻域 (Radius Search)
            [k, idx, _] = pcd_tree.search_radius_vector_3d(points_np[i], self.filter_radius)
            
            is_kept = False
            
            if k >= 4: # 至少需要几个点才能算PCA
                neighbors = points_np[idx, :]
                cov = np.cov(neighbors.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # 特征值从小到大: e1, e2, e3 (对应向量 v1, v2, v3)
                # 线性物体：e3 (主方向) 应该远大于 e2
                e1, e2, e3 = eigenvalues
                
                # 1. 线性度检查
                if e3 > 0:
                    linearity = (e3 - e2) / e3
                    if linearity > self.min_linearity:
                        # 2. 方向检查
                        if use_dir_filter:
                            curr_dir = eigenvectors[:, 2] # 主方向
                            dot_val = np.abs(np.dot(curr_dir, ref_vec_norm))
                            dot_val = np.clip(dot_val, 0, 1)
                            angle_deg = np.degrees(np.arccos(dot_val))
                            
                            if angle_deg <= max_angle:
                                is_kept = True
                        else:
                            is_kept = True # 不查方向，只查线性度
            
            if is_kept:
                keep_indices.append(i)
            else:
                removed_indices.append(i)
        
        # 收集第二阶段被剔除的点
        if len(removed_indices) > 0:
            removed_part_2 = points_to_check[removed_indices]
            removed_points_list.append(removed_part_2[:, :3])
            print(f"  -> [几何过滤] 剔除 {len(removed_indices)} 个点 (杂乱或方向错误)")

        # -----------------------------------------------------
        # 3. 更新与返回
        # -----------------------------------------------------
        # 更新类内部存储的点云为最终结果
        self.wire_points_world = points_to_check[keep_indices]
        print(f"  -> 最终剩余导线点数: {len(self.wire_points_world)}")       
        
        # 返回所有被移除的点 (合并列表)
        if removed_points_list:
            return np.vstack(removed_points_list)
        else:
            return np.array([]) 

    def get_projected_seeds(self, current_pose_matrix, img_shape, rvec, tvec, K, dist_coeffs):
        if len(self.wire_points_world) == 0:
            return []
        
        # 仅使用 XYZ 进行投影
        xyz_world = self.wire_points_world[:, :3]
        
        T_world_to_current = np.linalg.inv(current_pose_matrix)
        points_homo = np.hstack([xyz_world, np.ones((len(xyz_world), 1))])
        points_current = (T_world_to_current @ points_homo.T).T[:, :3]
        
        imgpts, _, mask = self._project_points(points_current, rvec, tvec, K, dist_coeffs)
        h, w = img_shape[:2]
        valid = mask & (imgpts[:, 0] >= 0) & (imgpts[:, 0] < w) & (imgpts[:, 1] >= 0) & (imgpts[:, 1] < h)
        
        if np.any(valid):
            seeds = imgpts[valid].astype(np.int32)
            seeds = np.unique(seeds, axis=0)
            return seeds.tolist()
        return []
    
    def _project_points(self, points, rvec, tvec, K, dist_coeffs):
        import cv2
        imgpts, _ = cv2.projectPoints(points.reshape(-1, 1, 3), rvec, tvec, K, dist_coeffs)
        imgpts = imgpts.reshape(-1, 2)
        
        R_mat, _ = cv2.Rodrigues(rvec)
        pts_cam = (R_mat @ points.T) + tvec
        Z = pts_cam[2, :]
        mask_in_front = Z > 1e-6
        
        return imgpts, Z, mask_in_front
    
    def get_all_points_world(self):
        """只返回 XYZ (N, 3) 用于可视化"""
        if len(self.wire_points_world) > 0:
            return self.wire_points_world[:, :3]
        return np.array([])
    
    def get_all_points_with_intensity(self):
        """返回 XYZI (N, 4) 用于保存"""
        if len(self.wire_points_world) > 0:
            return self.wire_points_world
        return np.array([])

    def get_point_count(self):
        return len(self.wire_points_world)
    
    def clear(self):
        self.wire_points_world = np.empty((0, 4), dtype=np.float64)
        self.wire_points_timestamps = []
        self.kdtree = None
        self.curve_model = None
    

