import cv2
import numpy as np
import open3d as o3d

class PointCloudProcessor:
    """点云处理器类"""
    
    def __init__(self, config):
        self.config = config
        self.K = config.get('camera.K')
        self.dist_coeffs = config.get('camera.dist_coeffs')
        self.rvec = config.get('camera.rotation_vector')
        self.tvec = config.get('camera.translation_vector')
        
        # 点云参数
        pc_config = config.get('point_cloud', {})
        self.point_radius = pc_config.get('point_radius', 2)
        self.point_color = pc_config.get('point_color', (0, 0, 255))
        self.history_point_color = pc_config.get('history_point_color', (255, 0, 0))
        self.point_alpha = pc_config.get('point_alpha', 0.8)
    
    def load_points_from_pcd(self, path):
        """从PCD文件加载点云"""
        try:
            pcd = o3d.io.read_point_cloud(path)
            if pcd.is_empty():
                print("PCD文件为空")
                return None, None
            
            pts = np.asarray(pcd.points, dtype=np.float64)
            intensity = None
            
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                intensity = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
            elif hasattr(pcd, 'intensity'):
                intensity = np.asarray(pcd.intensity, dtype=np.float64)
                
            return pts, intensity
            
        except Exception as e:
            print(f"读取 PCD 文件失败: {e}")
            return None, None
    
    def project_points_to_image_with_extrinsics(self, pts_lidar, rvec=None, tvec=None, K=None, dist_coeffs=None):
        """将点云投影到图像平面"""
        if rvec is None:
            rvec = self.rvec
        if tvec is None:
            tvec = self.tvec
        if K is None:
            K = self.K
        if dist_coeffs is None:
            dist_coeffs = self.dist_coeffs
        
        pts = np.asarray(pts_lidar, dtype=np.float64).reshape(-1, 3)
        R_mat, _ = cv2.Rodrigues(rvec)
        pts_cam = (R_mat @ pts.T) + tvec
        Z = pts_cam[2, :]
        mask_in_front = Z > 1e-6
        
        imgpts, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, tvec, K, dist_coeffs)
        imgpts = imgpts.reshape(-1, 2)
        
        return imgpts, Z, mask_in_front
    
    def overlay_points_on_image_with_mask(self, img, imgpts, depths, mask_in_front, image_mask,
                                         segmentor=None, point_color=None, 
                                         use_inliers_only=None):
        """在图像上叠加点云（带掩码过滤）"""
        if point_color is None:
            point_color = self.point_color
        
        # 使用配置中的标志
        use_inliers_only = self.config.get('flags.use_inliers_only', True) if use_inliers_only is None else use_inliers_only
        
        # 1. 绘制掩码（如果有）
        if image_mask is not None and segmentor is not None:
            vis = segmentor.visualize_mask_inline(img, image_mask)
        else:
            vis = img.copy()
        
        h, w = img.shape[:2]
        valid = mask_in_front & (imgpts[:, 0] >= 0) & (imgpts[:, 0] < w) & (imgpts[:, 1] >= 0) & (imgpts[:, 1] < h)
        
        # 2. 筛选要绘制的点
        if use_inliers_only:
            if image_mask is not None:
                # 情况 A: 有掩码，只画掩码内的点
                imgpts_int = imgpts[valid].astype(np.int32)
                mask_values = image_mask[imgpts_int[:, 1], imgpts_int[:, 0]]
                mask_valid = mask_values > 0
                valid_indices = np.where(valid)[0][mask_valid]
            else:
                # 情况 B: 有要求但无掩码，不画任何点
                return vis
        else:
            # 情况 C: 画所有点
            valid_indices = np.where(valid)[0]
        
        if len(valid_indices) == 0:
            return vis
        
        # 3. 绘制点
        pts2d = imgpts[valid_indices].astype(np.int32)
        depths_valid = depths[valid_indices]
        order = np.argsort(depths_valid)[::-1]  # 从远到近绘制
        
        overlay = vis.copy()
        for idx in order:
            x, y = pts2d[idx]
            cv2.circle(overlay, (int(x), int(y)), self.point_radius, point_color, -1, lineType=cv2.LINE_AA)
        
        vis = cv2.addWeighted(overlay, self.point_alpha, vis, 1.0 - self.point_alpha, 0)
        
        return vis