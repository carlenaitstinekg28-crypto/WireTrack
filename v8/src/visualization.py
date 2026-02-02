import open3d as o3d
import numpy as np
import cv2

class Visualizer3D:
    """3D可视化类"""
    
    def __init__(self, config):
        self.config = config
        self.vis = None
        self.pcd_history_vis = None
        self.pcd_new_vis = None
        self.pcd_bg_vis = None
        self.pcd_removed_vis = None
        self.pcd_depth_removed_vis = None
        # [修改] 改为列表存储多个球体网格
        self.selection_meshes = []

    def init_visualizer(self, window_name="3D Wire Tracking"):
        """初始化3D可视化窗口"""
        vis_config = self.config.get('visualization', {})
        width = vis_config.get('window_width', 800)
        height = vis_config.get('window_height', 600)
        top = vis_config.get('window_top', 50)
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, 
                            width=width, height=height, 
                            left=0, top=top)
        
        # 1. 历史点 (蓝)
        self.pcd_history_vis = o3d.geometry.PointCloud()  
        self.vis.add_geometry(self.pcd_history_vis)
        
        # 2. 当前帧新增点 (红)
        self.pcd_new_vis = o3d.geometry.PointCloud() 
        self.vis.add_geometry(self.pcd_new_vis)
        
        # 3. 背景点 (灰)
        self.pcd_bg_vis = o3d.geometry.PointCloud() 
        self.vis.add_geometry(self.pcd_bg_vis)

        # 4. 几何过滤点 (黄)
        self.pcd_removed_vis = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_removed_vis)

        # 5. 深度过滤点 (紫)
        self.pcd_depth_removed_vis = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd_depth_removed_vis)
        
        # 坐标轴
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
        
        return self.vis

    # ... [中间的 update_removed_points 等方法保持不变] ...
    
    def update_removed_points(self, points):
        if points is not None and len(points) > 0:
            self.pcd_removed_vis.points = o3d.utility.Vector3dVector(points)
            self.pcd_removed_vis.paint_uniform_color([1, 1, 0])  # Yellow
            self.vis.update_geometry(self.pcd_removed_vis)
        else:
            self.pcd_removed_vis.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self.vis.update_geometry(self.pcd_removed_vis)

    def update_depth_removed_points(self, points):
        if points is not None and len(points) > 0:
            self.pcd_depth_removed_vis.points = o3d.utility.Vector3dVector(points)
            self.pcd_depth_removed_vis.paint_uniform_color([1, 0, 1])  # Magenta
            self.vis.update_geometry(self.pcd_depth_removed_vis)
        else:
            self.pcd_depth_removed_vis.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self.vis.update_geometry(self.pcd_depth_removed_vis)

    def update_history_points(self, points):
        if points is not None and len(points) > 0:
            self.pcd_history_vis.points = o3d.utility.Vector3dVector(points)
            self.pcd_history_vis.paint_uniform_color([0, 0, 1])  # Blue
            self.vis.update_geometry(self.pcd_history_vis)
    
    def update_new_points(self, points):
        if points is not None and len(points) > 0:
            self.pcd_new_vis.points = o3d.utility.Vector3dVector(points)
            self.pcd_new_vis.paint_uniform_color([1, 0, 0])  # Red
            self.vis.update_geometry(self.pcd_new_vis)
        else:
            self.pcd_new_vis.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self.vis.update_geometry(self.pcd_new_vis)
    
    def update_background_points(self, points, downsample_factor=10):
        if points is not None and len(points) > 0:
            if downsample_factor > 1:
                points = points[::downsample_factor]
            self.pcd_bg_vis.points = o3d.utility.Vector3dVector(points)
            self.pcd_bg_vis.paint_uniform_color([0.8, 0.8, 0.8])  # Grey
            self.vis.update_geometry(self.pcd_bg_vis)
        else:
            self.pcd_bg_vis.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self.vis.update_geometry(self.pcd_bg_vis)

    # === [修改] 支持更新多个不同颜色的点 ===
    def update_selected_points(self, points_data, radius=0.01):
        """
        更新选中的点 -> 渲染为多个彩色球体
        :param points_data: 列表，每个元素为 (point_world, color_rgb)
                            point_world: (3,) np.array
                            color_rgb: [r, g, b] list or array
        """
        # 1. 移除旧的所有球体
        for mesh in self.selection_meshes:
            self.vis.remove_geometry(mesh, reset_bounding_box=False)
        self.selection_meshes = []

        # 2. 添加新球体
        if points_data:
            for point_world, color in points_data:
                if point_world is None or len(point_world) == 0:
                    continue
                
                # 获取中心坐标
                center = point_world.reshape(-1)[:3]
                
                # 创建球体
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                mesh.translate(center)
                mesh.paint_uniform_color(color)
                mesh.compute_vertex_normals() 
                
                self.vis.add_geometry(mesh, reset_bounding_box=False)
                self.selection_meshes.append(mesh)

    def poll_events(self):
        if self.vis: self.vis.poll_events()
    
    def update_renderer(self):
        if self.vis: self.vis.update_renderer()
    
    def reset_view_point(self):
        if self.vis: self.vis.reset_view_point(True)
    
    def destroy_window(self):
        if self.vis: self.vis.destroy_window()

# ... [create_2d_visualization 保持不变] ...
def create_2d_visualization(img, config, point_cloud_processor, segmentor, 
                           imgpts, depths, mask_in_front, selected_mask,
                           wire_tracker=None, current_pose_matrix=None,
                           rvec=None, tvec=None, K=None, dist_coeffs=None):
    # (此函数内容保持原样)
    flags = config.get('flags', {})
    vis_2d = img.copy()
    
    # 1. Draw Current
    if flags.get('show_current_frame_points', True) and selected_mask is not None:
        vis_2d = point_cloud_processor.overlay_points_on_image_with_mask(
            img, imgpts, depths, mask_in_front, selected_mask,
            segmentor=segmentor, point_color=point_cloud_processor.point_color
        )
    
    # 2. Draw History
    if flags.get('show_all_history_points', True) and wire_tracker is not None and current_pose_matrix is not None:
        hist_pts_world = wire_tracker.get_all_points_world()
        if len(hist_pts_world) > 0:
            T_world_to_current = np.linalg.inv(current_pose_matrix)
            points_homo = np.hstack([hist_pts_world, np.ones((len(hist_pts_world), 1))])
            points_current = (T_world_to_current @ points_homo.T).T[:, :3]
            
            # Project
            if rvec is None: rvec = point_cloud_processor.rvec
            if tvec is None: tvec = point_cloud_processor.tvec
            if K is None: K = point_cloud_processor.K
            if dist_coeffs is None: dist_coeffs = point_cloud_processor.dist_coeffs
            
            hist_imgpts, _, hist_mask = point_cloud_processor.project_points_to_image_with_extrinsics(
                points_current, rvec, tvec, K, dist_coeffs
            )
            
            h, w = img.shape[:2]
            valid_hist = hist_mask & (hist_imgpts[:, 0] >= 0) & (hist_imgpts[:, 0] < w) & \
                        (hist_imgpts[:, 1] >= 0) & (hist_imgpts[:, 1] < h)
            
            if np.any(valid_hist):
                valid_hist_pts = hist_imgpts[valid_hist].astype(np.int32)
                for pt in valid_hist_pts:
                    cv2.circle(vis_2d, tuple(pt), 2, point_cloud_processor.history_point_color, -1, cv2.LINE_AA)
    
    return vis_2d