import cv2
import time
import sys
import os
import numpy as np
import open3d as o3d

# 添加项目根目录到Python路径q
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from src.image_segmentor import ImageSegmentor
from src.point_cloud_processor import PointCloudProcessor
from src.wire_tracker import WireTracker
from src.visualization import Visualizer3D, create_2d_visualization
from src.utils import load_lidar_poses, get_transform_matrix, find_closest_timestamp, get_image_pcd_pairs, extract_timestamp_from_filename

class WireTrackingApp:
    """导线跟踪主应用程序"""
    def __init__(self, config_path="D:/CloudProject/wire_project/v8/config/config.yaml"):
        # 加载配置
        self.config = Config(config_path)
        
        # 初始化各个模块
        self.segmentor = ImageSegmentor(self.config)
        self.point_processor = PointCloudProcessor(self.config)
        self.wire_tracker = WireTracker(self.config)
        self.visualizer_3d = Visualizer3D(self.config)
        
        # 加载位姿数据
        poses_file = self.config.get('paths.poses_file')
        self.lidar_poses = load_lidar_poses(poses_file)
        
        # 获取文件列表
        image_folder = self.config.get('paths.image_folder')
        pcd_folder = self.config.get('paths.pcd_folder')
        self.image_files, self.pcd_files = get_image_pcd_pairs(image_folder, pcd_folder)
        
        self.image_timestamps = [extract_timestamp_from_filename(f) for f in self.image_files]
        
        self.auto_play_next_frame = self.config.get('flags.auto_play_next_frame', False)
        self.save_results = self.config.get('flags.save_results', True)
        self.show_background_points_flag = self.config.get('flags.show_background_points', True)
        self.auto_play_delay = self.config.get('auto_play.auto_play_delay', 0.05)
        
        self.selected_mask = None
        self.refined_mask = None
        self.click_point = None
        self.continue_mode = False
        self.last_frame_edge_seeds = []

        # 用于存储最后一帧的数据以进行坐标系验证
        self.last_frame_raw_pts = None
        self.last_frame_pose = None

        # 累积存储被深度过滤掉的点（用于可视化）
        self.accumulated_depth_removed_points = []
        
        # [核心新增] 用于在帧间传递严格筛选后的球体点 (World Frame)
        self.current_frame_sphere_points = None
    
    def run(self):
        """运行主程序"""
        if not self.lidar_poses:
            print("错误：无法加载激光雷达位姿")
            return
        
        if len(self.image_files) == 0 or len(self.pcd_files) == 0:
            print("错误：未找到图像或点云文件")
            return
        
        # 初始化窗口
        self.visualizer_3d.init_visualizer(window_name="Wire Tracking")
        
        for i, (img_path, pcd_path) in enumerate(zip(self.image_files, self.pcd_files)):
            print(f"\n处理第 {i+1}/{len(self.image_files)} 帧")
            
            # 每一帧开始前，清空上一帧的球体点缓存
            self.current_frame_sphere_points = None
            
            img = cv2.imread(img_path)
            if img is None: continue
            
            pts, intensity = self.point_processor.load_points_from_pcd(pcd_path)
            if pts is None: continue
            
            img_timestamp = self.image_timestamps[i]
            closest_pose_timestamp = find_closest_timestamp(img_timestamp, self.lidar_poses.keys())
            position, quaternion = self.lidar_poses[closest_pose_timestamp]
            current_pose_matrix = get_transform_matrix(position, quaternion)

            self.last_frame_raw_pts = pts
            self.last_frame_pose = current_pose_matrix
            
            # 预计算当前帧所有点的世界坐标 (用于后续的距离查找)
            pts_homo = np.hstack([pts, np.ones((len(pts), 1))])
            all_pts_world = (current_pose_matrix @ pts_homo.T).T[:, :3]
            
            edges = self.segmentor.compute_edges(self.segmentor.preprocess(img))
            
            # 计算投影 (得到 2D 点坐标, 深度, 以及是否在前方)
            imgpts, depths, mask_in_front = self.point_processor.project_points_to_image_with_extrinsics(pts)
            
            # 默认：显示历史点(蓝色)
            current_total_points_xyz = self.wire_tracker.get_all_points_world()
            self.visualizer_3d.update_history_points(current_total_points_xyz)
            
            self.selected_mask = None
            mask_base = None
            mask_diff = None

            if i == 0:
                print("第一帧：手动分割模式")
                # 进入手动分割逻辑 (这里会生成 self.current_frame_sphere_points)
                self.handle_manual_segmentation(img, edges, imgpts, depths, mask_in_front, current_pose_matrix, pts, all_pts_world)
                
                if self.selected_mask is not None:
                    mask_base = self.selected_mask.copy()
            else:
                # === 第二帧及后续帧的处理逻辑 ===
                
                # 1. 常规自动分割 (获取绿色掩码)
                if self.last_frame_edge_seeds:
                    print(f"  -> 使用跨帧边界种子: {len(self.last_frame_edge_seeds)} 个")

                self.selected_mask = self.segmentor.auto_image_segmentation(
                    img, edges, self.wire_tracker, current_pose_matrix,
                    self.point_processor.rvec, self.point_processor.tvec,
                    self.point_processor.K, self.point_processor.dist_coeffs,
                    extra_seeds=self.last_frame_edge_seeds
                )
                if self.selected_mask is not None:
                    mask_base = self.selected_mask.copy()
                
                if self.selected_mask is not None:
                    self.selected_mask = self.segmentor.extend_mask_iteratively(
                        img.shape, edges, self.selected_mask, debug_mode=False 
                    )
                    if mask_base is not None:
                        mask_diff = cv2.bitwise_xor(self.selected_mask, mask_base)

                # 2. [核心修改] 基于 Tracker 历史点寻找当前帧的 3 个锚点并生成球体点云
                if self.selected_mask is not None and len(current_total_points_xyz) > 0:
                    self.process_next_frame_spheres(
                        current_total_points_xyz, self.selected_mask, 
                        imgpts, depths, mask_in_front, pts, all_pts_world
                    )
            
            # 更新边界种子 (供下一帧使用)
            self.last_frame_edge_seeds = []
            if self.selected_mask is not None:
                h, w = self.selected_mask.shape[:2]
                margin = 10
                ys, xs = np.where(self.selected_mask > 0)
                if len(xs) > 0:
                    border_mask = (xs < margin) | (xs > w - margin) | (ys < margin) | (ys > h - margin)
                    b_xs, b_ys = xs[border_mask], ys[border_mask]
                    if len(b_xs) > 0:
                        indices = np.arange(0, len(b_xs), step=5) 
                        self.last_frame_edge_seeds = list(zip(b_xs[indices], b_ys[indices]))

            # 提取点云并添加到 Tracker
            # 注意：这里的逻辑已经修改，会优先检查 self.current_frame_sphere_points
            added_points = self.extract_and_add_points(pts, intensity, imgpts, depths, mask_in_front, 
                                                      current_pose_matrix, img_timestamp)
            
            self.visualizer_3d.update_new_points(added_points)
            
            if self.show_background_points_flag:
                self.update_background_points(pts, current_pose_matrix)
            else:
                self.visualizer_3d.update_background_points(np.array([]))
            
            self.visualizer_3d.poll_events()
            self.visualizer_3d.update_renderer()
            if i == 0:
                self.visualizer_3d.reset_view_point()
            
            display_base = mask_base if mask_base is not None else self.selected_mask
            vis_background = self.segmentor.visualize_mask_inline(img, display_base, secondary_mask=mask_diff)
            
            vis_2d = create_2d_visualization(
                vis_background, self.config, self.point_processor, segmentor=None,                 
                imgpts=imgpts, depths=depths, mask_in_front=mask_in_front, selected_mask=self.selected_mask, 
                wire_tracker=self.wire_tracker, current_pose_matrix=current_pose_matrix,
                rvec=self.point_processor.rvec, tvec=self.point_processor.tvec,
                K=self.point_processor.K, dist_coeffs=self.point_processor.dist_coeffs
            )
            
            # 显示窗口
            vis_config = self.config.get('visualization', {})
            w = vis_config.get('window_width', 800)
            h = vis_config.get('window_height', 600)
            top = vis_config.get('window_top', 50)
            vis_2d_resized = cv2.resize(vis_2d, (w, h))
            window_name = 'Result'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, w, h)
            cv2.moveWindow(window_name, w, top)
            cv2.imshow(window_name, vis_2d_resized)
            
            if not self.handle_playback_control():
                break
        
        self.cleanup()

    def process_next_frame_spheres(self, hist_pts_world, current_mask, imgpts, depths, mask_in_front, raw_pts, all_pts_world):
            """
            [修正版] 修复了坐标系转换缺失的问题
            """
            # === 核心修复步骤：世界坐标 -> 当前帧雷达坐标 ===
            # 1. 获取当前位姿的逆矩阵 (World -> Body)
            if self.last_frame_pose is None:
                return
            current_pose_matrix = self.last_frame_pose
            T_world_to_current = np.linalg.inv(current_pose_matrix)
            
            # 2. 将历史点从世界坐标转换到当前帧坐标
            # hist_pts_world 是 (N, 3)
            pts_homo = np.hstack([hist_pts_world, np.ones((len(hist_pts_world), 1))])
            # (4, 4) @ (N, 4).T -> (4, N) -> .T -> (N, 4)
            points_current = (T_world_to_current @ pts_homo.T).T[:, :3]
            
            # 3. 再进行投影 (Body -> Image)
            hist_imgpts, _, hist_mask_front = self.point_processor.project_points_to_image_with_extrinsics(
                points_current, # 注意：这里传入转换后的点
                self.point_processor.rvec, 
                self.point_processor.tvec, 
                self.point_processor.K, 
                self.point_processor.dist_coeffs
            )
            # ============================================
            
            h, w = current_mask.shape[:2]
            valid_proj = hist_mask_front & (hist_imgpts[:, 0] >= 0) & (hist_imgpts[:, 0] < w) & \
                        (hist_imgpts[:, 1] >= 0) & (hist_imgpts[:, 1] < h)
            
            valid_indices = np.where(valid_proj)[0]
            if len(valid_indices) == 0:
                print("  [警告] 历史点投影未落在图像范围内")
                return

            # 增加容差 (保持之前的优化)
            kernel_size = 11 
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            mask_tolerant = cv2.dilate(current_mask, kernel, iterations=1)
            
            pts_int = hist_imgpts[valid_indices].astype(np.int32)
            
            # 检查
            in_mask_bool = mask_tolerant[pts_int[:, 1], pts_int[:, 0]] > 0
            final_hist_indices = valid_indices[in_mask_bool]
            
            # === 调试打印 ===
            if len(final_hist_indices) == 0:
                print(f"  [调试] 投影点总数: {len(valid_indices)}, 命中掩码数: 0")
                # 抽样打印几个坐标看看
                sample_pts = pts_int[:5]
                print(f"  [调试] 前5个投影坐标 (X,Y): {sample_pts.tolist()}")
                # 检查对应掩码位置的值
                mask_vals = [mask_tolerant[p[1], p[0]] for p in sample_pts if 0<=p[0]<w and 0<=p[1]<h]
                print(f"  [调试] 对应掩码值: {mask_vals}")
                return
            # ================

            # 找到 左、中、右 三个关键位置
            target_points_2d = hist_imgpts[final_hist_indices]
            sorted_indices = np.argsort(target_points_2d[:, 0])
            
            left_pt = target_points_2d[sorted_indices[0]]
            right_pt = target_points_2d[sorted_indices[-1]]
            center_pt = target_points_2d[sorted_indices[len(sorted_indices)//2]] 
            
            anchors_2d = [
                {'pos': left_pt,   'name': 'Left'},
                {'pos': center_pt, 'name': 'Center'},
                {'pos': right_pt,  'name': 'Right'}
            ]
            
            collected_sphere_points = [] 
            search_radius_px = 15 
            search_radius_3d = 0.20 
            
            print(f"  [自动化] 正在基于投影寻找3个锚点 (命中: {len(final_hist_indices)}个)...")

            for anchor in anchors_2d:
                ax, ay = anchor['pos']
                
                # A. 在 像素半径内，找当前帧离相机最近的点
                dists_sq = (imgpts[:, 0] - ax)**2 + (imgpts[:, 1] - ay)**2
                in_circle_mask = (dists_sq <= search_radius_px**2) & mask_in_front
                circle_indices = np.where(in_circle_mask)[0]
                
                if len(circle_indices) > 0:
                    # 找 Min Depth
                    local_depths = depths[circle_indices]
                    best_local_idx = np.argmin(local_depths)
                    best_global_idx = circle_indices[best_local_idx]
                    
                    # 获取该点的世界坐标 (作为 3D 锚点)
                    anchor_world = all_pts_world[best_global_idx]
                    
                    # B. 在 20cm 空间半径内，找所有邻域点
                    dists_3d = np.linalg.norm(all_pts_world - anchor_world, axis=1)
                    nearby_indices = np.where(dists_3d < search_radius_3d)[0]
                    
                    if len(nearby_indices) > 0:
                        nearby_pts_world = all_pts_world[nearby_indices]
                        collected_sphere_points.append(nearby_pts_world)
            
            # 4. 存入结果
            if len(collected_sphere_points) > 0:
                self.current_frame_sphere_points = np.vstack(collected_sphere_points)
                self.current_frame_sphere_points = np.unique(self.current_frame_sphere_points, axis=0)
                print(f"  -> 成功锁定 {len(self.current_frame_sphere_points)} 个导线点 (基于锚点球体)")
            else:
                print("  [警告] 未找到任何符合条件的球体邻域点")




    def handle_manual_segmentation(self, img, edges, imgpts, depths, mask_in_front, current_pose_matrix, raw_pts, all_pts_world):
            """
            手动分割逻辑：点击 -> 三锚点 -> 按 'n' 生成 20cm 邻域球体 -> 存入 self.current_frame_sphere_points
            """
            search_radius_3d = 0.20  # 20cm
            sphere_radius = 0.003
            
            self.indices_in_mask = []
            self.visual_circles = [] 
            
            self.current_anchor_data = [] # 存红黄青三个球的数据 (point, color)
            self.current_3d_anchors = []  # 只存坐标
            
            def update_3d_scene(selected_points_data):
                self.current_anchor_data = selected_points_data
                self.current_3d_anchors = [p[0] for p in selected_points_data]
                
                self.visualizer_3d.update_history_points(np.array([]))
                self.visualizer_3d.update_removed_points(np.array([]))
                self.visualizer_3d.update_depth_removed_points(np.array([]))
                self.visualizer_3d.update_background_points(all_pts_world)
                self.visualizer_3d.update_selected_points(selected_points_data, radius=sphere_radius)

            def on_mouse(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    print(f"\n[操作] 点击坐标: {x}, {y}")
                    offset = 30; check_radius = 15
                    targets = [
                        {'pos': (x, y), 'name': 'Center', 'color_bgr': (255, 0, 255), 'color_rgb': [1, 0, 1]},
                        {'pos': (x - offset, y), 'name': 'Left',   'color_bgr': (0, 255, 255), 'color_rgb': [1, 1, 0]},
                        {'pos': (x + offset, y), 'name': 'Right',  'color_bgr': (255, 255, 0), 'color_rgb': [0, 1, 1]}
                    ]
                    self.visual_circles = []
                    h, w = img.shape[:2]
                    all_seeds = []
                    for t in targets:
                        tx, ty = t['pos']
                        if 0 <= tx < w and 0 <= ty < h:
                            seeds = self.segmentor.find_nearest_edge_seeds(edges, (tx, ty), radius=15, max_seeds=8)
                            all_seeds.extend(seeds)
                            self.visual_circles.append((tx, ty, check_radius, t['color_bgr']))
                    
                    if not all_seeds:
                        print("  -> 未找到任何边缘种子，请重新点击")
                        update_3d_scene([])
                        self.indices_in_mask = []
                        return
                    
                    new_mask = self.segmentor.directional_region_growing_multi_seed(edges, all_seeds, max_angle_diff=50, pca_win=21)
                    if np.count_nonzero(new_mask) > 0:
                        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
                        new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)
                        new_mask = cv2.dilate(new_mask, kernel, iterations=1)
                    
                    if self.continue_mode and self.selected_mask is not None:
                        self.selected_mask = cv2.bitwise_or(self.selected_mask, new_mask)
                        self.continue_mode = False
                    else:
                        self.selected_mask = new_mask
                        self.refined_mask = None

                    valid_proj = mask_in_front & (imgpts[:, 0] >= 0) & (imgpts[:, 0] < w) & (imgpts[:, 1] >= 0) & (imgpts[:, 1] < h)
                    valid_indices = np.where(valid_proj)[0]
                    self.indices_in_mask = [] 
                    if len(valid_indices) > 0 and self.selected_mask is not None:
                        pts_int = imgpts[valid_indices].astype(np.int32)
                        in_mask_bool = self.selected_mask[pts_int[:, 1], pts_int[:, 0]] > 0
                        self.indices_in_mask = valid_indices[in_mask_bool]
                    
                    print(f"  -> 掩码覆盖了 {len(self.indices_in_mask)} 个 3D投影点 (蓝色)")
                    points_3d_to_draw = [] 
                    if len(self.indices_in_mask) > 0:
                        candidate_pts_2d = imgpts[self.indices_in_mask]
                        candidate_depths = depths[self.indices_in_mask]
                        for t in targets:
                            tx, ty = t['pos']
                            if not (0 <= tx < w and 0 <= ty < h): continue
                            dists_sq = (candidate_pts_2d[:, 0] - tx)**2 + (candidate_pts_2d[:, 1] - ty)**2
                            in_circle_mask = dists_sq <= (check_radius ** 2)
                            circle_indices_local = np.where(in_circle_mask)[0]
                            if len(circle_indices_local) > 0:
                                local_depths = candidate_depths[circle_indices_local]
                                best_local_idx = np.argmin(local_depths)
                                global_idx = self.indices_in_mask[circle_indices_local[best_local_idx]]
                                pt_world = all_pts_world[global_idx]
                                points_3d_to_draw.append((pt_world, t['color_rgb']))
                    update_3d_scene(points_3d_to_draw)

            cv2.namedWindow('Manual Seg', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Manual Seg', on_mouse)
            
            print("\n" + "="*50)
            print("【第一帧操作说明】")
            print(" 1. 点击图片 -> 生成红/黄/青三个锚点")
            print(f" 2. 按 'n' -> 查找锚点周围 {search_radius_3d}m 内的点 (变成红色)")
            print(" 3. 按 's' 确认并进入下一帧 (Tracker将只存储这些红点)")
            print("="*50 + "\n")
            
            if len(all_pts_world) > 0:
                self.visualizer_3d.update_history_points(all_pts_world)
                self.visualizer_3d.update_background_points(np.array([]))
                self.visualizer_3d.update_selected_points([])

            while True:
                vis_2d = img.copy()
                display_mask = self.refined_mask if self.refined_mask is not None else self.selected_mask
                if display_mask is not None:
                    vis_2d = self.segmentor.visualize_mask_inline(vis_2d, display_mask)
                if len(self.indices_in_mask) > 0:
                    pts_to_draw = imgpts[self.indices_in_mask].astype(np.int32)
                    for pt in pts_to_draw:
                        cv2.circle(vis_2d, tuple(pt), 2, (255, 0, 0), -1)
                for (cx, cy, r, color_bgr) in self.visual_circles:
                    cv2.circle(vis_2d, (int(cx), int(cy)), int(r), color_bgr, 2)
                cv2.imshow('Manual Seg', vis_2d)
                self.visualizer_3d.poll_events()
                self.visualizer_3d.update_renderer()

                key = cv2.waitKey(20) & 0xFF
                if key == ord('s') and self.selected_mask is not None:
                    # 退出时，清空可视化选择
                    self.visualizer_3d.update_selected_points([]) 
                    break
                elif key == ord('n'): 
                    if not self.current_3d_anchors:
                        print("  [提示] 请先点击生成3D锚点")
                    else:
                        nearby_indices = set()
                        for anchor in self.current_3d_anchors:
                            dists = np.linalg.norm(all_pts_world - anchor, axis=1)
                            found = np.where(dists < search_radius_3d)[0]
                            nearby_indices.update(found)
                        
                        if nearby_indices:
                            nearby_pts = all_pts_world[list(nearby_indices)]
                            purple_color = [1, 0, 0] # 红色
                            
                            # 保存这些点，供 Tracker 使用
                            self.current_frame_sphere_points = nearby_pts
                            
                            # 更新可视化
                            neighbor_spheres = [(pt, purple_color) for pt in nearby_pts]
                            combined_spheres = self.current_anchor_data + neighbor_spheres
                            self.visualizer_3d.update_selected_points(combined_spheres, radius=sphere_radius)
                            print(f"  -> 已选中 {len(nearby_pts)} 个点 (红色)")
                        else:
                            print("  -> 未找到任何邻域点")
                elif key == ord('g') and self.selected_mask is not None:
                    self.refined_mask = self.segmentor.refine_with_grabcut(img, self.selected_mask)
                elif key == ord('c'): 
                    self.continue_mode = True
                elif key == ord('q'):
                    self.visualizer_3d.destroy_window()
                    sys.exit(0)

            cv2.destroyWindow('Manual Seg')
            if self.refined_mask is not None: self.selected_mask = self.refined_mask

    def extract_and_add_points(self, pts, intensity, imgpts, depths, mask_in_front, current_pose_matrix, timestamp):
        """
        修改后的提取逻辑：
        优先检查是否有 self.current_frame_sphere_points (即通过球体筛选的高置信度点)。
        如果有，直接使用；否则回退到掩码提取。
        """
        pts_to_add_world = None
        
        # 1. 优先使用基于锚点球体的精确点集
        if self.current_frame_sphere_points is not None and len(self.current_frame_sphere_points) > 0:
            print(f"  [提取] 使用球体筛选点集 (数量: {len(self.current_frame_sphere_points)})")
            pts_to_add_world = self.current_frame_sphere_points
            
            # 由于 self.current_frame_sphere_points 已经是世界坐标，不需要再变换
            # 但我们需要对应的 intensity。
            # 这是一个难点，因为我们只存了坐标。
            # 简化处理：在这个模式下，暂时不存储 intensity，或者设为默认值。
            # 如果非常需要 intensity，需要反向查找索引，比较耗时。
            wire_intensities_current = None
            
        else:
            # 2. 回退到原来的逻辑 (基于掩码和深度的宽泛提取)
            if self.selected_mask is None:
                return np.array([])
            
            h, w = self.selected_mask.shape[:2]
            valid = mask_in_front & (imgpts[:, 0] >= 0) & (imgpts[:, 0] < w) & \
                    (imgpts[:, 1] >= 0) & (imgpts[:, 1] < h)
            
            if not np.any(valid):
                return np.array([])
            
            imgpts_int = imgpts[valid].astype(np.int32)
            mask_values = self.selected_mask[imgpts_int[:, 1], imgpts_int[:, 0]]
            mask_valid = mask_values > 0
            
            if not np.any(mask_valid):
                return np.array([])
            
            valid_indices_in_pts = np.where(valid)[0][mask_valid]
            
            # 深度分离策略
            candidate_depths = depths[valid_indices_in_pts]
            if len(candidate_depths) == 0: return np.array([])

            sorted_order = np.argsort(candidate_depths)
            sorted_depths = candidate_depths[sorted_order]
            min_d, max_d = sorted_depths[0], sorted_depths[-1]
            depth_span = max_d - min_d
            
            final_indices = valid_indices_in_pts
            
            if depth_span > 0.5:
                diffs = np.diff(sorted_depths)
                max_gap_idx = np.argmax(diffs)
                if diffs[max_gap_idx] > 0.74:
                    split_depth = (sorted_depths[max_gap_idx] + sorted_depths[max_gap_idx+1]) / 2.0
                    foreground_mask = candidate_depths < split_depth
                    if np.sum(foreground_mask) > 5:
                        final_indices = valid_indices_in_pts[foreground_mask]
            
            wire_points_3d_current = pts[final_indices]
            if intensity is not None and len(intensity) == len(pts):
                wire_intensities_current = intensity[final_indices]
            else:
                wire_intensities_current = None
                
            points_homo = np.hstack([wire_points_3d_current, np.ones((len(wire_points_3d_current), 1))])
            pts_to_add_world = (current_pose_matrix @ points_homo.T).T[:, :3]

        # 执行添加
        if pts_to_add_world is not None and len(pts_to_add_world) > 0:
            added_points = self.wire_tracker.add_points(
                pts_to_add_world, timestamp, current_pose_matrix, 
                intensities=wire_intensities_current
            )
            print(f"  -> Tracker新增点数: {len(added_points)}，总累计: {self.wire_tracker.get_point_count()}")
            return added_points
        else:
            return np.array([])

    def update_background_points(self, pts, current_pose_matrix, downsample_factor=10):
        if len(pts) == 0:
            self.visualizer_3d.update_background_points(np.array([]))
            return
        current_pts_homo = np.hstack([pts, np.ones((len(pts), 1))])
        pts_world = (current_pose_matrix @ current_pts_homo.T).T[:, :3]
        self.visualizer_3d.update_background_points(pts_world, downsample_factor)

    def handle_playback_control(self):
        if self.auto_play_next_frame:
            key = cv2.waitKey(int(self.auto_play_delay * 1000)) & 0xFF
            if key == ord('q'): return False
            elif key == ord('r'):
                self.wire_tracker.clear()
                print("已重置 Tracker")
            elif key == ord('p'):
                self.auto_play_next_frame = not self.auto_play_next_frame
        else:
            while True:
                self.visualizer_3d.poll_events()
                self.visualizer_3d.update_renderer()
                key = cv2.waitKey(10) & 0xFF
                if key == ord(' '): break
                elif key == ord('q'): return False
                elif key == ord('r'):
                    self.wire_tracker.clear()
                    print("已重置 Tracker")
                    break
                elif key == ord('p'):
                    self.auto_play_next_frame = True
                    break
        return True

    def cleanup(self):
        print("\n[结束] 正在执行最终处理流程...")
        # 1. 几何过滤
        if self.config.get('filter.enable', False):
            removed_points = self.wire_tracker.apply_advanced_filter()
            if len(removed_points) > 0:
                self.visualizer_3d.update_removed_points(removed_points)
        
        # 2. 几何矫正
        if self.config.get('flags.enable_geometry_correction', True): 
            self.wire_tracker.apply_geometry_correction()

        final_points = self.wire_tracker.get_all_points_world()
        self.visualizer_3d.update_history_points(final_points)
        
        print("\n>>> 按 'q' 键退出并保存 <<<")
        while True:
            self.visualizer_3d.poll_events()
            self.visualizer_3d.update_renderer()
            if (cv2.waitKey(10) & 0xFF) == ord('q'): break
        
        if self.save_results:
            results_folder = self.config.get('paths.results_folder', 'results')
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
            
            all_points_with_feat = self.wire_tracker.get_all_points_with_intensity()
            if len(all_points_with_feat) > 0:
                xyz = all_points_with_feat[:, :3]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                full_path = os.path.join(results_folder, "final_wire_world.pcd")
                o3d.io.write_point_cloud(full_path, pcd, write_ascii=True)
                print(f"已保存导线: {full_path}")
                
        self.visualizer_3d.destroy_window()
        cv2.destroyAllWindows()
        print("程序退出")

if __name__ == "__main__":
    app = WireTrackingApp()
    app.run()