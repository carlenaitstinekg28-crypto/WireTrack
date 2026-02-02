import cv2
import numpy as np
from collections import deque

class ImageSegmentor:
    """图像分割器类，封装所有分割相关功能"""
    
    def __init__(self, config):
        self.config = config
        
        # 从配置中获取基础分割参数
        seg_config = config.get('segmentation', {})
        self.canny_thresh1 = seg_config.get('canny_thresh1', 50)
        self.canny_thresh2 = seg_config.get('canny_thresh2', 150)
        self.gaussian_blur_ksize = seg_config.get('gaussian_blur_ksize', 3)
        self.grabcut_iter = seg_config.get('grabcut_iter', 5)
        self.min_mask_block_size = seg_config.get('min_mask_block_size', 50)
        # 自动分割参数
        self.auto_seed_radius = seg_config.get('auto_seed_radius', 15)
        self.min_auto_points = seg_config.get('min_auto_points', 10)
        self.auto_grow_angle_diff = seg_config.get('auto_grow_angle_diff', 45)
        self.search_radius = seg_config.get('search_radius', 30)
        self.erosion_start_point_count = seg_config.get('erosion_start_point_count', 13500)
        
        # === 蔓延（补全）算法配置 ===
        prop_config = seg_config.get('propagation', {})
        self.prop_enable = prop_config.get('enable', True)
        self.prop_max_iters = prop_config.get('max_iterations', 10)
        self.prop_step = prop_config.get('step_size', 20)
        self.prop_search_r = prop_config.get('search_radius', 30)
        self.prop_pca_win = prop_config.get('local_pca_window', 30)
        self.prop_angle_dev = prop_config.get('max_angle_deviation', 45)
        
        # === 新增：连接算法参数 ===
        self.connect_gap_threshold = seg_config.get('connect_gap_threshold', 30)  # 最大连接距离
        self.connect_min_angle_diff = seg_config.get('connect_min_angle_diff', 60)  # 最小角度差
        
        # 蔓延部分的颜色配置 (BGR) - 默认为黄色
        self.prop_fill_color = tuple(prop_config.get('fill_color', (0, 255, 255)))
        self.prop_outline_color = tuple(prop_config.get('outline_color', (0, 200, 200)))
        
        # 基础显示参数
        self.display_widen = seg_config.get('display_widen', 10)
        self.alpha = seg_config.get('alpha', 0.6)
        self.outline_thickness = seg_config.get('outline_thickness', 1)
        self.outline_color = seg_config.get('outline_color', (0, 128, 0))
        self.fill_color = seg_config.get('fill_color', (0, 255, 0))
    
    def preprocess(self, img):
        """图像预处理"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.gaussian_blur_ksize, self.gaussian_blur_ksize), 0)
        return blur
    
    def compute_edges(self, img_gray):
        """计算边缘"""
        return cv2.Canny(img_gray, self.canny_thresh1, self.canny_thresh2, apertureSize=3)
    
    @staticmethod
    def estimate_pca_angle(pts):
        """估计PCA主方向角度"""
        if pts.shape[0] < 2:
            return 0.0
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        idx = np.argmax(eigvals)
        v = eigvecs[:, idx]
        angle = np.arctan2(v[1], v[0])
        return angle
    
    @staticmethod
    def normalize_angle(angle):
        """标准化角度到[-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def directional_region_growing_multi_seed(self, edge_img, seeds, max_angle_diff=50, pca_win=21):
        """多种子点方向性区域生长"""
        h, w = edge_img.shape
        visited = np.zeros((h, w), dtype=bool)
        mask = np.zeros((h, w), dtype=np.uint8)
        all_pts = []
        
        for sx, sy in seeds:
            y0, y1 = max(0, sy-pca_win//2), min(h, sy+pca_win//2+1)
            x0, x1 = max(0, sx-pca_win//2), min(w, sx+pca_win//2+1)
            sub = edge_img[y0:y1, x0:x1]
            ys, xs = np.where(sub > 0)
            if len(xs) > 0:
                pts = np.column_stack([xs + x0, ys + y0])
                all_pts.append(pts)
        
        if len(all_pts) == 0:
            return mask
        
        pts_concat = np.vstack(all_pts)
        main_angle = self.estimate_pca_angle(pts_concat)
        
        q = deque()
        for sx, sy in seeds:
            if sx < 0 or sx >= w or sy < 0 or sy >= h:
                continue
            if not visited[sy, sx]:
                visited[sy, sx] = True
                mask[sy, sx] = 255
                q.append((sx, sy))
        
        neighbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        
        while q:
            cx, cy = q.popleft()
            for dx, dy in neighbors:
                nx, ny = cx+dx, cy+dy
                if nx<0 or nx>=w or ny<0 or ny>=h:
                    continue
                if visited[ny, nx] or edge_img[ny, nx] == 0:
                    continue
                
                move_angle = np.arctan2(dy, dx)
                angle_diff = abs(np.degrees(self.normalize_angle(move_angle - main_angle)))
                if angle_diff < max_angle_diff or angle_diff > (180 - max_angle_diff):
                    visited[ny, nx] = True
                    mask[ny, nx] = 255
                    q.append((nx, ny))
        
        return mask
    
    def find_nearest_edge_seeds(self, edge_img, click, radius=12, max_seeds=8):
        """找到最近的边缘种子点"""
        h, w = edge_img.shape
        cx, cy = click
        x0, x1 = max(0, cx-radius), min(w, cx+radius+1)
        y0, y1 = max(0, cy-radius), min(h, cy+radius+1)
        sub = edge_img[y0:y1, x0:x1]
        ys, xs = np.where(sub > 0)
        if len(xs) == 0:
            return []
        
        pts = list(zip(xs + x0, ys + y0))
        pts.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        return pts[:max_seeds]
    
    def refine_with_grabcut(self, img, init_mask):
        """使用GrabCut精细化掩码"""
        h, w = init_mask.shape
        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[init_mask == 255] = cv2.GC_PR_FGD
        
        # 设置边界为背景
        gc_mask[0:3, :] = cv2.GC_BGD
        gc_mask[-3:, :] = cv2.GC_BGD
        gc_mask[:, 0:3] = cv2.GC_BGD
        gc_mask[:, -3:] = cv2.GC_BGD
        
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(img, gc_mask, None, bgdModel, fgdModel, 
                   self.grabcut_iter, cv2.GC_INIT_WITH_MASK)
        
        res_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        return res_mask
    
    def visualize_mask_inline(self, image, mask, secondary_mask=None):
        """
        可视化掩码
        :param image: 原图
        :param mask: 基础掩码（通常显示为绿色）
        :param secondary_mask: [可选] 第二层掩码（蔓延部分，通常显示为黄色）
        """
        vis_base = image.copy()
        if mask is None and secondary_mask is None:
            return vis_base
        
        overlay = vis_base.copy()
        
        # 膨胀核设置
        ksize = max(1, int(self.display_widen))
        if ksize % 2 == 0: ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

        # 1. 绘制基础掩码 (绿色)
        if mask is not None:
            mask_bin = (mask > 0).astype(np.uint8) * 255
            mask_wide = cv2.dilate(mask_bin, kernel, iterations=1)
            overlay[mask_wide == 255] = self.fill_color

        # 2. 绘制第二层掩码 (黄色 - 蔓延部分)
        if secondary_mask is not None:
            sec_bin = (secondary_mask > 0).astype(np.uint8) * 255
            sec_wide = cv2.dilate(sec_bin, kernel, iterations=1)
            # 覆盖在基础掩码之上
            overlay[sec_wide == 255] = self.prop_fill_color

        # 3. 混合图像
        vis = cv2.addWeighted(overlay, self.alpha, vis_base, 1.0 - self.alpha, 0)
        
        # 4. 绘制轮廓 (基础)
        if mask is not None:
            contours_base, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_base:
                cv2.drawContours(vis, contours_base, -1, self.outline_color, 
                            thickness=self.outline_thickness, lineType=cv2.LINE_AA)

        # 5. 绘制轮廓 (蔓延)
        if secondary_mask is not None:
            contours_sec, _ = cv2.findContours((secondary_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_sec:
                cv2.drawContours(vis, contours_sec, -1, self.prop_outline_color, 
                            thickness=self.outline_thickness, lineType=cv2.LINE_AA)
        
        return vis

    # ================= 新增：掩码连接方法 =================
    
    def connect_broken_mask(self, mask, edges, max_gap=30, min_angle_diff=60):
        """
        连接断开的掩码部分
        参数:
            mask: 当前掩码
            edges: 边缘图
            max_gap: 最大连接距离
            min_angle_diff: 最小角度差（避免连接垂直物体）
        返回:
            连接后的掩码
        """
        if mask is None:
            return None
            
        h, w = mask.shape
        
        # 找到掩码的连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
        
        if num_labels < 3:  # 背景 + 1个连通域，不需要连接
            return mask
        
        # 获取所有连通域（排除背景）
        components = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 10:  # 忽略太小的区域
                # 获取该连通域的像素点
                y, x = np.where(labels == i)
                if len(x) > 0:
                    # 计算连通域的方向
                    pts = np.column_stack([x, y])
                    angle = self.estimate_pca_angle(pts)
                    
                    # 找到连通域的端点
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    proj = pts[:, 0] * cos_a + pts[:, 1] * sin_a
                    
                    min_idx = np.argmin(proj)
                    max_idx = np.argmax(proj)
                    
                    endpoints = [pts[min_idx], pts[max_idx]]
                    components.append({
                        'id': i,
                        'points': pts,
                        'endpoints': endpoints,
                        'angle': angle,
                        'centroid': centroids[i]
                    })
        
        # 如果只有1个连通域，不需要连接
        if len(components) < 2:
            return mask
        
        # 尝试连接断开的连通域
        connected_mask = mask.copy()
        
        for i in range(len(components)):
            for j in range(i+1, len(components)):
                comp1, comp2 = components[i], components[j]
                
                # 检查两个连通域的角度是否一致（导线方向）
                angle_diff = abs(np.degrees(self.normalize_angle(comp1['angle'] - comp2['angle'])))
                angle_diff = min(angle_diff, 180 - angle_diff)  # 考虑反向情况
                
                if angle_diff > min_angle_diff:  # 角度差异太大，可能是不同的物体
                    continue
                
                # 计算最近端点距离
                min_dist = float('inf')
                best_pair = None
                
                for ep1 in comp1['endpoints']:
                    for ep2 in comp2['endpoints']:
                        dist = np.sqrt((ep1[0]-ep2[0])**2 + (ep1[1]-ep2[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (ep1, ep2)
                
                # 如果距离在阈值内，尝试连接
                if min_dist < max_gap:
                    # 检查端点之间是否有边缘
                    ep1, ep2 = best_pair
                    x1, y1 = int(ep1[0]), int(ep1[1])
                    x2, y2 = int(ep2[0]), int(ep2[1])
                    
                    # 创建连接线
                    line_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
                    
                    # 检查连接线上是否有边缘点
                    line_edges = cv2.bitwise_and(line_mask, edges)
                    edge_count = np.count_nonzero(line_edges)
                    
                    # 如果有足够的边缘点，说明可以连接
                    if edge_count > max(3, min_dist / 5):
                        # 使用形态学操作连接
                        kernel_size = int(min_dist / 2) + 1
                        if kernel_size % 2 == 0:
                            kernel_size += 1
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                        
                        # 合并两个连通域
                        comp1_mask = (labels == comp1['id']).astype(np.uint8) * 255
                        comp2_mask = (labels == comp2['id']).astype(np.uint8) * 255
                        combined = cv2.bitwise_or(comp1_mask, comp2_mask)
                        
                        # 使用闭运算连接
                        connected = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
                        connected_mask = cv2.bitwise_or(connected_mask, connected)
        
        return connected_mask


    # ================= 蔓延（补全）相关方法 =================

    def get_mask_endpoints(self, mask):
            """
            找到当前Mask所有连通域的端点（支持断开的多个片段）
            返回: List of [(point, dir_vec), ...]
            """
            if mask is None:
                return []

            # 1. 连通域分析：将断开的部分区分开
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
            
            endpoints_info = []
            
            # 遍历每一个连通域（忽略背景 0）
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 10:  # 忽略太小的噪点
                    continue
                    
                # 获取该连通域的所有点
                pts_y, pts_x = np.where(labels == i)
                pts = np.column_stack([pts_x, pts_y])
                
                # 计算该片段的质心
                comp_center = centroids[i]
                
                # --- 对该片段进行 PCA 分析找端点 ---
                if len(pts) < 5:
                    continue

                angle = self.estimate_pca_angle(pts)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                
                # 投影到主轴
                proj = pts[:, 0] * cos_a + pts[:, 1] * sin_a
                min_idx = np.argmin(proj)
                max_idx = np.argmax(proj)
                
                # 检查该片段的两个端点
                for idx in [min_idx, max_idx]:
                    tip_pt = pts[idx]
                    
                    # 局部PCA优化方向
                    dists = np.sum((pts - tip_pt)**2, axis=1)
                    local_mask = dists < (self.prop_pca_win ** 2)
                    local_pts = pts[local_mask]
                    
                    if len(local_pts) < 5:
                        continue
                        
                    local_angle = self.estimate_pca_angle(local_pts)
                    vx, vy = np.cos(local_angle), np.sin(local_angle)
                    
                    # 关键：确保方向是指向该连通域外部的
                    # 逻辑：Tip + Vector 应该离该连通域的质心更远
                    check_p1 = tip_pt + np.array([vx, vy]) * 10
                    check_p2 = tip_pt - np.array([vx, vy]) * 10
                    
                    d1 = np.sum((check_p1 - comp_center)**2)
                    d2 = np.sum((check_p2 - comp_center)**2)
                    
                    if d2 > d1: 
                        vx, vy = -vx, -vy
                        
                    endpoints_info.append((tip_pt, (vx, vy)))
                    
            return endpoints_info
    def extend_mask_iteratively(self, img_shape, edges, current_mask, debug_mode=False):
        """
        对当前Mask进行迭代式蔓延补全 (带调试可视化)
        新增约束：避免蔓延到垂直物体
        """
        if not self.prop_enable or current_mask is None:
            return current_mask

        h, w = img_shape[:2]
        accumulated_mask = current_mask.copy()
        
        # === DEBUG: 创建调试画布 ===
        if debug_mode:
            # 将边缘图转为彩色，方便画彩色的调试信息
            debug_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # 把已有的 Mask (绿色) 叠加上去，方便看起点
            debug_vis[accumulated_mask > 0] = [0, 255, 0]

        for i in range(self.prop_max_iters):
            # 1. 寻找当前Mask的生长端点
            tips = self.get_mask_endpoints(accumulated_mask)
            new_seeds = []
            
            if debug_mode:
                print(f"--- Iter {i} ---")
                if not tips:
                    print("  No tips found.")

            for tip_pt, (vx, vy) in tips:
                cx, cy = tip_pt
                
                # 2. 预测新位置 (向外延伸 step 像素)
                pred_x = int(cx + vx * self.prop_step)
                pred_y = int(cy + vy * self.prop_step)

                # === DEBUG: 可视化预测行为 ===
                if debug_mode:
                    # 画蓝色箭头：表示预测的方向和步长
                    cv2.arrowedLine(debug_vis, (int(cx), int(cy)), (pred_x, pred_y), (255, 0, 0), 2, tipLength=0.3)
                    # 画青色空心圆：表示搜索范围 (search_radius)
                    cv2.circle(debug_vis, (pred_x, pred_y), self.prop_search_r, (255, 255, 0), 1)

                # 停止条件1: 超出图像边界
                if not (0 <= pred_x < w and 0 <= pred_y < h):
                    continue
                
                # 3. 在预测点周围寻找边缘种子
                local_seeds = self.find_nearest_edge_seeds(
                    edges, (pred_x, pred_y), 
                    radius=self.prop_search_r, max_seeds=5
                )
                
                # === 新增：过滤不符合导线方向的种子 ===
                filtered_seeds = []
                for sx, sy in local_seeds:
                    # 停止条件2: 新种子已经被包含在Mask里了
                    if accumulated_mask[sy, sx] > 0:
                        continue
                    
                    # 检查种子点方向是否与导线方向一致
                    # 获取种子点附近的局部边缘方向
                    seed_angle = self._get_local_edge_angle(edges, (sx, sy), radius=10)
                    if seed_angle is not None:
                        # 计算种子点方向与导线方向的差异
                        seed_dir = np.array([np.cos(seed_angle), np.sin(seed_angle)])
                        tip_dir = np.array([vx, vy])
                        
                        # 计算夹角
                        dot_product = np.dot(seed_dir, tip_dir)
                        angle_diff = np.degrees(np.arccos(np.clip(abs(dot_product), 0.0, 1.0)))
                        
                        # 如果角度差异太大（接近垂直），可能是其他物体
                        if angle_diff > 45:  # 45度阈值
                            if debug_mode:
                                print(f"    跳过垂直物体种子: 角度差 {angle_diff:.1f}度")
                            continue
                    
                    filtered_seeds.append((sx, sy))
                    
                    # === DEBUG: 画出找到的种子 ===
                    if debug_mode:
                        cv2.circle(debug_vis, (sx, sy), 2, (0, 0, 255), -1)
                
                new_seeds.extend(filtered_seeds)
            
            if not new_seeds:
                # if debug_mode: print("  No new seeds found, stopping.")
                break
                
            # 4. 基于新种子进行生长
            new_chunk_mask = self.directional_region_growing_multi_seed(
                edges, new_seeds, 
                max_angle_diff=self.prop_angle_dev, 
                pca_win=self.prop_pca_win
            )
            
            # 停止条件3: 如果生长的区域非常小，则停止
            diff = cv2.bitwise_and(new_chunk_mask, cv2.bitwise_not(accumulated_mask))
            new_pixel_count = np.count_nonzero(diff)
            
            if debug_mode:
                if new_pixel_count > 0:
                    debug_vis[diff > 0] = [0, 165, 255]
                    print(f"  Grown {new_pixel_count} pixels.")
                else:
                    print("  Seeds found but failed to grow (Angle mismatch?).")
            
            if new_pixel_count < 5:
                # print("    新增区域过小，停止")
                break
                
            # 合并 Mask
            accumulated_mask = cv2.bitwise_or(accumulated_mask, new_chunk_mask)

        # === DEBUG: 显示调试窗口 ===
        if debug_mode:
            cv2.imshow("Debug Propagation", debug_vis)
            # cv2.waitKey(0)
            
        return accumulated_mask

    def _get_local_edge_angle(self, edges, point, radius=10):
        """获取局部边缘方向"""
        h, w = edges.shape
        x, y = int(point[0]), int(point[1])
        
        x0 = max(0, x - radius)
        x1 = min(w, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(h, y + radius + 1)
        
        patch = edges[y0:y1, x0:x1]
        pts_y, pts_x = np.where(patch > 0)
        
        if len(pts_x) < 3:
            return None
        
        pts = np.column_stack([pts_x + x0, pts_y + y0])
        return self.estimate_pca_angle(pts)

    # =======================================================
    
    def auto_image_segmentation(self, image, edges, wire_tracker, current_pose_matrix, 
                                rvec, tvec, K, dist_coeffs, extra_seeds=None):
            """
            自动图像分割（带磁吸修正 + 热身机制 + 噪点过滤）
            """
            # 1. 获取原始投影种子点 (从3D跟踪器)
            raw_seeds = wire_tracker.get_projected_seeds(current_pose_matrix, image.shape, 
                                                    rvec, tvec, K, dist_coeffs)
            
            # 合并额外种子
            if extra_seeds and len(extra_seeds) > 0:
                if not isinstance(raw_seeds, list):
                    raw_seeds = list(raw_seeds)
                raw_seeds.extend(extra_seeds)
            
            if len(raw_seeds) < self.min_auto_points:
                return None
            
            # 2. 种子磁吸修正
            h, w = edges.shape
            snapped_seeds = []
            
            # 为了提高效率，如果种子太多，可以进行降采样
            sample_step = 1 if len(raw_seeds) < 100 else 3
            sample_seeds = raw_seeds[::sample_step]
            
            for (x, y) in sample_seeds:
                x, y = int(x), int(y)
                if not (0 <= x < w and 0 <= y < h):
                    continue
                
                if edges[y, x] > 0:
                    snapped_seeds.append((x, y))
                    continue
                
                x0, x1 = max(0, x - self.search_radius), min(w, x + self.search_radius)
                y0, y1 = max(0, y - self.search_radius), min(h, y + self.search_radius)
                patch = edges[y0:y1, x0:x1]
                ey, ex = np.nonzero(patch)
                
                if len(ex) > 0:
                    dists = (ex + x0 - x)**2 + (ey + y0 - y)**2
                    min_idx = np.argmin(dists)
                    if dists[min_idx] < self.search_radius**2:
                        best_x = ex[min_idx] + x0
                        best_y = ey[min_idx] + y0
                        snapped_seeds.append((best_x, best_y))
            
            if len(snapped_seeds) < 5:
                final_seeds = raw_seeds
            else:
                final_seeds = list(set(snapped_seeds))
            
            # 3. 使用修正后的种子进行生长
            selected_mask = self.directional_region_growing_multi_seed(edges, final_seeds, 
                                                                    max_angle_diff=self.auto_grow_angle_diff, 
                                                                    pca_win=21)
            
            if selected_mask is None or np.count_nonzero(selected_mask) < 10:
                return None
            
            # 4. 连接断开的掩码部分
            selected_mask = self.connect_broken_mask(selected_mask, edges, 
                                                    max_gap=self.connect_gap_threshold,
                                                    min_angle_diff=self.connect_min_angle_diff)
            
            # 5. 动态腐蚀策略
            current_tracker_points = len(wire_tracker.wire_points_world)
            final_mask = selected_mask # 默认
            
            if current_tracker_points > self.erosion_start_point_count:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                thinned_mask = cv2.erode(selected_mask, kernel, iterations=1)
                final_pixel_count = np.count_nonzero(thinned_mask)
                
                if final_pixel_count > 10:
                    final_mask = thinned_mask
                else:
                    print("自动分割警告: 腐蚀后掩码消失")
                    return None
            
            # === 6. [新增] 过滤细小噪点块 ===
            if final_mask is not None:
                # 连通域分析
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, 8, cv2.CV_32S)
                
                # 如果有多个连通域，进行过滤
                if num_labels > 1:
                    filtered_mask = np.zeros_like(final_mask)
                    has_valid_block = False
                    
                    for i in range(1, num_labels): # 0是背景，从1开始
                        area = stats[i, cv2.CC_STAT_AREA]
                        
                        # 只有大于配置阈值的块才保留
                        if area >= self.min_mask_block_size:
                            filtered_mask[labels == i] = 255
                            has_valid_block = True
                    
                    if has_valid_block:
                        return filtered_mask
                    else:
                        return None # 所有块都被过滤掉了
                
            return final_mask