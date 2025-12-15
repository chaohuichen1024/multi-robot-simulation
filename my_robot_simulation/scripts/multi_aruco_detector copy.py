#!/usr/bin/env python3
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import ast
import tf
import tf2_ros
import geometry_msgs.msg
import math
from typing import List, Tuple
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion

class PerspectiveCorrector:
    def __init__(self):
        # 边缘检测参数
        self.canny_threshold = 50
        self.canny_threshold_max = 200
        self.hough_threshold = 50
        self.min_line_length = 100
        self.max_line_gap = 10
        
        # 最大检测直线数量
        self.max_lines_num = 12
        
    def auto_perspective_correction(self, image: np.ndarray) -> np.ndarray:
        """
        自动透视变换矫正主函数
        """
        if image is None:
            raise ValueError("输入图像为空")
            
        # 1. 预处理图像
        gray = self.preprocess_image(image)
        
        # 2. 边缘检测和直线提取
        lines = self.edge_detection_and_line_extraction(gray)
        
        if len(lines) < 4:
            rospy.logdebug("检测到的直线数量不足，无法进行透视变换")
            return image
        
        # 3. 提取和过滤关键点
        points = self.extract_and_filter_points(lines, image.shape)
        
        if len(points) < 4:
            rospy.logdebug("提取的关键点数量不足")
            return image
        
        # 4. 计算四个角点
        corners = self.calculate_four_corners(points, image.shape)
        
        if corners is None:
            rospy.logdebug("无法计算四个角点")
            return image
        
        # 5. 执行透视变换
        result = self.apply_perspective_transform(image, corners)
        
        return result
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 高斯模糊去噪
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return gray
    
    def edge_detection_and_line_extraction(self, gray: np.ndarray) -> List:
        """
        边缘检测和直线提取
        """
        lines = []
        canny_threshold = self.canny_threshold
        
        # 迭代调整Canny阈值，直到检测到的直线数量合适
        while canny_threshold <= self.canny_threshold_max:
            # Canny边缘检测
            edges = cv2.Canny(gray, canny_threshold, canny_threshold * 2)
            
            # 霍夫直线检测
            detected_lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, self.hough_threshold,
                minLineLength=self.min_line_length, 
                maxLineGap=self.max_line_gap
            )
            
            if detected_lines is None:
                canny_threshold += 10
                continue
                
            detected_lines = detected_lines.reshape(-1, 4).tolist()
            
            # 过滤过于贴近边缘的直线
            filtered_lines = self.filter_edge_lines(detected_lines, gray.shape)
            
            if len(filtered_lines) <= self.max_lines_num:
                lines = filtered_lines
                break
            else:
                canny_threshold += 10
        
        rospy.logdebug(f"最终检测到 {len(lines)} 条直线")
        return lines
    
    def filter_edge_lines(self, lines: List, image_shape: Tuple) -> List:
        """
        过滤过于贴近图像边缘的直线
        """
        height, width = image_shape
        margin = 20  # 边缘阈值
        
        filtered_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # 过滤水平或垂直的短直线
            if (abs(x1 - x2) < 10 and abs(y1 - y2) > 10) or \
               (abs(y1 - y2) < 10 and abs(x1 - x2) > 10):
                continue
            
            # 检查是否过于贴近边缘
            if (x1 < margin and x2 < margin) or \
               (x1 > width - margin and x2 > width - margin) or \
               (y1 < margin and y2 < margin) or \
               (y1 > height - margin and y2 > height - margin):
                continue
                
            filtered_lines.append(line)
        
        return filtered_lines
    
    def extract_and_filter_points(self, lines: List, image_shape: Tuple) -> List[Tuple]:
        """
        从直线中提取关键点并过滤
        """
        points = []
        
        # 提取所有线段的端点
        for line in lines:
            x1, y1, x2, y2 = line
            points.append((x1, y1))
            points.append((x2, y2))
        
        # 过滤相近的点
        filtered_points = self.filter_close_points(points)
        
        # 按到原点的距离排序
        filtered_points.sort(key=lambda p: p[0] + p[1])
        
        return filtered_points
    
    def filter_close_points(self, points: List[Tuple], threshold: int = 10) -> List[Tuple]:
        """
        过滤距离过近的点
        """
        filtered_points = []
        
        for i, point1 in enumerate(points):
            keep_point = True
            
            for j, point2 in enumerate(points):
                if i == j:
                    continue
                    
                distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
                
                if distance < threshold:
                    # 如果找到更接近原点的点，保留那个点
                    if (point2[0] + point2[1]) < (point1[0] + point1[1]):
                        keep_point = False
                        break
            
            if keep_point and point1 not in filtered_points:
                filtered_points.append(point1)
        
        return filtered_points
    
    def calculate_four_corners(self, points: List[Tuple], image_shape: Tuple) -> np.ndarray:
        """
        计算四个角点
        """
        if len(points) < 4:
            return None
        
        height, width = image_shape[0], image_shape[1]
        
        # 方法1：基于距离的方法
        corners1 = self.method1_distance_based(points.copy())
        
        # 方法2：基于倾斜方向的方法
        corners2 = self.method2_tilt_based(points.copy(), image_shape)
        
        # 如果两种方法都成功，取平均值
        if corners1 is not None and corners2 is not None:
            # 检查两个结果是否接近
            avg_corners = self.average_corners(corners1, corners2)
            return avg_corners
        elif corners1 is not None:
            return corners1
        elif corners2 is not None:
            return corners2
        else:
            return None
    
    def method1_distance_based(self, points: List[Tuple]) -> np.ndarray:
        """
        方法1：基于最大距离的角点计算
        """
        if len(points) < 4:
            return None
        
        # 左上和右下点（排序后的第一个和最后一个）
        left_top = points[0]
        right_down = points[-1]
        
        # 分离右上和左下点簇
        right_top_cluster = []
        left_down_cluster = []
        
        for point in points[1:-1]:  # 排除已经确定的两个点
            x, y = point
            
            # 右上点簇：x > 左上x 且 y < 右下y
            if x > left_top[0] and y < right_down[1]:
                right_top_cluster.append(point)
            
            # 左下点簇：y > 左上y 且 x < 右下x
            if y > left_top[1] and x < right_down[0]:
                left_down_cluster.append(point)
        
        if not right_top_cluster or not left_down_cluster:
            return None
        
        # 在点簇中寻找距离最远的点对
        max_distance = 0
        best_right_top = right_top_cluster[0]
        best_left_down = left_down_cluster[0]
        
        for rt_point in right_top_cluster:
            for ld_point in left_down_cluster:
                distance = (rt_point[0] - ld_point[0])**2 + (rt_point[1] - ld_point[1])**2
                if distance > max_distance:
                    max_distance = distance
                    best_right_top = rt_point
                    best_left_down = ld_point
        
        corners = np.array([
            left_top,
            best_right_top,
            right_down,
            best_left_down
        ], dtype=np.float32)
        
        return corners
    
    def method2_tilt_based(self, points: List[Tuple], image_shape: Tuple) -> np.ndarray:
        """
        方法2：基于倾斜方向的角点计算
        """
        if len(points) < 4:
            return None
        
        height, width = image_shape[0], image_shape[1]
        
        left_top = points[0]
        right_down = points[-1]
        
        # 分离点簇
        right_top_cluster = []
        left_down_cluster = []
        
        for point in points[1:-1]:
            x, y = point
            
            if x > left_top[0] and y < right_down[1]:
                right_top_cluster.append(point)
            
            if y > left_top[1] and x < right_down[0]:
                left_down_cluster.append(point)
        
        if not right_top_cluster or not left_down_cluster:
            return None
        
        # 判断图像倾斜方向
        image_state = self.determine_image_tilt(right_top_cluster, left_top)
        
        # 根据倾斜方向确定真正的右上和左下点
        if image_state == "lean_to_right":
            # 向右倾斜：右上点取横坐标最大，左下点取横坐标最小
            right_top_cluster.sort(key=lambda p: p[0], reverse=True)
            left_down_cluster.sort(key=lambda p: p[0])
        elif image_state == "lean_to_left":
            # 向左倾斜：右上点取纵坐标最小，左下点取纵坐标最大
            right_top_cluster.sort(key=lambda p: p[1])
            left_down_cluster.sort(key=lambda p: p[1], reverse=True)
        else:
            # 正常状态：使用默认排序
            right_top_cluster.sort(key=lambda p: p[0] + p[1])
            left_down_cluster.sort(key=lambda p: p[0] + p[1], reverse=True)
        
        true_right_top = right_top_cluster[0] if right_top_cluster else None
        true_left_down = left_down_cluster[0] if left_down_cluster else None
        
        if true_right_top is None or true_left_down is None:
            return None
        
        corners = np.array([
            left_top,
            true_right_top,
            right_down,
            true_left_down
        ], dtype=np.float32)
        
        return corners
    
    def determine_image_tilt(self, right_top_cluster: List[Tuple], left_top: Tuple) -> str:
        """
        判断图像倾斜方向
        """
        # 如果所有右上点的y值都大于左上点的y值，说明图像向右倾斜
        all_y_greater = all(point[1] > left_top[1] for point in right_top_cluster)
        
        if all_y_greater:
            return "lean_to_right"
        else:
            # 检查是否有明显向左倾斜的特征
            # 这里可以添加更复杂的判断逻辑
            return "lean_to_left"
    
    def average_corners(self, corners1: np.ndarray, corners2: np.ndarray) -> np.ndarray:
        """
        平均两种方法得到的角点
        """
        return (corners1 + corners2) / 2
    
    def apply_perspective_transform(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        应用透视变换
        """
        # 重新排列角点顺序：左上、右上、右下、左下
        corners = self.reorder_corners(corners)
        
        # 计算变换后的宽度和高度
        width = max(
            np.linalg.norm(corners[0] - corners[1]),
            np.linalg.norm(corners[2] - corners[3])
        )
        height = max(
            np.linalg.norm(corners[0] - corners[3]),
            np.linalg.norm(corners[1] - corners[2])
        )
        
        # 目标点
        dst_points = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        
        # 应用透视变换
        result = cv2.warpPerspective(image, matrix, (int(width), int(height)))
        
        return result
    
    def reorder_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        重新排列角点顺序
        """
        # 计算中心点
        center = np.mean(corners, axis=0)
        
        # 排序角点
        def angle_from_center(point):
            return math.atan2(point[1] - center[1], point[0] - center[0])
        
        sorted_corners = sorted(corners, key=angle_from_center)
        
        # 重新排列为：左上、右上、右下、左下
        # 找到最左上的点作为起点
        start_index = np.argmin([p[0] + p[1] for p in sorted_corners])
        
        reordered = []
        for i in range(4):
            reordered.append(sorted_corners[(start_index + i) % 4])
        
        return np.array(reordered, dtype=np.float32)

class EnhancedArucoDetector:
    def __init__(self):
        rospy.init_node('enhanced_aruco_detector', anonymous=True)
        
        self.bridge = CvBridge()
        
        # 安全地获取机器人列表参数
        robot_names_param = rospy.get_param('~robot_names', '["robot1", "robot2", "robot3"]')
        
        # 解析机器人名称列表
        try:
            if isinstance(robot_names_param, str):
                self.robot_names = ast.literal_eval(robot_names_param)
            else:
                self.robot_names = robot_names_param
        except:
            rospy.logwarn("无法解析robot_names参数，使用默认值")
            self.robot_names = ["robot1", "robot2", "robot3"]
        
        # AR码检测参数
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters_create()
        
        # AR码实际尺寸 (0.2x0.2米)
        self.marker_length = 0.2
        
        # 存储每个机器人的发布器和相机参数
        self.image_publishers = {}
        self.camera_matrix = {}
        self.dist_coeffs = {}
        self.has_camera_info = {}
        self.undistort_maps = {}  # 新增：存储畸变矫正映射
        
        # 新增：AR码坐标发布器
        self.aruco_pose_publishers = {}
        
        # 透视矫正器
        self.perspective_corrector = PerspectiveCorrector()
        
        # TF广播器
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        # 设置订阅器和发布器
        self.setup_robot_connections()
        
        rospy.loginfo(f"增强版AR码检测器已启动，监控机器人: {self.robot_names}")
        
    def setup_robot_connections(self):
        """为每个机器人设置订阅器和发布器，包括前方和右侧相机"""
        for robot in self.robot_names:
            # 初始化相机参数状态
            self.has_camera_info[robot] = {
                'front': False,
                'right': False
            }
            self.camera_matrix[robot] = {
                'front': None,
                'right': None
            }
            self.dist_coeffs[robot] = {
                'front': None,
                'right': None
            }
            self.undistort_maps[robot] = {
                'front': None,
                'right': None
            }
            
            # 构建合法的话题名称
            # 前方相机
            front_image_topic = f"/{robot}/camera/image"
            front_camera_info_topic = f"/{robot}/camera/camera_info"
            front_result_topic = f"/aruco_detection/{robot}/front_result"
            
            # 右侧相机
            right_image_topic = f"/{robot}/right_camera/image_right"
            right_camera_info_topic = f"/{robot}/right_camera/camera_info_right"
            right_result_topic = f"/aruco_detection/{robot}/right_result"
            
            # 订阅前方相机话题
            rospy.Subscriber(front_image_topic, Image, 
                           lambda msg, robot_name=robot, camera_type='front': self.image_callback(msg, robot_name, camera_type))
            rospy.Subscriber(front_camera_info_topic, CameraInfo,
                           lambda msg, robot_name=robot, camera_type='front': self.camera_info_callback(msg, robot_name, camera_type))
            
            # 订阅右侧相机话题
            rospy.Subscriber(right_image_topic, Image, 
                           lambda msg, robot_name=robot, camera_type='right': self.image_callback(msg, robot_name, camera_type))
            rospy.Subscriber(right_camera_info_topic, CameraInfo,
                           lambda msg, robot_name=robot, camera_type='right': self.camera_info_callback(msg, robot_name, camera_type))
            
            # 创建结果图像发布器
            self.image_publishers[robot] = {
                'front': rospy.Publisher(front_result_topic, Image, queue_size=1),
                'right': rospy.Publisher(right_result_topic, Image, queue_size=1)
            }
            
            # 新增：创建AR码坐标发布器
            self.aruco_pose_publishers[robot] = {
                'front': rospy.Publisher(f"/aruco_detection/{robot}/front_poses", PoseArray, queue_size=1),
                'right': rospy.Publisher(f"/aruco_detection/{robot}/right_poses", PoseArray, queue_size=1)
            }
            
            rospy.loginfo(f"已订阅 {robot} 前方相机: {front_image_topic}")
            rospy.loginfo(f"已订阅 {robot} 右侧相机: {right_image_topic}")
    
    def camera_info_callback(self, msg, robot_name, camera_type):
        """处理相机参数信息并生成畸变矫正映射"""
        if not self.has_camera_info[robot_name][camera_type]:
            # 提取相机内参矩阵
            self.camera_matrix[robot_name][camera_type] = np.array(msg.K).reshape(3, 3)
            
            # 提取畸变系数
            self.dist_coeffs[robot_name][camera_type] = np.array(msg.D)
            
            # 生成畸变矫正映射（实时矫正用）
            image_size = (msg.width, msg.height)
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix[robot_name][camera_type],
                self.dist_coeffs[robot_name][camera_type],
                image_size, 1, image_size
            )
            
            # 计算畸变矫正映射
            map1, map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix[robot_name][camera_type],
                self.dist_coeffs[robot_name][camera_type],
                None, new_camera_matrix, image_size, cv2.CV_16SC2
            )
            
            self.undistort_maps[robot_name][camera_type] = (map1, map2, roi)
            
            self.has_camera_info[robot_name][camera_type] = True
            rospy.loginfo(f"✅ 已获取 {robot_name} {camera_type}相机的相机参数和畸变映射")
    
    def undistort_image(self, cv_image, robot_name, camera_type):
        """对图像进行实时畸变矫正"""
        if (not self.has_camera_info[robot_name][camera_type] or 
            self.undistort_maps[robot_name][camera_type] is None):
            return cv_image
        
        try:
            map1, map2, roi = self.undistort_maps[robot_name][camera_type]
            undistorted = cv2.remap(cv_image, map1, map2, cv2.INTER_LINEAR)
            
            # 裁剪ROI区域
            x, y, w, h = roi
            if w > 0 and h > 0:
                undistorted = undistorted[y:y+h, x:x+w]
            
            return undistorted
        except Exception as e:
            rospy.logwarn(f"畸变矫正失败: {e}")
            return cv_image
    
    def perspective_correction(self, cv_image):
        """对图像进行透视变换矫正"""
        try:
            corrected = self.perspective_corrector.auto_perspective_correction(cv_image)
            return corrected
        except Exception as e:
            rospy.logwarn(f"透视变换矫正失败: {e}")
            return cv_image
    
    def transform_coordinate_system(self, rvec, tvec):
        """
        将坐标从OpenCV坐标系（X右，Y下，Z前）转换到相机坐标系（X前，Z上）
        OpenCV坐标系：X向右，Y向下，Z向前
        目标坐标系：X向前，Z向上，Y向左（右手坐标系）
        """
        # 创建从OpenCV到目标坐标系的旋转矩阵
        # 这个旋转矩阵将X轴从右转向前，Z轴从前转向上
        R_cv_to_cam = np.array([
            [0, 0, 1],   # X_cam = Z_cv (前)
            [-1, 0, 0],  # Y_cam = -X_cv (左)  
            [0, -1, 0]   # Z_cam = -Y_cv (上)
        ], dtype=np.float64)
        
        # 将旋转向量转换为旋转矩阵
        R_obj, _ = cv2.Rodrigues(rvec)
        
        # 应用坐标系变换
        R_obj_cam = R_cv_to_cam @ R_obj
        tvec_cam = R_cv_to_cam @ tvec.reshape(3, 1)
        
        # 将旋转矩阵转换回旋转向量
        rvec_cam, _ = cv2.Rodrigues(R_obj_cam)
        
        return rvec_cam, tvec_cam.flatten()
    
    def detect_aruco_markers_with_pose(self, cv_image, robot_name, camera_type):
        """
        检测AR码并估计位姿，在图像上绘制3D轴和边界框
        返回带标记的图像、检测到的ID列表和位姿信息
        """
        if not self.has_camera_info[robot_name][camera_type]:
            # 如果没有相机参数，使用基本检测
            return self.basic_detection(cv_image, robot_name, camera_type)
        
        # 转换为灰度图像
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 检测AR码
        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        detected_ids = []
        poses = []
        
        if ids is not None:
            # 估计AR码位姿
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, 
                self.camera_matrix[robot_name][camera_type], 
                self.dist_coeffs[robot_name][camera_type]
            )
            
            # 将ID转换为列表
            detected_ids = ids.flatten().tolist()
            
            # 绘制检测结果
            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                rvec = rvecs[i]
                tvec = tvecs[i]
                
                # 坐标系统转换 - 将OpenCV坐标转换为相机坐标
                rvec_cam, tvec_cam = self.transform_coordinate_system(rvec, tvec)
                
                # 存储位姿信息（使用转换后的坐标）
                poses.append({
                    'id': marker_id,
                    'rvec': rvec_cam,
                    'tvec': tvec_cam,
                    'camera_type': camera_type
                })
                
                # 根据相机类型选择颜色
                if camera_type == 'front':
                    color = (0, 255, 0)  # 绿色 - 前方相机
                else:
                    color = (255, 0, 0)  # 蓝色 - 右侧相机
                
                # 绘制AR码边界框
                cv2.polylines(cv_image, [corner.astype(int)], True, color, 2)
                
                # 绘制3D坐标轴（使用原始坐标进行绘制，因为OpenCV的绘制函数期望OpenCV坐标系）
                axis_length = self.marker_length * 0.5
                cv2.drawFrameAxes(cv_image, self.camera_matrix[robot_name][camera_type], 
                                self.dist_coeffs[robot_name][camera_type], rvec, tvec, axis_length)
                
                # 计算中心点用于显示ID
                center = corner[0].mean(axis=0).astype(int)
                
                # 绘制ID标签和距离信息（使用转换后的距离）
                distance = np.linalg.norm(tvec_cam)
                text = f"ID:{marker_id} {camera_type} Dist:{distance:.2f}m"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # 标签背景
                cv2.rectangle(cv_image, 
                            (center[0] - text_size[0]//2 - 5, center[1] - text_size[1] - 5),
                            (center[0] + text_size[0]//2 + 5, center[1] + 5),
                            color, -1)
                
                # ID文本
                cv2.putText(cv_image, text, 
                          (center[0] - text_size[0]//2, center[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # 发布TF变换（使用转换后的坐标）
                self.publish_tf_transform(rvec_cam, tvec_cam, marker_id, robot_name, camera_type)
        
        return cv_image, detected_ids, poses
    
    def basic_detection(self, cv_image, robot_name, camera_type):
        """基础检测（当没有相机参数时使用）"""
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        detected_ids = []
        
        if ids is not None:
            detected_ids = ids.flatten().tolist()
            
            # 根据相机类型选择颜色
            if camera_type == 'front':
                color = (0, 255, 0)  # 绿色
            else:
                color = (255, 0, 0)  # 蓝色
            
            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                corner = corner.astype(int)
                
                # 绘制边界框
                cv2.polylines(cv_image, [corner], True, color, 3)
                
                center = corner[0].mean(axis=0).astype(int)
                text = f"ID:{marker_id} {camera_type}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                cv2.rectangle(cv_image, 
                            (center[0] - text_size[0]//2 - 5, center[1] - text_size[1] - 5),
                            (center[0] + text_size[0]//2 + 5, center[1] + 5),
                            color, -1)
                
                cv2.putText(cv_image, text, 
                          (center[0] - text_size[0]//2, center[1]), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return cv_image, detected_ids, []
    
    def publish_tf_transform(self, rvec, tvec, marker_id, robot_name, camera_type):
        """发布AR码的TF变换，区分不同相机"""
        try:
            # 将旋转向量转换为旋转矩阵
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # 构建齐次变换矩阵
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = tvec.flatten()
            
            # 转换到TF格式
            transform = tf.transformations.rotation_matrix(0, (0, 0, 0))  # 创建单位矩阵
            transform[:3, :3] = rotation_matrix
            transform[:3, 3] = tvec.flatten()
            
            # 提取四元数
            quaternion = tf.transformations.quaternion_from_matrix(transform)
            
            # 根据相机类型确定父坐标系
            if camera_type == 'front':
                parent_frame = f"{robot_name}/camera_link"
            else:
                parent_frame = f"{robot_name}/right_camera_link"
            
            # 发布TF，添加相机类型前缀
            self.tf_broadcaster.sendTransform(
                tvec.flatten().tolist(),
                quaternion.tolist(),
                rospy.Time.now(),
                f"{camera_type}_aruco_{marker_id}",
                parent_frame
            )
            
        except Exception as e:
            rospy.logwarn(f"发布TF变换时出错: {e}")
    
    def publish_aruco_poses(self, poses, robot_name, camera_type):
        """发布AR码的坐标信息"""
        if not poses:
            return
            
        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp = rospy.Time.now()
        
        # 设置坐标系
        if camera_type == 'front':
            pose_array.header.frame_id = f"{robot_name}/camera_link"
        else:
            pose_array.header.frame_id = f"{robot_name}/right_camera_link"
        
        for pose_info in poses:
            pose = Pose()
            
            # 设置位置
            tvec = pose_info['tvec'].flatten()
            pose.position = Point(x=tvec[0], y=tvec[1], z=tvec[2])
            
            # 设置方向（将旋转向量转换为四元数）
            rvec = pose_info['rvec']
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # 将旋转矩阵转换为四元数
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            quaternion = tf.transformations.quaternion_from_matrix(transform_matrix)
            
            pose.orientation = Quaternion(
                x=quaternion[0],
                y=quaternion[1],
                z=quaternion[2],
                w=quaternion[3]
            )
            
            pose_array.poses.append(pose)
        
        # 发布AR码坐标信息
        self.aruco_pose_publishers[robot_name][camera_type].publish(pose_array)
        
        # 打印坐标信息（可选，用于调试）
        if poses:
            for pose_info in poses:
                tvec = pose_info['tvec'].flatten()
                rospy.loginfo(f"📊 {robot_name.upper()} {camera_type.upper()}相机 AR码 {pose_info['id']} 坐标: "
                            f"X={tvec[0]:.3f}m, Y={tvec[1]:.3f}m, Z={tvec[2]:.3f}m")
    
    def image_callback(self, msg, robot_name, camera_type):
        """处理图像回调，先进行畸变矫正再识别AR码"""
        try:
            # 转换ROS图像到OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 实时畸变矫正 - 在识别前先矫正图像
            undistorted_image = self.undistort_image(cv_image, robot_name, camera_type)
            
            # 透视变换矫正 - 进一步矫正图像倾斜
            perspective_corrected_image = self.perspective_correction(undistorted_image)
            
            # 检测AR码（带位姿估计）
            result_image, detected_ids, poses = self.detect_aruco_markers_with_pose(perspective_corrected_image, robot_name, camera_type)
            
            # 新增：发布AR码坐标信息
            self.publish_aruco_poses(poses, robot_name, camera_type)
            
            # 在图像左上角添加相机类型标识
            cv2.putText(result_image, f"{camera_type.upper()} CAMERA - {robot_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (255, 255, 255) if camera_type == 'front' else (255, 200, 100), 2)
            
            # 打印检测结果
            if detected_ids:
                pose_info = ""
                for pose in poses:
                    tvec = pose['tvec'].flatten()
                    pose_info += f" ID{pose['id']}: pos({tvec[0]:.2f}, {tvec[1]:.2f}, {tvec[2]:.2f})"
                
                rospy.loginfo(f"🤖 {robot_name.upper()} {camera_type.upper()}相机 检测到AR码: IDs={detected_ids}{pose_info}")
            else:
                # 减少未检测到时的日志输出频率
                current_time = rospy.get_time()
                time_key = f'last_no_detection_time_{robot_name}_{camera_type}'
                
                if not hasattr(self, time_key):
                    setattr(self, time_key, 0)
                
                if current_time - getattr(self, time_key) > 5.0:
                    if self.has_camera_info[robot_name][camera_type]:
                        rospy.loginfo(f"🔍 {robot_name.upper()} {camera_type.upper()}相机 视野内未检测到AR码")
                    else:
                        rospy.logwarn(f"⚠️ {robot_name.upper()} {camera_type.upper()}相机 等待相机参数...")
                    setattr(self, time_key, current_time)
            
            # 发布结果图像
            try:
                result_msg = self.bridge.cv2_to_imgmsg(result_image, "bgr8")
                self.image_publishers[robot_name][camera_type].publish(result_msg)
            except CvBridgeError as e:
                rospy.logerr(f"发布图像时出错 ({robot_name} {camera_type}): {e}")
                
        except CvBridgeError as e:
            rospy.logerr(f"图像转换错误 ({robot_name} {camera_type}): {e}")
        except Exception as e:
            rospy.logerr(f"处理图像时发生未知错误 ({robot_name} {camera_type}): {e}")
    
    def run(self):
        """运行节点"""
        rospy.loginfo("增强版AR码检测器运行中...按Ctrl+C退出")
        rospy.loginfo("支持实时畸变矫正、透视变换矫正和双相机AR码识别")
        rospy.loginfo("AR码坐标话题格式: /aruco_detection/<robot_name>/<camera_type>_poses")
        rospy.loginfo("坐标系已转换为: X轴朝前，Z轴朝上")
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = EnhancedArucoDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("增强版AR码检测器已关闭")