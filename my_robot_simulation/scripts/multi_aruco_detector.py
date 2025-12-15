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
from geometry_msgs.msg import Point, Pose, PoseArray
from std_msgs.msg import Header, Int32MultiArray

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
        self.pose_publishers = {}  # 新增：AR码位姿发布器
        
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
            front_pose_topic = f"/aruco_detection/{robot}/front_poses"  # 新增：前方相机AR码位姿话题
            
            # 右侧相机
            right_image_topic = f"/{robot}/right_camera/image_right"
            right_camera_info_topic = f"/{robot}/right_camera/camera_info_right"
            right_result_topic = f"/aruco_detection/{robot}/right_result"
            right_pose_topic = f"/aruco_detection/{robot}/right_poses"  # 新增：右侧相机AR码位姿话题
            
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
            
            # 新增：创建AR码位姿发布器
            self.pose_publishers[robot] = {
                'front': rospy.Publisher(front_pose_topic, PoseArray, queue_size=1),
                'right': rospy.Publisher(right_pose_topic, PoseArray, queue_size=1)
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
    
    def transform_pose_to_camera_frame(self, rvec, tvec):
        """
        将OpenCV坐标系下的位姿转换到相机坐标系
        OpenCV: Z向前, Y向下, X向右
        相机坐标系: X向前, Z向上, Y向左
        """
        # 创建从OpenCV到相机坐标系的旋转矩阵
        # 这个矩阵将:
        #   Z(前) -> X(前)
        #   X(右) -> -Y(左) 
        #   Y(下) -> Z(上)
        R_cv_to_cam = np.array([
            [0, 0, 1],   # Z -> X
            [-1, 0, 0],  # X -> -Y
            [0, -1, 0]   # Y -> -Z (注意: 这里应该是Y -> Z, 但需要验证)
        ])
        
        # 将旋转向量转换为旋转矩阵
        R_cv, _ = cv2.Rodrigues(rvec)
        
        # 应用坐标系变换
        R_cam = R_cv_to_cam @ R_cv
        
        # 转换平移向量
        t_cam = R_cv_to_cam @ tvec.flatten()
        
        # 将旋转矩阵转换回旋转向量
        rvec_cam, _ = cv2.Rodrigues(R_cam)
        
        return rvec_cam, t_cam
    
    def publish_aruco_poses(self, poses, robot_name, camera_type):
        """发布检测到的AR码位姿信息"""
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
            
            # 设置位置 (tvec)
            tvec = pose_info['tvec'].flatten()
            pose.position.x = tvec[0]
            pose.position.y = tvec[1]
            pose.position.z = tvec[2]
            
            # 设置方向 (从旋转向量转换为四元数)
            rvec = pose_info['rvec']
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix
            quaternion = tf.transformations.quaternion_from_matrix(transform)
            
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]
            
            pose_array.poses.append(pose)
        
        # 发布位姿信息
        self.pose_publishers[robot_name][camera_type].publish(pose_array)
    
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
                
                # 根据相机类型选择颜色
                if camera_type == 'front':
                    color = (0, 255, 0)  # 绿色 - 前方相机
                else:
                    color = (255, 0, 0)  # 蓝色 - 右侧相机
                
                # 绘制AR码边界框
                cv2.polylines(cv_image, [corner.astype(int)], True, color, 2)
                
                # 绘制3D坐标轴（红色：X，绿色：Y，蓝色：Z）
                axis_length = self.marker_length * 0.5
                cv2.drawFrameAxes(cv_image, self.camera_matrix[robot_name][camera_type], 
                                self.dist_coeffs[robot_name][camera_type], rvec, tvec, axis_length)
                
                # 计算中心点用于显示ID
                center = corner[0].mean(axis=0).astype(int)
                
                # 绘制ID标签和距离信息
                distance = np.linalg.norm(tvec)
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
                
                # 转换坐标系：从OpenCV坐标系到相机坐标系
                rvec_cam, tvec_cam = self.transform_pose_to_camera_frame(rvec, tvec)
                
                # 存储转换后的位姿信息
                poses.append({
                    'id': marker_id,
                    'rvec': rvec_cam,
                    'tvec': tvec_cam,
                    'camera_type': camera_type
                })
                
                # 发布TF变换（使用转换后的位姿）
                self.publish_tf_transform(rvec_cam, tvec_cam, marker_id, robot_name, camera_type)
            
            # 新增：发布AR码位姿信息
            self.publish_aruco_poses(poses, robot_name, camera_type)
        
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
    
    def image_callback(self, msg, robot_name, camera_type):
        """处理图像回调，先进行畸变矫正再识别AR码"""
        try:
            # 转换ROS图像到OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 实时畸变矫正 - 在识别前先矫正图像
            undistorted_image = self.undistort_image(cv_image, robot_name, camera_type)
            
            # 检测AR码（带位姿估计）
            result_image, detected_ids, poses = self.detect_aruco_markers_with_pose(undistorted_image, robot_name, camera_type)
            
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
        rospy.loginfo("支持实时畸变矫正和双相机AR码识别")
        rospy.loginfo("AR码坐标话题格式: /aruco_detection/{robot_name}/{front/right}_poses")
        rospy.loginfo("已应用坐标系转换: OpenCV(Z前,Y下,X右) -> 相机(X前,Z上,Y左)")
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = EnhancedArucoDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("增强版AR码检测器已关闭")