#!/usr/bin/env python3
import rospy
import yaml
import math
import os
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from tf.transformations import euler_from_quaternion
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse
import threading

class RobotController:
    def __init__(self, robot_name, params):
        self.robot_name = robot_name
        self.params = params
        
        # 初始化变量
        self.current_pose = None
        self.path = None
        self.current_waypoint = 0
        self.is_active = False
        self.finished = False
        self.lock = threading.Lock()
        
        # 创建发布者
        self.cmd_vel_pub = rospy.Publisher(f'/{robot_name}/cmd_vel', Twist, queue_size=10)
        self.path_pub = rospy.Publisher(f'/{robot_name}/nav_path', Path, queue_size=10)
        
        # 创建订阅者
        self.odom_sub = rospy.Subscriber(f'/{robot_name}/odom', Odometry, self.odom_callback)
        
        # 创建服务
        self.start_service = rospy.Service(f'/{robot_name}/start_path_following', Trigger, self.start_path_following_callback)
        self.stop_service = rospy.Service(f'/{robot_name}/stop_path_following', Trigger, self.stop_path_following_callback)
        
        # 读取路径
        self.load_path()
        
        # 立即发布路径，使RViz能够显示
        self.publish_path()
        
        rospy.loginfo(f"Robot controller initialized for {robot_name}")
    
    def resolve_path(self, path_string):
        """解析ROS路径中的$(find package)语法"""
        if path_string.startswith('$(find '):
            # 提取包名和相对路径
            parts = path_string.split(')')
            if len(parts) > 1:
                find_part = parts[0]  # $(find package_name/path
                relative_path = ')'.join(parts[1:])  # 剩余部分
                
                # 提取包名
                package_part = find_part.replace('$(find ', '')
                package_name = package_part.split('/')[0]
                
                try:
                    # 获取包路径
                    import rospkg
                    rospack = rospkg.RosPack()
                    package_path = rospack.get_path(package_name)
                    
                    # 构建完整路径
                    if '/' in package_part:
                        sub_path = package_part.split('/', 1)[1]
                        full_path = os.path.join(package_path, sub_path, relative_path.lstrip('/'))
                    else:
                        full_path = os.path.join(package_path, relative_path.lstrip('/'))
                    
                    return full_path
                except (rospkg.ResourceNotFound, ImportError) as e:
                    rospy.logwarn(f"Failed to resolve ROS path {path_string}: {str(e)}")
                    return path_string
        return path_string
    
    def load_path(self):
        """从YAML文件加载路径"""
        try:
            path_file = self.params.get('path_file', 'path.yaml')
            
            # 解析ROS路径语法
            resolved_path = self.resolve_path(path_file)
            rospy.loginfo(f"Loading path from: {resolved_path}")
            
            with open(resolved_path, 'r') as f:
                path_data = yaml.safe_load(f)
            
            # 创建Path消息
            self.path = Path()
            self.path.header.frame_id = 'world'
            self.path.header.stamp = rospy.Time.now()
            
            # 解析YAML数据
            if 'waypoints' in path_data:
                for waypoint in path_data['waypoints']:
                    pose = PoseStamped()
                    pose.header.frame_id = 'world'
                    pose.pose.position.x = waypoint.get('x', 0.0)
                    pose.pose.position.y = waypoint.get('y', 0.0)
                    pose.pose.position.z = waypoint.get('z', 0.0)
                    pose.pose.orientation.x = waypoint.get('ox', 0.0)
                    pose.pose.orientation.y = waypoint.get('oy', 0.0)
                    pose.pose.orientation.z = waypoint.get('oz', 0.0)
                    pose.pose.orientation.w = waypoint.get('ow', 1.0)
                    self.path.poses.append(pose)
            
            rospy.loginfo(f"Loaded path with {len(self.path.poses)} waypoints for {self.robot_name}")
            
        except Exception as e:
            rospy.logerr(f"Failed to load path for {self.robot_name}: {str(e)}")
    
    def publish_path(self):
        """发布路径供RViz显示"""
        if self.path:
            self.path.header.stamp = rospy.Time.now()
            self.path_pub.publish(self.path)
    
    def odom_callback(self, msg):
        """处理里程计消息，获取当前位置"""
        with self.lock:
            self.current_pose = msg.pose.pose
    
    def calculate_distance(self, x1, y1, x2, y2):
        """计算两点之间的距离"""
        return math.hypot(x2 - x1, y2 - y1)
    
    def start_path_following(self):
        """开始路径跟踪"""
        with self.lock:
            if self.is_active:
                return False, "Path following is already active"
            
            if not self.path or len(self.path.poses) == 0:
                return False, "No path loaded"
            
            self.is_active = True
            self.finished = False
            self.current_waypoint = 0
            rospy.loginfo(f"Path following started for {self.robot_name}")
            
            return True, f"Path following started successfully for {self.robot_name}"
    
    def start_path_following_callback(self, req):
        """服务回调函数，开始路径跟踪"""
        success, message = self.start_path_following()
        return TriggerResponse(success, message)
    
    def stop_path_following(self):
        """停止路径跟踪"""
        with self.lock:
            if not self.is_active:
                return False, "Path following is not active"
            
            self.is_active = False
            # 发布零速度
            twist = Twist()
            self.cmd_vel_pub.publish(twist)
            rospy.loginfo(f"Path following stopped for {self.robot_name}")
            
            return True, f"Path following stopped successfully for {self.robot_name}"
    
    def stop_path_following_callback(self, req):
        """服务回调函数，停止路径跟踪"""
        success, message = self.stop_path_following()
        return TriggerResponse(success, message)
    
    def update_path_display(self):
        """更新路径显示，无论是否激活都会执行"""
        # 发布路径（持续发布以便RViz能接收到）
        self.publish_path()
    
    def control_loop(self):
        """控制循环"""
        with self.lock:
            # 检查是否激活且未完成
            if not self.is_active or self.finished:
                return
            
            # 等待获取当前位置
            if self.current_pose is None:
                return
            
            # 检查是否所有路径点都已到达
            if self.current_waypoint >= len(self.path.poses):
                rospy.loginfo(f"All waypoints reached for {self.robot_name}. Stopping.")
                self.cmd_vel_pub.publish(Twist())  # 发布零速度
                self.is_active = False
                self.finished = True
                return
            
            # 获取当前目标点
            target_pose = self.path.poses[self.current_waypoint].pose
            
            # 计算当前位置与目标点的距离
            current_x = self.current_pose.position.x
            current_y = self.current_pose.position.y
            target_x = target_pose.position.x
            target_y = target_pose.position.y
            
            distance = self.calculate_distance(current_x, current_y, target_x, target_y)
            
            # 如果到达当前目标点，切换到下一个
            if distance < self.params.get('target_tolerance', 0.1):
                rospy.loginfo(f"{self.robot_name}: Reached waypoint {self.current_waypoint + 1}/{len(self.path.poses)}")
                self.current_waypoint += 1
                return
            
            # 计算目标方向
            target_yaw = math.atan2(target_y - current_y, target_x - current_x)
            
            # 获取当前机器人朝向
            current_quat = self.current_pose.orientation
            _, _, current_yaw = euler_from_quaternion(
                [current_quat.x, current_quat.y, current_quat.z, current_quat.w]
            )
            
            # 计算角度差（归一化到[-pi, pi]）
            angle_error = target_yaw - current_yaw
            angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi
            
            # 计算速度指令
            twist = Twist()
            
            # 只有当角度误差足够小时才前进
            angular_tolerance = self.params.get('angular_tolerance', 0.1)
            if abs(angle_error) < angular_tolerance:
                # 线速度与距离成正比，但不超过最大值
                linear_gain = self.params.get('linear_gain', 1.0)
                max_linear_speed = self.params.get('max_linear_speed', 0.5)
                twist.linear.x = min(linear_gain * distance, max_linear_speed)
            else:
                twist.linear.x = 0.0  # 角度误差大时不前进，只转向
            
            # 角速度与角度误差成正比，但不超过最大值
            angular_gain = self.params.get('angular_gain', 2.0)
            max_angular_speed = self.params.get('max_angular_speed', 1.0)
            twist.angular.z = min(max(angular_gain * angle_error, -max_angular_speed), max_angular_speed)
            
            # 发布速度指令
            self.cmd_vel_pub.publish(twist)

class MultiRobotPathFollower:
    def __init__(self):
        # 初始化节点
        rospy.init_node('multi_robot_path_follower', anonymous=True)
        
        # 读取参数
        self.robot_configs = rospy.get_param('~robots', {})
        
        # 初始化机器人控制器字典
        self.robot_controllers = {}
        
        # 创建总服务
        self.start_all_service = rospy.Service('/start_all_robots', Trigger, self.start_all_robots_callback)
        self.stop_all_service = rospy.Service('/stop_all_robots', Trigger, self.stop_all_robots_callback)
        
        # 初始化所有配置的机器人
        self.initialize_robots()
        
        rospy.loginfo("Multi-robot path follower initialized")
        
        # 主循环
        self.rate = rospy.Rate(10)  # 10Hz
        self.main_loop()
    
    def initialize_robots(self):
        """初始化所有配置的机器人"""
        for robot_name, params in self.robot_configs.items():
            self.add_robot_controller(robot_name, params)
    
    def add_robot_controller(self, robot_name, params):
        """添加单个机器人控制器"""
        if robot_name in self.robot_controllers:
            rospy.logwarn(f"Robot {robot_name} already exists")
            return False
        
        try:
            controller = RobotController(robot_name, params)
            self.robot_controllers[robot_name] = controller
            rospy.loginfo(f"Successfully added robot controller for {robot_name}")
            return True
        except Exception as e:
            rospy.logerr(f"Failed to add robot controller for {robot_name}: {str(e)}")
            return False
    
    def start_all_robots_callback(self, req):
        """启动所有机器人的服务回调"""
        results = []
        success_count = 0
        
        for robot_name, controller in self.robot_controllers.items():
            success, message = controller.start_path_following()
            results.append(f"{robot_name}: {message}")
            if success:
                success_count += 1
        
        all_success = success_count == len(self.robot_controllers)
        message = f"Started {success_count}/{len(self.robot_controllers)} robots. Details: " + "; ".join(results)
        
        return TriggerResponse(all_success, message)
    
    def stop_all_robots_callback(self, req):
        """停止所有机器人的服务回调"""
        results = []
        success_count = 0
        
        for robot_name, controller in self.robot_controllers.items():
            success, message = controller.stop_path_following()
            results.append(f"{robot_name}: {message}")
            if success:
                success_count += 1
        
        all_success = success_count == len(self.robot_controllers)
        message = f"Stopped {success_count}/{len(self.robot_controllers)} robots. Details: " + "; ".join(results)
        
        return TriggerResponse(all_success, message)
    
    def main_loop(self):
        """主循环"""
        while not rospy.is_shutdown():
            # 为每个机器人执行路径显示和控制循环
            for robot_name, controller in self.robot_controllers.items():
                try:
                    # 无论是否激活，都更新路径显示
                    controller.update_path_display()
                    # 只有在激活状态时才执行控制逻辑
                    controller.control_loop()
                except Exception as e:
                    rospy.logerr(f"Error in control loop for {robot_name}: {str(e)}")
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        MultiRobotPathFollower()
    except rospy.ROSInterruptException:
        rospy.loginfo("Multi-robot path follower interrupted.")