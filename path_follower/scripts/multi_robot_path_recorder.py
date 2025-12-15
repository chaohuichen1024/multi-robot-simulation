#!/usr/bin/env python3
'''
Descripttion: 多机器人里程计记录节点
Author: chaohui_chen
Date: 2024-06-08 17:22:35
LastEditors: chaohui_chen1024
LastEditTime: 2025-11-19 11:21:27
'''

import rospy
import yaml
import os
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
import math
import threading
from datetime import datetime

class MultiRobotPathRecorder:
    def __init__(self):
        # 获取参数
        self.robot_count = rospy.get_param('~robot_count', 3)
        self.record_interval = rospy.get_param('~record_interval', 2.0)  # 记录间隔（秒）
        self.movement_threshold = rospy.get_param('~movement_threshold', 0.05)  # 运动阈值（米）
        self.angular_threshold = rospy.get_param('~angular_threshold', 0.1)  # 角速度阈值（弧度）
        self.output_directory = rospy.get_param('~output_directory', '/home/cch/sensor_ws/src/path_follower/')  # 输出目录
        
        # 展开家目录
        self.output_directory = os.path.expanduser(self.output_directory)
        
        # 确保输出目录存在
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            rospy.loginfo(f"Created output directory: {self.output_directory}")
        
        # 存储每个机器人的数据
        self.robot_data = {}
        self.robot_poses = {}
        self.last_record_time = {}
        self.last_pose = {}
        self.lock = threading.Lock()
        
        # 初始化每个机器人的数据结构
        for i in range(1, self.robot_count + 1):
            robot_name = f"robot{i}"
            self.robot_data[robot_name] = []
            self.robot_poses[robot_name] = None
            self.last_record_time[robot_name] = rospy.Time.now()
            self.last_pose[robot_name] = None
            
            # 订阅每个机器人的里程计话题
            topic_name = f'/{robot_name}/odom'
            rospy.Subscriber(topic_name, Odometry, 
                           self.odom_callback, 
                           callback_args=robot_name,
                           queue_size=10)
        
        # 定时保存线程
        self.save_timer = rospy.Timer(rospy.Duration(5.0), self.auto_save_callback)
        
        # 手动保存服务（可选）
        rospy.on_shutdown(self.save_all_paths)
        
        rospy.loginfo(f"Multi-robot path recorder initialized for {self.robot_count} robots")
        rospy.loginfo(f"Record interval: {self.record_interval}s, Movement threshold: {self.movement_threshold}m")
        rospy.loginfo(f"Output directory: {self.output_directory}")

    def odom_callback(self, msg, robot_name):
        """里程计数据回调函数"""
        current_time = rospy.Time.now()
        
        with self.lock:
            # 更新当前位姿
            current_pose = {
                'position': msg.pose.pose.position,
                'orientation': msg.pose.pose.orientation
            }
            
            self.robot_poses[robot_name] = current_pose
            
            # 检查是否需要记录
            should_record = self.should_record(robot_name, current_pose, current_time)
            
            if should_record:
                self.record_waypoint(robot_name, current_pose)
                self.last_record_time[robot_name] = current_time
                self.last_pose[robot_name] = current_pose.copy()
                
                rospy.logdebug(f"Recorded waypoint for {robot_name}: "
                             f"({current_pose['position'].x:.2f}, "
                             f"{current_pose['position'].y:.2f}, "
                             f"{current_pose['position'].z:.2f})")

    def should_record(self, robot_name, current_pose, current_time):
        """判断是否需要记录当前位姿"""
        # 如果是第一个点，总是记录
        if self.last_pose[robot_name] is None:
            return True
        
        # 检查时间间隔
        time_diff = (current_time - self.last_record_time[robot_name]).to_sec()
        if time_diff < self.record_interval:
            return False
        
        # 检查是否在运动（位置变化）
        last_pos = self.last_pose[robot_name]['position']
        current_pos = current_pose['position']
        
        distance = math.sqrt(
            (current_pos.x - last_pos.x) ** 2 +
            (current_pos.y - last_pos.y) ** 2 +
            (current_pos.z - last_pos.z) ** 2
        )
        
        # 检查方向变化（使用四元数的简单方法）
        last_orient = self.last_pose[robot_name]['orientation']
        current_orient = current_pose['orientation']
        
        # 计算四元数点积来估计角度变化
        dot_product = (last_orient.x * current_orient.x +
                      last_orient.y * current_orient.y +
                      last_orient.z * current_orient.z +
                      last_orient.w * current_orient.w)
        
        # 确保点积在有效范围内
        dot_product = max(min(dot_product, 1.0), -1.0)
        angle_change = 2 * math.acos(abs(dot_product))
        
        # 如果位置或方向变化超过阈值，则记录
        return (distance > self.movement_threshold or 
                angle_change > self.angular_threshold)

    def record_waypoint(self, robot_name, pose):
        """记录路径点"""
        waypoint = {
            'x': pose['position'].x,
            'y': pose['position'].y,
            'z': pose['position'].z,
            'ox': pose['orientation'].x,
            'oy': pose['orientation'].y,
            'oz': pose['orientation'].z,
            'ow': pose['orientation'].w
        }
        
        self.robot_data[robot_name].append(waypoint)
        rospy.loginfo(f"Recorded waypoint {len(self.robot_data[robot_name])} for {robot_name}")

    def save_robot_path(self, robot_name):
        """保存单个机器人的路径到YAML文件"""
        if not self.robot_data[robot_name]:
            rospy.logwarn(f"No data to save for {robot_name}")
            return False
        
        filename = os.path.join(self.output_directory, f"{robot_name}_path.yaml")
        
        data_to_save = {
            'waypoints': self.robot_data[robot_name]
        }
        
        try:
            with open(filename, 'w') as file:
                yaml.dump(data_to_save, file, default_flow_style=False)
            
            rospy.loginfo(f"Saved {len(self.robot_data[robot_name])} waypoints for {robot_name} to {filename}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Failed to save path for {robot_name}: {e}")
            return False

    def save_all_paths(self):
        """保存所有机器人的路径"""
        rospy.loginfo("Saving all robot paths...")
        
        success_count = 0
        for robot_name in self.robot_data:
            if self.save_robot_path(robot_name):
                success_count += 1
        
        rospy.loginfo(f"Successfully saved paths for {success_count}/{self.robot_count} robots")

    def auto_save_callback(self, event):
        """定时自动保存回调"""
        rospy.logdebug("Auto-saving robot paths...")
        # 在实际使用中，可以定期保存，但这里我们只在关闭时保存
        pass

    def print_status(self):
        """打印当前状态"""
        with self.lock:
            for robot_name in self.robot_data:
                waypoint_count = len(self.robot_data[robot_name])
                rospy.loginfo(f"{robot_name}: {waypoint_count} waypoints recorded")

    def run(self):
        """主循环"""
        rospy.loginfo("Multi-robot path recorder is running...")
        rospy.loginfo("Press Ctrl+C to stop and save all paths")
        
        # 定期打印状态
        status_timer = rospy.Timer(rospy.Duration(10.0), 
                                 lambda event: self.print_status())
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Received Ctrl+C, shutting down...")
        finally:
            self.save_all_paths()

if __name__ == '__main__':
    rospy.init_node('multi_robot_path_recorder')
    recorder = MultiRobotPathRecorder()
    recorder.run()