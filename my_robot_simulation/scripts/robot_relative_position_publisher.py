#!/usr/bin/env python3
'''
Descripttion: 灵活的机器人相对位置发布器
Author: chaohui_chen1024
Date: 2025-10-18 16:13:21
LastEditors: chaohui_chen1024
LastEditTime: 2025-10-18 21:23:15
'''

import rospy
import tf2_ros
import geometry_msgs.msg
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header

class RobotRelativePositionPublisher:
    def __init__(self):
        rospy.init_node('robot_relative_position_publisher')
        
        # 从参数服务器获取机器人配置
        self.robot_names = rospy.get_param('~robot_names', ['robot1', 'robot2', 'robot3'])
        self.reference_robot = rospy.get_param('~reference_robot', 'robot1')
        
        # 创建TF缓冲区和监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 为每个机器人创建独立的发布器
        self.publishers = {}
        for robot in self.robot_names:
            if robot != self.reference_robot:
                topic_name = f'/robot_relative_positions/{self.reference_robot}_to_{robot}'
                self.publishers[robot] = rospy.Publisher(topic_name, PointStamped, queue_size=10)
        
        # 同时创建一个综合发布器，发布所有相对位置
        self.combined_pub = rospy.Publisher('/robot_relative_positions/all', PointStamped, queue_size=10)
        
        rospy.loginfo(f"初始化完成，参考机器人: {self.reference_robot}")
        rospy.loginfo(f"监控的机器人: {self.robot_names}")
        
    def publish_relative_positions(self):
        current_time = rospy.Time.now()
        all_positions_available = True
        
        for robot in self.robot_names:
            if robot != self.reference_robot:
                try:
                    # 获取变换
                    transform = self.tf_buffer.lookup_transform(
                        f"{self.reference_robot}/base_link",
                        f"{robot}/base_link",
                        current_time,
                        rospy.Duration(1.0)  # 缩短超时时间
                    )
                    
                    # 创建相对位置消息
                    rel_pos = PointStamped()
                    rel_pos.header.stamp = current_time
                    rel_pos.header.frame_id = f"{self.reference_robot}/base_link"
                    rel_pos.point.x = transform.transform.translation.x
                    rel_pos.point.y = transform.transform.translation.y
                    rel_pos.point.z = transform.transform.translation.z
                    
                    # 发布到独立话题
                    self.publishers[robot].publish(rel_pos)
                    
                    # 也发布到综合话题（带机器人标识）
                    combined_pos = PointStamped()
                    combined_pos.header.stamp = current_time
                    combined_pos.header.frame_id = robot  # 使用机器人名称作为frame_id来标识
                    combined_pos.point.x = transform.transform.translation.x
                    combined_pos.point.y = transform.transform.translation.y
                    combined_pos.point.z = transform.transform.translation.z
                    self.combined_pub.publish(combined_pos)
                    
                    # 实时输出所有机器人的位置信息
                    rospy.loginfo(
                        f"{robot} 相对于 {self.reference_robot} 的位置: "
                        f"({transform.transform.translation.x:.2f}, "
                        f"{transform.transform.translation.y:.2f}, "
                        f"{transform.transform.translation.z:.2f})"
                    )
                    
                except (tf2_ros.LookupException, 
                        tf2_ros.ConnectivityException, 
                        tf2_ros.ExtrapolationException) as e:
                    rospy.logwarn(f"无法获取 {robot} 相对于 {self.reference_robot} 的变换: {e}")
                    all_positions_available = False
        
        return all_positions_available
    
    def run(self):
        rate = rospy.Rate(10)  # 10Hz
        rospy.loginfo("开始发布机器人相对位置信息...")
        
        while not rospy.is_shutdown():
            try:
                self.publish_relative_positions()
                rate.sleep()
                
            except rospy.ROSInterruptException:
                break

def main():
    try:
        publisher = RobotRelativePositionPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()