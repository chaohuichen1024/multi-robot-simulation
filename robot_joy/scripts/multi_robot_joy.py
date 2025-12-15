#!/usr/bin/env python3
'''
Descripttion: 多机器人手柄控制节点
Author: chaohui_chen
Date: 2024-06-08 17:22:35
LastEditors: chaohui_chen1024
LastEditTime: 2025-10-21 11:52:52
'''
  
import rospy  
from sensor_msgs.msg import Joy  
from geometry_msgs.msg import Twist  

class MultiRobotJoy:  
    def __init__(self):  
        # 获取参数
        self.linear_axis = rospy.get_param('~axis_linear', 1)  
        self.angular_axis = rospy.get_param('~axis_angular', 3)  
        self.scale_linear = rospy.get_param('~scale_linear', 0.5)  
        self.scale_angular = rospy.get_param('~scale_angular', 1.0)  
        
        # 机器人数量和控制切换按钮
        self.robot_count = rospy.get_param('~robot_count', 3)
        self.prev_button = rospy.get_param('~prev_button', 4)  # LB按钮 - 上一个机器人
        self.next_button = rospy.get_param('~next_button', 5)  # RB按钮 - 下一个机器人
        
        # 当前控制的机器人索引
        self.current_robot = 0
        self.last_switch_time = rospy.Time.now()
        self.switch_cooldown = rospy.Duration(0.3)  # 切换冷却时间
        
        # 为每个机器人创建速度发布器
        self.vel_publishers = []
        for i in range(self.robot_count):
            topic_name = f'robot{i+1}/cmd_vel'
            pub = rospy.Publisher(topic_name, Twist, queue_size=1)
            self.vel_publishers.append(pub)
        
        # 订阅手柄话题
        self.joy_sub = rospy.Subscriber('joy', Joy, self.joy_callback) 
        
        rospy.loginfo(f"multi_robot_joy_node has been started. Controlling {self.robot_count} robots.")
        rospy.loginfo(f"Use LB (button {self.prev_button}) to select previous robot, RB (button {self.next_button}) to select next robot.")
        rospy.loginfo(f"Currently controlling robot{self.current_robot + 1}")

    def joy_callback(self, joy_msg): 
        current_time = rospy.Time.now()
        
        # 检查按钮状态
        prev_pressed = (len(joy_msg.buttons) > self.prev_button and 
                       joy_msg.buttons[self.prev_button] == 1)
        next_pressed = (len(joy_msg.buttons) > self.next_button and 
                       joy_msg.buttons[self.next_button] == 1)
        
        # 处理机器人切换（带冷却时间）
        if (current_time - self.last_switch_time) > self.switch_cooldown:
            if prev_pressed and not next_pressed:
                # 切换到上一个机器人
                self.current_robot = (self.current_robot - 1) % self.robot_count
                self.last_switch_time = current_time
                rospy.loginfo(f"Switched to control robot{self.current_robot + 1}")
                return
            elif next_pressed and not prev_pressed:
                # 切换到下一个机器人
                self.current_robot = (self.current_robot + 1) % self.robot_count
                self.last_switch_time = current_time
                rospy.loginfo(f"Switched to control robot{self.current_robot + 1}")
                return
            elif prev_pressed and next_pressed:
                # 同时按下LB和RB，回到第一个机器人
                if self.current_robot != 0:
                    self.current_robot = 0
                    self.last_switch_time = current_time
                    rospy.loginfo(f"Returned to control robot1")
                return
        
        # 创建速度消息
        twist = Twist()  
        linear_val = joy_msg.axes[self.linear_axis] * self.scale_linear
        angular_val = joy_msg.axes[self.angular_axis] * self.scale_angular
        
        # 添加死区处理，避免摇杆微小移动导致机器人抖动
        if abs(linear_val) < 0.1:
            linear_val = 0.0
        if abs(angular_val) < 0.1:
            angular_val = 0.0
            
        twist.linear.x = linear_val
        twist.angular.z = angular_val
        
        # 发布到当前选择的机器人
        self.vel_publishers[self.current_robot].publish(twist)
        
        # 调试输出
        if abs(linear_val) > 0 or abs(angular_val) > 0:
            rospy.loginfo_throttle(1, f"Robot{self.current_robot + 1} - Linear: {linear_val:.2f}, Angular: {angular_val:.2f}")

if __name__ == '__main__':  
    rospy.init_node('multi_robot_joy_node')  
    teleop_turtle = MultiRobotJoy() 
    rospy.spin()