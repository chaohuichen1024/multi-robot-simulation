#!/usr/bin/env python3
'''
Descripttion: 多机器人键盘控制节点
Author: chaohui_chen
Date: 2024-06-08 17:22:35
LastEditors: chaohui_chen1024
LastEditTime: 2025-10-23 21:00:59
'''
  
import rospy  
import sys
import select
import termios
import tty
from geometry_msgs.msg import Twist  

class MultiRobotKeyboard:  
    def __init__(self):  
        # 获取参数
        self.scale_linear = rospy.get_param('~scale_linear', 0.5)  
        self.scale_angular = rospy.get_param('~scale_angular', 1.0)  
        
        # 机器人数量
        self.robot_count = rospy.get_param('~robot_count', 3)
        
        # 当前控制的机器人索引
        self.current_robot = 0
        
        # 为每个机器人创建速度发布器
        self.vel_publishers = []
        for i in range(self.robot_count):
            topic_name = f'robot{i+1}/cmd_vel'
            pub = rospy.Publisher(topic_name, Twist, queue_size=1)
            self.vel_publishers.append(pub)
        
        # 保存终端设置
        self.settings = termios.tcgetattr(sys.stdin)
        
        # 控制指令
        self.linear_x = 0.0
        self.angular_z = 0.0
        
        rospy.loginfo(f"multi_robot_keyboard_node has been started. Controlling {self.robot_count} robots.")
        self.print_instructions()

    def print_instructions(self):
        print("")
        print("Control Your Robots!")
        print("---------------------------")
        print("Robot Selection:")
        print("   q : Previous robot")
        print("   e : Next robot")
        print("")
        print("Movement Control:")
        print("   w : Forward")
        print("   s : Backward") 
        print("   a : Turn left")
        print("   d : Turn right")
        print("   x : Stop")
        print("")
        print("Speed Adjustment:")
        print("   1 : Increase linear speed by 10%")
        print("   2 : Decrease linear speed by 10%")
        print("   3 : Increase angular speed by 10%")
        print("   4 : Decrease angular speed by 10%")
        print("")
        print("CTRL-C to quit")
        print(f"Currently controlling robot{self.current_robot + 1}")
        print(f"Linear speed: {self.scale_linear:.2f}, Angular speed: {self.scale_angular:.2f}")
        print("")

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def process_key(self, key):
        # 机器人选择
        if key == 'q':
            self.current_robot = (self.current_robot - 1) % self.robot_count
            print(f"Switched to control robot{self.current_robot + 1}")
            return
        elif key == 'e':
            self.current_robot = (self.current_robot + 1) % self.robot_count
            print(f"Switched to control robot{self.current_robot + 1}")
            return
        
        # 运动控制
        if key == 'w':
            self.linear_x = self.scale_linear
            self.angular_z = 0.0
        elif key == 's':
            self.linear_x = -self.scale_linear
            self.angular_z = 0.0
        elif key == 'a':
            self.linear_x = 0.0
            self.angular_z = self.scale_angular
        elif key == 'd':
            self.linear_x = 0.0
            self.angular_z = -self.scale_angular
        elif key == 'x':
            self.linear_x = 0.0
            self.angular_z = 0.0
        
        # 速度调节
        elif key == '1':
            self.scale_linear *= 1.1
            print(f"Linear speed increased to: {self.scale_linear:.2f}")
        elif key == '2':
            self.scale_linear *= 0.9
            print(f"Linear speed decreased to: {self.scale_linear:.2f}")
        elif key == '3':
            self.scale_angular *= 1.1
            print(f"Angular speed increased to: {self.scale_angular:.2f}")
        elif key == '4':
            self.scale_angular *= 0.9
            print(f"Angular speed decreased to: {self.scale_angular:.2f}")

    def run(self):
        try:
            while not rospy.is_shutdown():
                key = self.get_key()
                if key != '':
                    # 退出
                    if key == '\x03':  # CTRL-C
                        break
                    
                    self.process_key(key)
                
                # 发布速度命令
                twist = Twist()
                twist.linear.x = self.linear_x
                twist.angular.z = self.angular_z
                self.vel_publishers[self.current_robot].publish(twist)
                
                # 显示状态信息（限制频率）
                if rospy.Time.now().to_sec() % 1.0 < 0.1:
                    if abs(self.linear_x) > 0 or abs(self.angular_z) > 0:
                        status = f"Robot{self.current_robot + 1} - Linear: {self.linear_x:.2f}, Angular: {self.angular_z:.2f}"
                        print(status, end='               \r')
                
        except Exception as e:
            rospy.logerr(f"Error in keyboard control: {e}")
        finally:
            # 停止所有机器人
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            for pub in self.vel_publishers:
                pub.publish(twist)
            
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            print("\nControl node has been shut down.")

if __name__ == '__main__':  
    rospy.init_node('multi_robot_keyboard_node')  
    controller = MultiRobotKeyboard() 
    controller.run()