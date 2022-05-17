#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import csv

def callback(odom):
    print(odom.pose.pose)
    print(odom.pose.pose.position.x)
    print(odom.pose.pose.position.y)
    print(odom.pose.pose.orientation.z)
    
    data = [odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.orientation.z]
    
    with open('map_exploration.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(data)


if __main__ == "__main__":
    
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    rospy.init_node('exomy_map_exploration_test')
    
    odom_sub = rospy.Subscriber('/rtabmap/odom', Odometry, callback)
    # depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_callback)
    
    rospy.spin()