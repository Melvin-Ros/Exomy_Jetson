#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import csv

def callback(odom):
    #print(odom.pose.pose)
    #print(odom.pose.pose.position.x)
    #print(odom.pose.pose.position.y)
    #print(odom.pose.pose.orientation.z)
    
    orientation_quat = [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]
    orientation_euler = euler_from_quaternion (orientation_quat)
    orientation_euler = [orientation_euler[0],orientation_euler[1],orientation_euler[2]]
    
    data = [odom.pose.pose.position.x, odom.pose.pose.position.y, orientation_euler[2]]
    print(data)
    with open('/home/jetson-xavier/Desktop/map_exploration.csv', 'a+') as file:
        writer = csv.writer(file)
        writer.writerow(data)


    
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

rospy.init_node('exomy_map_exploration_test')
    
odom_sub = rospy.Subscriber('/rtabmap/odom', Odometry, callback)
    # depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_callback)
    
rospy.spin()
