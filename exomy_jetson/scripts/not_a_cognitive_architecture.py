#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import cv2
import onnx
import onnxruntime as ort
from torch import nn
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge 
from tf.transformations import euler_from_quaternion, quaternion_from_euler



#Load model
path = '/home/jetson-xavier/model_TowardsGoal.onnx'
ort_sess = ort.InferenceSession(path)



#global variables for observations
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
action = torch.tensor([0,0])
action_dict = torch.tensor([0,0])
progress_buf = torch.tensor([0])
cv_image_tensor = torch.tensor([0])

def callback(odom):
    global cv_image_tensor
    if(cv_image_tensor.nelement != 0):
    	global action
    	goal = [1,0]
    	obs = calculate_observations(odom.pose.pose, cv_image_tensor, goal, action)
    
    	[lin_vel,ang_vel]= not_ca(obs)
    	move_rover(lin_vel, ang_vel)
    
def depth_callback(depth_data):
    br = CvBridge()
    current_frame = br.imgmsg_to_cv2(depth_data)
    dim = (10, 20)

    current_frame = cv2.resize(current_frame, dim)
    cv_image_array = np.array(current_frame, dtype = np.dtype('f8'))
    global cv_image_tensor
    cv_image_tensor = torch.tensor(cv_image_array/1000)
    #np.savetxt('test.out', cv_image_array, delimiter=',',fmt='%i')
    #cv2.imwrite('IMAGE.PNG', (cv_image_array/3000) * 255)
    
    
def move_rover(lin,ang):
    move_cmd = Twist()
    move_cmd.linear.x = lin*4
    move_cmd.angular.z = ang*4
    pub.publish(move_cmd)
    
def not_ca(obs):
    global ort_sess
    #act = ort_sess.run(['mu', 'sigma', 'value'], {'obs': obs})
    act = ort_sess.run(None, {'obs': obs})
    global action
    action = torch.tensor([act[0][0][0],act[0][0][1]])
    global action_dict
    print("action: ", action)
    
    # action clamp
    lin_condition = torch.where(action[0] > action_dict[0], 1, -1)
    ang_condition = torch.where(action[1] > action_dict[1], 1, -1)
    lin_vel_difference = abs(action[0] - action_dict[0])
    ang_vel_difference = abs(action[1] - action_dict[1])
    lin_vel_dif_clamp = torch.clamp(lin_vel_difference, 0, 0.02)
    ang_vel_dif_clamp = torch.clamp(ang_vel_difference, 0, 0.02)
    clamped = action
    # Clamped actions are saved
    clamped[0] = action_dict[0] + lin_condition * lin_vel_dif_clamp
    clamped[1] = action_dict[1] + ang_condition * ang_vel_dif_clamp
    print("clamped action: ", clamped)
    
    # update action_dict
    action_dict = action
    return action

def calculate_observations(robot_pose, depth_stream, target, action):
    global progress_buf
    pos = torch.tensor([-robot_pose.position.y, -robot_pose.position.x])
    goal = torch.tensor([target[0], target[1]])
    
    orientation_quat = [robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w]
    orientation_euler = euler_from_quaternion (orientation_quat)
    orientation_euler = torch.tensor([orientation_euler[0],orientation_euler[1],orientation_euler[2]])
    #orientation_euler[2] -= torch.pi
    
    # Calculate target_vector
    target_vector = goal - pos
    
    # Calculate heading diff
    eps = 1e-7
    dot = ((target_vector[0] * torch.cos(orientation_euler[2] - (torch.pi/2))) + (target_vector[1] * torch.sin(orientation_euler[2] - (torch.pi/2)))) / ((torch.sqrt(torch.square(target_vector[0]) + torch.square(target_vector[1]))) * torch.sqrt(torch.square(torch.cos(orientation_euler[2] - (torch.pi/2))) + torch.square(torch.sin(orientation_euler[2] - (torch.pi/2)))))
    
    condition = ((target_vector[0] * torch.cos(orientation_euler[2])) + (target_vector[1] * torch.sin(orientation_euler[2]))) / ((torch.sqrt(torch.square(target_vector[0]) + torch.square(target_vector[1]))) * torch.sqrt(torch.square(torch.cos(orientation_euler[2])) + torch.square(torch.sin(orientation_euler[2]))))
    
    angle = torch.clamp(dot, min = (-1 + eps), max = (1 - eps))
    obs_heading_diff = torch.where(condition < 0, -1 * torch.arccos(angle), torch.arccos(angle))/torch.pi
    
    # Calculate target_dist
    target_dist = (torch.sqrt(torch.square(target_vector).sum(-1)))
    
    # Calculate progress_buf
    if target_dist < 0.2:
        progress_buf = 0
    progress_buf = torch.add(progress_buf,1/3000)
    
    input_tensor = torch.zeros([55])
    input_tensor[0] = action[0]
    input_tensor[1] = action[1]
    input_tensor[2] = obs_heading_diff
    input_tensor[3] = target_dist/5
    input_tensor[4] = progress_buf
    input_tensor[5:15] = cv_image_tensor[2]
    input_tensor[15:25] = cv_image_tensor[6]
    input_tensor[25:35] = cv_image_tensor[9]
    input_tensor[35:45] = cv_image_tensor[12]
    input_tensor[45:55] = cv_image_tensor[16]
    #input_tensor = torch.tensor([0.4367, -0.1000,  0.1150,  0.4492,  0.0293, -1.0047, -1.0011, -0.9979,
    #    -0.9957, -0.9917, -0.9880, -0.9851, -0.9826, -0.9790, -0.9754, -0.6968,
    #    -0.6951, -0.6934, -0.6919, -0.6903, -0.6886, -0.6868, -0.6852, -0.6838,
    #    -0.6822, -0.5331, -0.5321, -0.5312, -0.5303, -0.5294, -0.5284, -0.5274,
    #    -0.5264, -0.5254, -0.5245, -0.4317, -0.4311, -0.4304, -0.4298, -0.4292,
    #    -0.4287, -0.4280, -0.4274, -0.4268, -0.4261, -0.3627, -0.3622, -0.3618,
    #    -0.3614, -0.3610, -0.3606, -0.3601, -0.3597, -0.3593, -0.3588])
    #input_tensor[5:55] = torch.tensor([-1.0108, -1.0094, -1.0085,
    #    -1.0064, -1.0043, -1.0028, -1.0013, -0.9998, -0.9986, -0.9972, -0.6981,
    #    -0.6973, -0.6967, -0.6961, -0.6954, -0.6946, -0.6938, -0.6930, -0.6921,
    #    -0.6916, -0.5334, -0.5330, -0.5325, -0.5320, -0.5315, -0.5311, -0.5306,
    #    -0.5301, -0.5296, -0.5292, -0.4315, -0.4312, -0.4309, -0.4305, -0.4302,
    #    -0.4300, -0.4296, -0.4293, -0.4290, -0.4287, -0.3622, -0.3620, -0.3618,
    #    -0.3616, -0.3614, -0.3612, -0.3610, -0.3608, -0.3606, -0.3603])
    
    #Debugging prints
    print("goal:" , goal)
    print("position: ", pos)
    print("orientation: ", orientation_euler)
    print("target_dist", target_dist)
    print("progress_buf", progress_buf)
    print("heading_diff", obs_heading_diff)
    obs = input_tensor.unsqueeze(0).numpy()
    #obs = input_tensor.unsqueeze(0).numpy
    return obs
    
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
rospy.init_node('totally_not_a_CA')
odom_sub = rospy.Subscriber('/rtabmap/odom', Odometry, callback)
depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw',Image,depth_callback)
rospy.spin()

