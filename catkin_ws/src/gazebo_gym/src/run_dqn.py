#! /usr/bin/env python

import rospy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import os

class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 300)
        self.out = nn.Linear(300, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class RunModel(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Start Node" %self.node_name)
        self.n_states = 180
        self.n_actions = 21
        
        self.model_root = '/media/arg_ws3/5E703E3A703E18EB/research/rl/saved_models/'
        self.model = rospy.get_param('~model','dqn_2200.pth')
        MODEL_PATH = os.path.join(self.model_root, self.model)
        self.net = Net(self.n_states, self.n_actions)
        self.net.load_state_dict(torch.load(MODEL_PATH))
        self.net = self.net.cuda()
        rospy.loginfo("Finish network loading...")

        self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.laser = LaserScan()

        self.sub_laser = rospy.Subscriber('/scan', LaserScan, self.cb_laser, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_pub)

    def timer_pub(self, event):
        observation = self.get_observation()
        state, _ = self.calculate_observation(observation)
        if len(state) != self.n_states:
            print(len(state))
            return
        action = self.choose_action(state)
        max_ang_speed = 0.5
        ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.2
        cmd_vel.angular.z = ang_vel
        self.pub_twist.publish(cmd_vel)

    def get_observation(self):
        return self.laser

    def calculate_observation(self, data):
        laser = []
        done = False
        min_dis = 0.2
        dis_threshold = 0.45
        for i, dis in enumerate(list(data.ranges)):
            if i >= 180:
                break
            if dis == np.inf or dis > dis_threshold:
                laser.append(0)
            else:
                laser.append(1)
            if dis < min_dis:
                done = True
        return np.array(laser), done

    def cb_laser(self, msg):
        self.laser = msg

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).cuda()
        actions_value = self.net.forward(x)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        return action

    def on_shutdown(self):
        rospy.loginfo("Close")


if __name__ == '__main__':
    rospy.init_node('rl_dqn')
    runmodel = RunModel()
    rospy.on_shutdown(runmodel.on_shutdown)
    rospy.spin()
