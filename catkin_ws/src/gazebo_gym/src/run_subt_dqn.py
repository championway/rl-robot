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
        self.fc1 = nn.Linear(n_states, 30)
        self.out = nn.Linear(30, n_actions)
        # self.fc1.weight.data.normal_(0, 0.1)   # initialization
        # self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class RunModel(object):
    def __init__(self):
        self.node_name = rospy.get_name()
        rospy.loginfo("[%s] Start Node" %self.node_name)
        self.actions = [[0.5, -0.8],
                        [1.5, -0.8],
                        [1.5, -0.4],
                        [1.5, 0.0],
                        [1.5, 0.4],
                        [1.5, 0.8],
                        [0.5, 0.8]]
        self.n_states = 42
        self.n_actions = len(self.actions)
        self.cuda = True

        self.model_root = '/media/arg_ws3/5E703E3A703E18EB/research/rl/saved_models/subt/'
        self.model = rospy.get_param('~model','dqn_1350.pth')
        MODEL_PATH = os.path.join(self.model_root, self.model)
        self.net = Net(self.n_states, self.n_actions)
        self.net.load_state_dict(torch.load(MODEL_PATH))
        if self.cuda:
            self.net = self.net.cuda()
        rospy.loginfo("Finish network loading...")

        self.pub_twist = rospy.Publisher('/X1/cmd_vel', Twist, queue_size=1)
        
        self.laser_upper = LaserScan()
        self.laser_lower = LaserScan()
        self.sub_laser_upper = rospy.Subscriber(
            '/RL/scan/upper', LaserScan, self.cb_laser_upper, queue_size=1)
        self.sub_laser_lower = rospy.Subscriber(
            '/RL/scan/lower', LaserScan, self.cb_laser_lower, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_pub)

    def timer_pub(self, event):
        observation = self.get_observation()
        if observation is None:
            return
        state, _ = self.calculate_observation(observation)
        if len(state) != self.n_states:
            return
        action = self.choose_action(state)
        cmd_vel = Twist()
        cmd_vel.linear.x = self.actions[action][0]
        cmd_vel.angular.z = self.actions[action][1]
        self.pub_twist.publish(cmd_vel)

    def cb_laser_upper(self, msg):
        self.laser_upper = msg

    def cb_laser_lower(self, msg):
        self.laser_lower = msg

    def get_observation(self):
        if self.laser_lower is not None and self.laser_upper is not None:
            return self.laser_upper, self.laser_lower
        else:
            return None

    def calculate_observation(self, data):
        data1 = data[0]
        data2 = data[1]
        laser = []
        done = False
        min_dis = 0.8
        dis_1 = 1.2
        dis_2 = 2.2
        value = None
        for i, dis in enumerate(list(data1.ranges)):
            if dis == np.inf or dis > dis_2:
                value = 2
            elif dis > dis_1:
                value = 1
            else:
                value = 0
            if dis < min_dis:
                done = True
            laser.append(value)
        for i, dis in enumerate(list(data2.ranges)):
            if dis == np.inf or dis > dis_2:
                value = 2
            elif dis > dis_1:
                value = 1
            else:
                value = 0
            if dis < min_dis:
                done = True
            laser.append(value)
        return np.array(laser), done

    def cb_laser(self, msg):
        self.laser = msg

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if self.cuda:
            x = x.cuda()
        actions_value = self.net.forward(x)
        action = torch.max(actions_value, 1)[1].data.cpu().numpy()
        action = action[0]
        return action

    def on_shutdown(self):
        rospy.loginfo("Close")


if __name__ == '__main__':
    rospy.init_node('rl_dqn')
    runmodel = RunModel()
    rospy.on_shutdown(runmodel.on_shutdown)
    rospy.spin()
