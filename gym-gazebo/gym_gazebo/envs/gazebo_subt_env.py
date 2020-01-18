import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

STEP_TIME = 0.01

# [x,y,z,x,y,z,w]
INITIAL_STATES = [[14.38, -0.05, -0.74, 0, 0.14, 0, 1],
                  [89.1, 0, -19.87, 0, 0, 0, -1],
                  [100, 22.5, -20, 0, 0, -0.7, -0.7],
                  [100, 63.13, -20, 0, 0, -0.7, -0.7],
                  [80, 51, -20, 0, 0, -0.74, 0.66],
                  [159, 0, -20, 0, 0, 0, -1],
                  [80, 60, -20, 0, 0, -0.7, -0.7],
                  [116, 40, -20, 0, 0, 1, 0],
                  [140, 43, -20, 0, 0, 0.7, 0.7]]

class GazeboSubtEnv(gym.Env):
    metadata = {'render.modes': ['laser']}

    def __init__(self):
        rospy.init_node('gym_gazebo_subt')
        self.laser_upper = LaserScan()
        self.laser_lower = LaserScan()
        self.n_action = 21
        self.actions = [[0.5, -0.8],
                        [1.5, -0.8],
                        [1.5, -0.4],
                        [1.5, 0.0],
                        [1.5, 0.4],
                        [1.5, 0.8],
                        [0.5, 0.8]]

        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.reset_model = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        self.pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        self.pub_twist = rospy.Publisher('/X1/cmd_vel', Twist, queue_size=1)

        self.sub_laser_upper = rospy.Subscriber(
            '/RL/scan/upper', LaserScan, self.cb_laser_upper, queue_size=1)
        self.sub_laser_lower = rospy.Subscriber(
            '/RL/scan/lower', LaserScan, self.cb_laser_lower, queue_size=1)
        
        self.reset()

    def cb_laser_upper(self, msg):
        self.laser_upper = msg

    def cb_laser_lower(self, msg):
        self.laser_lower = msg

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        cmd_vel = Twist()
        # cmd_vel.linear.x = self.actions[action][0]
        # cmd_vel.angular.z = self.actions[action][1]
        cmd_vel.linear.x = self.get_action(action)[1]
        cmd_vel.angular.z = self.get_action(action)[0]

        self.pub_twist.publish(cmd_vel)

        # Get environment laserscan state observation
        observation = self.get_observation()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # Caculate reward and check if it's done
        info = {}
        state, done = self.get_state(observation)
        
        # factor = 3-abs(action-3)
        # reward = 2**factor
        # reward = round(15*(0.8*2 - abs(cmd_vel.angular.z)), 2)
        n_factor = int((self.n_action-1)/2)
        reward = round((n_factor - abs(action - n_factor))*5/n_factor, 2)

        too_close = False
        for i, dis in enumerate(state):
            if dis < 1 :
                too_close = True
        if too_close:
            reward = -50
        if done:
            reward = -200
        # print action, reward

        return state, reward, done, info

    def reset(self):
        self.reset_model(self.get_initial_state(
            np.random.randint(0, len(INITIAL_STATES))))

        # Resets the state of the environment and returns an initial observation.
        # rospy.wait_for_service('/gazebo/reset_simulation')
        # try:
        #     self.reset_sim()
        # except (rospy.ServiceException) as e:
        #     print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # Read laser data
        # rospy.loginfo('reset model')

        observation = self.get_observation()

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            #resp_pause = pause.call()
            self.pause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        state, done = self.get_state(observation)

        return state

    def get_observation(self):
        data1 = None
        data2 = None
        while data1 is None or data2 is None:
            try:
                data1 = rospy.wait_for_message(
                    '/RL/scan/upper', LaserScan, timeout=5)
                data2 = rospy.wait_for_message(
                    '/RL/scan/lower', LaserScan, timeout=5)
            except:
				pass
        return data1, data2

    def render(self, mode='laser'):
        pass

    def close(self):
        self.unpause_physics()
        rospy.signal_shutdown('Shutdown')

    def get_state(self, data):
        data1 = data[0]
        data2 = data[1]
        laser = []
        done = False
        min_dis = 0.8
        max_dis = 1.5
        value = None
        for i, dis in enumerate(list(data1.ranges)):
            if dis == np.inf or dis > max_dis:
                value = max_dis
            else:
                value = dis
            if dis < min_dis:
                done = True
            laser.append(value)
        for i, dis in enumerate(list(data2.ranges)):
            if dis == np.inf or dis > max_dis:
                value = max_dis
            else:
                value = dis
            if dis < min_dis:
                done = True
            laser.append(value)
        return np.array(laser), done

    def discretize_observation(self, data, new_ranges):
        discretized_ranges = []
        min_range = 0.2
        done = False
        mod = len(data.ranges)/new_ranges
        for i, item in enumerate(data.ranges):
            if (i%mod==0):
                if data.ranges[i] == float ('Inf'):
                    discretized_ranges.append(6)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0)
                else:
                    discretized_ranges.append(int(data.ranges[i]))
            if (min_range > data.ranges[i] > 0):
                done = True
        return discretized_ranges, done

    def get_action(self, id):
        # y = -((1.3*x)**2) + 1.5
        # range: -0.8~0.8
        delta = float(1.6/(self.n_action - 1))
        index = int(-(self.n_action - 1)/2) + id
        x = float(index) * delta
        y = -((1.3*x)**2) + 1.5
        return x, y

    def get_initial_state(self, id):
        # start position
        state_msg = ModelState()
        state_msg.model_name = 'X1'
        state_msg.pose.position.x = INITIAL_STATES[id][0]
        state_msg.pose.position.y = INITIAL_STATES[id][1]
        state_msg.pose.position.z = INITIAL_STATES[id][2]
        state_msg.pose.orientation.x = INITIAL_STATES[id][3]
        state_msg.pose.orientation.y = INITIAL_STATES[id][4]
        state_msg.pose.orientation.z = INITIAL_STATES[id][5]
        state_msg.pose.orientation.w = INITIAL_STATES[id][6]
        return state_msg