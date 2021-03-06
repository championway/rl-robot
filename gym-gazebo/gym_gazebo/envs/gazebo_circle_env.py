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

class GazeboCircleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        rospy.init_node('gym_gazebo_circle')
        self.laser = LaserScan()
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

        self.pub_twist = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self.reset()

    def cb_laser(self, msg):
        self.laser = msg

    def step(self, action):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_physics()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # Do action
        # cmd_vel = Twist()
        # cmd_vel.linear.x = self.actions[action][0]
        # cmd_vel.angular.z = self.actions[action][1]
        # self.pub_twist.publish(cmd_vel)
        max_ang_speed = 0.5
        ang_vel = (action-10)*max_ang_speed*0.1 #from (-0.33 to + 0.33)
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.2
        cmd_vel.angular.z = ang_vel
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
        state, done = self.calculate_observation(observation)
        
        if not done:
            # Straight reward = 5, Max angle reward = 0.5
            reward = round(15*(max_ang_speed - abs(ang_vel) +0.0335), 2)
            # print ("Action : "+str(action)+" Ang_vel : "+str(ang_vel)+" reward="+str(reward))
        else:
            reward = -200

        return state, reward, done, info

    def reset(self):
        # self.reset_model(self.get_initial_state(
        #     np.random.randint(0, len(INITIAL_STATES))))

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_sim()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

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

        state, done = self.calculate_observation(observation)

        return state

    def get_observation(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
				pass
        return data

    def render(self, mode='laser'):
        pass

    def close(self):
        self.unpause_physics()
        rospy.signal_shutdown('Shutdown')

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