# rl-robot
Reinforcement learning with Gym, Gazebo and real robot

# Develop environment

- ROS kinetic 
- Ubuntu 16.04

# Setup environment

```
$ git clone https://github.com/championway/rl-robot.git
$ cd ~/rl-robot && source environment.sh
```

## Install gym-gazebo
```
$ pip install gym
$ cd gym-gazebo
$ pip install -e .
```
Add "--user" when pip install encounter permission deny

## Turtlebot3 Gazebo environment
```
$ sudo apt-get install ros-kinetic-turtlebot3-bringup
$ cd ~/rl-robot/catkin_ws/src/
$ git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
$ cd ../ && catkin_make
```
Reference: http://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/

# How to run

## Build environment
(Do this everytime when you start a new terminal)
```
$ cd ~/rl-robot/
$ source environment.sh
```

## Open Gazebo environment

- TurtleBot3 world
```
$ export TURTLEBOT3_MODEL=burger
$ roslaunch turtlebot3_gazebo turtlebot3_world.launch
```

## Gym-gazebo Training
- DQN
```
$ cd ~/rl-robot/rl-algorithm/DQN
$ python circuit_turtlebot_dqn.py
```
See training [tips](#Tips)

## Tips
### gazebo speed up
physics
- realtime update rate 0
- max step size 0.01