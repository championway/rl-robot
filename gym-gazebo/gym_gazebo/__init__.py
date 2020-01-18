from gym.envs.registration import register

register(
    id='gazebo-circle-v0',
    entry_point='gym_gazebo.envs:GazeboCircleEnv',
)
register(
    id='gazebo-subt-v0',
    entry_point='gym_gazebo.envs:GazeboSubtEnv',
)