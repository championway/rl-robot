import gym
import numpy as np
import torch
from dqn_pytorch import DQN
import os

env = gym.make('gym_gazebo:gazebo-circle-v0')
env = env.unwrapped

BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
REPLACE_TARGET_ITER = 300   # target update frequency
MEMORY_CAPACITY = 4000
MODEL_ROOT='/media/arg_ws3/5E703E3A703E18EB/research/rl/saved_models/'
if not os.path.exists(MODEL_ROOT):
  os.makedirs(MODEL_ROOT)

dqn = DQN(#n_actions=len(env.actions),
          # n_states=len(env.get_observation()),
          n_actions=21,
          n_states=180,
          learning_rate=LR,
          e_greedy=EPSILON,
          replace_target_iter=REPLACE_TARGET_ITER,
          memory_size=MEMORY_CAPACITY,
          e_greedy_increment=0.001,
          batch_size = BATCH_SIZE,
          gamma=GAMMA,
          cuda=True
          )

print('\nCollecting experience...')

for i_episode in range(5000):

    state = env.reset()

    ep_r = 0
    while True:
        # env.render()
        action = dqn.choose_action(state)

        # take action
        state_, reward, done, info = env.step(action)
        # print(len(observation_))

        dqn.store_transition(state, action, reward, state_)

        ep_r += reward
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
        if done:
            if i_episode % 20 == 0:
                print('Ep: ', i_episode,
                    '| Ep_r: ', round(ep_r, 2),
                    '| Epsilon: ', round(dqn.epsilon, 2))
            break
        state = state_
    if i_episode % 50 == 0 and i_episode != 0:
        torch.save(dqn.eval_net.state_dict(), MODEL_ROOT + 'dqn_%d.pth' % (i_episode))  
torch.save(dqn.eval_net.state_dict(), MODEL_ROOT + 'dqn.pth')
    