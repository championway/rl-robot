import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 30)
        self.out = nn.Linear(30, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(
            self,
            n_actions,
            n_states,
            learning_rate=0.01,
            e_greedy=0.9,   # greedy policy
            gamma = 0.9,    # reward discount
            replace_target_iter=100,    # target update frequency
            memory_size=2000,
            batch_size=32,
            e_greedy_increment=None,
            cuda=True
    ):
        self.cuda = cuda
        self.n_actions = n_actions
        self.n_states = n_states
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.eval_net, self.target_net = Net(self.n_states, self.n_actions), Net(self.n_states, self.n_actions)
        self.learn_step_counter = 0                                   # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((self.memory_size, self.n_states * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        if self.cuda:
            self.eval_net = self.eval_net.cuda()
            self.target_net = self.target_net.cuda()
            self.loss_func = self.loss_func.cuda()
        
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < self.epsilon:   # greedy
            if self.cuda:
                x = x.cuda()
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, self.n_actions)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        # increasing epsilon
        if self.learn_step_counter % 3 == 0:
            self.epsilon = self.epsilon + \
            self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.memory_size, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:])

        if self.cuda:
            b_s = b_s.cuda()
            b_a = b_a.cuda()
            b_r = b_r.cuda()
            b_s_ = b_s_.cuda()
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()