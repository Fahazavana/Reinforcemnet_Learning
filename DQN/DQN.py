import torch
from pyroapi import optim
from torch import nn
from Utility.ReplayMemory import ReplayMemory, Transition


class DQN(nn.Module):
    def __init__(self, num_state, num_action, hidden_size=(64, 64)):
        super(DQN, self).__init__()
        self.dqn = nn.Sequential(
            nn.Linear(num_state, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], num_action),
        )
    def forward(self, x):
        return self.dqn(x)



class MiniGridDQN():
    def __init__(self, env, policy, gamm=0.9):
        self.env = env
        self.gamma = 0.9
        self.policy = policy

    def train(self, target, alpha, batch_size, strategy, sync_freq, memory_size):
        steps_done = 0
        update = 0
        target.eval()
        memory = ReplayMemory(memory_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.policy.parameters(), lr=alpha)
        print("Start Training")
