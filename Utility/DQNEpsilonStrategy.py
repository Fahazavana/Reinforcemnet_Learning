import random

import numpy as np
import torch


class EpsilonGreedy:
    def __init__(self, epsilon=0.1, num_actions=3, seed=42):
        random.seed(seed)
        self.epsilon = epsilon
        self.num_actions = num_actions

    def select(self, state, policy_net):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return policy_net(state).argmax(1).unsqueeze(0).detach()
        return torch.tensor([[random.randrange(self.num_actions)]], dtype=torch.long)


class DecreasingEpsilon:
    def __init__(self, start_epsilon, stop_epsilon, decay_rate, num_actions=3):
        self.epsGreedy = EpsilonGreedy(start_epsilon, num_actions)
        self.num_actions = num_actions
        self.start_epsilon = start_epsilon
        self.stop_epsilon = stop_epsilon
        self.decay_rate = decay_rate

    def reset(self):
        self.epsGreedy = EpsilonGreedy(self.start_epsilon, self.num_actions)

    def select(self, state, policy_net, steps):
        res = self.epsGreedy.select(state, policy_net)
        self.epsGreedy.epsilon = (self.start_epsilon - self.stop_epsilon) * np.exp(
            -1 * steps / self.decay_rate) + self.stop_epsilon
        return res
