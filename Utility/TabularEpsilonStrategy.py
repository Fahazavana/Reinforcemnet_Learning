import numpy as np


class EpsilonGreedy:
    def __init__(self, epsilon=0.1, num_actions=3, seed=42):
        self.epsilon = epsilon
        self.actions = range(num_actions)
        np.random.seed(seed)

    def select_action(self, state, qTable):
        if np.random.rand() > self.epsilon:
            return np.argmax(qTable[state])
        return np.random.choice(self.actions)


class DecreasingEpsilon:
    def __init__(self, start_epsilon, stop_epsilon, decay_rate, num_actions=3):
        self.epsGreedy = EpsilonGreedy(start_epsilon, num_actions)
        self.num_actions = num_actions
        self.start_epsilon = start_epsilon
        self.stop_epsilon = stop_epsilon
        self.decay_rate = decay_rate

    def reset(self):
        self.epsGreedy = EpsilonGreedy(self.start_epsilon, self.num_actions)

    def select_action(self, state, qTable, steps):
        res = self.epsGreedy.select_action(state, qTable)
        self.epsGreedy.epsilon = (self.start_epsilon - self.stop_epsilon) * np.exp(
                -1 * steps / self.decay_rate) + self.stop_epsilon
        return res
