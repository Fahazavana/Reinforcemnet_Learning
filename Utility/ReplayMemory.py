import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('currentState', 'action', 'nextState', 'reward',))


class ReplayMemory:
    def __init__(self, capacity, seed=42):
        random.seed(seed)
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
