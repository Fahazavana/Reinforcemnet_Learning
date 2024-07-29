import hashlib
import platform

import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


def get_device():
    if platform.platform().lower().startswith("mac"):  # macOS
        return "mps" if torch.backends.mps.is_available() else "cpu"
    else:  # Linux, Windows
        return "cuda" if torch.cuda.is_available() else "cpu"


class MiniGridBase():
    def __init__(self, env_name="MiniGrid-Empty-8x8-v0", render_mode=None):
        self.env = gym.make(env_name, render_mode=render_mode if render_mode is not None else None)
        self.numRow = self.env.unwrapped.height - 1
        self.numCol = self.env.unwrapped.width - 1
        self.numStates = self.numCol * self.numRow
        self.maxSteps = self.env.unwrapped.max_steps
        self.numActions = 3

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def step_count(self):
        return self.env.unwrapped.step_count

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()


class MiniGridHash(MiniGridBase):
    def __init__(self, env_name="MiniGrid-Empty-8x8-v0", render_mode=None):
        super().__init__(env_name, render_mode)
        self.env = ImgObsWrapper(self.env)

    def reset(self):
        obs, _ = self.env.reset()
        return self.__hash(obs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self.__hash(obs), reward, done, truncated

    def __hash(self, obs):
        state_str = str(obs.flatten())
        return hashlib.md5(state_str.encode()).hexdigest()


class MiniGridRaw(MiniGridBase):
    """
    This class use the provided base code for deep Q-net
    """

    def __init__(self, env_name="MiniGrid-Empty-8x8-v0", render_mode=None):
        super().__init__(env_name, render_mode)
        self.env = ImgObsWrapper(self.env)

    def __extractObjectInformation_(self, observation):
        (rows, cols, x) = observation.shape
        view = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                view[r, c] = observation[r, c, 0]
        return view

    def __getObjectInformation(self, observation):
        (rows, cols, x) = observation.shape
        tmp = np.reshape(observation, (rows * cols * x, 1), 'F')[0:rows * cols]
        return np.reshape(tmp, (rows, cols), 'C')

    def __normalize(self, observation, max_value=10.0):
        return observation / max_value

    def __flatten(self, observation):
        return torch.from_numpy(np.array(observation).flatten()).float().unsqueeze(0)

    def __preprocess(self, observation):
        return self.__flatten(self.__normalize(self.__getObjectInformation(observation), 10))

    def step(self, action):
        obs, reward, done, truncated, _ = self.env.step(action)
        if done or truncated:
            state = None
        else:
            state = self.__preprocess(obs)
        return state, reward, done, truncated

    def reset(self):
        obs, _ = self.env.reset()
        return self.__preprocess(obs)


class MiniGridImage(MiniGridBase):
    """
    This class use the provided base code for the RGBImage technique
    """

    def __init__(self, render_mode=None):
        super().__init__(env_name='MiniGrid-Empty-8x8-v0', render_mode=render_mode)
        self.env = ImgObsWrapper(RGBImgPartialObsWrapper(self.env))
        self.screen_height, self.screen_width = 56, 56

    def __tograyscale(self, frame):
        return np.mean(frame, -1)

    def __normalize(self, frame, max_value=255.0):
        return frame / max_value

    def __toframe(self, frame):
        return self.__normalize(self.__tograyscale(frame))

    def step(self, action):
        obs, reward, done, truncated, _ = self.env.step(action)
        state = self.__toframe(obs)
        return state, reward, done, truncated

    def reset(self):
        obs, _ = self.env.reset()
        return self.__toframe(obs)
