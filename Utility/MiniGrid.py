import gymnasium as gym
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


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


class MiniGridXY(MiniGridBase):
    """
    This class use agent position (y,x) as observation,
    and removing unnecessary information.
    """

    def __init__(self, env_name="MiniGrid-Empty-8x8-v0", render_mode=None):
        super().__init__(env_name, render_mode)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if done:
            obs = None
        else:
            obs = self.env.unwrapped.agent_pos
        return obs, reward, done, truncated

    def reset(self):
        _ = self.env.reset()
        return self.env.unwrapped.agent_pos


class MiniGridID(MiniGridBase):
    """
    This class use agent state based on the ceel coordinate as observation,
    """

    def __init__(self, env_name="MiniGrid-Empty-8x8-v0", render_mode=None):
        super().__init__(env_name, render_mode)

    def reset(self):
        _ = self.env.reset()
        return self.__get_state_id()

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if done:
            obs = None
        else:
            obs = self.__get_state_id()
        return self.env.unwrapped.agent_pos, reward, done, truncated

    def __get_state_id(self):
        row, col = self.env.unwrapped.agent_pos
        cell_id = (row - 1) + (col - 1) * self.numCol
        return cell_id


class MiniGridOneHot(MiniGridBase):
    def __init__(self, env_name="MiniGrid-Empty-8x8-v0", render_mode=None):
        super().__init__(env_name, render_mode)

    def step(self, action):
        _, reward, done, truncated, _ = self.env.step(action)
        if done or truncated:
            state = None
        else:
            state = self.__onehot()
        return state, reward, done, truncated

    def reset(self):
        _ = self.env.reset()
        return self.__onehot()

    def __onehot(self):
        row, col = self.env.unwrapped.agent_pos
        cellId = (row - 1) + (col - 1) * self.numCol
        oneHot = torch.zeros(self.numStates)
        oneHot[cellId] = 1
        return oneHot.unsqueeze(0)


class MiniGridRaw(MiniGridBase):
    """
    This class use the provided base code for deep Q-net
    """

    def __init__(self, env_name="MiniGrid-Empty-8x8-v0", render_mode=None):
        super().__init__(env_name, render_mode)
        self.env = ImgObsWrapper(super().env)

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
        self.env = RGBImgPartialObsWrapper(ImgObsWrapper(super().env))
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
