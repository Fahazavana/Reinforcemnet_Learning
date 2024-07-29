from collections import deque
from datetime import datetime
import os
import numpy as np
import torch
from torch import nn
from torch import optim

from Utility.DQNEpsilonStrategy import DecreasingEpsilon, EpsilonGreedy
from Utility.MiniGrid import MiniGridImage, get_device
from Utility.Plots import LivePlots, Logs
from Utility.ReplayMemory import ReplayMemory

device = get_device()


class CNN_DQN(nn.Module):
    def __init__(self, height, width, numActions, hiddenLayerSize=(512,)):
        super(CNN_DQN, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=3, stride=2, bias=False),  # 16x27x27
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=False),  # 32x13x13
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, bias=False),  # 64x6x6
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, bias=False),  # 128x2x2
                nn.BatchNorm2d(128),
                nn.Flatten(1, 3)
        )
        self.approximator = nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, numActions)
        )

    def forward(self, x):
        x = self.features(x)
        return self.approximator(x)


class FrameStack:
    def __init__(self, width, height, stack_size):
        self.width = width
        self.height = height
        self.stack_size = stack_size
        self.stack = deque([np.zeros((width, height)) for _ in range(stack_size)], maxlen=stack_size)

    def reset(self):
        self.stack = deque([np.zeros((self.width, self.height)) for _ in range(self.stack_size)],
                           maxlen=self.stack_size)

    def push(self, frame, new_episode=False):
        if new_episode:
            self.reset()
            for _ in range(self.stack_size):
                self.stack.append(frame)
        else:
            self.stack.append(frame)
        return torch.from_numpy(np.stack(self.stack, axis=0)).float().unsqueeze(0).to(device)


class MiniGridDQN():
    def __init__(self, env, policy_net, target_net, gamma, alpha, memory_size):
        self.env = env
        self.gamma = gamma
        self.policy_net = policy_net
        self.target_net = target_net
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(memory_size)
        self.stack = FrameStack(56, 56, 4)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)

    def optimize_model(self, target_net, batch_size):
        if len(self.memory) < batch_size:
            return 2

        batch = self.memory.sample(batch_size)
        state_batch = torch.cat(batch.currentState).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)

        # Q(s,a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        non_final_next_states = (s for s in batch.nextState if s is not None)
        non_final_next_states = torch.cat(tuple(non_final_next_states)).to(device)
        mask = tuple(map(lambda s: s is not None, batch.nextState))
        non_final_mask = torch.tensor(mask, dtype=torch.bool).to(device)
        next_state_values = torch.zeros(batch_size).to(device)

        # max_a Q(s,a',)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        TD_target = reward_batch + next_state_values * self.gamma

        self.optimizer.zero_grad()
        loss = self.criterion(state_action_values, TD_target.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        return loss.item()

    def train(self, batch_size, strategy, sync_freq, episodes):
        start_time = datetime.now()
        print(f"Training started: {start_time.strftime('%H:%M:%S')}")

        logs = Logs()
        live_plots = LivePlots()
        steps_done = 0
        update = 0
        for episode in range(1, episodes + 1):
            current_state = self.env.reset()
            current_stack = self.stack.push(current_state, True)
            for step in range(self.env.maxSteps):
                action = strategy.select_action(current_stack, self.policy_net, steps_done).to(device)
                next_state, reward, done, truncated = self.env.step(action)
                reward = torch.tensor([reward], dtype=torch.float)
                next_stack = self.stack.push(next_state, False)
                self.memory.push(current_stack, action, next_stack, reward)
                steps_done += 1
                update += 1

                loss = self.optimize_model(self.target_net, batch_size)
                if update > sync_freq:
                    update = 0
                    print("Updating target network")
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                if done or truncated:
                    logs.steps_done.append(steps_done)
                    logs.steps_taken.append(self.env.step_count())
                    logs.td_error_sq.append(loss)
                    logs.rewards.append(reward.item())
                    if done:
                        logs.finish_counter += 1
                        print(
                                f"Episode {episode} completed using {self.env.step_count()} steps, and received {reward.item()} reward.")
                    if truncated:
                        print(
                            f"Episode {episode} truncated after {self.env.step_count()} steps, and received {reward.item()} reward.")
                    break
                current_stack = next_stack
            live_plots.e.append(strategy.epsGreedy.epsilon)
            live_plots.l.append(loss)
            live_plots.r.append(reward.item())
            if episode % 5 == 0:
                live_plots.update_plot()

        live_plots.update_plot()
        end_time = datetime.now()
        print(f"\nTraining ended: {end_time.strftime('%H:%M:%S')}")
        delta = datetime(1, 1, 1) + (end_time - start_time)
        print(f"Training took {delta.strftime('%H:%M:%S')}")
        print('\n====== TRAIN SUMMARY ======')
        print(f"Completion rate: {logs.finish_counter / episodes}")
        print(f"Average Reward : {sum(logs.rewards) / episodes:.3f}")
        print(f"Average steps  : {sum(logs.steps_taken) / episodes:.3f}")
        date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        logs.save_log(f"DQNIMAGE_TRAIN_{date_time}.json")
        live_plots.save_and_close(f"DQNIMAGE_LIVE_PLOT_{date_time}.json")


def eval(env, policy_net, strategy, episodes):
    print("\n\nEvaluation...")
    frame_stack = FrameStack(56, 56, 4)
    steps_list, rewards_list = [], []
    finish_counter = 0
    policy_net.eval()
    for episode in range(1, episodes + 1):
        current_state = env.reset()
        current_stack = frame_stack.push(current_state, True)
        for step in range(env.maxSteps):
            action = strategy.select_action(current_stack, policy_net)
            next_state, reward, done, truncated = env.step(action.item())
            next_stack = frame_stack.push(next_state, False)
            if done or truncated:
                steps_list.append(env.step_count())
                if done:
                    rewards_list.append(reward)
                    finish_counter += 1
                break
            current_stack = next_stack

    print('====== EVALUATION SUMMARY ======')
    print(f"Evaluation episodes: {episodes}")
    print(f"Completion rate    : {finish_counter / episodes}")
    print(f"Average Reward     : {sum(rewards_list) / episodes:.3f}")
    print(f"Average steps      : {sum(steps_list) / episodes:.3f}")


if __name__ == '__main__':
    env = MiniGridImage()
    if not os.path.exists('dqn_image.pth'):
        policy_net = CNN_DQN(56, 56, 3).to(device)
        target_net = CNN_DQN(56, 56, 3).to(device)
        minigrid_dqn = MiniGridDQN(env=env,
                                   policy_net=policy_net,
                                   target_net=target_net,
                                   gamma=0.9,
                                   alpha=1e-4,
                                   memory_size=4096)

        strategy = DecreasingEpsilon(start_epsilon=1,
                                     stop_epsilon=0.01,
                                     decay_rate=1e4)

        minigrid_dqn.train(batch_size=128, strategy=strategy, episodes=2000, sync_freq=2048)
        torch.save(minigrid_dqn.policy_net.state_dict(), 'dqn_image.pth')
    else:
        policy_net = CNN_DQN(56, 56, 3).to(device)
        policy_net.load_state_dict(torch.load('dqn_image.pth', map_location=device))
    eval(env, policy_net, EpsilonGreedy(0), 1000)
