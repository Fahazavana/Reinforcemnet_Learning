import os

import torch
from torch import nn
from torch import optim
from datetime import datetime

from Utility.DQNEpsilonStrategy import DecreasingEpsilon, EpsilonGreedy
from Utility.MiniGrid import MiniGridRaw
from Utility.Plots import LivePlots, Logs
from Utility.ReplayMemory import ReplayMemory


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
    def __init__(self, env, policy_net, target_net, gamma, alpha, memory_size):
        self.env = env
        self.gamma = gamma
        self.policy_net = policy_net
        self.target_net = target_net
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(memory_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)

    def optimize_model(self, target_net, batch_size):
        if len(self.memory) < batch_size:
            return 2

        batch = self.memory.sample(batch_size)
        state_batch = torch.cat(batch.currentState)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Q(s,a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        non_final_next_states = (s for s in batch.nextState if s is not None)
        non_final_next_states = torch.cat(tuple(non_final_next_states))
        mask = tuple(map(lambda s: s is not None, batch.nextState))
        non_final_mask = torch.tensor(mask, dtype=torch.bool)
        next_state_values = torch.zeros(batch_size)

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
            for step in range(self.env.maxSteps):
                action = strategy.select_action(current_state, self.policy_net, steps_done)
                next_state, reward, done, truncated = self.env.step(action)
                reward = torch.tensor([reward], dtype=torch.float)
                self.memory.push(current_state, action, next_state, reward)
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
                current_state = next_state
            live_plots.e.append(strategy.epsGreedy.epsilon)
            live_plots.s.append(self.env.step_count())
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
        logs.save_log(f"DQN_train.json")
        live_plots.save_and_close(f"DQN_live_plot.json")


def eval(env, policy_net, strategy, episodes):
    print("Evaluation...")
    steps_list, rewards_list = [], []
    finish_counter = 0
    policy_net.eval()
    for episode in range(1, episodes + 1):
        current_state = env.reset()
        for step in range(env.maxSteps):
            action = strategy.select_action(current_state, policy_net)
            next_state, reward, done, truncated = env.step(action.item())
            if done or truncated:
                steps_list.append(env.step_count())
                rewards_list.append(reward)
                if done:
                    finish_counter += 1
                break
            current_state = next_state

    print('====== EVALUATION SUMMARY ======')
    print(f"Evaluation episodes: {episodes}")
    print(f"Completion rate    : {finish_counter / episodes}")
    print(f"Average Reward     : {sum(rewards_list) / episodes:.3f}")
    print(f"Average steps      : {sum(steps_list) / episodes:.3f}")


if __name__ == '__main__':
    env = MiniGridRaw()
    if not os.path.exists('dqn.pth'):
        policy_net = DQN(env.numStates, env.numActions, (64, 32))
        target_net = DQN(env.numStates, env.numActions, (64, 32))
        minigrid_dqn = MiniGridDQN(env=env,
                                   policy_net=policy_net,
                                   target_net=target_net,
                                   gamma=0.9,
                                   alpha=1e-4,
                                   memory_size=1024 * 4)

        strategy = DecreasingEpsilon(start_epsilon=1,
                                     stop_epsilon=0.01,
                                     decay_rate=1e4)

        minigrid_dqn.train(batch_size=256, strategy=strategy, episodes=2000, sync_freq=2048)
        torch.save(minigrid_dqn.policy_net.state_dict(), 'dqn.pth')
    else:
        policy_net = DQN(env.numStates, env.numActions, (64, 32))
        policy_net.load_state_dict(torch.load('dqn.pth'))

    eval(env, policy_net, EpsilonGreedy(0), 1000)
