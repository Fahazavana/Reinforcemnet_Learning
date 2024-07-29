import os
import pickle
from datetime import datetime

import numpy as np

from Utility.MiniGrid import MiniGridHash
from Utility.Plots import LivePlots, Logs
from Utility.TabularEpsilonStrategy import DecreasingEpsilon, EpsilonGreedy


class QLearning:
    def __init__(self, env):
        self.env = env
        self.q_table = {}

    def train(self, gamma, alpha, strategy: DecreasingEpsilon, episodes):
        start_time = datetime.now()
        print(f"Training started: {start_time.strftime('%H:%M:%S')}")

        logs = Logs()
        live_plots = LivePlots()
        steps_done = 0
        for episode in range(1, episodes + 1):
            current_state = self.env.reset()
            if current_state not in self.q_table:
                self.q_table[current_state] = np.ones(self.env.numActions)

            for step in range(self.env.maxSteps):
                action = strategy.select_action(current_state, self.q_table, steps_done)
                next_state, reward, done, truncated = self.env.step(action)
                steps_done += 1
                if next_state not in self.q_table:
                    self.q_table[next_state] = np.ones(self.env.numActions)
                if done:
                    self.q_table[next_state] = np.zeros(self.env.numActions)

                td_target = reward + gamma * np.max(self.q_table[next_state])
                td_error = td_target - self.q_table[current_state][action]

                self.q_table[current_state][action] += alpha * td_error

                if done or truncated:
                    logs.steps_done.append(steps_done)
                    logs.steps_taken.append(self.env.step_count())
                    logs.td_error_sq.append(td_error ** 2)
                    logs.rewards.append(reward)
                    if done:
                        logs.finish_counter += 1
                        print(
                                f"Episode {episode} completed, Use {self.env.step_count()} steps, and received {reward} reward.")
                    if truncated:
                        print(f"Episode {episode} truncated after {self.env.step_count()} steps, and received {reward}")
                    break
                current_state = next_state
            live_plots.e.append(strategy.epsGreedy.epsilon)
            live_plots.s.append(self.env.step_count())
            live_plots.l.append(td_error ** 2)
            live_plots.r.append(reward)
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
        logs.save_log(f"QLearning_train.json")
        live_plots.save_and_close(f"QLearning_live_plot.json")


def eval(env, q_table, strategy, episodes):
    print("Evaluation...")
    steps_list, rewards_list = [], []
    finish_counter = 0
    for episode in range(1, episodes + 1):
        current_state = env.reset()
        for step in range(env.maxSteps):
            action = strategy.select_action(current_state, q_table)
            next_state, reward, done, truncated = env.step(action)
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


def main(env, alpha, gamma, strategy, episodes):
    q_table = QLearning(env)
    q_table.train(gamma, alpha, strategy, episodes)
    return q_table.q_table


if __name__ == '__main__':
    env = MiniGridHash()
    # Check if there is already a saved table
    if os.path.exists("q_learning_table.pkl"):
        with open('q_learning_table.pkl', 'rb') as f:
            q_table = pickle.load(f)
    else:
        # Train and save
        strategy = DecreasingEpsilon(1, 0.01, 3000)
        q_table = main(env, 0.1, 0.9, strategy, 512)
        with open('q_learning_table.pkl', 'wb') as f:
            pickle.dump(q_table, f)

    # Evaluation

    eval(env, q_table, EpsilonGreedy(0, 3), 1000)
