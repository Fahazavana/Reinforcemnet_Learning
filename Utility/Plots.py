import json
from time import sleep

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class Logs:
    def __init__(self):
        self.steps_done = []
        self.rewards = []
        self.epsilons = []
        self.steps_taken = []
        self.td_error_sq = []
        self.finish_counter = 0

    def save_log(self, path):
        with open(path, "w") as f:
            json.dump({
                    'steps_done'    : self.steps_done,
                    'td_error_sq'   : self.td_error_sq,
                    'rewards'       : self.rewards,
                    'epsilons'      : self.epsilons,
                    'steps_taken'   : self.steps_taken,
                    'finish_counter': self.finish_counter

            }, f)


class LivePlots:

    def __init__(self):
        plt.ion()
        mpl.use("QtAgg")
        self.l = []
        self.r = []
        self.e = []
        self.s = []
        self.fig, (self.al, self.ar) = plt.subplots(2, 1, figsize=(6, 8))
        self.al.grid(True)
        self.ar.grid(True)
        self.lline, = self.al.plot([], [], label="Loss")
        # self.sline, = self.ar.plot([], [], label="Steps Taken")
        self.rline, = self.ar.plot([], [], label="Reward")
        self.eline, = self.al.plot([], [], label="Epsilon")
        self.al.set_ylim(0, 1.1)
        self.ar.set_ylim(0, 1.1)
        self.al.set_xlim(0, 10)
        self.ar.set_xlim(0, 10)
        self.al.set_title("Cumulative Averaged Loss,and\n Start Epsilon per Episode")
        self.ar.set_title("Cumulative Averaged Rewards")
        self.al.set_xlabel("Episode")
        self.ar.set_xlabel("Episode")
        self.al.set_ylabel("Loss  and Epsilon")
        self.ar.set_ylabel("Reward")
        self.al.legend(loc='upper right')
        self.ar.legend(loc='upper right')
        plt.tight_layout()

    def update_plot(self):
        if len(self.l) == 0:
            return

        max_x = len(self.l)
        if max_x > self.al.get_xlim()[1]:
            self.al.set_xlim(0, max_x + 10)
            self.ar.set_xlim(0, max_x + 10)

        T = np.arange(1, max_x + 1)
        r = np.cumsum(self.r) / T
        l = np.cumsum(self.l) / T
        # s = np.cumsum(self.s) / T

        self.lline.set_data(T, l)
        self.rline.set_data(T, r)
        # self.sline.set_data(T, s)
        self.eline.set_data(T, self.e)

        self.al.figure.canvas.draw()
        self.ar.figure.canvas.draw()
        plt.pause(0.001)

    def save_and_close(self, path):
        with open(path, "w") as f:
            json.dump({
                    "episode"    : list(range(1, len(self.l) + 1)),
                    "epsilons"   : self.e,
                    "rewards"    : self.r,
                    "losses"     : self.l,
                    "steps_taken": self.s,
            }, f)
        plt.ioff()
        plt.close(self.fig)


def smooth(scalars, weight: float):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_live_data(x, y, title, epsilon=None, avg=True):
    x = np.array(x)
    y = np.array(y)
    smoothed_y = smooth(y, 0.7)
    acc_avg = np.cumsum(y) / x

    if epsilon is not None:
        plt.plot(x, epsilon, label='Epsilon', linestyle='--', linewidth=1)
    plt.plot(x, y, 'b--', alpha=0.25, label=title['raw'], linewidth=1)
    plt.plot(x, smoothed_y, 'c', label=title['smooth'])
    if avg:
        plt.plot(x, acc_avg, label=title['acc'])
    plt.xlabel(title['xlab'])
    plt.ylabel(title['ylab'])
    plt.title(title['title'])
    plt.legend()
    plt.grid(True)


def plot_train_data(x, y, title, steps_taken=None, avg=True):
    x = np.array(x)
    y = np.array(y)
    smoothed_y = smooth(y, 0.7)
    acc_avg = np.cumsum(y) / x

    if steps_taken is not None:
        s = np.array(steps_taken)
        t = np.arange(1, len(s) + 1)
        plt.plot(t, np.cumsum(s) / t, label='Averaged Steps', linewidth=1)
    plt.plot(x, y, 'b--', alpha=0.25, label=title['raw'], linewidth=1)
    plt.plot(x, smoothed_y, 'c', label=title['smooth'])
    if avg:
        plt.plot(x, acc_avg, label=title['acc'])
    plt.xlabel(title['xlab'])
    plt.ylabel(title['ylab'])
    plt.title(title['title'])
    plt.legend()
    plt.grid(True)


def plot_steps_taken(x, y, title=''):
    x = np.array(x)
    y = np.array(y)
    smoothed_y = smooth(y, 0.7)
    s = np.cumsum(y) / x
    plt.plot(x, y, '--', alpha=0.25, linewidth=1, label="Raw Steps")
    plt.plot(x, smoothed_y, label='Smoothed Steps')
    plt.plot(x, s, 'c', label='Accumulated Averaged Steps')
    plt.legend(loc='best')
    plt.xlabel("Episode")
    plt.grid(True)
    plt.ylabel("Steps")
    plt.title(f"{title}: Steps taken per Episode")


if __name__ == "__main__":
    plotter = LivePlots()
    # Simulating data update
    for i in range(100):
        plotter.l.append(np.random.rand())
        plotter.r.append(np.random.rand())
        plotter.e.append(np.random.rand())
        if i % 10 == 0:
            plotter.update_plot()
            plt.pause(0.01)
        sleep(0.1)
