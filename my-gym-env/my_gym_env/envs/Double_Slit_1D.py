import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import math


class DoubleSlit1D(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    T = 2.0
    dt = 0.02
    slit_t = 1.0
    slit_n = int(slit_t / dt)
    max_n = int(T / dt)

    x_min = -10.0
    x_max = 10.0
    slit1 = [-6.0, -4.0]
    # slit1 = [-9.5, -0.5]
    slit2 = [4.0, 6.0]
    # slit2 = [0.5, 9.5]

    def __init__(self):
        super(DoubleSlit1D, self).__init__()

        self.action_space = spaces.Box(-15, 15, (1, ), dtype=np.float32)
        self.observation_space = spaces.Box(-float('inf'),
                                            float('inf'), (2, ), dtype=np.float32)

        self.v = 1.0
        self.R = 0.1

        self.seed()

    def _stage_cost(self, x, n):
        return 1e6 if n == self.slit_n and (not ((
            self.slit1[0] < x < self.slit1[1]) or (self.slit2[0] < x < self.slit2[1]))) else 0

    def terminate_cost(self, x):
        return 0.5 * x ** 2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, x=0.0, n=0, x_array=None):
        self.x = x
        self.wiener = 0.0
        self.n = n

        self.t_array = np.arange(0, self.T, self.dt)
        self.t = self.t_array[self.n]
        if x_array is not None:
            self.x_array = x_array
        else:
            self.x_array = np.zeros_like(self.t_array)
            self.x_array[self.n] = self.x
        self.noise_array = np.zeros_like(self.t_array)

        return np.array([self.x, self.t_array[self.n]])

    def step(self, u):
        self.n += 1
        noise = self.v * np.sqrt(self.dt) * np.random.randn()
        self.noise_array[self.n] = noise
        self.x += u * self.dt + noise
        self.x_array[self.n] = self.x

        stage_cost = self._stage_cost(self.x, self.n)
        done = (self.n + 1 == self.max_n or stage_cost > 1e5 or not (
            self.x_min < self.x < self.x_max))
        return np.array([self.x, self.t_array[self.n]]), stage_cost, done, {}

    def render(self, mode='rgb_array'):
        self.fig, self.ax = plt.subplots()
        self.ax.set_ylim(self.x_min, self.x_max)
        self.ax.set_xlim(0, self.T)
        self.ax.set_aspect(0.08)
        self.ax.plot([self.slit_t, self.slit_t],
                     [self.x_min, self.slit1[0]], color="black")
        self.ax.plot([self.slit_t, self.slit_t], [
            self.slit1[1], self.slit2[0]], color="black")
        self.ax.plot([self.slit_t, self.slit_t],
                     [self.slit2[1], self.x_max], color="black")

        self.ax.plot(self.t_array[: self.n + 1], self.x_array[: self.n + 1])

        rgb_array = self.fig2array()[:, :, :3]
        return rgb_array

    # matplotlibの画像データをnumpyに変換 ref: https://agirobots.com/openai-gym-custom-env/
    def fig2array(self):
        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.fromstring(self.fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        return buf

    def render_multiple_path(self, traj, cost=None):
        fig, ax = plt.subplots()
        ax.set_ylim(self.x_min, self.x_max)
        ax.set_xlim(0, self.T)
        ax.set_aspect(0.08)
        ax.plot([self.slit_t, self.slit_t],
                [self.x_min, self.slit1[0]], color="black")
        ax.plot([self.slit_t, self.slit_t], [
            self.slit1[1], self.slit2[0]], color="black")
        ax.plot([self.slit_t, self.slit_t],
                [self.slit2[1], self.x_max], color="black")

        for i, (n, x_array) in enumerate(traj):
            if cost is None:
                ax.plot(self.t_array[: n + 1], x_array[: n + 1])
            else:
                ax.plot(self.t_array[: n + 1],
                        x_array[: n + 1], alpha=max(0.1, cost[i][0]), color="b")
        plt.show()
