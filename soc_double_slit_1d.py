import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
import my_gym_env


class DoubleSlit1DAnalytical:
    def __init__(self, env):
        self.env = env
        self.tf = env.T
        self.slit_t = env.slit_t
        self.dt = env.dt
        self.v = env.v
        self.R = env.R

        self.slit1 = env.slit1
        self.slit2 = env.slit2
        self.x_min = env.x_min
        self.x_max = env.x_max

        self.sigma = lambda t: math.sqrt(self.v * (self.tf - t))
        self.sigma_1 = lambda t: math.sqrt(self.sigma(
            t) ** 2 * self.v * self.R / (self.sigma(t) ** 2 + self.v * self.R))
        self.A = lambda t: 1 / (self.slit_t - t) + 1 / \
            (self.R + self.tf - self.slit_t)
        self.B = lambda x, t: x / (self.slit_t - t)
        self.F = lambda x_0, x, t: math.erf(
            math.sqrt(self.A(t) / (2 * self.v)) * (x_0 - (self.B(x, t) / self.A(t))))
        self.P = lambda x, t: self.F(
            self.slit1[1], x, t) - self.F(self.slit1[0], x, t) + self.F(self.slit2[1], x, t) - self.F(self.slit2[0], x, t)
        self.J = lambda x, t: self.v * self.R * math.log(self.sigma(t) / self.sigma_1(t)) + 0.5 * (
            self.sigma_1(t) * x / self.sigma(t)) ** 2 - self.v * self.R * math.log(0.5 * (self.P(x, t)) + 1e-5) if t < self.slit_t else self.v * self.R * math.log(self.sigma(t) / self.sigma_1(t)) + 0.5 * (
            self.sigma_1(t) * x / self.sigma(t)) ** 2
        self.partial_xF = lambda x_0, x, t: 2 / math.sqrt(math.pi) * math.exp(-math.sqrt(
            self.A(t) / (2 * self.v)) * (x_0 - self.B(x, t) / self.A(t)) ** 2)
        self.partial_xP = lambda x, t: self.partial_xF(self.slit1[1], x, t) - self.partial_xF(
            self.slit1[0], x, t) + self.partial_xF(self.slit2[1], x, t) - self.partial_xF(self.slit2[0], x, t)

        self.optimal_u = lambda x, t: - (self.v * x) / (self.R + self.tf - t) - (self.partial_xP(x, t) / self.P(x, t)) * (self.v / (
            math.sqrt(2 * self.v * self.A(t)) * (self.slit_t - t))) if t < self.slit_t else - (x) / (self.R + (self.tf - t))

    def command(self, x, t):
        return self.optimal_u(x, t)

    def draw_J(self):
        fig, ax = plt.subplots()
        x = np.arange(self.x_min, self.x_max, 0.01)
        for t in [0.0, self.slit_t - self.dt, self.slit_t + self.dt, self.tf - self.dt]:
            ax.plot(x, list(map(lambda x: self.J(x, t), x)),
                    label="t={}".format(t))
        ax.legend()
        plt.show()


if __name__ == "__main__":
    env = gym.make('DoubleSlit1D-v0')
    ctrl = DoubleSlit1DAnalytical(env)

    traj = []
    for _ in tqdm(range(10)):
        state = env.reset(x=0.0)
        done = False
        while not done:
            u = ctrl.command(state[0], state[1])
            state, cost, done, _ = env.step(u)
        traj.append((env.n, env.x_array))
    env.render_multiple_path(traj)
    ctrl.draw_J()
