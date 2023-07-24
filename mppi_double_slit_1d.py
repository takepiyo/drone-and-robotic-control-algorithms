import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
import my_gym_env


class DoubleSlit1DMPPI:
    def __init__(self, env, K=1000, horizon=50, sigma=0.4, l=1.0, plot_per_iteration=50):
        self.env = env
        self.K = K
        self.horizon = horizon
        self.sigma = sigma
        self.dt = env.dt
        self.v = env.v
        self.l = l

        self.plot_per_iteration = plot_per_iteration
        self.iteration = 0
        # self.normalize = lambda x: (x - np.mean(x)) / np.std(x)
        self.normalize = lambda x: x
        self.cost_alpha = lambda x: (self.normalize(x) - np.min(self.normalize(x))) / (
            np.max(self.normalize(x)) - np.min(self.normalize(x))
        )

        self.u = np.zeros((self.horizon, 1))

    def command(self, x, t, x_array):
        x_0 = x
        t_0 = t
        sequence_cost = np.zeros((self.K, 1))
        epsilon = self.sigma * np.random.randn(self.K, self.horizon)
        traj = []
        for k in range(self.K):
            state = self.env.reset(x=x_0, n=int(t_0 / self.dt), x_array=np.copy(x_array))
            for h in range(self.horizon):
                state, cost, done, _ = self.env.step(self.u[h][0] + epsilon[k, h])
                x = state[0]
                sequence_cost[k] += cost + np.sqrt(np.square(x)) + self.l * self.u[h][0] * epsilon[k, h] / self.sigma

                if done or h == self.horizon - 1:
                    sequence_cost[k] += self.env.terminate_cost(x) if self.env.n + 1 == self.env.max_n else 0
                    traj.append((self.env.n, self.env.x_array))
                    break
        if self.iteration % self.plot_per_iteration == 0:
            self.env.render_multiple_path(traj, cost=self.cost_alpha(-sequence_cost))
        beta = np.min(sequence_cost)
        importance_weight = np.exp(-(sequence_cost - beta) / self.l)
        importance_weight /= np.sum(importance_weight)
        self.u += epsilon.T.dot(importance_weight)
        u_0 = self.u[0]
        self.u = np.roll(self.u, -1)
        self.u[-1] = 0

        self.iteration += 1

        return u_0[0]


if __name__ == "__main__":
    env = gym.make("DoubleSlit1D-v0")
    sim_env = gym.make("DoubleSlit1D-v0")
    ctrl = DoubleSlit1DMPPI(sim_env, 1000, horizon=50, sigma=4.0, l=1, plot_per_iteration=40)

    traj = []
    for _ in tqdm(range(1)):
        state = env.reset(x=0.0)
        done = False
        while not done:
            u = ctrl.command(state[0], state[1], env.x_array)
            state, cost, done, _ = env.step(u)
        traj.append((env.n, env.x_array))
    env.render_multiple_path(traj)
