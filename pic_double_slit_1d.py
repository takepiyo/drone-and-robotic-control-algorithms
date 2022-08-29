import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
import my_gym_env


class DoubleSlit1DPathIntegralSampling:
    def __init__(self, env, K):
        self.env = env
        self.K = K
        self.dt = env.dt
        self.v = env.v
        self.phi = env.terminate_cost

        self.sampled_cost = []
        self.noise_history = []
        self.traj = []
        for _ in tqdm(range(K)):
            state = self.env.reset()
            done = False
            while not done:
                state, cost, done, _ = self.env.step(0)
            if self.env.n + 1 == self.env.max_n:
                self.sampled_cost.append(
                    math.exp(-self.phi(state[0]) / self.v))
                self.noise_history.append(self.env.noise_array)
            self.traj.append((self.env.n, self.env.x_array))
        if not len(self.noise_history) > 0:
            print('All sample did not get to the goal')
            return
        self.noise_history = np.stack(self.noise_history)
        # env.render_multiple_path(self.traj)

        self.psi = sum(self.sampled_cost) / K

    def command(self):
        return np.sum(self.sampled_cost * (self.noise_history[:, 1])) / (self.psi * self.K)


if __name__ == "__main__":
    env = gym.make('DoubleSlit1D-v0')
    ctrl = DoubleSlit1DPathIntegralSampling(env, 100000)

    traj = []
    for _ in tqdm(range(10)):
        state = env.reset(x=2.0)
        done = False
        while not done:
            u = ctrl.command()
            state, cost, done, _ = env.step(u)
        traj.append((env.n, env.x_array))
    env.render_multiple_path(traj)
