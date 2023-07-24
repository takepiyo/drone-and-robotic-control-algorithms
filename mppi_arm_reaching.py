import os
from tqdm import tqdm

import gym
import mujoco_py
import numpy as np
import matplotlib.pyplot as plt


class ReacherMPPI:
    def __init__(self, env, K=100, horizon=50, sigma=0.4, l=1.0, plot_per_iteration=50):
        self.sim_env = env
        self.sim_env.reset()
        self.K = K
        self.horizon = horizon
        self.sigma = sigma
        self.dt = env.dt
        self.l = l

        self.plot_per_iteration = plot_per_iteration
        self.iteration = 0
        # self.normalize = lambda x: (x - np.mean(x)) / np.std(x)
        self.normalize = lambda x: x
        self.cost_alpha = lambda x: (self.normalize(x) - np.min(self.normalize(x))) / (
            np.max(self.normalize(x)) - np.min(self.normalize(x))
        )

        self.action_dim = self.sim_env.action_space.shape[0]
        self.u = np.zeros((self.horizon, self.action_dim))

    def command(self, env):
        sequence_cost = np.zeros(self.K)
        epsilon = self.sigma * np.random.randn(self.K, self.horizon, self.action_dim)
        traj = []
        for k in range(self.K):
            # for k in tqdm(range(self.K)):
            obs = self.sim_env.set_state(env.data.qpos, env.data.qvel)
            for h in range(self.horizon):
                obs, _, _, info = self.sim_env.step(self.u[h] + epsilon[k, h])
                reward = 10 * info["reward_dist"] + info["reward_ctrl"]
                sequence_cost[k] += -reward + self.l * self.u[h].dot(epsilon[k, h]) / self.sigma

                # if h == self.horizon - 1:
                #     traj.append((self.sim_env.n, self.sim_env.x_array))
                #     break
        # if self.iteration % self.plot_per_iteration == 0:
        #     self.sim_env.render_multiple_path(
        #         traj, cost=self.cost_alpha(-sequence_cost))
        beta = np.min(sequence_cost)
        importance_weight = np.exp(-(sequence_cost - beta) / self.l)
        importance_weight /= np.sum(importance_weight)
        self.u += np.sum(epsilon * np.tile(importance_weight, (self.action_dim, self.horizon, 1)).T, axis=0)
        # self.u += epsilon.T.dot(importance_weight)
        u_0 = self.u[0]
        self.u = np.roll(self.u, -1, axis=0)
        self.u[-1] = 0

        self.iteration += 1

        return u_0


env_name = "Reacher-v2"
env = gym.make(env_name)
sim_env = gym.make(env_name)
ctrl = ReacherMPPI(sim_env, K=500, horizon=10, sigma=0.3)
obs = env.reset()
env.render()
while True:
    u = ctrl.command(env)
    obs, reward, _, info = env.step(u)
    print(f"{info=}")
    # obs, reward, _, _ = env.step(env.action_space.sample())
    env.render()

# mj_path = mujoco_py.utils.discover_mujoco()
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
# model = mujoco_py.load_model_from_path(xml_path)
# sim = mujoco_py.MjSim(model)

# print(sim.data.qpos)
