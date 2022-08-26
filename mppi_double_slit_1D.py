import math
import argparse
from re import L

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
        self.normalize = lambda x: (x - np.mean(x)) / np.std(x)
        self.cost_alpha = lambda x: (self.normalize(
            x) - np.min(self.normalize(x))) / (np.max(self.normalize(x)) - np.min(self.normalize(x)))

        self.u = np.zeros((self.horizon, 1))

    def command(self, x, t, x_array):
        x_0 = x
        t_0 = t
        sequence_cost = np.zeros((self.K, 1))
        epsilon = self.sigma * np.random.randn(self.K, self.horizon)
        traj = []
        for k in range(self.K):
            x, _, cost, done = self.env.reset(
                x=x_0, n=int(t_0 / self.dt), x_array=np.copy(x_array))
            for h in range(self.horizon):
                x, _, cost, done = self.env.step(self.u[h] + epsilon[k, h])
                sequence_cost[k] += cost + np.sqrt(np.square(x)) + \
                    self.l * self.u[h] * epsilon[k, h] / self.sigma

                if done or h == self.horizon - 1:
                    sequence_cost[k] += env.phi(x) if env.n + \
                        1 == env.max_n else 0
                    traj.append((self.env.n, self.env.x_array))
                    break
        if self.iteration % self.plot_per_iteration == 0:
            self.env.render_multiple_path(
                traj, self.cost_alpha(-sequence_cost))
        beta = np.min(sequence_cost)
        importance_weight = np.exp(-(sequence_cost - beta) / self.l)
        importance_weight /= np.sum(importance_weight)
        self.u += epsilon.T.dot(importance_weight)
        u_0 = self.u[0]
        self.u = np.roll(self.u, -1)
        self.u[-1] = 0

        self.iteration += 1

        return u_0[0]


class DoubleSlit1DAnalytical:
    def __init__(self, env):
        self.env = env
        self.tf = env.T
        self.slit_t = env.slit_t
        self.dt = env.dt
        self.v = env.v
        self.R = env.R
        self.alpha = env.alpha

        self.slit1 = env.slit1
        self.slit2 = env.slit2
        self.x_min = env.x_min
        self.x_max = env.x_max

        self.sigma = lambda t: math.sqrt(self.v * (self.tf - t))
        self.sigma_1 = lambda t: math.sqrt(self.sigma(
            t) ** 2 * self.v * self.R / (self.alpha * self.sigma(t) ** 2 + self.v * self.R))
        self.A = lambda t: 1 / (self.slit_t - t) + 1 / \
            (self.R + self.tf - self.slit_t)
        self.B = lambda x, t: x / (self.slit_t - t)
        self.F = lambda x_0, x, t: math.erf(
            math.sqrt(self.A(t) / (2 * self.v)) * (x_0 - (self.B(x, t) / self.A(t))))
        self.P = lambda x, t: self.F(
            self.slit1[1], x, t) - self.F(self.slit1[0], x, t) + self.F(self.slit2[1], x, t) - self.F(self.slit2[0], x, t)
        self.J = lambda x, t: self.v * self.R * math.log(self.sigma(t) / self.sigma_1(t)) + 0.5 * (
            self.sigma_1(t) * x / self.sigma(t)) ** 2 - self.v * self.R * math.log(0.5 * (self.P(x, t)) + 1e-5) if t < self.slit_t else self.v * self.R * math.log(self.sigma(t) / self.sigma_1(t)) + 0.5 * (
            self.sigma_1(t) * x / self.sigma(t)) ** 2 * self.alpha
        self.partial_xF = lambda x_0, x, t: 2 / math.sqrt(math.pi) * math.exp(-math.sqrt(
            self.A(t) / (2 * self.v)) * (x_0 - self.B(x, t) / self.A(t)) ** 2)
        self.partial_xP = lambda x, t: self.partial_xF(self.slit1[1], x, t) - self.partial_xF(
            self.slit1[0], x, t) + self.partial_xF(self.slit2[1], x, t) - self.partial_xF(self.slit2[0], x, t)

        self.optimal_u = lambda x, t: - (self.v * x) / (self.R + self.tf - t) - (self.partial_xP(x, t) / self.P(x, t)) * (self.v / (
            math.sqrt(2 * self.v * self.A(t)) * (self.slit_t - t))) if t < self.slit_t else - (self.alpha * x) / (self.R + self.alpha * (self.tf - t))

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


class DoubleSlit1DPathIntegralSampling:
    def __init__(self, env, K):
        self.env = env
        self.K = K
        self.dt = env.dt
        self.v = env.v
        self.phi = env.phi

        self.sampled_cost = []
        self.noise_history = []
        self.traj = []
        for _ in tqdm(range(K)):
            x, t, _, done = env.reset()
            while not done:
                x, t, _, done = env.step(u=0)
            if env.n + 1 == env.max_n:
                self.sampled_cost.append(math.exp(-self.phi(x) / self.v))
                self.noise_history.append(env.noise_array)
            self.traj.append((env.n, env.x_array))
        if not len(self.noise_history) > 0:
            print('All sample did not get to the goal')
            return
        self.noise_history = np.stack(self.noise_history)
        # env.render_multiple_path(self.traj)

        self.psi = sum(self.sampled_cost) / K

    def command(self, x, t):
        return np.sum(self.sampled_cost * (self.noise_history[:, 1])) / (self.psi * self.K)


class DoubleSlit1D:
    def __init__(self):
        self.T = 2.0
        self.dt = 0.02
        self.slit_t = 1.0
        self.slit_n = int(self.slit_t / self.dt)
        self.max_n = int(self.T / self.dt)

        self.x_min = -10.0
        self.x_max = 10.0
        self.slit1 = [-6.0, -4.0]
        self.slit2 = [4.0, 6.0]
        self.V = lambda x, n: 1000000 if n == self.slit_n and (not ((
            self.slit1[0] < x < self.slit1[1]) or (self.slit2[0] < x < self.slit2[1]))) else 0
        self.alpha = 1.0
        self.phi = lambda x: 0.5 * self.alpha * x ** 2
        self.v = 1.0
        self.R = 0.1

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

        instantaneous_cost = self.V(self.x, self.n)
        done = (self.n + 1 == self.max_n or instantaneous_cost > 0 or not (
            self.x_min < self.x < self.x_max))
        return self.x, self.t_array[self.n], instantaneous_cost, done

    def step(self, u):
        self.n += 1
        noise = np.sqrt(self.dt) * np.random.randn()
        self.noise_array[self.n] = noise
        self.x += u * self.dt + noise
        self.x_array[self.n] = self.x

        instantaneous_cost = self.V(self.x, self.n)
        done = (self.n + 1 == self.max_n or instantaneous_cost > 0 or not (
            self.x_min < self.x < self.x_max))
        return self.x, self.t_array[self.n], instantaneous_cost, done

    def render(self):
        figure, ax = plt.subplots()
        ax.set_ylim(self.x_min, self.x_max)
        ax.set_xlim(0, self.T)
        ax.set_aspect(0.08)
        ax.plot([self.slit_t, self.slit_t],
                [self.x_min, self.slit1[0]], color="black")
        ax.plot([self.slit_t, self.slit_t], [
            self.slit1[1], self.slit2[0]], color="black")
        ax.plot([self.slit_t, self.slit_t],
                [self.slit2[1], self.x_max], color="black")

        ax.plot(self.t_array[: self.n + 1], self.x_array[: self.n + 1])
        plt.show()

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
                        # x_array[: n + 1], alpha=cost[i][0])
                        x_array[: n + 1], alpha=1.0, color="b")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Path Integral Control Example(Double Slit 1D Environment)")
    parser.add_argument('--method', choices=[
                        'Analytical', 'Sampling', 'MPPI'], default='Analytical')
    parser.add_argument('-n', default=1000, type=int)
    parser.add_argument('--n_sampling', default=100000, type=int)
    args = parser.parse_args()
    env = DoubleSlit1D()
    if args.method == 'Analytical':
        ctrl = DoubleSlit1DAnalytical(DoubleSlit1D())
    elif args.method == 'Sampling':
        ctrl = DoubleSlit1DPathIntegralSampling(
            DoubleSlit1D(), args.n_sampling)
    elif args.method == 'MPPI':
        ctrl = DoubleSlit1DMPPI(DoubleSlit1D(
        ), args.n_sampling, horizon=50, sigma=4.0, l=1, plot_per_iteration=40)

    traj = []
    for _ in tqdm(range(args.n)):
        x, t, cost, done = env.reset(x=0.0)
        while not done:
            u = ctrl.command(x, t, env.x_array)
            x, t, cost, done = env.step(u=u)
        traj.append((env.n, env.x_array))
    env.render_multiple_path(traj)
    # ctrl.draw_J()
