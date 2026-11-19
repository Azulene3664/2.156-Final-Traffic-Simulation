"""Gymnasium environment: one RL-controlled AV among 21 human drivers on a ring road.

Observation (local, 5 numbers): own speed, speed difference to the leader, gap to the leader,
and the running averages (30 s) of the AV's own speed and gap.
Action: a target speed between 0 and 10 m/s; a proportional controller turns it into an
acceleration, which is passed through a safety filter.
Reward: mean speed of ALL vehicles minus a small acceleration penalty (as in Wu et al.), divided by
the uniform-flow speed of the current ring so that all ring lengths weigh equally, minus a penalty
on the spread of speeds across vehicles (a direct measure of the wave).
Each episode: random ring length, a warm-up in which the AV drives like a human so that
stop-and-go waves form (they need about 350 s to develop fully), then 300 s of control.
"""
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim  # noqa: E402

CTRL_EVERY = 5                 # one decision every 0.5 s
WARMUP_S, HORIZON_S = 350.0, 300.0
WARMUP_TRAIN = (50.0, 400.0)    # training episodes start from partly and fully developed waves
V_TARGET_MAX, K_SPEED = 10.0, 1.0
V_SCALE, H_SCALE = 10.0, 20.0
ACC_PENALTY = 0.1
SPREAD_PENALTY = 0.5


def observe(s):
    i = s.av
    h = s.gaps()[i]
    v_lead = s.v[(i + 1) % sim.N_VEH]
    return np.array([s.v[i] / V_SCALE, (v_lead - s.v[i]) / V_SCALE, min(h, 2 * H_SCALE) / H_SCALE,
                     s.avg_v / V_SCALE, min(s.avg_h, 2 * H_SCALE) / H_SCALE], dtype=np.float32)


def to_acc(action, v):
    """Policy output in [-1, 1] -> target speed in [0, V_TARGET_MAX] -> acceleration command."""
    v_target = (float(np.clip(action, -1.0, 1.0)) + 1.0) / 2.0 * V_TARGET_MAX
    return float(np.clip(K_SPEED * (v_target - v), sim.AV_ACC_MIN, sim.AV_ACC_MAX))


class RingEnv(gym.Env):
    def __init__(self, length_range=(220.0, 270.0), seed=None, spread_penalty=SPREAD_PENALTY):
        super().__init__()
        self.length_range, self.spread_penalty = length_range, spread_penalty
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(5,), dtype=np.float32)
        self.n_decisions = int(HORIZON_S / (sim.DT * CTRL_EVERY))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        L = self.rng.uniform(*self.length_range)
        self.sim = sim.RingSim(L, self.rng, av_index=0)
        self.v_eq = sim.equilibrium_speed(L)
        for _ in range(int(self.rng.uniform(*WARMUP_TRAIN) / sim.DT)):
            self.sim.step(None)
        self.k = 0
        return observe(self.sim), {}

    def step(self, action):
        speed, spread, cmd = 0.0, 0.0, 0.0
        for _ in range(CTRL_EVERY):
            cmd = to_acc(action[0], self.sim.v[self.sim.av])
            self.sim.step(cmd, tau=sim.DT * CTRL_EVERY)
            speed += self.sim.v.mean()
            spread += self.sim.v.std()
        self.k += 1
        reward = (speed / CTRL_EVERY - ACC_PENALTY * abs(cmd) - self.spread_penalty * spread / CTRL_EVERY) / self.v_eq
        terminated = self.sim.collided
        if terminated:
            reward = 0.0
        return observe(self.sim), float(reward), terminated, self.k >= self.n_decisions, {}
