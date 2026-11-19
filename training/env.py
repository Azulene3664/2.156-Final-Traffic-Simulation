"""Training environment v2 for the shared AV policy.

Differences from the environment in test2.ipynb:
  * same road, inflow rule and vehicle parameters as the evaluation simulation
    (evaluation/evaluate.py), so there is no train / test mismatch;
  * the other AVs on the road are driven by a copy of the policy being trained
    (refreshed after every PPO rollout) instead of by IDM;
  * hard-braking disturbances are injected ahead of the ego vehicle during training;
  * the reward looks at the platoon behind the ego vehicle, and leaving the road is a
    time-limit truncation (value is bootstrapped), not a terminal state with a bonus.
"""
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation"))
from evaluate import DT, ROAD, V0, Vehicle, av_obs, idm_acc  # noqa: E402

N_FOLLOW = 5          # followers that enter the reward
MAX_STEPS = 250       # 125 s
BRAKE_STEPS = 11      # 5.5 s at -4 m/s^2, as in the evaluation scenario


def make_numpy_policy(w):
    def policy(obs):
        h = np.tanh(w["mlp_extractor.policy_net.0.weight"] @ obs + w["mlp_extractor.policy_net.0.bias"])
        h = np.tanh(w["mlp_extractor.policy_net.2.weight"] @ h + w["mlp_extractor.policy_net.2.bias"])
        return float(np.clip((w["action_net.weight"] @ h + w["action_net.bias"])[0], -3.0, 2.0))
    return policy


class MixedTrafficEnv(gym.Env):
    def __init__(self, residual=True, av_ratio_range=(0.1, 0.4), p_disturb=0.8, w_follow=0.5, w_acc=0.03, w_aeb=1.0, w_jam=0.5, seed=None):
        super().__init__()
        self.action_space = spaces.Box(-3.0, 2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 2.0, shape=(7,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.av_ratio_range, self.p_disturb, self.residual = av_ratio_range, p_disturb, residual
        self.w_follow, self.w_acc, self.w_aeb, self.w_jam = w_follow, w_acc, w_aeb, w_jam
        self.other_policy = None
        self.vehicles = None
        self.episodes = 0

    # called from the training callback
    def set_policy_weights(self, weights):
        self.other_policy = make_numpy_policy(weights)

    # ---------------------------------------------------------------- simulation
    def _new_sim(self):
        self.vehicles, self.timer, self.n = [], 0.0, 0
        self.av_ratio = self.rng.uniform(*self.av_ratio_range)
        self.braking = {}                      # vehicle -> remaining forced-brake steps
        for _ in range(140):                   # fill the road
            self._advance(None, None)

    def _advance(self, ego, action):
        r = self.rng
        self.timer -= DT
        if self.timer <= 0 and (not self.vehicles or self.vehicles[-1].x > 25.0):
            is_av = r.random() < self.av_ratio
            T = 1.0 if is_av else float(np.clip(r.normal(1.5, 0.3), 0.8, 2.5))
            a = 1.0 if is_av else float(np.clip(r.normal(1.0, 0.2), 0.5, 1.5))
            self.vehicles.append(Vehicle(self.n, is_av, T, a, 0.0))
            self.n += 1
            self.timer = r.uniform(0.8, 1.0)
        acc, aeb = [], False
        for i, veh in enumerate(self.vehicles):
            s, dv = 999.0, 0.0
            if i > 0:
                s, dv = self.vehicles[i - 1].x - veh.x, veh.v - self.vehicles[i - 1].v
            use_policy = veh is ego or (veh.is_av and self.other_policy is not None)
            if use_policy:
                a = float(action) if veh is ego else self.other_policy(av_obs(veh, self.vehicles))
                if self.residual:              # policy output is a correction on top of IDM
                    a = float(np.clip(idm_acc(veh, s, dv) + a, -4.0, 2.0))
                if s < 2.0 + veh.v * 0.8:
                    a = -4.0
                    aeb = aeb or veh is ego
            else:
                a = idm_acc(veh, s, dv)
            if self.braking.get(veh, 0) > 0:
                a = -4.0
                self.braking[veh] -= 1
            acc.append(a)
        for a, veh in zip(acc, self.vehicles):
            veh.v = float(np.clip(veh.v + a * DT, 0.0, veh.v0))
            veh.x += veh.v * DT
        return aeb

    def _remove_exited(self):
        self.vehicles = [v for v in self.vehicles if v.x < ROAD]
        self.braking = {v: k for v, k in self.braking.items() if k > 0 and v.x < ROAD}

    # ---------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.vehicles is None or self.episodes % 10 == 0:
            self._new_sim()
        self.episodes += 1
        while True:                            # wait for an AV in the upstream part of the road
            cands = [v for v in self.vehicles if v.is_av and v.x < 200.0]
            if cands:
                break
            self._advance(None, None)
            self._remove_exited()
        self.ego = cands[self.rng.integers(len(cands))]
        self.t = 0
        self.disturb_at = int(self.rng.integers(0, 50)) if self.rng.random() < self.p_disturb else -1
        return av_obs(self.ego, self.vehicles).astype(np.float32), {}

    def step(self, action):
        a_cmd = float(np.clip(action[0], -3.0, 2.0))
        if self.t == self.disturb_at:
            idx = self.vehicles.index(self.ego)
            ahead = self.vehicles[max(0, idx - 8):idx]
            if ahead:
                self.braking[ahead[self.rng.integers(len(ahead))]] = BRAKE_STEPS
        aeb = self._advance(self.ego, a_cmd)
        self.t += 1

        idx = self.vehicles.index(self.ego)
        followers = [v for v in self.vehicles[idx + 1: idx + 1 + N_FOLLOW] if self.ego.x - v.x < 150.0]
        v_follow = np.mean([v.v for v in followers]) if followers else self.ego.v
        group = [self.ego] + followers
        jam_frac = np.mean([v.v < 5.0 for v in group])
        reward = ((1 - self.w_follow) * self.ego.v / V0 + self.w_follow * v_follow / V0
                  - self.w_acc * a_cmd ** 2 - self.w_aeb * float(aeb) - self.w_jam * jam_frac)

        obs = av_obs(self.ego, self.vehicles).astype(np.float32)
        truncated = self.ego.x >= ROAD or self.t >= MAX_STEPS
        self._remove_exited()
        return obs, float(reward), False, bool(truncated), {}
