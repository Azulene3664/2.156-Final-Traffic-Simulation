"""Train the ring-road AV policy with PPO.

    python ring/train.py --steps 2000000 --out models/ring_ppo

Writes <out>.zip (Stable-Baselines3 model) and <out>_trainlog.csv, then prints a quick
validation on held-out seeds (seeds >= 1000; the report uses seeds 0-9 per ring length).
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor  # noqa: E402

import sim  # noqa: E402
from env import RingEnv, V_TARGET_MAX  # noqa: E402
from rollout import rollout  # noqa: E402


class Log(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path, self.rows = path, []

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if buf:
            # mean speed over the episode is roughly return * scale / decisions
            self.rows.append((self.num_timesteps, float(np.mean([e["r"] for e in buf])), float(np.mean([e["l"] for e in buf]))))
            with open(self.path, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["timesteps", "ep_rew_mean", "ep_len_mean"]); w.writerows(self.rows)


def make_env(rank, seed, spread_penalty):
    def f():
        return RingEnv(seed=seed * 1000 + rank, spread_penalty=spread_penalty)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "ring_ppo"))
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--log_std", type=float, default=-2.0)
    ap.add_argument("--spread_penalty", type=float, default=0.5)
    ap.add_argument("--resume", default=None, help="continue training from this model zip")
    ap.add_argument("--init_target", type=float, default=2.5, help="initial target speed of the AV in m/s")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    venv = VecMonitor(SubprocVecEnv([make_env(i, args.seed, args.spread_penalty) for i in range(args.envs)]))
    model = PPO("MlpPolicy", venv, learning_rate=lambda frac: args.lr * max(frac, 0.1), n_steps=1200, batch_size=1200, n_epochs=10,
                gamma=args.gamma, gae_lambda=0.97, ent_coef=0.0,
                policy_kwargs=dict(net_arch=[32, 32], log_std_init=args.log_std), seed=args.seed, verbose=0)
    # start from a cautious policy: initial target speed below the equilibrium speed of every ring length
    import torch
    with torch.no_grad():
        model.policy.action_net.bias.fill_(2.0 * args.init_target / V_TARGET_MAX - 1.0)
    if args.resume:
        model.set_parameters(args.resume)
    log = Log(args.out + "_trainlog.csv")
    model.learn(total_timesteps=args.steps, callback=log)
    model.save(args.out)
    venv.close()
    print(f"trained {log.rows[-1][0]} decisions | mean episode return {log.rows[0][1]:.0f} -> {log.rows[-1][1]:.0f} "
          f"(episode length {log.rows[0][2]:.0f} -> {log.rows[-1][2]:.0f}) | saved {args.out}.zip")

    policy = lambda obs: model.predict(obs, deterministic=True)[0][0]
    print("validation (seeds 1000-1004): L | equilibrium | human-only | with AV   [mean speed, last 100 s]   | speed std human / AV | collisions")
    for L in (220, 230, 250, 260, 270):
        hum = [rollout(L, 1000 + s, None) for s in range(5)]
        av = [rollout(L, 1000 + s, policy) for s in range(5)]
        m = lambda rs, k: np.mean([r[k] for r in rs])
        print(f"  {L} | {sim.equilibrium_speed(L):.2f} | {m(hum, 'mean_speed'):.2f} | {m(av, 'mean_speed'):.2f} | "
              f"{m(hum, 'speed_std'):.2f} / {m(av, 'speed_std'):.2f} | {sum(r['collided'] for r in av)}")


if __name__ == "__main__":
    main()
