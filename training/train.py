"""Train the shared AV policy (v2) with PPO and validate it on held-out seeds.

    python training/train.py --steps 400000 --out models/ppo_v2 --tag run1

Validation uses seeds 100-129 of the evaluation scenario (evaluation/evaluate.py);
seeds 0-29 are kept for the final numbers in the report.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor  # noqa: E402

import evaluate as ev  # noqa: E402
from env import MixedTrafficEnv  # noqa: E402


def weights_of(model):
    return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.policy.state_dict().items()
            if k.startswith("mlp_extractor.policy_net") or k.startswith("action_net")}


class SelfPlayAndLog(BaseCallback):
    """After every rollout: push the new weights to the simulated AVs and log the return."""
    def __init__(self, log_path):
        super().__init__()
        self.log_path, self.rows = log_path, []

    def _on_rollout_start(self):
        self.training_env.env_method("set_policy_weights", weights_of(self.model))

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if buf:
            self.rows.append((self.num_timesteps, float(np.mean([e["r"] for e in buf])), float(np.mean([e["l"] for e in buf]))))
            with open(self.log_path, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["timesteps", "ep_rew_mean", "ep_len_mean"]); w.writerows(self.rows)


def make_env(rank, seed, kwargs):
    def f():
        return MixedTrafficEnv(seed=seed * 1000 + rank, **kwargs)
    return f


def validate(model_zip, seeds, rl_mode):
    """Held-out seeds, disturbance applied to one fixed vehicle (fixed_target=True)."""
    policy = ev.load_policy(model_zip)
    conds = [("HV only", "idm", 0.0), ("IDM T=1.0", "idm", 0.2), ("PPO", rl_mode, 0.2)]
    res = {n: [ev.run(m, r, s, policy, fixed_target=True)[0] for s in seeds] for n, m, r in conds}
    g = lambda n, k: np.array([x[k] for x in res[n]], dtype=float)
    print(f"validation seeds {seeds[0]}-{seeds[-1]} (mean ± std)")
    for n in res:
        print(f"  {n:10s} exited {g(n, 'exited').mean():6.1f} | follower jam {g(n, 'jam_followers').mean():5.1f}±{g(n, 'jam_followers').std():4.1f}"
              f" (max {g(n, 'jam_followers').max():5.1f}) | vehicles slowed {g(n, 'n_slowed').mean():4.2f} | delay {g(n, 'delay').mean():6.1f}")
    jb, jr = g("HV only", "jam_followers"), g("PPO", "jam_followers")
    eb, er = g("HV only", "exited"), g("PPO", "exited")
    print(f"  PPO vs HV only: jam smaller in {(jr < jb).sum()}/{len(jb)}, larger in {(jr > jb).sum()} | exited more in {(er > eb).sum()}, fewer in {(er < eb).sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "ppo_v2"))
    ap.add_argument("--w_follow", type=float, default=0.5)
    ap.add_argument("--w_acc", type=float, default=0.03)
    ap.add_argument("--w_aeb", type=float, default=1.0)
    ap.add_argument("--w_jam", type=float, default=0.5)
    ap.add_argument("--p_disturb", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--val_seeds", type=int, default=30)
    ap.add_argument("--direct", action="store_true", help="policy outputs the acceleration itself instead of a correction to IDM")
    ap.add_argument("--log_std", type=float, default=-1.2)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    kwargs = dict(residual=not args.direct, w_follow=args.w_follow, w_acc=args.w_acc, w_aeb=args.w_aeb, w_jam=args.w_jam, p_disturb=args.p_disturb)
    venv = VecMonitor(SubprocVecEnv([make_env(i, args.seed, kwargs) for i in range(args.envs)]))
    model = PPO("MlpPolicy", venv, learning_rate=args.lr, n_steps=1024, batch_size=512, n_epochs=10, gamma=args.gamma,
                gae_lambda=0.95, ent_coef=0.0, policy_kwargs=dict(log_std_init=args.log_std), seed=args.seed, verbose=0)
    cb = SelfPlayAndLog(args.out + "_trainlog.csv")
    model.learn(total_timesteps=args.steps, callback=cb)
    model.save(args.out)
    venv.close()
    first, last = cb.rows[0], cb.rows[-1]
    print(f"trained {last[0]} steps | mean episode return {first[1]:.1f} -> {last[1]:.1f} | saved {args.out}.zip")
    validate(args.out + ".zip", list(range(100, 100 + args.val_seeds)), "rl" if args.direct else "rl_res")


if __name__ == "__main__":
    main()
