"""Run one evaluation episode on the ring and return metrics (and optionally trajectories)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim  # noqa: E402
from env import CTRL_EVERY, HORIZON_S, WARMUP_S, observe, to_acc  # noqa: E402


def rollout(ring_length, seed, policy=None, record=False, last_s=100.0, warmup_s=WARMUP_S):
    """policy: function obs -> action in [-1, 1], or None for 22 human drivers.
    The random stream is identical with and without the AV until control starts."""
    s = sim.RingSim(ring_length, np.random.default_rng(seed), av_index=0)
    n_warm, n_ctrl = int(warmup_s / sim.DT), int(HORIZON_S / sim.DT)
    T, X, V = [], [], []
    action = None
    for k in range(n_warm + n_ctrl):
        if k >= n_warm and policy is not None and (k - n_warm) % CTRL_EVERY == 0:
            action = policy(observe(s))
        cmd = to_acc(action, s.v[s.av]) if action is not None else None
        s.step(cmd, tau=sim.DT * CTRL_EVERY)
        if record or k >= n_warm + n_ctrl - int(last_s / sim.DT):
            T.append(s.t); X.append(s.x.copy()); V.append(s.v.copy())
        if s.collided:
            break
    Vl = np.array(V)[-int(last_s / sim.DT):]
    out = dict(
        ring_length=ring_length, seed=seed, collided=bool(s.collided),
        mean_speed=float(Vl.mean()),                 # system-level average speed, last 100 s
        speed_std=float(Vl.std()),                   # 0 for perfectly uniform flow
        min_speed=float(Vl.min()),
        frac_stopped=float((Vl < 0.5).mean()),       # share of vehicle-time (nearly) at standstill
    )
    if record:
        out.update(t=np.array(T), x=np.array(X), v=np.array(V))
    return out
