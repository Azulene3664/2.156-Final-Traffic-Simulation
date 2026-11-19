"""Multi-seed evaluation of the trained PPO policy on the open single-lane road.

Re-implements the simulation loop of test2.ipynb (run_simulation) so that it can be
run for many seeds and several control conditions, and writes the figures and the
results table used in the mid-term report.

Usage (from the repo root):
    python evaluation/evaluate.py                  # 30 seeds, figures -> evaluation/figures
    python evaluation/evaluate.py --original_rule  # brake rule exactly as in test2.ipynb

Only numpy and matplotlib are needed. The policy weights are read straight out of
ppo_av_platooning.zip with a restricted unpickler, so torch / stable-baselines3 are
not required and no pickled code is executed.
"""
import argparse
import collections
import csv
import io
import json
import os
import pickle
import re
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DT, STEPS, ROAD = 0.5, 600, 800.0
V0 = 15.0
JAM_SPEED = 5.0          # m/s, a vehicle slower than this counts as "in a jam"
T_DISTURB = 60.0         # s, the forced brake happens at steps 120..130


# --------------------------------------------------------------------------
# Policy: SB3 MlpPolicy actor (7 -> 64 -> 64 -> 1, tanh), deterministic action
# --------------------------------------------------------------------------
class _Storage:
    def __init__(self, dtype):
        self.dtype = dtype


def _rebuild_tensor(storage, offset, size, stride, *unused):
    return storage[offset:offset + int(np.prod(size))].reshape(size).copy()


class _WeightsOnlyUnpickler(pickle.Unpickler):
    _DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64}

    def __init__(self, f, archive, prefix):
        super().__init__(f)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if module == "torch" and name in self._DTYPES:
            return _Storage(self._DTYPES[name])
        raise pickle.UnpicklingError(f"blocked {module}.{name}")

    def persistent_load(self, pid):
        _, storage, key, _, _ = pid
        return np.frombuffer(self.archive.read(f"{self.prefix}/data/{key}"), dtype=storage.dtype)


def load_policy(model_zip):
    outer = zipfile.ZipFile(model_zip)
    inner = zipfile.ZipFile(io.BytesIO(outer.read("policy.pth")))
    pkl = [n for n in inner.namelist() if n.endswith("data.pkl")][0]
    state = _WeightsOnlyUnpickler(inner.open(pkl), inner, pkl.rsplit("/", 1)[0]).load()
    w = {k: np.asarray(v, dtype=np.float64) for k, v in state.items()}

    def policy(obs):
        h = np.tanh(w["mlp_extractor.policy_net.0.weight"] @ obs + w["mlp_extractor.policy_net.0.bias"])
        h = np.tanh(w["mlp_extractor.policy_net.2.weight"] @ h + w["mlp_extractor.policy_net.2.bias"])
        a = w["action_net.weight"] @ h + w["action_net.bias"]
        return float(np.clip(a[0], -3.0, 2.0))

    return policy


# --------------------------------------------------------------------------
# Simulation (same logic as run_simulation in test2.ipynb)
# --------------------------------------------------------------------------
class Vehicle:
    def __init__(self, vid, is_av, T, a, t0):
        self.id, self.is_av, self.T, self.a = vid, is_av, T, a
        self.x, self.v = 0.0, 12.0
        self.b, self.s0, self.v0 = 1.5, 2.0, V0
        self.t0 = t0
        self.ht, self.hx, self.hv = [], [], []
        self.exited = False


def av_obs(veh, vehicles):
    s_lead, dv_lead, s_av, v_av, s_lag, v_lag = 100.0, 0.0, 100.0, 15.0, 100.0, veh.v0
    idx = vehicles.index(veh)
    if idx > 0:
        lead = vehicles[idx - 1]
        s_lead, dv_lead = min(lead.x - veh.x, 100.0), veh.v - lead.v
        for i in range(idx - 1, -1, -1):
            if vehicles[i].is_av:
                s_av, v_av = min(vehicles[i].x - veh.x, 100.0), vehicles[i].v
                break
    if idx < len(vehicles) - 1:
        lag = vehicles[idx + 1]
        s_lag, v_lag = min(veh.x - lag.x, 100.0), lag.v
    return np.array([veh.v / 15, s_lead / 100, (dv_lead + 15) / 30, s_lag / 100, v_lag / 15, s_av / 100, v_av / 15])


def idm_acc(veh, s, dv):
    s_star = max(veh.s0 + veh.v * veh.T + veh.v * dv / (2 * np.sqrt(veh.a * veh.b)), veh.s0)
    a = veh.a * (1 - (veh.v / veh.v0) ** 4 - (s_star / max(s, 0.1)) ** 2)
    if s < veh.s0:
        a = -4.0
    return float(np.clip(a, -4.0, 2.0))


def run(mode, av_ratio, seed, policy=None, fixed_target=False):
    """mode: 'rl'       AVs use the PPO policy (+ AEB shield)
             'rl_res'   AVs use IDM (T = 1.0 s) plus a correction from the PPO policy (+ AEB shield)
             'idm'      AVs are plain IDM vehicles with T = 1.0 s, a = 1.0 m/s^2
             'throttle' AVs always command +2 m/s^2 (+ AEB shield)"""
    np.random.seed(seed)
    vehicles, done, n, timer = [], [], 0, 0.0
    braked = set()
    target = None
    for t in range(STEPS):
        timer -= DT
        if timer <= 0 and (not vehicles or vehicles[-1].x > 25.0):
            rand_T = np.clip(np.random.normal(1.5, 0.3), 0.8, 2.5)
            rand_a = np.clip(np.random.normal(1.0, 0.2), 0.5, 1.5)
            timer = np.random.uniform(0.8, 1.0)
            is_av = np.random.rand() < av_ratio
            vehicles.append(Vehicle(n, is_av, 1.0 if is_av else rand_T, 1.0 if is_av else rand_a, t * DT))
            n += 1
        acc = []
        for i, veh in enumerate(vehicles):
            s, dv = 999.0, 0.0
            if i > 0:
                s, dv = vehicles[i - 1].x - veh.x, veh.v - vehicles[i - 1].v
            if veh.is_av and mode in ("rl", "rl_res", "throttle"):
                if mode == "rl":
                    a = policy(av_obs(veh, vehicles))
                elif mode == "rl_res":
                    a = float(np.clip(idm_acc(veh, s, dv) + policy(av_obs(veh, vehicles)), -4.0, 2.0))
                else:
                    a = 2.0
                if s < 2.0 + veh.v * 0.8:      # AEB shield
                    a = -4.0
            else:
                a = idm_acc(veh, s, dv)
            if t == 120 and i == 5:
                target = veh
            # forced hard brake: original rule = whoever is 6th in line at each step (the index drifts
            # when a vehicle exits); fixed_target = the same vehicle for all 11 steps
            hit = (veh is target) if fixed_target else (len(vehicles) > 5 and i == 5)
            if 120 <= t <= 130 and hit:
                a = -4.0
                braked.add((veh.id, veh.is_av))
            acc.append(a)
        for i, veh in enumerate(vehicles):
            veh.v = float(np.clip(veh.v + acc[i] * DT, 0.0, veh.v0))
            veh.x += veh.v * DT
            veh.ht.append(t * DT); veh.hx.append(veh.x); veh.hv.append(veh.v)
        still = []
        for veh in vehicles:
            if veh.x >= ROAD:
                veh.exited = True
                done.append(veh)
            else:
                still.append(veh)
        vehicles = still
    everyone = done + vehicles
    after = [(tt, vv) for veh in everyone for tt, vv in zip(veh.ht, veh.hv) if tt >= T_DISTURB]
    metrics = dict(
        seed=seed,
        spawned=n,
        exited=len(done),
        jam_veh_s=sum(DT for _, vv in after if vv < JAM_SPEED),
        mean_speed=float(np.mean([vv for _, vv in after])),
        travel_time=float(np.mean([len(veh.ht) * DT for veh in done])),
        braked_is_av=any(b[1] for b in braked),
        n_braked=len(braked),
        # the same quantities for the vehicles that were not themselves forced to brake
        jam_followers=sum(DT for veh in everyone if veh.id not in {b[0] for b in braked}
                          for tt, vv in zip(veh.ht, veh.hv) if tt >= T_DISTURB and vv < JAM_SPEED),
        n_slowed=sum(1 for veh in everyone if veh.id not in {b[0] for b in braked}
                     and any(vv < JAM_SPEED for tt, vv in zip(veh.ht, veh.hv) if tt >= T_DISTURB)),
        delay=sum(DT * (1 - vv / V0) for _, vv in after),
    )
    return metrics, everyone


CONDITIONS = [  # label, mode, AV ratio, colour, which policy
    ("HV only", "idm", 0.0, "#898781", None),
    ("20% AV: PPO v1", "rl", 0.2, "#2a78d6", "v1"),
    ("20% AV: IDM, T = 1.0 s", "idm", 0.2, "#eb6834", None),
    ("20% AV: full throttle + AEB", "throttle", 0.2, "#1baf7a", None),
    ("20% AV: PPO v2 (IDM + correction)", "rl_res", 0.2, "#6250d6", "v2"),
]


# --------------------------------------------------------------------------
# Training log (parsed from the stored output of test2.ipynb, nothing is executed)
# --------------------------------------------------------------------------
def training_log():
    nb = json.load(open(os.path.join(ROOT, "test2.ipynb"), encoding="utf-8"))
    text = "".join("".join(o.get("text", "")) for o in nb["cells"][0]["outputs"] if o["output_type"] == "stream")
    steps, rew, length = [], [], []
    for block in text.split("| rollout/")[1:]:
        r = re.search(r"ep_rew_mean\s*\|\s*([-\d.e]+)", block)
        l = re.search(r"ep_len_mean\s*\|\s*([-\d.e]+)", block)
        s = re.search(r"total_timesteps\s*\|\s*(\d+)", block)
        if r and l and s:
            steps.append(int(s.group(1))); rew.append(float(r.group(1))); length.append(float(l.group(1)))
    return np.array(steps), np.array(rew), np.array(length)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
AV_COLOR = "#eb6834"


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.dpi": 300, "figure.dpi": 150,
    })


def fig_inputs(samples, out):
    import matplotlib.pyplot as plt
    hv = [v for v in samples if not v.is_av]
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 1.9), constrained_layout=True)
    ax[0].hist([v.T for v in hv], bins=30, color="#2a78d6", edgecolor="white", linewidth=0.3)
    ax[0].axvline(1.0, color=AV_COLOR, lw=1.5)
    ax[0].text(1.04, 0.98, "AV: 1.0 s", color=INK2, transform=ax[0].get_xaxis_transform(), va="top", fontsize=6.5)
    ax[0].set(title="(a) HV time headway T", xlabel="T (s)", ylabel="vehicles")
    ax[1].hist([v.a for v in hv], bins=30, color="#2a78d6", edgecolor="white", linewidth=0.3)
    ax[1].axvline(1.0, color=AV_COLOR, lw=1.5)
    ax[1].text(1.02, 0.98, "AV: 1.0 m/s²", color=INK2, transform=ax[1].get_xaxis_transform(), va="top", fontsize=6.5)
    ax[1].set(title="(b) HV max. acceleration a", xlabel="a (m/s²)")
    gaps = []
    by_seed = collections.defaultdict(list)
    for v in samples:
        by_seed[v.seed].append(v.t0)
    for ts in by_seed.values():
        gaps.extend(np.diff(sorted(ts)))
    vals, counts = np.unique(np.round(gaps, 1), return_counts=True)
    ax[2].bar(vals, counts / counts.sum() * 100, width=0.3, color="#2a78d6")
    ax[2].set(title="(c) Entry time gap", xlabel="gap between arrivals (s)", ylabel="% of arrivals")
    ax[2].set_xticks(vals)
    for a in ax:
        a.grid(axis="x", visible=False)
    for a in ax[:2]:
        a.set_ylim(0, a.get_ylim()[1] * 1.22)
    fig.savefig(out)
    plt.close(fig)


def fig_spacetime(panels, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    cmap = LinearSegmentedColormap.from_list("speed", ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef"])
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5, 2.45), sharex=True, sharey=True, constrained_layout=True)
    sc = None
    for ax, (title, vehicles) in zip(axes, panels):
        for v in vehicles:
            if v.is_av:
                ax.plot(v.ht, v.hx, color=AV_COLOR, lw=0.8, zorder=5)
            else:
                sc = ax.scatter(v.ht, v.hx, c=v.hv, cmap=cmap, vmin=0, vmax=15, s=0.6, linewidths=0, zorder=3, rasterized=True)
        ax.set_title(title)
        ax.grid(False)
        ax.set(xlim=(40, 200), ylim=(300, ROAD), xlabel="time (s)")
    axes[0].set_ylabel("position (m)")
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label("HV speed (m/s)")
    cb.outline.set_visible(False)
    axes[1].legend(handles=[Line2D([], [], color=AV_COLOR, lw=1.2, label="AV")],
                   loc="lower right", frameon=True, framealpha=0.9, edgecolor="none")
    fig.savefig(out)
    plt.close(fig)


def fig_results(results, out):
    import matplotlib.pyplot as plt
    steps, rew, length = training_log()
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.35), constrained_layout=True,
                           gridspec_kw={"width_ratios": [0.85, 1.25, 1.25]})
    ax[0].plot(steps / 1000, rew, color="#2a78d6", lw=1.5)
    ax[0].set(title="(a) PPO v1 training", xlabel="environment steps (thousands)", ylabel="mean episode return",
              ylim=(0, 100), xlim=(0, 102))
    ax[0].annotate(f"{rew[0]:.0f}", (steps[0] / 1000, rew[0]), xytext=(2, 5), textcoords="offset points", fontsize=6.5, color=INK2)
    ax[0].annotate(f"{rew[-1]:.0f}", (steps[-1] / 1000, rew[-1]), xytext=(-9, 5), textcoords="offset points", fontsize=6.5, color=INK2)
    short = ["HV\nonly", "PPO\nv1", "IDM\nT=1.0", "Throttle\n+AEB", "PPO\nv2"]
    rng = np.random.default_rng(0)
    for k, (key, title, ylabel) in enumerate([("jam_followers", "(b) Jam severity per seed", "follower vehicle-seconds below 5 m/s"),
                                              ("exited", "(c) Throughput per seed", "vehicles exited in 300 s")]):
        a = ax[k + 1]
        for j, (label, _, _, color, _) in enumerate(CONDITIONS):
            y = np.array([r[key] for r in results[label]], dtype=float)
            if key == "jam_followers":
                y = np.maximum(y, 1.0)
            a.scatter(j + rng.uniform(-0.2, 0.2, len(y)), y, s=9, color=color, alpha=0.75, linewidths=0, zorder=3)
            m = np.median(y)
            a.plot([j - 0.3, j + 0.3], [m, m], color=INK, lw=1.2, zorder=4)
        a.set_xticks(range(len(CONDITIONS)))
        a.set_xticklabels(short)
        a.tick_params(axis="x", colors=INK2, length=0)
        a.set(title=title, ylabel=ylabel, xlim=(-0.6, len(CONDITIONS) - 0.4))
        a.grid(axis="x", visible=False)
    ax[1].set_yscale("log")
    ax[1].set_yticks([1, 10, 100])
    ax[1].set_yticklabels(["≤1", "10", "100"])
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--model_v1", default=os.path.join(ROOT, "ppo_av_platooning.zip"))
    ap.add_argument("--model_v2", default=os.path.join(ROOT, "models", "ppo_v2_residual.zip"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    ap.add_argument("--original_rule", action="store_true",
                    help="use the drifting 'sixth vehicle in line' brake rule of test2.ipynb instead of one fixed vehicle")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    policies = {"v1": load_policy(args.model_v1), "v2": load_policy(args.model_v2), None: None}
    fixed = not args.original_rule

    results, samples = {}, []
    for label, mode, ratio, _, pk in CONDITIONS:
        results[label] = []
        for seed in range(args.seeds):
            m, vehicles = run(mode, ratio, seed, policies[pk], fixed_target=fixed)
            results[label].append(m)
            if pk == "v1":
                for v in vehicles:
                    v.seed = seed
                samples.extend(vehicles)

    keys = ["seed", "spawned", "exited", "jam_followers", "n_slowed", "jam_veh_s", "delay", "mean_speed", "travel_time", "braked_is_av", "n_braked"]
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition"] + keys)
        for label, rows in results.items():
            for r in rows:
                w.writerow([label] + [r[k] for k in keys])

    base = results["HV only"]
    bj, be = np.array([r["jam_followers"] for r in base]), np.array([r["exited"] for r in base])
    print(f"{args.seeds} seeds, mean ± std, brake rule: {'one fixed vehicle' if fixed else 'original (drifting index)'}")
    print(f"{'condition':36s} {'exited':>10s} {'follower jam':>14s} {'median':>7s} {'max':>6s} {'slowed':>7s} {'jam<base':>9s} {'jam>base':>9s} {'thr>base':>9s}")
    for label, rows in results.items():
        g = lambda k: np.array([r[k] for r in rows], dtype=float)
        jam, ex = g("jam_followers"), g("exited")
        print(f"{label:36s} {ex.mean():5.1f}±{ex.std():3.1f} {jam.mean():7.1f}±{jam.std():5.1f} {np.median(jam):7.1f} {jam.max():6.1f} {g('n_slowed').mean():7.2f} "
              f"{int((jam < bj).sum()):>6d}/{len(bj)} {int((jam > bj).sum()):>6d}/{len(bj)} {int((ex > be).sum()):>6d}/{len(be)}")
    print("vehicles sampled for Fig. 1:", len(samples), "of which HV:", sum(not v.is_av for v in samples))

    # the single run shown in the first draft, under both brake rules (all-vehicle jam metric)
    for rule, fx in (("original rule", False), ("fixed vehicle", True)):
        a, b = run("idm", 0.0, 666, fixed_target=fx)[0], run("rl", 0.2, 666, policies["v1"], fixed_target=fx)[0]
        print(f"seed 666, {rule}: HV only jam {a['jam_veh_s']} ({a['n_braked']} vehicles braked) | PPO v1 jam {b['jam_veh_s']} ({b['n_braked']} braked)")

    # representative seed for Fig. 2: the one where PPO v1 is at its median jam severity
    j1 = np.array([r["jam_followers"] for r in results["20% AV: PPO v1"]])
    rep_seed = int(np.argsort(j1)[len(j1) // 2])
    j2 = [r["jam_followers"] for r in results["20% AV: PPO v2 (IDM + correction)"]]
    print(f"Fig. 2 seed {rep_seed}: follower jam HV only {bj[rep_seed]} | PPO v1 {j1[rep_seed]} | PPO v2 {j2[rep_seed]}")

    style()
    fig_inputs(samples, os.path.join(args.out, "fig1_inputs.png"))
    panels = [("HV only", run("idm", 0.0, rep_seed, fixed_target=fixed)[1]),
              ("20% AV, PPO v1", run("rl", 0.2, rep_seed, policies["v1"], fixed_target=fixed)[1]),
              ("20% AV, PPO v2", run("rl_res", 0.2, rep_seed, policies["v2"], fixed_target=fixed)[1])]
    fig_spacetime(panels, os.path.join(args.out, "fig2_spacetime.png"))
    fig_results(results, os.path.join(args.out, "fig3_results.png"))
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
