"""Evaluate the ring-road AV policy on test seeds and draw the figures.

    python ring/evaluate.py                     # uses models/ring_ppo.zip, 10 seeds per ring length

Conditions: 22 human drivers; 21 humans + 1 AV with the PPO policy; 21 humans + 1 AV that holds
0.8 x the uniform-flow speed (a hand-written reference that is told the ring length).
Each run: 350 s warm-up with human driving only, then 300 s with the AV active; metrics are taken
over the last 100 s. The random stream is identical across conditions until the AV takes over.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import sim  # noqa: E402
from env import V_TARGET_MAX, WARMUP_S  # noqa: E402
from rollout import rollout  # noqa: E402

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_HUMAN, C_PPO, C_REF, C_AV = "#898781", "#2a78d6", "#1baf7a", "#eb6834"
LENGTHS = (210, 220, 230, 240, 250, 260, 270, 280, 290)      # training range is 220-270 m


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


def fig_spacetime(runs, L, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("speed", ["#0d366b", "#1c5cab", "#3987e5", "#b7d3f6"])
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5, 2.5), sharey=True, constrained_layout=True)
    for ax, (title, r) in zip(axes, runs):
        t, x, v = r["t"][::5], r["x"][::5], r["v"][::5]
        for i in range(1, sim.N_VEH):
            sc = ax.scatter(t, x[:, i], c=v[:, i], cmap=cmap, vmin=0, vmax=8, s=0.25, linewidths=0, rasterized=True)
        if "AV" in title:
            ax.scatter(t, x[:, 0], color=C_AV, s=0.6, linewidths=0, zorder=5, rasterized=True, label="AV")
            ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", markerscale=8)
        else:
            ax.scatter(t, x[:, 0], c=v[:, 0], cmap=cmap, vmin=0, vmax=8, s=0.25, linewidths=0, rasterized=True)
        ax.axvline(WARMUP_S, color=INK2, lw=0.8, ls="--")
        ax.set_title(title)
        ax.grid(False)
        ax.set(xlim=(250, 650), ylim=(0, L), xlabel="time (s)")
    axes[0].set_ylabel("position on ring (m)")
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.015)
    cb.set_label("speed (m/s)")
    cb.outline.set_visible(False)
    fig.savefig(out)
    plt.close(fig)


def fig_distributions(v_human, v_av, out):
    """Inputs of the simulator and the resulting distribution of vehicle speeds (output)."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 1.9), constrained_layout=True)
    dens = sim.N_VEH / rng.uniform(220, 270, 20000) * 1000
    ax[0].hist(dens, bins=30, color=C_PPO, edgecolor="white", linewidth=0.3, weights=np.full(dens.size, 100 / dens.size))
    ax[0].set(title="(a) Traffic density in training", xlabel="vehicles/km", ylabel="% of episodes")
    noise = rng.normal(0, sim.NOISE_STD, 20000)
    ax[1].hist(noise, bins=40, color=C_PPO, edgecolor="white", linewidth=0.3, weights=np.full(noise.size, 100 / noise.size))
    ax[1].set(title="(b) Human acceleration noise", xlabel="m/s²", ylabel="% of time steps")
    bins = np.linspace(0, 9, 37)
    ax[2].hist(v_human, bins=bins, color=C_HUMAN, alpha=0.9, weights=np.full(v_human.size, 100 / v_human.size), label="22 human drivers")
    ax[2].hist(v_av, bins=bins, color=C_PPO, alpha=0.8, weights=np.full(v_av.size, 100 / v_av.size), label="with 1 AV (PPO)")
    ax[2].set(title="(c) Vehicle speeds, 260 m ring", xlabel="m/s", ylabel="% of vehicle-time", ylim=(0, 55))
    ax[2].legend(loc="upper right", frameon=False, fontsize=6)
    for a in ax:
        a.grid(axis="x", visible=False)
    fig.savefig(out)
    plt.close(fig)


def fig_summary(rows, train_log, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.3), constrained_layout=True)
    Ls = np.array(LENGTHS)
    dens = sim.N_VEH / Ls * 1000
    if train_log is not None:
        ax[0].plot(train_log[:, 0] / 1e6, train_log[:, 1], color=C_PPO, lw=1.2)
        ax[0].set(title="(a) PPO training", xlabel="decisions (millions)", ylabel="mean episode return")
    for k, (key, title, ylabel) in enumerate([("mean_speed", "(b) Average speed, all vehicles", "m/s"),
                                              ("speed_std", "(c) Spread of speeds (wave size)", "m/s")]):
        a = ax[k + 1]
        a.axvspan(sim.N_VEH / 270 * 1000, sim.N_VEH / 220 * 1000, color="#f0efec", zorder=0)
        if key == "mean_speed":
            a.plot(dens, [sim.equilibrium_speed(L) for L in Ls], color=INK2, lw=0.9, ls="--", label="uniform flow (ideal)")
        for cond, color in (("human", C_HUMAN), ("reference", C_REF), ("ppo", C_PPO)):
            y = np.array([[r[key] for r in rows if r["cond"] == cond and r["ring_length"] == L] for L in Ls])
            label = {"human": "22 human drivers", "reference": "1 AV: fixed 0.8 × ideal", "ppo": "1 AV: PPO"}[cond]
            a.errorbar(dens, y.mean(1), yerr=y.std(1), color=color, lw=1.3, marker="o", ms=3, capsize=1.5, elinewidth=0.7, label=label)
        a.set(title=title, xlabel="density (vehicles/km)", ylabel=ylabel)
        a.set_ylim(bottom=0)
    ax[1].set_ylim(0, 10.5)
    ax[1].legend(loc="upper right", frameon=False, fontsize=6)
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "ring_ppo.zip"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    ppo = lambda obs: model.predict(obs, deterministic=True)[0][0]

    rows = []
    for L in LENGTHS:
        ref_action = 2.0 * 0.8 * sim.equilibrium_speed(L) / V_TARGET_MAX - 1.0
        for cond, policy in (("human", None), ("ppo", ppo), ("reference", lambda obs: ref_action)):
            for seed in range(args.seeds):
                r = rollout(L, seed, policy)
                r["cond"] = cond
                rows.append(r)
    keys = ["cond", "ring_length", "seed", "mean_speed", "speed_std", "min_speed", "frac_stopped", "collided"]
    with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for r in rows:
            w.writerow([r[k] for k in keys])

    print(f"{args.seeds} seeds per ring length, metrics over the last 100 s (mean ± std over seeds)")
    print(f"{'L':>4s} {'ideal':>6s} | {'human speed':>12s} {'PPO speed':>12s} {'ref speed':>12s} | {'human spread':>12s} {'PPO spread':>11s} | {'stopped h/PPO':>13s} | PPO waves gone | PPO faster | crashes")
    for L in LENGTHS:
        g = lambda c, k: np.array([r[k] for r in rows if r["cond"] == c and r["ring_length"] == L], dtype=float)
        gone = int((g("ppo", "speed_std") < 0.5).sum())
        faster = int((g("ppo", "mean_speed") > g("human", "mean_speed")).sum())
        print(f"{L:>4d} {sim.equilibrium_speed(L):6.2f} | {g('human','mean_speed').mean():5.2f}±{g('human','mean_speed').std():4.2f}  "
              f"{g('ppo','mean_speed').mean():5.2f}±{g('ppo','mean_speed').std():4.2f}  {g('reference','mean_speed').mean():5.2f}±{g('reference','mean_speed').std():4.2f}  | "
              f"{g('human','speed_std').mean():5.2f}±{g('human','speed_std').std():4.2f}  {g('ppo','speed_std').mean():5.2f}±{g('ppo','speed_std').std():4.2f} | "
              f"{g('human','frac_stopped').mean():5.2f} / {g('ppo','frac_stopped').mean():4.2f} | {gone:>6d}/{args.seeds}     | {faster:>4d}/{args.seeds}    | {int(g('ppo','collided').sum())}")

    style()
    L = 260
    runs = [("22 human drivers", rollout(L, 0, None, record=True)), ("21 human drivers + 1 AV (PPO)", rollout(L, 0, ppo, record=True))]
    fig_spacetime(runs, L, os.path.join(args.out, "ring_spacetime.png"))
    n = int(100.0 / sim.DT)
    vh = np.concatenate([rollout(L, s_, None, record=True)["v"][-n:].ravel() for s_ in range(args.seeds)])
    recs = [rollout(L, s_, ppo, record=True) for s_ in range(args.seeds)]
    va = np.concatenate([r["v"][-n:].ravel() for r in recs])
    fig_distributions(vh, va, os.path.join(args.out, "ring_distributions.png"))
    gap = lambda r: ((np.roll(r["x"][-n:], -1, axis=1) - r["x"][-n:]) % L - sim.VEH_LEN)
    print(f"L={L}, last 100 s with AV: AV gap {np.mean([gap(r)[:, 0].mean() for r in recs]):.1f} m, human gap {np.mean([gap(r)[:, 1:].mean() for r in recs]):.1f} m, "
          f"AV speed {np.mean([r['v'][-n:, 0].mean() for r in recs]):.2f} m/s; human-only: share of vehicle-time below 0.5 m/s {np.mean(vh < 0.5):.2f}")
    log_path = args.model.replace(".zip", "_trainlog.csv")
    train_log = np.loadtxt(log_path, delimiter=",", skiprows=1) if os.path.exists(log_path) else None
    fig_summary(rows, train_log, os.path.join(args.out, "ring_summary.png"))
    print("figures written to", args.out)


if __name__ == "__main__":
    main()
