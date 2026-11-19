"""Single-lane ring road with IDM human drivers and (optionally) one controlled AV.

Set-up follows the mixed-autonomy ring experiment of Wu et al., "Flow: A Modular Learning
Framework for Mixed Autonomy Traffic" (arXiv:1710.05465), which is a simulation of the
field experiment of Sugiyama et al. (2008): 22 vehicles of 5 m on a ring of about 230 m.
Human drivers: IDM with v0 = 30 m/s, T = 1 s, a = 1 m/s^2, b = 1.5 m/s^2, s0 = 2 m,
plus Gaussian acceleration noise (std 0.2 m/s^2).
"""
import numpy as np

N_VEH = 22
VEH_LEN = 5.0
DT = 0.1
V0, T_HW, A_MAX, B_COMF, S0, DELTA = 30.0, 1.0, 1.0, 1.5, 2.0, 4
NOISE_STD = 0.2
AV_ACC_MIN, AV_ACC_MAX = -1.5, 1.0     # range of the learned command
AV_BRAKE_MAX = 5.0                     # deceleration available to the safety filter
MEMORY_S = 30.0                        # time constant of the AV's running averages


def equilibrium_speed(ring_length, n=N_VEH):
    """Uniform-flow speed: every vehicle at the same gap and speed (IDM acceleration = 0)."""
    h = ring_length / n - VEH_LEN
    lo, hi = 0.0, V0
    for _ in range(60):
        v = 0.5 * (lo + hi)
        acc = 1 - (v / V0) ** DELTA - ((S0 + v * T_HW) / h) ** 2
        lo, hi = (v, hi) if acc > 0 else (lo, v)
    return 0.5 * (lo + hi)


class RingSim:
    def __init__(self, ring_length, rng, av_index=None, noise_std=NOISE_STD):
        self.L, self.rng, self.av, self.noise_std = float(ring_length), rng, av_index, noise_std
        self.x = np.arange(N_VEH) * self.L / N_VEH          # vehicle i follows vehicle i+1
        self.v = np.zeros(N_VEH)
        self.t = 0.0
        self.collided = False
        self.avg_v, self.avg_h = 0.0, self.L / N_VEH - VEH_LEN   # running averages of the AV's own speed and gap

    def gaps(self):
        lead = np.roll(self.x, -1)
        return (lead - self.x) % self.L - VEH_LEN

    def idm(self, h, v, v_lead):
        s_star = S0 + np.maximum(0.0, v * T_HW + v * (v - v_lead) / (2 * np.sqrt(A_MAX * B_COMF)))
        return A_MAX * (1 - (v / V0) ** DELTA - (s_star / np.maximum(h, 0.01)) ** 2)

    def safe_acc(self, acc, h, v, v_lead, tau):
        """Safety filter for the AV: never exceed the speed from which it can still stop behind its
        leader, assuming the leader may brake at AV_BRAKE_MAX (Krauss safe speed)."""
        b = AV_BRAKE_MAX
        v_safe = -b * tau + np.sqrt((b * tau) ** 2 + v_lead ** 2 + 2 * b * max(h - 1.0, 0.0))
        return float(np.clip(min(acc, (v_safe - v) / tau), -b, AV_ACC_MAX))

    def step(self, av_acc=None, tau=DT):
        """Advance one time step. av_acc=None: the AV (if any) drives like a human."""
        h = self.gaps()
        v_lead = np.roll(self.v, -1)
        acc = self.idm(h, self.v, v_lead) + self.rng.normal(0.0, self.noise_std, N_VEH)
        acc = np.clip(acc, -6.0, A_MAX + 0.5)
        if self.av is not None and av_acc is not None:
            i = self.av
            acc[i] = self.safe_acc(av_acc, h[i], self.v[i], v_lead[i], tau)
        self.v = np.maximum(self.v + acc * DT, 0.0)
        self.x = (self.x + self.v * DT) % self.L
        self.t += DT
        g = self.gaps()
        if self.av is not None:
            k = DT / MEMORY_S
            self.avg_v += k * (self.v[self.av] - self.avg_v)
            self.avg_h += k * (g[self.av] - self.avg_h)
        if g.min() <= 0.0:
            self.collided = True
        return acc
