"""What a rational team does with a joule, and a three-parameter prior for it.

Teams run near time-optimal energy management. The structure is simple enough
to use as a prior without importing anyone's full model, and it is worth
deriving rather than asserting because two of the project's results fall out of
it.

Time to cover ds at speed v is ds/v. Adding kinetic energy dE raises v by
dE/(m v), so

    d(time)/dE = -ds / (m v^3)

A joule is worth a great deal at corner exit and almost nothing near top speed.
Under Pontryagin the optimal policy is bang-bang against a near-constant
costate lambda -- the shadow price of stored energy for that lap:

    deploy   where  1/(m v^3) > lambda
    harvest  where  v is highest, for the same reason inverted
    coast    otherwise

Two consequences, both of which the repo already measured by other means:

1. It derives the Zone A versus Zone C result. Spending E over a straight of
   length L at speed v adds power E*v/L, and near terminal speed drag power
   goes as v^3, so dv = E / (1.5 * rho * CdA * v * L). The long straight into
   the hairpin is the worst place to spend, not the best, because both v and L
   are large. Nobody typed that in; it is the v^3.

2. It predicts super-clipping at the end of straights. Harvesting where the
   joule is cheapest means harvesting where v is highest -- which is exactly
   where the 290-355 km/h taper makes deployment nearly worthless anyway.

Deliberately not deep inverse RL: three parameters are identifiable from one
lap and a network is not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .regs import p_dep_max, taper


def joule_value(m, v, ds: float = 1.0):
    """Seconds saved per joule of kinetic energy added, over ds metres.

    The 1/(m v^3) that drives everything else in this module.
    """
    m = np.asarray(m, float)
    v = np.maximum(np.asarray(v, float), 1.0)
    return ds / (m * v ** 3)


def dv_per_mj(rho: float, cda: float, v_end: float, straight_m: float,
             energy_j: float = 1.0e6) -> float:
    """Speed gained at the braking point per MJ spent down the straight.

    Near the end of a long straight the car is power-limited, so the useful
    form is dv = dP / (dP_drag/dv) with P_drag = 1/2 rho CdA v^3. Treating the
    energy as pure kinetic gain instead gives dv = E/(m v), which is about
    seven times too large and saturates every overtake opportunity at p = 1.
    """
    v_end = max(float(v_end), 1.0)
    d_power = energy_j * v_end / max(float(straight_m), 1.0)
    return float(d_power / max(3.0 * 0.5 * rho * cda * v_end ** 2, 1e-9))


@dataclass(frozen=True)
class PolicyFit:
    """Three interpretable numbers per lap, and what they imply."""
    v_cut: float      # m/s, deployment cut-off: above this the joule is too cheap
    v_harv: float     # m/s, super-clip onset
    width: float      # m/s, softness of both transitions
    residual_rms: float
    n_samples: int

    @property
    def lam(self) -> float:
        """The implied costate, in seconds per joule per metre.

        lambda is the shadow price of the lap's energy, and v_cut is where the
        marginal value crosses it -- so a low cut-off means the driver is
        hoarding and a high one means they are spending.
        """
        return float(joule_value(790.0, self.v_cut))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def pmp_deployment(v, v_cut: float, v_harv: float, width: float,
                   in_zone, regs, on_power=None, braking=None):
    """Store-side power the prior expects, W. Positive deploys.

    Bang-bang softened by `width`: a real driver's transition is not a step,
    and a step has no gradient for a fit to follow.
    """
    v = np.asarray(v, float)
    cap = p_dep_max(v, in_zone, regs)
    dep = cap * _sigmoid((v_cut - v) / max(width, 1e-3))
    harv = regs.p_harv_max * _sigmoid((v - v_harv) / max(width, 1e-3))
    if on_power is not None:
        gate = np.asarray(on_power, bool)
        if braking is not None:
            gate = gate | np.asarray(braking, bool)
        harv = np.where(gate, harv, 0.0)
        dep = np.where(np.asarray(on_power, bool), dep, 0.0)
    if braking is not None:
        dep = np.where(np.asarray(braking, bool), 0.0, dep)
    return dep - harv


def fit_policy(v, p_k_obs, in_zone, regs, on_power=None, braking=None,
               v0: tuple = (70.0, 85.0, 4.0)) -> PolicyFit:
    """Fit (v_cut, v_harv, width) to a reconstructed store-side power trace.

    Three parameters against a lap of samples, so it is well posed and stays
    interpretable. The fit is a *prior* for the estimator: it tilts the
    posterior inside the identified set and never overrides a bound.
    """
    v = np.asarray(v, float)
    y = np.asarray(p_k_obs, float)
    good = np.isfinite(v) & np.isfinite(y)
    if good.sum() < 12:
        raise ValueError(f"only {int(good.sum())} usable samples to fit a policy")

    def resid(p):
        pred = pmp_deployment(v[good], p[0], p[1], p[2], np.asarray(in_zone)[good],
                              regs,
                              None if on_power is None else np.asarray(on_power)[good],
                              None if braking is None else np.asarray(braking)[good])
        return (pred - y[good]) / 1.0e4

    sol = least_squares(resid, np.array(v0, float), loss="soft_l1", f_scale=3.0,
                        bounds=([20.0, 20.0, 0.5], [110.0, 120.0, 30.0]))
    return PolicyFit(v_cut=float(sol.x[0]), v_harv=float(sol.x[1]),
                     width=float(sol.x[2]),
                     residual_rms=float(np.sqrt(np.mean((resid(sol.x) * 1e4) ** 2))),
                     n_samples=int(good.sum()))


def aggression(fit: PolicyFit, v_lo: float, v_hi: float) -> float:
    """0 (hoarding) to 1 (spending): the fraction of this lap's speed range over
    which the driver is willing to spend.

    Two earlier versions of this were wrong in ways worth recording, because the
    number ends up on a per-driver fingerprint where nobody would catch it.

    `taper(v_cut)` is inverted: taper falls with speed, so a driver who stops
    deploying at 180 km/h -- a hoarder -- scored 1.00 and one who deployed to
    342 km/h scored 0.20. And `1 - taper(v_cut)` is no better, because taper is
    flat at 1 below 290 km/h, so every realistic cut-off scores exactly 0 and
    the metric has no resolution where the drivers actually differ.

    Referencing the lap's own speed range keeps it interpretable and comparable
    between circuits, which is what the fingerprint view needs.
    """
    span = max(float(v_hi) - float(v_lo), 1e-6)
    return float(np.clip((fit.v_cut - float(v_lo)) / span, 0.0, 1.0))
