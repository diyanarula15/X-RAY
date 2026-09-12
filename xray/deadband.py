"""Where the driver stops deploying, estimated before the filter runs.

Invariant 9 in its sharpest form: between the deployment cut-off v_cut and the
super-clip onset v_harv the policy says P_K = 0, so the interval balance there
is pure ICE against drag and the residual is delta_CdA * 0.5 rho v^3 with
nothing else in it. That is Stage 1's high-speed window relocated to a speed
that exists on a real circuit -- no 340 km/h threshold, no clean-sample filter.

The reason this is its own module and not a particle dimension is measured. Let
the filter infer v_cut jointly and the particles escape the test rather than
pass it: v_cut ran to 98 m/s (353 km/h, above the car's top speed) where the
dead band is empty, and corr(CdA, log weight) came out -0.00. The likelihood
looked wired in and was doing nothing. Swept against the *true* dead band the
same residual minimises at CdA_X = 0.66, exactly the truth -- so the statistic
was never the problem, the joint inference was.

So v_cut is estimated first, from the shape of implied wheel power against
speed, and handed to the filter as an input.

Two constraints do all the work here and both were found by their absence.

1. **No free intercept.** Over the 210-355 km/h range a v^3 column and a
   constant are nearly collinear, so an unconstrained intercept buys fit by
   pushing drag up and the offset down: CdA_X 0.976 with a -156 kW intercept,
   against a true 0.660. Rolling resistance enters as its own v-proportional
   column, which is what it physically is.
2. **Non-negative coefficients.** With F_rr free in sign the fit chose -1459 N
   and used it as the intercept it had just been denied (CdA_X 0.980). Drag
   area, rolling resistance and deployment amplitude are all non-negative by
   physics, so the fit is a non-negative least squares and the escape is
   closed. That single change moved the no-changepoint fit from 0.980 to 0.613.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from .balance import M_REF_KG, fuel_mass
from .constants import G
from .regs import FULL_THROTTLE_FRAC, RegSet, p_ice_max, taper

# Fraction of gated samples that must fall on each side of a candidate cut-off.
# A changepoint with three samples above it is a fit to noise, and the BIC
# penalty alone does not stop it because the amplitude column goes very nearly
# constant there and absorbs the offset the fit was denied.
SIDE_MIN_FRAC = 0.10
# Grid step for the scan, m/s. 0.5 m/s is 1.8 km/h; the CdA estimate moves by
# less than 0.01 per 4 m/s of v_cut, so finer is spurious precision.
V_GRID_STEP = 0.5
# Minimum dead-band samples for the answer to be worth reporting at all.
N_DEAD_MIN = 30
# BIC improvement over the no-changepoint model below which the step is not
# claimed. 2 is the conventional "positive evidence" threshold on a BIC
# difference; measured gains where a step really exists are two orders larger
# (169.7 on seed 42), so the threshold is not what decides anything.
BIC_MARGIN = 2.0


@dataclass(frozen=True)
class CutSpeeds:
    """The dead band, as measured. `v_harv = inf` means no super-clip step."""
    v_cut: float
    v_harv: float
    cda_hint: float        # drag area from the three-column fit itself
    f_rr_hint: float
    amplitude_w: float     # fitted deployment amplitude below v_cut
    n_samples: int         # gated intervals the fit saw
    n_dead: int            # of those, how many land in the dead band
    bic_gain: float        # against the no-changepoint model
    declined: bool
    note: str

    @property
    def dead_band(self) -> tuple:
        return (self.v_cut, self.v_harv)


def _bic(sse: float, n: int, k: int) -> float:
    return n * np.log(max(sse, 1e-12) / n) + k * np.log(n)


def _nnls_sse(cols, y):
    A = np.column_stack(cols)
    c, _ = nnls(A, y)
    return c, float(np.sum((A @ c - y) ** 2))


def gated_terms(feed, regs: RegSet, rho: float, eta_d: float,
                m_published: float, fuel_start: float,
                fuel_burn_per_lap: float):
    """Per-interval regression terms on full-throttle, brakes-off intervals.

    Returns (v, drag_col, roll_col, y, mask) where

        y = eta_d * P_ice,max - dE_mech/dt
          = 0.5 rho CdA v^3 + F_rr * vbar * m/M_ref - P_K

    Gated to full throttle because below it the ICE output is unknown, and
    recovering it needs an engine map: an earlier real-data version inferred
    ICE output from the throttle trace and produced CdA lower bounds above
    13 m^2. Brakes-off because a braking interval is harvesting, not a dead
    band. Labelling is conservative on both -- the *minimum* throttle and the
    *either* endpoint brake flag across the interval (invariant 4).
    """
    v, t, z = (np.asarray(getattr(feed, a), float) for a in ("v", "t", "z"))
    dt = np.diff(t)
    v0, v1 = v[:-1], v[1:]
    vbar = 0.5 * (v0 + v1)
    v3bar = 0.5 * (v0 ** 3 + v1 ** 3)
    m_nom = m_published + fuel_mass(np.asarray(feed.lap_frac, float)[:-1],
                                    fuel_start, fuel_burn_per_lap)
    d_e = m_nom * (0.5 * (v1 ** 2 - v0 ** 2) + G * np.diff(z))
    thr_lo = np.minimum(feed.throttle[:-1], feed.throttle[1:])
    thr_hi = np.maximum(feed.throttle[:-1], feed.throttle[1:])
    rpm_hi = np.maximum(feed.rpm[:-1], feed.rpm[1:])
    brake = np.asarray(feed.brake, float) > 0.5
    mask = (thr_lo >= FULL_THROTTLE_FRAC) & ~(brake[:-1] | brake[1:])
    y = eta_d * p_ice_max(rpm_hi, thr_hi, regs) - d_e / dt
    return (vbar[mask], (0.5 * rho * v3bar)[mask],
            (vbar * m_nom / M_REF_KG)[mask], y[mask], mask)


def fit_cut_speeds(feed, regs: RegSet, rho: float, eta_d: float = 0.95,
                   m_published: float = 790.0, fuel_start: float = 70.0,
                   fuel_burn_per_lap: float = 1.15) -> CutSpeeds:
    """Scan for the deployment cut-off, then for the super-clip onset above it.

    Model, on full-throttle brakes-off intervals:

        y(v) = 0.5 rho CdA v^3 + F_rr v m/M_ref
               - A * taper(v) * 1[v < v_cut]
               + H * 1[v > v_harv]

    with every coefficient non-negative. v_cut is scanned; v_harv is scanned
    afterwards over the dead band only, and *declined* if it buys no BIC. On
    the Stage 1 simulator it is always declined, because `vehicle.step` harvests
    on the brakes only -- there is no off-throttle super-clipping to find, so
    the dead band runs from v_cut to the top of the speed range. Guessing a
    v_harv there would invent the very number the fit is supposed to measure.
    """
    v, drag, roll, y, mask = gated_terms(feed, regs, rho, eta_d, m_published,
                                         fuel_start, fuel_burn_per_lap)
    n = len(y)
    if n < 4 * N_DEAD_MIN:
        return CutSpeeds(np.nan, np.inf, np.nan, np.nan, 0.0, n, 0, 0.0, True,
                         f"only {n} full-throttle brakes-off intervals; a "
                         f"changepoint scan needs {4 * N_DEAD_MIN}")
    tp = taper(v, False)
    c_null, sse_null = _nnls_sse((drag, roll), y)
    bic_null = _bic(sse_null, n, 2)

    best = None
    for v_c in np.arange(v.min(), v.max(), V_GRID_STEP):
        below = v < v_c
        if below.sum() < SIDE_MIN_FRAC * n or (~below).sum() < SIDE_MIN_FRAC * n:
            continue
        c, sse = _nnls_sse((drag, roll, -tp * below), y)
        b = _bic(sse, n, 3)
        if best is None or b < best[0]:
            best = (b, float(v_c), c)
    if best is None or best[0] > bic_null - BIC_MARGIN:
        return CutSpeeds(np.nan, np.inf, float(c_null[0]), float(c_null[1]),
                         0.0, n, 0, 0.0, True,
                         "no deployment cut-off in speed: the no-changepoint "
                         "model is not beaten, so the dead band is not "
                         "identified and no v_cut is claimed")
    bic_cut, v_cut, c_cut = best

    # v_harv, above v_cut only. v_harv >= v_cut always; the reverse is a
    # contradiction, not a policy, and drawing the two independently is what
    # gave particles empty dead bands in the joint version.
    dead = v >= v_cut
    v_harv, bic_harv = np.inf, None
    if dead.sum() >= 2 * N_DEAD_MIN:
        for v_h in np.arange(v_cut, v.max(), V_GRID_STEP):
            above = v > v_h
            if above.sum() < SIDE_MIN_FRAC * n or (v >= v_cut).sum() - above.sum() < N_DEAD_MIN:
                continue
            c, sse = _nnls_sse((drag, roll, -tp * (v < v_cut),
                                above.astype(float)), y)
            b = _bic(sse, n, 4)
            if bic_harv is None or b < bic_harv:
                bic_harv, v_harv = b, float(v_h)
        if bic_harv is None or bic_harv > bic_cut - BIC_MARGIN:
            v_harv = np.inf

    n_dead = int(((v >= v_cut) & (v <= v_harv)).sum())
    note = (f"v_cut {v_cut * 3.6:.0f} km/h from {n} full-throttle brakes-off "
            f"intervals, BIC gain {bic_null - bic_cut:.1f}; "
            + ("no super-clip step above it, so the dead band is open at the "
               "top" if not np.isfinite(v_harv)
               else f"v_harv {v_harv * 3.6:.0f} km/h"))
    return CutSpeeds(v_cut=v_cut, v_harv=v_harv, cda_hint=float(c_cut[0]),
                     f_rr_hint=float(c_cut[1]), amplitude_w=float(c_cut[2]),
                     n_samples=n, n_dead=n_dead,
                     bic_gain=float(bic_null - bic_cut), declined=False,
                     note=note)


def dead_band_cda(feed, cuts: CutSpeeds, regs: RegSet, rho: float,
                  f_rr: float, eta_d: float = 0.95, m_published: float = 790.0,
                  fuel_start: float = 70.0,
                  fuel_burn_per_lap: float = 1.15) -> float:
    """One-parameter least squares for CdA inside the dead band.

    This is the whole point of estimating v_cut first: with the band fixed and
    P_K = 0 by policy, CdA is the only free parameter left and the fit is a
    single ratio. No intercept -- see the module docstring for the 0.976 that
    an intercept bought.

    Measured on seed 42 at the fitted v_cut: 0.631 against a true 0.660, and
    0.629 to 0.665 as the band is narrowed from 234 to 324 km/h. The residual
    3-5% is contamination: the simulator's policy is positional, so a little
    deployment survives above any speed cut-off, and it can only bias drag
    *down*.
    """
    v, drag, roll, y, _ = gated_terms(feed, regs, rho, eta_d, m_published,
                                      fuel_start, fuel_burn_per_lap)
    dead = (v >= cuts.v_cut) & (v <= cuts.v_harv)
    if dead.sum() < N_DEAD_MIN:
        raise ValueError(
            f"{int(dead.sum())} dead-band intervals, need {N_DEAD_MIN}: this "
            "trace does not spend enough time above the deployment cut-off at "
            "full throttle to identify drag from the dead band")
    d, r, yy = drag[dead], roll[dead], y[dead]
    return float(np.sum(d * (yy - f_rr * r)) / np.sum(d * d))
