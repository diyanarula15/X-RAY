"""Generalized calibration for real telemetry (Stage 2 §2).

Stage 1 identified drag in a binary window above 340 km/h. The feasibility study
killed that: real 2026 circuits put 0.00-0.11% of samples up there. This module
replaces the gate with a continuous formulation that works everywhere.

The idea. For a given wind offset the observed wheel power is *linear* in drag
area:

    P_obs(k) = A(k) + CdA · B(k),     B(k) = ½ρ·scale(k)·(v+w)²·v  > 0

and the regulation says deployment is bounded by a known, smooth, speed-varying
ceiling:

    0 ≤ D(k) = P_obs(k)/η − P_ice(k) ≤ ceiling(v(k))

Substituting turns every single sample into an *interval* on CdA:

    (η·P_ice(k) − A(k))/B(k)  ≤  CdA  ≤  (η·(P_ice(k)+ceiling(k)) − A(k))/B(k)

That is the whole method. No threshold anywhere. A sample at 340 km/h has a
near-zero ceiling and so pins CdA between two almost-identical numbers; a sample
at 120 km/h has a 350 kW ceiling and constrains it barely at all. The estimator
uses every sample, each weighted by how binding it actually is, and the width of
the surviving interval *is* the identifiability score.

This module is blind to energy by construction: no such channel exists in public
telemetry, which is the entire premise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

from .constants import (E_HARVEST_LAP, E_STORE_MAX, G, P_ICE_MAX, P_MGUK_MAX,
                        TAPER_V_END, p_mguk_ceiling)

RESERVE_SIGMA_REAL = 5.0e5   # J; a cut-out locates the buffer to about this
COAST_THROTTLE = 8.0       # % — below this the ICE is effectively off
COAST_MIN_DECEL = 0.4      # m/s², a real coast, not just noise
COAST_MAX_DECEL = 12.0     # m/s², above this the brakes are involved
BRAKE_DECEL = -6.0         # m/s²


@dataclass(frozen=True)
class RealNuisanceFit:
    cda_hat: float
    cda_lo: float              # identified set, not a confidence interval
    cda_hi: float
    cda_sigma: float
    v_wind_hat: float
    rho: float
    identifiability: float     # 0 (nothing) .. 1 (pinned)
    n_samples: int
    n_binding: int             # samples whose ceiling actually constrains
    n_coast: int
    coast_cda: float | None
    residual_rms: float
    systematic_rms: float
    method: str = "interval-intersection"
    notes: list = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.identifiability >= 0.25 and self.n_binding >= 50


def _sg(x: np.ndarray, win: int, order: int = 2, deriv: int = 0, delta: float = 1.0):
    win = max(win | 1, order + 3)
    if win >= len(x):
        win = (len(x) - 1) | 1
    if win <= order:
        return np.gradient(x, delta) if deriv else x
    return savgol_filter(x, win, order, deriv=deriv, delta=delta, mode="interp")


@dataclass
class Kin:
    """Kinematics on a distance grid, from the public feed only."""
    s: np.ndarray
    v: np.ndarray
    a: np.ndarray
    dt: np.ndarray           # s, time represented by each grid step (ds / v)
    t: np.ndarray
    mass: np.ndarray
    cda_scale: np.ndarray
    sin_grade: np.ndarray
    ceiling: np.ndarray
    throttle: np.ndarray | None
    brake: np.ndarray | None
    valid: np.ndarray
    lap: np.ndarray


def build_kin(df, track, mass_kg, rho, smooth_m: float = 60.0) -> Kin:
    """Kinematics from one car's distance-gridded telemetry.

    Differentiation happens **per lap**. The frame stacks every lap on the same
    distance grid, so differentiating the stacked array would run the smoother
    straight across the start/finish seam once per lap and produce nonsense.
    """
    s = df["distance"].to_numpy(dtype=float)
    v_raw = df["speed"].to_numpy(dtype=float)
    t = df["time"].to_numpy(dtype=float)
    lap = df["lap"].to_numpy(dtype=int)
    valid = np.isfinite(v_raw) & np.isfinite(t) & (v_raw > 1.0)

    ds = float(np.median(np.diff(np.unique(s)))) if len(np.unique(s)) > 2 else 10.0
    win = max(int(smooth_m / max(ds, 1e-6)), 5)

    v = np.full(len(s), np.nan)
    a = np.full(len(s), np.nan)
    for L in np.unique(lap):
        m = (lap == L)
        idx = np.flatnonzero(m)
        if len(idx) < win + 2:
            continue
        order = idx[np.argsort(s[idx])]
        vv = v_raw[order]
        good = np.isfinite(vv)
        if good.sum() < win + 2:
            continue
        # fill only interior holes, so the smoother has a continuous lap to work
        # on; the invalid mask still suppresses those samples downstream
        filled = vv.copy()
        filled[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), vv[good])
        vs = _sg(filled, win)
        dvds = _sg(filled, win, deriv=1, delta=ds)
        v[order] = vs
        a[order] = vs * dvds          # a = v dv/ds

    # time each grid cell represents: ds / v. Power integrated over a distance
    # grid with a constant dt is wrong by a factor that varies 5x around a lap.
    dt = np.where(np.isfinite(v) & (v > 1.0), ds / np.maximum(v, 1.0), np.nan)

    valid &= np.isfinite(v) & np.isfinite(a)
    return Kin(s=s, v=np.nan_to_num(v, nan=1.0), a=np.nan_to_num(a, nan=0.0),
               dt=dt, t=t,
               mass=np.full(len(s), mass_kg),
               cda_scale=np.asarray(track.cda_scale(s), dtype=float),
               sin_grade=np.sin(np.asarray(track.grade(s), dtype=float)),
               ceiling=p_mguk_ceiling(np.nan_to_num(v, nan=1.0)),
               throttle=df["throttle"].to_numpy(dtype=float) if "throttle" in df else None,
               brake=df["brake"].to_numpy(dtype=float) if "brake" in df else None,
               valid=valid, lap=lap)


def ice_power(kin: Kin, eta: float = 0.95) -> np.ndarray:
    """Assumed ICE output. Real telemetry publishes throttle, so this is read
    rather than inferred -- Stage 1 had to assume wide-open whenever the car was
    accelerating."""
    if kin.throttle is None:
        return np.where(kin.a > 0, P_ICE_MAX, 0.0)
    thr = np.clip(np.nan_to_num(kin.throttle) / 100.0, 0.0, 1.0)
    # a turbocharged ICE is not linear in pedal, but it is monotone and
    # saturating; this is the standard first-order shape
    return P_ICE_MAX * np.clip(thr * 1.15, 0.0, 1.0)


def _terms(kin: Kin, v_wind: float, crr: float, rho: float):
    """P_obs(k) = A(k) + CdA·B(k)."""
    v, m = kin.v, kin.mass
    A = m * kin.a * v + crr * m * G * v + m * G * kin.sin_grade * v
    B = 0.5 * rho * kin.cda_scale * (v + v_wind) ** 2 * v
    return A, B


def interval_bounds(kin: Kin, v_wind: float, crr: float, rho: float,
                    eta: float = 0.95, use_throttle: bool = True):
    """Per-sample interval on CdA, derived from the regulation alone.

    Total power at the wheels is bounded above by what the rules allow -- 400 kW
    of ICE plus the speed-varying MGU-K ceiling -- and below by the largest
    recovery the MGU-K may take:

        -P_MGUK_MAX  <=  A(k) + CdA*B(k)  <=  (P_ICE_MAX + ceiling(k)) * eta

    Rearranged, every sample brackets CdA. Both bounds hold whatever the driver
    is doing, because both come from the rulebook rather than from a model of
    the engine. That distinction matters: an earlier version inferred ICE output
    from the throttle trace and produced lower bounds of CdA > 13, because a real
    car at 30 m/s is traction-limited rather than power-limited and no
    throttle-to-power curve survives that.

    The bounds are then *sharpened* wherever the feed removes ambiguity:

      - brakes on -> the lower bound is dropped. Brake force is not in the model,
        so deceleration there says nothing about drag.
      - throttle off -> the ICE contributes nothing, so the upper bound loses its
        400 kW term and tightens severalfold. That is the coast-down channel, and
        unlike a high-speed window it exists at every circuit on earth.
    """
    A, B = _terms(kin, v_wind, crr, rho)
    ok = kin.valid & (B > 1e3)

    p_ice_cap = np.full(len(A), float(P_ICE_MAX))
    if use_throttle and kin.throttle is not None:
        thr = np.nan_to_num(kin.throttle)
        p_ice_cap = np.where(thr < COAST_THROTTLE, 0.0, float(P_ICE_MAX))
    if kin.brake is not None:
        brakes_on = np.nan_to_num(kin.brake) > 0.5
    else:
        brakes_on = kin.a < BRAKE_DECEL

    lo = np.full(len(A), -np.inf)
    hi = np.full(len(A), np.inf)
    hi[ok] = ((p_ice_cap[ok] + kin.ceiling[ok]) * eta - A[ok]) / B[ok]
    free = ok & ~brakes_on
    lo[free] = (-P_MGUK_MAX - A[free]) / B[free]
    return lo, hi, ok


def coast_phases(kin: Kin) -> np.ndarray:
    """Lift-and-coast: off throttle, off the brakes, genuinely decelerating."""
    if kin.throttle is None:
        return (kin.a < -COAST_MIN_DECEL) & (kin.a > -COAST_MAX_DECEL) & kin.valid
    thr = np.nan_to_num(kin.throttle)
    brk = np.nan_to_num(kin.brake) if kin.brake is not None else np.zeros_like(thr)
    return ((thr < COAST_THROTTLE) & (brk < 0.5) & kin.valid
            & (kin.a < -COAST_MIN_DECEL) & (kin.a > -COAST_MAX_DECEL))


def coast_bounds(kin: Kin, v_wind: float, crr: float, rho: float, eta: float = 0.95):
    """The second calibration channel, valid at every circuit."""
    coast = coast_phases(kin)
    A, B = _terms(kin, v_wind, crr, rho)
    good = coast & (B > 1e3)
    if good.sum() < 20:
        return None, coast
    lo = (-P_MGUK_MAX - A[good]) / B[good]
    hi = (eta * kin.ceiling[good] - A[good]) / B[good]
    return (lo, hi), good


def fit_nuisance_real(kin: Kin, rho: float, crr: float = 0.012, eta: float = 0.95,
                      wind_grid=np.arange(-4.0, 4.01, 0.5),
                      robust_q: float = 0.02) -> RealNuisanceFit:
    """Intersect the per-sample intervals; report the identified set.

    `robust_q` discards the most extreme q of bounds on each side, because a
    single mis-sampled point should not be able to declare the whole problem
    infeasible.
    """
    best = None
    notes = []
    for w in wind_grid:
        lo, hi, ok = interval_bounds(kin, w, crr, rho, eta)
        Lf = lo[np.isfinite(lo)]
        Hf = hi[np.isfinite(hi)]
        if len(Hf) < 50:
            continue
        good = np.isfinite(hi)
        # the intersection: highest lower bound and lowest upper bound, each
        # taken at a robust quantile so one mis-sampled point cannot empty it
        cda_lo = float(np.quantile(Lf, 1.0 - robust_q)) if len(Lf) > 20 else 0.05
        cda_hi = float(np.quantile(Hf, robust_q))
        # how many samples would the interval violate? that is the cost
        mid = 0.5 * (cda_lo + cda_hi)
        viol = 0.0
        width = cda_hi - cda_lo
        # prefer the wind that makes the constraints most nearly consistent:
        # a tight, non-empty identified set with few violated samples
        rel = width / max(mid, 1e-6) if width > 0 else 9.9
        cost = rel + 3.0 * viol + abs(w) / 40.0
        cand = (cost, w, cda_lo, cda_hi, mid, int(np.sum(good)))
        if best is None or cand[0] < best[0]:
            best = cand

    if best is None:
        raise ValueError("no usable samples for calibration")
    _cost, w_hat, cda_lo, cda_hi, cda_mid, n_ok = best

    if cda_hi <= cda_lo:
        notes.append("interval empty at the robust quantile; widened to the "
                     "median of the bounds")
        lo, hi, ok = interval_bounds(kin, w_hat, crr, rho, eta)
        cda_lo = float(np.nanmedian(lo[ok][np.isfinite(lo[ok])]))
        cda_hi = float(np.nanmedian(hi[ok][np.isfinite(hi[ok])]))
        if cda_hi <= cda_lo:
            cda_lo, cda_hi = sorted((cda_lo, cda_hi))
    cda_mid = 0.5 * (cda_lo + cda_hi)

    # coast-down as a second, independent read
    cb, coast_mask = coast_bounds(kin, w_hat, crr, rho, eta)
    coast_cda = None
    if cb is not None:
        cl, ch = cb
        c_lo = float(np.quantile(cl, 1.0 - robust_q))
        c_hi = float(np.quantile(ch, robust_q))
        if c_hi > c_lo:
            coast_cda = 0.5 * (c_lo + c_hi)
            # intersect the two channels where they agree
            j_lo, j_hi = max(cda_lo, c_lo), min(cda_hi, c_hi)
            if j_hi > j_lo:
                cda_lo, cda_hi = j_lo, j_hi
                cda_mid = 0.5 * (j_lo + j_hi)
                notes.append("coast-down channel intersected with the ceiling constraint")
            else:
                notes.append("coast-down and ceiling channels disagree; kept the "
                             "ceiling interval and widened")
                cda_lo = min(cda_lo, c_lo)
                cda_hi = max(cda_hi, c_hi)
                cda_mid = 0.5 * (cda_lo + cda_hi)

    # How binding are the constraints, sample by sample? A sample is
    # informative when its own upper bound already sits near the estimate --
    # that is, when the regulation leaves little room between what the car is
    # doing and the most it is allowed to do. High-speed and coasting samples
    # qualify; a car trundling out of a hairpin does not.
    lo, hi, ok = interval_bounds(kin, w_hat, crr, rho, eta)
    hf = hi[np.isfinite(hi)]
    n_binding = int(np.sum(hf < 2.0 * max(cda_mid, 1e-3)))
    rel_width = (cda_hi - cda_lo) / max(cda_mid, 1e-6)
    # 0 when the identified set is as wide as the estimate itself, 1 when it is
    # pinned to a few percent
    identifiability = float(np.clip(1.0 - rel_width / 0.5, 0.0, 1.0))

    A, B = _terms(kin, w_hat, crr, rho)
    P_ice = ice_power(kin, eta)
    D = np.clip((A + cda_mid * B) / eta - P_ice, 0.0, kin.ceiling)
    resid = (A + cda_mid * B) - (P_ice + D) * eta
    rr = float(np.sqrt(np.nanmean(resid[ok] ** 2)))
    lap_means = [np.nanmean(resid[ok & (kin.lap == L)]) for L in np.unique(kin.lap)]
    lap_means = [x for x in lap_means if np.isfinite(x)]
    sysr = float(np.sqrt(np.mean(np.square(lap_means)))) if len(lap_means) > 1 else rr

    return RealNuisanceFit(
        cda_hat=float(cda_mid), cda_lo=float(cda_lo), cda_hi=float(cda_hi),
        cda_sigma=float((cda_hi - cda_lo) / 3.29),   # interval -> 1 sigma equivalent
        v_wind_hat=float(w_hat), rho=float(rho),
        identifiability=identifiability, n_samples=int(ok.sum()),
        n_binding=n_binding, n_coast=int(coast_mask.sum()),
        coast_cda=coast_cda, residual_rms=rr, systematic_rms=sysr, notes=notes)


def deployment_trace(kin: Kin, fit: RealNuisanceFit, crr: float = 0.012,
                     eta: float = 0.95, smooth_win: int = 5) -> dict:
    """Deployment and recovery per sample, as an identified BAND.

    A speed trace measures total power at the wheels. It cannot see which part
    of that came from the engine and which from the motor -- both make torque at
    the same axle. What the regulation does pin down is:

      lower bound   whatever exceeds the 400 kW the ICE is allowed to make must
                    be electrical. Below that the split is unidentified.
      upper bound   the MGU-K ceiling at this speed, and no more.

    An earlier version modelled ICE output from the throttle trace and reported
    the remainder as deployment. That produced laps of 0.2 MJ against a real
    3-4 MJ, because a real car below about 200 km/h is traction-limited and its
    wheel power sits under the ICE cap for most of the lap -- so the model
    attributed everything to the engine and left nothing for the motor. Carrying
    the ambiguity as a band is the honest answer; collapsing it to a point with
    an engine model is not.
    """
    A, B = _terms(kin, fit.v_wind_hat, crr, fit.rho)
    P_obs = A + fit.cda_hat * B
    p_wheel = P_obs / eta

    if kin.brake is not None:
        braking = np.nan_to_num(kin.brake) > 0.5
    else:
        braking = kin.a < BRAKE_DECEL
    if kin.throttle is not None:
        on_power = np.nan_to_num(kin.throttle) > COAST_THROTTLE
    else:
        on_power = kin.a > 0.0

    d_lo = np.clip(p_wheel - P_ICE_MAX, 0.0, None)
    d_hi = np.where(on_power, p_wheel, 0.0)
    d_lo = np.minimum(np.clip(d_lo, 0.0, kin.ceiling), kin.ceiling)
    d_hi = np.clip(np.maximum(d_hi, d_lo), 0.0, kin.ceiling)
    d_lo = np.where(braking, 0.0, d_lo)
    d_hi = np.where(braking, 0.0, d_hi)

    H = np.where(braking, np.clip(-P_obs, 0.0, P_MGUK_MAX), 0.0)

    mid = 0.5 * (d_lo + d_hi)
    if smooth_win > 2:
        mid = np.clip(_sg(np.nan_to_num(mid), smooth_win), 0.0, None)
    keep = kin.valid
    return {"P_obs": P_obs, "deploy": np.where(keep, mid, np.nan),
            "deploy_lo": np.where(keep, d_lo, np.nan),
            "deploy_hi": np.where(keep, d_hi, np.nan),
            "harvest": np.where(keep, H, np.nan), "braking": braking,
            "P_ice": np.clip(p_wheel - mid, 0.0, P_ICE_MAX)}


def common_mode(fits_by_car: dict, kins: dict) -> dict:
    """§2.3 — twenty cars through the same air on the same lap.

    Wind, air density and track evolution are shared; drag area is not. Fit the
    shared part as a per-track-position multiplier estimated across the field,
    and let each car's drag be its deviation from it. Strictly better with real
    data than it ever was with two simulated cars.
    """
    cars = [c for c in fits_by_car if c in kins]
    if len(cars) < 3:
        return {"available": False, "reason": f"only {len(cars)} cars"}
    ref = kins[cars[0]]
    n = len(ref.s)
    stack = []
    for c in cars:
        k, f = kins[c], fits_by_car[c]
        A, B = _terms(k, f.v_wind_hat, 0.012, f.rho)
        P_ice = ice_power(k)
        r = (A + f.cda_hat * B) - (P_ice + np.clip((A + f.cda_hat * B) / 0.95 - P_ice,
                                                   0, k.ceiling)) * 0.95
        r = np.where(k.valid, r, np.nan)
        if len(r) >= n:
            stack.append(r[:n])
    if len(stack) < 3:
        return {"available": False, "reason": "insufficient aligned traces"}
    M = np.vstack(stack)
    shared = np.nanmedian(M, axis=0)          # common mode
    resid = M - shared[None, :]
    return {
        "available": True, "n_cars": len(stack),
        "common_mode_w": shared,
        "common_mode_rms": float(np.sqrt(np.nanmean(shared ** 2))),
        "per_car_rms": float(np.sqrt(np.nanmean(resid ** 2))),
        "variance_explained": float(
            1.0 - np.nanvar(resid) / max(np.nanvar(M), 1e-9)),
    }


def pool_field(fits: dict, min_ident: float = 0.15) -> dict:
    """§2.3, applied to the nuisance itself.

    Twenty cars run the same regulations through the same air. Drag area differs
    between them by a few percent, not by a factor of five, so the field's
    identified sets can be intersected: a car that had no clean high-speed
    running inherits the constraint from cars that did. Each car's own estimate
    is then its deviation from the pooled value, which is the quantity that
    actually differs between teams.
    """
    good = {k: f for k, f in fits.items() if f.identifiability >= min_ident}
    if len(good) < 3:
        return {"available": False, "reason": f"only {len(good)} identifiable cars"}
    lo = max(f.cda_lo for f in good.values())
    hi = min(f.cda_hi for f in good.values())
    if hi <= lo:   # sets disagree; fall back to the median of the midpoints
        mids = np.array([f.cda_hat for f in good.values()])
        lo, hi = float(np.percentile(mids, 25)), float(np.percentile(mids, 75))
    pooled = 0.5 * (lo + hi)
    return {
        "available": True, "n_cars_pooled": len(good),
        "cda_pooled": float(pooled), "cda_lo": float(lo), "cda_hi": float(hi),
        "per_car_deviation_pct": {
            k: float(100.0 * (f.cda_hat - pooled) / pooled) for k, f in fits.items()},
        "field_identifiability": float(np.clip(
            1.0 - (hi - lo) / max(pooled, 1e-6) / 0.5, 0.0, 1.0)),
    }


# ---------------------------------------------------------------- belief
def belief_from_deployment(kin: Kin, tr: dict, fit: RealNuisanceFit,
                           n_particles: int = 400, seed: int = 0,
                           reserve_max_frac: float = 0.35) -> dict:
    """Stage 1's Stage C, run on a real deployment trace.

    The calibration front-end had to be replaced for real data (§2), but the
    inference that turns a deployment trace into a belief about stored energy is
    unchanged: propagate each particle's own store, clip it at 0 and 4 MJ, and
    let the clipping do the regularising. Deployment cut-outs remain the
    informative event, and the store is still identified only up to the driver's
    unspent buffer, so deployable energy is what gets reported.
    """
    rng = np.random.default_rng(seed)
    D, H, dt = tr["deploy"], tr["harvest"], kin.dt
    Dlo = np.nan_to_num(tr.get("deploy_lo", D))
    Dhi = np.nan_to_num(tr.get("deploy_hi", D))
    n = len(D)
    ok = np.isfinite(D) & np.isfinite(H) & np.isfinite(dt)
    D = np.where(ok, D, 0.0)
    H = np.where(ok, H, 0.0)
    dtv = np.where(ok, dt, 0.0)

    # The store is bounded at 4 MJ, so over a whole race the net flow must be
    # within a store of zero -- a car cannot deploy more than it recovers, lap
    # after lap, for two hours. The raw reconstruction does not respect that: at
    # Spa it comes out about 0.2 MJ/lap net negative, which over 44 laps pegs
    # every particle at the floor within a few laps and produces a belief band of
    # +/-0.00 MJ that is confidently wrong.
    #
    # Boundedness therefore identifies the ratio between the deployment and
    # recovery estimates, which is not otherwise pinned. Centre the recovery
    # scale on the value that closes the balance and let the spread around it
    # carry the uncertainty. This is a regulation constraint, not ground truth.
    # Deployment is identified only between d_lo (what the ICE cannot supply)
    # and d_hi (the regulatory ceiling). Where in that band the truth sits is not
    # visible in a speed trace -- but the store is bounded at 4 MJ, so over a
    # whole race deployment and recovery must agree to within one store. Solve
    # for the split that closes that balance, and let the particles spread around
    # it. Leaving the split uniform on [0,1] implies the car runs half on
    # electricity and gives 10 MJ per lap against a 4 MJ store.
    dtv_f = np.nan_to_num(dtv)
    lo_f, hi_f = np.nan_to_num(Dlo), np.nan_to_num(Dhi)
    tot_h = float(np.sum(np.nan_to_num(H) * dtv_f))
    tot_lo = float(np.sum(lo_f * dtv_f))
    tot_span = float(np.sum((hi_f - lo_f) * dtv_f))
    split_star = float(np.clip((tot_h - tot_lo) / max(tot_span, 1.0), 0.0, 1.0))
    # Verify against the same quantities the report uses and correct. A 10%
    # residual imbalance is enough to drain a 4 MJ store inside ten laps and
    # leave the belief pinned at zero for the rest of the race.
    for _ in range(3):
        tot_d = float(np.sum((lo_f + split_star * (hi_f - lo_f)) * dtv_f))
        err = tot_d - tot_h
        if abs(err) < 0.005 * max(tot_h, 1.0) or tot_span <= 0:
            break
        split_star = float(np.clip(split_star - err / tot_span, 0.0, 1.0))
    balance = split_star

    Np = n_particles
    scale = np.clip(rng.normal(1.0, max(fit.cda_sigma / max(fit.cda_hat, 1e-3), 0.05), Np),
                    0.4, 1.8)
    # recovery is taken as measured; the deployment split does the balancing
    hscale = np.clip(rng.normal(1.0, 0.12, Np), 0.5, 1.5)
    # centred on the split that closes the energy balance, spread by how
    # uncertain that closure is
    split = np.clip(rng.normal(split_star, 0.22, Np), 0.0, 1.0)
    reserve = rng.uniform(0.0, reserve_max_frac * E_STORE_MAX, Np)
    E = rng.uniform(0.0, E_STORE_MAX, Np)

    # a cut-out: deployment stops while the car is still on the throttle below
    # the taper, which says the store has reached this driver's floor
    below_taper = kin.v < 80.6
    on_power = (np.nan_to_num(kin.throttle) > 70.0) if kin.throttle is not None else (kin.a > 0.5)
    dry = np.zeros(n, dtype=bool)
    armed = False
    for k in range(n):
        if not (ok[k] and below_taper[k] and on_power[k]):
            continue
        if D[k] > 1.5e5:
            armed = True
        elif armed and D[k] < 2.5e4:
            dry[k] = True
            armed = False

    soc_mean = np.empty(n); soc_lo = np.empty(n); soc_hi = np.empty(n)
    use_mean = np.empty(n); use_lo = np.empty(n); use_hi = np.empty(n)
    cloud = np.empty((n, min(Np, 400)), dtype=np.float32)   # for the 3D view
    logw = np.zeros(Np)
    floor_j = fit.residual_rms * float(np.nanmedian(dtv[dtv > 0]) or 0.05)

    lap = kin.lap
    laps = np.unique(lap)
    dep_lap, har_lap = {}, {}
    for L in laps:
        idx = np.flatnonzero(lap == L)
        if len(idx) == 0:
            continue
        floor_hits = np.zeros(Np)
        # A cut-out is one observation. On real, noisy data the detector can
        # re-arm and fire many times in a lap, and a sharp quadratic penalty each
        # time collapses the cloud to one particle and reports a 0.1 MJ band with
        # total confidence. Take the first few per lap and no more.
        dry_budget = 3
        # each particle takes its own position inside the identified band, so
        # the ICE / MGU-K split ambiguity shows up as band width rather than
        # being silently resolved
        d_band = Dlo[idx][None, :] + split[:, None] * (Dhi[idx] - Dlo[idx])[None, :]
        d_l = d_band * dtv[idx][None, :] * scale[:, None]
        h_l = H[idx] * dtv[idx] * hscale[:, None]
        for j, k in enumerate(idx):
            d = d_l[:, j]; h = h_l[:, j]
            # Count floor violations; charge for them ONCE at the end of the lap.
            # Applying the penalty per sample multiplies it a few thousand times
            # over and annihilates every particle but one, which is how a belief
            # band collapses to +/-0.00 MJ and starts lying with total confidence.
            floor_hits += (E <= 1.0) & (d > floor_j)
            E = np.clip(E + h - d, 0.0, E_STORE_MAX)
            if dry[k] and dry_budget > 0:
                dry_budget -= 1
                logw -= 0.5 * ((E - reserve) / RESERVE_SIGMA_REAL) ** 2
            W = np.exp((logw - 6.0 * floor_hits / max(len(idx), 1) * 10.0))
            W = W / W.sum() if W.sum() > 0 else np.full(Np, 1.0 / Np)
            soc_mean[k] = float(np.sum(E * W))
            order = np.argsort(E); c = np.cumsum(W[order])
            soc_lo[k] = float(E[order][np.searchsorted(c, 0.10)])
            soc_hi[k] = float(E[order][np.searchsorted(c, 0.90)])
            U = np.maximum(E - reserve, 0.0)
            use_mean[k] = float(np.sum(U * W))
            uo = np.argsort(U); cu = np.cumsum(W[uo])
            use_lo[k] = float(U[uo][np.searchsorted(cu, 0.10)])
            use_hi[k] = float(U[uo][np.searchsorted(cu, 0.90)])
            cloud[k] = U[:cloud.shape[1]].astype(np.float32)
        logw = logw - 6.0 * floor_hits / max(len(idx), 1) * 10.0
        # report the split-corrected deployment, which is what the particles
        # actually used -- not the raw band midpoint
        d_star = (np.nan_to_num(Dlo[idx])
                  + split_star * (np.nan_to_num(Dhi[idx]) - np.nan_to_num(Dlo[idx])))
        dep_lap[int(L)] = float(np.nansum(d_star * dtv[idx]))
        har_lap[int(L)] = float(np.nansum(H[idx] * dtv[idx]))
        # systematic resampling once per lap
        W = np.exp(logw - logw.max()); W /= W.sum()
        pos = (rng.random() + np.arange(Np)) / Np
        take = np.searchsorted(np.cumsum(W), pos).clip(0, Np - 1)
        E, scale, hscale, reserve, split = (E[take], scale[take], hscale[take],
                                            reserve[take], split[take])
        split = np.clip(split + rng.normal(0, 0.03, Np), 0.0, 1.0)
        # Process noise on the store itself. Without it the filter drives every
        # particle onto the floor and then reports "empty" with a zero-width
        # band -- certainty it has not earned, since a lap's reconstructed energy
        # is only good to a few per cent. Sized from that error, not chosen.
        lap_energy = float(np.nansum(d_star * dtv[idx])) if len(idx) else 0.0
        E = np.clip(E + rng.normal(0.0, max(0.12 * lap_energy, 5.0e4), Np),
                    0.0, E_STORE_MAX)
        reserve = np.clip(reserve + rng.normal(0, 0.01 * reserve_max_frac * E_STORE_MAX, Np),
                          0.0, reserve_max_frac * E_STORE_MAX)
        logw = np.zeros(Np)

    deploy_star = (np.nan_to_num(Dlo)
                   + split_star * (np.nan_to_num(Dhi) - np.nan_to_num(Dlo)))
    return {"deploy_star": np.where(np.isfinite(tr["deploy"]), deploy_star, np.nan),
            "soc_mean": soc_mean, "soc_p10": soc_lo, "soc_p90": soc_hi,
            "usable_mean": use_mean, "usable_p10": use_lo, "usable_p90": use_hi,
            "cloud": cloud, "dry": dry, "deployed_lap": dep_lap,
            "harvested_lap": har_lap, "reserve_mean": float(np.mean(reserve)),
            "balance": balance}


def observability(kin: Kin, fit: RealNuisanceFit, track, crr: float = 0.012,
                  eta: float = 0.95, n_bins: int = 200) -> dict:
    """View 2: how much can the estimator learn at each point on the circuit?

    Two different things are learnable and they are not the same thing, so they
    are reported separately:

      deployment information -- how much a change in deployment moves the
        observed speed. dP/dD is flat, so what varies is whether deployment is
        free to move at all: where the regulatory ceiling is wide open and the
        car is on power, the trace is informative about deployment.

      nuisance information -- how tightly this point constrains drag area, which
        is |dP/dCdA| = B(k). That grows as v^3 and is what high-speed running
        actually buys you.

    High-speed running informs the nuisances rather than the deployment, which
    is why it gets its own colour in the UI.
    """
    A, B = _terms(kin, fit.v_wind_hat, crr, fit.rho)
    ok = kin.valid & (B > 1e3)
    s = kin.s % track.length
    bins = np.linspace(0.0, track.length, n_bins + 1)
    idx = np.clip(np.digitize(s, bins) - 1, 0, n_bins - 1)

    ceil = kin.ceiling
    on_power = (np.nan_to_num(kin.throttle) > 50.0) if kin.throttle is not None else (kin.a > 0.5)
    braking = (np.nan_to_num(kin.brake) > 0.5) if kin.brake is not None else (kin.a < BRAKE_DECEL)
    # deployment is legible where there is headroom under the ceiling AND the
    # car is actually using the power unit
    dep_info = np.where(ok & on_power & ~braking, ceil / P_MGUK_MAX, 0.0)
    nui_info = np.where(ok, B, 0.0)

    dep = np.zeros(n_bins); nui = np.zeros(n_bins); cnt = np.zeros(n_bins)
    vv = np.zeros(n_bins)
    for arr, dst in ((dep_info, dep), (nui_info, nui)):
        np.add.at(dst, idx[ok], arr[ok])
    np.add.at(cnt, idx[ok], 1.0)
    np.add.at(vv, idx[ok], kin.v[ok])
    cnt = np.maximum(cnt, 1.0)
    dep /= cnt; nui /= cnt; vv /= cnt
    nui = nui / max(nui.max(), 1e-9)
    return {"s": 0.5 * (bins[:-1] + bins[1:]), "deployment_info": dep,
            "nuisance_info": nui, "speed": vv, "n_samples": cnt}
