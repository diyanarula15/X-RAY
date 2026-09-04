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
    """Deployment and recovery, per sample, from the calibrated model."""
    A, B = _terms(kin, fit.v_wind_hat, crr, fit.rho)
    P_obs = A + fit.cda_hat * B
    P_ice = ice_power(kin, eta)
    D = np.clip(P_obs / eta - P_ice, 0.0, kin.ceiling)
    if kin.brake is not None:
        braking = np.nan_to_num(kin.brake) > 0.5
    else:
        braking = kin.a < BRAKE_DECEL
    D = np.where(braking, 0.0, D)
    H = np.where(braking, np.clip(-P_obs, 0.0, P_MGUK_MAX), 0.0)
    D = np.where(kin.valid, D, np.nan)
    H = np.where(kin.valid, H, np.nan)
    if smooth_win > 2:
        D = _sg(np.nan_to_num(D), smooth_win) * np.where(kin.valid, 1.0, np.nan)
        D = np.clip(D, 0.0, None)
    return {"P_obs": P_obs, "deploy": D, "harvest": H, "braking": braking,
            "P_ice": P_ice}


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
    n = len(D)
    ok = np.isfinite(D) & np.isfinite(H) & np.isfinite(dt)
    D = np.where(ok, D, 0.0)
    H = np.where(ok, H, 0.0)
    dtv = np.where(ok, dt, 0.0)

    Np = n_particles
    scale = np.clip(rng.normal(1.0, max(fit.cda_sigma / max(fit.cda_hat, 1e-3), 0.05), Np),
                    0.4, 1.8)
    hscale = np.clip(rng.normal(1.0, 0.12, Np), 0.5, 1.5)
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
        d_l = D[idx] * dtv[idx] * scale[:, None]
        h_l = H[idx] * dtv[idx] * hscale[:, None]
        for j, k in enumerate(idx):
            d = d_l[:, j]; h = h_l[:, j]
            logw -= 6.0 * ((E <= 1.0) & (d > floor_j))
            E = np.clip(E + h - d, 0.0, E_STORE_MAX)
            if dry[k]:
                logw -= 0.5 * ((E - reserve) / 2.5e5) ** 2
            W = np.exp(logw - logw.max()); W /= W.sum()
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
        dep_lap[int(L)] = float(np.nansum(D[idx] * dtv[idx]))
        har_lap[int(L)] = float(np.nansum(H[idx] * dtv[idx]))
        # systematic resampling once per lap
        W = np.exp(logw - logw.max()); W /= W.sum()
        pos = (rng.random() + np.arange(Np)) / Np
        take = np.searchsorted(np.cumsum(W), pos).clip(0, Np - 1)
        E, scale, hscale, reserve = E[take], scale[take], hscale[take], reserve[take]
        reserve = np.clip(reserve + rng.normal(0, 0.01 * reserve_max_frac * E_STORE_MAX, Np),
                          0.0, reserve_max_frac * E_STORE_MAX)
        logw = np.zeros(Np)

    return {"soc_mean": soc_mean, "soc_p10": soc_lo, "soc_p90": soc_hi,
            "usable_mean": use_mean, "usable_p10": use_lo, "usable_p90": use_hi,
            "cloud": cloud, "dry": dry, "deployed_lap": dep_lap,
            "harvested_lap": har_lap, "reserve_mean": float(np.mean(reserve))}


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
