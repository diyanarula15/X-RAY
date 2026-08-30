"""X-RAY estimator: hidden energy state from a speed trace alone.

STRUCTURALLY BLIND. This module imports `constants`, `observe` and `track` --
regulation, the feed, and the circuit map. It does not import `sim`, `policy`
or `vehicle`, and it never sees an energy, throttle, brake or deployment
channel. `tests/test_estimator.py::test_estimator_is_blind` parses this file's
imports and fails the build if that ever changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from .constants import (
    E_HARVEST_LAP,
    E_STORE_MAX,
    FUEL_BURN_PER_LAP_KG,
    FUEL_START_KG,
    G,
    MASS_CAR_MIN,
    P_ICE_MAX,
    P_MGUK_MAX,
    TAPER_V_END,
    TAPER_V_START,
    p_mguk_ceiling,
)
from .observe import Observation

BRAKE_A_THRESHOLD = -5.0   # m/s^2; braking is unmistakable in a speed trace
CORNER_V_FRACTION = 0.93   # of the published apex limit
TAPER_WINDOW_MARGIN = 15 / 3.6  # m/s below TAPER_V_END
FIT_V_LO = 320 / 3.6       # bottom of the taper calibration band. The brief
                           # suggests 340 km/h, where deployment is "near zero";
                           # in practice the ceiling there (<=81 kW) is too small
                           # to separate from drag and the fit walks a ridge. At
                           # 320 km/h the ceiling still spans 188 kW down to 0
                           # across the band, which identifies the deployment
                           # share outright -- and above 320 km/h the ceiling has
                           # already fallen below any plausible zone deployment
                           # demand, so the car is either off the power or on the
                           # ceiling. Measured: 4.3% worst-case CdA error vs 9.5%.
PHI_PRIOR_SIGMA = 0.29     # std of Uniform[0,1]: we know the bounds, nothing more
TRAFFIC_GAP_S = 2.5        # a tow is still worth ~1% of drag area at this gap,
                           # so the calibration window rejects anything closer


class EstimatorError(RuntimeError):
    pass


@dataclass(frozen=True)
class PublicPriors:
    """Everything the estimator is allowed to assume it knows.

    Note what is *absent*: there is no CdA field. Drag area is the thing being
    identified. Air density is a weather observation; rolling resistance and
    drivetrain efficiency are assumed constants -- see the README.
    """
    mass_car: float = MASS_CAR_MIN
    fuel_start: float = FUEL_START_KG
    fuel_burn_per_lap: float = FUEL_BURN_PER_LAP_KG
    crr: float = 0.012
    rho: float = 1.20
    drivetrain_eff: float = 0.95


@dataclass(frozen=True)
class NuisanceFit:
    cda_hat: float
    cda_sigma: float
    v_wind_hat: float
    v_wind_sigma: float
    n_samples: int
    residual_rms: float
    deploy_frac_hat: float = 0.0   # fitted share of the taper ceiling still in use
    traffic_rejected: int = 0      # window samples dropped for running in a tow


@dataclass(frozen=True)
class BeliefTrace:
    t: np.ndarray
    soc_mean: np.ndarray
    soc_p10: np.ndarray
    soc_p90: np.ndarray
    deployed_lap: np.ndarray       # per-lap totals, J
    harvested_lap: np.ndarray
    nuisance: NuisanceFit
    lap_index: np.ndarray          # lap id for each entry of *_lap
    p_mguk_mean: np.ndarray = field(default=None)   # W, per sample
    harvest_mean: np.ndarray = field(default=None)  # W, per sample
    ess: np.ndarray = field(default=None)           # effective sample size per lap
    dry_events: np.ndarray = field(default=None)    # bool per sample


# ---------------------------------------------------------------- smoothing
def _dilate(mask: np.ndarray, half: int) -> np.ndarray:
    """Grow a boolean mask by ``half`` samples either side."""
    out = mask.copy()
    for k in range(1, half + 1):
        out[k:] |= mask[:-k]
        out[:-k] |= mask[k:]
    return out


def smooth_window(rate_hz: float, seconds: float = 0.8, polyorder: int = 2) -> int:
    w = int(round(seconds * rate_hz))
    if w % 2 == 0:
        w += 1
    return max(w, polyorder + 3)


def smooth_speed(obs: Observation, seconds: float = 0.8, polyorder: int = 2):
    """Savitzky-Golay smoothing and its analytic derivative.

    Never `np.diff` -- differentiating a quantised, noisy speed trace directly
    puts ~100 kW of noise on every power sample.
    """
    dt = 1.0 / obs.sample_rate_hz
    w = smooth_window(obs.sample_rate_hz, seconds, polyorder)
    w = min(w, len(obs.v) - 1 if len(obs.v) % 2 == 0 else len(obs.v))
    if w <= polyorder:
        raise EstimatorError("trace too short to smooth")
    v_s = savgol_filter(obs.v, w, polyorder, mode="interp")
    a_s = savgol_filter(obs.v, w, polyorder, deriv=1, delta=dt, mode="interp")
    return v_s, a_s


def _mass(obs: Observation, track, priors: PublicPriors) -> np.ndarray:
    lap_frac = obs.lap + np.clip(obs.s / track.length, 0.0, 1.0)
    fuel = np.maximum(priors.fuel_start - priors.fuel_burn_per_lap * lap_frac, 0.0)
    return priors.mass_car + fuel


# ----------------------------------------------------- Stage A: nuisance fit
def fit_nuisance(obs: Observation, track, priors: PublicPriors = PublicPriors(),
                 wind_prior_sigma: float = 1.0,
                 v_lo: float = FIT_V_LO) -> NuisanceFit:
    """Identify drag area and a wind offset in the high-speed window.

    High up the speed range the regulatory taper has squeezed deployment to a
    small, *known* ceiling, so the trace there is close to a pure ICE-versus-drag
    balance. What deployment is left is not assumed to be zero: the share of the
    ceiling still in use is carried as a fitted nuisance, identified by the fact
    that the ceiling falls steeply with speed across the band while drag climbs
    as v^3. See FIT_V_LO for why the band starts at 320 km/h.
    """
    v_s, a_s = smooth_speed(obs, )
    is_corner = track.is_corner(obs.s)
    fast = v_s > v_lo
    clear = np.isnan(obs.gap_to_leader) | (obs.gap_to_leader > TRAFFIC_GAP_S)
    guard = _dilate(a_s < BRAKE_A_THRESHOLD,
                    smooth_window(obs.sample_rate_hz) // 2 + 1)
    base = fast & (~is_corner) & (a_s > BRAKE_A_THRESHOLD) & (~guard)
    mask = base & clear
    traffic_rejected = int(np.sum(base & ~clear))

    n = int(mask.sum())
    if n < 30:
        raise EstimatorError(
            f"only {n} samples above {v_lo * 3.6:.0f} km/h "
            f"in clear air (need 30). The track has no straight long enough to open the "
            f"taper calibration window, or the car never deployed hard enough to use it.")

    v = v_s[mask]
    a = a_s[mask]
    m = _mass(obs, track, priors)[mask]
    grade = track.grade(obs.s[mask])
    ceiling = p_mguk_ceiling(v)
    eta, rho, crr = priors.drivetrain_eff, priors.rho, priors.crr

    def resid(theta, v_wind, sigma):
        cda, phi = theta
        p_wheel = (P_ICE_MAX + phi * ceiling) * eta
        drag = 0.5 * rho * cda * (v + v_wind) ** 2 * v
        r = m * a * v + drag + crr * m * G * v + m * G * np.sin(grade) * v - p_wheel
        return np.concatenate([r / sigma, [(phi - 0.5) / PHI_PRIOR_SIGMA]])

    # Profile over the wind offset rather than fitting it jointly: over an
    # 8%-wide speed window drag and wind are collinear, and a joint fit walks
    # the ridge to whichever bound the noise happens to favour. Marginalising
    # instead puts the ridge where it belongs -- in cda_sigma.
    winds = np.arange(-3.0, 3.0 + 1e-9, 0.5)
    cdas, phis, sigs, costs = [], [], [], []
    sigma = 1.0e5
    for _ in range(2):
        cdas, phis, sigs, costs = [], [], [], []
        for w in winds:
            sol = least_squares(resid, np.array([1.0, 0.5]), args=(w, sigma),
                                loss="soft_l1", f_scale=1.5,
                                bounds=([0.05, 0.0], [4.0, 1.0]))
            J = sol.jac
            try:
                cov = np.linalg.inv(J.T @ J)
                sig_c = float(np.sqrt(abs(cov[0, 0])))
            except np.linalg.LinAlgError:
                sig_c = np.inf
            cdas.append(sol.x[0]); phis.append(sol.x[1])
            sigs.append(sig_c); costs.append(2.0 * sol.cost)
        best = int(np.argmin(costs))
        sigma = float(np.sqrt(np.mean(resid([cdas[best], phis[best]], winds[best], 1.0)[:n] ** 2)))
    cdas = np.array(cdas); phis = np.array(phis)
    sigs = np.array(sigs); costs = np.array(costs)

    logp = -0.5 * costs - 0.5 * (winds / wind_prior_sigma) ** 2
    wgt = np.exp(logp - logp.max())
    wgt /= wgt.sum()

    cda_hat = float(np.sum(wgt * cdas))
    cda_var = float(np.sum(wgt * (cdas ** 2 + np.where(np.isfinite(sigs), sigs, 0.0) ** 2))
                    - cda_hat ** 2)
    v_wind_hat = float(np.sum(wgt * winds))
    v_wind_var = float(np.sum(wgt * winds ** 2) - v_wind_hat ** 2)
    phi_hat = float(np.sum(wgt * phis))

    return NuisanceFit(
        cda_hat=cda_hat, cda_sigma=float(np.sqrt(max(cda_var, 1e-12))),
        v_wind_hat=v_wind_hat, v_wind_sigma=float(np.sqrt(max(v_wind_var, 1e-12))),
        n_samples=n, residual_rms=float(sigma),
        deploy_frac_hat=phi_hat, traffic_rejected=traffic_rejected)


# --------------------------------------------- Stage B: power reconstruction
@dataclass(frozen=True)
class Kinematics:
    """Everything Stage B needs that does not depend on the nuisance draw."""
    v: np.ndarray
    a: np.ndarray
    s: np.ndarray
    lap: np.ndarray
    dt: float
    mass_nom: np.ndarray
    cda_scale: np.ndarray
    sin_grade: np.ndarray
    ceiling: np.ndarray
    brake: np.ndarray
    corner: np.ndarray
    accel: np.ndarray
    in_zone: np.ndarray
    below_taper: np.ndarray


def kinematics(obs: Observation, track, priors: PublicPriors = PublicPriors()) -> Kinematics:
    v_s, a_s = smooth_speed(obs)
    is_corner = track.is_corner(obs.s)
    v_lim = track.v_limit(obs.s)
    brake = a_s < BRAKE_A_THRESHOLD
    corner = is_corner & (v_s >= CORNER_V_FRACTION * v_lim) & ~brake
    accel = ~brake & ~corner
    in_zone = np.array([
        (z is not None and s_i < z.s_straight_end)
        for s_i, z in ((s_i, track.zone_at(s_i)) for s_i in obs.s)])
    return Kinematics(
        v=v_s, a=a_s, s=obs.s, lap=obs.lap, dt=1.0 / obs.sample_rate_hz,
        mass_nom=_mass(obs, track, priors), cda_scale=track.cda_scale(obs.s),
        sin_grade=np.sin(track.grade(obs.s)), ceiling=p_mguk_ceiling(v_s),
        brake=brake, corner=corner, accel=accel, in_zone=in_zone,
        below_taper=v_s < TAPER_V_START)


def powers(kin: Kinematics, cda, v_wind, mass_off, priors: PublicPriors,
           sl: slice = slice(None)):
    """Wheel power, deployment and recovery for a bundle of particles.

    ``cda``/``v_wind``/``mass_off`` are (Np,) arrays; the return values are
    (Np, n) arrays over the requested sample slice.
    """
    cda = np.atleast_1d(cda)[:, None]
    v_wind = np.atleast_1d(v_wind)[:, None]
    mass_off = np.atleast_1d(mass_off)[:, None]

    v = kin.v[sl][None, :]
    a = kin.a[sl][None, :]
    m = kin.mass_nom[sl][None, :] + mass_off
    cda_eff = cda * kin.cda_scale[sl][None, :]
    v_app = v + v_wind

    p_obs = (m * a * v
             + 0.5 * priors.rho * cda_eff * v_app * v_app * v
             + priors.crr * m * G * v
             + m * G * kin.sin_grade[sl][None, :] * v)

    eta = priors.drivetrain_eff
    # ICE runs wide open whenever the car is accelerating and not corner-limited
    mguk = np.clip(p_obs / eta - P_ICE_MAX, 0.0, kin.ceiling[sl][None, :])
    mguk = np.where(kin.accel[sl][None, :], mguk, 0.0)
    harv = np.where(kin.brake[sl][None, :], np.clip(-p_obs, 0.0, P_MGUK_MAX), 0.0)
    return p_obs, mguk, harv


def dry_events(kin: Kinematics, mguk_mean: np.ndarray,
               on_w: float = 1.5e5, off_w: float = 2.5e4) -> np.ndarray:
    """Samples where the trace says deployment *stopped* while the car was
    still accelerating below the taper, inside an overtake zone.

    This is the single most informative thing a speed trace contains about a
    hidden store: a car that could deploy and suddenly is not, is empty. It is
    what makes the belief band collapse on a straight rather than drifting.
    """
    out = np.zeros(len(mguk_mean), dtype=bool)
    armed = False
    prev_key = None
    for k in range(len(mguk_mean)):
        key = (kin.lap[k], kin.in_zone[k])
        if key != prev_key:
            armed = False
            prev_key = key
        if not (kin.in_zone[k] and kin.accel[k] and kin.below_taper[k]):
            continue
        if mguk_mean[k] > on_w:
            armed = True
        elif armed and mguk_mean[k] < off_w:
            out[k] = True
    return out


# ------------------------------------------- Stage C: SoC continuity + band
def _weighted_quantiles(vals: np.ndarray, w: np.ndarray, qs) -> list[np.ndarray]:
    order = np.argsort(vals, axis=0)
    vs = np.take_along_axis(vals, order, axis=0)
    ws = np.take_along_axis(w, order, axis=0)
    cum = np.cumsum(ws, axis=0)
    cum /= cum[-1:]
    cols = np.arange(vals.shape[1])
    return [vs[np.argmax(cum >= q, axis=0), cols] for q in qs]


def _systematic_resample(w: np.ndarray, rng) -> np.ndarray:
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(w / w.sum()), pos).clip(0, n - 1)


def estimate(obs: Observation, track, priors: PublicPriors = PublicPriors(),
             n_particles: int = 400, seed: int = 0,
             deploy_scale_sigma: float = 0.06, dry_event_e_scale: float = 4.0e5,
             floor_violation_penalty: float = 6.0,
             nuisance: NuisanceFit | None = None) -> BeliefTrace:
    """Observation -> BeliefTrace. The whole pipeline, Stages A through C."""
    rng = np.random.default_rng(seed)
    if nuisance is None:
        nuisance = fit_nuisance(obs, track, priors)
    kin = kinematics(obs, track, priors)
    n = len(obs.t)
    dt = kin.dt

    # the point-estimate reconstruction, used only to locate dry events
    _, mguk_pt, _ = powers(kin, nuisance.cda_hat, nuisance.v_wind_hat, 0.0, priors)
    dry = dry_events(kin, mguk_pt[0])

    Np = n_particles
    cda = np.clip(rng.normal(nuisance.cda_hat, max(nuisance.cda_sigma, 1e-3), Np), 0.05, 4.0)
    v_wind = rng.normal(nuisance.v_wind_hat, max(nuisance.v_wind_sigma, 1e-3), Np)
    mass_off = rng.normal(0.0, 3.0, Np)
    dep_scale = np.clip(rng.normal(1.0, deploy_scale_sigma, Np), 0.5, 1.5)
    E = rng.uniform(0.0, E_STORE_MAX, Np)

    soc_mean = np.empty(n)
    soc_p10 = np.empty(n)
    soc_p90 = np.empty(n)
    mguk_mean = np.empty(n)
    harv_mean = np.empty(n)

    laps = np.unique(obs.lap)
    dep_lap = np.zeros(len(laps))
    har_lap = np.zeros(len(laps))
    ess_lap = np.zeros(len(laps))

    for li, lap in enumerate(laps):
        idx = np.flatnonzero(obs.lap == lap)
        sl = slice(idx[0], idx[-1] + 1)
        _, mguk, harv = powers(kin, cda, v_wind, mass_off, priors, sl)
        mguk = mguk * dep_scale[:, None]
        nk = mguk.shape[1]

        Eh = np.empty((Np, nk))
        logw = np.zeros(Np)
        logW = np.empty((Np, nk))
        harv_cum = np.zeros(Np)
        dry_sl = dry[sl]
        for k in range(nk):
            d = mguk[:, k] * dt
            h = harv[:, k] * dt
            # a particle that says the car is deploying while its own store is
            # empty is describing something that cannot happen
            floor_hit = (E <= 1.0) & (d > 1.0)
            E = np.clip(E + h - d, 0.0, E_STORE_MAX)
            harv_cum += h
            logw = logw - floor_violation_penalty * floor_hit
            if dry_sl[k]:
                logw = logw - E / dry_event_e_scale
            Eh[:, k] = E
            logW[:, k] = logw

        # per-lap harvest cap
        logw = logw - 3.0 * np.maximum(harv_cum - E_HARVEST_LAP, 0.0) / 1.0e6
        logW[:, -1] = logw

        W = np.exp(logW - logW.max(axis=0, keepdims=True))
        W /= W.sum(axis=0, keepdims=True)
        soc_mean[sl] = np.sum(Eh * W, axis=0)
        soc_p10[sl], soc_p90[sl] = _weighted_quantiles(Eh, W, (0.10, 0.90))
        mguk_mean[sl] = np.sum(mguk * W, axis=0)
        harv_mean[sl] = np.sum(harv * W, axis=0)
        dep_lap[li] = float(np.sum(mguk_mean[sl]) * dt)
        har_lap[li] = float(np.sum(harv_mean[sl]) * dt)

        w = W[:, -1]
        ess_lap[li] = 1.0 / np.sum(w ** 2)
        take = _systematic_resample(w, rng)
        E, cda, v_wind, mass_off, dep_scale = (
            E[take], cda[take], v_wind[take], mass_off[take], dep_scale[take])
        # jitter the nuisance draws slightly so resampling cannot collapse the
        # parameter cloud to a single point over a long stint
        cda = cda + rng.normal(0.0, 0.15 * max(nuisance.cda_sigma, 1e-3), Np)
        dep_scale = np.clip(dep_scale + rng.normal(0.0, 0.15 * deploy_scale_sigma, Np),
                            0.5, 1.5)

    return BeliefTrace(
        t=obs.t.copy(), soc_mean=soc_mean, soc_p10=soc_p10, soc_p90=soc_p90,
        deployed_lap=dep_lap, harvested_lap=har_lap, nuisance=nuisance,
        lap_index=laps, p_mguk_mean=mguk_mean, harvest_mean=harv_mean,
        ess=ess_lap, dry_events=dry)
