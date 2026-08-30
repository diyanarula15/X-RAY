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
RESERVE_MAX_FRAC = 0.35    # racing prior: a driver holds back at most about a
                           # third of the store as buffer. The actual buffer is
                           # inferred per particle -- it is a policy parameter.
RESERVE_SIGMA = 2.5e5      # J, how tightly a cut-out pins the store to the buffer
RESERVE_RELEASE_LAPS = 3.0  # a buffer held all stint is spent over the last few
                            # laps. That release is what makes the buffer -- and
                            # so the absolute level of the store -- observable at
                            # all: the level a driver cuts out at drops towards
                            # zero as the stint ends, and by how much it drops is
                            # how big the buffer was. Without this the store is
                            # identifiable only up to an unknown constant.
TOW_TAU_S = 0.8            # public assumption: how fast a tow decays with gap
TOW_K_PRIOR = (0.22, 0.09)  # mean, sigma of the drag-area fraction recovered
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
    systematic_rms: float = 0.0    # W, lap-to-lap drift of the model error


@dataclass(frozen=True)
class BeliefTrace:
    t: np.ndarray
    soc_mean: np.ndarray       # J, belief about the raw store
    soc_p10: np.ndarray
    soc_p90: np.ndarray
    usable_mean: np.ndarray    # J, belief about energy the rival can actually
    usable_p10: np.ndarray     # spend: store minus the buffer they hold back.
    usable_p90: np.ndarray     # This is the identified quantity -- see README.
    deployed_lap: np.ndarray       # per-lap totals, J
    harvested_lap: np.ndarray
    nuisance: NuisanceFit
    deploy_scale_sigma: float
    lap_index: np.ndarray          # lap id for each entry of *_lap
    p_mguk_mean: np.ndarray = field(default=None)   # W, per sample
    harvest_mean: np.ndarray = field(default=None)  # W, per sample
    ess: np.ndarray = field(default=None)           # effective sample size per lap
    dry_events: np.ndarray = field(default=None)    # bool per sample
    reserve_mean: float = 0.0                       # J, inferred driver buffer
    reserve_sigma: float = 0.0


# ---------------------------------------------------------------- smoothing
def _dilate(mask: np.ndarray, half: int) -> np.ndarray:
    """Grow a boolean mask by ``half`` samples either side."""
    out = mask.copy()
    for k in range(1, half + 1):
        out[k:] |= mask[:-k]
        out[:-k] |= mask[k:]
    return out


def _erode(mask: np.ndarray, half: int) -> np.ndarray:
    out = mask.copy()
    for k in range(1, half + 1):
        out[k:] &= mask[:-k]
        out[:-k] &= mask[k:]
    return out


def brake_mask(a_s: np.ndarray) -> np.ndarray:
    """Braking samples, with the smoothing smear taken back off.

    Smoothing widens a braking event by about one window; thresholding at half
    of each event's own peak deceleration recovers the true duration, and
    duration is what recovered energy is: the 350 kW cap binds throughout, so
    harvest per event is simply the cap times how long the car was on the
    brakes.
    """
    cand = a_s < BRAKE_A_THRESHOLD
    out = np.zeros_like(cand)
    k = 0
    n = len(cand)
    while k < n:
        if not cand[k]:
            k += 1
            continue
        j = k
        while j < n and cand[j]:
            j += 1
        peak = a_s[k:j].min()
        thr = min(0.5 * peak, BRAKE_A_THRESHOLD)
        out[k:j] = a_s[k:j] < thr
        k = j
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
    v_wind_mode = float(winds[int(np.argmax(wgt))])

    # Refinement. Modelling the leftover deployment as one average fraction of
    # the ceiling is a mean-field approximation, and it is wrong in a specific
    # way: deployment in this band is binary, either off or hard against the
    # ceiling. Classify each sample instead and re-solve for drag with the
    # deployment known. Two or three passes converge, and it takes the CdA bias
    # from -1.8% to under 0.1%.
    cda_em, phi_em = cda_hat, float(np.sum(wgt * phis))
    for _ in range(3):
        p_obs = (m * a * v + 0.5 * rho * cda_em * (v + v_wind_mode) ** 2 * v
                 + crr * m * G * v + m * G * np.sin(grade) * v)
        deploying = (p_obs / eta - P_ICE_MAX) > 0.5 * ceiling
        p_mg = np.where(deploying, ceiling, 0.0)
        target = ((P_ICE_MAX + p_mg) * eta - m * a * v - crr * m * G * v
                  - m * G * np.sin(grade) * v)
        basis = 0.5 * rho * (v + v_wind_mode) ** 2 * v
        cda_em = float(np.sum(target * basis) / np.sum(basis * basis))
        phi_em = float(deploying.mean())
    cda_hat = float(np.clip(cda_em, 0.05, 4.0))

    # Two different error scales matter for two different things. `residual_rms`
    # is per-sample and mostly white -- it averages away over a lap. What makes
    # two particles' stored energy diverge over a stint is the part of the model
    # error that does *not* average away: the lap-to-lap drift of the mean
    # residual. Measure that separately; it sets the belief band's width.
    p_obs = (m * a * v + 0.5 * rho * cda_hat * (v + v_wind_mode) ** 2 * v
             + crr * m * G * v + m * G * np.sin(grade) * v)
    resid_w = p_obs - (P_ICE_MAX + np.where(deploying, ceiling, 0.0)) * eta
    lap_ids = obs.lap[mask]
    lap_means = np.array([resid_w[lap_ids == L].mean() for L in np.unique(lap_ids)
                          if np.sum(lap_ids == L) >= 5])
    systematic_rms = float(np.sqrt(np.mean(lap_means ** 2))) if len(lap_means) >= 2 else sigma
    cda_var = float(np.sum(wgt * (cdas ** 2 + np.where(np.isfinite(sigs), sigs, 0.0) ** 2))
                    - cda_hat ** 2)
    v_wind_hat = float(np.sum(wgt * winds))
    v_wind_var = float(np.sum(wgt * winds ** 2) - v_wind_hat ** 2)
    phi_hat = float(np.sum(wgt * phis))

    return NuisanceFit(
        cda_hat=cda_hat, cda_sigma=float(np.sqrt(max(cda_var, 1e-12))),
        v_wind_hat=v_wind_hat, v_wind_sigma=float(np.sqrt(max(v_wind_var, 1e-12))),
        n_samples=n, residual_rms=float(sigma),
        deploy_frac_hat=phi_em, traffic_rejected=traffic_rejected,
        systematic_rms=systematic_rms)


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
    tow_decay: np.ndarray   # exp(-gap/tau); 0 in clear air
    brake: np.ndarray
    corner: np.ndarray
    accel: np.ndarray
    in_zone: np.ndarray
    below_taper: np.ndarray


def kinematics(obs: Observation, track, priors: PublicPriors = PublicPriors()) -> Kinematics:
    v_s, a_s = smooth_speed(obs)
    is_corner = track.is_corner(obs.s)
    v_lim = track.v_limit(obs.s)
    brake = brake_mask(a_s)
    corner = is_corner & (v_s >= CORNER_V_FRACTION * v_lim) & ~brake
    accel = ~brake & ~corner
    in_zone = np.array([
        (z is not None and s_i < z.s_straight_end)
        for s_i, z in ((s_i, track.zone_at(s_i)) for s_i in obs.s)])
    return Kinematics(
        v=v_s, a=a_s, s=obs.s, lap=obs.lap, dt=1.0 / obs.sample_rate_hz,
        mass_nom=_mass(obs, track, priors), cda_scale=track.cda_scale(obs.s),
        sin_grade=np.sin(track.grade(obs.s)), ceiling=p_mguk_ceiling(v_s),
        tow_decay=np.where(np.isnan(obs.gap_to_leader), 0.0,
                           np.exp(-np.maximum(obs.gap_to_leader, 0.0) / TOW_TAU_S)),
        brake=brake, corner=corner, accel=accel, in_zone=in_zone,
        below_taper=v_s < TAPER_V_START)


def powers(kin: Kinematics, cda, v_wind, mass_off, priors: PublicPriors,
           sl: slice = slice(None), tow_k=0.0):
    """Wheel power, deployment and recovery for a bundle of particles.

    ``cda``/``v_wind``/``mass_off``/``tow_k`` are (Np,) arrays; the return
    values are (Np, n) arrays over the requested sample slice. ``tow_k`` is the
    fraction of drag area a car gets back when it is running in another car's
    wake -- the feed publishes the gap, so the estimator knows when to apply it
    even though it does not know how strong the effect is.
    """
    cda = np.atleast_1d(cda)[:, None]
    v_wind = np.atleast_1d(v_wind)[:, None]
    mass_off = np.atleast_1d(mass_off)[:, None]
    tow_k = np.atleast_1d(tow_k)[:, None]

    v = kin.v[sl][None, :]
    a = kin.a[sl][None, :]
    m = kin.mass_nom[sl][None, :] + mass_off
    cda_eff = (cda * kin.cda_scale[sl][None, :]
               * (1.0 - tow_k * kin.tow_decay[sl][None, :]))
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
            armed = False   # one episode is one observation, not one per sample
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
             deploy_scale_sigma: float | None = None, dry_event_e_scale: float = RESERVE_SIGMA,
             floor_violation_penalty: float = 6.0,
             nuisance: NuisanceFit | None = None) -> BeliefTrace:
    """Observation -> BeliefTrace. The whole pipeline, Stages A through C."""
    rng = np.random.default_rng(seed)
    if nuisance is None:
        nuisance = fit_nuisance(obs, track, priors)
    kin = kinematics(obs, track, priors)
    n = len(obs.t)
    dt = kin.dt

    # the point-estimate reconstruction: used to report per-lap flows, to locate
    # dry events, and to size the belief band
    _, mguk_pt, harv_pt = powers(kin, nuisance.cda_hat, nuisance.v_wind_hat, 0.0,
                                 priors, tow_k=TOW_K_PRIOR[0])
    mguk_pt, harv_pt = mguk_pt[0], harv_pt[0]
    dry = dry_events(kin, mguk_pt)

    # How wrong is the reconstructed per-lap energy likely to be? The estimator
    # answers that from its own Stage A residuals, with no reference to truth.
    # Two scales, and they matter at different sample rates: the per-sample
    # residual is largely white and integrates up as sqrt(N) over a lap, while
    # the lap-to-lap drift of the mean residual integrates up as N. At 100 Hz
    # the drift dominates; at 3.7 Hz the white term does. The belief band's
    # width is a consequence of this number, not a setting.
    n_laps_seen = max(len(np.unique(obs.lap)), 1)
    if deploy_scale_sigma is None:
        n_accel = max(float(np.sum(kin.accel)) / n_laps_seen, 1.0)
        white = nuisance.residual_rms * dt * np.sqrt(n_accel)
        syst = nuisance.systematic_rms * dt * n_accel
        dep_ref = max(float(np.sum(mguk_pt) * dt) / n_laps_seen, 1.0e5)
        deploy_scale_sigma = float(np.clip(np.hypot(white, syst) / dep_ref, 0.02, 0.40))

    # A particle sitting on the floor while the reconstruction shows a trickle
    # of deployment is not describing an impossibility -- it is describing our
    # own noise. Only deployment above the reconstruction's residual scale
    # counts as a genuine floor violation.
    floor_deploy_j = nuisance.residual_rms * dt

    Np = n_particles
    cda = np.clip(rng.normal(nuisance.cda_hat, max(nuisance.cda_sigma, 1e-3), Np), 0.05, 4.0)
    v_wind = rng.normal(nuisance.v_wind_hat, max(nuisance.v_wind_sigma, 1e-3), Np)
    mass_off = rng.normal(0.0, 3.0, Np)
    dep_scale = np.clip(rng.normal(1.0, deploy_scale_sigma, Np), 0.4, 1.6)
    har_scale = np.clip(rng.normal(1.0, deploy_scale_sigma, Np), 0.4, 1.6)
    tow_k = np.clip(rng.normal(*TOW_K_PRIOR, Np), 0.0, 0.5)
    # the rival's held-back buffer: unknown, and the thing a deployment cut-out
    # actually reveals. Inferring it is inferring one of the four policy
    # parameters, which is what decision.py later samples over.
    reserve = rng.uniform(0.0, RESERVE_MAX_FRAC * E_STORE_MAX, Np)
    E = rng.uniform(0.0, E_STORE_MAX, Np)

    soc_mean = np.empty(n)
    soc_p10 = np.empty(n)
    soc_p90 = np.empty(n)
    use_mean = np.empty(n)
    use_p10 = np.empty(n)
    use_p90 = np.empty(n)
    mguk_mean = np.empty(n)
    harv_mean = np.empty(n)

    laps = np.unique(obs.lap)
    n_laps_total = int(laps.max()) + 1
    dep_lap = np.zeros(len(laps))
    har_lap = np.zeros(len(laps))
    ess_lap = np.zeros(len(laps))

    for li, lap in enumerate(laps):
        idx = np.flatnonzero(obs.lap == lap)
        sl = slice(idx[0], idx[-1] + 1)
        _, mguk, harv = powers(kin, cda, v_wind, mass_off, priors, sl, tow_k)
        mguk = mguk * dep_scale[:, None]
        harv = harv * har_scale[:, None]
        nk = mguk.shape[1]

        Eh = np.empty((Np, nk))
        logw = np.zeros(Np)
        logW = np.empty((Np, nk))
        harv_cum = np.zeros(Np)
        floor_frac = np.zeros(Np)
        ceil_frac = np.zeros(Np)
        dry_sl = dry[sl]
        laps_left = max(n_laps_total - int(lap), 0)
        reserve_eff = reserve * min(1.0, laps_left / RESERVE_RELEASE_LAPS)
        for k in range(nk):
            d = mguk[:, k] * dt
            h = harv[:, k] * dt
            # a particle claiming the car is deploying out of an empty store is
            # describing something that cannot happen; count how often, and pay
            # for it once at the end of the lap rather than at every sample
            floor_frac += (E <= 1.0) & (d > floor_deploy_j)
            # the mirror constraint, and the only thing that bounds the belief
            # from above: a particle pinned at the ceiling is claiming the team
            # threw recovered energy away lap after lap. Possible, but a team
            # that did that would not be in this fight, so it is penalised --
            # softly, because unlike the floor it is not impossible.
            ceil_frac += (E >= E_STORE_MAX - 1.0) & (h > floor_deploy_j)
            E = np.clip(E + h - d, 0.0, E_STORE_MAX)
            harv_cum += h
            if dry_sl[k]:
                # deployment stopped while the car could still have used it.
                # That does not mean the store is empty -- it means the store
                # has reached whatever buffer this driver refuses to spend.
                logw = logw - 0.5 * ((E - reserve_eff) / dry_event_e_scale) ** 2
            Eh[:, k] = E
            logW[:, k] = logw
        logw = logw - floor_violation_penalty * floor_frac / max(nk, 1) * 10.0
        logw = logw - 0.35 * floor_violation_penalty * ceil_frac / max(nk, 1) * 10.0
        logW[:, -1] = logw

        # per-lap harvest cap
        logw = logw - 3.0 * np.maximum(harv_cum - E_HARVEST_LAP, 0.0) / 1.0e6
        logW[:, -1] = logw

        W = np.exp(logW - logW.max(axis=0, keepdims=True))
        W /= W.sum(axis=0, keepdims=True)
        soc_mean[sl] = np.sum(Eh * W, axis=0)
        soc_p10[sl], soc_p90[sl] = _weighted_quantiles(Eh, W, (0.10, 0.90))
        Uh = np.maximum(Eh - reserve_eff[:, None], 0.0)
        use_mean[sl] = np.sum(Uh * W, axis=0)
        use_p10[sl], use_p90[sl] = _weighted_quantiles(Uh, W, (0.10, 0.90))
        mguk_mean[sl] = np.sum(mguk * W, axis=0)
        harv_mean[sl] = np.sum(harv * W, axis=0)
        # Per-lap flows are reported from the point estimate, not from the
        # particle cloud. Two reasons: the SoC weights carry information about
        # the store's absolute *level*, which is only weakly identified (see the
        # README on the reserve degeneracy) and would import that ambiguity into
        # a quantity the power balance measures directly; and clipping
        # deployment at zero is convex, so averaging it over a spread of drag
        # draws biases the total upward.
        dep_lap[li] = float(np.sum(mguk_pt[sl]) * dt)
        har_lap[li] = float(np.sum(harv_pt[sl]) * dt)

        w = W[:, -1]
        ess_lap[li] = 1.0 / np.sum(w ** 2)
        take = _systematic_resample(w, rng)
        E, cda, v_wind, mass_off, dep_scale, har_scale, tow_k, reserve = (
            E[take], cda[take], v_wind[take], mass_off[take], dep_scale[take],
            har_scale[take], tow_k[take], reserve[take])
        reserve = np.clip(reserve + rng.normal(0.0, 0.05 * RESERVE_MAX_FRAC * E_STORE_MAX, Np),
                          0.0, RESERVE_MAX_FRAC * E_STORE_MAX)
        # jitter the nuisance draws slightly so resampling cannot collapse the
        # parameter cloud to a single point over a long stint
        cda = cda + rng.normal(0.0, 0.15 * max(nuisance.cda_sigma, 1e-3), Np)
        dep_scale = np.clip(dep_scale + rng.normal(0.0, 0.15 * deploy_scale_sigma, Np),
                            0.4, 1.6)
        har_scale = np.clip(har_scale + rng.normal(0.0, 0.15 * deploy_scale_sigma, Np),
                            0.4, 1.6)

    return BeliefTrace(
        t=obs.t.copy(), soc_mean=soc_mean, soc_p10=soc_p10, soc_p90=soc_p90,
        usable_mean=use_mean, usable_p10=use_p10, usable_p90=use_p90,
        deployed_lap=dep_lap, harvested_lap=har_lap, nuisance=nuisance,
        deploy_scale_sigma=deploy_scale_sigma, lap_index=laps, p_mguk_mean=mguk_mean, harvest_mean=harv_mean,
        ess=ess_lap, dry_events=dry,
        reserve_mean=float(np.mean(reserve)), reserve_sigma=float(np.std(reserve)))
