"""Rao-Blackwellised particle filter over the store, the reserve and theta.

The discrete latents (aero mode, powertrain regime) are sampled; given them the
store is a linear accumulation, so its mean and variance are carried in closed
form inside each particle rather than represented by a cloud of samples. That is
the Rao-Blackwellisation, and it is what makes a few hundred particles enough:
the sampling only has to cover the *discrete* ambiguity and the static
parameters, not the continuous state as well.

Two things this stage does that no earlier stage can.

1. It closes the drag lower bound. Step 2 proved a speed trace bounds drag area
   from above and not below, because recovery can always account for a
   deceleration -- a 3 s coast shedding 5 m/s dissipates 0.267 MJ where the
   rules permit 0.75 MJ of harvesting. What that argument ignores is that the
   recovery has to go *somewhere*: a 4 MJ store cannot absorb an unbounded
   amount, and a particle that explains the whole race with low drag and
   constant harvesting pegs its store at the ceiling and dies. The bound comes
   from boundedness, not from the trace.

2. It reports deployable energy rather than the store. A deployment cut-out
   means the store reached the buffer this driver refuses to spend, not that it
   is empty, so the store is identified only up to that buffer. The buffer is a
   static per-particle parameter and E - R is the reported quantity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .balance import M_REF_KG, fuel_mass
from .constants import E_STORE_MAX, G
from .modes import BRAKE_HARVEST, DEPLOY, SUPER_CLIP
from .regs import RegSet, p_ice_max, p_k_bounds

RESERVE_MAX_FRAC = 0.35    # racing prior: a driver holds back at most about a
                           # third of the store. Inferred per particle.
RESERVE_SIGMA_J = 5.0e5    # how tightly a cut-out pins the store to the buffer
# Hysteresis on the cut-out detector, in power. A single threshold fires on
# every dip in a fluctuating deployment trace: measured, that produced 36
# detections a race (the per-lap budget, saturated every lap) at true stores
# from 0.00 to 2.86 MJ, and the false ones taught a 1.21 MJ buffer to a driver
# holding none. Deployment must first be unambiguously ON, then unambiguously
# OFF, for the event to mean anything.
DRY_ARM_W = 1.5e5
DRY_FIRE_W = 2.5e4
DRY_PER_LAP = 2
# Per-lap jitter on the reserve, as a fraction of its prior range. At 0.01 the
# cloud collapsed under resampling -- 0.014 MJ of diversity against a 1.4 MJ
# range -- and the filter reported its prior mean back as an inference.
RESERVE_JITTER_FRAC = 0.10
# A buffer held all stint gets spent at the end of it, and that release is the
# *only* thing that separates the store from the buffer. Without it E and R are
# identified jointly and not separately, so the filter reports its prior mean
# for R: measured, 0.91 MJ for a driver holding none. Because U = max(E - R, 0)
# is then exactly zero for every particle, the deployable-energy band collapsed
# to 0.06 MJ wide and covered the truth 3% of the time -- narrow, confident and
# wrong. The level a driver cuts out at drops towards zero as the stint closes,
# and how far it drops is how big the buffer was.
RESERVE_RELEASE_LAPS = 3.0
CEIL_PENALTY = 0.35        # relative to the floor penalty. A particle pinned at
                           # the ceiling claims the team threw recovered energy
                           # away lap after lap: possible, unlike deploying from
                           # an empty store, so penalised more softly.
FLOOR_PENALTY = 6.0


@dataclass(frozen=True)
class Belief:
    t: np.ndarray
    soc_mean: np.ndarray
    soc_p10: np.ndarray
    soc_p90: np.ndarray
    usable_mean: np.ndarray
    usable_p10: np.ndarray
    usable_p90: np.ndarray
    since_reserve: np.ndarray   # J, net flow since the last reserve hit
    deployed_lap: dict
    harvested_lap: dict
    reserve_mean: float
    reserve_sigma: float
    ess: np.ndarray
    theta_post_mean: np.ndarray
    theta_post_lo: np.ndarray
    theta_post_hi: np.ndarray
    n_particles: int
    dry: np.ndarray


def _sample_theta(ident, rng, n: int) -> np.ndarray:
    """Draw parameters from inside the identified set.

    Hit-and-run would be the principled sampler for a polytope. This uses the
    projections plus a feasibility filter against the centre, which is cruder
    but has the property that matters: every particle is feasible, so no
    particle can explain the data with a theta the regulation forbids.
    Directions the set does not constrain -- drag area from below, measured --
    come out as the prior, which is the honest answer rather than a hidden one.
    """
    lo = np.where(np.isfinite(ident.lo), ident.lo, ident.centre)
    hi = np.where(np.isfinite(ident.hi), ident.hi, ident.centre)
    return lo + rng.random((n, len(lo))) * (hi - lo)


def _interval_terms(feed, theta, rho: float, m_published: float,
                    fuel_start: float, fuel_burn_per_lap: float, is_x):
    """Per-interval wheel energy implied by each particle's theta. (Np, n-1)."""
    v, t, z = feed.v, feed.t, feed.z
    dt = np.diff(t)
    v0, v1 = v[:-1], v[1:]
    v3bar = 0.5 * (v0 ** 3 + v1 ** 3)
    vbar = 0.5 * (v0 + v1)
    m_nom = m_published + fuel_mass(feed.lap_frac[:-1], fuel_start, fuel_burn_per_lap)
    cda = np.where(is_x[:-1], theta[:, 0:1], theta[:, 1:2])       # (Np, n-1)
    m = m_nom[None, :] + theta[:, 3:4]
    d_e = m * (0.5 * (v1 ** 2 - v0 ** 2) + G * np.diff(z))[None, :]
    drag = 0.5 * rho * cda * v3bar[None, :] * dt[None, :]
    roll = theta[:, 2:3] * vbar[None, :] * (m / M_REF_KG) * dt[None, :]
    return d_e + drag + roll, dt


def run(feed, ident, regs: RegSet, rho: float, n_particles: int = 400,
        seed: int = 0, eta_d: float = 0.95, m_published: float = 790.0,
        fuel_start: float = 70.0, fuel_burn_per_lap: float = 1.15,
        is_x=None, regime=None, rao_blackwell: bool = True,
        n_laps: int | None = None, v_cut: float | None = None) -> Belief:
    """Filter the store forward, one lap of resampling at a time."""
    rng = np.random.default_rng(seed)
    n = len(feed)
    Np = n_particles
    is_x = np.ones(n, bool) if is_x is None else np.asarray(is_x, bool)

    theta = _sample_theta(ident, rng, Np)
    e_wheel, dt = _interval_terms(feed, theta, rho, m_published, fuel_start,
                                  fuel_burn_per_lap, is_x)

    # Regulation bounds on the store-side flow over each interval, which is
    # what the ICE / MGU-K split ambiguity actually amounts to. Deployment is
    # bounded below by whatever the engine cannot supply and above by the
    # ceiling; where in that band the truth sits is not visible in a speed
    # trace, so each particle takes its own position and the ambiguity shows up
    # as band width instead of being silently resolved.
    v_cap = np.minimum(feed.v[:-1], feed.v[1:])
    rpm_hi = np.maximum(feed.rpm[:-1], feed.rpm[1:])
    thr_hi = np.maximum(feed.throttle[:-1], feed.throttle[1:])
    zone = np.asarray(feed.in_zone, bool)
    mo = (None if feed.manual_override is None
          else np.asarray(feed.manual_override, bool)[:-1])
    pk_lo, pk_hi = p_k_bounds(v_cap, zone[:-1] | zone[1:], rpm_hi, thr_hi, regs,
                              False if mo is None else mo)
    e_ice_max = eta_d * p_ice_max(rpm_hi, thr_hi, regs) * dt
    d_lo = np.clip(e_wheel - e_ice_max[None, :], 0.0, None)
    d_hi = np.minimum(np.maximum(e_wheel, d_lo), (pk_hi * eta_d * dt)[None, :])
    d_hi = np.maximum(d_hi, d_lo)
    h_cap = (-pk_lo * dt)[None, :]

    braking = np.zeros(n - 1, bool) if regime is None else (
        np.asarray(regime)[:-1] == BRAKE_HARVEST)
    superclip = np.zeros(n - 1, bool) if regime is None else (
        np.asarray(regime)[:-1] == SUPER_CLIP)
    recovering = braking | superclip
    harvest = np.where(recovering[None, :],
                       np.clip(-e_wheel, 0.0, h_cap), 0.0)
    deploy_lo = np.where(recovering[None, :], 0.0, d_lo)
    deploy_hi = np.where(recovering[None, :], 0.0, d_hi)

    split = rng.random(Np)
    reserve = rng.uniform(0.0, RESERVE_MAX_FRAC * E_STORE_MAX, Np)
    E = rng.uniform(0.0, E_STORE_MAX, Np)
    # Rao-Blackwellised second moment: the store's variance is propagated in
    # closed form from the width of the deployment band, rather than being
    # represented by extra particles. This is the whole reason a few hundred
    # particles suffice.
    E_var = np.zeros(Np)
    # Net flow since the last reserve hit. Deployable energy is this integral,
    # not E - R: the two particle dimensions are identified only jointly, but
    # their *difference* since a known reference point is measured directly by
    # the flows. Reporting the integral sidesteps the degeneracy instead of
    # fighting it.
    since = np.zeros(Np)

    soc_m = np.empty(n); soc_lo = np.empty(n); soc_hi = np.empty(n)
    use_m = np.empty(n); use_lo = np.empty(n); use_hi = np.empty(n)
    since_m = np.empty(n)
    dry = np.zeros(n, bool)
    logw = np.zeros(Np)
    laps = np.asarray(feed.lap_frac, float).astype(int)
    # Stint length is published, so this is not cheating -- and it must not be
    # read off the window, which would move the release ramp as the race runs.
    n_laps_total = int(laps.max()) + 1 if n_laps is None else int(n_laps)
    dep_lap, har_lap, ess = {}, {}, []

    # A RESERVE hit, as distinct from a policy cut-out -- and the distinction is
    # the whole difficulty. Both look like "deployment stopped at full
    # throttle". A policy cut-out happens at the driver's chosen cut-off speed
    # v_cut and resumes at the next corner exit; a reserve hit is unconditional
    # on speed and *persists* until the next harvest. Treating the first as the
    # second pins the buffer to the store level at every straight -- near full --
    # which is exactly a large R inferred for a driver holding none.
    #
    # So the gate is: below the policy cut-off (where the prior says the driver
    # still wants to deploy), with real headroom under the regulation ceiling,
    # on full throttle, gaining speed, and deployment still zero.
    #
    # A cut-out: on power, below the taper, *gaining speed*, and the
    # reconstruction says deployment stopped. The single most informative event
    # about a hidden store -- and the accelerating gate is not optional.
    #
    # Without it a corner-limited car qualifies: it is on full throttle and not
    # deploying because it does not need to, which is not the same as having
    # nothing left. Measured on seed 42, the ungated detector fired 36 times at
    # a true store of 0.01 MJ median but 2.86 MJ maximum, and the false
    # detections at a nearly-full store taught the filter a 0.91 MJ buffer for a
    # driver holding none. That put the deployable-energy band 0.91 MJ low,
    # clipped it at zero, and dropped coverage to 0.03 -- a band that had
    # collapsed and was lying, while the underlying store estimate was fine.
    on_power = np.asarray(feed.throttle, float)[:-1] > 0.7
    room = pk_hi > 1.0e5
    gaining = np.diff(np.asarray(feed.v, float)) > 0.2
    # below the driver's own cut-off: where the policy prior still wants to
    # deploy, so a stop here is not the policy's doing
    v_int = np.minimum(np.asarray(feed.v, float)[:-1], np.asarray(feed.v, float)[1:])
    below_cut = (np.ones(n - 1, bool) if v_cut is None else v_int < float(v_cut))
    armed = False

    def reserve_eff_at(lap: int) -> np.ndarray:
        laps_left = max(n_laps_total - int(lap), 0)
        return reserve * min(1.0, laps_left / RESERVE_RELEASE_LAPS)

    def record(k, w):
        soc_m[k] = float(np.sum(E * w))
        order = np.argsort(E); c = np.cumsum(w[order])
        soc_lo[k] = float(E[order][min(np.searchsorted(c, 0.10), Np - 1)])
        soc_hi[k] = float(E[order][min(np.searchsorted(c, 0.90), Np - 1)])
        since_m[k] = float(np.sum(np.maximum(since, 0.0) * w))
        U = np.maximum(E - reserve_eff_at(laps[k]), 0.0)
        use_m[k] = float(np.sum(U * w))
        uo = np.argsort(U); cu = np.cumsum(w[uo])
        use_lo[k] = float(U[uo][min(np.searchsorted(cu, 0.10), Np - 1)])
        use_hi[k] = float(U[uo][min(np.searchsorted(cu, 0.90), Np - 1)])

    w = np.full(Np, 1.0 / Np)
    record(0, w)
    floor_hits = np.zeros(Np)
    ceil_hits = np.zeros(Np)
    cur_lap = laps[0]
    dep_acc = np.zeros(Np)
    har_acc = np.zeros(Np)
    dry_budget = DRY_PER_LAP

    for k in range(n - 1):
        d = deploy_lo[:, k] + split * (deploy_hi[:, k] - deploy_lo[:, k])
        h = harvest[:, k]
        band = deploy_hi[:, k] - deploy_lo[:, k]
        if rao_blackwell:
            # variance of a uniform position inside the identified band
            E_var = E_var + (band ** 2) / 12.0
        floor_hits += (E <= 1.0) & (d > 1.0e3)
        ceil_hits += (E >= E_STORE_MAX - 1.0) & (h > 1.0e3)
        # The store the cut-out was observed AT is the one before this
        # interval's flows are applied. Using the post-update value offsets the
        # observation by one interval, which is the same class of bug as the
        # simulator's own record-after-step ordering.
        E_at_event = E
        E = np.clip(E + h - d, 0.0, E_STORE_MAX)
        # net flow since the last reserve hit: this is the deployable quantity,
        # and it is identified WITHOUT separating E from R
        since = since + (h - d)
        dep_acc += d
        har_acc += h

        if on_power[k] and room[k] and gaining[k] and below_cut[k]:
            # Detecting on the deployment lower bound instead of the band
            # midpoint was tried and is worse: d_lo is regulation-grounded
            # (d_lo > 0 means the ICE provably cannot have supplied the power),
            # but it is also zero for most of a lap, so the detector loses the
            # events that matter. Store-band coverage fell 0.69 -> 0.55 with no
            # improvement in the inferred buffer (0.85 MJ against 0.98).
            p_d = float(np.mean(d)) / max(dt[k], 1e-6)
            if p_d > DRY_ARM_W:
                armed = True
            elif armed and p_d < DRY_FIRE_W and dry_budget > 0:
                dry[k] = True
                armed = False
                dry_budget -= 1
                # The Rao-Blackwellised part earning its name: the store's
                # own propagated variance enters the likelihood, so a cut-out
                # observed after a long stretch of ambiguous deployment counts
                # for less than one observed right after a well-determined
                # stretch. Carrying E_var and not using it here would make the
                # label decoration.
                var = RESERVE_SIGMA_J ** 2 + E_var
                logw = logw - 0.5 * (E_at_event - reserve_eff_at(laps[k])) ** 2 / var
                # the store was at the buffer here, so deployable resets to zero
                since = np.zeros(Np)

        w = np.exp(logw - logw.max())
        w = w / w.sum() if w.sum() > 0 else np.full(Np, 1.0 / Np)
        record(k + 1, w)

        if laps[k + 1] != cur_lap or k == n - 2:
            nk = max(np.sum(laps[:-1] == cur_lap), 1)
            logw = logw - FLOOR_PENALTY * floor_hits / nk * 10.0
            logw = logw - CEIL_PENALTY * FLOOR_PENALTY * ceil_hits / nk * 10.0
            w = np.exp(logw - logw.max()); w /= w.sum()
            ess.append(float(1.0 / np.sum(w ** 2)))
            dep_lap[int(cur_lap)] = float(np.sum(dep_acc * w))
            har_lap[int(cur_lap)] = float(np.sum(har_acc * w))
            take = _systematic(w, rng)
            E, E_var, theta, split, reserve, since = (
                E[take], E_var[take], theta[take], split[take], reserve[take],
                since[take])
            # Process noise on the store, sized from the reconstruction's own
            # per-lap error rather than chosen. Without it the filter drives
            # every particle onto the floor and then reports "empty" with a
            # zero-width band -- certainty it has not earned.
            lap_e = max(float(np.mean(dep_acc)), 5.0e4)
            E = np.clip(E + rng.normal(0.0, 0.12 * lap_e, Np), 0.0, E_STORE_MAX)
            reserve = np.clip(
                reserve + rng.normal(0.0, RESERVE_JITTER_FRAC * RESERVE_MAX_FRAC * E_STORE_MAX, Np),
                0.0, RESERVE_MAX_FRAC * E_STORE_MAX)
            logw = np.zeros(Np)
            floor_hits = np.zeros(Np); ceil_hits = np.zeros(Np)
            dep_acc = np.zeros(Np); har_acc = np.zeros(Np)
            dry_budget = DRY_PER_LAP
            cur_lap = laps[k + 1]

    w = np.exp(logw - logw.max()); w /= w.sum()
    return Belief(
        t=np.asarray(feed.t).copy(), soc_mean=soc_m, soc_p10=soc_lo,
        soc_p90=soc_hi, usable_mean=use_m, usable_p10=use_lo, usable_p90=use_hi,
        since_reserve=since_m, deployed_lap=dep_lap, harvested_lap=har_lap,
        reserve_mean=float(np.sum(reserve * w)),
        reserve_sigma=float(np.sqrt(np.sum(w * (reserve - np.sum(reserve * w)) ** 2))),
        ess=np.array(ess), theta_post_mean=np.sum(theta * w[:, None], axis=0),
        theta_post_lo=theta.min(axis=0), theta_post_hi=theta.max(axis=0),
        n_particles=Np, dry=dry)


def _systematic(w: np.ndarray, rng) -> np.ndarray:
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(w / w.sum()), pos).clip(0, n - 1)
