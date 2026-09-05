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
FLOOR_PENALTY = 6.0        # retained only for the historical note above

# Priors on the three policy parameters, in m/s. Wide: they are a *proposal*
# for where inside the deployment band P_K sits, not a claim about the driver.
V_CUT_PRIOR = (45.0, 100.0)
V_HARV_PRIOR = (70.0, 105.0)
WIDTH_PRIOR = (2.0, 9.0)

# Store closure per lap. A car cannot deploy materially more than it recovers
# lap after lap on a 4 MJ store; the Manual Override allowance is 0.5 MJ, so the
# scale is one allowance plus room for the store to breathe.
CLOSURE_SIGMA_J = 1.0e6

# Bisection on the policy prior's amplitude. 24 steps on [0, LAMBDA_MAX] is
# 1e-7 of the range, far below the energy resolution, and the cost is 24
# vectorised clips per lap.
LAMBDA_MAX = 4.0
LAMBDA_BISECT_STEPS = 24


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


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
    notes: tuple = ()


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
        n_laps: int | None = None, closure_sigma_j: float = CLOSURE_SIGMA_J,
        policy_prior: bool = True, reject_outside_box: bool = True) -> Belief:
    """Filter the store forward. Policy prior proposes, the store box rejects.

    Inside the identified polytope the trace is uninformative *by construction*:
    every theta in P explains the data exactly, which is what set membership
    means. So the only things that can discriminate between particles are

      (a) the store box -- 0 to 4 MJ, as a rejection;
      (b) store closure per lap -- deployed and harvested must agree to within
          one store plus the Manual Override allowance;
      (c) the strategy prior, which is what decides where inside the deployment
          band each particle's P_K actually sits.

    Without (c) the assignment of P_K within the band is arbitrary -- uniform,
    or the midpoint -- and then the store-floor penalty is the only selection
    pressure left. Measured, that biased drag low (less drag, less inferred
    deployment, fuller store, fewer floor hits) and did double duty as the only
    thing making the energy estimate work at all: at FLOOR_PENALTY = 0 the CdA
    posterior was exactly the uniform draw mean and per-lap energy was 344.7%
    out; at 6.0 the posterior sat at 0.277 against a true 0.660 with ESS 12 of
    400. There is no setting of that knob that is not trading drag bias against
    energy accuracy, which is why it is gone.
    """
    rng = np.random.default_rng(seed)
    n = len(feed)
    Np = n_particles
    is_x = np.ones(n, bool) if is_x is None else np.asarray(is_x, bool)

    theta = _sample_theta(ident, rng, Np)
    e_wheel, dt = _interval_terms(feed, theta, rho, m_published, fuel_start,
                                  fuel_burn_per_lap, is_x)

    v_cap = np.minimum(feed.v[:-1], feed.v[1:])
    rpm_hi = np.maximum(feed.rpm[:-1], feed.rpm[1:])
    thr_hi = np.maximum(feed.throttle[:-1], feed.throttle[1:])
    zone = np.asarray(feed.in_zone, bool)
    zone_i = zone[:-1] | zone[1:]
    mo = (None if feed.manual_override is None
          else np.asarray(feed.manual_override, bool)[:-1])
    pk_lo, pk_hi = p_k_bounds(v_cap, zone_i, rpm_hi, thr_hi, regs,
                              False if mo is None else mo)
    e_ice_max = eta_d * p_ice_max(rpm_hi, thr_hi, regs) * dt
    d_lo = np.clip(e_wheel - e_ice_max[None, :], 0.0, None)
    d_hi = np.minimum(np.maximum(e_wheel, d_lo), (pk_hi * eta_d * dt)[None, :])
    d_hi = np.maximum(d_hi, d_lo)

    braking = np.zeros(n - 1, bool) if regime is None else (
        np.asarray(regime)[:-1] == BRAKE_HARVEST)
    superclip = np.zeros(n - 1, bool) if regime is None else (
        np.asarray(regime)[:-1] == SUPER_CLIP)
    recovering = braking | superclip
    harvest = np.where(recovering[None, :],
                       np.clip(-e_wheel, 0.0, (-pk_lo * dt)[None, :]), 0.0)
    d_lo = np.where(recovering[None, :], 0.0, d_lo)
    d_hi = np.where(recovering[None, :], 0.0, d_hi)

    # (c) the strategy prior, as the proposal for where in the band P_K sits.
    # Three interpretable parameters per particle -- see strategy.py for the
    # 1/(m v^3) argument that makes them the right three.
    v_cut = rng.uniform(*V_CUT_PRIOR, Np)
    v_harv = rng.uniform(*V_HARV_PRIOR, Np)
    width = rng.uniform(*WIDTH_PRIOR, Np)
    split = rng.random(Np)          # fallback when the prior is switched off

    E = rng.uniform(0.0, E_STORE_MAX, Np)
    E_var = np.zeros(Np)
    alive = np.ones(Np, bool)
    # Cumulative flow, and its running minimum. Deployable energy is the gap
    # between them and needs no cut-out detector at all: the store is known up
    # to a constant, E_k = c + F_k, so if the driver has touched the reserve at
    # least once then E_min = R and D_k = F_k - min_{j<=k} F_j exactly, with c
    # and R both cancelling. Before the first touch it is a lower bound.
    F = np.zeros(Np)
    F_min = np.zeros(Np)

    soc_m = np.empty(n); soc_lo = np.empty(n); soc_hi = np.empty(n)
    use_m = np.empty(n); use_lo = np.empty(n); use_hi = np.empty(n)
    since_m = np.empty(n)
    logw = np.zeros(Np)
    laps = np.asarray(feed.lap_frac, float).astype(int)
    n_laps_total = int(laps.max()) + 1 if n_laps is None else int(n_laps)
    dep_lap, har_lap, ess = {}, {}, []
    notes = []

    def weights():
        lw = np.where(alive, logw, -np.inf)
        if not np.isfinite(lw).any():
            return None
        w = np.exp(lw - lw.max())
        return w / w.sum()

    def record(k, w):
        soc_m[k] = float(np.sum(E * w))
        order = np.argsort(E); c = np.cumsum(w[order])
        soc_lo[k] = float(E[order][min(np.searchsorted(c, 0.10), Np - 1)])
        soc_hi[k] = float(E[order][min(np.searchsorted(c, 0.90), Np - 1)])
        D = np.maximum(F - F_min, 0.0)
        use_m[k] = float(np.sum(D * w))
        uo = np.argsort(D); cu = np.cumsum(w[uo])
        use_lo[k] = float(D[uo][min(np.searchsorted(cu, 0.10), Np - 1)])
        use_hi[k] = float(D[uo][min(np.searchsorted(cu, 0.90), Np - 1)])
        since_m[k] = use_m[k]

    w = np.full(Np, 1.0 / Np)
    record(0, w)
    dep_acc = np.zeros(Np)
    har_acc = np.zeros(Np)

    # Blocked by lap, because the policy prior's *scale* has to be solved rather
    # than assumed. The prior gives the shape of P_K -- deploy below the cut-off
    # speed, taper above it -- but its amplitude saturates at the regulation
    # ceiling, so taking it at face value simply pins deployment to the top of
    # the band: measured, 3.82 MJ/lap against a true 2.41, a net flow of
    # -1.71 MJ/lap against a true -0.25, and a store that drifts 20 MJ out of a
    # 4 MJ box. Store closure is what sets the amplitude, and it has to be
    # *solved* for, not weighted: no particle satisfies it by luck.
    interval_lap = laps[:-1]
    for lap_id in np.unique(interval_lap):
        sl = np.flatnonzero(interval_lap == lap_id)
        if len(sl) == 0:
            continue
        h_lap = harvest[:, sl]
        want_lap = ((pk_hi[sl] * eta_d * dt[sl])[None, :]
                    * _sigmoid((v_cut[:, None] - v_cap[sl][None, :])
                               / np.maximum(width, 1e-3)[:, None]))
        lo_lap, hi_lap = d_lo[:, sl], d_hi[:, sl]

        if policy_prior:
            # d(lam) is monotone non-decreasing in lam, so bisection is exact
            # to tolerance in a fixed number of steps and needs no derivative.
            target = h_lap.sum(axis=1)
            lam_lo = np.zeros(Np)
            lam_hi = np.full(Np, LAMBDA_MAX)
            for _ in range(LAMBDA_BISECT_STEPS):
                lam = 0.5 * (lam_lo + lam_hi)
                tot = np.clip(lam[:, None] * want_lap, lo_lap, hi_lap).sum(axis=1)
                too_much = tot > target
                lam_hi = np.where(too_much, lam, lam_hi)
                lam_lo = np.where(too_much, lam_lo, lam)
            lam = 0.5 * (lam_lo + lam_hi)
            d_lap = np.clip(lam[:, None] * want_lap, lo_lap, hi_lap)
        else:
            d_lap = lo_lap + split[:, None] * (hi_lap - lo_lap)

        for j, k in enumerate(sl):
            d = d_lap[:, j]
            h = h_lap[:, j]
            if rao_blackwell:
                E_var = E_var + ((hi_lap[:, j] - lo_lap[:, j]) ** 2) / 12.0
            E_new = E + h - d
            if reject_outside_box:
                alive = alive & (E_new >= -1.0) & (E_new <= E_STORE_MAX + 1.0)
            E = np.clip(E_new, 0.0, E_STORE_MAX)
            F = F + (h - d)
            F_min = np.minimum(F_min, F)
            dep_acc += d
            har_acc += h
            w = weights()
            if w is None:
                notes.append(
                    f"every particle left the 0-4 MJ store box by sample {k}: "
                    f"the deployment band and the identified drag cannot be "
                    f"reconciled with a 4 MJ store on this trace")
                w = np.full(Np, 1.0 / Np)
                alive = np.ones(Np, bool)
                logw = np.zeros(Np)
            record(k + 1, w)

        # Store closure, with the Rao-Blackwellised variance in the
        # denominator. This is where E_var earns the name: a particle whose
        # deployment band was wide all lap has a genuinely uncertain net flow
        # and should not be judged against closure as harshly as one whose band
        # was tight. Carrying E_var and never using it would make
        # "Rao-Blackwellised" decoration -- which is what an earlier version of
        # this module did.
        net = har_acc - dep_acc
        logw = logw - 0.5 * net ** 2 / (closure_sigma_j ** 2 + E_var)
        w = weights()
        if w is None:
            w = np.full(Np, 1.0 / Np); alive = np.ones(Np, bool); logw = np.zeros(Np)
        ess.append(float(1.0 / np.sum(w ** 2)))
        dep_lap[int(lap_id)] = float(np.sum(dep_acc * w))
        har_lap[int(lap_id)] = float(np.sum(har_acc * w))
        take = _systematic(w, rng)
        E, E_var, theta, split, F, F_min = (
            E[take], E_var[take], theta[take], split[take], F[take], F_min[take])
        v_cut, v_harv, width = v_cut[take], v_harv[take], width[take]
        alive = np.ones(Np, bool)
        v_cut = np.clip(v_cut + rng.normal(0.0, 1.5, Np), *V_CUT_PRIOR)
        v_harv = np.clip(v_harv + rng.normal(0.0, 1.5, Np), *V_HARV_PRIOR)
        logw = np.zeros(Np)
        dep_acc = np.zeros(Np); har_acc = np.zeros(Np)

    w = weights()
    if w is None:
        w = np.full(Np, 1.0 / Np)
    return Belief(
        t=np.asarray(feed.t).copy(), soc_mean=soc_m, soc_p10=soc_lo,
        soc_p90=soc_hi, usable_mean=use_m, usable_p10=use_lo, usable_p90=use_hi,
        since_reserve=since_m, deployed_lap=dep_lap, harvested_lap=har_lap,
        reserve_mean=float(np.sum(-F_min * w)),
        reserve_sigma=float(np.sqrt(np.sum(w * (-F_min - np.sum(-F_min * w)) ** 2))),
        ess=np.array(ess), theta_post_mean=np.sum(theta * w[:, None], axis=0),
        theta_post_lo=theta.min(axis=0), theta_post_hi=theta.max(axis=0),
        n_particles=Np, dry=np.zeros(n, bool), notes=tuple(notes))


def _systematic(w: np.ndarray, rng) -> np.ndarray:
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(w / w.sum()), pos).clip(0, n - 1)
