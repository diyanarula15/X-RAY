"""Decision engine: when to spend the energy you have, given what you believe
about the energy they have.

Backward induction over the remaining stint. The only inputs that touch the
rival are the belief trace's numbers -- this module never reads the rival's
true state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import E_STORE_MAX, P_ICE_MAX, P_MGUK_MAX, p_mguk_ceiling
from .overtake import p_pass
from .policy import DeploymentPolicy
from .track import Track, Zone
from .vehicle import CarState, VehicleParams, set_car_mass, step

N_BINS = 20
R_PASS = 1.0   # value of completing a pass with a full stint still to run


# --------------------------------------------------------------- calibration
@dataclass(frozen=True)
class ZoneModel:
    """What a zone does with energy, measured on the vehicle model itself."""
    name: str
    braking_severity: float
    energy_grid: np.ndarray   # J spent on the straight
    speed_grid: np.ndarray    # m/s reached at the braking point
    dv_per_mj: float          # local slope, m/s per MJ, for reporting
    # ---- P1, all optional so every P0 construction and consumer is unchanged.
    # Own and rival no longer have to share a map: they can carry different CdA,
    # ClA and tyre state, and `delta_v` used to assume the difference between
    # two energies on ONE curve, which silently asserted identical cars.
    rival_energy_grid: np.ndarray | None = None
    rival_speed_grid: np.ndarray | None = None
    rival_dv_per_mj: float | None = None
    # Tyre-aware surfaces, shape (n_wear, n_energy), indexed by `wear_grid`.
    wear_grid: np.ndarray | None = None
    speed_grid_by_wear: np.ndarray | None = None
    rival_speed_grid_by_wear: np.ndarray | None = None
    # Energy an attack can physically deploy over the ATTACK window (the zone
    # straight), at the entry speed it was measured for. Distinct from
    # `energy_grid.max()`, which covers run-up + straight. None when nobody has
    # measured it, in which case consumers fall back to the energy axis.
    executable_ceiling_j: float | None = None
    executable_ceiling_entry_v_mps: float | None = None

    def own_speed(self, e: float, wear: float | None = None) -> float:
        return _interp_surface(e, wear, self.energy_grid, self.speed_grid,
                               self.wear_grid, self.speed_grid_by_wear)

    def rival_speed(self, e: float, wear: float | None = None) -> float:
        eg = self.energy_grid if self.rival_energy_grid is None else self.rival_energy_grid
        sg = self.speed_grid if self.rival_speed_grid is None else self.rival_speed_grid
        return _interp_surface(e, wear, eg, sg, self.wear_grid,
                               self.rival_speed_grid_by_wear)


def _interp_surface(e, wear, energy_grid, speed_grid, wear_grid, surface):
    """Energy -> speed, optionally on a wear axis. Bilinear, clamped.

    Falls back to the 1-D map whenever there is no surface or no wear was
    supplied, which is what keeps every P0 call bit-identical.
    """
    if surface is None or wear is None or wear_grid is None:
        return float(np.interp(e, energy_grid, speed_grid))
    w = float(np.clip(wear, wear_grid[0], wear_grid[-1]))
    j = int(np.clip(np.searchsorted(wear_grid, w) - 1, 0, len(wear_grid) - 2))
    lo, hi = wear_grid[j], wear_grid[j + 1]
    f = 0.0 if hi <= lo else (w - lo) / (hi - lo)
    a = np.interp(e, energy_grid, surface[j])
    b = np.interp(e, energy_grid, surface[j + 1])
    return float(a + f * (b - a))


def calibrate_zone(track: Track, params: VehicleParams, zone: Zone,
                   n_points: int = 7, dt: float = 0.005,
                   physics_context=None, from_s: float | None = None,
                   entry_v_mps: float | None = None) -> ZoneModel:
    """Run the real longitudinal model down the zone straight at a range of
    deployment budgets and record where it ends up.

    Nothing here is a fitted constant: the map from energy to end-of-straight
    speed is whatever the physics says it is.

    WHICH WINDOW this covers matters, and conflating it with the attack window
    is what made P2's budget axis meaningless. By default the run starts at the
    last corner BEFORE the zone (`_run_up`), so `energy_grid` is energy over
    run-up + straight -- on Circuit Sigma zone A that is 1700 m entered at
    200 km/h, and it reaches 1.602 MJ because the low-speed part of the run-up
    sits under the full 350 kW ceiling. An attack in-race only spans the 1100 m
    straight, entered at 314 km/h where the taper allows about 89 kW, so at most
    1.109 MJ is deployable there. The two numbers describe different runs.

    `from_s` / `entry_v_mps` override the start so the same function can measure
    the attack window itself. See `executable_attack_ceiling`.
    """
    set_car_mass(params.mass_car)
    s_from, entry_v = _run_up(track, zone)
    if from_s is not None:
        s_from = float(from_s)
    if entry_v_mps is not None:
        entry_v = float(entry_v_mps)
    # The tyre reaches the braking point mostly through the corner BEFORE the
    # straight, not along it. A short straight is power-and-drag limited, so a
    # worn tyre loses almost nothing on it -- but it exits the preceding corner
    # slower, and arrives slower for the whole run. Same sqrt(grip) law that
    # `vehicle._regime` applies to every other corner (m v^2 / r = mu N), read
    # off the same TyreState, so this is the existing physics being applied at
    # the one point the calibration starts from rather than a second model.
    ctx_tyre = None if physics_context is None else getattr(physics_context, "tyre", None)
    if ctx_tyre is not None:
        entry_v *= float(np.sqrt(max(getattr(ctx_tyre, "grip_scale", 1.0), 0.0)))
    # wrap-safe: on Circuit Sigma the Zone A run-up crosses the start line
    run_m = (zone.s_straight_end - s_from) % track.length
    budgets = np.linspace(0.0, 2.6e6, n_points)
    speeds, spent = [], []
    for budget in budgets:
        st = CarState(s=s_from, s_total=0.0, v=entry_v, E=budget,
                      fuel=params.fuel_start * 0.5)
        used, peak, guard = 0.0, entry_v, 0
        while st.s_total < run_m and guard < 400_000:
            demand = P_MGUK_MAX if st.E > 0 else 0.0
            out = step(track, st, params, demand, dt,
                       physics_context=physics_context)
            used += out["p_mguk"] * dt
            peak = max(peak, st.v)
            guard += 1
        speeds.append(peak)
        spent.append(used)
    spent = np.array(spent)
    speeds = np.array(speeds)
    order = np.argsort(spent)
    spent, speeds = spent[order], speeds[order]
    slope = float(np.polyfit(spent / 1e6, speeds, 1)[0]) if len(spent) > 1 else 0.0
    return ZoneModel(zone.name, zone.braking_severity, spent, speeds, slope)


def _same_context(a, b, wear_grid) -> bool:
    """Do two context factories produce physically identical tyre states?

    Compared on the fields `vehicle.step` actually reads -- grip_scale, and the
    environment object -- rather than on object identity, because the factories
    are closures and are never the same object.
    """
    if a is b:
        return True
    try:
        for w in wear_grid:
            ca, cb = a(float(w)), b(float(w))
            if ca.environment is not cb.environment:
                return False
            if ca.downforce_factor != cb.downforce_factor:
                return False
            ta, tb = ca.tyre, cb.tyre
            if (ta is None) != (tb is None):
                return False
            if ta is not None and (
                    ta.compound != tb.compound
                    or abs(ta.grip_scale - tb.grip_scale) > 1e-12):
                return False
    except Exception:
        return False
    return True


def executable_attack_ceiling(track: Track, params: VehicleParams, zone: Zone,
                              entry_v_mps: float, dt: float = 0.005,
                              physics_context=None,
                              store_j: float = E_STORE_MAX) -> dict:
    """Most electrical energy an attack can actually put down in THIS zone.

    Measured by commanding `P_MGUK_MAX` every step through the shared integrator
    and summing the `p_mguk` it returns. Nothing is integrated in closed form,
    because two closed forms are both wrong here:

    * Integrating the taper over the window overstates it. On Circuit Sigma zone
      A from a 314 km/h entry that gives 0.981 MJ against 0.448 MJ delivered --
      it counts the ceiling through the braking and off-throttle stretch, where
      the car draws nothing whatever the rules permit.
    * The taper is not exogenous. Deploying harder raises v, which LOWERS the
      ceiling at the next step, so the bound depends on the trajectory it is
      meant to bound. The car crosses 345 km/h after 378 m of zone A's 1100 m
      straight, and the normal curve is exactly 0 kW beyond that: no budget of
      any size buys a single joule over the remaining 660 m.

    Context-dependent by construction. Zone A absorbs 1.603 MJ entered at
    200 km/h and 0.448 MJ entered at 314 km/h; a per-zone constant cannot say
    both, which is why P2's budget axis -- built from `ZoneModel.energy_grid`,
    measured over run-up + straight from a slow corner exit -- offered budgets up
    to 1.602 MJ for a zone that could execute 0.45.
    """
    set_car_mass(params.mass_car)
    run_m = (zone.s_straight_end - zone.s_straight_start) % track.length
    st = CarState(s=float(zone.s_straight_start), s_total=0.0,
                  v=float(entry_v_mps), E=float(store_j),
                  fuel=params.fuel_start * 0.5)
    deployed_j = 0.0
    taper_zero_at_m = None
    guard = 0
    while st.s_total < run_m and guard < 400_000:
        if taper_zero_at_m is None and p_mguk_ceiling(st.v) <= 0.0:
            taper_zero_at_m = float(st.s_total)
        out = step(track, st, params, P_MGUK_MAX, dt,
                   physics_context=physics_context)
        deployed_j += out["p_mguk"] * dt
        guard += 1
    return {"zone": zone.name, "entry_v_mps": float(entry_v_mps),
            "window_m": float(run_m),
            "executable_ceiling_j": float(deployed_j),
            "taper_zero_after_m": taper_zero_at_m,
            "exit_v_mps": float(st.v)}


def calibrate_zone_with_wear(track: Track, params: VehicleParams, zone: Zone,
                             context_for_wear, wear_levels=(0.0, 0.5, 1.0),
                             rival_params: VehicleParams | None = None,
                             rival_context_for_wear=None,
                             n_points: int = 7, dt: float = 0.005) -> ZoneModel:
    """A ZoneModel carrying an energy-by-wear speed surface.

    `context_for_wear(wear) -> PhysicsContext` is supplied by the caller, so the
    tyre model lives in xray.tyres and the physics lives in vehicle.step; this
    function only decides which points to evaluate. No force equation is written
    here, which is the rule that keeps decision.py from growing a second physics.

    The rival surface is built the same way from its own params and context, so
    two cars with different aero or different tyres get different maps.
    """
    wear_grid = np.asarray(wear_levels, dtype=float)
    own_models = [calibrate_zone(track, params, zone, n_points, dt,
                                 physics_context=context_for_wear(float(w)))
                  for w in wear_grid]
    # The base map is the surface's own first row, not a seventh calibration of
    # the same state. Recomputing wear = 0 separately cost a full zone run per
    # zone per context for nothing.
    base = own_models[int(np.argmin(np.abs(wear_grid)))]
    own = np.vstack([m.speed_grid for m in own_models])

    riv_surface = None
    riv_energy = riv_speed = None
    riv_slope = None
    if rival_params is not None:
        rctx = rival_context_for_wear or context_for_wear
        # Physically identical inputs give physically identical maps. When both
        # cars share a chassis AND a tyre context -- the same-compound case,
        # which is common -- the rival surface IS the own surface, and running
        # it again is arithmetic we already have the answer to.
        same = (rival_params is params
                and _same_context(rctx, context_for_wear, wear_grid))
        if same:
            riv_surface = own
            riv_energy = base.energy_grid
            riv_speed = base.speed_grid
            riv_slope = base.dv_per_mj
        else:
            rows, rz = [], None
            for w in wear_grid:
                rz = calibrate_zone(track, rival_params, zone, n_points, dt,
                                    physics_context=rctx(float(w)))
                rows.append(rz.speed_grid)
            riv_surface = np.vstack(rows)
            riv_energy, riv_speed, riv_slope = (rz.energy_grid, rz.speed_grid,
                                                rz.dv_per_mj)
    return ZoneModel(
        base.name, base.braking_severity, base.energy_grid, base.speed_grid,
        base.dv_per_mj,
        rival_energy_grid=riv_energy, rival_speed_grid=riv_speed,
        rival_dv_per_mj=riv_slope,
        wear_grid=wear_grid, speed_grid_by_wear=own,
        rival_speed_grid_by_wear=riv_surface)


def _run_up(track: Track, zone: Zone) -> tuple[float, float]:
    """Where the run to this zone's braking point really begins.

    The last corner before the zone straight, which is not always the corner
    immediately adjacent to it -- Circuit Sigma's pit straight runs into the
    Zone A straight with nothing between them, so a car arrives at Zone A
    already doing 300 km/h, not at hairpin speed.
    """
    best = None
    for start, end, vlim in track.corners:
        back = (zone.s_straight_start - end) % track.length
        if best is None or back < best[0]:
            best = (back, end, vlim)
    _back, s_end, vlim = best
    return float(s_end), float(vlim)


def delta_v(zm: ZoneModel, e_own: float, e_riv: float,
            wear_own: float | None = None,
            wear_riv: float | None = None) -> float:
    """Speed advantage at the braking point implied by two energy states.

    With no rival map and no wear this is exactly the P0 expression: one curve,
    two energies. When a rival map or a wear surface is present the two cars are
    read off their own curves, because the P0 form quietly asserted that the
    rival's car and tyres were identical to ours.
    """
    return zm.own_speed(e_own, wear_own) - zm.rival_speed(e_riv, wear_riv)


# ---------------------------------------------------------------- the DP
@dataclass(frozen=True)
class DecisionModel:
    zones: list[ZoneModel]
    recharge_per_lap: float     # J recovered per lap
    own_spend_per_lap: float    # J we deploy on a lap we are NOT attacking on.
                                # Attacking costs `attack_cost` on top of this;
                                # without the distinction a lap's recovery pays
                                # for the attack and the DP attacks for free.
    rival_spend_per_lap: float  # J the rival deploys per lap, from the belief.
                                # This is the whole reason waiting can be worth
                                # anything: a rival spending more than they
                                # recover is draining, and the DP can see it.
    attack_cost: float          # J spent on an attack
    defend_cost: float          # J the rival spends defending. Zero by default:
                                # the rival's spending is already described by
                                # the belief trace, and crediting our attack with
                                # draining them would let the DP attack for free.
    gap_s: float                # nominal gap at the braking point
    n_laps: int
    fail_cost: float = 0.0      # fraction of continuation value lost when a
                                # lunge fails: you yield the place back and
                                # arrive at the next chance further adrift
    bins: np.ndarray = field(default=None)

    @property
    def edges(self) -> np.ndarray:
        return self.bins


def make_bins(n: int = N_BINS) -> np.ndarray:
    return np.linspace(0.0, E_STORE_MAX, n)


def _bin_index(bins: np.ndarray, e) -> np.ndarray:
    return np.clip(np.searchsorted(bins, np.asarray(e), side="left"), 0, len(bins) - 1)


@dataclass(frozen=True)
class DPSolution:
    V: np.ndarray            # (laps+1, bins, bins) value
    best_zone: np.ndarray    # (laps+1, bins, bins) index, -1 = wait
    tau: np.ndarray          # (laps+1, bins) threshold opportunity quality
    model: DecisionModel

    def action(self, laps_left: int, e_own: float, e_riv: float):
        k = int(np.clip(laps_left, 0, self.V.shape[0] - 1))
        i = int(_bin_index(self.model.bins, e_own))
        j = int(_bin_index(self.model.bins, e_riv))
        z = int(self.best_zone[k, i, j])
        return (None if z < 0 else self.model.zones[z].name)

    def quality(self, laps_left: int, e_own: float, e_riv: float) -> float:
        """Best p_pass available this lap at this pair of energy states."""
        return max(_zone_p(zm, e_own, e_riv, self.model) for zm in self.model.zones)


def _zone_p(zm: ZoneModel, e_own, e_riv, model: DecisionModel) -> float:
    return p_pass(delta_v(zm, e_own, e_riv), model.gap_s, zm)


def solve(model: DecisionModel) -> DPSolution:
    """Backward induction. State is (laps left, own energy, rival energy)."""
    bins = model.bins
    nb = len(bins)
    K = model.n_laps
    V = np.zeros((K + 1, nb, nb))
    best = np.full((K + 1, nb, nb), -1, dtype=np.int8)
    tau = np.zeros((K + 1, nb))

    E_own = bins[:, None]
    E_riv = bins[None, :]
    own_next = np.clip(E_own + model.recharge_per_lap - model.own_spend_per_lap,
                       0.0, E_STORE_MAX)
    wait_own = _bin_index(bins, own_next)
    riv_next = np.clip(E_riv + model.recharge_per_lap - model.rival_spend_per_lap,
                       0.0, E_STORE_MAX)
    wait_riv = _bin_index(bins, riv_next)
    att_own = _bin_index(bins, np.clip(own_next - model.attack_cost, 0.0, E_STORE_MAX))
    att_riv = _bin_index(bins, np.clip(riv_next - model.defend_cost, 0.0, E_STORE_MAX))

    P = np.stack([np.array([[_zone_p(zm, eo, er, model) for er in bins] for eo in bins])
                  for zm in model.zones])  # (nz, nb, nb)

    for k in range(1, K + 1):
        v_next = V[k - 1]
        v_wait = v_next[wait_own, wait_riv]
        v_fail = v_next[att_own, att_riv]
        # What a pass is worth depends on when it happens: the reward is the
        # share of the stint spent in front, so passing on lap 3 is worth more
        # than passing on the last lap. Without this the value function
        # saturates at 1 everywhere, every attack looks free, and the threshold
        # collapses to zero -- attack always, which is not a strategy.
        reward = R_PASS * k / model.n_laps
        v_attack = (P * reward
                    + (1.0 - P) * (1.0 - model.fail_cost) * v_fail[None, :, :])
        z_best = np.argmax(v_attack, axis=0)
        v_best = np.max(v_attack, axis=0)
        take = v_best > v_wait
        V[k] = np.where(take, v_best, v_wait)
        best[k] = np.where(take, z_best, -1)

        # threshold: the opportunity quality at which attacking starts to win.
        # V_attack = q*reward + (1-q)*V_fail, so q* = (V_wait-V_fail)/(reward-V_fail).
        vf = (1.0 - model.fail_cost) * v_fail
        q_star = np.clip((v_wait - vf) / np.maximum(reward - vf, 1e-9), 0.0, 1.0)
        tau[k] = q_star.mean(axis=1)

    return DPSolution(V=V, best_zone=best, tau=tau, model=model)


def fail_cost_from_geometry(gap_s: float, yield_m: float = 10.0,
                            v_ref: float = 60.0, dv_ref: float = 8.0) -> float:
    """How much a failed lunge costs, in the currency the DP uses.

    The simulator makes a failed attacker yield the place and drop `yield_m`
    behind. At racing speed that is an extra `yield_m / v_ref` seconds of gap,
    and the pass model says exactly what that does to the next opportunity. The
    number is read off the pass model, not chosen.
    """
    class _Z:
        braking_severity = 1.0
    before = p_pass(dv_ref, gap_s, _Z())
    after = p_pass(dv_ref, gap_s + yield_m / v_ref, _Z())
    return float(np.clip(1.0 - after / max(before, 1e-9), 0.0, 0.9))


def build_model(track: Track, params: VehicleParams, n_laps: int,
                recharge_per_lap: float, rival_spend_per_lap: float,
                own_spend_per_lap: float | None = None, gap_s: float = 0.45,
                attack_cost: float | None = None,
                defend_cost: float = 0.0) -> DecisionModel:
    zones = [calibrate_zone(track, params, z) for z in track.zones]
    cost = attack_cost if attack_cost is not None else float(zones[0].energy_grid.max())
    return DecisionModel(
        zones=zones, recharge_per_lap=recharge_per_lap,
        own_spend_per_lap=(own_spend_per_lap if own_spend_per_lap is not None
                           else 0.6 * recharge_per_lap),
        rival_spend_per_lap=rival_spend_per_lap, attack_cost=cost,
        defend_cost=defend_cost, gap_s=gap_s, n_laps=n_laps, bins=make_bins(),
        fail_cost=fail_cost_from_geometry(gap_s))


# ------------------------------------------------- opponent policy posterior
def policy_posterior(belief, track, n_samples: int = 200, seed: int = 0
                     ) -> list[DeploymentPolicy]:
    """Bootstrap posterior over the rival's deployment policy.

    Resample the estimated per-lap deployment totals with replacement; each
    resample implies a different picture of how much the rival spends and how
    early, which maps onto `front_loading` and `reserve`. Nothing here reads
    the simulator: it is built from `belief`, which came from a speed trace.
    """
    rng = np.random.default_rng(seed)
    dep = np.asarray(belief.deployed_lap, dtype=float)
    dep = dep[np.isfinite(dep) & (dep > 0)]
    if len(dep) < 2:
        dep = np.array([2.0e6, 2.0e6])
    reserve_mu = belief.reserve_mean / E_STORE_MAX
    reserve_sd = max(belief.reserve_sigma / E_STORE_MAX, 0.02)

    out = []
    for _ in range(n_samples):
        draw = rng.choice(dep, size=len(dep), replace=True)
        spend = float(draw.mean())
        # a car that spends more per lap than it can recover is front-loading
        front = float(np.clip(spend / (2.0 * belief.harvested_lap.mean() + 1.0), 0.0, 1.0))
        reserve = float(np.clip(rng.normal(reserve_mu, reserve_sd), 0.0, 0.5))
        gap_sens = float(np.clip(rng.normal(0.8, 0.4), 0.0, 2.0))
        prefs = np.clip(rng.dirichlet([3.0, 2.0, 1.0]) * 3.0, 0.05, 1.5)
        out.append(DeploymentPolicy(
            name="POSTERIOR", front_loading=front, reserve=reserve,
            gap_sensitivity=gap_sens,
            zone_preference={z.name: float(p) for z, p in zip(track.zones, prefs)}))
    return out


def robustness(model: DecisionModel, policies: list[DeploymentPolicy],
               laps_left: int, e_own: float, e_riv: float) -> dict:
    """Re-solve the DP under each sampled opponent policy and report how often
    the recommendation is unchanged.

    A sampled policy changes the rival's recharge and how much they hold back,
    which moves their effective energy -- so each sample is a different DP.
    """
    from dataclasses import replace
    base = solve(model).action(laps_left, e_own, e_riv)
    agree = 0
    actions = []
    for pol in policies:
        e_riv_eff = max(e_riv - pol.reserve * E_STORE_MAX, 0.0)
        m = replace(model,
                    rival_spend_per_lap=model.rival_spend_per_lap
                    * (0.7 + 0.6 * pol.front_loading))
        act = solve(m).action(laps_left, e_own, e_riv_eff)
        actions.append(act)
        agree += (act == base)
    return {"action": base, "fraction_agreeing": agree / max(len(policies), 1),
            "actions": actions}


# ------------------------------------------------------- counterfactual eval
def simulate_stint(model: DecisionModel, choose, rng, n_laps: int,
                   e_own0: float, e_riv0: float) -> dict:
    """Lap-level Monte Carlo of a stint.

    Energy bookkeeping and the zone speed maps are the calibrated ones from the
    vehicle model; the pass itself is resolved with `p_pass`. `choose` is a
    callable (laps_left, e_own, e_riv_belief) -> zone name or None.

    This is a reduced-order race: it decides once per lap rather than
    integrating the full 200 Hz two-car simulation. `scripts/03.b_run_decision_eval.py`
    reruns the headline comparison at full fidelity.
    """
    e_own, e_riv = float(e_own0), float(e_riv0)
    zmap = {zm.name: zm for zm in model.zones}
    for k in range(n_laps, 0, -1):
        pick = choose(k, e_own, e_riv)
        e_own = float(np.clip(e_own + model.recharge_per_lap - model.own_spend_per_lap,
                              0.0, E_STORE_MAX))
        e_riv = float(np.clip(e_riv + model.recharge_per_lap - model.rival_spend_per_lap,
                              0.0, E_STORE_MAX))
        if pick is None:
            continue
        zm = zmap[pick]
        spend = min(model.attack_cost, e_own)
        p = p_pass(delta_v(zm, spend, min(e_riv, model.attack_cost)), model.gap_s, zm)
        e_own = max(e_own - spend, 0.0)
        if rng.random() < p:
            return {"passed": 1, "lap_passed": n_laps - k + 1, "laps_ahead": k}
    return {"passed": 0, "lap_passed": None, "laps_ahead": 0}


def blind_chooser(model: DecisionModel):
    """The baseline: a driver with no read on the rival's store.

    Attacks the highest-braking-severity zone whenever the battery will cover
    it. This is what "race by feel" looks like -- it is not a straw man, it is
    the rule you use when the energy channel does not exist.
    """
    best = max(model.zones, key=lambda z: z.braking_severity).name

    def choose(laps_left, e_own, e_riv_belief):
        return best if e_own >= model.attack_cost else None
    return choose


def xray_chooser(sol: DPSolution, belief_e_riv, belief_sigma: float, rng):
    """Uses the reconstructed rival store, with its uncertainty, not the truth."""
    def choose(laps_left, e_own, e_riv_true):
        seen = float(np.clip(np.interp(laps_left, [1, len(belief_e_riv)],
                                       [belief_e_riv[-1], belief_e_riv[0]])
                             + rng.normal(0.0, belief_sigma), 0.0, E_STORE_MAX))
        return sol.action(laps_left, e_own, seen)
    return choose


def compare_policies(model: DecisionModel, sol: DPSolution, belief_e_riv,
                     belief_sigma: float, n_races: int = 50, n_laps: int = 12,
                     e_own0: float = 2.0e6, e_riv0: float = 3.0e6,
                     seed: int = 0) -> dict:
    """Paired comparison: same seed, same rival, two decision rules."""
    blind = blind_chooser(model)
    xr = []
    bl = []
    for i in range(n_races):
        rng_x = np.random.default_rng(seed * 1000 + i)
        rng_b = np.random.default_rng(seed * 1000 + i)  # paired coin flips
        chooser = xray_chooser(sol, belief_e_riv, belief_sigma,
                               np.random.default_rng(seed * 7919 + i))
        xr.append(simulate_stint(model, chooser, rng_x, n_laps, e_own0, e_riv0)["passed"])
        bl.append(simulate_stint(model, blind, rng_b, n_laps, e_own0, e_riv0)["passed"])
    xr = np.array(xr, dtype=float)
    bl = np.array(bl, dtype=float)
    d = xr - bl
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    return {"xray_pass_rate": float(xr.mean()), "blind_pass_rate": float(bl.mean()),
            "mean_gain": float(d.mean()), "ci95": (float(d.mean() - 1.96 * se),
                                                   float(d.mean() + 1.96 * se)),
            "n_races": n_races}


@dataclass(frozen=True)
class TyreDecisionContext:
    """The tyre axis of the DP, and where its numbers came from.

    `wear_per_lap_hold` and `wear_per_lap_attack` are DERIVED by integrating
    xray.tyres.advance over a lap at two utilisations -- they are not penalty
    constants, and if the tyre model changes they change with it. Nothing in
    this module computes wear; it only indexes it.

    `pit_at_laps_left` is when a KNOWN stop happens, expressed in the DP's own
    horizon coordinate. None means no known stop, and that is the common case
    for a rival: there is no validated pit-inference model, so the honest
    continuation is the worn state carrying on rather than an invented reset.
    """
    wear_grid: np.ndarray
    wear_per_lap_hold: float
    wear_per_lap_attack: float
    pit_at_laps_left: int | None = None
    # The rival's CURRENT wear. Exogenous, like their energy track: our wear is
    # the DP's state because our choices move it, theirs is a fixed number we
    # read off their tyre state at this opportunity. None => their map is read
    # without a wear argument, which is the P0 behaviour.
    rival_wear: float | None = None
    compound: str = "UNKNOWN"
    pit_source: str = "unknown"
    pit_confidence: float = 0.0

    def __post_init__(self):
        if self.wear_per_lap_attack < self.wear_per_lap_hold:
            raise ValueError("attacking cannot wear the tyre less than holding")
        g = np.asarray(self.wear_grid, dtype=float)
        if len(g) < 2 or np.any(np.diff(g) <= 0):
            raise ValueError("wear_grid must be increasing with at least 2 bins")
        # No resolution requirement: the solver interpolates along this axis, so
        # a coarse grid represents a small wear increment exactly rather than
        # rounding it to nothing. What the grid must do is SPAN the states the
        # car can reach, or the transition silently clips instead.
        if g[0] > 0.0 or g[-1] < 1.0:
            raise ValueError("wear_grid must span 0..1")


def _wear_index(grid: np.ndarray, w):
    """Nearest wear bin. For LOOKUP only -- the solver interpolates instead."""
    g = np.asarray(grid, dtype=float)
    w = np.clip(np.asarray(w, dtype=float), g[0], g[-1])
    hi = np.clip(np.searchsorted(g, w, side="left"), 1, len(g) - 1)
    lo = hi - 1
    return np.where(np.abs(w - g[lo]) <= np.abs(g[hi] - w), lo, hi)


def _wear_weights(grid: np.ndarray, w):
    """(lo, hi, frac) for linear interpolation along the wear axis.

    Snapping a wear transition to the nearest bin quantises it away: one
    attacking lap costs a few thousandths of tyre life, a usable grid has bins
    of a few hundredths, so hold and attack round to the SAME bin and the DP
    prices the entire tyre cost at exactly zero. It did, silently.

    Making the grid fine enough instead is the obvious fix and the wrong one --
    it needs hundreds of bins and the DP is O(laps x energy x wear x zones).
    Interpolating the continuation value keeps the axis coarse and the
    transition exact, which is the standard treatment for a continuous state on
    a discrete grid.
    """
    g = np.asarray(grid, dtype=float)
    w = np.clip(np.asarray(w, dtype=float), g[0], g[-1])
    hi = np.clip(np.searchsorted(g, w, side="left"), 1, len(g) - 1)
    lo = hi - 1
    span = g[hi] - g[lo]
    frac = np.where(span > 0, (w - g[lo]) / np.where(span > 0, span, 1.0), 0.0)
    return lo, hi, frac


# --------------------------------------- known rival trajectory (the real case)
@dataclass(frozen=True)
class ExogenousSolution:
    """DP over (laps left, own energy) when the rival's energy at the decision
    point is already known per lap from the belief trace.

    This is the situation X-RAY actually creates. Guessing at the rival's
    lap-to-lap energy dynamics is only necessary while you cannot see them; once
    the speed trace has given you the trajectory, the rival stops being a state
    to be modelled and becomes a schedule to be read.
    """
    V: np.ndarray            # (laps+1, bins)
    best_zone: np.ndarray    # (laps+1, bins), -1 = wait
    tau: np.ndarray          # (laps+1, bins)
    q: np.ndarray            # (laps+1, bins) best available pass probability
    model: DecisionModel
    rival_track: np.ndarray  # J, deployable energy at the decision point per lap
    tyre: "TyreDecisionContext | None" = None   # present => arrays carry a wear axis

    def _lap_of(self, laps_left: int) -> int:
        return int(np.clip(self.model.n_laps - laps_left, 0, len(self.rival_track) - 1))

    def _idx(self, laps_left: int, e_own: float, wear: float | None):
        """(k, i) for P0, (k, i, j) once a wear axis exists."""
        k = int(np.clip(laps_left, 0, self.V.shape[0] - 1))
        i = int(_bin_index(self.model.bins, e_own))
        if self.tyre is None:
            return (k, i)
        j = int(_wear_index(self.tyre.wear_grid, 0.0 if wear is None else wear))
        return (k, i, j)

    def action(self, laps_left: int, e_own: float, wear: float | None = None):
        z = int(self.best_zone[self._idx(laps_left, e_own, wear)])
        return None if z < 0 else self.model.zones[z].name

    def quality(self, laps_left: int, e_own: float, wear: float | None = None) -> float:
        return float(self.q[self._idx(laps_left, e_own, wear)])

    def threshold(self, laps_left: int, e_own: float, wear: float | None = None) -> float:
        return float(self.tau[self._idx(laps_left, e_own, wear)])


def explain_exogenous_action(sol: ExogenousSolution, laps_left: int,
                             own_usable_energy_j: float,
                             wear: float | None = None,
                             rival_wear: float | None = None,
                             own_tyre=None, rival_tyre=None,
                             pit_context=None, environment=None) -> dict:
    """Expose the DP terms behind the exogenous-rival recommendation.

    This is the canonical inspection interface used by the API and tests. It
    reads values out of the solved finite-horizon problem rather than rebuilding
    a separate web threshold or pass-probability calculation.
    """
    model = sol.model
    bins = model.bins
    k = int(np.clip(laps_left, 0, sol.V.shape[0] - 1))
    i = int(_bin_index(bins, own_usable_energy_j))
    lap = sol._lap_of(k)
    rival_usable_energy_j = float(sol.rival_track[lap])
    own_grid_energy_j = float(bins[i])
    own_next_j = float(np.clip(own_grid_energy_j + model.recharge_per_lap
                              - model.own_spend_per_lap, 0.0, E_STORE_MAX))
    wait_i = int(_bin_index(bins, own_next_j))
    attack_next_j = float(np.clip(own_next_j - model.attack_cost, 0.0, E_STORE_MAX))
    attack_i = int(_bin_index(bins, attack_next_j))
    v_next = sol.V[k - 1] if k > 0 else sol.V[0]

    # The wear axis, mirrored from the solver rather than re-derived. Every
    # index below has to match `_solve_exogenous_tyre` exactly or the
    # explanation stops being an explanation -- which is what the
    # solver-agreement test sweeps for.
    t = sol.tyre
    wear_now = None
    wear_penalty = 0.0
    pit_resets_next_lap = False
    if t is None:
        value_wait = float(v_next[wait_i])
        value_fail = float((1.0 - model.fail_cost) * v_next[attack_i])
    else:
        wear_now = float(np.clip(0.0 if wear is None else wear, 0.0, 1.0))
        j = int(_wear_index(t.wear_grid, wear_now))
        pit_now = (t.pit_at_laps_left is not None and k == int(t.pit_at_laps_left))
        base_w = float(t.wear_grid[j])
        nh = 0.0 if pit_now else min(base_w + t.wear_per_lap_hold, 1.0)
        na = 0.0 if pit_now else min(base_w + t.wear_per_lap_attack, 1.0)

        def _v(i_energy, w_next):
            lo, hi, f = _wear_weights(t.wear_grid, w_next)
            return float((1.0 - f) * v_next[i_energy, int(lo)]
                         + f * v_next[i_energy, int(hi)])

        value_wait = _v(wait_i, nh)
        value_fail = float((1.0 - model.fail_cost) * _v(attack_i, na))
        # The price of the EXTRA wear an attack causes, isolated: same energy
        # bin, same lap, only the wear index differs. This is the entire
        # mechanism by which a pit stop changes the call -- at the stop jh and
        # ja are both 0, so the penalty is exactly zero and attacking is free of
        # tyre cost. Published because a claim this load-bearing should be
        # readable rather than inferred from two value differences.
        wear_penalty = _v(attack_i, nh) - _v(attack_i, na)
        pit_resets_next_lap = bool(pit_now)
    reward = R_PASS * k / max(model.n_laps, 1)
    # The solver masks unaffordable bins to -inf before its argmax. Leaving that
    # gate out here did not change the recommendation -- `decision` is read from
    # best_zone either way -- but it published a finite `value_attack` for an
    # attack the solver had already rejected, so the web could show
    # V_attack > V_wait next to HOLD. The explanation must be able to reproduce
    # the choice, not just accompany it.
    affordable = own_grid_energy_j >= model.attack_cost * 0.75

    rows = []
    for zi, zm in enumerate(model.zones):
        own_deploy_j = min(own_grid_energy_j, model.attack_cost)
        rival_deploy_j = min(rival_usable_energy_j, model.attack_cost)
        # Two cars, two curves. `rival_wear` defaults to whatever the solver was
        # given, so the explanation cannot read a different rival than the DP.
        riv_w = sol.tyre.rival_wear if (rival_wear is None and sol.tyre is not None) \
            else rival_wear
        dv = delta_v(zm, own_deploy_j, rival_deploy_j,
                     wear_own=wear_now, wear_riv=riv_w)
        q = p_pass(dv, model.gap_s, zm)
        value_hypothetical = float(q * reward + (1.0 - q) * value_fail)
        value_attack = value_hypothetical if affordable else -np.inf
        rows.append({
            "zone_index": zi,
            "zone": zm.name,
            "pass_probability": float(q),
            "predicted_delta_v_mps": float(dv),
            "predicted_own_speed_mps": float(zm.own_speed(own_deploy_j, wear_now)),
            "predicted_rival_speed_mps": float(zm.rival_speed(rival_deploy_j, riv_w)),
            "value_attack": float(value_attack),
            # What the attack would be worth if it were affordable. Kept apart
            # from `value_attack` so a rejected attack cannot be displayed as a
            # real one, and so the panel still has a number to show.
            "value_attack_hypothetical": value_hypothetical,
            "affordable": bool(affordable),
        })
    selected = int(sol.best_zone[sol._idx(k, own_usable_energy_j, wear_now)])
    # Report the zone the solver picked, not a second argmax over the same rows.
    # They agree on affordable bins, but only one of them decides.
    if selected >= 0:
        best_row = next(r for r in rows if r["zone_index"] == selected)
    else:
        best_row = (max(rows, key=lambda r: r["value_attack_hypothetical"])
                    if rows else None)
    return {
        "laps_left": k,
        "own_usable_energy_j": float(own_usable_energy_j),
        "rival_usable_energy_j": rival_usable_energy_j,
        "attack_threshold": float(sol.threshold(k, own_usable_energy_j, wear_now)),
        "value_wait": value_wait,
        "value_attack": float(best_row["value_attack"]) if best_row else -np.inf,
        "attack_affordable": bool(affordable),
        "best_zone": None if selected < 0 else model.zones[selected].name,
        "decision": "HOLD" if selected < 0 else "ATTACK",
        "best_attack": best_row,
        "zones": rows,
        # P1 state, reported so the web can explain a call without recomputing
        # any of it. None throughout when no tyre/stint context was supplied,
        # which is the P0 shape.
        "wear_fraction": wear_now,
        "rival_wear_fraction": (sol.tyre.rival_wear if sol.tyre is not None
                                else None),
        "attack_wear_continuation_penalty": wear_penalty,
        "pit_resets_next_lap": pit_resets_next_lap,
        "tyre_model": None if t is None else {
            "compound": t.compound,
            "wear_per_lap_hold": float(t.wear_per_lap_hold),
            "wear_per_lap_attack": float(t.wear_per_lap_attack),
            "attack_extra_wear_per_lap": float(t.wear_per_lap_attack
                                               - t.wear_per_lap_hold),
            "pit_at_laps_left": t.pit_at_laps_left,
            "pit_source": t.pit_source,
            "pit_confidence": float(t.pit_confidence),
        },
        "own_tyre": None if own_tyre is None else _tyre_row(own_tyre),
        "rival_tyre": None if rival_tyre is None else _tyre_row(rival_tyre),
        "pit_context": None if pit_context is None else {
            "planned_pit_lap": pit_context.planned_pit_lap,
            "laps_to_pit_mean": pit_context.laps_to_pit_mean,
            "pit_within_1_lap_prob": pit_context.pit_within_1_lap_prob,
            "pit_within_2_laps_prob": pit_context.pit_within_2_laps_prob,
            "source": pit_context.source,
            "confidence": pit_context.confidence,
        },
        "environment": None if environment is None else {
            "t": environment.t, "rho": environment.rho,
            "track_temp_c": environment.track_temp_c,
            "air_temp_c": environment.air_temp_c,
            "rainfall": environment.rainfall,
            "track_wetness_index": environment.track_wetness_index,
            "source": environment.source, "age_s": environment.age_s,
        },
    }


def _tyre_row(st) -> dict:
    """TyreState -> JSON. tyre_life stays age; wear stays modelled."""
    return {"compound": st.compound, "tyre_life_laps": st.tyre_life,
            "fresh_tyre": st.fresh_tyre,
            "estimated_temp_c": st.estimated_temp_c,
            "wear_fraction": st.wear_fraction,
            "grip_scale": st.grip_scale, "source": st.source}


def solve_exogenous(model: DecisionModel, rival_track: np.ndarray,
                    tyre: "TyreDecisionContext | None" = None) -> ExogenousSolution:
    """Canonical finite-horizon solver.

    With `tyre=None` this is the P0 problem, V[k, energy], untouched. With a
    tyre context the state gains a wear axis, V[k, energy, wear], and the reason
    "close to a pit stop" changes the answer is that wear RESETS at the stop --
    not because a near-pit multiplier was added to the reward. Attacking costs
    extra wear; if that wear is about to be thrown away it costs almost nothing,
    and if it has to be carried for ten more opportunities it costs a great
    deal. The horizon does that arithmetic by itself.
    """
    if tyre is not None:
        return _solve_exogenous_tyre(model, rival_track, tyre)
    bins = model.bins
    nb = len(bins)
    K = model.n_laps
    V = np.zeros((K + 1, nb))
    best = np.full((K + 1, nb), -1, dtype=np.int8)
    tau = np.zeros((K + 1, nb))
    qbest = np.zeros((K + 1, nb))

    own_next = np.clip(bins + model.recharge_per_lap - model.own_spend_per_lap,
                       0.0, E_STORE_MAX)
    wait_own = _bin_index(bins, own_next)
    att_own = _bin_index(bins, np.clip(own_next - model.attack_cost, 0.0, E_STORE_MAX))
    affordable = bins >= model.attack_cost * 0.75

    for k in range(1, K + 1):
        lap = int(np.clip(K - k, 0, len(rival_track) - 1))
        e_riv = float(rival_track[lap])
        P = np.array([[p_pass(delta_v(zm, min(eo, model.attack_cost),
                                      min(e_riv, model.attack_cost)),
                              model.gap_s, zm) for eo in bins]
                      for zm in model.zones])
        reward = R_PASS * k / K
        v_next = V[k - 1]
        v_wait = v_next[wait_own]
        v_fail = v_next[att_own]
        v_attack = (P * reward
                    + (1.0 - P) * (1.0 - model.fail_cost) * v_fail[None, :])
        v_attack = np.where(affordable[None, :], v_attack, -np.inf)
        z_best = np.argmax(v_attack, axis=0)
        v_best = np.max(v_attack, axis=0)
        take = v_best > v_wait
        V[k] = np.where(take, v_best, v_wait)
        best[k] = np.where(take, z_best, -1)
        qbest[k] = np.max(P, axis=0)
        vf = (1.0 - model.fail_cost) * v_fail
        tau[k] = np.clip((v_wait - vf) / np.maximum(reward - vf, 1e-9), 0.0, 1.0)

    return ExogenousSolution(V=V, best_zone=best, tau=tau, q=qbest, model=model,
                             rival_track=np.asarray(rival_track, dtype=float))


def _solve_exogenous_tyre(model: DecisionModel, rival_track: np.ndarray,
                          tyre: TyreDecisionContext) -> ExogenousSolution:
    """V[k, energy, wear]. Same reward and same pass model as the P0 path.

    Transitions:
        HOLD    e' = clip(e + recharge - spend)     w' = w + wear_hold
        ATTACK  e' = clip(e' - attack_cost)         w' = w + wear_attack
        PIT     at k == pit_at_laps_left, w' = 0 regardless of which was chosen

    The pit reset is applied to the CONTINUATION index, so the wear a lap
    generates is still paid for on that lap and only the future is cleared.
    """
    bins = model.bins
    wear = np.asarray(tyre.wear_grid, dtype=float)
    nb, nw, K = len(bins), len(wear), model.n_laps
    V = np.zeros((K + 1, nb, nw))
    best = np.full((K + 1, nb, nw), -1, dtype=np.int8)
    tau = np.zeros((K + 1, nb, nw))
    qbest = np.zeros((K + 1, nb, nw))

    own_next = np.clip(bins + model.recharge_per_lap - model.own_spend_per_lap,
                       0.0, E_STORE_MAX)
    wait_own = _bin_index(bins, own_next)
    att_own = _bin_index(bins, np.clip(own_next - model.attack_cost, 0.0, E_STORE_MAX))
    affordable = bins >= model.attack_cost * 0.75

    for k in range(1, K + 1):
        lap = int(np.clip(K - k, 0, len(rival_track) - 1))
        e_riv = float(rival_track[lap])
        reward = R_PASS * k / K
        v_next = V[k - 1]

        # Where this lap's wear lands, before any stop.
        next_hold = np.clip(wear + tyre.wear_per_lap_hold, 0.0, 1.0)
        next_att = np.clip(wear + tyre.wear_per_lap_attack, 0.0, 1.0)
        if tyre.pit_at_laps_left is not None and k == int(tyre.pit_at_laps_left):
            # The stop happens on the way out of this lap: everything continues
            # from a fresh tyre. This single line is the whole "close to pit"
            # effect -- there is no near-pit reward multiplier anywhere.
            next_hold = np.zeros_like(next_hold)
            next_att = np.zeros_like(next_att)
        lo_h, hi_h, f_h = _wear_weights(wear, next_hold)
        lo_a, hi_a, f_a = _wear_weights(wear, next_att)

        # P[zone, energy, wear]: the rival's speed does not depend on OUR wear,
        # but ours does, so the pass probability carries the axis.
        P = np.empty((len(model.zones), nb, nw))
        for zi, zm in enumerate(model.zones):
            for j, w in enumerate(wear):
                P[zi, :, j] = [p_pass(delta_v(zm, min(eo, model.attack_cost),
                                              min(e_riv, model.attack_cost),
                                              wear_own=float(w),
                                              wear_riv=tyre.rival_wear),
                                      model.gap_s, zm) for eo in bins]

        v_wait = ((1.0 - f_h)[None, :] * v_next[wait_own[:, None], lo_h[None, :]]
                  + f_h[None, :] * v_next[wait_own[:, None], hi_h[None, :]])
        v_fail = ((1.0 - f_a)[None, :] * v_next[att_own[:, None], lo_a[None, :]]
                  + f_a[None, :] * v_next[att_own[:, None], hi_a[None, :]])
        v_attack = P * reward + (1.0 - P) * (1.0 - model.fail_cost) * v_fail[None, :, :]
        v_attack = np.where(affordable[None, :, None], v_attack, -np.inf)

        z_best = np.argmax(v_attack, axis=0)
        v_best = np.max(v_attack, axis=0)
        take = v_best > v_wait
        V[k] = np.where(take, v_best, v_wait)
        best[k] = np.where(take, z_best, -1)
        qbest[k] = np.max(P, axis=0)
        vf = (1.0 - model.fail_cost) * v_fail
        tau[k] = np.clip((v_wait - vf) / np.maximum(reward - vf, 1e-9), 0.0, 1.0)

    return ExogenousSolution(V=V, best_zone=best, tau=tau, q=qbest, model=model,
                             rival_track=np.asarray(rival_track, dtype=float),
                             tyre=tyre)


def rival_energy_at_zone(belief, obs, track, n_laps: int, zone_name: str = "A"
                         ) -> np.ndarray:
    """Believed deployable energy of the rival where the pass would happen.

    Not the lap minimum and not the lap maximum: the value at the entry to the
    overtaking zone, which is the only moment that decides anything.
    """
    z = track.zone_by_name(zone_name)
    out = np.zeros(n_laps)
    for L in range(n_laps):
        m = np.flatnonzero(obs.lap == L)
        if len(m) == 0:
            out[L] = out[L - 1] if L else 0.0
            continue
        j = m[int(np.argmin(np.abs(obs.s[m] - z.s_straight_start)))]
        out[L] = float(belief.usable_mean[j])
    return out


def simulate_stint_exogenous(model: DecisionModel, sol_or_chooser, rng, n_laps: int,
                             e_own0: float, rival_track: np.ndarray,
                             blind: bool = False) -> dict:
    """Stint played against a rival whose energy schedule is fixed and real."""
    zmap = {zm.name: zm for zm in model.zones}
    e_own = float(e_own0)
    best_zone = max(model.zones, key=lambda z: z.braking_severity).name
    rows = []
    for k in range(n_laps, 0, -1):
        lap = n_laps - k
        e_riv = float(rival_track[min(lap, len(rival_track) - 1)])
        if blind:
            pick = best_zone if e_own >= model.attack_cost else None
        else:
            pick = sol_or_chooser.action(k, e_own)
        e_own = float(np.clip(e_own + model.recharge_per_lap - model.own_spend_per_lap,
                              0.0, E_STORE_MAX))
        q, hit = 0.0, False
        if pick is not None:
            zm = zmap[pick]
            spend = min(model.attack_cost, e_own)
            q = p_pass(delta_v(zm, spend, min(e_riv, model.attack_cost)), model.gap_s, zm)
            e_own = max(e_own - spend, 0.0)
            hit = bool(rng.random() < q)
        rows.append((lap, q, pick is not None, hit))
        if hit:
            return {"passed": 1, "lap_passed": lap + 1, "laps": rows}
    return {"passed": 0, "lap_passed": 0, "laps": rows}


def compare_exogenous(model: DecisionModel, sol: ExogenousSolution,
                      rival_track: np.ndarray, n_races: int, n_laps: int,
                      e_own0: float, seed: int = 0) -> dict:
    xr, bl = [], []
    for i in range(n_races):
        xr.append(simulate_stint_exogenous(
            model, sol, np.random.default_rng(seed * 977 + i), n_laps, e_own0,
            rival_track)["passed"])
        bl.append(simulate_stint_exogenous(
            model, None, np.random.default_rng(seed * 977 + i), n_laps, e_own0,
            rival_track, blind=True)["passed"])
    xr, bl = np.array(xr, float), np.array(bl, float)
    d = xr - bl
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    return {"xray_pass_rate": float(xr.mean()), "blind_pass_rate": float(bl.mean()),
            "mean_gain": float(d.mean()),
            "ci95": (float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)),
            "n_races": n_races}
