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


def calibrate_zone(track: Track, params: VehicleParams, zone: Zone,
                   n_points: int = 7, dt: float = 0.005) -> ZoneModel:
    """Run the real longitudinal model down the zone straight at a range of
    deployment budgets and record where it ends up.

    Nothing here is a fitted constant: the map from energy to end-of-straight
    speed is whatever the physics says it is.
    """
    set_car_mass(params.mass_car)
    s_from, entry_v = _run_up(track, zone)
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
            out = step(track, st, params, demand, dt)
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


def delta_v(zm: ZoneModel, e_own: float, e_riv: float) -> float:
    """Speed advantage at the braking point implied by two energy states."""
    return float(np.interp(e_own, zm.energy_grid, zm.speed_grid)
                 - np.interp(e_riv, zm.energy_grid, zm.speed_grid))


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

    def _lap_of(self, laps_left: int) -> int:
        return int(np.clip(self.model.n_laps - laps_left, 0, len(self.rival_track) - 1))

    def action(self, laps_left: int, e_own: float):
        k = int(np.clip(laps_left, 0, self.V.shape[0] - 1))
        i = int(_bin_index(self.model.bins, e_own))
        z = int(self.best_zone[k, i])
        return None if z < 0 else self.model.zones[z].name

    def quality(self, laps_left: int, e_own: float) -> float:
        k = int(np.clip(laps_left, 0, self.V.shape[0] - 1))
        return float(self.q[k, int(_bin_index(self.model.bins, e_own))])

    def threshold(self, laps_left: int, e_own: float) -> float:
        k = int(np.clip(laps_left, 0, self.V.shape[0] - 1))
        return float(self.tau[k, int(_bin_index(self.model.bins, e_own))])


def explain_exogenous_action(sol: ExogenousSolution, laps_left: int,
                             own_usable_energy_j: float) -> dict:
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
    value_wait = float(v_next[wait_i])
    value_fail = float((1.0 - model.fail_cost) * v_next[attack_i])
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
        dv = delta_v(zm, own_deploy_j, rival_deploy_j)
        q = p_pass(dv, model.gap_s, zm)
        value_hypothetical = float(q * reward + (1.0 - q) * value_fail)
        value_attack = value_hypothetical if affordable else -np.inf
        rows.append({
            "zone_index": zi,
            "zone": zm.name,
            "pass_probability": float(q),
            "predicted_delta_v_mps": float(dv),
            "predicted_own_speed_mps": float(np.interp(own_deploy_j, zm.energy_grid,
                                                       zm.speed_grid)),
            "predicted_rival_speed_mps": float(np.interp(rival_deploy_j, zm.energy_grid,
                                                         zm.speed_grid)),
            "value_attack": float(value_attack),
            # What the attack would be worth if it were affordable. Kept apart
            # from `value_attack` so a rejected attack cannot be displayed as a
            # real one, and so the panel still has a number to show.
            "value_attack_hypothetical": value_hypothetical,
            "affordable": bool(affordable),
        })
    selected = int(sol.best_zone[k, i]) if k < sol.best_zone.shape[0] else -1
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
        "attack_threshold": float(sol.threshold(k, own_usable_energy_j)),
        "value_wait": value_wait,
        "value_attack": float(best_row["value_attack"]) if best_row else -np.inf,
        "attack_affordable": bool(affordable),
        "best_zone": None if selected < 0 else model.zones[selected].name,
        "decision": "HOLD" if selected < 0 else "ATTACK",
        "best_attack": best_row,
        "zones": rows,
    }


def solve_exogenous(model: DecisionModel, rival_track: np.ndarray) -> ExogenousSolution:
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
