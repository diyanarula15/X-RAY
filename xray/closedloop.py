"""Closed-loop strategy evaluation: run the P2 policy inside the simulator.

Historical replay cannot answer "would this policy have been better?", because
the rival never reacted to the action X-RAY did not take. The simulator can: the
recommendation changes the deployment, the deployment changes the speed at the
braking point, and the pass then either happens or it does not.

The one thing that must not happen here is the estimator's blindfold slipping.
The simulator knows the rival's store exactly. A policy that reads it is not
X-RAY, it is an oracle, and it will look wonderful. So the belief every policy
acts on comes from `observe()` -> `estimate()` and nothing else:

    GroundTruth ──► observe()  ──► estimate()  ──► rival usable-energy belief
        │                                                    │
        │                                                    ▼
        └──────────────────────────► metrics only        P2 / baselines
                                                             │
                                                             ▼
                                                   AttackPlan (lap, zone, budget)
                                                             │
                                                             ▼
                                                   Simulator executes it

`oracle_p2_plan` is the one deliberate exception and is labelled `oracle` in its
own source field; it exists to separate "the strategy is limited" from "the
estimator is limited" and never shares a row with an X-RAY result.

Paired comparison is exact here. `Simulator` draws from its rng at exactly one
site -- the pass dice in the overtake resolver -- so the same seed gives every
policy the same sequence of coin flips, and a difference in outcome is a
difference in strategy rather than in luck.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .constants import E_STORE_MAX, P_MGUK_MAX, p_mguk_ceiling
from .decision import (DecisionModel, build_model, fail_cost_from_geometry,
                       make_bins, rival_energy_at_zone, solve_exogenous)
from .estimator import EstimatorError, estimate
from .observe import observe
from .policy import CONSERVATIVE, DeploymentPolicy, get_policy
from .sim import FOLLOWER, LEADER, Simulator
from .vehicle import VehicleParams


@dataclass(frozen=True)
class AttackPlan:
    """What a policy decided, in the form the simulator can execute."""
    source: str
    lap: int | None = None
    zone: str | None = None
    budget_j: float | None = None      # None = legacy "spend everything"
    detail: dict = field(default_factory=dict)

    @property
    def attacks(self) -> bool:
        return self.lap is not None and self.zone is not None


@dataclass(frozen=True)
class BudgetedAttack(DeploymentPolicy):
    """The existing attack hook, with a deployment BUDGET attached.

    `DeploymentPolicy.demand` already has the called-attack branch -- lap, zone,
    spend everything. P2's whole point is that "everything" is one choice among
    several, so this subclass stops the attack once `attack_budget_j` has left
    the store and falls back to the parent's normal behaviour.

    Spend is measured as MGU-K energy DELIVERED by the integrator, reported back
    through `DeploymentPolicy.note_deployed`. It used to be measured as store
    drawdown from the level at the first attack sample, justified by "the accel
    regime harvests nothing". That is false, and it is why P2's audit reported
    zone A as absorbing only 0.449 MJ against a 1.602 MJ calibrated budget: the
    zone straight ends in a braking zone, so the store is already refilling while
    the attack runs. Across zone A drawdown goes
    0 -> 0.4281 -> 0.4492 -> 0.4492 -> -0.1353 MJ, peaking mid-straight and
    ending NEGATIVE, while delivered energy rises monotonically
    0.0002 -> 0.4283 -> 0.4494 -> 0.4494 -> 0.4564 MJ. Against a non-monotone,
    eventually-negative quantity a budget of any size stops binding, so the
    "executable" figure was the peak of the drawdown curve rather than a
    measurement of deployment.
    """
    attack_budget_j: float | None = None
    _spend: dict = field(default_factory=dict, compare=False, repr=False)
    # Deployed joules per lap, filled in by the simulator after each step.
    _delivered: dict = field(default_factory=dict, compare=False, repr=False)

    def note_deployed(self, lap: int, energy_j: float) -> None:
        if energy_j > 0.0:
            self._delivered[lap] = self._delivered.get(lap, 0.0) + float(energy_j)

    def budget_exhausted(self, lap: int) -> bool:
        """Has this lap's allocation been spent? The production rule, readable.

        Exists so a test can OBSERVE the rule instead of restating it. The old
        test restated it -- recomputing `start - E >= budget` in its own spy -- and
        when the accounting moved from store drawdown to delivered energy the spy
        silently compared a ~0 J baseline against a ~2.4 MJ store level, so every
        sample classified as "still on the floor" and the budget looked inert. A
        test that reimplements the rule it checks cannot detect the rule changing,
        and would have passed the broken version too.
        """
        if self.attack_budget_j is None:
            return False
        start = self._spend.get((lap, self.attack_zone))
        if start is None:
            return False
        return (float(self._delivered.get(lap, 0.0)) - start) >= self.attack_budget_j

    def deployed_during_attack(self) -> float:
        """Energy put down on the attack lap, for tests and diagnostics."""
        if self.attack_lap is None:
            return 0.0
        return float(self._delivered.get(self.attack_lap, 0.0))

    def demand(self, track, s, v, E, gap_ahead, gap_behind, laps_left,
               is_corner=False, lap=-1):
        if (self.attack_budget_j is not None and self.attack_lap is not None
                and lap == self.attack_lap and not is_corner):
            zone = track.zone_at(s)
            in_zone = zone is not None and s < zone.s_straight_end and (
                self.attack_zone is None or zone.name == self.attack_zone)
            if in_zone:
                key = (lap, self.attack_zone)
                # Baseline the count at the first attack sample so earlier zones
                # on the same lap do not consume this zone's allocation.
                start = self._spend.setdefault(
                    key, float(self._delivered.get(lap, 0.0)))
                # The tactical allocation is a FLOOR on deployment while it
                # lasts, not a ceiling on it. Two wrong versions came first:
                # handing straight back to the base policy made the budget
                # non-binding (0.4 MJ and 1.6 MJ both put 2.812 MJ into the
                # lap), and cutting to zero made a small allocation an
                # ANTI-attack -- a 0.400 MJ "attack" capped deployment below
                # what the driver would have used anyway and dropped peak speed
                # from 365.3 to 364.9 km/h. Spending less than normal is not an
                # attack, it is a lift.
                base = DeploymentPolicy.demand(
                    self, track, s, v, E, gap_ahead, gap_behind, laps_left,
                    is_corner=is_corner, lap=-1)
                spent = float(self._delivered.get(lap, 0.0)) - start
                if spent >= self.attack_budget_j:
                    return base
                return max(base, min(P_MGUK_MAX, p_mguk_ceiling(v)))
        return DeploymentPolicy.demand(self, track, s, v, E, gap_ahead,
                                       gap_behind, laps_left,
                                       is_corner=is_corner, lap=lap)


def policy_for(base: DeploymentPolicy, plan: AttackPlan) -> DeploymentPolicy:
    """Attach a plan to a base driver policy, leaving the base untouched."""
    import dataclasses
    if not plan.attacks:
        return base
    vals = {f.name: getattr(base, f.name)
            for f in dataclasses.fields(DeploymentPolicy)}
    vals.update(name=f"{base.name}+{plan.source}", attack_lap=plan.lap,
                attack_zone=plan.zone, save_until=plan.lap)
    return BudgetedAttack(attack_budget_j=plan.budget_j, **vals)


# ------------------------------------------------------------------ belief
@dataclass(frozen=True)
class RivalBelief:
    """Blind reconstruction of the rival's deployable energy, per lap."""
    usable_by_lap_j: np.ndarray
    p10_by_lap_j: np.ndarray
    p90_by_lap_j: np.ndarray
    source: str
    n_laps: int


def blind_rival_belief(gt, rival: str, cfg: dict, seed: int,
                       n_particles: int = 400) -> RivalBelief:
    """observe() then estimate(). The simulator's stores are never read.

    This is the blindfold in executable form: the only argument that touches the
    ground truth is `observe`, which produces the public feed, and `estimate`
    has no access to anything else.
    """
    obs = observe(gt, rival, rate_hz=cfg["observe"]["rate_hz"],
                  speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=seed + 1)
    bel = estimate(obs, gt.track, n_particles=n_particles, seed=seed + 2)
    n = gt.n_laps
    usable = np.zeros(n)
    p10 = np.zeros(n)
    p90 = np.zeros(n)
    for lap in range(n):
        m = np.flatnonzero(obs.lap == lap)
        if len(m) == 0:
            usable[lap] = usable[lap - 1] if lap else 0.0
            p10[lap] = p10[lap - 1] if lap else 0.0
            p90[lap] = p90[lap - 1] if lap else 0.0
            continue
        i = int(m[0])
        usable[lap] = float(bel.usable_mean[i])
        p10[lap] = float(bel.usable_p10[i])
        p90[lap] = float(bel.usable_p90[i])
    return RivalBelief(usable, p10, p90, "observe+estimate (blind)", n)


def oracle_rival_belief(gt, rival: str) -> RivalBelief:
    """TRUE per-lap store. EVALUATION ONLY -- labelled, never mixed with X-RAY."""
    tr = gt.cars[rival]
    n = gt.n_laps
    e = np.array([float(tr.e_lap_open[min(l, len(tr.e_lap_open) - 1)])
                  for l in range(n)])
    return RivalBelief(e, e.copy(), e.copy(), "ORACLE simulator truth", n)


# ----------------------------------------------------------------- policies
# Every policy has the same signature: (belief, gt_for_geometry_only, cfg, car)
# -> AttackPlan. `gt` is passed for the TRACK and the lap count only; reading a
# store out of it is what `oracle_*` is for and nothing else may.
def hold_plan(belief, gt, cfg, car) -> AttackPlan:
    """Baseline A: never initiate an attack. The pre-decision-engine behaviour."""
    return AttackPlan(source="hold")


def _decision_model(gt, cfg, gap_s: float):
    params = VehicleParams.from_config(cfg)
    return build_model(gt.track, params, n_laps=gt.n_laps,
                       recharge_per_lap=2.2e6, rival_spend_per_lap=2.4e6,
                       gap_s=gap_s)


def _median_gap(gt) -> float:
    g = np.asarray(gt.gap_s, dtype=float)
    g = g[np.isfinite(g) & (g > 0.0)]
    return float(np.median(g)) if len(g) else 0.6


def fixed_cost_plan(belief, gt, cfg, car) -> AttackPlan:
    """Baseline B: the pre-P2 decision engine.

    `solve_exogenous` is the canonical P0/P1 solver and is used unchanged. Its
    ATTACK spends one fixed `attack_cost`, which is exactly the limitation P2
    exists to lift, so it is the comparator that matters.
    """
    model = _decision_model(gt, cfg, _median_gap(gt))
    sol = solve_exogenous(model, belief.usable_by_lap_j)
    e_own = 0.75 * E_STORE_MAX
    for lap in range(gt.n_laps):
        laps_left = gt.n_laps - lap
        zone = sol.action(laps_left, e_own)
        if zone is not None:
            return AttackPlan(source="fixed_cost", lap=lap, zone=zone,
                              budget_j=None,
                              detail={"attack_cost_j": model.attack_cost})
    return AttackPlan(source="fixed_cost")


def _p2_plan(belief, gt, cfg, car, source: str) -> AttackPlan:
    """P2: choose lap, zone AND deployment budget over the opportunity horizon."""
    from . import opportunity as opp_mod
    from . import tyres as tyremod

    gap = _median_gap(gt)
    model = _decision_model(gt, cfg, gap)
    zones = {z.name: z for z in model.zones}
    e_own = 0.75 * E_STORE_MAX
    tyre_params = tyremod.params_from_config(cfg, "MEDIUM")
    lap_m = float(gt.track.length)
    per_lap = max(len(model.zones), 1)
    transition = opp_mod.transition_model_from(
        model, tyre_params, distance_m=lap_m / per_lap, track_temp_c=30.0,
        opportunities_per_lap=per_lap)

    best = None
    for lap in range(gt.n_laps):
        horizon = []
        for k, (l, zname) in enumerate(
                [(l, z.name) for l in range(lap, gt.n_laps) for z in model.zones]):
            if len(horizon) >= 9:
                break
            e_riv = float(belief.usable_by_lap_j[min(l, belief.n_laps - 1)])
            p10 = float(belief.p10_by_lap_j[min(l, belief.n_laps - 1)])
            p90 = float(belief.p90_by_lap_j[min(l, belief.n_laps - 1)])
            horizon.append(opp_mod.DecisionOpportunity(
                opportunity_id=f"L{l}-{zname}", lap=l, zone_name=zname,
                decision_s=float(zones[zname].braking_severity) + 100.0 * len(horizon),
                decision_time_s=None, zone=zones[zname], gap_s=gap,
                own_usable_energy_j=e_own, rival_usable_energy_mean_j=e_riv,
                rival_usable_energy_p10_j=p10, rival_usable_energy_p90_j=p90,
                source="closed_loop_belief", confidence=0.5))
        if not horizon:
            continue
        sol = opp_mod.solve_opportunities(model, horizon, transition)
        if sol.decision == opp_mod.ATTACK:
            best = (lap, sol)
            break
    if best is None:
        return AttackPlan(source=source)
    lap, sol = best
    return AttackPlan(source=source, lap=lap, zone=sol.chosen.zone_name,
                      budget_j=float(sol.chosen.deployment_budget_j),
                      detail={"value_action": sol.value_action,
                              "value_hold": sol.value_hold,
                              "decision_margin": sol.decision_margin,
                              "action_consensus": sol.action_consensus,
                              "expected_regret": sol.expected_regret,
                              "ceiling_j": float(sol.opportunities[0].deploy_ceiling_j),
                              "action_values": dict(sol.action_values)})


def p2_plan(belief, gt, cfg, car) -> AttackPlan:
    return _p2_plan(belief, gt, cfg, car, "p2")


def max_deploy_plan(belief, gt, cfg, car) -> AttackPlan:
    """Baseline C: attack where P2 attacks, but always spend the zone maximum.

    Isolates the deployment-budget feature: same opportunity, same information,
    the only difference is that the budget is pinned to the ceiling.
    """
    plan = _p2_plan(belief, gt, cfg, car, "max_deploy")
    if not plan.attacks:
        return plan
    ceiling = plan.detail.get("ceiling_j")
    return replace(plan, budget_j=None if ceiling is None else float(ceiling))


def oracle_p2_plan(belief_ignored, gt, cfg, car) -> AttackPlan:
    """ORACLE upper bound: P2 with the rival's TRUE store. Evaluation only."""
    rival = LEADER if car == FOLLOWER else FOLLOWER
    truth = oracle_rival_belief(gt, rival)
    return _p2_plan(truth, gt, cfg, car, "oracle_p2")


POLICIES = {
    "hold": hold_plan,
    "fixed_cost": fixed_cost_plan,
    "max_deploy": max_deploy_plan,
    "p2": p2_plan,
}
ORACLE_POLICIES = {"oracle_p2": oracle_p2_plan}


# --------------------------------------------------------------- evaluation
@dataclass(frozen=True)
class EpisodeResult:
    """Realised outcome, measured from the simulator's own event record."""
    seed: int
    policy: str
    plan: AttackPlan
    attempts: int
    successes: int
    failures: int
    ahead_fraction: float
    finished_ahead: bool
    energy_deployed_j: float
    energy_unused_j: float
    attack_deployed_j: float
    mean_p_pass: float
    mean_delta_v: float

    @property
    def energy_per_success_mj(self) -> float:
        return (float("inf") if self.successes == 0
                else self.energy_deployed_j / self.successes / 1e6)


def run_episode(cfg: dict, seed: int, plan: AttackPlan, car: str = FOLLOWER,
                base_policy: DeploymentPolicy | None = None) -> EpisodeResult:
    """Execute one plan in the simulator and measure what actually happened."""
    base = base_policy or get_policy(cfg["sim"]["follower_policy"])
    rival = LEADER if car == FOLLOWER else FOLLOWER
    policies = {car: policy_for(base, plan),
                rival: get_policy(cfg["sim"]["leader_policy"])}
    gt = Simulator(cfg, policies=policies, seed=seed).run()

    ev = [e for e in gt.events if e.kind == "overtake" and e.car == car]
    succ = [e for e in ev if e.detail.get("success")]
    # Race order from cumulative distance, not from `gt.order`. `order` is a
    # uint8 index, so comparing it to the car-id STRING is silently always False
    # -- the first version of this metric reported ahead_fraction = 0.0000 for
    # every policy in every world and looked like a physics result. s_total is
    # the quantity the simulator itself uses to decide who is in front.
    rival_id = LEADER if car == FOLLOWER else FOLLOWER
    lead = np.asarray(gt.cars[car].s_total) - np.asarray(gt.cars[rival_id].s_total)
    ahead = float(np.mean(lead > 0.0)) if len(lead) else 0.0
    tr = gt.cars[car]
    deployed = float(np.sum(tr.deployed_lap))
    attack_dep = 0.0
    if plan.attacks and plan.lap is not None and plan.lap < len(tr.deployed_lap):
        attack_dep = float(tr.deployed_lap[plan.lap])
    return EpisodeResult(
        seed=seed, policy=plan.source, plan=plan,
        attempts=len(ev), successes=len(succ), failures=len(ev) - len(succ),
        ahead_fraction=ahead,
        finished_ahead=bool(len(lead) and lead[-1] > 0.0),
        energy_deployed_j=deployed,
        energy_unused_j=float(tr.E[-1]),
        attack_deployed_j=attack_dep,
        mean_p_pass=float(np.mean([e.detail["p"] for e in ev])) if ev else 0.0,
        mean_delta_v=float(np.mean([e.detail["delta_v"] for e in ev])) if ev else 0.0)


def evaluate(cfg: dict, seeds, policies=None, car: str = FOLLOWER,
             include_oracle: bool = False, n_particles: int = 400) -> dict:
    """Paired comparison: one world per seed, every policy run inside it.

    The belief is reconstructed ONCE per seed from a baseline observation and
    shared by every policy, so the policies differ in what they decide and not
    in what they were told. The staleness this implies is real and stated: our
    attack changes the gap, which changes the rival's tow, so a belief taken
    from the baseline run is slightly off for the attacking runs. It is shared
    deliberately -- giving each policy its own belief would confound the
    comparison with estimator noise.
    """
    fns = dict(POLICIES if policies is None else policies)
    if include_oracle:
        fns.update(ORACLE_POLICIES)
    rival = LEADER if car == FOLLOWER else FOLLOWER
    rows, beliefs = [], {}
    for seed in seeds:
        baseline = Simulator(cfg, seed=seed).run()
        try:
            belief = blind_rival_belief(baseline, rival, cfg, seed, n_particles)
        except EstimatorError as exc:
            # A refusal is a correct output, not an error to paper over.
            rows.append({"seed": seed, "policy": "REFUSED", "reason": str(exc)})
            continue
        beliefs[seed] = belief
        for name, fn in fns.items():
            plan = fn(belief, baseline, cfg, car)
            rows.append(run_episode(cfg, seed, plan, car))
    return {"rows": rows, "beliefs": beliefs, "car": car,
            "seeds": list(seeds), "policies": list(fns)}


# ------------------------------------------------------------------ worlds
# Every varied parameter is an EXISTING config key with an existing meaning.
# Nothing is widened beyond what the simulator already supports, and the ranges
# are stated rather than tuned: a randomisation chosen to make P2 look good is
# not a randomisation, it is a fixture.
WORLD_VARIATION = {
    "sim.start_gap_s": "0.25-0.95 s -- from 'on the gearbox' to 'out of range'",
    "sim.e_start_frac": "0.55-0.90 of the store, both cars",
    "sim.leader_policy": "AGGRESSIVE | BALANCED | MOM_ATTACK (existing presets)",
    "observe.speed_noise_ms": "0.25-0.45 m/s -- estimator input quality",
}


def world(cfg: dict, seed: int) -> dict:
    """One synthetic world. Deterministic in `seed`, so pairing is exact."""
    rng = np.random.default_rng(10_000 + seed)
    leader = str(rng.choice(["AGGRESSIVE", "BALANCED", "MOM_ATTACK"]))
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    out["sim"] = dict(cfg["sim"])
    out["observe"] = dict(cfg["observe"])
    out["sim"]["start_gap_s"] = float(rng.uniform(0.25, 0.95))
    out["sim"]["e_start_frac"] = float(rng.uniform(0.55, 0.90))
    out["sim"]["leader_policy"] = leader
    out["observe"]["speed_noise_ms"] = float(rng.uniform(0.25, 0.45))
    return out


def evaluate_worlds(cfg: dict, seeds, policies=None, car: str = FOLLOWER,
                    include_oracle: bool = False, n_particles: int = 400) -> dict:
    """Paired evaluation across randomised worlds, one world per seed."""
    rows, beliefs, worlds = [], {}, {}
    fns = dict(POLICIES if policies is None else policies)
    if include_oracle:
        fns.update(ORACLE_POLICIES)
    rival = LEADER if car == FOLLOWER else FOLLOWER
    for seed in seeds:
        wcfg = world(cfg, seed)
        worlds[seed] = {"start_gap_s": wcfg["sim"]["start_gap_s"],
                        "e_start_frac": wcfg["sim"]["e_start_frac"],
                        "leader_policy": wcfg["sim"]["leader_policy"],
                        "speed_noise_ms": wcfg["observe"]["speed_noise_ms"]}
        baseline = Simulator(wcfg, seed=seed).run()
        try:
            belief = blind_rival_belief(baseline, rival, wcfg, seed, n_particles)
        except EstimatorError as exc:
            rows.append({"seed": seed, "policy": "REFUSED", "reason": str(exc)})
            continue
        beliefs[seed] = belief
        for name, fn in fns.items():
            plan = fn(belief, baseline, wcfg, car)
            rows.append(run_episode(wcfg, seed, plan, car))
    return {"rows": rows, "beliefs": beliefs, "worlds": worlds, "car": car,
            "seeds": list(seeds), "policies": list(fns)}


def summarise(out: dict) -> dict:
    """Per-policy aggregates plus paired win/loss against a named baseline."""
    rows = [r for r in out["rows"] if not isinstance(r, dict)]
    by = {}
    for r in rows:
        by.setdefault(r.policy, []).append(r)
    agg = {}
    for name, rs in by.items():
        agg[name] = {
            "episodes": len(rs),
            "attempts": sum(r.attempts for r in rs),
            "successes": sum(r.successes for r in rs),
            "failures": sum(r.failures for r in rs),
            "mean_ahead_fraction": float(np.mean([r.ahead_fraction for r in rs])),
            "finished_ahead": sum(1 for r in rs if r.finished_ahead),
            "mean_energy_deployed_mj": float(np.mean(
                [r.energy_deployed_j for r in rs])) / 1e6,
            "mean_unused_mj": float(np.mean([r.energy_unused_j for r in rs])) / 1e6,
            "attacked": sum(1 for r in rs if r.plan.attacks),
            "interior_budget": sum(
                1 for r in rs if r.plan.attacks and r.plan.budget_j is not None
                and r.plan.detail.get("ceiling_j")
                and r.plan.budget_j < 0.999 * r.plan.detail["ceiling_j"]),
            "mean_p_pass": float(np.mean([r.mean_p_pass for r in rs])),
        }
    return agg


def paired(out: dict, a: str, b: str) -> dict:
    """Win/loss/tie of policy `a` against `b` on the same seeds."""
    rows = [r for r in out["rows"] if not isinstance(r, dict)]
    ra = {r.seed: r for r in rows if r.policy == a}
    rb = {r.seed: r for r in rows if r.policy == b}
    seeds = sorted(set(ra) & set(rb))
    wins = losses = ties = 0
    diffs = []
    for s in seeds:
        da = (ra[s].successes, ra[s].ahead_fraction)
        db = (rb[s].successes, rb[s].ahead_fraction)
        diffs.append(ra[s].ahead_fraction - rb[s].ahead_fraction)
        if da > db:
            wins += 1
        elif da < db:
            losses += 1
        else:
            ties += 1
    return {"a": a, "b": b, "n": len(seeds), "wins": wins, "losses": losses,
            "ties": ties,
            "mean_ahead_delta": float(np.mean(diffs)) if diffs else 0.0}
