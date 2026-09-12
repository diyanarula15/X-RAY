"""P2: which opportunity, and how much energy.

P1 answers what physically happens if a car arrives at one braking point with a
given energy, tyre and weather state. This module answers the question above
that: across the ordered overtaking opportunities still to come, is this the one
to take, and how many joules is it worth spending on it?

Three things make that a real optimisation rather than a threshold:

1. **The horizon is opportunities, not laps.** A lap is an accounting period; an
   overtaking chance is a physical event with a zone, a gap and a tyre state.
2. **ATTACK carries a budget.** The legacy action spent one fixed `attack_cost`,
   which quietly asserted that the only question was whether to attack. Zone A
   physically saturates at 1.42 MJ on this track -- asking for 2.6 MJ there buys
   exactly nothing, and a solver that cannot express 0.7 MJ cannot notice.
3. **Resources are priced by their consequences.** Energy spent now is energy a
   later opportunity does not have; wear added now is grip a later opportunity
   does not have, unless a pit stop throws it away first. No term in this file
   penalises energy or wear directly.

What this module is NOT: a second physics engine. Every speed comes from the P1
`ZoneModel` surfaces, every wear increment from `xray.tyres`, every pass
probability from `xray.overtake.p_pass`. It chooses between actions; it does not
compute what an action does.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

from .constants import E_STORE_MAX
from .decision import R_PASS, DecisionModel, ZoneModel, delta_v
from .overtake import p_pass
from .stint import PitContext
from . import tyres as tyremod

HOLD = "HOLD"
ATTACK = "ATTACK"

# Deployment levels as FRACTIONS of the zone's own physical ceiling, not absolute
# joules. The ceiling is `ZoneModel.energy_grid.max()` -- what the P1 calibration
# actually managed to deploy down that straight under the 2026 MGU-K curve -- so
# a fraction adapts to each zone and to any future regulation change, where a
# hard-coded "1.5 MJ" would silently become wrong. 1.0 is included so the legacy
# single-budget behaviour remains representable exactly.
DEFAULT_DEPLOYMENT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)

# Three-point discretisation of the rival's usable-energy posterior. ASSUMED
# quadrature: p10/p50/p90 with these weights is a standard coarse stand-in for a
# unimodal distribution. It is a decision-side approximation, not a claim about
# the estimator's posterior shape, and it is config-driven for that reason.
DEFAULT_SCENARIO_WEIGHTS = (0.3, 0.4, 0.3)

# State discretisation for the backward induction. The continuation value is a
# function of (energy, wear); evaluating it at full float resolution makes every
# path a distinct memo entry and the recursion exponential -- measured 45 s for a
# 12-opportunity horizon. These are the same energy bins the P0/P1 DP already
# uses (E_STORE_MAX / 19 ~ 210 kJ) plus a wear step finer than one opportunity's
# wear increment, so the grid resolves what the transitions actually do.
# `test_memoisation_resolution_does_not_change_the_answer` pins the
# approximation against a finer grid.
ENERGY_MEMO_QUANTUM_J = E_STORE_MAX / 19.0
WEAR_MEMO_QUANTUM = 0.02


def _snap(value: float, quantum: float) -> float:
    return float(round(float(value) / quantum) * quantum)


# --------------------------------------------------------------- the horizon
@dataclass(frozen=True)
class DecisionOpportunity:
    """One overtaking chance, with every input read at the SAME decision point.

    Field names follow P1 where P1 already named the thing (`own_tyre`,
    `rival_tyre`, `*_pit_context` are the P1 types, not new ones). The spec's
    `own_physics_context` is expressed as the tyre state plus the shared
    environment, because that is what P1 actually carries into a speed surface.
    """
    opportunity_id: str
    lap: int
    zone_name: str
    decision_s: float
    decision_time_s: float | None
    zone: ZoneModel

    gap_s: float
    own_usable_energy_j: float
    rival_usable_energy_mean_j: float
    rival_usable_energy_p10_j: float | None = None
    rival_usable_energy_p90_j: float | None = None

    own_tyre: object | None = None          # xray.tyres.TyreState
    rival_tyre: object | None = None
    own_pit_context: PitContext | None = None
    rival_pit_context: PitContext | None = None
    environment: object | None = None       # xray.environment.EnvironmentalState

    source: str = "decision_service"
    confidence: float = 0.0

    # Energy an attack can actually deploy over THIS zone from THIS entry speed,
    # measured by `decision.executable_attack_ceiling`. None when nobody measured
    # it, in which case the energy axis is used and the budget axis may offer
    # budgets the car cannot execute -- see `deploy_ceiling_j`.
    executable_ceiling_j: float | None = None
    # The same quantity measured at the RIVAL's observed entry speed. Separate
    # because the ceiling is set almost entirely by entry speed, so sharing one
    # number between two cars makes our own arrival speed move the rival's
    # predicted speed -- which it cannot.
    rival_executable_ceiling_j: float | None = None

    @property
    def own_wear(self) -> float | None:
        return None if self.own_tyre is None else float(self.own_tyre.wear_fraction)

    @property
    def rival_wear(self) -> float | None:
        return None if self.rival_tyre is None else float(self.rival_tyre.wear_fraction)

    @property
    def deploy_ceiling_j(self) -> float:
        """Most energy an attack here can actually deploy.

        Prefers the measured executable ceiling. The fallback,
        `max(zone.energy_grid)`, is the axis `calibrate_zone` swept, and that
        covers RUN-UP PLUS STRAIGHT from the previous corner exit -- 1700 m
        entered at 200 km/h for Circuit Sigma zone A, where the low-speed part
        sits under the full 350 kW. An attack in-race spans the 1100 m straight
        entered at 314 km/h, and the taper is 0 kW after 387 m of it: 0.448 MJ,
        not 1.602 MJ. Using the calibration axis as the budget ceiling is what
        made P2 offer budgets up to 1.602 MJ in a zone that could execute 0.45,
        so every budget above roughly a quarter of the axis was the same action
        wearing different labels.
        """
        if self.executable_ceiling_j is not None:
            return float(self.executable_ceiling_j)
        return float(np.max(self.zone.energy_grid))

    @property
    def rival_deploy_ceiling_j(self) -> float | None:
        """None means "do not clip" -- the speed surface clamps on its own axis.

        Deliberately NOT defaulted to our own ceiling. It was, implicitly, and it
        made HOLD's predicted delta_v move with our deployment ceiling: on the
        P1 service fixture, tightening our zone-A ceiling from the calibration
        axis (1.422 MJ) to the measured executable value (0.354 MJ) moved HOLD's
        delta_v from -10.55 to -2.31 m/s, because `min(rival_energy, ceiling)`
        was quietly throttling the rival with our number. HOLD deploys nothing,
        so nothing about our ceiling may touch it.
        """
        return (None if self.rival_executable_ceiling_j is None
                else float(self.rival_executable_ceiling_j))

    @property
    def deploy_ceiling_is_measured(self) -> bool:
        """False means the ceiling is the calibration axis, not this context."""
        return self.executable_ceiling_j is not None


def order_opportunities(opps: Sequence[DecisionOpportunity]) -> list[DecisionOpportunity]:
    """Strict race chronology, and a loud failure on a duplicate id.

    Sorted by (lap, decision_s) rather than by decision_time_s, because time can
    be missing on a synthetic fixture while the geometry never is. Ties are
    impossible after the id check: two opportunities at the same lap and the same
    metre are the same opportunity.
    """
    seen = {}
    for o in opps:
        if o.opportunity_id in seen:
            raise ValueError(f"duplicate opportunity_id {o.opportunity_id!r}")
        seen[o.opportunity_id] = o
    out = sorted(opps, key=lambda o: (int(o.lap), float(o.decision_s)))
    for a, b in zip(out, out[1:]):
        if (a.lap, a.decision_s) == (b.lap, b.decision_s):
            raise ValueError(
                f"two opportunities at lap {a.lap} s={a.decision_s}: "
                f"{a.opportunity_id} and {b.opportunity_id}")
    return out


# ---------------------------------------------------------------- the action
@dataclass(frozen=True)
class DecisionAction:
    kind: str                       # HOLD | ATTACK
    zone_name: str | None = None
    deployment_budget_j: float = 0.0

    def __post_init__(self):
        if self.kind not in (HOLD, ATTACK):
            raise ValueError(f"unknown action kind {self.kind!r}")
        if self.kind == HOLD and self.deployment_budget_j != 0.0:
            raise ValueError("HOLD spends no tactical energy by definition")
        if self.kind == ATTACK and not self.zone_name:
            raise ValueError("ATTACK needs a zone")

    @property
    def label(self) -> str:
        if self.kind == HOLD:
            return HOLD
        return f"ATTACK({self.zone_name}, {self.deployment_budget_j / 1e6:.3f} MJ)"


@dataclass(frozen=True)
class ActionOutcome:
    """What the P1 physics says this action buys at this opportunity."""
    action: DecisionAction
    feasible: bool
    infeasible_reason: str
    requested_budget_j: float
    actual_deployed_j: float
    saturated: bool
    saturation_reason: str
    own_speed_mps: float
    rival_speed_mps: float
    delta_v_mps: float
    pass_probability: float
    pass_model_calibration: str = "synthetic"


def enumerate_actions(opp: DecisionOpportunity,
                      fractions: Sequence[float] = DEFAULT_DEPLOYMENT_FRACTIONS
                      ) -> list[DecisionAction]:
    """HOLD plus one ATTACK per deployment level, budgets in joules.

    Budgets are generated even when they are unaffordable or physically
    impossible; `evaluate_action` marks those infeasible and the solver masks
    them. Generating and rejecting is deliberate -- it makes "we considered
    2.6 MJ and it was not deployable" a reportable fact rather than an absence.
    """
    ceiling = opp.deploy_ceiling_j
    out = [DecisionAction(HOLD)]
    for f in sorted(set(float(x) for x in fractions)):
        if f <= 0.0:
            continue
        out.append(DecisionAction(ATTACK, opp.zone_name, float(f) * ceiling))
    return out


def evaluate_action(opp: DecisionOpportunity, action: DecisionAction,
                    own_wear: float | None = None,
                    rival_usable_energy_j: float | None = None,
                    rival_wear: float | None = None,
                    own_usable_energy_j: float | None = None) -> ActionOutcome:
    """Ask P1 what this action does. No physics is computed here.

    Feasibility has three separate causes and they are reported separately,
    because "you cannot afford it" and "the straight is too short for it" are
    different facts about the same number of joules:

      affordability  budget <= current usable energy
      deployability  budget <= what the zone physically absorbed during
                     calibration, which already encodes the 2026 MGU-K ceiling
                     and the zone's own length and speed profile
      store bounds   enforced upstream by the P1 integrator; the calibration
                     could not have returned energy the store could not supply
    """
    own_e = (opp.own_usable_energy_j if own_usable_energy_j is None
             else float(own_usable_energy_j))
    riv_e = (opp.rival_usable_energy_mean_j if rival_usable_energy_j is None
             else float(rival_usable_energy_j))
    w_own = opp.own_wear if own_wear is None else own_wear
    w_riv = opp.rival_wear if rival_wear is None else rival_wear
    ceiling = opp.deploy_ceiling_j
    requested = float(action.deployment_budget_j)

    reason, feasible = "", True
    if action.kind == ATTACK:
        if requested <= 0.0:
            feasible, reason = False, "attack with no budget"
        elif requested > own_e + 1e-9:
            feasible, reason = False, (
                f"budget {requested/1e6:.3f} MJ exceeds usable {own_e/1e6:.3f} MJ")

    # What actually goes out of the battery: never more than is there, never
    # more than the zone can physically take.
    actual = float(np.clip(min(requested, own_e, ceiling), 0.0, E_STORE_MAX))
    if action.kind == HOLD:
        actual = 0.0

    saturated, sat_reason = False, ""
    if action.kind == ATTACK and feasible:
        if requested > ceiling + 1e-6:
            saturated = True
            sat_reason = (f"zone absorbs at most {ceiling/1e6:.3f} MJ; "
                          f"{(requested - ceiling)/1e6:.3f} MJ undeployable")

    # Both speeds off the P1 surfaces, each car on its own curve and its own wear.
    own_v = float(opp.zone.own_speed(actual, w_own))
    riv_cap = opp.rival_deploy_ceiling_j
    riv_deployed = riv_e if riv_cap is None else min(riv_e, riv_cap)
    riv_v = float(opp.zone.rival_speed(riv_deployed, w_riv))
    dv = own_v - riv_v
    q = float(p_pass(dv, opp.gap_s, opp.zone)) if action.kind == ATTACK else 0.0
    return ActionOutcome(
        action=action, feasible=feasible, infeasible_reason=reason,
        requested_budget_j=requested, actual_deployed_j=actual,
        saturated=saturated, saturation_reason=sat_reason,
        own_speed_mps=own_v, rival_speed_mps=riv_v, delta_v_mps=dv,
        pass_probability=q)


# ------------------------------------------------------------- transitions
@dataclass(frozen=True)
class TransitionModel:
    """Per-opportunity state dynamics, all of it delegated to P0/P1.

    `recharge_per_step_j` and `normal_spend_per_step_j` are the DecisionModel's
    per-lap figures divided by the number of opportunities in a lap: the car
    harvests and deploys continuously, and an opportunity is a fraction of a lap.
    The wear increments come from integrating `tyres.advance` over the distance
    between opportunities at two utilisations -- the same function the simulator
    steps and P1's lap-level DP already uses.
    """
    recharge_per_step_j: float
    normal_spend_per_step_j: float
    wear_hold_per_step: float
    wear_attack_per_step: float
    fail_cost: float
    compound: str = "UNKNOWN"

    def __post_init__(self):
        if self.wear_attack_per_step < self.wear_hold_per_step:
            raise ValueError("attacking cannot wear the tyre less than holding")


def transition_model_from(model: DecisionModel, tyre_params, distance_m: float,
                          track_temp_c: float, wetness: float = 0.0,
                          opportunities_per_lap: int = 1,
                          start_wear: float = 0.0) -> TransitionModel:
    """Build the per-opportunity dynamics from canonical P1 pieces."""
    n = max(int(opportunities_per_lap), 1)
    hold = tyremod.wear_per_lap(tyre_params, distance_m, track_temp_c,
                                tyremod.ASSUMED_UTIL_NORMAL, wetness,
                                start_wear=start_wear)
    extra = tyremod.attack_extra_wear(tyre_params, distance_m, track_temp_c,
                                      wetness, start_wear=start_wear)
    return TransitionModel(
        recharge_per_step_j=float(model.recharge_per_lap) / n,
        normal_spend_per_step_j=float(model.own_spend_per_lap) / n,
        wear_hold_per_step=hold, wear_attack_per_step=hold + extra,
        fail_cost=float(model.fail_cost), compound=tyre_params.compound)


def next_energy(e_j: float, attack_deployed_j: float, tm: TransitionModel) -> float:
    """E' = clip(E - attack - normal + harvest, 0, E_store_max).

    The P0 store semantics, kept separate: tactical attack energy, normal
    deployment and harvest are three different quantities and are never merged.
    """
    return float(np.clip(e_j - float(attack_deployed_j)
                         - tm.normal_spend_per_step_j + tm.recharge_per_step_j,
                         0.0, E_STORE_MAX))


def next_wear(wear: float, attacked: bool, tm: TransitionModel,
              pit_before_next: bool = False) -> float:
    """Wear advances; a pit throws it away. Nothing here is a penalty.

    `pit_before_next` comes from a CAUSAL PitContext only -- the caller is
    responsible for never passing an oracle-derived stop, and `stint.require_causal`
    is the gate that enforces it upstream.
    """
    if pit_before_next:
        return 0.0
    inc = tm.wear_attack_per_step if attacked else tm.wear_hold_per_step
    return float(np.clip(float(wear) + inc, 0.0, 1.0))


# ------------------------------------------------------------- the solver
@dataclass(frozen=True)
class ScenarioSet:
    """Discrete rival-energy scenarios with normalised weights."""
    energies_j: tuple
    weights: tuple
    source: str = "p10_p50_p90_quadrature"

    def __post_init__(self):
        if len(self.energies_j) != len(self.weights):
            raise ValueError("scenario energies and weights differ in length")
        total = float(sum(self.weights))
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"scenario weights must sum to 1, got {total}")


def scenarios_for(opp: DecisionOpportunity,
                  weights: Sequence[float] = DEFAULT_SCENARIO_WEIGHTS) -> ScenarioSet:
    """p10 / mean / p90 of the rival's usable energy, or a point mass.

    Collapses to a single scenario when the estimator gave no interval or the
    interval has no width, which is what makes the belief-aware solver reduce to
    the deterministic one rather than approximate it.
    """
    mean = float(opp.rival_usable_energy_mean_j)
    lo, hi = opp.rival_usable_energy_p10_j, opp.rival_usable_energy_p90_j
    if lo is None or hi is None or abs(float(hi) - float(lo)) < 1.0:
        return ScenarioSet((mean,), (1.0,), source="point_mass_no_interval")
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return ScenarioSet((float(lo), mean, float(hi)), tuple(float(x) for x in w))


@dataclass(frozen=True)
class OpportunitySolution:
    """Solver output. Every reported number is read from here, never recomputed."""
    opportunities: tuple
    chosen: DecisionAction
    chosen_outcome: ActionOutcome
    value_action: float
    value_hold: float
    action_values: dict          # action label -> belief-weighted value
    outcomes: dict               # action label -> ActionOutcome (mean scenario)
    next_best_action: DecisionAction | None
    next_best_value: float
    decision_margin: float
    action_consensus: float
    expected_regret: float
    policy_posterior: tuple
    scenarios: ScenarioSet
    robustness: dict
    horizon_len: int
    pass_model_calibration: str = "synthetic"

    @property
    def decision(self) -> str:
        return self.chosen.kind


def _future_value(opps, i, e_j, wear, tm, scen_energy_j, fractions,
                  pit_at_index, memo):
    """V_i(E, W) by backward induction over the remaining opportunities.

    Same shape as the P1 lap DP and the same reward semantics: passing earlier
    is worth more because more of the race is left to hold the place. The only
    P2 change is that the max runs over (zone, budget) pairs instead of zones,
    and that the state carries energy AND wear forward through P1 transitions.
    """
    if i >= len(opps):
        return 0.0
    # Snap the state, then USE the snapped value, so the memo and the evaluation
    # always agree about which state this is.
    e_j = float(np.clip(_snap(e_j, ENERGY_MEMO_QUANTUM_J), 0.0, E_STORE_MAX))
    wear = float(np.clip(_snap(wear, WEAR_MEMO_QUANTUM), 0.0, 1.0))
    key = (i, e_j, wear)
    if key in memo:
        return memo[key]

    opp = opps[i]
    n = len(opps)
    reward = R_PASS * (n - i) / n
    pit_now = (pit_at_index is not None and i == int(pit_at_index))

    best = -np.inf
    for action in enumerate_actions(opp, fractions):
        out = evaluate_action(opp, action, own_wear=wear,
                              rival_usable_energy_j=scen_energy_j,
                              own_usable_energy_j=e_j)
        if not out.feasible:
            continue
        attacked = action.kind == ATTACK
        e_next = next_energy(e_j, out.actual_deployed_j, tm)
        w_next = next_wear(wear, attacked, tm, pit_before_next=pit_now)
        cont = _future_value(opps, i + 1, e_next, w_next, tm, scen_energy_j,
                             fractions, pit_at_index, memo)
        if attacked:
            q = out.pass_probability
            v = q * reward + (1.0 - q) * (1.0 - tm.fail_cost) * cont
        else:
            v = cont
        best = max(best, v)
    memo[key] = best if np.isfinite(best) else 0.0
    return memo[key]


def solve_opportunities(model: DecisionModel,
                        opportunities: Sequence[DecisionOpportunity],
                        transition: TransitionModel,
                        deployment_fractions: Sequence[float] = DEFAULT_DEPLOYMENT_FRACTIONS,
                        scenario_weights: Sequence[float] = DEFAULT_SCENARIO_WEIGHTS,
                        pit_at_index: int | None = None,
                        pass_model_calibration: str = "synthetic"
                        ) -> OpportunitySolution:
    """Choose zone AND deployment budget at the first opportunity.

    Belief-aware in the QMDP sense that `xray.qmdp.decide` already uses on the
    lap problem:

        Q(a) = sum_j w_j Q(a | rival_energy_j)

    `qmdp.decide` is not called directly because its Q is single-lap, budgetless
    and wear-blind (`_q_attack` reads `delta_v` with no wear argument), so it
    cannot express a P2 action. The expectation form and the `consensus`
    definition are deliberately identical to it.
    """
    opps = order_opportunities(opportunities)
    if not opps:
        raise ValueError("no opportunities to solve")
    first = opps[0]
    scen = scenarios_for(first, scenario_weights)
    actions = enumerate_actions(first, deployment_fractions)

    # Q(a | scenario) for every action and every scenario.
    per_scenario, feasible_actions = {}, []
    for action in actions:
        probe = evaluate_action(first, action)
        if not probe.feasible:
            continue
        feasible_actions.append(action)

    n = len(opps)
    reward = R_PASS * n / n
    pit_now = (pit_at_index is not None and int(pit_at_index) == 0)
    for si, (e_riv, w_s) in enumerate(zip(scen.energies_j, scen.weights)):
        memo = {}
        vals = {}
        for action in feasible_actions:
            out = evaluate_action(first, action, rival_usable_energy_j=e_riv)
            attacked = action.kind == ATTACK
            e_next = next_energy(first.own_usable_energy_j, out.actual_deployed_j,
                                 transition)
            w_next = next_wear(first.own_wear or 0.0, attacked, transition,
                               pit_before_next=pit_now)
            cont = _future_value(opps, 1, e_next, w_next, transition, e_riv,
                                 deployment_fractions, pit_at_index, memo)
            if attacked:
                q = out.pass_probability
                vals[action.label] = (q * reward
                                      + (1.0 - q) * (1.0 - transition.fail_cost) * cont)
            else:
                vals[action.label] = cont
        per_scenario[si] = vals

    # Belief-weighted value of each action.
    action_values = {}
    for action in feasible_actions:
        action_values[action.label] = float(sum(
            w * per_scenario[si][action.label]
            for si, w in enumerate(scen.weights)))

    by_label = {a.label: a for a in feasible_actions}
    ranked = sorted(action_values.items(), key=lambda kv: kv[1], reverse=True)
    chosen_label, chosen_value = ranked[0]
    chosen = by_label[chosen_label]
    next_best = by_label[ranked[1][0]] if len(ranked) > 1 else None
    next_best_value = ranked[1][1] if len(ranked) > 1 else float("-inf")

    # consensus: belief mass whose own optimal action is the one we report.
    # Same definition as qmdp.decide's `consensus`.
    consensus, regret, posterior = 0.0, 0.0, []
    for si, w in enumerate(scen.weights):
        vals = per_scenario[si]
        best_label = max(vals, key=vals.get)
        if best_label == chosen_label:
            consensus += w
        # expected regret: what this scenario would have gained by acting on its
        # own optimum instead of the reported one.
        regret += w * (vals[best_label] - vals[chosen_label])
        posterior.append({
            "scenario_index": si,
            "rival_usable_energy_mj": scen.energies_j[si] / 1e6,
            "weight": float(w),
            "optimal_action": best_label,
            "optimal_value": float(vals[best_label]),
            "chosen_action_value": float(vals[chosen_label]),
        })

    outcomes = {a.label: evaluate_action(first, a) for a in actions}
    hold_label = DecisionAction(HOLD).label
    value_hold = float(action_values.get(hold_label, 0.0))

    spread = (0.0 if len(scen.energies_j) < 2
              else float(max(scen.energies_j) - min(scen.energies_j)))
    robustness = {
        "action_consensus": float(consensus),
        "decision_margin": float(chosen_value - next_best_value)
        if np.isfinite(next_best_value) else None,
        "expected_regret": float(regret),
        "n_scenarios": len(scen.energies_j),
        "rival_energy_spread_mj": spread / 1e6,
        "scenario_source": scen.source,
        "n_feasible_actions": len(feasible_actions),
        "n_actions_considered": len(actions),
        "horizon_len": len(opps),
    }
    return OpportunitySolution(
        opportunities=tuple(opps), chosen=chosen,
        chosen_outcome=outcomes[chosen_label],
        value_action=float(chosen_value), value_hold=value_hold,
        action_values=action_values, outcomes=outcomes,
        next_best_action=next_best,
        next_best_value=float(next_best_value) if np.isfinite(next_best_value) else 0.0,
        decision_margin=float(chosen_value - next_best_value)
        if np.isfinite(next_best_value) else 0.0,
        action_consensus=float(consensus), expected_regret=float(regret),
        policy_posterior=tuple(posterior), scenarios=scen,
        robustness=robustness, horizon_len=len(opps),
        pass_model_calibration=pass_model_calibration)
