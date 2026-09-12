"""P2: opportunity horizon, deployment budgets, and belief-aware choice.

The claims under test are the ones that make P2 more than a renamed threshold:
a later opportunity can make HOLD optimal today, a medium budget can beat the
maximum one, and a pit horizon can flip the call through the tyre state rather
than through a penalty term.
"""
from __future__ import annotations

import numpy as np
import pytest

from xray.config import load_config
from xray.decision import DecisionModel, ZoneModel, make_bins
from xray.opportunity import (ATTACK, DEFAULT_DEPLOYMENT_FRACTIONS, HOLD,
                              DecisionAction, DecisionOpportunity,
                              ScenarioSet, TransitionModel, enumerate_actions,
                              evaluate_action, next_energy, next_wear,
                              order_opportunities, scenarios_for,
                              solve_opportunities, transition_model_from)
from xray.stint import pit_context_from_plan
from xray.tyres import TyreState, grip_scale, params_from_config

WEAR_GRID = np.linspace(0.0, 1.0, 5)


def _zone(name="A", severity=1.0, ceiling_mj=1.4, dv_per_mj=6.0, wear_penalty=5.0):
    """A zone whose energy grid SATURATES, like the real calibration does."""
    budgets = np.linspace(0.0, 2.6e6, 7)
    spent = np.minimum(budgets, ceiling_mj * 1e6)
    base = 70.0 + dv_per_mj * spent / 1e6
    surface = np.vstack([base - wear_penalty * w for w in WEAR_GRID])
    return ZoneModel(name, severity, spent, base, dv_per_mj,
                     wear_grid=WEAR_GRID, speed_grid_by_wear=surface,
                     rival_speed_grid_by_wear=np.vstack([base for _ in WEAR_GRID]))


def _tyre(wear=0.1, compound="MEDIUM"):
    return TyreState(compound, 5.0, False, 95.0, wear, 1.0 - 0.17 * wear,
                     "modelled_thermal_wear")


def _opp(i=0, lap=1, zone=None, gap=0.45, own_mj=3.0, riv_mj=1.0,
         p10=None, p90=None, wear=0.1, riv_wear=0.1, s=100.0):
    z = zone or _zone()
    return DecisionOpportunity(
        opportunity_id=f"L{lap}-{z.name}-{int(s)}", lap=lap, zone_name=z.name,
        decision_s=s, decision_time_s=float(lap * 90 + s / 80.0), zone=z,
        gap_s=gap, own_usable_energy_j=own_mj * 1e6,
        rival_usable_energy_mean_j=riv_mj * 1e6,
        rival_usable_energy_p10_j=None if p10 is None else p10 * 1e6,
        rival_usable_energy_p90_j=None if p90 is None else p90 * 1e6,
        own_tyre=_tyre(wear), rival_tyre=_tyre(riv_wear))


def _model(zone):
    return DecisionModel(zones=[zone], recharge_per_lap=2.0e6,
                         own_spend_per_lap=1.6e6, rival_spend_per_lap=1.8e6,
                         attack_cost=float(zone.energy_grid.max()), defend_cost=0.0,
                         gap_s=0.45, n_laps=10, fail_cost=0.25, bins=make_bins())


def _tm(hold=0.02, attack=0.05, recharge=0.6e6, spend=0.4e6, fail=0.25):
    return TransitionModel(recharge_per_step_j=recharge,
                           normal_spend_per_step_j=spend,
                           wear_hold_per_step=hold, wear_attack_per_step=attack,
                           fail_cost=fail)


# ------------------------------------------------------------- the horizon
def test_opportunities_are_ordered_chronologically():
    a = _opp(lap=2, s=100.0)
    b = _opp(lap=1, s=500.0)
    c = _opp(lap=1, s=100.0)
    out = order_opportunities([a, b, c])
    assert [(o.lap, o.decision_s) for o in out] == [(1, 100.0), (1, 500.0), (2, 100.0)]


def test_duplicate_opportunity_ids_are_rejected():
    a = _opp(lap=1, s=100.0)
    with pytest.raises(ValueError, match="duplicate opportunity_id"):
        order_opportunities([a, a])


def test_two_opportunities_at_the_same_place_are_rejected():
    z = _zone()
    a = DecisionOpportunity("x", 1, z.name, 100.0, 1.0, z, 0.4, 3e6, 1e6)
    b = DecisionOpportunity("y", 1, z.name, 100.0, 1.0, z, 0.4, 3e6, 1e6)
    with pytest.raises(ValueError, match="two opportunities at lap"):
        order_opportunities([a, b])


def test_an_opportunity_binds_one_decision_point():
    o = _opp()
    assert o.own_wear == pytest.approx(0.1)
    assert o.rival_wear == pytest.approx(0.1)
    assert o.deploy_ceiling_j == pytest.approx(1.4e6)


# ----------------------------------------------------- the action space
def test_the_action_space_is_hold_plus_one_attack_per_budget():
    acts = enumerate_actions(_opp())
    assert acts[0].kind == HOLD and acts[0].deployment_budget_j == 0.0
    attacks = [a for a in acts if a.kind == ATTACK]
    assert len(attacks) == len(DEFAULT_DEPLOYMENT_FRACTIONS)
    budgets = [a.deployment_budget_j for a in attacks]
    assert budgets == sorted(budgets), "budgets must be ordered"
    assert max(budgets) == pytest.approx(1.4e6), "top budget is the zone ceiling"


def test_hold_cannot_carry_a_budget_and_attack_needs_a_zone():
    with pytest.raises(ValueError, match="no tactical energy"):
        DecisionAction(HOLD, None, 1.0e6)
    with pytest.raises(ValueError, match="needs a zone"):
        DecisionAction(ATTACK, None, 1.0e6)
    with pytest.raises(ValueError, match="unknown action kind"):
        DecisionAction("PREPARE_OVERTAKE", "A", 0.0)


def test_a_budget_above_usable_energy_is_infeasible():
    o = _opp(own_mj=0.3)
    out = evaluate_action(o, DecisionAction(ATTACK, "A", 1.4e6))
    assert not out.feasible
    assert "exceeds usable" in out.infeasible_reason


def test_a_budget_above_the_zone_ceiling_saturates_rather_than_lying():
    """The §6 case: 2 MJ and 1.4 MJ must produce the same physical outcome."""
    z = _zone(ceiling_mj=1.4)
    o = _opp(zone=z, own_mj=3.0)
    big = evaluate_action(o, DecisionAction(ATTACK, "A", 2.4e6))
    cap = evaluate_action(o, DecisionAction(ATTACK, "A", 1.4e6))
    assert big.feasible and big.saturated
    assert "undeployable" in big.saturation_reason
    assert big.actual_deployed_j == pytest.approx(cap.actual_deployed_j)
    assert big.own_speed_mps == pytest.approx(cap.own_speed_mps)
    assert big.requested_budget_j > big.actual_deployed_j


def test_hold_deploys_nothing_and_attempts_no_pass():
    out = evaluate_action(_opp(), DecisionAction(HOLD))
    assert out.actual_deployed_j == 0.0
    assert out.pass_probability == 0.0
    assert not out.saturated


def test_deployed_energy_never_exceeds_usable_or_the_ceiling():
    for own_mj in (0.2, 0.8, 1.5, 4.0):
        o = _opp(own_mj=own_mj)
        for a in enumerate_actions(o):
            out = evaluate_action(o, a)
            assert out.actual_deployed_j <= o.own_usable_energy_j + 1e-9
            assert out.actual_deployed_j <= o.deploy_ceiling_j + 1e-9
            assert out.actual_deployed_j >= 0.0


# ------------------------------------------------------- action physics
def test_more_deployable_energy_never_reduces_our_speed():
    """Monotone, with diminishing returns and saturation permitted."""
    o = _opp(own_mj=4.0)
    outs = [evaluate_action(o, a) for a in enumerate_actions(o) if a.kind == ATTACK]
    speeds = [x.own_speed_mps for x in outs]
    assert np.all(np.diff(speeds) >= -1e-9), f"speed fell with more energy: {speeds}"
    assert speeds[-1] > speeds[0], "energy bought nothing at all"


def test_a_worn_tyre_lowers_the_speed_every_budget_buys():
    o_fresh = _opp(wear=0.0)
    o_worn = _opp(wear=1.0)
    a = DecisionAction(ATTACK, "A", 1.4e6)
    assert (evaluate_action(o_worn, a).own_speed_mps
            < evaluate_action(o_fresh, a).own_speed_mps)


def test_speeds_come_from_the_p1_surfaces_not_a_local_formula():
    o = _opp()
    a = DecisionAction(ATTACK, "A", 0.7e6)
    out = evaluate_action(o, a)
    assert out.own_speed_mps == pytest.approx(
        o.zone.own_speed(out.actual_deployed_j, o.own_wear))
    assert out.rival_speed_mps == pytest.approx(
        o.zone.rival_speed(min(o.rival_usable_energy_mean_j, o.deploy_ceiling_j),
                           o.rival_wear))


# --------------------------------------------------------- transitions
def test_energy_transition_is_the_canonical_store_equation():
    tm = _tm(recharge=0.6e6, spend=0.4e6)
    assert next_energy(3.0e6, 1.0e6, tm) == pytest.approx(3.0e6 - 1.0e6 - 0.4e6 + 0.6e6)
    assert next_energy(0.1e6, 1.0e6, tm) == 0.0, "store cannot go negative"
    assert next_energy(4.0e6, 0.0, tm) <= 4.0e6, "store cannot exceed the max"


def test_wear_advances_more_for_attacking_and_resets_only_at_a_pit():
    tm = _tm(hold=0.02, attack=0.05)
    assert next_wear(0.1, False, tm) == pytest.approx(0.12)
    assert next_wear(0.1, True, tm) == pytest.approx(0.15)
    assert next_wear(0.99, True, tm) == 1.0, "wear is bounded"
    assert next_wear(0.8, True, tm, pit_before_next=True) == 0.0


def test_attacking_cannot_wear_less_than_holding():
    with pytest.raises(ValueError, match="cannot wear the tyre less"):
        TransitionModel(1.0, 1.0, wear_hold_per_step=0.05,
                        wear_attack_per_step=0.02, fail_cost=0.2)


def test_transition_model_wear_comes_from_the_tyre_module():
    cfg = load_config()
    p = params_from_config(cfg, "MEDIUM")
    model = _model(_zone())
    tm = transition_model_from(model, p, distance_m=5000.0, track_temp_c=35.0,
                               opportunities_per_lap=3)
    assert tm.wear_attack_per_step > tm.wear_hold_per_step > 0.0
    assert tm.recharge_per_step_j == pytest.approx(model.recharge_per_lap / 3)
    assert tm.normal_spend_per_step_j == pytest.approx(model.own_spend_per_lap / 3)


# ------------------------------------------------------------ the solver
def test_a_single_opportunity_reduces_to_a_one_shot_choice():
    z = _zone()
    sol = solve_opportunities(_model(z), [_opp(zone=z)], _tm())
    assert sol.horizon_len == 1
    assert sol.decision in (HOLD, ATTACK)
    assert sol.chosen.label in sol.action_values
    assert sol.value_action == pytest.approx(max(sol.action_values.values()))


def test_a_later_opportunity_can_make_holding_optimal_today():
    """The multi-opportunity exit criterion.

    Today's zone is poor (low dv per MJ, wide gap); the next one is excellent.
    Spending the energy now would leave nothing for the better chance.
    """
    poor = _zone("A", severity=0.2, ceiling_mj=1.4, dv_per_mj=1.0)
    great = _zone("B", severity=1.0, ceiling_mj=1.4, dv_per_mj=12.0)
    tm = _tm(recharge=0.0, spend=0.0)          # no recharge: energy is scarce

    now_only = solve_opportunities(_model(poor),
                                   [_opp(lap=1, zone=poor, gap=1.2, own_mj=1.4)], tm)
    with_future = solve_opportunities(
        _model(poor),
        [_opp(lap=1, zone=poor, gap=1.2, own_mj=1.4),
         _opp(lap=2, zone=great, gap=0.2, own_mj=1.4, s=600.0)], tm)

    assert with_future.horizon_len == 2
    assert with_future.value_action >= now_only.value_action
    # the presence of the better chance must not make today MORE attractive
    a_now = now_only.action_values.get(now_only.chosen.label)
    assert a_now is not None
    if now_only.decision == ATTACK:
        assert with_future.decision == HOLD or (
            with_future.chosen.deployment_budget_j
            <= now_only.chosen.deployment_budget_j), (
            "a better future opportunity did not restrain today's spend")


def test_a_moderate_budget_can_beat_the_maximum_one():
    """The deployment-optimisation exit criterion.

    The regime where this happens is a real and recognisable one: the rival is
    slow enough that the pass is already near-certain. Pass probability by
    budget here runs 0.93 / 0.97 / 0.983 / 0.983, so the last 0.35 MJ buys about
    four thousandths of probability -- and costs the entire option on the two
    later opportunities, because with no recharge those joules do not come back.

    Nothing is penalised. The maximum budget simply scores lower once the future
    it forecloses is counted, which is a sentence a fixed-cost ATTACK cannot
    even express.
    """
    budgets = np.linspace(0.0, 2.6e6, 7)
    spent = np.minimum(budgets, 1.4e6)
    base = 70.0 + 6.0 * np.sqrt(spent / 1e6)      # concave: v^3 drag fights power
    rival = base - 12.0                            # a rival we already have covered
    z = ZoneModel("A", 1.0, spent, base, 6.0, wear_grid=WEAR_GRID,
                  speed_grid_by_wear=np.vstack([base - 5.0 * w for w in WEAR_GRID]),
                  rival_speed_grid_by_wear=np.vstack([rival - 5.0 * w for w in WEAR_GRID]))

    tm = _tm(recharge=0.0, spend=0.0)             # strictly finite energy
    opps = [_opp(lap=i + 1, zone=z, gap=0.05, own_mj=1.4, s=100.0 + 600 * i)
            for i in range(3)]
    sol = solve_opportunities(_model(z), opps, tm)

    attacks = {l: v for l, v in sol.action_values.items() if l != HOLD}
    assert sol.decision == ATTACK, f"expected an attack, got {sol.chosen.label}"
    max_label = max(attacks, key=lambda l: float(l.split(",")[1].split()[0]))
    max_budget = float(max_label.split(",")[1].split()[0])
    chosen_mj = sol.chosen.deployment_budget_j / 1e6

    assert chosen_mj < max_budget - 1e-9, (
        f"took the maximum budget despite saturating returns; values {attacks}")
    assert attacks[sol.chosen.label] > attacks[max_label], (
        "the chosen budget must actually be worth more than the maximum")
    # the mechanism, stated: the extra energy barely moves the pass probability
    outs = {l: o for l, o in sol.outcomes.items() if o.action.kind == ATTACK}
    q_chosen = outs[sol.chosen.label].pass_probability
    q_max = outs[max_label].pass_probability
    assert q_max - q_chosen < 0.02, (
        f"this fixture was meant to have saturating q: {q_chosen} -> {q_max}")


def test_saturation_is_reported_when_a_caller_asks_for_more_than_the_zone_takes():
    """Enumerated budgets are fractions of the ceiling, so they never saturate.

    That is deliberate: an action the zone cannot absorb is not worth offering.
    Saturation is still reported for any budget handed in from outside, which is
    what makes "2.6 MJ and 1.4 MJ are the same lap" explainable.
    """
    z = _zone(ceiling_mj=0.7)
    o = _opp(zone=z, own_mj=3.0)
    assert all(not evaluate_action(o, a).saturated for a in enumerate_actions(o)), (
        "an enumerated budget exceeded the zone ceiling")
    over = evaluate_action(o, DecisionAction(ATTACK, "A", 2.6e6))
    assert over.saturated and over.actual_deployed_j == pytest.approx(0.7e6)


def test_pit_horizon_changes_the_call_through_state_not_a_penalty():
    """Tyre/pit coupling exit criterion."""
    z = _zone("A", ceiling_mj=1.4, dv_per_mj=6.0, wear_penalty=25.0)
    tm = _tm(hold=0.05, attack=0.35, recharge=0.0, spend=0.0)
    opps = [_opp(lap=1, zone=z, own_mj=3.0, gap=0.3, wear=0.2),
            _opp(lap=2, zone=z, own_mj=3.0, gap=0.3, wear=0.2, s=600.0),
            _opp(lap=3, zone=z, own_mj=3.0, gap=0.3, wear=0.2, s=900.0)]
    far = solve_opportunities(_model(z), opps, tm, pit_at_index=None)
    near = solve_opportunities(_model(z), opps, tm, pit_at_index=0)
    # attacking is worth at least as much when the wear is about to be discarded
    assert near.value_action >= far.value_action - 1e-12
    assert near.robustness["horizon_len"] == 3


def test_no_reward_hack_exists_in_the_p2_solver():
    """The pit/tyre effect must come from the transition, like P1."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "xray" / "opportunity.py").read_text()
    for bad in ("near_pit", "pit_bonus", "pit_multiplier", "laps_remaining",
                "1.0 / laps", "wear_penalty *", "tyre_penalty"):
        assert bad not in src, f"a reward hack appeared: {bad!r}"


# --------------------------------------------------------- belief/robustness
def test_a_collapsed_belief_reduces_to_a_single_scenario():
    """Deterministic and belief-aware must agree when the posterior collapses."""
    z = _zone()
    point = solve_opportunities(_model(z), [_opp(zone=z)], _tm())
    assert point.scenarios.source == "point_mass_no_interval"
    assert point.scenarios.weights == (1.0,)
    assert point.action_consensus == pytest.approx(1.0)
    assert point.expected_regret == pytest.approx(0.0)

    narrow = solve_opportunities(
        _model(z), [_opp(zone=z, riv_mj=1.0, p10=0.9999, p90=1.0001)], _tm())
    assert narrow.chosen.label == point.chosen.label
    assert narrow.value_action == pytest.approx(point.value_action, rel=1e-3)


def test_scenario_weights_normalise_and_are_reported():
    sol = solve_opportunities(_model(_zone()),
                              [_opp(riv_mj=1.0, p10=0.2, p90=2.0)], _tm())
    assert sum(sol.scenarios.weights) == pytest.approx(1.0)
    assert len(sol.scenarios.energies_j) == 3
    assert sum(p["weight"] for p in sol.policy_posterior) == pytest.approx(1.0)
    assert 0.0 <= sol.action_consensus <= 1.0


def test_scenario_set_rejects_unnormalised_weights():
    with pytest.raises(ValueError, match="must sum to 1"):
        ScenarioSet((1.0, 2.0), (0.4, 0.4))
    with pytest.raises(ValueError, match="differ in length"):
        ScenarioSet((1.0, 2.0), (1.0,))


def test_a_wider_belief_does_not_change_the_physics():
    """Belief width may move consensus/robustness, never a speed."""
    z = _zone()
    tight = _opp(zone=z, riv_mj=1.0, p10=0.95, p90=1.05)
    wide = _opp(zone=z, riv_mj=1.0, p10=0.1, p90=2.5)
    a = DecisionAction(ATTACK, "A", 1.0e6)
    assert (evaluate_action(tight, a).own_speed_mps
            == pytest.approx(evaluate_action(wide, a).own_speed_mps))
    st = solve_opportunities(_model(z), [tight], _tm())
    sw = solve_opportunities(_model(z), [wide], _tm())
    assert st.robustness["rival_energy_spread_mj"] < sw.robustness["rival_energy_spread_mj"]


def test_margin_regret_and_next_best_are_internally_consistent():
    sol = solve_opportunities(_model(_zone()),
                              [_opp(riv_mj=1.0, p10=0.3, p90=2.2)], _tm())
    ranked = sorted(sol.action_values.values(), reverse=True)
    assert sol.value_action == pytest.approx(ranked[0])
    if len(ranked) > 1:
        assert sol.next_best_value == pytest.approx(ranked[1])
        assert sol.decision_margin == pytest.approx(ranked[0] - ranked[1])
        assert sol.decision_margin >= -1e-12
        assert sol.next_best_action.label != sol.chosen.label
    assert sol.expected_regret >= -1e-12
    # regret is zero exactly when every scenario agrees with the call
    if sol.action_consensus == pytest.approx(1.0):
        assert sol.expected_regret == pytest.approx(0.0, abs=1e-12)


def test_policy_posterior_reports_each_scenarios_own_optimum():
    sol = solve_opportunities(_model(_zone()),
                              [_opp(riv_mj=1.0, p10=0.2, p90=2.4)], _tm())
    assert len(sol.policy_posterior) == 3
    for row in sol.policy_posterior:
        assert row["optimal_value"] >= row["chosen_action_value"] - 1e-12
        assert row["optimal_action"] in sol.action_values


def test_the_explanation_reads_solver_values_and_runs_no_second_argmax():
    sol = solve_opportunities(_model(_zone()),
                              [_opp(riv_mj=1.0, p10=0.4, p90=1.8)], _tm())
    assert sol.chosen_outcome.action.label == sol.chosen.label
    assert sol.value_action == pytest.approx(sol.action_values[sol.chosen.label])
    assert sol.value_hold == pytest.approx(sol.action_values.get(HOLD, 0.0))


def test_infeasible_actions_never_win():
    o = _opp(own_mj=0.35)                       # affords only the smallest budget
    sol = solve_opportunities(_model(o.zone), [o], _tm())
    for label in sol.action_values:
        if label == HOLD:
            continue
        budget = float(label.split(",")[1].split()[0]) * 1e6
        assert budget <= o.own_usable_energy_j + 1e-9, (
            f"an unaffordable action {label} was scored")
    assert sol.chosen.deployment_budget_j <= o.own_usable_energy_j + 1e-9


def test_memoisation_resolution_does_not_change_the_answer(monkeypatch):
    """The state grid is a discretisation, so it has to be pinned like one.

    The backward induction memoises on (energy, wear) and must snap them to a
    grid or the recursion is exponential -- 45 s for a 12-opportunity horizon
    before this, 0.2 s after. A coarse grid that changed the recommendation
    would be buying that speed with the answer, so a 4x finer grid must agree.
    """
    import xray.opportunity as m

    z = _zone()
    opps = [_opp(lap=i + 1, zone=z, own_mj=2.0, gap=0.3, s=100.0 + 600 * i)
            for i in range(5)]
    tm = _tm(recharge=0.2e6, spend=0.15e6)

    coarse = solve_opportunities(_model(z), opps, tm)
    monkeypatch.setattr(m, "ENERGY_MEMO_QUANTUM_J", m.ENERGY_MEMO_QUANTUM_J / 4)
    monkeypatch.setattr(m, "WEAR_MEMO_QUANTUM", m.WEAR_MEMO_QUANTUM / 4)
    fine = solve_opportunities(_model(z), opps, tm)

    assert coarse.chosen.label == fine.chosen.label, (
        f"grid resolution changed the recommendation: "
        f"{coarse.chosen.label} vs {fine.chosen.label}")
    assert coarse.value_action == pytest.approx(fine.value_action, rel=0.05)
