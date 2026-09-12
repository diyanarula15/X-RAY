"""P2 closure: the policy actually drives the simulator, and drives it blind.

Two things are being defended here. The first is that the recommendation reaches
the physics at all -- a strategy layer that nobody executes is a spreadsheet.
The second is the blindfold: the simulator knows the rival's store exactly, and
a policy that reads it would look excellent and mean nothing.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from xray.closedloop import (POLICIES, AttackPlan, BudgetedAttack, EpisodeResult,
                             WORLD_VARIATION, blind_rival_belief, evaluate,
                             fixed_cost_plan, hold_plan, oracle_rival_belief,
                             p2_plan, policy_for, run_episode, world)
from xray.config import load_config
from xray.policy import DeploymentPolicy, get_policy
from xray.sim import FOLLOWER, LEADER, Simulator


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c = {**c, "sim": {**c["sim"], "start_gap_s": 0.35, "n_laps": 6}}
    return c


@pytest.fixture(scope="module")
def gt(cfg):
    return Simulator(cfg, seed=42).run()


# ------------------------------------------------------- policy integration
def test_the_plan_reaches_the_simulator(cfg):
    """A recommendation nobody executes is not a decision."""
    base = get_policy(cfg["sim"]["follower_policy"])
    rival = {LEADER: get_policy(cfg["sim"]["leader_policy"])}

    hold = Simulator(cfg, policies={FOLLOWER: base, **rival}, seed=42).run()
    attack_pol = policy_for(base, AttackPlan("t", lap=2, zone="A", budget_j=1.2e6))
    atk = Simulator(cfg, policies={FOLLOWER: attack_pol, **rival}, seed=42).run()

    assert isinstance(attack_pol, BudgetedAttack)
    a = hold.cars[FOLLOWER].deployed_lap[2]
    b = atk.cars[FOLLOWER].deployed_lap[2]
    assert b > a, f"the attack lap deployed no more than holding: {b} vs {a}"


def test_the_deployment_budget_binds(cfg):
    """A budget that does not change the control is not a budget.

    It binds on DEPLOYMENT, not on peak speed, and the distinction is measured
    rather than assumed. Circuit Sigma's zone A absorbs only about 0.45 MJ from
    the store in race conditions -- the MGU-K taper and the zone length, not the
    optimiser, set that -- so every budget at or above it is the same physical
    action. Asserting distinct peak speeds would be asserting something the
    physics does not support; what must hold is that a smaller allocation is
    actually enforced, and that a larger one is never slower.
    """
    from xray.policy import DeploymentPolicy
    base = get_policy(cfg["sim"]["follower_policy"])
    rival = {LEADER: get_policy(cfg["sim"]["leader_policy"])}

    seen = {}
    for budget in (0.15e6, 0.35e6, 1.4e6):
        pol = policy_for(base, AttackPlan("t", lap=2, zone="A", budget_j=budget))
        stats = {"floor": 0, "reverted": 0}
        orig = BudgetedAttack.demand

        def spy(self, track, s, v, E, ga, gb, ll, is_corner=False, lap=-1,
                _s=stats, _o=orig):
            r = _o(self, track, s, v, E, ga, gb, ll, is_corner=is_corner, lap=lap)
            if lap == self.attack_lap and not is_corner:
                z = track.zone_at(s)
                if z is not None and s < z.s_straight_end and z.name == self.attack_zone:
                    start = self._spend.get((lap, self.attack_zone))
                    if start is not None:
                        key = ("reverted" if start - float(E) >= self.attack_budget_j
                               else "floor")
                        _s[key] += 1
            return r

        BudgetedAttack.demand = spy
        try:
            g = Simulator(cfg, policies={FOLLOWER: pol, **rival}, seed=42).run()
        finally:
            BudgetedAttack.demand = orig
        tr = g.cars[FOLLOWER]
        seen[budget] = {"peak": float(tr.v[tr.lap == 2].max()), **stats}

    # the small allocation must actually run out; the large one must not
    assert seen[0.15e6]["reverted"] > 0, "a 0.15 MJ allocation was never exhausted"
    assert seen[1.4e6]["reverted"] == 0, "a 1.4 MJ allocation ran out in one zone"
    assert seen[0.15e6]["reverted"] > seen[0.35e6]["reverted"], (
        "a smaller allocation must be exhausted sooner")
    # and more allocation is never slower
    peaks = [seen[b]["peak"] for b in (0.15e6, 0.35e6, 1.4e6)]
    assert peaks[0] <= peaks[1] + 1e-9 <= peaks[2] + 1e-9, f"non-monotone: {peaks}"


def test_an_attack_never_deploys_less_than_the_base_policy_would(cfg):
    """The allocation is a floor, not a cap.

    An earlier version cut deployment to zero once the allocation was spent,
    which turned a small "attack" into a lift: a 0.400 MJ attack capped the car
    below its own normal deployment and dropped zone peak speed from 365.3 to
    364.9 km/h. Attacking must never be slower than not attacking.
    """
    base = get_policy(cfg["sim"]["follower_policy"])
    rival = {LEADER: get_policy(cfg["sim"]["leader_policy"])}
    hold = Simulator(cfg, policies={FOLLOWER: base, **rival}, seed=42).run()
    v_hold = float(hold.cars[FOLLOWER].v[hold.cars[FOLLOWER].lap == 2].max())
    for budget in (0.1e6, 0.4e6, 1.6e6):
        pol = policy_for(base, AttackPlan("t", lap=2, zone="A", budget_j=budget))
        g = Simulator(cfg, policies={FOLLOWER: pol, **rival}, seed=42).run()
        tr = g.cars[FOLLOWER]
        v = float(tr.v[tr.lap == 2].max())
        assert v >= v_hold - 1e-9, (
            f"a {budget/1e6:.1f} MJ attack was slower than holding: {v} < {v_hold}")


def test_a_plan_with_no_attack_leaves_the_base_policy_untouched(cfg):
    base = get_policy(cfg["sim"]["follower_policy"])
    assert policy_for(base, AttackPlan("hold")) is base


def test_energy_cannot_exceed_what_is_physically_available(cfg):
    """The budget is a request; the store and the regulation are the limit."""
    base = get_policy(cfg["sim"]["follower_policy"])
    pol = policy_for(base, AttackPlan("t", lap=2, zone="A", budget_j=99.0e6))
    g = Simulator(cfg, policies={FOLLOWER: pol,
                                 LEADER: get_policy(cfg["sim"]["leader_policy"])},
                  seed=42).run()
    tr = g.cars[FOLLOWER]
    from xray.constants import E_STORE_MAX
    assert np.all(tr.E >= -1e-6) and np.all(tr.E <= E_STORE_MAX + 1e-6)
    # a whole lap cannot deploy more than the store plus what it harvested
    for lap in range(g.n_laps):
        assert tr.deployed_lap[lap] <= E_STORE_MAX + tr.harvested_lap[lap] + 1.0


def test_tyre_and_pit_machinery_still_runs_under_a_p2_policy(cfg):
    """Closed loop must not bypass the P1 transitions."""
    base = get_policy(cfg["sim"]["follower_policy"])
    pol = policy_for(base, AttackPlan("t", lap=2, zone="A", budget_j=0.8e6))
    g = Simulator(cfg, policies={FOLLOWER: pol,
                                 LEADER: get_policy(cfg["sim"]["leader_policy"])},
                  seed=42).run()
    tr = g.cars[FOLLOWER]
    k = g.n_laps - 1
    delta = np.diff(tr.e_lap_open)[:k]
    flows = (tr.harvested_lap - tr.deployed_lap)[:k]
    assert np.abs(delta - flows).max() < 1000.0, "the store stopped closing"
    assert np.all(np.isfinite(tr.v)) and tr.v.max() > 50.0


# ------------------------------------------------------- no truth leakage
def test_the_belief_is_built_from_public_data_only():
    """Structural: blind_rival_belief may not read a store."""
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "xray" / "closedloop.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "blind_rival_belief")
    body = ast.dump(fn)
    for banned in ("e_lap_open", "'E'", "cars"):
        assert banned not in body, (
            f"blind_rival_belief touches simulator state: {banned}")
    assert "observe" in body and "estimate" in body


def test_hidden_truth_cannot_reach_the_p2_policy(cfg, gt):
    """Mutate the rival's TRUE store; the belief and the plan must not move.

    The observation is taken first and held fixed, so this isolates the illegal
    path -- a policy reading `gt.cars[LEADER].E` instead of the estimate.
    """
    import copy
    belief = blind_rival_belief(gt, LEADER, cfg, seed=42, n_particles=100)
    before = p2_plan(belief, gt, cfg, FOLLOWER)

    tampered = copy.deepcopy(gt)
    tampered.cars[LEADER].E[:] = 0.0
    tampered.cars[LEADER].e_lap_open[:] = 0.0
    after = p2_plan(belief, tampered, cfg, FOLLOWER)

    assert after.lap == before.lap and after.zone == before.zone
    assert after.budget_j == before.budget_j


def test_the_oracle_reads_truth_and_is_labelled_as_such(cfg, gt):
    """The one sanctioned exception, and it must announce itself."""
    truth = oracle_rival_belief(gt, LEADER)
    assert "ORACLE" in truth.source
    blind = blind_rival_belief(gt, LEADER, cfg, seed=42, n_particles=100)
    assert "blind" in blind.source
    # and they must actually differ, or the oracle proves nothing
    assert not np.allclose(truth.usable_by_lap_j, blind.usable_by_lap_j)


def test_oracle_policies_are_kept_out_of_the_default_set():
    from xray.closedloop import ORACLE_POLICIES
    assert "oracle_p2" not in POLICIES
    assert "oracle_p2" in ORACLE_POLICIES


# --------------------------------------------------------- reproducibility
def test_the_same_seed_and_policy_reproduce_exactly(cfg):
    plan = AttackPlan("t", lap=2, zone="A", budget_j=0.9e6)
    a = run_episode(cfg, 42, plan)
    b = run_episode(cfg, 42, plan)
    assert (a.attempts, a.successes, a.ahead_fraction, a.energy_deployed_j) == \
           (b.attempts, b.successes, b.ahead_fraction, b.energy_deployed_j)


def test_different_policies_can_produce_different_control(cfg):
    hold = run_episode(cfg, 42, AttackPlan("hold"))
    atk = run_episode(cfg, 42, AttackPlan("t", lap=2, zone="A", budget_j=1.4e6))
    assert hold.plan.attacks is False and atk.plan.attacks is True
    # the control differs even where the outcome does not
    assert hold.energy_deployed_j != atk.energy_deployed_j or \
        hold.attempts != atk.attempts or hold.ahead_fraction != atk.ahead_fraction


def test_race_order_uses_distance_not_the_order_index(cfg):
    """The metric bug this test exists because of.

    `gt.order` is a uint8 index; comparing it to the car-id string is silently
    always False, so every policy scored ahead_fraction = 0.0000 and it looked
    like a physics result rather than a broken comparison.
    """
    r = run_episode(cfg, 42, AttackPlan("hold"))
    g = Simulator(cfg, seed=42).run()
    lead = np.asarray(g.cars[FOLLOWER].s_total) - np.asarray(g.cars[LEADER].s_total)
    assert r.ahead_fraction == pytest.approx(float(np.mean(lead > 0.0)))
    assert np.asarray(g.order).dtype != object, (
        "gt.order is not a car id; do not compare it to one")


# --------------------------------------------------------------- baselines
def test_every_baseline_produces_a_plan(cfg, gt):
    belief = blind_rival_belief(gt, LEADER, cfg, seed=42, n_particles=100)
    for name, fn in POLICIES.items():
        plan = fn(belief, gt, cfg, FOLLOWER)
        assert plan.source == name
        if plan.attacks:
            assert plan.zone in {z.name for z in gt.track.zones}
            assert 0 <= plan.lap < gt.n_laps


def test_hold_never_attacks(cfg, gt):
    belief = blind_rival_belief(gt, LEADER, cfg, seed=42, n_particles=100)
    assert not hold_plan(belief, gt, cfg, FOLLOWER).attacks


def test_the_fixed_cost_baseline_uses_the_canonical_p0_solver():
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "xray" / "closedloop.py").read_text()
    i = src.index("def fixed_cost_plan")
    body = src[i:i + 1400]
    assert "solve_exogenous" in body, (
        "baseline B must be the real pre-P2 solver, not a re-implementation")


def test_worlds_are_deterministic_and_varied(cfg):
    a, b = world(cfg, 42), world(cfg, 42)
    assert a["sim"]["start_gap_s"] == b["sim"]["start_gap_s"]
    c = world(cfg, 7)
    assert c["sim"]["start_gap_s"] != a["sim"]["start_gap_s"]
    for key in WORLD_VARIATION:
        section, field = key.split(".")
        assert field in a[section], f"{key} is not an existing config key"


def test_legacy_simulation_is_unchanged(cfg):
    """A run with no policies argument must behave exactly as before."""
    a = Simulator(cfg, seed=42).run()
    b = Simulator(cfg, seed=42).run()
    assert np.array_equal(a.cars[FOLLOWER].v, b.cars[FOLLOWER].v)
    assert type(a.cars[FOLLOWER].v) is np.ndarray
    # the default follower policy is the plain one, not a BudgetedAttack
    assert not isinstance(get_policy(cfg["sim"]["follower_policy"]), BudgetedAttack)
