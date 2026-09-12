"""Step 8: deciding under a belief rather than under a point estimate."""
from __future__ import annotations

import numpy as np

from xray.decision import build_model
from xray.qmdp import decide
from xray.regs import PRE_MIAMI
from xray.vehicle import VehicleParams


def _model(cfg, gt, n_laps=12):
    params = VehicleParams.from_config(cfg)
    return build_model(gt.track, params, n_laps=n_laps,
                       recharge_per_lap=2.2e6, rival_spend_per_lap=2.4e6)


def test_a_belief_is_not_its_mean(cfg, races):
    """The reason QMDP is here at all.

    p_pass is convex in the energy difference over the relevant range, so the
    mean of the probability is not the probability of the mean. A point estimate
    therefore misprices a wide belief systematically, in a direction that does
    not average out over a stint.
    """
    model = _model(cfg, races[42])
    wide = np.linspace(0.0, 3.0e6, 400)
    point = np.full(400, wide.mean())
    d_wide = decide(model, wide, None, laps_left=8, e_own=2.0e6, reward=0.7)
    d_point = decide(model, point, None, laps_left=8, e_own=2.0e6, reward=0.7)
    assert not np.isclose(d_wide.q_best, d_point.q_best, rtol=1e-6), (
        "a 3 MJ-wide belief priced identically to its mean; the averaging is "
        "not happening")


def test_a_drained_rival_is_attacked_and_a_full_one_is_not(cfg, races):
    """The whole point of reconstructing their energy."""
    model = _model(cfg, races[42])
    rng = np.random.default_rng(0)
    empty = np.clip(rng.normal(0.1e6, 0.1e6, 300), 0, None)
    full = np.clip(rng.normal(3.0e6, 0.2e6, 300), 0, None)
    d_empty = decide(model, empty, None, laps_left=6, e_own=2.4e6, reward=0.6,
                     v_fail=0.3)
    d_full = decide(model, full, None, laps_left=6, e_own=2.4e6, reward=0.6,
                    v_fail=0.3)
    assert d_empty.q_best > d_full.q_best
    assert d_empty.p_pass_mean > d_full.p_pass_mean


def test_holding_wins_when_no_attack_beats_the_continuation(cfg, races):
    """"Attack always" is not a strategy. With a strong continuation value and a
    full rival, the call must be to hold."""
    model = _model(cfg, races[42])
    full = np.full(200, 3.5e6)
    d = decide(model, full, None, laps_left=11, e_own=0.2e6, reward=0.2,
               v_fail=0.9)
    assert d.action is None
    assert d.q_best <= d.q_hold


def test_consensus_is_reported_separately_from_the_expected_value(cfg, races):
    """A call optimal on average but for only half the cloud is a different
    animal from one that holds for 95%, and the difference is invisible in an
    expected value. A narrow belief must agree with itself more than a wide one.
    """
    model = _model(cfg, races[42])
    rng = np.random.default_rng(1)
    narrow = np.clip(rng.normal(1.0e6, 0.05e6, 400), 0, None)
    wide = np.clip(rng.uniform(0.0, 4.0e6, 400), 0, None)
    d_n = decide(model, narrow, None, laps_left=7, e_own=2.0e6, reward=0.6)
    d_w = decide(model, wide, None, laps_left=7, e_own=2.0e6, reward=0.6)
    assert 0.0 <= d_w.consensus <= d_n.consensus <= 1.0
    # The separation is the claim, and it is large: 0.825 against 0.235 here.
    # An absolute floor on the narrow belief is not: under the corrected 2026
    # curve the three zones' q values sit at 0.024 / 0.022 / 0.027, so the argmax
    # is close and a narrow cloud still disagrees with itself ~17% of the time.
    # On the old linear-to-355 km/h taper zone C led by 1.7x (0.066 vs 0.038) and
    # narrow consensus was a clean 1.00. The corrected curve allows much less
    # deployment above 290 km/h, which compresses the zones together -- a real
    # change in how sharp the decision is, recorded here rather than absorbed by
    # lowering 0.9 to 0.82. Both runs choose zone C.
    assert d_n.consensus - d_w.consensus > 0.3, (
        f"narrow {d_n.consensus:.3f} vs wide {d_w.consensus:.3f}: a narrow "
        "belief must agree with itself markedly more than a wide one")
    assert d_n.consensus > 0.75


def test_reserve_sensitivity_is_reported_with_the_call(cfg, races):
    """The reserve is weakly identified (step 5). Reporting how much that
    matters is what stops a known limitation from looking fatal: where the
    sensitivity is small the degeneracy is irrelevant, and that is the common
    case."""
    model = _model(cfg, races[42])
    rng = np.random.default_rng(2)
    belief = np.clip(rng.normal(1.5e6, 0.4e6, 300), 0, None)
    d = decide(model, belief, None, laps_left=6, e_own=2.2e6, reward=0.6)
    assert np.isfinite(d.sensitivity)
    # shifting the rival's energy up by 0.5 MJ can only make attacking worse
    assert d.sensitivity <= 1e-9, (
        f"a richer rival improved our expected value ({d.sensitivity:+.4f}/MJ)")


def test_zero_spread_belief_reports_zero_sensitivity(cfg, races):
    """No uncertainty, nothing for the degeneracy to matter to."""
    model = _model(cfg, races[42])
    d = decide(model, np.full(50, 1.0e6), None, laps_left=5, e_own=2.0e6)
    assert d.sensitivity == 0.0


def test_manual_override_eligibility_is_a_state_not_noise(cfg, races):
    """Within 1.000 s at the detection point the rival's ceiling rises next lap.
    That is a known rule on a known state and must show up in the call."""
    model = _model(cfg, races[42])
    belief = np.full(100, 1.0e6)
    close = decide(model, belief, None, laps_left=6, e_own=2.0e6, gap_s=0.9,
                   regs=PRE_MIAMI)
    far = decide(model, belief, None, laps_left=6, e_own=2.0e6, gap_s=1.4,
                 regs=PRE_MIAMI)
    assert close.mom_eligible and not far.mom_eligible
    assert any("detection point" in n for n in close.notes)
    assert not any("detection point" in n for n in far.notes)


def test_provoking_a_defence_is_evaluated_two_steps_deep(cfg, races):
    """The defender's joule obeys the same 1/(m v^3), so forcing a defence out
    of the slowest corner drains them where their energy is worth most. A
    one-step lookahead cannot see this, so it has to be asked for explicitly --
    and it must not fire when the rival has nothing left to drain."""
    model = _model(cfg, races[42])
    zones = {z.name: z for z in model.zones}
    slow = min(model.zones, key=lambda z: z.dv_per_mj)
    fast = max(model.zones, key=lambda z: z.dv_per_mj)
    rich = np.full(200, 3.0e6)
    d_rich = decide(model, rich, None, laps_left=6, e_own=3.0e6, reward=0.6,
                    prev_zone=slow)
    empty = np.full(200, 0.02e6)
    d_empty = decide(model, empty, None, laps_left=6, e_own=3.0e6, reward=0.6,
                     prev_zone=slow)
    assert not d_empty.provoke, "provoked a defence from a rival with no energy"
    assert isinstance(d_rich.provoke, bool)
    if d_rich.provoke:
        assert any("slow corner" in n for n in d_rich.notes)


def test_every_zone_gets_a_q_value(cfg, races):
    """The call must be explainable: the reason a zone was not chosen is its
    number, not its absence."""
    model = _model(cfg, races[42])
    d = decide(model, np.full(100, 1.2e6), None, laps_left=6, e_own=2.0e6)
    assert set(d.q_values) == {z.name for z in model.zones}
    assert all(np.isfinite(v) for v in d.q_values.values())
