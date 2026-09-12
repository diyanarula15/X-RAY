"""P1.6: pit proximity, and keeping the answer out of the question.

The failure this guards against is not subtle but it is very easy: replaying a
historical race, the lap the car actually pitted is a column in the Laps table,
sitting next to the columns you legitimately need. Reading it makes every
strategic call look brilliant.
"""
from __future__ import annotations

import pytest

from xray.stint import (CAUSAL_SOURCES, ORACLE_SOURCE, PitContext, StintState,
                        UNKNOWN_PIT, actual_pit_lap_from_rows,
                        infer_pit_context, oracle_pit_context,
                        pit_context_from_plan, pit_context_from_window,
                        require_causal, stint_from_lap_row)


# --------------------------------------------------------------- known plans
def test_a_team_plan_is_causal_and_gives_a_horizon_not_a_boolean():
    ctx = pit_context_from_plan(current_lap=10, planned_pit_lap=12)
    assert ctx.source == "team_plan" and ctx.is_causal
    assert ctx.laps_to_pit_mean == pytest.approx(2.0)
    assert ctx.pit_within_2_laps_prob == 1.0
    assert ctx.pit_within_1_lap_prob == 0.0
    assert ctx.confidence > 0.5


def test_the_horizon_distinguishes_half_a_lap_from_eight():
    """A boolean would make these the same situation. They are not."""
    near = pit_context_from_plan(10, 11)
    mid = pit_context_from_plan(10, 12)
    far = pit_context_from_plan(10, 18)
    assert near.laps_to_pit_mean < mid.laps_to_pit_mean < far.laps_to_pit_mean
    assert near.pit_within_1_lap_prob == 1.0
    assert mid.pit_within_1_lap_prob == 0.0
    assert far.pit_within_2_laps_prob == 0.0


def test_a_window_is_uniform_over_its_laps():
    ctx = pit_context_from_window(current_lap=10, start_lap=12, end_lap=15)
    assert ctx.planned_pit_lap is None
    assert ctx.pit_window_start_lap == 12 and ctx.pit_window_end_lap == 15
    assert ctx.laps_to_pit_mean == pytest.approx(3.5)   # 2,3,4,5
    assert ctx.pit_within_2_laps_prob == pytest.approx(0.25)
    assert ctx.confidence < 0.95, "a window is weaker evidence than a lap"


def test_a_plan_source_must_actually_be_a_plan():
    with pytest.raises(ValueError, match="not a plan source"):
        pit_context_from_plan(10, 12, source="inferred")
    with pytest.raises(ValueError, match="not a plan source"):
        pit_context_from_plan(10, 12, source=ORACLE_SOURCE)


def test_no_plan_means_no_horizon():
    ctx = pit_context_from_plan(10, None)
    assert ctx.laps_to_pit_mean is None and not ctx.is_known
    assert ctx.confidence == 0.0
    assert ctx.is_causal, "absence of a plan is still a causal statement"


# ------------------------------------------------------- refusing to guess
def test_rival_pit_inference_returns_unknown_because_no_model_is_validated():
    """Tyre age does not tell you when a team will stop."""
    st = StintState(driver="RIV", lap=20, stint=1, compound="MEDIUM",
                    tyre_life=24.0, fresh_tyre=False)
    ctx = infer_pit_context(st, laps_seen=20)
    assert ctx is UNKNOWN_PIT
    assert ctx.source == "unknown" and ctx.confidence == 0.0
    assert ctx.laps_to_pit_mean is None, (
        "an unknown horizon must be None, not a plausible-looking number")
    assert ctx.is_causal, "'we do not know' is a legitimate causal answer"


def test_an_injected_inference_model_must_label_itself():
    st = StintState(driver="RIV", lap=20)
    good = PitContext(planned_pit_lap=25, laps_to_pit_mean=5.0,
                      source="inferred", confidence=0.4)
    assert infer_pit_context(st, 20, model=lambda *_: good) is good

    with pytest.raises(ValueError, match="label its output"):
        infer_pit_context(st, 20, model=lambda *_: PitContext(source="team_plan"))
    with pytest.raises(TypeError):
        infer_pit_context(st, 20, model=lambda *_: 5)


# ------------------------------------------------------- the oracle boundary
def test_oracle_context_is_not_causal_and_is_refused_at_the_gate():
    """The whole point of the module."""
    ctx = oracle_pit_context(current_lap=10, actual_pit_lap=13)
    assert ctx.source == ORACLE_SOURCE
    assert ctx.laps_to_pit_mean == pytest.approx(3.0)
    assert ctx.confidence == 1.0, "the oracle is certain -- that is the problem"
    assert not ctx.is_causal
    with pytest.raises(ValueError, match="never feed a decision"):
        require_causal(ctx)


def test_every_causal_source_passes_the_gate():
    for src in CAUSAL_SOURCES:
        assert require_causal(PitContext(source=src)).source == src
    assert ORACLE_SOURCE not in CAUSAL_SOURCES


def test_future_pit_events_cannot_change_a_causal_context():
    """Mutating the actual stop must not move a plan or an inference."""
    st = StintState(driver="RIV", lap=10, compound="MEDIUM", tyre_life=8.0)
    plan_before = pit_context_from_plan(10, 14)
    infer_before = infer_pit_context(st, 10)

    rows = [{"lap": 10}, {"lap": 13, "pit_in_time_s": 4321.0}]
    assert actual_pit_lap_from_rows(rows) == 13
    rows[1]["pit_in_time_s"] = 9999.0
    rows.append({"lap": 11, "pit_in_time_s": 1.0})
    assert actual_pit_lap_from_rows(rows) == 11, "oracle itself did change"

    # ...and none of that touched the causal contexts
    assert pit_context_from_plan(10, 14) == plan_before
    assert infer_pit_context(st, 10) == infer_before


# ------------------------------------------------------------- payload rows
def test_stint_from_lap_row_keeps_age_as_age_and_tolerates_absence():
    st = stint_from_lap_row("VER", {"lap": 7, "compound": "SOFT",
                                    "tyre_life": 5, "stint": 2,
                                    "fresh_tyre": True})
    assert st.compound == "SOFT" and st.tyre_life == 5.0
    assert st.stint == 2 and st.fresh_tyre is True
    assert not hasattr(st, "wear_fraction"), "wear is modelled elsewhere, not metadata"

    bare = stint_from_lap_row("VER", {"lap": 7, "compound": "SOFT", "tyre_life": 5})
    assert bare.stint is None and bare.fresh_tyre is None, (
        "missing FastF1 fields must stay missing, not become defaults")


def test_no_pit_entry_means_no_oracle_lap():
    assert actual_pit_lap_from_rows([{"lap": 1}, {"lap": 2}]) is None
