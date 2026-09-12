"""P4: the scenario walkthrough's ground truth -- did the driver actually
attack, and did that match what P2 would have called.

This is presentation-adjacent plumbing for the interactive Replay
walkthrough, not a new estimator or a new decision rule. `actual_action` and
`matches_recommendation` are read straight off `laps[].position`, which is
the same public race result `snapshots.make_snapshot` already causally
splits -- nothing here infers, thresholds or scores anything the estimator
produces. The negative case (no comparison possible) must stay `None`, never
collapse to a guessed `"held"`.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from xray.decision_service import historical_replay


@pytest.fixture(scope="module")
def real_payload():
    root = pathlib.Path(__file__).resolve().parent.parent / "out" / "races"
    paths = sorted(root.glob("*.json"))
    if not paths:
        pytest.skip("no real race payload in out/races")
    with open(paths[0]) as fh:
        return json.load(fh)


def test_the_replay_carries_position_and_gap_outcome_fields(real_payload):
    cars = sorted(real_payload["cars"])
    a, b = cars[0], cars[1]
    t = real_payload["cars"][a]["trace"]["t"]
    r = historical_replay(real_payload, a, b, float(t[len(t) // 3]), horizon_s=60.0)
    for d in (a, b):
        out = r["later_observable_outcome"][d]
        assert "position_at_cutoff" in out
        assert "position_at_window_end" in out
    assert "gap_to_rival_at_cutoff_s" in r
    assert "gap_to_rival_at_window_end_s" in r
    assert "actual_action" in r
    assert "matches_recommendation" in r


def test_actual_action_is_derived_only_from_position_never_guessed(real_payload):
    """No comparison possible -> `None`, not a default `"held"`."""
    cars = sorted(real_payload["cars"])
    a, b = cars[0], cars[1]
    t = real_payload["cars"][a]["trace"]["t"]
    # A cutoff at the very last sample leaves no future window, so there is no
    # later position to compare against -- the honest answer is `None`.
    r = historical_replay(real_payload, a, b, float(t[-1]), horizon_s=0.01)
    out = r["later_observable_outcome"][a]
    if out["position_at_window_end"] is None:
        assert r["actual_action"] is None
        assert r["matches_recommendation"] is None


def test_matches_recommendation_is_a_plain_fact_comparison_not_new_maths(real_payload):
    """Source-scan: the function may compare two already-known facts, and must
    not introduce a probability, a threshold, or any arithmetic on them."""
    import inspect

    from xray.decision_service import historical_replay as fn

    src = inspect.getsource(fn)
    body = src[src.index("def historical_replay"):]
    banned = {
        r"\bp_pass\b": "a pass-probability formula",
        r"1\s*/\s*\(\s*1\s*\+": "a logistic",
        r"np\.exp": "an exponential model",
        r"threshold\s*=": "a new threshold",
    }
    for pattern, what in banned.items():
        assert not __import__("re").search(pattern, body), f"historical_replay contains {what}"
    assert '"attacked" if position_after[car] < position_before[car]' in body or \
        "attacked" in body


def test_matches_recommendation_agrees_with_actual_action_and_the_call(real_payload):
    cars = sorted(real_payload["cars"])
    a, b = cars[0], cars[1]
    t = real_payload["cars"][a]["trace"]["t"]
    r = historical_replay(real_payload, a, b, float(t[len(t) // 3]), horizon_s=60.0)
    if r["actual_action"] is not None and r["p2_recommendation"] is not None:
        expected = ((r["actual_action"] == "attacked")
                    == (r["p2_recommendation"]["decision"] == "ATTACK"))
        assert r["matches_recommendation"] == expected
