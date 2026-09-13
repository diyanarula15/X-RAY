"""Situations-tab semantics: what is observed, what is recommended, what is not known.

Two repairs are pinned here.

1. A zone whose straight starts AT the start/finish line (`s_straight_start ==
   0.0`, Zandvoort zone B, the only one on disk) had no same-lap sample at or
   before its decision point, because `ingest.grid_lap` places the first cell
   centre at s = 10 m. The refusal propagated out of the whole decision trace:
   8 of 8 Zandvoort battles returned "no causal trace samples for lap 1 at
   s <= 0.0 m" and the tab was blank for the entire race. The boundary now reads
   the last sample of the PREVIOUS lap, and an opportunity that still has no
   causal sample (lap 1, which has no previous lap) is skipped, not fatal.

2. The tab printed MATCHED / DIVERGED per row from a "driver action" inferred
   from the position delta over the evaluation window. A driver who attacked and
   failed was recorded as having held, and a car promoted by the pit stop of the
   car ahead was recorded as having attacked. The delta survives as an OBSERVED
   OUTCOME; the driver's action is reported as unobserved and the counterfactual
   as unresolved.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from xray import decision_service as ds

REPO = Path(__file__).resolve().parents[1]
RACES = REPO / "out" / "races"
ZANDVOORT = RACES / "2026_r12_R.json"


# ---------------------------------------------------------------------------
# belief_at_position: the lap boundary, and the absence of future leakage
# ---------------------------------------------------------------------------

def _trace(samples):
    """`(lap, s, t, usable)` tuples -> the trace dict `belief_at_position` reads."""
    return {
        "lap": [x[0] for x in samples],
        "s": [x[1] for x in samples],
        "t": [x[2] for x in samples],
        "usable_mean": [x[3] for x in samples],
        "usable_p10": [x[3] for x in samples],
        "usable_p90": [x[3] for x in samples],
    }


# Two laps on the same 10 m-offset grid the real ingest produces. The lap-2
# values are deliberately far from the lap-1 values so a leak is visible in the
# returned number, not just in the index.
TWO_LAPS = _trace([(1, 10.0, 1.0, 1.0), (1, 90.0, 2.0, 1.1), (1, 170.0, 3.0, 1.2),
                   (2, 10.0, 4.0, 9.0), (2, 90.0, 5.0, 9.1), (2, 170.0, 6.0, 9.2)])


def test_s_zero_on_a_later_lap_reads_the_last_sample_of_the_previous_lap():
    b = ds.belief_at_position(TWO_LAPS, 2, 0.0)
    assert b.source == "belief.usable_mean_at_last_sample_of_previous_lap"
    assert TWO_LAPS["lap"][b.sample_index] == 1
    assert TWO_LAPS["s"][b.sample_index] == 170.0      # the lap-boundary crossing
    assert b.sample_time_s == 3.0


def test_s_zero_never_reads_a_future_same_lap_sample():
    """The leak this forbids: resolving s = 0 on lap 2 from lap 2's own samples.

    Every lap-2 sample sits at s > 0, i.e. after the decision point, and carries
    usable 9.0-9.2 MJ against lap 1's 1.0-1.2. A same-lap read would return
    9.0e6 J and an index inside lap 2.
    """
    b = ds.belief_at_position(TWO_LAPS, 2, 0.0)
    assert b.sample_index < TWO_LAPS["lap"].index(2)
    assert TWO_LAPS["lap"][b.sample_index] < 2
    assert b.usable_energy_j == pytest.approx(1.2e6)
    assert b.sample_time_s < min(t for lap, _s, t, _u in
                                 [(1, 10.0, 1.0, 1.0), (1, 90.0, 2.0, 1.1),
                                  (1, 170.0, 3.0, 1.2), (2, 10.0, 4.0, 9.0),
                                  (2, 90.0, 5.0, 9.1), (2, 170.0, 6.0, 9.2)]
                                 if lap == 2)


def test_s_zero_on_lap_one_still_refuses_because_there_is_no_previous_lap():
    """Nothing is invented for the first crossing. Refusal is the right output."""
    with pytest.raises(ValueError, match="no causal trace samples for lap 1"):
        ds.belief_at_position(TWO_LAPS, 1, 0.0)


def test_a_mid_lap_decision_point_is_unchanged_and_never_reaches_back_a_lap():
    """Non-boundary behaviour is byte-identical: same-lap, at-or-before, or refuse.

    The previous-lap branch is gated on `s_m <= 0.0` precisely so a mid-lap
    telemetry hole cannot silently resolve a belief from a whole lap earlier.
    """
    b = ds.belief_at_position(TWO_LAPS, 2, 100.0)
    assert b.source == "belief.usable_mean_at_or_before_zone"
    assert TWO_LAPS["lap"][b.sample_index] == 2 and TWO_LAPS["s"][b.sample_index] == 90.0

    hole = _trace([(1, 10.0, 1.0, 1.0), (1, 90.0, 2.0, 1.1), (2, 900.0, 3.0, 9.0)])
    with pytest.raises(ValueError, match="no causal trace samples for lap 2"):
        ds.belief_at_position(hole, 2, 100.0)


# ---------------------------------------------------------------------------
# an unresolvable opportunity is skipped, not fatal
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def zandvoort():
    if not ZANDVOORT.exists():
        pytest.skip("Zandvoort artefact not analysed")
    return json.loads(ZANDVOORT.read_text(encoding="utf-8"))


def test_zandvoort_is_the_s_zero_circuit_and_no_other_artefact_is(zandvoort):
    assert any(z["s_straight_start"] == 0.0
               for z in zandvoort["circuit_geometry"]["zones"])
    for p in sorted(RACES.glob("*.json")):
        if p == ZANDVOORT:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        starts = [z["s_straight_start"] for z in d["circuit_geometry"]["zones"]]
        # If this ever fails it is not a test to relax: another circuit gained a
        # boundary zone and its numbers are expected to move.
        assert min(starts) > 0.0, f"{p.name} now has an s = 0 zone"


def test_one_unresolvable_opportunity_is_skipped_not_fatal(zandvoort, monkeypatch):
    """The Zandvoort failure mode, reproduced at the opportunity level.

    `_evaluate_lap` must drop a zone whose belief refuses and keep the others.
    The real lap 1 at Zandvoort does exactly this: zone B (s = 0.0) has no
    previous lap, zones A and C resolve, and the lap is still evaluated.
    """
    real = ds.belief_at_position
    calls = {"refused": 0}

    def spy(trace, lap, s_m):
        try:
            return real(trace, lap, s_m)
        except ValueError:
            calls["refused"] += 1
            raise

    monkeypatch.setattr(ds, "belief_at_position", spy)
    track = ds.track_from_payload(zandvoort)
    params = ds.params_from_payload(zandvoort)
    car, rival = zandvoort["battles"][0]["car"], zandvoort["battles"][0]["ahead"]
    laps = ds.common_trace_laps(zandvoort["cars"][car], zandvoort["cars"][rival])
    row = ds._evaluate_lap(zandvoort, track, params, car, rival, laps, 0, laps[0])
    assert row is not None, "one refused zone must not take the lap down"
    assert calls["refused"] >= 1, "the s = 0 zone on lap 1 must genuinely refuse"
    evaluated = {c["requested_zone"] for c in row["zone_candidates"]}
    boundary = {z.name for z in track.zones if z.s_straight_start == 0.0}
    assert evaluated and not (evaluated & boundary)


def test_every_opportunity_unresolvable_is_a_reported_refusal_not_a_traceback(
        zandvoort, monkeypatch):
    """A trace with nothing causal left refuses with the reason, and names the laps."""
    def always_refuse(trace, lap, s_m):
        raise ValueError(f"no causal trace samples for lap {lap} at s <= {s_m:.1f} m")

    monkeypatch.setattr(ds, "belief_at_position", always_refuse)
    car, rival = zandvoort["battles"][0]["car"], zandvoort["battles"][0]["ahead"]
    with pytest.raises(ValueError, match="no causal decision opportunity on any common lap"):
        ds.evaluate_decision_trace_from_payload(zandvoort, car, rival)


# ---------------------------------------------------------------------------
# observed outcome vs driver action vs recommendation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def replay():
    p = RACES / "2026_r1_R.json"
    if not p.exists():
        pytest.skip("Melbourne artefact not analysed")
    d = json.loads(p.read_text(encoding="utf-8"))
    car, rival = "GAS", "OCO"
    t = d["cars"][car]["trace"]["t"]
    cutoff = float(t[len(t) // 2])          # mid-race, so the window closes inside it
    return ds.historical_replay(d, car, rival, cutoff, horizon_s=150.0)


def test_observed_outcome_fields_are_present_and_agree_with_the_delta(replay):
    for k in ("observed_position_before", "observed_position_after",
              "observed_position_delta", "observed_outcome",
              "observed_outcome_basis", "driver_action_observed",
              "counterfactual_status"):
        assert k in replay, k
    before = replay["observed_position_before"]
    after = replay["observed_position_after"]
    delta = replay["observed_position_delta"]
    if before is None or after is None:
        assert delta is None                 # unknown is never folded into zero
        assert "not published" in replay["observed_outcome"]
    else:
        assert delta == before - after
        if delta > 0:
            assert replay["observed_outcome"] == (
                f"gained {delta} position" + ("s" if delta > 1 else ""))
        elif delta < 0:
            assert replay["observed_outcome"] == (
                f"lost {-delta} position" + ("s" if -delta > 1 else ""))
        else:
            assert replay["observed_outcome"] == "no position change"


def test_driver_action_is_never_claimed_and_the_counterfactual_is_unresolved(replay):
    assert replay["driver_action_observed"] is False
    assert replay["counterfactual_status"] == "unresolved_from_historical_telemetry"


def test_the_legacy_verdict_is_retained_but_marked_legacy(replay):
    # Retained for compatibility only. If these disappear, readers break; if they
    # are ever rendered, the invalid claim is back.
    assert replay["actual_action_is_legacy"] is True
    assert replay["matches_recommendation_is_legacy"] is True
    assert "actual_action" in replay and "matches_recommendation" in replay


def test_the_row_builder_surfaces_observed_outcome_and_marks_the_legacy_fields():
    src = (REPO / "simulation" / "api" / "main.py").read_text(encoding="utf-8")
    for f in ("observed_position_delta", "observed_outcome",
              "counterfactual_status", "driver_action_observed"):
        assert f'"{f}"' in src, f
    assert '"matches_recommendation_is_legacy": True' in src
    assert '"actual_action_is_legacy": True' in src
    # The bundle cache key must move with the row shape, or a pre-repair bundle is
    # served against post-repair semantics: an empty outcome column next to a
    # recommendation, with nothing saying why.
    assert "SITUATIONS_SCHEMA = 3" in src


def _code_lines(src: str) -> list[str]:
    """Source with comment-only and JSX-comment lines dropped.

    The prohibition is on RENDERING a matched/diverged verdict. The comments are
    required to keep saying why it is gone, so they must not trip the scan.
    """
    out, in_block = [], False
    for raw in src.splitlines():
        line = raw.strip()
        if in_block:
            if "*/" in line:
                in_block = False
            continue
        if line.startswith("/*") or line.startswith("{/*"):
            if "*/" not in line:
                in_block = True
            continue
        if line.startswith("//") or line.startswith("*"):
            continue
        out.append(raw)
    return out


def test_situations_view_renders_no_driver_followed_or_disobeyed_verdict():
    src = (REPO / "simulation" / "app" / "src" / "views"
           / "Situations.tsx").read_text(encoding="utf-8")
    code = "\n".join(_code_lines(src))
    # The position-derived action and the verdict built on it are not read at all.
    for banned in ("matches_recommendation", "actual_action",
                   "inferred_action_from_position"):
        assert banned not in code, banned
    # And no chip or cell may word an outcome as obedience.
    for word in ("diverged", "matched", "followed", "disobeyed", "obeyed"):
        assert word not in code.lower(), word
    # What must be there instead.
    assert "observed_outcome" in code
    assert "recommendation === 'ATTACK'" in code          # ATTACK filter on canonical P2
    assert "Counterfactual: unresolved from historical telemetry" in src
