"""P2 §8: the dataset audit that decides whether p_pass may become empirical.

The point of these tests is not that a model fits well. It is that the code
REFUSES to fit when the data cannot support the claim, and that the label is
built from decision-time features plus a forward-looking target and nothing else.
"""
from __future__ import annotations

import pytest

from xray.passmodel import (CONFIRMATION_LAPS, DatasetAudit, PassSample,
                            audit_dataset, build_dataset, dataset_fingerprint,
                            fit, load_payloads)


def _race(n_laps=12):
    """A synthetic payload in the real analysed-race schema."""
    laps, gaps = [], []
    for L in range(1, n_laps + 1):
        for i, drv in enumerate(("AAA", "BBB")):
            # BBB starts behind and gets ahead on lap 5, permanently
            pos = (1 if drv == "AAA" else 2)
            if L >= 5 and drv == "BBB":
                pos = 1
            elif L >= 5 and drv == "AAA":
                pos = 2
            laps.append({"driver": drv, "lap": L, "position": pos,
                         "lap_time": 90.0, "t_end": 90.0 * L,
                         "compound": "MEDIUM", "tyre_life": L})
        gaps.append({"lap": L, "car": "BBB", "ahead": "AAA",
                     "gap_s": 0.6, "position": 2})
    return {"id": "test_race", "circuit": "Test", "laps": laps, "gaps": gaps}


def test_the_label_requires_the_pass_to_stick():
    d = build_dataset(_race())
    by_lap = {s.lap: s for s in d}
    assert by_lap[4].pass_success == 1, "ahead next lap and held -> a pass"
    assert by_lap[1].pass_success == 0, "no position change -> not a pass"
    for s in d:
        if s.pass_success is not None:
            assert s.pass_success in (0, 1)


def test_the_attacker_must_start_behind():
    d = build_dataset(_race())
    assert all(s.position_delta > 0 for s in d), (
        "a car already ahead is not attempting a pass")


def test_a_pit_cycle_is_excluded_using_only_published_fields():
    """Position changes across a stop are not overtakes, and the data says so."""
    r = _race()
    for row in r["laps"]:
        if row["driver"] == "AAA" and row["lap"] >= 5:
            row["compound"] = "HARD"        # a real compound change = a stop
            row["tyre_life"] = row["lap"] - 4
    d = build_dataset(r)
    lap4 = next(s for s in d if s.lap == 4)
    assert lap4.excluded
    assert "pit cycle" in lap4.exclusion_reason
    assert lap4.pass_success is not None, "the label is still computed, just unused"


def test_a_tyre_life_reset_alone_is_enough_to_detect_a_stop():
    r = _race()
    for row in r["laps"]:
        if row["driver"] == "BBB" and row["lap"] >= 5:
            row["tyre_life"] = row["lap"] - 4      # same compound, younger tyre
    d = build_dataset(r)
    assert any(s.excluded and "pit cycle" in s.exclusion_reason for s in d)


def test_the_confirmation_window_cannot_run_past_the_race():
    """A pass cannot be confirmed with laps that do not exist.

    Uses a race where the attacker never gets by, so there IS a sample at every
    lap -- in `_race` the attacker stops being an attacker once it is ahead.
    """
    r = _race(n_laps=6)
    for row in r["laps"]:                       # nobody ever passes
        row["position"] = 1 if row["driver"] == "AAA" else 2
    d = build_dataset(r)
    tail = [x for x in d if x.lap > 6 - CONFIRMATION_LAPS]
    assert tail, "fixture produced no late-race samples"
    assert all(x.excluded for x in tail)
    assert all("past the end" in x.exclusion_reason for x in tail)
    early = [x for x in d if x.lap <= 6 - CONFIRMATION_LAPS]
    assert early and all(not x.excluded for x in early)
    assert all(x.pass_success == 0 for x in early), "nobody passed in this fixture"


def test_features_are_decision_time_only():
    """No field on a sample may be a future observation except the label."""
    s = build_dataset(_race())[0]
    feature_fields = {f for f in PassSample.__dataclass_fields__
                      if f not in ("pass_success", "excluded", "exclusion_reason")}
    for banned in ("future", "next_lap", "final_position", "outcome", "result"):
        assert not any(banned in f for f in feature_fields), (
            f"a future-looking feature appeared: {banned}")
    assert s.pass_success in (0, 1, None)


def test_the_audit_blocks_fitting_and_fit_refuses():
    d = build_dataset(_race())
    a = audit_dataset(d, [_race()])
    assert not a.fit_permitted
    assert a.blocking_reasons
    with pytest.raises(RuntimeError, match="remains SYNTHETIC"):
        fit(d, a)


def test_the_audit_names_the_signals_it_does_not_have():
    a = audit_dataset(build_dataset(_race()), [_race()])
    joined = " ".join(a.missing_signals)
    assert "safety car" in joined or "VSC" in joined
    assert "delta_v" in joined
    assert any("no safety-car/VSC channel" in r for r in a.blocking_reasons)


def test_a_dataset_that_would_otherwise_qualify_is_still_blocked_on_signals():
    """Even with plenty of rows, the qualitative blockers stand.

    This is the test that stops a future contributor from fixing the sample
    count and declaring the model empirical.
    """
    a = DatasetAudit(n_raw=99999, n_usable=50000, n_positive=9000,
                     races=[f"r{i}" for i in range(20)])
    a2 = audit_dataset(build_dataset(_race()), [_race()])
    qualitative = [r for r in a2.blocking_reasons
                   if "safety-car" in r or "delta_v" in r]
    assert len(qualitative) == 2, "the qualitative blockers were removed"


def test_the_fingerprint_changes_with_the_data():
    a = build_dataset(_race())
    assert dataset_fingerprint(a) == dataset_fingerprint(build_dataset(_race())), (
        "the same data must fingerprint the same, or provenance is meaningless")
    changed = _race()
    for g in changed["gaps"]:
        g["gap_s"] = 1.9                        # a feature the samples carry
    assert dataset_fingerprint(build_dataset(changed)) != dataset_fingerprint(a)


def test_the_real_payloads_produce_a_blocked_audit():
    """The actual out/races payloads, if present. Skipped when absent."""
    from pathlib import Path
    races = Path(__file__).resolve().parent.parent / "out" / "races"
    payloads = load_payloads(races)
    if not payloads:
        pytest.skip("no analysed race payloads present")
    samples = []
    for p in payloads:
        samples += build_dataset(p)
    a = audit_dataset(samples, payloads)
    assert a.n_usable > 0
    assert not a.fit_permitted, (
        "the real dataset unexpectedly passed the audit; if that is genuine, "
        "a fitting path and a held-out calibration report are now required")
    # pit contamination must actually be being caught
    assert any("pit cycle" in k for k in a.exclusions)
