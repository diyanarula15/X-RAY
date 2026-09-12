from __future__ import annotations

import pytest

from xray.p3_validation import (build_validation_examples, fixed_energy_by_target,
                                leave_one_race_splits)


def _payload(race_id: str, energy: float = 1.0):
    return {
        "id": race_id,
        "event": race_id,
        "circuit": "Test",
        "circuit_geometry": {
            "length": 1000.0,
            "s": [0.0, 500.0, 999.0],
            "grade": [0.0, 0.0, 0.0],
            "zones": [{"name": "A", "s_straight_start": 100.0,
                       "s_straight_end": 500.0, "s_end": 650.0,
                       "braking_severity": 0.5, "apex_v": 35.0}],
        },
        "cars": {
            "AAA": {
                "cda": 0.8,
                "cda_lo": 0.7,
                "cda_hi": 0.9,
                "trace": {
                    "t": [0.0, 1.0, 3.1, 5.1, 8.2, 10.4, 13.6],
                    "s": [100.0, 160.0, 230.0, 310.0, 390.0, 470.0, 540.0],
                    "v": [40.0, 42.0, 45.0, 48.0, 51.0, 53.0, 45.0],
                    "lap": [1, 1, 1, 1, 1, 1, 1],
                    "usable_mean": [energy] * 7,
                },
            }
        },
    }


def test_part2_targets_are_strictly_after_cutoff():
    examples, _ = build_validation_examples(_payload("r1"), stride=1)
    assert examples
    assert all(e.target_t > e.cutoff_t for e in examples)


def test_leave_one_race_out_split_has_no_race_leakage():
    e1, _ = build_validation_examples(_payload("r1", 1.0), stride=2)
    e2, _ = build_validation_examples(_payload("r2", 3.0), stride=2)
    splits = leave_one_race_splits(e1 + e2)
    for race, split in splits.items():
        assert {e.race_id for e in split["test"]} == {race}
        assert race not in {e.race_id for e in split["train"]}


def test_fixed_energy_baseline_is_train_only_by_target():
    train, _ = build_validation_examples(_payload("train", 2.5), stride=1)
    test, _ = build_validation_examples(_payload("test", 99.0), stride=1)
    fixed = fixed_energy_by_target(train)
    for target in fixed:
        if any(e.target == target for e in train):
            assert fixed[target] == pytest.approx(2.5e6)
    assert all(e.xray_energy_j == pytest.approx(99.0e6) for e in test)
