"""P3.5 Part 3: fixed-E leakage guard and conditional deployment semantics."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from xray.hardened_eval import (deterministic_training_subset,
                                score_with_independent_baseline,
                                train_optimal_fixed_energy)
from xray.p3_validation import TARGETS, ValidationExample
from xray.realfit import (Kin, RealNuisanceFit, _conditional_deployment_step,
                          belief_from_deployment)


def _example(race: str, i: int, target: str = TARGETS[0],
             target_v: float = 2.0, xray_energy_j: float = 0.0) -> ValidationExample:
    return ValidationExample(
        race_id=race, event=race, circuit="unit", driver=f"D{i % 3}",
        target=target, cutoff_t=float(i), target_t=float(i + 5),
        horizon_s=5.0, s0=float(i * 10), v0=50.0, lap=i // 10,
        target_s=float(i * 10 + 100), target_v=target_v,
        xray_energy_j=xray_energy_j, cda_straight=0.7,
        cda_lo=None, cda_hi=None,
    )


def _payloads(*races: str) -> list[dict]:
    return [{"id": r} for r in races]


def _patch_rollout(monkeypatch):
    def fake_rollout(payload, ex, energy_j, cfg=None):
        return float(energy_j) / 1e6
    monkeypatch.setattr("xray.p3_validation.rollout_speed", fake_rollout)


def test_held_out_race_never_participates_in_fixed_e_selection(monkeypatch):
    _patch_rollout(monkeypatch)
    examples = ([_example("A", i, target_v=1.0) for i in range(6)]
                + [_example("B", i, target_v=3.0) for i in range(6)]
                + [_example("C", i, target_v=0.0) for i in range(6)])
    out = score_with_independent_baseline(
        _payloads("A", "B", "C"), examples, cfg={}, max_examples=8)

    fit = out["chosen_fixed_energy_by_fold"]["C"][TARGETS[0]]
    assert "C" not in fit["race_counts_used"]
    assert fit["n_train_available"] == 12
    assert fit["n_train_used"] == 8


def test_changing_held_out_targets_does_not_change_training_fixed_e(monkeypatch):
    _patch_rollout(monkeypatch)
    examples = ([_example("A", i, target_v=2.0) for i in range(5)]
                + [_example("B", i, target_v=2.0) for i in range(5)]
                + [_example("C", i, target_v=0.0) for i in range(5)])
    changed = [replace(e, target_v=4.0) if e.race_id == "C" else e
               for e in examples]

    a = score_with_independent_baseline(
        _payloads("A", "B", "C"), examples, cfg={})["chosen_fixed_energy_by_fold"]
    b = score_with_independent_baseline(
        _payloads("A", "B", "C"), changed, cfg={})["chosen_fixed_energy_by_fold"]

    assert a["C"][TARGETS[0]]["energy_j"] == b["C"][TARGETS[0]]["energy_j"]
    assert a["C"][TARGETS[0]]["subset_fingerprint"] == \
        b["C"][TARGETS[0]]["subset_fingerprint"]


def test_changing_estimator_outputs_does_not_change_train_optimal_fixed_e(monkeypatch):
    _patch_rollout(monkeypatch)
    train = [_example("A", i, target_v=1.0, xray_energy_j=0.0) for i in range(4)]
    shifted = [replace(e, xray_energy_j=4.0e6) for e in train]
    by_id = {"A": {"id": "A"}}

    a = train_optimal_fixed_energy(train, by_id, cfg={})
    b = train_optimal_fixed_energy(shifted, by_id, cfg={})

    assert a[TARGETS[0]]["energy_j"] == b[TARGETS[0]]["energy_j"]
    assert a[TARGETS[0]]["subset_fingerprint"] == b[TARGETS[0]]["subset_fingerprint"]


def test_deterministic_subset_selection_repeats_the_same_rows_and_energy(monkeypatch):
    _patch_rollout(monkeypatch)
    examples = []
    for race in ("A", "B", "C"):
        examples.extend(_example(race, i, target_v=2.0) for i in range(20))
    by_id = {r: {"id": r} for r in ("A", "B", "C")}

    s1, m1 = deterministic_training_subset(examples, max_examples=11)
    s2, m2 = deterministic_training_subset(list(reversed(examples)), max_examples=11)
    e1 = train_optimal_fixed_energy(s1, by_id, cfg={}, max_examples=11)
    e2 = train_optimal_fixed_energy(s2, by_id, cfg={}, max_examples=11)

    assert [e.cutoff_t for e in s1] == [e.cutoff_t for e in s2]
    assert m1["subset_fingerprint"] == m2["subset_fingerprint"]
    assert e1[TARGETS[0]]["energy_j"] == e2[TARGETS[0]]["energy_j"]


def test_conditional_low_energy_particle_is_not_penalized_when_support_intersects():
    E = np.array([100.0])
    h = np.array([0.0])
    lo = np.array([0.0])
    hi = np.array([500.0])
    split = np.array([0.5])

    avail, deployed, contradiction, empty = _conditional_deployment_step(
        E, h, lo, hi, split)

    assert avail[0] == 100.0
    assert deployed[0] == 50.0
    assert contradiction[0] == 0.0
    assert not bool(empty[0])


def test_conditional_empty_intersection_applies_contradiction_penalty():
    _, deployed, contradiction, empty = _conditional_deployment_step(
        np.array([100.0]), np.array([0.0]), np.array([200.0]),
        np.array([500.0]), np.array([0.5]))

    assert deployed[0] == 100.0
    assert contradiction[0] == 100.0
    assert bool(empty[0])


def test_regulatory_upper_bound_remains_a_bound_not_the_deployment_location():
    _, deployed, contradiction, empty = _conditional_deployment_step(
        np.array([1000.0]), np.array([0.0]), np.array([100.0]),
        np.array([350.0]), np.array([0.25]))

    assert deployed[0] == 162.5
    assert deployed[0] < 350.0
    assert contradiction[0] == 0.0
    assert not bool(empty[0])


def _kin(n=240):
    return Kin(s=np.linspace(0, 2000, n), v=np.full(n, 60.0), a=np.zeros(n),
               dt=np.full(n, 0.1), t=np.linspace(0, 24, n),
               mass=np.full(n, 800.0), cda_scale=np.ones(n),
               sin_grade=np.zeros(n), in_deployment_zone=np.ones(n, bool),
               ceiling=np.full(n, 3.0e5), throttle=np.full(n, 90.0),
               brake=np.zeros(n), valid=np.ones(n, bool),
               lap=np.repeat(np.arange(3), n // 3)[:n])


def _fit():
    return RealNuisanceFit(
        cda_hat=0.8, cda_lo=0.6, cda_hi=1.0, cda_sigma=0.08,
        v_wind_hat=0.0, rho=1.2, identifiability=0.5, n_samples=240,
        n_binding=120, n_coast=20, coast_cda=0.8,
        residual_rms=2.0e4, systematic_rms=2.0e4)


def test_usable_floor_probability_is_bounded_and_reserve_not_falsely_identified():
    kin = _kin()
    n = len(kin.v)
    tr = {"deploy": np.full(n, 1.0e5),
          "deploy_lo": np.zeros(n),
          "deploy_hi": np.full(n, 2.0e5),
          "harvest": np.full(n, 0.8e5),
          "braking": np.zeros(n, bool),
          "P_obs": np.full(n, 1.0e5),
          "P_ice": np.zeros(n)}

    bel = belief_from_deployment(kin, tr, _fit(), n_particles=120, seed=11,
                                 boundary="conditional")

    p = np.asarray(bel["usable_p_at_floor"], float)
    p = p[np.isfinite(p)]
    assert len(p)
    assert float(p.min()) >= 0.0
    assert float(p.max()) <= 1.0
    assert bel["reserve_identified"] is False
