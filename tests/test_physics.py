"""Physics acceptance tests."""
from __future__ import annotations

import numpy as np
import pytest

from xray.constants import (E_STORE_MAX, P_MGUK_MAX, TAPER_V_END, TAPER_V_START,
                            p_mguk_ceiling)
from xray.sim import FOLLOWER, LEADER, run_sim
from xray.track import circuit_sigma
from xray.vehicle import VehicleParams, braking_distance

CARS = (LEADER, FOLLOWER)


def test_taper_ceiling():
    assert p_mguk_ceiling(289 / 3.6) == pytest.approx(P_MGUK_MAX)
    assert p_mguk_ceiling(TAPER_V_START) == pytest.approx(P_MGUK_MAX)
    assert p_mguk_ceiling(356 / 3.6) == 0.0
    assert p_mguk_ceiling(TAPER_V_END) == 0.0
    vs = np.linspace(TAPER_V_START, TAPER_V_END, 200)
    ceil = p_mguk_ceiling(vs)
    assert np.all(np.diff(ceil) <= 1e-9), "ceiling must be monotonically decreasing"
    assert ceil[0] == pytest.approx(P_MGUK_MAX)
    assert ceil[-1] == pytest.approx(0.0)


def test_energy_balance_closes(gt):
    """Per lap: E_end - E_start == harvested + manual-override - deployed."""
    for car in CARS:
        tr = gt.cars[car]
        k = gt.n_laps - 1
        delta = np.diff(tr.e_lap_open)[:k]
        flows = (tr.harvested_lap - tr.deployed_lap)[:k] + tr.mom_lap[1:k + 1]
        assert np.abs(delta - flows).max() < 1000.0, f"{car} energy does not close"


def test_store_and_harvest_limits_respected(gt):
    for car in CARS:
        tr = gt.cars[car]
        assert tr.E.min() >= -1e-9 and tr.E.max() <= E_STORE_MAX + 1e-6
        assert tr.harvested_lap.max() <= 7.0e6 + 1.0
        # deployment is set from the speed at the start of the step and the
        # trace records the speed at the end of it, so compare against the
        # previous sample's ceiling
        # (sample 0 has no recorded predecessor, so start at 1)
        assert (tr.P_mguk[1:] <= p_mguk_ceiling(tr.v[:-1]) + 1.0).all(), "taper violated"


def test_determinism(cfg):
    a = run_sim(cfg, seed=42, n_laps=3)
    b = run_sim(cfg, seed=42, n_laps=3)
    for car in CARS:
        for field in ("v", "E", "P_mguk", "s_total"):
            assert np.array_equal(getattr(a.cars[car], field),
                                  getattr(b.cars[car], field)), f"{car}.{field}"
    assert np.array_equal(a.gap_s, b.gap_s)
    assert [e.detail for e in a.events] == [e.detail for e in b.events]


def test_lap_time_and_top_speed_are_plausible(gt):
    for car in CARS:
        tr = gt.cars[car]
        lt = tr.lap_time[np.isfinite(tr.lap_time)]
        assert 85.0 <= lt.mean() <= 95.0, f"{car} mean lap time {lt.mean():.1f}s"
        assert 330.0 <= tr.v.max() * 3.6 <= 370.0


def test_taper_window_is_actually_exercised(gt):
    """The estimator's calibration band has to exist, or Stage A has nothing."""
    for car in CARS:
        v = gt.cars[car].v
        assert (v > 340 / 3.6).mean() > 0.0, f"{car} never reaches the taper window"
        assert v.max() > 340 / 3.6


def test_braking_distance_monotonic(cfg):
    p = VehicleParams.from_config(cfg)
    d = [braking_distance(v, 20.0, 838.0, p.cda_straight, p)
         for v in np.linspace(25, 95, 40)]
    assert np.all(np.diff(d) > 0)
    assert braking_distance(30.0, 40.0, 838.0, p.cda_straight, p) == 0.0


def test_track_geometry():
    t = circuit_sigma()
    assert 4800 <= t.length <= 5600
    assert len(t.zones) == 3
    assert [z.name for z in t.zones] == ["A", "B", "C"]
    sev = {z.name: z.braking_severity for z in t.zones}
    assert sev["A"] > sev["B"] > sev["C"]
    for z in t.zones:
        # detection point sits 200 m before the zone straight begins
        assert (z.s_straight_start - z.detection_point) % t.length == pytest.approx(200.0)


def test_mom_grants_respect_the_gap_rule(gt):
    grants = [e for e in gt.events if e.kind == "mom_grant"]
    assert grants, "no Manual Override grants at all"
    assert max(e.detail["gap_s"] for e in grants) <= 1.0
    seen = {}
    for e in grants:
        key = (e.car, e.lap)
        seen[key] = seen.get(key, 0) + 1
    assert max(seen.values()) == 1, "more than one grant per car per lap"
