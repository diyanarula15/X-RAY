"""Physics acceptance tests."""
from __future__ import annotations

import numpy as np
import pytest

from xray.constants import (E_STORE_MAX, P_MGUK_MAX, TAPER_V_END,
                            TAPER_V_START, mguk_power_limit_normal,
                            mguk_power_limit_overtake, p_mguk_ceiling)
from xray.sim import FOLLOWER, LEADER, run_sim
from xray.track import circuit_sigma
from xray.vehicle import VehicleParams, braking_distance

CARS = (LEADER, FOLLOWER)


def test_taper_ceiling():
    assert p_mguk_ceiling(289 / 3.6) == pytest.approx(P_MGUK_MAX)
    assert p_mguk_ceiling(TAPER_V_START) == pytest.approx(P_MGUK_MAX)
    assert p_mguk_ceiling(346 / 3.6) == 0.0
    assert p_mguk_ceiling(TAPER_V_END) == 0.0
    vs = np.linspace(TAPER_V_START, TAPER_V_END, 200)
    ceil = p_mguk_ceiling(vs)
    assert np.all(np.diff(ceil) <= 1e-9), "ceiling must be monotonically decreasing"
    assert ceil[0] == pytest.approx(P_MGUK_MAX)
    assert ceil[-1] == pytest.approx(0.0)


def test_2026_mguk_power_curve_boundaries():
    eps = 0.01 / 3.6
    assert mguk_power_limit_normal((290 / 3.6) - eps) == pytest.approx(P_MGUK_MAX)
    assert mguk_power_limit_normal(290 / 3.6) == pytest.approx(P_MGUK_MAX)
    assert mguk_power_limit_normal((290 / 3.6) + eps) < P_MGUK_MAX
    assert mguk_power_limit_normal(337.5 / 3.6) == pytest.approx(112_500.0)
    assert mguk_power_limit_normal(340 / 3.6) == pytest.approx(100_000.0)
    assert mguk_power_limit_normal((345 / 3.6) - eps) > 0.0
    assert mguk_power_limit_normal(345 / 3.6) == 0.0
    assert mguk_power_limit_normal(355 / 3.6) == 0.0

    assert mguk_power_limit_overtake(337.5 / 3.6) == pytest.approx(P_MGUK_MAX)
    assert mguk_power_limit_overtake((337.5 / 3.6) + eps) < P_MGUK_MAX
    assert mguk_power_limit_overtake(340 / 3.6) == pytest.approx(300_000.0)
    assert mguk_power_limit_overtake((345 / 3.6) - eps) > mguk_power_limit_normal((345 / 3.6) - eps)
    assert mguk_power_limit_overtake(345 / 3.6) == pytest.approx(200_000.0)
    assert mguk_power_limit_overtake((355 / 3.6) - eps) > 0.0
    assert mguk_power_limit_overtake(355 / 3.6) == 0.0


def test_energy_balance_closes(gt):
    """Per lap: E_end - E_start == harvested - deployed.

    Manual Override is a legal deployment allocation, not instant stored energy.
    """
    for car in CARS:
        tr = gt.cars[car]
        k = gt.n_laps - 1
        delta = np.diff(tr.e_lap_open)[:k]
        flows = (tr.harvested_lap - tr.deployed_lap)[:k]
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


def test_one_regulation_curve_and_every_module_uses_it():
    """Invariant 10, as a test rather than a convention.

    `regs.py` kept its own linear 290->355 km/h ramp after `constants.py` moved to
    the published piecewise curve. The simulator deployed on the curve and
    `balance.py` bounded it with the ramp, so between 336 and 340 km/h the bound
    permitted 18 kW less than the car actually used. At dt = 5 ms that is 91 J,
    and it is exactly the 91.6 J by which the clean-truth containment test found
    the upper bound excluding the true theta -- a set that excludes the truth is
    not conservative, it is wrong. Nothing caught it for an entire iteration
    because six test modules, that one among them, were failing to collect.
    """
    from xray import regs
    from xray.constants import (POWER_UNIT_2026, mguk_power_limit,
                                mguk_power_limit_normal, mguk_power_limit_overtake)

    # regs must not restate the breakpoints, it must derive them.
    assert regs.TAPER_V_FULL_KMH == pytest.approx(
        POWER_UNIT_2026.normal_curve.full_power_until_mps * 3.6)
    assert regs.TAPER_V_ZERO_KMH == pytest.approx(
        POWER_UNIT_2026.normal_curve.zero_at_mps * 3.6)
    assert regs.MO_TAPER_V_ZERO_KMH == pytest.approx(
        POWER_UNIT_2026.overtake_curve.zero_at_mps * 3.6)

    # and its taper must BE the curve, at every speed, both modes.
    v = np.linspace(200 / 3.6, 380 / 3.6, 400)
    np.testing.assert_allclose(regs.taper(v) * P_MGUK_MAX,
                               mguk_power_limit_normal(v), rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(regs.taper(v, manual_override=True) * P_MGUK_MAX,
                               mguk_power_limit_overtake(v), rtol=1e-12, atol=1e-6)

    # the dispatcher agrees with both branches
    np.testing.assert_allclose(mguk_power_limit(v, False), mguk_power_limit_normal(v))
    np.testing.assert_allclose(mguk_power_limit(v, True), mguk_power_limit_overtake(v))

    # no module may hard-code a breakpoint in km/h
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "xray"
    for path in root.rglob("*.py"):
        if path.name == "constants.py":
            continue
        text = path.read_text()
        for literal in ("355.0", "345.0", "337.5"):
            assert f"= {literal}" not in text, (
                f"{path.name} restates a regulation breakpoint ({literal}); "
                "derive it from constants.POWER_UNIT_2026")


def test_overtake_allowance_is_a_recharge_ceiling_not_stored_energy():
    """The 0.5 MJ may raise what can be RECOVERED. It may not appear in the store.

    The previous model ran `st.E += min(0.5 MJ, room)` at the lap boundary, which
    materialised half a megajoule out of a regulation; the per-lap balance test
    only closed because it added the same mom term to both sides of the equation.
    """
    from xray.constants import POWER_UNIT_2026 as R
    from xray.constants import recharge_allowance_j
    from xray.vehicle import CarState

    assert recharge_allowance_j(False) == R.max_recharge_without_overtake_j
    assert recharge_allowance_j(True) == R.max_recharge_with_overtake_j
    assert (recharge_allowance_j(True) - recharge_allowance_j(False)
            == pytest.approx(R.overtake_allocation_j))

    # the allowance is a field of its own, and it is not the store
    st = CarState(E=1.0e6)
    assert st.manual_overtake_allocation_j == 0.0
    assert st.overtake_active is False
    st.manual_overtake_allocation_j = R.overtake_allocation_j
    assert st.E == 1.0e6, "granting the allowance must not touch stored energy"


def test_manual_override_does_not_create_energy_over_a_race(gt):
    """End to end: on every lap a car was granted Overtake, the store still only
    changes by harvested minus deployed."""
    for car in CARS:
        tr = gt.cars[car]
        k = gt.n_laps - 1
        granted = tr.mom_lap[:k] > 0.0
        if not granted.any():
            continue
        delta = np.diff(tr.e_lap_open)[:k]
        flows = (tr.harvested_lap - tr.deployed_lap)[:k]
        assert np.abs(delta[granted] - flows[granted]).max() < 1000.0, (
            f"{car}: Overtake laps do not close on harvest minus deployment")
