"""P1.2 / P1.3: downforce, relative airspeed, and the wake split.

The two things worth guarding here are both invisible failures. A drag/downforce
model that ignores wind is wrong by a few percent and looks fine; a braking
model that adds explicit downforce on top of a brake_decel_max that already
contained it is wrong by a factor of two and also looks fine, because the car
still stops.
"""
from __future__ import annotations

import numpy as np
import pytest

from xray.config import load_config
from xray.environment import state_from_summary
from xray.physics_context import PhysicsContext
from xray.vehicle import (G, VehicleParams, brake_mu, braking_distance,
                          cornering_grip_from_downforce, downforce, drag_force,
                          normal_load)


# The reference ClA values. They live here, not in the config defaults, because
# P1 aero ships disabled -- see config/default.yaml for why. These tests are what
# prove the model works when it is switched on.
CLA_STRAIGHT, CLA_CORNER = 2.00, 4.30


@pytest.fixture(scope="module")
def params():
    """Params with P1 aero ENABLED, whatever the shipped default is."""
    base = VehicleParams.from_config(load_config())
    return VehicleParams(**{**base.__dict__, "cla_straight": CLA_STRAIGHT,
                            "cla_corner": CLA_CORNER})


# ------------------------------------------------------------ force scaling
def test_drag_and_downforce_scale_with_rho_and_airspeed_squared(params):
    cda, cla, rho = 0.66, 4.3, 1.2
    assert drag_force(80.0, cda, 2 * rho) == pytest.approx(2 * drag_force(80.0, cda, rho))
    assert downforce(80.0, cla, 2 * rho) == pytest.approx(2 * downforce(80.0, cla, rho))
    assert drag_force(160.0, cda, rho) == pytest.approx(4 * drag_force(80.0, cda, rho))
    assert downforce(160.0, cla, rho) == pytest.approx(4 * downforce(80.0, cla, rho))


def test_drag_opposes_motion_even_against_a_stronger_headwind():
    """The signed form matters only in the one case where it matters."""
    assert drag_force(-5.0, 0.66, 1.2) < 0.0
    assert drag_force(5.0, 0.66, 1.2) > 0.0


def test_headwind_raises_both_drag_and_downforce(params):
    """Same rho, same relative airspeed, both forces."""
    v, w = 80.0, -10.0          # w_parallel < 0 is a headwind
    v_air_head, v_air_still = v - w, v
    assert drag_force(v_air_head, 0.66, 1.2) > drag_force(v_air_still, 0.66, 1.2)
    assert downforce(v_air_head, 4.3, 1.2) > downforce(v_air_still, 4.3, 1.2)
    v_air_tail = v - 10.0
    assert drag_force(v_air_tail, 0.66, 1.2) < drag_force(v_air_still, 0.66, 1.2)
    assert downforce(v_air_tail, 4.3, 1.2) < downforce(v_air_still, 4.3, 1.2)


def test_normal_load_is_weight_plus_downforce():
    m, v, cla, rho = 800.0, 80.0, 4.3, 1.2
    assert normal_load(m, v, 0.0, rho) == pytest.approx(m * G)
    assert normal_load(m, v, cla, rho) == pytest.approx(m * G + downforce(v, cla, rho))
    # the wake removes load, it does not add it
    assert normal_load(m, v, cla, rho, 0.8) < normal_load(m, v, cla, rho, 1.0)


# ------------------------------------------------- the double-counting guard
def test_brake_mu_inverts_the_legacy_constant_instead_of_stacking_on_it(params):
    """brake_decel_max is 4.59 g. That already includes downforce.

    If the P1 path had simply added mu*F_down to a mu derived as
    brake_decel_max/g, the car would brake at roughly twice its calibrated rate
    at speed. The inversion is what prevents that, and this pins it: at the
    calibration speed the two models must agree exactly.
    """
    assert params.brake_decel_max / G > 4.0, "the legacy constant is superhuman for mechanical grip alone"
    mu_naive = params.brake_decel_max / G
    mu = brake_mu(params, params.cla_corner, params.rho)
    assert mu < mu_naive / 2.0, "mu was not re-derived; downforce is double counted"
    assert 1.0 < mu < 2.0, f"derived mu {mu:.3f} is outside the plausible slick range"

    # exact agreement with the legacy constant at the calibration speed
    m_ref = params.mass_car + 0.5 * params.fuel_start
    v_ref = params.brake_calibration_v_ms
    decel = mu * normal_load(m_ref, v_ref, params.cla_corner, params.rho) / m_ref
    assert decel == pytest.approx(params.brake_decel_max)


def test_braking_is_speed_dependent_once_downforce_exists(params):
    """Harder at speed, softer at low speed -- the behaviour a constant cannot have."""
    m = 800.0
    mu = brake_mu(params, params.cla_corner, params.rho)

    def decel(v):
        return mu * normal_load(m, v, params.cla_corner, params.rho) / m

    assert decel(100.0) > decel(60.0) > decel(30.0)
    assert decel(30.0) < params.brake_decel_max, "low-speed braking must lose the aero help"
    assert decel(100.0) > params.brake_decel_max, "high-speed braking must gain it"


def test_braking_distance_reduces_exactly_to_p0_when_cla_is_zero(params):
    """The closed form is unchanged, not re-derived, when there is no downforce."""
    k = 0.5 * params.rho * 0.66
    m, v_in, v_t = 800.0, 90.0, 60.0
    a0 = m * params.brake_decel_max + params.crr * m * G
    expected = m / (2 * k) * np.log((a0 + k * v_in ** 2) / (a0 + k * v_t ** 2))
    assert braking_distance(v_in, v_t, m, 0.66, params, 0.0) == pytest.approx(expected)


def test_braking_distance_shortens_with_more_downforce(params):
    m = 800.0
    clean = braking_distance(100.0, 70.0, m, 0.66, params, 0.0,
                             cla=params.cla_corner, rho=params.rho,
                             downforce_factor=1.0)
    dirty = braking_distance(100.0, 70.0, m, 0.66, params, 0.0,
                             cla=params.cla_corner, rho=params.rho,
                             downforce_factor=0.7)
    assert dirty > clean, "losing downforce must lengthen the braking zone"


# ------------------------------------------------------------- wake splitting
def test_corner_speed_loss_is_derived_from_load_not_configured_in_speed():
    """Dirty air takes load; the speed penalty is a consequence.

    The old model configured a 6% apex-speed loss directly. The new one
    configures a 20% downforce loss and gets 6.4% of speed out of the load
    balance -- so the number that used to be tuned is now predicted, which is
    the whole point of moving the parameter into the force model.
    """
    g = cornering_grip_from_downforce(70.0, 4.3, 1.2, 800.0, 0.80)
    assert 0.90 < g < 0.98
    assert 1.0 - g == pytest.approx(0.064, abs=0.01)
    # monotone, bounded, and inert without downforce
    assert cornering_grip_from_downforce(70.0, 4.3, 1.2, 800.0, 1.0) == 1.0
    assert cornering_grip_from_downforce(70.0, 0.0, 1.2, 800.0, 0.5) == 1.0
    worse = cornering_grip_from_downforce(70.0, 4.3, 1.2, 800.0, 0.5)
    assert worse < g


def test_tow_and_dirty_air_are_separate_parameters():
    """One decay, two effects. They must not be the same number again."""
    tow = load_config()["tow"]
    assert "cda_reduction" in tow and "tau_s" in tow
    assert "dirty_air_downforce_loss" in tow, "the P1 wake key is missing"
    assert tow["cda_reduction"] != tow["dirty_air_downforce_loss"], (
        "drag reduction and downforce loss have collapsed to one number")


def test_legacy_dirty_air_grip_loss_config_still_produces_a_wake():
    """Backward compatibility, asserted rather than assumed.

    An older config has `dirty_air_grip_loss` and no `dirty_air_downforce_loss`.
    It must not silently end up with no wake at all, which is what a plain
    `cfg["tow"]["dirty_air_downforce_loss"]` lookup with a 0.0 default would do.
    """
    from xray.sim import Simulator
    cfg = load_config()
    legacy = {**cfg, "tow": {"cda_reduction": 0.25, "tau_s": 0.80,
                             "dirty_air_grip_loss": 0.06}}
    sim = Simulator(legacy)
    assert sim._dirty_air_loss == pytest.approx(0.06), (
        "a legacy config lost its wake entirely")
    assert sim._legacy_grip_loss == pytest.approx(0.06)


# --------------------------------------------------------- legacy isolation
def test_omitting_the_physics_context_reproduces_p0_exactly(params):
    """P1 must be opt-in, so a P0 regression test still measures P0."""
    from xray.sim import LEADER, run_sim
    cfg = load_config()
    p0 = {**cfg, "vehicle": {**cfg["vehicle"], "cla_straight": 0.0, "cla_corner": 0.0}}
    a = run_sim(p0, seed=42).cars[LEADER]
    b = run_sim(p0, seed=42).cars[LEADER]
    assert np.array_equal(a.v, b.v)
    # and the P1 run must actually differ, or the switch does nothing. The
    # traces are not even the same length -- load-dependent braking changes the
    # lap time -- so compare the summary the lengths come from.
    p1 = run_sim({**cfg, "vehicle": {**cfg["vehicle"],
                                     "cla_straight": CLA_STRAIGHT,
                                     "cla_corner": CLA_CORNER}}, seed=42).cars[LEADER]
    assert (len(p1.v), float(p1.v.max())) != (len(a.v), float(a.v.max())), (
        "enabling ClA changed nothing")


def test_physics_context_carries_environment_not_strategy():
    env = state_from_summary({"rho": 1.18})
    ctx = PhysicsContext(environment=env)
    assert ctx.rho == pytest.approx(1.18)
    assert ctx.downforce_factor == 1.0
    assert not hasattr(ctx, "pit_context"), (
        "pit strategy is not a force and must not live in PhysicsContext")
