"""Longitudinal dynamics and the energy store.

This is the ground-truth physics. `estimator.py` must never import it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import (
    E_STORE_MAX,
    G,
    P_ICE_MAX,
    P_MGUK_MAX,
    mguk_power_limit,
    recharge_allowance_j,
)

ACCEL, BRAKE, CORNER = "accel", "brake", "corner"


@dataclass(frozen=True)
class VehicleParams:
    mass_car: float
    fuel_start: float
    fuel_burn_per_lap: float
    cda_straight: float
    cda_corner: float
    crr: float
    rho: float
    brake_decel_max: float
    drivetrain_eff: float
    # P1: lift-area, the downforce analogue of CdA. SYNTHETIC/ASSUMED -- these
    # are not measured F1 values and nothing in this repo has calibrated them;
    # see config/default.yaml and docs/model_inventory.md. Defaulted so every
    # existing VehicleParams(...) call keeps working and P0 stays reproducible.
    cla_straight: float = 0.0
    cla_corner: float = 0.0
    # Speed at which the P1 tyre-limited brake model is made to agree with the
    # legacy constant brake_decel_max. ASSUMED; see brake_mu().
    brake_calibration_v_ms: float = 80.0

    @classmethod
    def from_config(cls, cfg: dict) -> "VehicleParams":
        return cls(**cfg["vehicle"])

    def cda(self, aero_mode: str) -> float:
        return self.cda_corner if aero_mode == "corner" else self.cda_straight

    def cla(self, aero_mode: str) -> float:
        """Lift-area in the same aero mode convention as cda()."""
        return self.cla_corner if aero_mode == "corner" else self.cla_straight


@dataclass
class CarState:
    s: float = 0.0            # m, distance along the current lap
    s_total: float = 0.0      # m, cumulative distance (defines race order)
    v: float = 0.0            # m/s
    E: float = 0.0            # J, energy store
    fuel: float = 70.0        # kg
    harvested_lap: float = 0.0
    deployed_lap: float = 0.0
    manual_overtake_allocation_j: float = 0.0   # J, this lap's extra legal
                              # RECHARGE allowance. Not stored energy, and
                              # not deployable on its own.
    overtake_active: bool = False   # selects the Overtake MGU-K power curve
    lap: int = 0

    @property
    def mass(self) -> float:
        return _MASS_CAR[0] + self.fuel


# module-level so CarState.mass stays cheap; set by Simulator on construction
_MASS_CAR = [768.0]


def set_car_mass(mass_car: float) -> None:
    _MASS_CAR[0] = mass_car


def drag_force(v: float, cda: float, rho: float) -> float:
    """Drag. `v` is RELATIVE AIRSPEED, not ground speed.

    They are the same number whenever there is no wind, which is every P0 call
    and every synthetic run -- Circuit Sigma has no physical heading, so no wind
    can be projected onto it. Under P1 with a real track and a configured frame
    orientation the caller passes v_air; the signed form below keeps the force
    opposing motion if a headwind ever exceeds ground speed.
    """
    return 0.5 * rho * cda * v * abs(v)


def downforce(v_air: float, cla: float, rho: float) -> float:
    """Aerodynamic normal load, same rho and same relative airspeed as drag."""
    return 0.5 * rho * cla * v_air * v_air


def normal_load(m: float, v_air: float, cla: float, rho: float,
                downforce_factor: float = 1.0) -> float:
    """N = m*g + F_down. The load the tyre actually has to work with."""
    return m * G + downforce(v_air, cla, rho) * downforce_factor


def brake_mu(params: VehicleParams, cla: float, rho: float) -> float:
    """Mechanical friction coefficient implied by the configured brake_decel_max.

    Derived, not invented, and the derivation is the point. `brake_decel_max` is
    45 m/s^2, which is 4.59 g -- no tyre does that on mechanical grip alone, so
    that number has ALWAYS had aerodynamic downforce baked into it at some
    unstated speed. Adding an explicit F_down on top of it would count the same
    load twice and make the car brake about twice as hard as it was calibrated
    to.

    So the P1 path inverts it instead: solve for the mu that reproduces exactly
    the legacy deceleration at the reference speed, then let load do the rest.
    Braking becomes speed-dependent -- stronger at 90 m/s than at 30 -- which is
    the physical behaviour the constant never had, while still agreeing with P0
    at the calibration point.
    """
    if cla <= 0.0:
        return params.brake_decel_max / G
    m_ref = params.mass_car + 0.5 * params.fuel_start
    v_ref = params.brake_calibration_v_ms
    a_aero = 0.5 * rho * cla * v_ref * v_ref / m_ref
    return float(params.brake_decel_max / (G + a_aero))


def braking_distance(v_in: float, v_target: float, m: float, cda: float,
                     params: VehicleParams, grade: float = 0.0,
                     cla: float = 0.0, rho: float | None = None,
                     downforce_factor: float = 1.0) -> float:
    """Distance needed to slow from v_in to v_target under brakes + drag.

    Closed form of  m v dv / (F_brake + F_roll + F_grade + k v^2).

    It survives downforce untouched, which is why it is still a closed form: the
    tyre limit mu*(m g + 0.5 rho ClA v^2) is also quadratic in v, so the aero
    term just joins the existing v^2 coefficient as k = 0.5 rho (CdA + mu ClA).
    With cla = 0 this is algebraically identical to the P0 expression.
    """
    if v_in <= v_target:
        return 0.0
    rho = params.rho if rho is None else rho
    mu = brake_mu(params, cla, rho)
    k = 0.5 * rho * (cda + mu * cla * downforce_factor)
    a0 = m * mu * G + params.crr * m * G + m * G * np.sin(grade)
    if a0 <= 0 or k <= 0:
        return float("inf")
    return float(m / (2 * k) * np.log((a0 + k * v_in * v_in) / (a0 + k * v_target * v_target)))


def cornering_grip_from_downforce(v_ref: float, cla: float, rho: float,
                                  m: float, downforce_factor: float) -> float:
    """Apex-speed scale implied by losing part of the aerodynamic normal load.

    The wake does not reach into the tyre and turn the rubber worse; it takes
    away load. For a corner of fixed radius the lateral balance is
    m v^2 / r = mu N, so v scales as sqrt(N) and the speed penalty follows from
    the load ratio instead of being a number in the config.

    Returns 1.0 when there is no downforce to lose, which keeps every P0 call
    (cla = 0) on exactly the old path.
    """
    if cla <= 0.0 or downforce_factor >= 1.0:
        return 1.0
    aero = 0.5 * rho * cla * v_ref * v_ref
    clean = m * G + aero
    dirty = m * G + aero * float(downforce_factor)
    return float(np.sqrt(max(dirty, 0.0) / clean))


def _regime(track, st: CarState, params: VehicleParams, grip: float,
            pt=None, margin: float = 1.02, ctx=None) -> tuple[str, float]:
    """Pick the driving regime for this step. Returns (regime, v_limit_here)."""
    if pt is None:
        pt = track.point(st.s)
    v_lim_raw, grade, is_corner, d1, v1, d2, v2 = pt
    v_lim_here = v_lim_raw * grip if is_corner else v_lim_raw
    if st.v > v_lim_here:
        return BRAKE, v_lim_here
    cda = params.cda_corner if is_corner else params.cda_straight
    cla = params.cla_corner if is_corner else params.cla_straight
    rho = params.rho if ctx is None else ctx.rho
    dff = 1.0 if ctx is None else ctx.downforce_factor
    if ctx is None:
        cla = 0.0            # P0 path: braking_distance reduces to its old form
    m = st.mass
    for d_ahead, v_target in ((d1, v1), (d2, v2)):
        if d_ahead > 400.0:
            break
        v_target *= grip
        if st.v > v_target and d_ahead <= margin * braking_distance(
                st.v, v_target, m, cda, params, grade,
                cla=cla, rho=rho, downforce_factor=dff):
            return BRAKE, v_lim_here
    if is_corner and st.v >= v_lim_here - 0.5:
        return CORNER, v_lim_here
    return ACCEL, v_lim_here


def step(track, st: CarState, params: VehicleParams, mguk_demand: float,
         dt: float, tow_factor: float = 1.0, grip: float = 1.0,
         physics_context=None) -> dict:
    """Advance one car by one fixed timestep. Semi-implicit Euler.

    ``mguk_demand`` is the policy's requested store-side MGU-K power (W); it is
    clipped by the regulatory taper and by what is actually in the store.
    Returns a per-step record used by the ground-truth trace.
    """
    ctx = physics_context
    pt = track.point(st.s)
    _v_lim_raw, grade, is_corner, _d1, _v1, _d2, _v2 = pt
    aero = "corner" if is_corner else "straight"
    cda = (params.cda_corner if is_corner else params.cda_straight) * tow_factor
    m = st.mass
    regime, v_lim_here = _regime(track, st, params, grip, pt, ctx=ctx)

    v = st.v
    # Relative airspeed, which equals ground speed whenever no wind was
    # projected -- every P0 call, and every synthetic run, because Circuit Sigma
    # has no physical heading to project onto.
    rho = params.rho if ctx is None else ctx.rho
    w_par = 0.0 if ctx is None else ctx.w_parallel_ms
    dff = 1.0 if ctx is None else ctx.downforce_factor
    cla = 0.0 if ctx is None else (params.cla_corner if is_corner
                                   else params.cla_straight)
    v_air = v - w_par
    f_drag = drag_force(v_air, cda, rho)
    f_down = downforce(v_air, cla, rho) * dff
    f_roll = params.crr * m * G
    f_grade = m * G * grade  # small-angle: sin(grade) ~ grade
    resist = f_drag + f_roll + f_grade

    p_ice = 0.0
    p_mguk = 0.0
    harvest = 0.0

    if regime == BRAKE:
        # Tyre-limited once there is a load to be limited by. mu comes from
        # brake_mu(), which inverts the legacy 45 m/s^2 rather than stacking a
        # second aero term on top of it -- see brake_mu for why that matters.
        if ctx is not None and cla > 0.0:
            decel = brake_mu(params, cla, rho) * (m * G + f_down) / m
        else:
            decel = params.brake_decel_max
        a = -(decel + resist / m)
        # regenerative braking: store-side power, capped by the regulation
        harvest = P_MGUK_MAX if m * decel * v > P_MGUK_MAX \
            else m * decel * v
    elif regime == CORNER:
        p_ice = resist * v / params.drivetrain_eff
        p_ice = 0.0 if p_ice < 0.0 else (P_ICE_MAX if p_ice > P_ICE_MAX else p_ice)
        f_trac = p_ice * params.drivetrain_eff / (v if v > 1.0 else 1.0)
        a = (f_trac - resist) / m
    else:  # ACCEL
        p_ice = P_ICE_MAX
        ceiling = mguk_power_limit(v, st.overtake_active)
        avail = st.E / dt if st.E > 0.0 else 0.0
        cap = ceiling if ceiling < avail else avail
        p_mguk = 0.0 if mguk_demand <= 0.0 else (cap if mguk_demand > cap else mguk_demand)
        f_trac = (p_ice + p_mguk) * params.drivetrain_eff / (v if v > 1.0 else 1.0)
        a = (f_trac - resist) / m

    v_new = v + a * dt
    if regime != BRAKE and v_new > v_lim_here:
        v_new = v_lim_here
    if v_new < 1.0:
        v_new = 1.0

    # ------------------------------------------------------- energy bookkeeping
    d_deploy = p_mguk * dt
    room = E_STORE_MAX - (st.E - d_deploy)
    lap_room = recharge_allowance_j(st.overtake_active) - st.harvested_lap
    d_harvest = harvest * dt
    if d_harvest > room:
        d_harvest = room
    if d_harvest > lap_room:
        d_harvest = lap_room
    if d_harvest < 0.0:
        d_harvest = 0.0
    e_new = st.E - d_deploy + d_harvest
    st.E = 0.0 if e_new < 0.0 else (E_STORE_MAX if e_new > E_STORE_MAX else e_new)
    st.harvested_lap += d_harvest
    st.deployed_lap += d_deploy

    ds = v_new * dt
    st.v = v_new
    st.s += ds
    st.s_total += ds
    st.fuel = max(st.fuel - params.fuel_burn_per_lap * ds / track.length, 0.0)
    crossed = False
    if st.s >= track.length:
        st.s -= track.length
        st.lap += 1
        crossed = True

    return {
        "regime": regime,
        "p_ice": p_ice,
        "p_mguk": p_mguk,
        "harvest": d_harvest / dt,
        "a": a,
        "aero": aero,
        "crossed_line": crossed,
    }
