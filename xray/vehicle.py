"""Longitudinal dynamics and the energy store.

This is the ground-truth physics. `estimator.py` must never import it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import (
    E_HARVEST_LAP,
    E_STORE_MAX,
    G,
    P_ICE_MAX,
    P_MGUK_MAX,
    p_mguk_ceiling,
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

    @classmethod
    def from_config(cls, cfg: dict) -> "VehicleParams":
        return cls(**cfg["vehicle"])

    def cda(self, aero_mode: str) -> float:
        return self.cda_corner if aero_mode == "corner" else self.cda_straight


@dataclass
class CarState:
    s: float = 0.0            # m, distance along the current lap
    s_total: float = 0.0      # m, cumulative distance (defines race order)
    v: float = 0.0            # m/s
    E: float = 0.0            # J, energy store
    fuel: float = 70.0        # kg
    harvested_lap: float = 0.0
    deployed_lap: float = 0.0
    mom_credit: float = 0.0   # J, unspent Manual Override allocation
    lap: int = 0

    @property
    def mass(self) -> float:
        return _MASS_CAR[0] + self.fuel


# module-level so CarState.mass stays cheap; set by Simulator on construction
_MASS_CAR = [768.0]


def set_car_mass(mass_car: float) -> None:
    _MASS_CAR[0] = mass_car


def drag_force(v: float, cda: float, rho: float) -> float:
    return 0.5 * rho * cda * v * v


def braking_distance(v_in: float, v_target: float, m: float, cda: float,
                     params: VehicleParams, grade: float = 0.0) -> float:
    """Distance needed to slow from v_in to v_target under brakes + drag.

    Closed form of  m v dv / (F_brake + F_roll + F_grade + k v^2)  which is
    exact for constant CdA, and cheap enough to call every timestep.
    """
    if v_in <= v_target:
        return 0.0
    k = 0.5 * params.rho * cda
    a0 = m * params.brake_decel_max + params.crr * m * G + m * G * np.sin(grade)
    if a0 <= 0:
        return float("inf")
    return float(m / (2 * k) * np.log((a0 + k * v_in * v_in) / (a0 + k * v_target * v_target)))


def _regime(track, st: CarState, params: VehicleParams, grip: float,
            margin: float = 1.02) -> tuple[str, float]:
    """Pick the driving regime for this step. Returns (regime, v_limit_here)."""
    v_lim_here = track.v_limit(st.s)
    if track.is_corner(st.s):
        v_lim_here *= grip
    if st.v > v_lim_here:
        return BRAKE, v_lim_here
    cda = params.cda(track.aero_mode(st.s))
    grade = track.grade(st.s)
    for d_ahead, v_target in track.next_corner(st.s, horizon=300.0):
        v_target *= grip
        if st.v > v_target and d_ahead <= margin * braking_distance(
                st.v, v_target, st.mass, cda, params, grade):
            return BRAKE, v_lim_here
    if track.is_corner(st.s) and st.v >= v_lim_here - 0.5:
        return CORNER, v_lim_here
    return ACCEL, v_lim_here


def step(track, st: CarState, params: VehicleParams, mguk_demand: float,
         dt: float, tow_factor: float = 1.0, grip: float = 1.0) -> dict:
    """Advance one car by one fixed timestep. Semi-implicit Euler.

    ``mguk_demand`` is the policy's requested store-side MGU-K power (W); it is
    clipped by the regulatory taper and by what is actually in the store.
    Returns a per-step record used by the ground-truth trace.
    """
    aero = track.aero_mode(st.s)
    cda = params.cda(aero) * tow_factor
    grade = track.grade(st.s)
    m = st.mass
    regime, v_lim_here = _regime(track, st, params, grip)

    f_drag = drag_force(st.v, cda, params.rho)
    f_roll = params.crr * m * G
    f_grade = m * G * np.sin(grade)
    resist = f_drag + f_roll + f_grade

    p_ice = 0.0
    p_mguk = 0.0
    harvest = 0.0

    if regime == BRAKE:
        a = -(params.brake_decel_max + resist / m)
        # regenerative braking: store-side power, capped by the regulation and
        # by what will actually fit in the store this lap
        harvest = min(P_MGUK_MAX, m * params.brake_decel_max * st.v)
    elif regime == CORNER:
        a = 0.0
        p_ice = float(np.clip(resist * st.v / params.drivetrain_eff, 0.0, P_ICE_MAX))
        f_trac = p_ice * params.drivetrain_eff / max(st.v, 1.0)
        a = (f_trac - resist) / m
    else:  # ACCEL
        p_ice = P_ICE_MAX
        avail = max(st.E, 0.0) / dt  # cannot draw more than the store holds
        p_mguk = float(np.clip(mguk_demand, 0.0, min(p_mguk_ceiling(st.v), avail)))
        f_trac = (p_ice + p_mguk) * params.drivetrain_eff / max(st.v, 1.0)
        a = (f_trac - resist) / m

    v_new = st.v + a * dt
    if regime != BRAKE and v_new > v_lim_here:
        v_new = v_lim_here
    v_new = max(v_new, 1.0)

    # ------------------------------------------------------- energy bookkeeping
    d_deploy = p_mguk * dt
    room = max(E_STORE_MAX - (st.E - d_deploy), 0.0)
    lap_room = max(E_HARVEST_LAP - st.harvested_lap, 0.0)
    d_harvest = min(harvest * dt, room, lap_room)
    st.E = float(np.clip(st.E - d_deploy + d_harvest, 0.0, E_STORE_MAX))
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
