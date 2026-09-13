"""Regulation constants and public knowledge.

Nothing in this module is a secret. Every value here is either written in the
2026 power-unit regulations or published in the timing feed, which is why the
estimator is allowed to import it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PowerCurve:
    full_power_until_mps: float
    zero_at_mps: float
    segments_kmh_kw: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class PowerUnitRegulation:
    max_ice_power_w: float
    max_mguk_power_w: float
    max_store_energy_j: float
    max_recharge_without_overtake_j: float
    max_recharge_with_overtake_j: float
    overtake_allocation_j: float
    normal_curve: PowerCurve
    overtake_curve: PowerCurve


POWER_UNIT_2026 = PowerUnitRegulation(
    max_ice_power_w=400_000.0,
    max_mguk_power_w=350_000.0,
    max_store_energy_j=4.0e6,
    max_recharge_without_overtake_j=7.0e6,
    max_recharge_with_overtake_j=7.5e6,
    overtake_allocation_j=0.5e6,
    normal_curve=PowerCurve(
        full_power_until_mps=290 / 3.6,
        zero_at_mps=345 / 3.6,
        segments_kmh_kw=((290.0, 340.0, 1800.0, -5.0),
                         (340.0, 345.0, 6900.0, -20.0)),
    ),
    overtake_curve=PowerCurve(
        full_power_until_mps=337.5 / 3.6,
        zero_at_mps=355 / 3.6,
        segments_kmh_kw=((337.5, 355.0, 7100.0, -20.0),),
    ),
)

# ---------------------------------------------------------------- regulation
P_ICE_MAX = POWER_UNIT_2026.max_ice_power_w
P_MGUK_MAX = POWER_UNIT_2026.max_mguk_power_w
TAPER_V_START = POWER_UNIT_2026.normal_curve.full_power_until_mps
NORMAL_TAPER_V_KMH = 340.0
NORMAL_TAPER_V_END = POWER_UNIT_2026.normal_curve.zero_at_mps
OVERTAKE_FULL_POWER_V = POWER_UNIT_2026.overtake_curve.full_power_until_mps
OVERTAKE_TAPER_V_END = POWER_UNIT_2026.overtake_curve.zero_at_mps
TAPER_V_END = NORMAL_TAPER_V_END
E_STORE_MAX = POWER_UNIT_2026.max_store_energy_j  # J, usable store
E_HARVEST_LAP = POWER_UNIT_2026.max_recharge_without_overtake_j  # J, per-lap harvest cap
MOM_DEPLOYMENT_ALLOWANCE_J = POWER_UNIT_2026.overtake_allocation_j  # J, legal allocation, not SOC
MOM_BONUS = MOM_DEPLOYMENT_ALLOWANCE_J  # Backward-compatible alias; do not add to store energy.
MOM_GAP_S = 1.0  # s, eligibility threshold at detection point

# The allocation is exactly the gap between the two recharge ceilings. Asserted
# rather than trusted: the three numbers are published separately and a typo in
# any one of them would silently hand a car free energy.
assert abs((POWER_UNIT_2026.max_recharge_with_overtake_j
            - POWER_UNIT_2026.max_recharge_without_overtake_j)
           - POWER_UNIT_2026.overtake_allocation_j) < 1.0


def recharge_allowance_j(overtake_active: bool = False) -> float:
    """Per-lap legal recharge ceiling, J.

    This is the ONLY thing the 0.5 MJ Manual Override allocation does to the
    energy budget: it raises how much the car may legally *recover* on that lap,
    from 7.0 to 7.5 MJ. It is not stored energy. The previous model ran
    `st.E += 0.5 MJ` at the lap boundary, which materialised half a megajoule
    out of a rulebook and closed the per-lap balance only because the test added
    the same term to both sides.
    """
    return float(POWER_UNIT_2026.max_recharge_with_overtake_j if overtake_active
                 else POWER_UNIT_2026.max_recharge_without_overtake_j)


def mguk_power_limit(v, overtake_active: bool = False):
    """Permitted MGU-K deployment power at speed v, W. Single dispatch point."""
    return (mguk_power_limit_overtake(v) if overtake_active
            else mguk_power_limit_normal(v))

G = 9.80665  # m/s^2

# ------------------------------------------------------------------- public
# Published or trivially inferable from the broadcast: minimum car mass, the
# declared start fuel and the per-lap burn. The estimator may use these.
MASS_CAR_MIN = 768.0  # kg
FUEL_START_KG = 70.0  # kg
FUEL_BURN_PER_LAP_KG = 1.4  # kg


def _as_power(out):
    return float(out) if np.ndim(out) == 0 else out


def _curve_limit(v, curve: PowerCurve):
    # `np.ndim(v) == 0`, not `type(v) is float`. `vehicle.step` integrates with
    # numpy scalars, and `type(np.float64(90.0)) is float` is False -- so every
    # scalar call fell through to the vectorised branch and allocated
    # asarray + full_like + three where + clip to compute one number. In one
    # `evaluate_decision_trace_from_payload` that is 6.0M calls and 69 s of a
    # 186 s trace, the single largest cost in the decision endpoint. Same
    # arithmetic either way; `test_physics` pins the curve itself.
    if np.ndim(v) == 0:
        v = float(v)
        if v <= curve.full_power_until_mps:
            return float(P_MGUK_MAX)
        if v >= curve.zero_at_mps:
            return 0.0
        v_kmh = v * 3.6
        for lo, hi, intercept_kw, slope_kw_per_kmh in curve.segments_kmh_kw:
            if lo <= v_kmh <= hi:
                return float(np.clip((intercept_kw + slope_kw_per_kmh * v_kmh) * 1000.0,
                                     0.0, P_MGUK_MAX))
        return 0.0

    v = np.asarray(v, dtype=float)
    v_kmh = v * 3.6
    out_kw = np.full_like(v, P_MGUK_MAX / 1000.0, dtype=float)
    for lo, hi, intercept_kw, slope_kw_per_kmh in curve.segments_kmh_kw:
        m = (v_kmh >= lo) & (v_kmh <= hi)
        out_kw = np.where(m, intercept_kw + slope_kw_per_kmh * v_kmh, out_kw)
    out = np.clip(out_kw * 1000.0, 0.0, P_MGUK_MAX)
    out = np.where(v <= curve.full_power_until_mps, P_MGUK_MAX, out)
    out = np.where(v >= curve.zero_at_mps, 0.0, out)
    return _as_power(out)


def mguk_power_limit_normal(v):
    """2026 normal MGU-K deployment ceiling as a function of speed. Returns W.

    Speeds are SI internally. The published piecewise curve is expressed in
    km/h and kW: capped at 350 kW through 290 km/h, then ``1800 - 5v`` through
    340 km/h, then ``6900 - 20v`` until normal deployment reaches zero at
    345 km/h.
    """
    return _curve_limit(v, POWER_UNIT_2026.normal_curve)


def mguk_power_limit_overtake(v):
    """2026 Overtake-active MGU-K deployment ceiling as a function of speed."""
    return _curve_limit(v, POWER_UNIT_2026.overtake_curve)


def p_mguk_ceiling(v):
    """Backward-compatible alias for the normal 2026 MGU-K power limit."""
    return mguk_power_limit_normal(v)


def fuel_estimate(lap, fuel_start=FUEL_START_KG, burn=FUEL_BURN_PER_LAP_KG):
    """Public-knowledge fuel mass at a given (0-based, fractional) lap index."""
    return np.maximum(fuel_start - burn * np.asarray(lap, dtype=float), 0.0)
