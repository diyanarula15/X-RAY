"""Regulation constants and public knowledge.

Nothing in this module is a secret. Every value here is either written in the
2026 power-unit regulations or published in the timing feed, which is why the
estimator is allowed to import it.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- regulation
P_ICE_MAX = 400_000  # W
P_MGUK_MAX = 350_000  # W
TAPER_V_START = 290 / 3.6  # m/s, full deployment up to here
TAPER_V_END = 355 / 3.6  # m/s, zero deployment at and above here
E_STORE_MAX = 4.0e6  # J, usable store
E_HARVEST_LAP = 7.0e6  # J, per-lap harvest cap
MOM_BONUS = 0.5e6  # J, Manual Override allocation
MOM_GAP_S = 1.0  # s, eligibility threshold at detection point

G = 9.80665  # m/s^2

# ------------------------------------------------------------------- public
# Published or trivially inferable from the broadcast: minimum car mass, the
# declared start fuel and the per-lap burn. The estimator may use these.
MASS_CAR_MIN = 768.0  # kg
FUEL_START_KG = 70.0  # kg
FUEL_BURN_PER_LAP_KG = 1.4  # kg


def p_mguk_ceiling(v):
    """Regulatory MGU-K deployment ceiling as a function of speed. Returns W.

    Linear taper: full power up to TAPER_V_START, zero at and above
    TAPER_V_END. Works on scalars and arrays.
    """
    v = np.asarray(v, dtype=float)
    frac = (TAPER_V_END - v) / (TAPER_V_END - TAPER_V_START)
    out = P_MGUK_MAX * np.clip(frac, 0.0, 1.0)
    return float(out) if out.ndim == 0 else out


def fuel_estimate(lap, fuel_start=FUEL_START_KG, burn=FUEL_BURN_PER_LAP_KG):
    """Public-knowledge fuel mass at a given (0-based, fractional) lap index."""
    return np.maximum(fuel_start - burn * np.asarray(lap, dtype=float), 0.0)
