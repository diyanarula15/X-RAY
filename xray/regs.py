"""2026 power-unit regulation, as bounds a car cannot violate.

Everything here is public: it is the rulebook plus the FIA event documents. The
estimator is allowed to import it for the same reason it may import
`constants` -- knowing the rules is not knowing the car.

Two regulation variants ship in the same season, so every bound is a function
of the race date. Getting the variant wrong does not degrade the estimate
gracefully: the deployment cap moves by 100 kW, which is 30% of the quantity
being estimated, so a mislabelled race produces a confidently wrong answer
rather than a noisy one. `RegSet.describe()` exists so the UI can state which
variant it used.

Provenance. The numbers below come from the project specification, not from a
machine-readable FIA source -- there isn't one. Anything the spec did not pin
exactly is named ASSUMED_* and carries its derivation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .constants import E_STORE_MAX, P_ICE_MAX, P_MGUK_MAX

# ---------------------------------------------------------------- fuel flow
# The regulation limits fuel *energy* flow on an RPM schedule, EF(n) =
# 0.27n + 165 MJ/h, rather than limiting ICE power directly. At the 10,500 rpm
# limit that is 3,000 MJ/h = 833 kW of fuel, and the 400 kW ICE cap implies a
# thermal efficiency of exactly 400/833 = 0.48. So the flat cap and the
# schedule agree at max revs and the schedule binds below it -- which is the
# point: a car at 7,000 rpm is allowed 2,055 MJ/h = 571 kW of fuel = 274 kW at
# the crank, not 400.
EF_SLOPE_MJ_PER_H_PER_RPM = 0.27
EF_INTERCEPT_MJ_PER_H = 165.0
N_RPM_MAX = 10_500.0
ASSUMED_ICE_THERMAL_EFF = P_ICE_MAX / (
    (EF_SLOPE_MJ_PER_H_PER_RPM * N_RPM_MAX + EF_INTERCEPT_MJ_PER_H) * 1e6 / 3600.0)

# ------------------------------------------------------------------- taper
TAPER_V_FULL_KMH = 290.0     # full deployment permitted up to here
TAPER_V_ZERO_KMH = 355.0     # zero at and above here
MO_TAPER_V_FULL_KMH = 337.0  # Manual Override holds full power far higher,
MO_TAPER_V_ZERO_KMH = 350.0  # which is the whole point of the override

# --------------------------------------------------------- regulation dates
# The mid-season change. 3 May 2026 is the Miami Grand Prix race date, and the
# package applied from that weekend per the F1 statement -- so this is the right
# date rather than a placeholder. Still tagged ASSUMED because it has not been
# checked against the FIA event document, which is the source that would settle
# it; the check belongs with the first real-data run, since that is when the
# document gets fetched anyway.
#
# Getting it wrong is not a graceful failure: the deployment cap moves 100 kW,
# which is 30% of the quantity being estimated, and the local drag floor goes
# from usable to dead. A mislabelled race produces a confidently wrong answer.
ASSUMED_MIAMI_2026 = date(2026, 5, 3)

# Pedal position below which the ICE is treated as making nothing. The feed
# publishes throttle as a percentage, and a real trace idles at a few percent
# rather than at zero.
COAST_THROTTLE_FRAC = 0.08

# Pedal position at or above which the ICE is treated as making its permitted
# maximum, less delta below.
FULL_THROTTLE_FRAC = 0.95

# ---------------------------------------------------- the ICE floor (delta)
# A speed trace cannot bound drag area from below. With P_ice >= 0, "engine at
# idle, motor harvesting" explains every straight and CdA = 0 is feasible. The
# lower bound has to come from a claim about ICE *output*, and that claim is not
# in the trace -- it is an assumption, tagged as one.
#
# At full throttle a competitive team is not leaving a tenth of the permitted
# fuel flow unused, so P_ice >= (1 - delta) * P_ice_max.
#
# What this buys depends on the regulation variant, and the difference is worth
# stating: pre-Miami the harvest cap is 250 kW, so 400*(1-delta) - 250 = 110 kW
# must go somewhere on a steady full-throttle straight, which pins CdA above
# about 0.44 at 300 km/h. Post-Miami super-clipping raises the cap to 350 kW and
# the same arithmetic leaves 10 kW. The April rule change made drag area
# lower-unidentifiable from power bounds alone.
ASSUMED_ICE_FLOOR_DELTA = 0.10

# ------------------------------------------------- fuel closure (the global bound)
# Independent of any harvest rule, and that is the point. Over a race the ICE
# does a known amount of work, and it has nowhere to go but drag, rolling
# resistance, the friction brakes and the store -- which cannot absorb more than
# 4 MJ net. No deployment strategy evades it.
ASSUMED_FUEL_LHV_MJ_PER_KG = 39.0   # pump-spec F1 fuel, lower heating value
ASSUMED_FUEL_MASS_FRAC_SIGMA = 0.03  # how well the burned mass is known, as a
                                     # fraction: teams know a stint's fuel far
                                     # better than a race's absolute load


@dataclass(frozen=True)
class RegSet:
    """The bounds in force for one race."""
    variant: str
    p_dep_max_zone: float      # W, cap inside a designated deployment zone
    p_dep_max_elsewhere: float  # W, cap outside one
    p_harv_max: float          # W, recovery cap -- applies at ANY throttle
    e_harvest_lap: float       # J, per-lap harvest allowance
    mom_bonus: float           # J, Manual Override allocation
    mom_gap_s: float           # s, eligibility gap at the detection point

    def describe(self) -> str:
        return (f"{self.variant}: deploy {self.p_dep_max_zone/1e3:.0f}/"
                f"{self.p_dep_max_elsewhere/1e3:.0f} kW (zone/elsewhere), "
                f"harvest {self.p_harv_max/1e3:.0f} kW, "
                f"harvest cap {self.e_harvest_lap/1e6:.1f} MJ/lap")


PRE_MIAMI = RegSet(
    variant="pre-Miami-2026",
    p_dep_max_zone=P_MGUK_MAX, p_dep_max_elsewhere=P_MGUK_MAX,
    p_harv_max=250_000.0, e_harvest_lap=7.0e6,
    mom_bonus=0.5e6, mom_gap_s=1.0)

POST_MIAMI = RegSet(
    variant="post-Miami-2026",
    # Deployment is cut outside the designated zones, which is what makes the
    # zone map part of the physics rather than decoration.
    p_dep_max_zone=P_MGUK_MAX, p_dep_max_elsewhere=250_000.0,
    # Recovery is raised and permitted at any throttle -- "super-clipping".
    # This is the change that breaks a braking-only harvest model: a car can
    # now recharge flat out at the end of a straight.
    p_harv_max=P_MGUK_MAX, e_harvest_lap=7.0e6,
    mom_bonus=0.5e6, mom_gap_s=1.0)


def regs_for(race_date: date | str | None,
             changeover: date = ASSUMED_MIAMI_2026) -> RegSet:
    """Which variant was in force. Defaults to pre-Miami when the date is
    unknown, because that variant's bounds are the wider pair and a wider
    bound cannot exclude the truth."""
    if race_date is None:
        return PRE_MIAMI
    if isinstance(race_date, str):
        race_date = date.fromisoformat(race_date[:10])
    return POST_MIAMI if race_date >= changeover else PRE_MIAMI


def _ramp(v, lo_kmh: float, hi_kmh: float):
    lo, hi = lo_kmh / 3.6, hi_kmh / 3.6
    return np.clip((hi - v) / (hi - lo), 0.0, 1.0)


def taper(v_ms, manual_override=False):
    """Fraction of the deployment cap the regulation still allows at speed v.

    `manual_override` is per sample, not per race: eligibility is decided lap by
    lap at the detection point, so a car can be on the override ramp for one lap
    and the normal ramp for the next.
    """
    v = np.asarray(v_ms, dtype=float)
    normal = _ramp(v, TAPER_V_FULL_KMH, TAPER_V_ZERO_KMH)
    mo = _ramp(v, MO_TAPER_V_FULL_KMH, MO_TAPER_V_ZERO_KMH)
    out = np.where(np.asarray(manual_override, dtype=bool), mo, normal)
    return float(out) if out.ndim == 0 else out


def p_dep_max(v_ms, in_zone, regs: RegSet, manual_override=False):
    """Upper bound on store-side deployment, per sample.

    Zone map times taper. Both factors are public; neither depends on the car.
    """
    in_zone = np.asarray(in_zone, dtype=bool)
    cap = np.where(in_zone, regs.p_dep_max_zone, regs.p_dep_max_elsewhere)
    return cap * taper(v_ms, manual_override)


def p_ice_max(rpm, throttle, regs: RegSet | None = None):
    """Upper bound on ICE output from the fuel-flow schedule and the pedal.

    Throttle is a *pedal position*, not a power fraction. Treating it as one is
    what produced CdA lower bounds above 13 m^2 in an earlier real-data
    version: a car at 30 m/s is traction-limited, its pedal is on the floor,
    and its engine is nowhere near 400 kW. Used only as an upper bound here, so
    the pedal can only ever relax the constraint.
    """
    n = np.clip(np.asarray(rpm, dtype=float), 0.0, N_RPM_MAX)
    thr = np.clip(np.asarray(throttle, dtype=float), 0.0, 1.0)
    fuel_w = (EF_SLOPE_MJ_PER_H_PER_RPM * n + EF_INTERCEPT_MJ_PER_H) * 1e6 / 3600.0
    return np.minimum(fuel_w * ASSUMED_ICE_THERMAL_EFF, float(P_ICE_MAX)) * thr


def p_ice_min(rpm, throttle, regs: RegSet | None = None,
              delta: float = ASSUMED_ICE_FLOOR_DELTA):
    """Lower bound on ICE output. ASSUMED -- see ASSUMED_ICE_FLOOR_DELTA.

    Zero unless the pedal is on the floor, because below full throttle the
    driver's intent is unknown and no floor is defensible.
    """
    thr = np.asarray(throttle, dtype=float)
    return np.where(thr >= FULL_THROTTLE_FRAC,
                    (1.0 - delta) * p_ice_max(rpm, 1.0, regs), 0.0)


def ice_work_from_fuel(fuel_kg, lhv_mj_per_kg: float = ASSUMED_FUEL_LHV_MJ_PER_KG,
                       thermal_eff: float = None) -> float:
    """Mechanical work the ICE must have done to burn this much fuel, J.

    thermal_eff defaults to the value the fuel-flow schedule implies at maximum
    revs (0.48), so the local and global bounds rest on the same number.
    """
    eff = ASSUMED_ICE_THERMAL_EFF if thermal_eff is None else thermal_eff
    return float(np.asarray(fuel_kg, float) * lhv_mj_per_kg * 1e6 * eff)


def p_k_bounds(v_ms, in_zone, rpm, throttle, regs: RegSet,
               manual_override=False):
    """Two-sided bound on store-side MGU-K power. Positive = deploying.

    The lower bound is the change that matters most versus the Stage 1 model,
    which allowed recovery only under braking. Post-Miami a car may recover at
    any throttle, so every sample now brackets P_K from both sides and no
    sample has to be discarded for being ambiguous.
    """
    hi = p_dep_max(v_ms, in_zone, regs, manual_override)
    lo = np.full(np.shape(hi), -float(regs.p_harv_max))
    return lo, hi


def store_bounds() -> tuple[float, float]:
    return 0.0, float(E_STORE_MAX)
