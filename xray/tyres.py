"""Reduced-order tyre state: temperature, wear, and the grip they imply.

Not a Pacejka model and not a thermal model. It is a bounded, interpretable
envelope whose every coefficient is configuration, so that a number nobody has
measured cannot hide inside an equation.

The one semantic rule that matters more than any equation here:

    FastF1's TyreLife is AGE IN LAPS. It is not wear.

A ten-lap-old hard on a cold track and a ten-lap-old soft after a qualifying
simulation have the same TyreLife and nothing else in common. `tyre_life` is
carried through untouched as the observed age; `wear_fraction` is modelled
separately and is never assigned from it. `test_tyres.py` asserts they cannot be
the same number.

Grip is the product of three bounded factors:

    mu_eff = mu_base * f_temp(T) * f_wear(wear) * f_wet(wetness, compound)

each of which is 1.0 in its own best case, so `mu_base` is the best the compound
ever does and every factor can only take grip away.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .vehicle import G

# Floor on the thermal factor. A tyre far outside its window is bad, not
# frictionless, and an unbounded factor makes the DP chase nonsense states.
MIN_THERMAL_GRIP = 0.70
# Ceiling on how much faster a badly-thermally-managed tyre wears.
MAX_THERMAL_WEAR_MULTIPLIER = 3.0


@dataclass(frozen=True)
class TyreModelParams:
    """Per-compound coefficients. ASSUMED -- see config/default.yaml."""
    compound: str
    mu_long_base: float
    mu_lat_base: float
    optimal_temp_c: float
    temp_window_c: float
    thermal_tau_s: float
    heating_gain: float
    base_wear_per_m: float
    utilisation_wear_gain: float
    thermal_wear_gain: float
    max_wear_grip_loss: float
    # Wet behaviour is compound-specific or it is nothing. A slick on a soaked
    # track is not "a bit slower", and an intermediate on a dry one overheats.
    wet_grip_loss: float = 0.0      # fraction of grip lost at wetness = 1
    dry_grip_loss: float = 0.0      # fraction lost at wetness = 0 (wet tyres)
    is_wet_compound: bool = False


@dataclass(frozen=True)
class TyreState:
    """What we believe about the rubber right now, and where it came from."""
    compound: str
    tyre_life: float                # OBSERVED age in laps. NOT wear.
    fresh_tyre: bool | None
    estimated_temp_c: float         # MODELLED; no public temperature channel
    wear_fraction: float            # MODELLED 0..1. NOT tyre_life.
    grip_scale: float               # combined multiplier on mu_base
    source: str

    @property
    def is_modelled(self) -> bool:
        return "modelled" in self.source


# ------------------------------------------------------------------ factors
def thermal_grip_factor(temp_c: float, p: TyreModelParams) -> float:
    """1.0 in the window, falling away on BOTH sides, floored.

    Deliberately not monotone in temperature. A single linear rule -- hotter is
    always worse, or always better -- is wrong in opposite directions either
    side of the operating window, and a tyre model that gets the sign of "cold"
    wrong will happily recommend attacking on an out-lap.
    """
    d = (float(temp_c) - p.optimal_temp_c) / max(p.temp_window_c, 1e-6)
    return float(np.clip(1.0 - (1.0 - MIN_THERMAL_GRIP) * d * d,
                         MIN_THERMAL_GRIP, 1.0))


def wear_grip_factor(wear_fraction: float, p: TyreModelParams) -> float:
    """Linear in wear, bounded by the compound's own worst case."""
    w = float(np.clip(wear_fraction, 0.0, 1.0))
    return float(1.0 - p.max_wear_grip_loss * w)


def wet_grip_factor(wetness: float, p: TyreModelParams) -> float:
    """Compound-specific, and neutral when the compound says nothing.

    A slick loses grip as the track wets; a wet-weather compound loses it as the
    track dries. With both coefficients at zero this returns 1.0 and the model
    makes no claim at all, which is the right behaviour for a compound nobody
    has parameterised.
    """
    w = float(np.clip(wetness, 0.0, 1.0))
    return float(np.clip(1.0 - p.wet_grip_loss * w - p.dry_grip_loss * (1.0 - w),
                         0.05, 1.0))


def grip_scale(temp_c: float, wear_fraction: float, wetness: float,
               p: TyreModelParams) -> float:
    return (thermal_grip_factor(temp_c, p) * wear_grip_factor(wear_fraction, p)
            * wet_grip_factor(wetness, p))


def mu_effective(temp_c: float, wear_fraction: float, wetness: float,
                 p: TyreModelParams) -> tuple[float, float]:
    """(mu_long, mu_lat) after temperature, wear and wetness."""
    k = grip_scale(temp_c, wear_fraction, wetness, p)
    return p.mu_long_base * k, p.mu_lat_base * k


# ------------------------------------------------------------------ envelope
def force_limits(mass_kg: float, f_down_n: float, temp_c: float,
                 wear_fraction: float, wetness: float,
                 p: TyreModelParams) -> tuple[float, float, float]:
    """(N, F_long_max, F_lat_max) from the aerodynamic normal load.

    This is the join between P1.2 and P1.5, and the reason downforce was worth
    adding: available force is no longer independent of speed.
    """
    n = float(mass_kg) * G + max(float(f_down_n), 0.0)
    mu_long, mu_lat = mu_effective(temp_c, wear_fraction, wetness, p)
    return n, mu_long * n, mu_lat * n


def utilisation(f_long: float, f_lat: float, f_long_max: float,
                f_lat_max: float) -> float:
    """Friction-ellipse usage in [0, inf), clipped for the wear model.

        U = sqrt((F_long/F_long_max)^2 + (F_lat/F_lat_max)^2)

    U <= 1 is inside the ellipse. Values above 1 are not forced back to 1 here
    because "how far past the limit" is exactly what the wear model wants; the
    clip belongs at the point of use, not in the definition.
    """
    a = float(f_long) / max(float(f_long_max), 1e-9)
    b = float(f_lat) / max(float(f_lat_max), 1e-9)
    return float(np.hypot(a, b))


# ------------------------------------------------------------- state update
def thermal_wear_multiplier(temp_c: float, p: TyreModelParams) -> float:
    """How much faster the tyre wears away from its window.

    Rises on both sides, like the grip factor and for the same reason: a cold
    tyre grains and a hot one blisters. Bounded, because an unbounded wear rate
    lets one bad sample retire a set of tyres.
    """
    d = (float(temp_c) - p.optimal_temp_c) / max(p.temp_window_c, 1e-6)
    return float(np.clip(1.0 + p.thermal_wear_gain * d * d,
                         1.0, MAX_THERMAL_WEAR_MULTIPLIER))


def advance(state: TyreState, p: TyreModelParams, dt_s: float, ds_m: float,
            track_temp_c: float, util: float, wetness: float = 0.0) -> TyreState:
    """One step of the thermal and wear state.

        target = track_temp + heating_gain * U
        T'     = T + (target - T) * (1 - exp(-dt/tau))
        dwear  = base_wear_per_m * ds * thermal_mult(T) * (1 + gain * U^2)

    Both integrations are bounded and neither can run backwards: wear is
    monotone non-decreasing within a stint, and only a tyre CHANGE resets it
    (see `fit_fresh`), never the passage of time.
    """
    u = float(np.clip(util, 0.0, 2.0))
    target = float(track_temp_c) + p.heating_gain * u
    alpha = 1.0 - float(np.exp(-max(dt_s, 0.0) / max(p.thermal_tau_s, 1e-6)))
    temp = float(state.estimated_temp_c + (target - state.estimated_temp_c) * alpha)

    d_wear = (p.base_wear_per_m * max(ds_m, 0.0)
              * thermal_wear_multiplier(temp, p)
              * (1.0 + p.utilisation_wear_gain * u * u))
    wear = float(np.clip(state.wear_fraction + d_wear, 0.0, 1.0))
    return replace(state, estimated_temp_c=temp, wear_fraction=wear,
                   grip_scale=grip_scale(temp, wear, wetness, p),
                   source="modelled_thermal_wear")


def fresh(compound: str, p: TyreModelParams, track_temp_c: float,
          wetness: float = 0.0, tyre_life: float = 0.0,
          source: str = "modelled_fresh") -> TyreState:
    """A new set. Wear is zero because the tyre is new, not because age is zero.

    `tyre_life` is accepted separately and may be non-zero: a set fitted under a
    safety car has age without meaningful wear, and conflating the two is the
    error this module exists to prevent.
    """
    temp = float(track_temp_c)
    return TyreState(compound=compound, tyre_life=float(tyre_life),
                     fresh_tyre=True, estimated_temp_c=temp, wear_fraction=0.0,
                     grip_scale=grip_scale(temp, 0.0, wetness, p), source=source)


def unknown(compound: str | None, tyre_life: float | None,
            fresh_tyre: bool | None = None,
            track_temp_c: float = 30.0) -> TyreState:
    """No model has run. Neutral grip, and a source that says so.

    Returned rather than a guess when a payload has age metadata but nothing has
    integrated a wear state. grip_scale is 1.0 so nothing downstream is silently
    penalised for an absence of information.
    """
    return TyreState(compound=str(compound) if compound else "UNKNOWN",
                     tyre_life=float(tyre_life) if tyre_life is not None else 0.0,
                     fresh_tyre=fresh_tyre, estimated_temp_c=float(track_temp_c),
                     wear_fraction=0.0, grip_scale=1.0, source="unknown")


# ----------------------------------------------------------------- config
def params_from_config(cfg: dict, compound: str | None) -> TyreModelParams:
    """Look up a compound, falling back to a neutral set for unknown strings.

    Unknown compound names are preserved rather than mapped onto a guess: a
    2026 compound this config has never heard of should behave neutrally and say
    `UNKNOWN`, not silently become a MEDIUM.
    """
    table = (cfg.get("tyres") or {}).get("compounds") or {}
    key = str(compound).upper() if compound else ""
    if key in table:
        return TyreModelParams(compound=key, **table[key])
    neutral = table.get("_NEUTRAL")
    if neutral:
        return TyreModelParams(compound=key or "UNKNOWN", **neutral)
    return TyreModelParams(
        compound=key or "UNKNOWN", mu_long_base=1.4, mu_lat_base=1.5,
        optimal_temp_c=95.0, temp_window_c=30.0, thermal_tau_s=45.0,
        heating_gain=25.0, base_wear_per_m=2.0e-6, utilisation_wear_gain=1.0,
        thermal_wear_gain=0.8, max_wear_grip_loss=0.15)


# --------------------------------------------------- wear over a whole lap
# Representative friction-ellipse usage. ASSUMED, and named rather than buried:
# a lap is not run at one utilisation, so these are the effective averages that
# reproduce a normal lap and an attacking lap. They are the only tunable in the
# DP's wear transition, and they live here with the rest of the tyre model
# instead of appearing as a constant inside decision.py.
ASSUMED_UTIL_NORMAL = 0.60
ASSUMED_UTIL_ATTACK = 0.95


def wear_per_lap(p: TyreModelParams, lap_distance_m: float, track_temp_c: float,
                 util: float, wetness: float = 0.0, n_steps: int = 40,
                 lap_time_s: float = 90.0, start_wear: float = 0.0) -> float:
    """Wear added by one lap at a given utilisation, from `advance`.

    Integrated through the same state update the simulator uses rather than
    given a closed form, so the DP's transition and the simulator's cannot drift
    apart. Returns the INCREMENT, not the new state.
    """
    st = TyreState(compound=p.compound, tyre_life=0.0, fresh_tyre=None,
                   estimated_temp_c=float(track_temp_c),
                   wear_fraction=float(np.clip(start_wear, 0.0, 1.0)),
                   grip_scale=1.0, source="modelled_thermal_wear")
    ds = float(lap_distance_m) / max(n_steps, 1)
    dt = float(lap_time_s) / max(n_steps, 1)
    for _ in range(n_steps):
        st = advance(st, p, dt_s=dt, ds_m=ds, track_temp_c=track_temp_c,
                     util=util, wetness=wetness)
    return float(st.wear_fraction - np.clip(start_wear, 0.0, 1.0))


def attack_extra_wear(p: TyreModelParams, lap_distance_m: float,
                      track_temp_c: float, wetness: float = 0.0,
                      start_wear: float = 0.0) -> float:
    """How much MORE a lap costs when it is an attacking lap.

    The difference between two integrations of the same model at two
    utilisations -- not a penalty constant. If the tyre model changes, this
    changes with it.
    """
    hard = wear_per_lap(p, lap_distance_m, track_temp_c, ASSUMED_UTIL_ATTACK,
                        wetness, start_wear=start_wear)
    normal = wear_per_lap(p, lap_distance_m, track_temp_c, ASSUMED_UTIL_NORMAL,
                          wetness, start_wear=start_wear)
    return float(max(hard - normal, 0.0))
