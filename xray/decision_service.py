"""Canonical overtake-decision service.

The API and tests call this module instead of recreating decision math. The
core solver still lives in :mod:`xray.decision`; this layer handles causal
state extraction from analysed telemetry payloads and packages a defensible,
unit-labelled result for the web.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .constants import E_HARVEST_LAP, E_STORE_MAX, POWER_UNIT_2026
from .environment import (EnvironmentalState, environment_at_time,
                          state_from_summary, timeline_from_samples, with_wetness)
from .stint import (PitContext, UNKNOWN_PIT, infer_pit_context,
                    pit_context_from_plan, require_causal, stint_from_lap_row)
from . import tyres as tyremod
from .decision import (DecisionModel, TyreDecisionContext, ZoneModel,
                       build_model, calibrate_zone_with_wear,
                       explain_exogenous_action, fail_cost_from_geometry,
                       make_bins, solve_exogenous)
from .physics_context import PhysicsContext
from .overtake import COEFFS
from .vehicle import VehicleParams

PHYSICS_CONFIG_VERSION = "decision-speed-map-v4-wear-surface"
ZONE_CALIBRATION_DT = 0.005


@dataclass(frozen=True)
class UsableEnergyAtOpportunity:
    """Current deployable store estimate at one decision location."""
    usable_energy_j: float
    usable_p10_j: float
    usable_p90_j: float
    sample_index: int
    sample_time_s: float
    source: str = "belief.usable_mean_at_or_before_zone"


@dataclass(frozen=True)
class OpportunityGap:
    gap_s: float
    source: str
    confidence: float
    age_s: float = 0.0
    method: str = "unknown"


@dataclass(frozen=True)
class TraceState:
    sample_index: int
    time_s: float
    lap: int
    s_m: float
    v_mps: float
    total_s_m: float


@dataclass(frozen=True)
class OvertakeDecisionInput:
    lap: int
    laps_left: int
    zone_name: str
    own_usable_energy_j: float
    rival_usable_energy_j: float
    rival_usable_p10_j: float
    rival_usable_p90_j: float
    actual_gap_s: float
    gap_source: str
    gap_confidence: float
    gap_age_s: float
    gap_method: str
    model: DecisionModel
    rival_track_j: np.ndarray
    # ---- P1, all optional so a P0 caller is unchanged.
    environment: EnvironmentalState | None = None
    own_tyre: Any = None
    rival_tyre: Any = None
    own_pit_context: PitContext | None = None
    rival_pit_context: PitContext | None = None
    tyre_decision: Any = None       # decision.TyreDecisionContext


def evaluate_overtake_decision(inp: OvertakeDecisionInput) -> dict[str, Any]:
    """Return one structured ATTACK/HOLD decision from the core DP."""
    # Oracle pit data cannot reach the solver: require_causal raises long before
    # this point, and this is the last gate before the DP sees anything.
    for ctx in (inp.own_pit_context, inp.rival_pit_context):
        if ctx is not None:
            require_causal(ctx)
    sol = solve_exogenous(inp.model, inp.rival_track_j, tyre=inp.tyre_decision)
    wear = None if inp.own_tyre is None else float(inp.own_tyre.wear_fraction)
    detail = explain_exogenous_action(
        sol, inp.laps_left, inp.own_usable_energy_j, wear=wear,
        own_tyre=inp.own_tyre, rival_tyre=inp.rival_tyre,
        pit_context=inp.own_pit_context, environment=inp.environment)
    zone_rows = detail["zones"]
    selected_zone = detail["best_zone"]
    if selected_zone is None:
        best = detail["best_attack"] or zone_rows[0]
    else:
        best = next(r for r in zone_rows if r["zone"] == selected_zone)
    gap_confidence = float(np.clip(inp.gap_confidence, 0.0, 1.0))
    calibration_confidence = 0.35  # p_pass coefficients are synthetic placeholders.
    confidence = float(np.clip(min(gap_confidence, calibration_confidence), 0.0, 1.0))
    return {
        "decision": detail["decision"],
        "zone": selected_zone or best["zone"],
        "pass_probability": best["pass_probability"],
        "attack_threshold": detail["attack_threshold"],
        "own_usable_energy_mj": inp.own_usable_energy_j / 1e6,
        "rival_usable_energy_mj": inp.rival_usable_energy_j / 1e6,
        "rival_usable_p10_mj": inp.rival_usable_p10_j / 1e6,
        "rival_usable_p90_mj": inp.rival_usable_p90_j / 1e6,
        "predicted_delta_v_mps": best["predicted_delta_v_mps"],
        "predicted_own_speed_mps": best["predicted_own_speed_mps"],
        "predicted_rival_speed_mps": best["predicted_rival_speed_mps"],
        "actual_gap_s": inp.actual_gap_s,
        "gap_s": inp.actual_gap_s,
        "gap_source": inp.gap_source,
        "gap_confidence": gap_confidence,
        "gap_age_s": inp.gap_age_s,
        "gap_method": inp.gap_method,
        # -inf is the DP's honest value for an attack it rejected, and it is not
        # representable in JSON. Publish null plus the flag, and keep the
        # hypothetical separately so the panel can still show what an attack
        # would have been worth.
        "value_attack": (float(best["value_attack"])
                         if np.isfinite(best["value_attack"]) else None),
        "value_attack_ranked": (float(best["value_attack"])
                                if np.isfinite(best["value_attack"]) else -1.0e18),
        "value_attack_hypothetical": float(best["value_attack_hypothetical"]),
        "attack_affordable": bool(detail["attack_affordable"]),
        "value_wait": detail["value_wait"],
        "confidence": confidence,
        "physics_model": "cached longitudinal vehicle.step zone speed map",
        "decision_model": "Core finite-horizon DP / solve_exogenous",
        "opponent_state": "inferred usable energy at current zone",
        "pass_model": {
            "model_type": "logistic",
            "calibration": "placeholder",
            "dataset_version": "synthetic-design-anchors",
            "coefficients": COEFFS.__dict__,
        },
        "model_calibration": "synthetic",
        "forecast_type": "heuristic_current_belief_plus_historical_rates",
        "future_energy_model": "bounded_historical_rate_transition",
        "future_energy_model_class": "heuristic",
        "future_energy_forecast_method": "bounded_historical_rate_transition",
        "forecast_confidence": 0.55,
        "future_energy_forecast_confidence": 0.55,
        "future_energy_forecast_mj": [float(x) / 1e6 for x in inp.rival_track_j],
        "all_zones": zone_rows,
        # ---- P1 state and provenance, so the web explains without recomputing
        "environment": detail.get("environment"),
        "weather_source": (None if inp.environment is None else inp.environment.source),
        "weather_age_s": (None if inp.environment is None else inp.environment.age_s),
        "track_wetness_index": (None if inp.environment is None
                                else inp.environment.track_wetness_index),
        "wetness_source": "inferred_rain_memory_assumed_coefficients",
        "own_tyre": detail.get("own_tyre"),
        "rival_tyre": detail.get("rival_tyre"),
        "tyre_model": detail.get("tyre_model"),
        "tyre_calibration": "synthetic",
        "wear_fraction": detail.get("wear_fraction"),
        "rival_wear_fraction": detail.get("rival_wear_fraction"),
        "attack_wear_continuation_penalty": detail.get("attack_wear_continuation_penalty"),
        "pit_resets_next_lap": detail.get("pit_resets_next_lap"),
        "pit_context": detail.get("pit_context"),
        "pit_source": (None if inp.own_pit_context is None
                       else inp.own_pit_context.source),
        "pit_confidence": (None if inp.own_pit_context is None
                           else inp.own_pit_context.confidence),
    }


def evaluate_decision_trace_from_payload(payload: dict[str, Any], car: str,
                                         rival: str,
                                         explicit_pit_plan: dict | None = None
                                         ) -> dict[str, Any]:
    """Evaluate every causal lap/zone opportunity in an analysed race payload."""
    if car not in payload["cars"] or rival not in payload["cars"]:
        raise KeyError("car not analysed")
    track = track_from_payload(payload)
    params = params_from_payload(payload)
    laps = common_trace_laps(payload["cars"][car], payload["cars"][rival])
    if len(laps) < 3:
        raise ValueError("not enough common laps for a decision trace")

    rows = []
    for i, lap in enumerate(laps):
        row = _evaluate_lap(payload, track, params, car, rival, laps, i, lap,
                            plan=explicit_pit_plan)
        rows.append(row)
    call = next((r for r in rows if r["attack"]), None)
    return {
        "car": car,
        "rival": rival,
        "laps": rows,
        "call": call,
        "fan": {"curves": [], "consensus_lap": call["lap"] if call else None,
                "consensus_fraction": None, "n_policies": 0},
        "zone_models": [{"name": z.name, "severity": z.braking_severity,
                         "dv_per_mj": round(z.dv_per_mj, 3)}
                        # Context-free maps: this block describes the TRACK for the
                        # UI, not the decision. The per-opportunity maps are
                        # built inside _evaluate_lap with the tyre/environment.
                        for z in _cached_zone_models(speed_map_cache_key(payload, params))],
        "metadata": {
            "decision_model": "Core finite-horizon DP / solve_exogenous",
            "physics": "cached longitudinal simulation",
            "opponent_state": "inferred usable energy at current zone",
            "pass_model": "synthetic placeholder logistic",
            "forecast": "heuristic current belief plus historical causal rates",
            "weather": "causal sample at or before the decision time; session "
                       "mean only when no trace exists",
            "wetness": "INFERRED from the Rainfall bool with assumed rain/drying "
                       "coefficients; not a FastF1 measurement",
            "tyre_state": "modelled wear and temperature; tyre_life is OBSERVED "
                          "AGE IN LAPS and is never wear",
            "tyre_calibration": "synthetic",
            "pit_context": "team/synthetic plan when supplied, otherwise unknown "
                           "-- no validated rival pit inference exists, and "
                           "actual pit times are oracle-only",
            "deployed_lap": "point-estimate deployed store energy over the lap, MJ",
            "deployed_lap_posterior_mean": "posterior mean deployed store energy over the lap, MJ",
            "usable_mean": "current deployable store energy belief at sample time, MJ",
            "reserve_mean": "inferred held-back buffer, J",
        },
    }


def _evaluate_lap(payload: dict[str, Any], track, params: VehicleParams,
                  car: str, rival: str, laps: list[int], i: int, lap: int,
                  plan: dict | None = None) -> dict[str, Any]:
    own_trace = payload["cars"][car]["trace"]
    rival_trace = payload["cars"][rival]["trace"]
    candidates = []
    n_laps = len(laps)
    for zi, zone in enumerate(track.zones):
        # One decision point per opportunity, and every input is read at it.
        # The gap used to be taken at `s_straight_end` -- the braking point,
        # which the ego reaches ~160 m AFTER the instant its energy belief is
        # sampled. That is a future observation feeding a past decision, and no
        # amount of "it is only 2 s later" makes it causal. Both the belief and
        # the gap are now read at the zone entry; carrying that gap into
        # `p_pass` as the gap at braking is an approximation, and it is labelled
        # one (`gap_reference_s`) rather than fixed by peeking.
        decision_s = zone.s_straight_start
        own_e = belief_at_position(own_trace, lap, decision_s)
        rival_e = belief_at_position(rival_trace, lap, decision_s)
        gap = gap_at_position(payload, car, rival, lap, decision_s)
        dyn = _historical_dynamics(payload, car, rival, lap)
        laps_left = n_laps - i
        # Everything below is read AT the decision time, never after it.
        env = environment_at_opportunity(payload, own_e.sample_time_s)
        cfg = _config()
        own_tyre = tyre_state_at_opportunity(payload, car, lap, env, cfg)
        rival_tyre = tyre_state_at_opportunity(payload, rival, lap, env, cfg)
        own_pit = pit_context_at_opportunity(payload, car, lap, plan)
        rival_pit = pit_context_at_opportunity(payload, rival, lap, plan)
        tyre_dec = tyre_decision_context(payload, own_tyre, rival_tyre, own_pit,
                                         laps, lap, laps_left)
        # The speed maps are rebuilt for THIS opportunity's physical context:
        # both compounds, the causal weather, and the wear axis. Cached on that
        # context, so the cost is paid once per context rather than per lap.
        surface = None
        if own_tyre is not None and own_tyre.source != "unknown":
            surface = ZoneSurfaceContext.build(own_tyre, rival_tyre, env,
                                               SURFACE_WEAR_LEVELS)
        zones = _cached_zone_models(
            speed_map_cache_key(payload, params, surface=surface))
        # The DP is given THIS zone only. Handing it every zone made it run its
        # own argmax over zones while the loop below ran a second one, so the
        # reported zone could be a zone whose gap and energy were never the ones
        # evaluated. One selection, here.
        zone_model = zones[zi]
        model = DecisionModel(
            zones=[zone_model], recharge_per_lap=dyn["recharge_per_lap_j"],
            own_spend_per_lap=dyn["own_spend_per_lap_j"],
            rival_spend_per_lap=dyn["rival_spend_per_lap_j"],
            attack_cost=float(zone_model.energy_grid.max()), defend_cost=0.0,
            gap_s=gap.gap_s, n_laps=laps_left, fail_cost=fail_cost_from_geometry(gap.gap_s),
            bins=make_bins())
        rival_track = _rival_energy_forecast(rival_e.usable_energy_j, dyn, laps_left)
        inp = OvertakeDecisionInput(
            lap=lap, laps_left=laps_left, zone_name=zone.name,
            own_usable_energy_j=own_e.usable_energy_j,
            rival_usable_energy_j=rival_e.usable_energy_j,
            rival_usable_p10_j=rival_e.usable_p10_j,
            rival_usable_p90_j=rival_e.usable_p90_j,
            actual_gap_s=gap.gap_s, gap_source=gap.source,
            gap_confidence=gap.confidence, gap_age_s=gap.age_s,
            gap_method=gap.method, model=model, rival_track_j=rival_track,
            environment=env, own_tyre=own_tyre, rival_tyre=rival_tyre,
            own_pit_context=own_pit, rival_pit_context=rival_pit,
            tyre_decision=tyre_dec)
        res = evaluate_overtake_decision(inp)
        candidate = {**res, "lap": lap, "requested_zone": zone.name,
                     "gap_reference_s": float(decision_s),
                     "decision_point_s": float(decision_s),
                     "decision_time_s": float(own_e.sample_time_s),
                     "own_sample_index": own_e.sample_index,
                     "rival_sample_index": rival_e.sample_index}
        candidates.append(candidate)

    # Attack candidates first, ranked by DP value; if none is affordable the
    # lap is a HOLD and the most valuable hypothetical is shown for context.
    chosen = max(candidates, key=lambda r: (r["decision"] == "ATTACK",
                                            r["value_attack_ranked"]))
    return {
        **chosen,
        "q": round(float(chosen["pass_probability"]), 4),
        "tau": round(float(chosen["attack_threshold"]), 4),
        "attack": chosen["decision"] == "ATTACK",
        "own_mj": round(float(chosen["own_usable_energy_mj"]), 4),
        "rival_mj": round(float(chosen["rival_usable_energy_mj"]), 4),
        "zone_candidates": candidates,
    }


@lru_cache(maxsize=1)
def _config() -> dict:
    from .config import load_config
    return load_config()


# The DP's wear axis. Independent of the SURFACE's axis and much finer is NOT
# required: `_solve_exogenous_tyre` interpolates the continuation value between
# levels, so a small wear increment is represented exactly rather than rounded
# away. This used to be sized from the wear rate -- up to 201 levels -- which
# was necessary while the DP snapped to the nearest bin and is pure cost now
# that it does not. 11 levels, verified against 101 by
# test_a_coarse_wear_grid_still_prices_a_small_attack_correctly.
DP_WEAR_LEVELS = 11


def _wear_grid_for(extra_wear: float, cap: int = DP_WEAR_LEVELS) -> np.ndarray:
    """The DP wear axis. Coarse, because the DP interpolates along it."""
    return np.round(np.linspace(0.0, 1.0, int(cap)), 6)


def tyre_decision_context(payload, own_tyre, rival_tyre, own_pit: PitContext,
                          laps: list[int], lap: int, n_laps: int):
    """Build the DP's tyre axis from the tyre model and a CAUSAL pit context.

    Returns None when there is no tyre state to reason about, which keeps the
    solver on its P0 path. The wear rates are integrated from xray.tyres, never
    chosen here.
    """
    if own_tyre is None or own_tyre.source == "unknown":
        return None
    cfg = _config()
    params = tyremod.params_from_config(cfg, own_tyre.compound)
    lap_m = float(payload["circuit_geometry"]["length"])
    temp = own_tyre.estimated_temp_c
    hold = tyremod.wear_per_lap(params, lap_m, temp, tyremod.ASSUMED_UTIL_NORMAL,
                                start_wear=own_tyre.wear_fraction)
    extra = tyremod.attack_extra_wear(params, lap_m, temp,
                                      start_wear=own_tyre.wear_fraction)
    if hold <= 0.0 and extra <= 0.0:
        return None
    pit_k = None
    if own_pit is not None and own_pit.planned_pit_lap is not None:
        if own_pit.planned_pit_lap in laps:
            pit_k = int(n_laps - laps.index(int(own_pit.planned_pit_lap)))
    return TyreDecisionContext(
        wear_grid=_wear_grid_for(extra if extra > 0 else hold),
        wear_per_lap_hold=hold, wear_per_lap_attack=hold + extra,
        pit_at_laps_left=pit_k, compound=own_tyre.compound,
        rival_wear=(None if rival_tyre is None or rival_tyre.source == "unknown"
                    else float(rival_tyre.wear_fraction)),
        pit_source=(own_pit.source if own_pit else "unknown"),
        pit_confidence=(own_pit.confidence if own_pit else 0.0))


def environment_at_opportunity(payload: dict[str, Any],
                               decision_time_s: float) -> EnvironmentalState:
    """Weather at the decision instant, causally.

    Falls back to the session summary only when there is no trace at all, and
    the returned `source` says which happened -- a mean is the right answer for
    a header and the wrong one for a decision, so the caller can tell.
    """
    rows = payload.get("weather_trace") or []
    if not rows:
        return state_from_summary(payload.get("weather") or {},
                                  t=float(decision_time_s))
    tl = with_wetness(timeline_from_samples(rows))
    return environment_at_time(tl, float(decision_time_s))


def _lap_rows_for(payload: dict[str, Any], driver: str) -> list[dict]:
    return [r for r in (payload.get("laps") or []) if r.get("driver") == driver]


def tyre_state_at_opportunity(payload: dict[str, Any], driver: str, lap: int,
                              env: EnvironmentalState, cfg: dict | None = None):
    """Tyre state at a decision, from causal lap metadata plus the wear model.

    Only laps STRICTLY BEFORE this one, plus this lap's own compound/age, are
    read. The wear integral runs forward from the last fresh set; `tyre_life`
    stays the observed age throughout and is never used as wear.
    """
    rows = sorted(_lap_rows_for(payload, driver), key=lambda r: int(r.get("lap", 0)))
    current = next((r for r in rows if int(r.get("lap", 0)) == int(lap)), None)
    if current is None:
        return tyremod.unknown(None, None, track_temp_c=env.track_temp_c or 30.0)
    st_meta = stint_from_lap_row(driver, current)
    params = tyremod.params_from_config(cfg or {}, st_meta.compound)
    track_temp = env.track_temp_c if env.track_temp_c is not None else 30.0

    # Where this stint began: the most recent fresh set at or before this lap.
    past = [r for r in rows if int(r.get("lap", 0)) <= int(lap)]
    start = 0
    saw_fitting = False
    for idx, r in enumerate(past):
        if r.get("fresh_tyre") is True or (
                r.get("stint") is not None and r.get("stint") != current.get("stint")):
            start = idx if r.get("fresh_tyre") is True else idx + 1
            saw_fitting = True
    stint_laps_seen = max(len(past) - start, 0)
    # Observed AGE sets how many laps the wear model is integrated over. This is
    # not the forbidden `wear = tyre_life / N` shortcut -- age is an input to the
    # integral, never its output. It matters because a set can have run laps
    # before our telemetry window opens: counting only the rows we happen to
    # hold would report a 30-lap-old tyre as brand new.
    stint_laps = stint_laps_seen
    if (not saw_fitting and st_meta.tyre_life is not None
            and np.isfinite(st_meta.tyre_life)):
        # Age fills in history we did NOT see. It does not override history we
        # did: if the set was fitted inside our window we watched it from new,
        # and `stint_laps_seen` is the better number even when the age column
        # disagrees. Extending only in the unobserved case is the difference
        # between using age as an input and letting it overrule an observation.
        stint_laps = max(stint_laps_seen, int(max(st_meta.tyre_life, 0.0)))
    if stint_laps == 0:
        return tyremod.fresh(st_meta.compound or "UNKNOWN", params, track_temp,
                             wetness=env.track_wetness_index,
                             tyre_life=st_meta.tyre_life or 0.0)

    state = tyremod.fresh(st_meta.compound or "UNKNOWN", params, track_temp,
                          wetness=env.track_wetness_index, tyre_life=0.0)
    lap_m = float(payload["circuit_geometry"]["length"])
    for _ in range(stint_laps):
        inc = tyremod.wear_per_lap(params, lap_m, track_temp,
                                   tyremod.ASSUMED_UTIL_NORMAL,
                                   env.track_wetness_index,
                                   start_wear=state.wear_fraction)
        state = tyremod.TyreState(
            compound=state.compound,
            tyre_life=st_meta.tyre_life if st_meta.tyre_life is not None else 0.0,
            fresh_tyre=st_meta.fresh_tyre,
            estimated_temp_c=track_temp + params.heating_gain * tyremod.ASSUMED_UTIL_NORMAL,
            wear_fraction=min(state.wear_fraction + inc, 1.0),
            grip_scale=tyremod.grip_scale(
                track_temp + params.heating_gain * tyremod.ASSUMED_UTIL_NORMAL,
                min(state.wear_fraction + inc, 1.0), env.track_wetness_index, params),
            source="modelled_thermal_wear")
    return state


def pit_context_at_opportunity(payload: dict[str, Any], driver: str, lap: int,
                               explicit_plan: dict | None = None) -> PitContext:
    """Pit horizon at a decision. Plans are allowed; the oracle is not.

    An explicit plan is our own team's, which we genuinely know at decision
    time. For anyone else there is no validated inference model in this
    checkout, so the answer is `unknown` -- and `require_causal` guarantees the
    actual pit laps sitting in `payload["laps"]` can never leak in here.
    """
    plan = (explicit_plan or {}).get(driver) if explicit_plan else None
    if plan is not None:
        if isinstance(plan, dict):
            return require_causal(pit_context_from_plan(
                lap, plan.get("planned_pit_lap"),
                source=plan.get("source", "team_plan"),
                confidence=float(plan.get("confidence", 0.9))))
        return require_causal(pit_context_from_plan(lap, int(plan)))
    rows = [r for r in _lap_rows_for(payload, driver)
            if int(r.get("lap", 0)) <= int(lap)]
    current = next((r for r in rows if int(r.get("lap", 0)) == int(lap)), None)
    if current is None:
        return UNKNOWN_PIT
    return require_causal(infer_pit_context(stint_from_lap_row(driver, current),
                                            laps_seen=len(rows)))


def belief_at_position(trace: dict[str, list], lap: int, s_m: float) -> UsableEnergyAtOpportunity:
    """Causal usable-energy belief at or before ``s_m`` on ``lap``."""
    lap_arr = np.asarray(trace["lap"], dtype=int)
    s_arr = np.asarray(trace["s"], dtype=float)
    m = np.flatnonzero((lap_arr == int(lap)) & (s_arr <= float(s_m)))
    if len(m) == 0:
        raise ValueError(f"no causal trace samples for lap {lap} at s <= {s_m:.1f} m")
    idx = int(m[-1])
    return UsableEnergyAtOpportunity(
        usable_energy_j=float(trace["usable_mean"][idx]) * 1e6,
        usable_p10_j=float(trace["usable_p10"][idx]) * 1e6,
        usable_p90_j=float(trace["usable_p90"][idx]) * 1e6,
        sample_index=idx,
        sample_time_s=float(trace["t"][idx]),
    )


def gap_at_position(payload: dict[str, Any], car: str, rival: str, lap: int,
                    s_m: float) -> OpportunityGap:
    """Actual gap at a decision location, with explicit fallbacks."""
    track_length = float(payload["circuit_geometry"]["length"])
    ego = _state_at_or_before_position(payload["cars"][car]["trace"], lap, s_m,
                                       track_length)
    if ego is not None:
        rival_now = _state_projected_from_last_observation(payload["cars"][rival]["trace"],
                                                           ego.time_s, track_length)
        if rival_now is not None:
            gap_m = rival_now.total_s_m - ego.total_s_m
            if 0.0 < gap_m < 2000.0:
                local_v = max(ego.v_mps, 1.0)
                age_s = max(ego.time_s - rival_now.time_s, 0.0)
                method = ("same_time_observation" if age_s <= 1e-6
                          else "causal_constant_velocity_projection")
                conf = 0.9 if age_s <= 1e-6 else float(np.clip(0.9 * np.exp(-age_s / 1.0),
                                                               0.25, 0.85))
                return OpportunityGap(float(gap_m / local_v),
                                      "same_time_position_trace", conf,
                                      age_s=age_s, method=method)

    tc = _time_at_or_before_position(payload["cars"][car]["trace"], lap, s_m)
    tr = _time_at_or_before_position(payload["cars"][rival]["trace"], lap, s_m)
    if tc is not None and tr is not None:
        gap = tc - tr
        if 0.0 < gap < 30.0:
            return OpportunityGap(float(gap), "historical_position_crossing", 0.75,
                                  age_s=0.0, method="historical_position_crossing")
    for row in payload.get("gaps", []):
        if int(row.get("lap", -1)) == int(lap) and row.get("car") == car and row.get("ahead") == rival:
            return OpportunityGap(float(row["gap_s"]), "lap_timing_fallback", 0.55,
                                  age_s=float("nan"), method="lap_timing_fallback")
    return OpportunityGap(0.45, "explicit_low_confidence_fallback", 0.2,
                          age_s=float("nan"), method="constant_low_confidence_fallback")


def common_trace_laps(own_car: dict[str, Any], rival_car: dict[str, Any]) -> list[int]:
    own = {int(x) for x in own_car["trace"]["lap"]}
    rival = {int(x) for x in rival_car["trace"]["lap"]}
    return sorted(x for x in own & rival if x > 0)


def params_from_payload(payload: dict[str, Any]) -> VehicleParams:
    pooled = payload.get("calibration", {}).get("pooled") or {}
    cda = float(pooled.get("cda_pooled") or 0.9)
    rho = float(payload.get("weather", {}).get("rho") or 1.20)
    return VehicleParams(
        mass_car=768.0, fuel_start=70.0, fuel_burn_per_lap=1.4,
        cda_straight=cda, cda_corner=1.9 * cda, crr=0.012, rho=rho,
        brake_decel_max=45.0, drivetrain_eff=0.95)


def track_from_payload(payload: dict[str, Any]):
    return _PayloadTrack(payload)


# The wear surface is a reusable physical object, so the cache key describes the
# CONTEXT it was built in, never the instantaneous state that reads it. Keying on
# `wear_fraction` -- which the first version did -- means 0.183742 and 0.183891
# are different cache entries and every single decision pays for a full
# recalibration. Wear is the surface's axis; the key holds the grid, not a point
# on it.
#
# Continuous context values are quantised so the cache is reusable at all. The
# steps below are far finer than the physical sensitivity of the map (a 0.005
# kg/m^3 rho step is ~0.4% of drag) and every one of them is a deliberate,
# documented approximation rather than an accident of float formatting.
RHO_QUANTUM = 0.005          # kg/m^3
TRACK_TEMP_QUANTUM_C = 2.0   # C
WETNESS_QUANTUM = 0.05       # dimensionless


def _q(value, quantum, default=None):
    if value is None:
        return default
    return round(round(float(value) / quantum) * quantum, 6)


@dataclass(frozen=True)
class ZoneSurfaceContext:
    """Everything that changes an energy-by-wear speed surface, and nothing else.

    Compounds are here because the tyre model's grip differs by compound;
    `wear_grid` is here because it defines the surface's axis; the environment
    fields are here because they reach `vehicle.step` through `PhysicsContext`.
    The CURRENT wear fractions are deliberately absent -- they select within the
    surface rather than defining it.
    """
    own_compound: str
    rival_compound: str
    wear_grid: tuple
    rho: float
    track_temp_c: float | None
    wetness: float

    @classmethod
    def build(cls, own_tyre, rival_tyre, env: EnvironmentalState | None,
              wear_grid) -> "ZoneSurfaceContext":
        return cls(
            own_compound=str(getattr(own_tyre, "compound", "UNKNOWN")),
            rival_compound=str(getattr(rival_tyre, "compound", "UNKNOWN")),
            wear_grid=tuple(round(float(w), 6) for w in wear_grid),
            rho=_q(getattr(env, "rho", None), RHO_QUANTUM, 1.2),
            track_temp_c=_q(getattr(env, "track_temp_c", None),
                            TRACK_TEMP_QUANTUM_C, None),
            wetness=_q(getattr(env, "track_wetness_index", 0.0),
                       WETNESS_QUANTUM, 0.0))


def speed_map_cache_key(payload: dict[str, Any], params: VehicleParams,
                        surface: "ZoneSurfaceContext | None" = None,
                        environment: EnvironmentalState | None = None) -> str:
    """Canonical cache key for the E -> braking-point speed map(s).

    Hashes a whitelist of the whole physics configuration rather than
    accumulating one field per bug, so UI metadata is absent by construction
    rather than by exclusion.
    """
    data = _speed_map_cache_payload(params_payload=payload, params=params)
    if surface is not None:
        data["surface"] = asdict(surface)
        # The tyre coefficients themselves are part of the physics: changing a
        # compound's mu or its optimal temperature changes the map without
        # changing anything else in this key.
        cfg = (_config().get("tyres") or {}).get("compounds") or {}
        data["tyre_model"] = {c: cfg.get(c) for c in
                              sorted({surface.own_compound, surface.rival_compound})}
    elif environment is not None:
        data["environment"] = {
            "rho": _q(environment.rho, RHO_QUANTUM, 1.2),
            "track_temp_c": _q(environment.track_temp_c, TRACK_TEMP_QUANTUM_C, None),
            "wetness": _q(environment.track_wetness_index, WETNESS_QUANTUM, 0.0),
        }
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def _speed_map_cache_payload(params_payload: dict[str, Any], params: VehicleParams) -> dict:
    payload = params_payload
    track = _PayloadTrack(payload)
    return {
        "version": PHYSICS_CONFIG_VERSION,
        "dt": ZONE_CALIBRATION_DT,
        "vehicle_params": asdict(params),
        "regulation": asdict(POWER_UNIT_2026),
        "track": {
            "length": round(float(track.length), 2),
            "s_signature": [round(float(x), 3) for x in track._grid_s.tolist()],
            "grade_signature": [round(float(x), 6) for x in track._grade.tolist()],
            "zones": [
                {
                    "name": z.name,
                    "s_straight_start": round(z.s_straight_start, 2),
                    "s_straight_end": round(z.s_straight_end, 2),
                    "s_end": round(z.s_end, 2),
                    "braking_severity": round(z.braking_severity, 3),
                    "apex_v": round(z.apex_v, 3),
                }
                for z in track.zones
            ],
        },
    }


# Zone calibrations are the expensive part of a decision, so the surface is
# built once per physical context and every wear fraction interpolates within it.
SURFACE_N_ENERGY = 7
SURFACE_DT = 0.005
# The wear axis of the SURFACE. Coarse on purpose: `ZoneModel._interp_surface`
# interpolates between levels, so more levels buy accuracy in the second decimal
# of a speed and cost a full zone calibration each. This is the surface's grid,
# not the DP's -- the DP has its own, finer, and they are independent.
SURFACE_WEAR_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _tyre_context_factory(compound: str, env: EnvironmentalState | None,
                          wetness: float, track_temp_c: float | None):
    """(wear) -> PhysicsContext for one car, built from the tyre model.

    The tyre TEMPERATURE is derived here rather than taken from the current
    state, which is what keeps the surface reusable: a map keyed on the
    instantaneous tyre temperature would miss the cache on almost every sample.
    It is the compound's own steady-state at normal utilisation, which is the
    condition the braking-point speed is actually reached in.
    """
    params = tyremod.params_from_config(_config(), compound)
    base_temp = 30.0 if track_temp_c is None else float(track_temp_c)
    temp = base_temp + params.heating_gain * tyremod.ASSUMED_UTIL_NORMAL

    def make(wear: float) -> PhysicsContext:
        st = tyremod.TyreState(
            compound=params.compound, tyre_life=0.0, fresh_tyre=None,
            estimated_temp_c=temp, wear_fraction=float(wear),
            grip_scale=tyremod.grip_scale(temp, float(wear), wetness, params),
            source="modelled_surface_calibration")
        return PhysicsContext(environment=env, tyre=st)

    return make


@lru_cache(maxsize=32)
def _cached_zone_models(cache_key: str) -> tuple[ZoneModel, ...]:
    """Zone speed maps for one physical context.

    Two paths, and the legacy one is not a fallback for failure -- it is the
    correct answer when there is no tyre state to reason about, and it keeps
    every P0 caller numerically identical.
    """
    data = json.loads(cache_key)
    track = _PayloadTrack.from_cache_payload(data["track"])
    params = VehicleParams(**data["vehicle_params"])
    surface = data.get("surface")
    if surface is None:
        model = build_model(track, params, n_laps=12,
                            recharge_per_lap=E_HARVEST_LAP * 0.4,
                            rival_spend_per_lap=E_HARVEST_LAP * 0.4)
        return tuple(model.zones)

    env = state_from_summary(
        {"rho": surface["rho"], "track_temp_c": surface["track_temp_c"]},
        source="zone_surface_context")
    wear_levels = tuple(surface["wear_grid"])
    own_ctx = _tyre_context_factory(surface["own_compound"], env,
                                    surface["wetness"], surface["track_temp_c"])
    riv_ctx = _tyre_context_factory(surface["rival_compound"], env,
                                    surface["wetness"], surface["track_temp_c"])
    # Same chassis (the payload carries one pooled CdA), different rubber. Tyre
    # independence is what is required here, not per-car aero.
    return tuple(
        calibrate_zone_with_wear(track, params, z, own_ctx,
                                 wear_levels=wear_levels,
                                 rival_params=params,
                                 rival_context_for_wear=riv_ctx,
                                 n_points=SURFACE_N_ENERGY, dt=SURFACE_DT)
        for z in track.zones)


def _historical_dynamics(payload: dict[str, Any], car: str, rival: str, before_lap: int) -> dict:
    own = payload["cars"][car]
    riv = payload["cars"][rival]
    return {
        "recharge_per_lap_j": _mean_lap_record_j(riv.get("harvested_lap", {}), before_lap,
                                                 default=E_HARVEST_LAP * 0.4),
        "own_spend_per_lap_j": _mean_lap_record_j(own.get("deployed_lap", {}), before_lap,
                                                  default=E_HARVEST_LAP * 0.25),
        "rival_spend_per_lap_j": _mean_lap_record_j(riv.get("deployed_lap", {}), before_lap,
                                                    default=E_HARVEST_LAP * 0.35),
    }


def _mean_lap_record_j(record: dict[str, float], before_lap: int, default: float) -> float:
    vals = [float(v) * 1e6 for k, v in record.items() if int(k) < int(before_lap)
            and np.isfinite(float(v)) and float(v) > 0.0]
    return float(np.mean(vals)) if vals else float(default)


def _rival_energy_forecast(current_usable_energy_j: float, dyn: dict,
                           laps_left: int) -> np.ndarray:
    vals = []
    e = float(np.clip(current_usable_energy_j, 0.0, E_STORE_MAX))
    harvested_per_lap_j = float(np.clip(dyn["recharge_per_lap_j"], 0.0, E_HARVEST_LAP))
    requested_deploy_per_lap_j = float(np.clip(dyn["rival_spend_per_lap_j"],
                                               0.0, E_STORE_MAX + E_HARVEST_LAP))
    for _ in range(max(int(laps_left), 1)):
        vals.append(e)
        deployed_j = min(requested_deploy_per_lap_j, e + harvested_per_lap_j)
        e = float(np.clip(e - deployed_j + harvested_per_lap_j, 0.0, E_STORE_MAX))
    return np.asarray(vals, dtype=float)


def _state_at_or_before_position(trace: dict[str, list], lap: int, s_m: float,
                                 track_length_m: float) -> TraceState | None:
    lap_arr = np.asarray(trace["lap"], dtype=int)
    s_arr = np.asarray(trace["s"], dtype=float)
    m = np.flatnonzero((lap_arr == int(lap)) & (s_arr <= float(s_m)))
    if len(m) == 0:
        return None
    return _state_at_index(trace, int(m[-1]), track_length_m)


def _state_at_or_before_time(trace: dict[str, list], time_s: float,
                             track_length_m: float) -> TraceState | None:
    t_arr = np.asarray(trace["t"], dtype=float)
    m = np.flatnonzero(t_arr <= float(time_s))
    if len(m) == 0:
        return None
    return _state_at_index(trace, int(m[-1]), track_length_m)


def _state_projected_from_last_observation(trace: dict[str, list], time_s: float,
                                           track_length_m: float) -> TraceState | None:
    t_arr = np.asarray(trace["t"], dtype=float)
    before = np.flatnonzero(t_arr <= float(time_s))
    if len(before) == 0:
        return None
    i0 = int(before[-1])
    t0 = float(t_arr[i0])
    lap = int(trace["lap"][i0])
    s0 = float(trace["s"][i0])
    v_mps = float(trace.get("v", [1.0])[i0])
    dt = max(float(time_s) - t0, 0.0)
    projected_total = (lap - 1) * float(track_length_m) + s0 + max(v_mps, 1.0) * dt
    projected_lap = int(projected_total // float(track_length_m)) + 1
    s_m = projected_total % float(track_length_m)
    return TraceState(
        sample_index=i0,
        time_s=t0,
        lap=projected_lap,
        s_m=s_m,
        v_mps=v_mps,
        total_s_m=projected_total,
    )


def _state_at_index(trace: dict[str, list], idx: int, track_length_m: float) -> TraceState:
    lap = int(trace["lap"][idx])
    s_m = float(trace["s"][idx])
    total = (lap - 1) * float(track_length_m) + s_m
    return TraceState(
        sample_index=idx,
        time_s=float(trace["t"][idx]),
        lap=lap,
        s_m=s_m,
        v_mps=float(trace.get("v", [1.0])[idx]),
        total_s_m=total,
    )


def _time_at_or_before_position(trace: dict[str, list], lap: int, s_m: float) -> float | None:
    lap_arr = np.asarray(trace["lap"], dtype=int)
    s_arr = np.asarray(trace["s"], dtype=float)
    t_arr = np.asarray(trace["t"], dtype=float)
    m = np.flatnonzero((lap_arr == int(lap)) & (s_arr <= float(s_m)))
    if len(m) == 0:
        return None
    return float(t_arr[int(m[-1])])


class _PayloadTrack:
    STRAIGHT_V_LIMIT = 400 / 3.6

    def __init__(self, payload: dict[str, Any]):
        geo = payload["circuit_geometry"]
        self.length = float(geo["length"])
        self.name = payload.get("circuit", "analysed circuit")
        self._grid_s = np.asarray(geo.get("s") or [0.0], dtype=float)
        self._grade = np.asarray(geo.get("grade") or np.zeros_like(self._grid_s), dtype=float)
        self.zones = [_PayloadZone(z, self._apex_speed(payload, z)) for z in geo["zones"]]
        self.corners = [(z.s_straight_end, z.s_end, z.apex_v) for z in self.zones]

    @classmethod
    def from_cache_payload(cls, data: dict):
        length = data["length"]
        zones = data["zones"]
        payload = {"circuit_geometry": {
            "length": length,
            "s": data.get("s_signature") or [0.0],
            "grade": data.get("grade_signature") or [0.0],
            "zones": zones}}
        return cls(payload)

    def point(self, s: float) -> tuple:
        s = float(s) % self.length
        is_corner = False
        v_lim = self.STRAIGHT_V_LIMIT
        for z in self.zones:
            if z.s_straight_end <= s < z.s_end:
                is_corner = True
                v_lim = z.apex_v
                break
        ahead = []
        for start, _end, v in self.corners:
            d = (start - s) % self.length
            if d <= 400.0:
                ahead.append((d, v))
        ahead.sort()
        while len(ahead) < 2:
            ahead.append((1e9, 1e9))
        return (float(v_lim), self.grade(s), is_corner,
                float(ahead[0][0]), float(ahead[0][1]),
                float(ahead[1][0]), float(ahead[1][1]))

    def grade(self, s):
        if len(self._grid_s) <= 1:
            return 0.0
        return float(np.interp(float(s) % self.length, self._grid_s, self._grade))

    def zone_by_name(self, name: str):
        return next(z for z in self.zones if z.name == name)

    def _apex_speed(self, payload: dict[str, Any], zone: dict[str, Any]) -> float:
        if "apex_v" in zone:
            return float(zone["apex_v"])
        # Causal API decisions cannot infer a current corner target from future
        # telemetry samples. Payloads produced by real analysis should include
        # zone ``apex_v``; this severity fallback is intentionally conservative.
        severity = float(zone.get("braking_severity", 0.5))
        return float((70.0 + 150.0 * (1.0 - severity)) / 3.6)


class _PayloadZone:
    def __init__(self, data: dict[str, Any], apex_v: float):
        self.name = str(data["name"])
        self.s_straight_start = float(data["s_straight_start"])
        self.s_straight_end = float(data["s_straight_end"])
        self.s_end = float(data.get("s_end", self.s_straight_end + 120.0))
        self.braking_severity = float(data["braking_severity"])
        self.detection_point = float(data.get("detection_point",
                                              self.s_straight_start - 200.0))
        self.energy_cost_hint = float(data.get("energy_cost_hint", 0.0))
        self.apex_v = float(apex_v)


# ======================================================================== P2
# Strategic opportunity / deployment optimisation. Additive and opt-in: the P0
# and P1 paths above are untouched, and `evaluate_decision_trace_from_payload`
# only calls this when asked. This function ORCHESTRATES -- it builds causal
# state and hands it to xray.opportunity. There is no Bellman recursion, no
# pass-model formula and no speed equation below this line.
P2_CONFIG_VERSION = "p2-opportunity-v1"


def build_opportunity_horizon(payload: dict[str, Any], car: str, rival: str,
                              from_lap: int | None = None,
                              plan: dict | None = None,
                              max_opportunities: int = 12) -> list:
    """The current opportunity, then a CAUSAL FORECAST of the ones after it.

    This is the part of P2 that is easiest to get wrong and hardest to notice.
    The horizon is the future, and the payload contains the future, so building
    opportunity 5 by reading lap 5's telemetry produces a beautifully accurate
    plan that a real car could never have made. The first version of this
    function did exactly that, and the causality test caught it: the chosen
    action and every physical quantity were identical, but `value_action` moved,
    because the value of waiting had been computed from data that had not
    happened yet.

    So only opportunity 0 is observed. Everything after it is forecast from what
    is known at the decision instant:

        zones      track geometry, known in advance
        gap        persistence of the current gap -- a heuristic, labelled
        rival E    `_rival_energy_forecast`, the P1 bounded causal forecast
        own E      not forecast at all; the DP carries it through its own
                   transition, so whatever is put here is overwritten
        tyre       likewise: the DP advances wear itself

    Everything downstream of the first opportunity therefore carries
    `source="causal_forecast"`, and no later telemetry sample is read.
    """
    from .opportunity import DecisionOpportunity, order_opportunities

    track = track_from_payload(payload)
    params = params_from_payload(payload)
    own_trace = payload["cars"][car]["trace"]
    rival_trace = payload["cars"][rival]["trace"]
    laps = common_trace_laps(payload["cars"][car], payload["cars"][rival])
    if from_lap is not None:
        laps = [x for x in laps if int(x) >= int(from_lap)]
    if not laps:
        return []
    cfg = _config()

    # ---- opportunity 0: observed, at its own decision point -----------------
    first = None
    for lap in laps:
        for zone in track.zones:
            decision_s = zone.s_straight_start
            try:
                own_e = belief_at_position(own_trace, lap, decision_s)
                rival_e = belief_at_position(rival_trace, lap, decision_s)
            except ValueError:
                continue
            gap = gap_at_position(payload, car, rival, lap, decision_s)
            env = environment_at_opportunity(payload, own_e.sample_time_s)
            own_tyre = tyre_state_at_opportunity(payload, car, lap, env, cfg)
            rival_tyre = tyre_state_at_opportunity(payload, rival, lap, env, cfg)
            surface = None
            if own_tyre is not None and own_tyre.source != "unknown":
                surface = ZoneSurfaceContext.build(own_tyre, rival_tyre, env,
                                                   SURFACE_WEAR_LEVELS)
            zones = _cached_zone_models(
                speed_map_cache_key(payload, params, surface=surface))
            first_zones = zones
            first = DecisionOpportunity(
                opportunity_id=f"L{int(lap)}-{zone.name}",
                lap=int(lap), zone_name=zone.name, decision_s=float(decision_s),
                decision_time_s=float(own_e.sample_time_s),
                zone=next(z for z in zones if z.name == zone.name),
                gap_s=float(gap.gap_s),
                own_usable_energy_j=float(own_e.usable_energy_j),
                rival_usable_energy_mean_j=float(rival_e.usable_energy_j),
                rival_usable_energy_p10_j=float(rival_e.usable_p10_j),
                rival_usable_energy_p90_j=float(rival_e.usable_p90_j),
                own_tyre=own_tyre, rival_tyre=rival_tyre,
                own_pit_context=pit_context_at_opportunity(payload, car, lap, plan),
                rival_pit_context=pit_context_at_opportunity(payload, rival, lap, plan),
                environment=env, source=gap.source, confidence=gap.confidence)
            break
        if first is not None:
            break
    if first is None:
        return []

    out = [first]
    if max_opportunities <= 1:
        return out
    # The zone maps of the context we are standing in. Forecast opportunities
    # reuse them deliberately: rebuilding a surface for a forecast tyre and a
    # forecast sky would be forecasting physics on top of forecast state, and
    # the extra precision would be invented rather than known.
    zone_by_name = {z.name: z for z in first_zones}

    # ---- the rest: forecast, never observed ---------------------------------
    dyn = _historical_dynamics(payload, car, rival, first.lap)
    n_zones = max(len(track.zones), 1)
    n_future_laps = (max_opportunities // n_zones) + 2
    rival_track = _rival_energy_forecast(first.rival_usable_energy_mean_j,
                                         dyn, n_future_laps)
    start_zi = next(i for i, z in enumerate(track.zones) if z.name == first.zone_name)
    k = 0
    lap = first.lap
    zi = start_zi
    while len(out) < max_opportunities:
        zi += 1
        if zi >= n_zones:
            zi = 0
            lap += 1
        k += 1
        zone = track.zones[zi]
        lap_offset = min(int(k / n_zones), len(rival_track) - 1)
        out.append(DecisionOpportunity(
            opportunity_id=f"L{int(lap)}-{zone.name}",
            lap=int(lap), zone_name=zone.name,
            decision_s=float(zone.s_straight_start),
            decision_time_s=None,               # a forecast has no observation time
            zone=zone_by_name[zone.name],
            gap_s=float(first.gap_s),           # persistence, heuristic
            own_usable_energy_j=float(first.own_usable_energy_j),  # DP overwrites
            rival_usable_energy_mean_j=float(rival_track[lap_offset]),
            rival_usable_energy_p10_j=None,
            rival_usable_energy_p90_j=None,
            own_tyre=first.own_tyre, rival_tyre=first.rival_tyre,
            own_pit_context=first.own_pit_context,
            rival_pit_context=first.rival_pit_context,
            environment=first.environment,
            source="causal_forecast_persistence_gap_bounded_rival_energy",
            confidence=0.4))
    return order_opportunities(out)[:max_opportunities]


def evaluate_opportunity_decision(payload: dict[str, Any], car: str, rival: str,
                                  from_lap: int | None = None,
                                  plan: dict | None = None,
                                  deployment_fractions=None,
                                  max_opportunities: int = 12) -> dict[str, Any]:
    """The P2 recommendation: which opportunity, which zone, how many joules."""
    from . import opportunity as opp_mod

    horizon = build_opportunity_horizon(payload, car, rival, from_lap, plan,
                                        max_opportunities)
    if not horizon:
        raise ValueError("no causal opportunities available")
    first = horizon[0]

    # Per-opportunity dynamics from canonical P1 pieces, never re-derived here.
    cfg = _config()
    tyre_params = tyremod.params_from_config(
        cfg, getattr(first.own_tyre, "compound", None))
    per_lap = max(len({o.zone_name for o in horizon}), 1)
    lap_m = float(payload["circuit_geometry"]["length"])
    env = first.environment
    dyn = _historical_dynamics(payload, car, rival, first.lap)
    model = DecisionModel(
        zones=[first.zone], recharge_per_lap=dyn["recharge_per_lap_j"],
        own_spend_per_lap=dyn["own_spend_per_lap_j"],
        rival_spend_per_lap=dyn["rival_spend_per_lap_j"],
        attack_cost=float(first.zone.energy_grid.max()), defend_cost=0.0,
        gap_s=first.gap_s, n_laps=len(horizon),
        fail_cost=fail_cost_from_geometry(first.gap_s), bins=make_bins())
    transition = opp_mod.transition_model_from(
        model, tyre_params, distance_m=lap_m / per_lap,
        track_temp_c=(env.track_temp_c if env and env.track_temp_c is not None else 30.0),
        wetness=(env.track_wetness_index if env else 0.0),
        opportunities_per_lap=per_lap,
        start_wear=first.own_wear or 0.0)

    # A KNOWN own stop inside the horizon resets the tyre. Causality is gated by
    # stint.require_causal upstream; an oracle context can never reach here.
    pit_at = None
    own_pit = first.own_pit_context
    if own_pit is not None and own_pit.is_causal and own_pit.planned_pit_lap is not None:
        for idx, o in enumerate(horizon):
            if int(o.lap) >= int(own_pit.planned_pit_lap):
                pit_at = idx
                break

    fr = (deployment_fractions if deployment_fractions is not None
          else tuple((cfg.get("decision") or {}).get("deployment_fractions")
                     or opp_mod.DEFAULT_DEPLOYMENT_FRACTIONS))
    sol = opp_mod.solve_opportunities(model, horizon, transition,
                                      deployment_fractions=fr,
                                      pit_at_index=pit_at)
    return _serialise_p2(sol, first, pit_at)


def _serialise_p2(sol, first, pit_at) -> dict[str, Any]:
    """Solver result -> JSON. Reads solver output; computes no decision."""
    out = sol.chosen_outcome
    nb = sol.next_best_action
    return {
        "decision": sol.decision,
        "zone": sol.chosen.zone_name,
        "opportunity_id": first.opportunity_id,
        "lap": first.lap,
        "decision_point_s": first.decision_s,
        "decision_time_s": first.decision_time_s,
        "deployment_budget_mj": sol.chosen.deployment_budget_j / 1e6,
        "actual_deployed_mj": out.actual_deployed_j / 1e6,
        "deployment_saturated": bool(out.saturated),
        "saturation_reason": out.saturation_reason,
        "predicted_own_speed_mps": out.own_speed_mps,
        "predicted_rival_speed_mps": out.rival_speed_mps,
        "predicted_delta_v_mps": out.delta_v_mps,
        "pass_probability": out.pass_probability,
        "pass_model_calibration": sol.pass_model_calibration,
        "value_action": sol.value_action,
        "value_hold": sol.value_hold,
        "decision_margin": sol.decision_margin,
        "next_best_action": None if nb is None else {
            "kind": nb.kind, "zone": nb.zone_name,
            "deployment_budget_mj": nb.deployment_budget_j / 1e6,
            "value": sol.next_best_value,
        },
        "action_consensus": sol.action_consensus,
        "expected_regret": sol.expected_regret,
        "policy_posterior": list(sol.policy_posterior),
        "robustness": dict(sol.robustness),
        "horizon": [
            {"opportunity_id": o.opportunity_id, "lap": o.lap, "zone": o.zone_name,
             "decision_s": o.decision_s, "decision_time_s": o.decision_time_s,
             "gap_s": o.gap_s,
             "own_usable_energy_mj": o.own_usable_energy_j / 1e6,
             "rival_usable_energy_mj": o.rival_usable_energy_mean_j / 1e6,
             "own_wear_fraction": o.own_wear, "rival_wear_fraction": o.rival_wear}
            for o in sol.opportunities],
        "candidate_actions": [
            {"action": lbl, "kind": o.action.kind,
             "requested_budget_mj": o.requested_budget_j / 1e6,
             "actual_deployed_mj": o.actual_deployed_j / 1e6,
             "feasible": o.feasible, "infeasible_reason": o.infeasible_reason,
             "saturated": o.saturated,
             "own_speed_mps": o.own_speed_mps,
             "rival_speed_mps": o.rival_speed_mps,
             "delta_v_mps": o.delta_v_mps,
             "pass_probability": o.pass_probability,
             "value": sol.action_values.get(lbl)}
            for lbl, o in sol.outcomes.items()],
        "pit_reset_index": pit_at,
        "input_confidence": {
            "gap_source": first.source, "gap_confidence": first.confidence,
            "weather_source": getattr(first.environment, "source", None),
            "own_tyre_source": getattr(first.own_tyre, "source", None),
            "rival_tyre_source": getattr(first.rival_tyre, "source", None),
            "own_pit_source": getattr(first.own_pit_context, "source", None),
            "rival_pit_source": getattr(first.rival_pit_context, "source", None),
        },
        "p2_version": P2_CONFIG_VERSION,
    }
