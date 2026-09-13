"""Reading a precomputed bundle, and the redaction that defines this evaluator.

Standard library only (see `xray/judge/__init__.py`).

Every function here is `dict -> dict`. Nothing computes physics, nothing calls
a solver, nothing reads a race artefact's telemetry payload. The only arithmetic
is `p90 - p10`, a subtraction of two numbers that are already in the file, and
it exists because a bracket width is the thing the judge is being asked to
notice.
"""
from __future__ import annotations

import json
from pathlib import Path

from .paths import DECISIONS, DECISIONS_INDEX, RACES


class EvidenceError(Exception):
    """A situation the bundle cannot serve. Returned to the model as a tool
    result, never raised past the agent loop -- one malformed bundle must not
    kill a batch that has spent a day of quota getting this far."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _loads(text: str) -> dict:
    """Parse a bundle, mapping the JSON non-finite tokens to None.

    `scripts/16.precompute_decisions.py` writes `json.dumps(out)` with
    `allow_nan` at its default True, so `_json_safe`'s scrubbing in the API does
    not survive that path for every field: `2026_r10_R__ANT__LEC.json` carries a
    literal `NaN` token for `gap_age_s`. `json.loads` accepts it and yields
    `float('nan')`, which then raises the moment anything re-serialises the
    value with `allow_nan=False` -- which is exactly what the verdict writer
    does. Mapped to None here, at the one boundary where every bundle enters.
    """
    return json.loads(text, parse_constant=lambda _: None)


def load_bundle(race: str, car: str, rival: str) -> dict:
    p = DECISIONS / f"{race}__{car}__{rival}.json"
    try:
        return _loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvidenceError(f"no bundle at {p.name}") from exc
    except ValueError as exc:
        raise EvidenceError(f"unparseable bundle {p.name}: {exc}") from exc


# The seven summary keys. `out/races/*.json` artefacts run to megabytes -- the
# RDD endpoint used to re-parse 130 MB of them per request -- and none of that
# payload is evidence about a single decision point. Loading the whole file to
# read `regulation` is wasteful; handing the whole file to a tool would also
# hand over `laps[].position`, which is the outcome.
_RACE_SUMMARY_KEYS = ("id", "year", "round", "circuit", "event", "date",
                      "regulation")


def load_race_meta(race: str) -> dict:
    p = RACES / f"{race}.json"
    try:
        d = _loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvidenceError(f"no race artefact for {race}") from exc
    except ValueError as exc:
        raise EvidenceError(f"unparseable artefact {race}: {exc}") from exc
    return {k: d.get(k) for k in _RACE_SUMMARY_KEYS}


def artefact_mtime_ns(race: str) -> int | None:
    try:
        return (RACES / f"{race}.json").stat().st_mtime_ns
    except OSError:
        return None


def read_decisions_index() -> dict:
    try:
        return json.loads(DECISIONS_INDEX.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def bundle_refusal(bundle: dict, race: str, car: str, rival: str,
                   index: dict | None = None) -> str | None:
    """Pair-level refusal, tolerant of bundles written before it existed.

    99 of the 102 bundles on disk have no `refusal` key at all: they predate the
    `build_bundle` that added it. A `bundle["refusal"]` would raise on 97% of
    the corpus, so this is a `.get` with a fallback to `_index.json`, which was
    rebuilt later and does carry the field.
    """
    if "refusal" in bundle:
        return bundle["refusal"]
    idx = read_decisions_index() if index is None else index
    return (idx.get(f"{race}__{car}__{rival}") or {}).get("refusal")


def pairs_on_disk() -> list[tuple[str, str, str]]:
    out = []
    for p in sorted(DECISIONS.glob("*__*__*.json")):
        parts = p.stem.split("__")
        if len(parts) == 3:
            out.append((parts[0], parts[1], parts[2]))
    return out


# --------------------------------------------------------------------------
# Situation identity
# --------------------------------------------------------------------------

# A refused pair has zero situations but is still judged, at pair level: whether
# refusing was the right call is exactly the kind of question this evaluator
# exists to ask, and sampling refusals out would bias the corpus toward the
# cases where the engine was willing to speak.
REFUSAL_SENTINEL = "refusal"


def key_of(race: str, car: str, rival: str, t: float | None) -> str:
    tail = REFUSAL_SENTINEL if t is None else f"{t:.2f}"
    return f"{race}__{car}__{rival}@{tail}"


def parse_key(key: str) -> tuple[str, str, str, float | None]:
    head, _, tail = key.rpartition("@")
    race, car, rival = head.split("__")
    return race, car, rival, (None if tail == REFUSAL_SENTINEL else float(tail))


def lap_row(bundle: dict, t: float) -> dict:
    """The `decision.laps` entry for one decision time.

    The join is exact, not nearest: `situations[i].decision_time_s` is copied
    verbatim from `decision.laps[i].decision_time_s` by `_situations_rows`, so a
    mismatch means the bundle is inconsistent and should refuse rather than
    silently grade the adjacent lap.
    """
    dec = bundle.get("decision") or {}
    for r in dec.get("laps") or []:
        ts = r.get("decision_time_s")
        if ts is not None and abs(float(ts) - t) < 1e-6:
            return r
    raise EvidenceError(f"no lap row at decision_time_s={t:.2f}")


def situation_row(bundle: dict, t: float) -> dict:
    for s in bundle.get("situations") or []:
        ts = s.get("decision_time_s")
        if ts is not None and abs(float(ts) - t) < 1e-6:
            return s
    raise EvidenceError(f"no situation at decision_time_s={t:.2f}")


# --------------------------------------------------------------------------
# Redaction -- the single most important rule in this package
# --------------------------------------------------------------------------

OUTCOME_KEYS = frozenset({
    "actual_action",
    "matches_recommendation",
    "replay_error",
    "position_before",
    "position_after",
    "actual_gap_s",
})
"""Fields that reveal what happened after the decision point.

Stripped recursively from every tool return, because the judge grades whether
the evidence supported the call and NOT whether the driver agreed. Those are
different questions, and the second one is already answered:
`decision_service.historical_replay` computes `matches_recommendation` with no
LLM anywhere near it. A judge that can see the outcome will grade the outcome,
and this evaluator would collapse into a worse copy of a function that already
exists.

`actual_gap_s` is in the list and does not look like an outcome. It is: on lap
1 of `2026_r1_R__HAM__LEC` it reads 1.3282 against `gap_s` 1.3282 with
`gap_source: "same_time_position_trace"` -- the OBSERVED gap, not the estimated
one the engine acted on. Handing it over lets the judge grade the call against a
fact the engine did not have, which is the same error in a quieter form.
"""


def redact_outcome(obj):
    """Drop every `OUTCOME_KEYS` entry, at any depth."""
    if isinstance(obj, dict):
        return {k: redact_outcome(v) for k, v in obj.items()
                if k not in OUTCOME_KEYS}
    if isinstance(obj, list):
        return [redact_outcome(v) for v in obj]
    return obj


def _pick(d: dict, keys) -> dict:
    return {k: d.get(k) for k in keys}


# --------------------------------------------------------------------------
# Evidence functions -- one per tool
# --------------------------------------------------------------------------

def headline(bundle: dict, t: float) -> dict:
    """What the engine called, and the top-line numbers behind it.

    Not a tool: this goes in the opening user message. Free-tier quota is the
    binding constraint on the whole harness (see `xray/judge/limiter.py`), and
    spending a round trip on data the agent needs every single time buys
    nothing.

    The call is read from `situations[].attack` and `laps[].decision`, NOT from
    `decision["call"]`, which is null in the bundles on disk -- it holds the
    race-level headline call, not a per-situation one.
    """
    row = lap_row(bundle, t)
    sit = situation_row(bundle, t)
    out = _pick(row, (
        "lap", "decision_time_s", "decision_point_s", "zone", "requested_zone",
        "decision", "attack", "attack_affordable", "attack_threshold",
        "pass_probability", "value_attack", "value_attack_ranked",
        "value_attack_hypothetical", "value_wait", "confidence",
        "decision_model", "physics_model", "opponent_state",
        "predicted_delta_v_mps", "predicted_own_speed_mps",
        "predicted_rival_speed_mps",
    ))
    out["situation_attack_flag"] = bool(sit.get("attack"))
    # `value_attack: null` is the DP's -inf mapped at the serialisation
    # boundary by `_json_safe` -- "not available at any price", not "worth
    # zero". Spelled out here because a model reading a bare null will
    # otherwise read it as a missing value.
    out["null_value_means"] = ("value_attack null = the DP's -inf = this action "
                               "is not available at any price. It is not zero.")
    return redact_outcome(out)


def rival_energy_bracket(bundle: dict, t: float) -> dict:
    row = lap_row(bundle, t)
    meta = (bundle.get("decision") or {}).get("metadata") or {}
    p10, p90 = row.get("rival_usable_p10_mj"), row.get("rival_usable_p90_mj")
    width = None
    if p10 is not None and p90 is not None:
        width = round(float(p90) - float(p10), 4)
    return redact_outcome({
        "rival_usable_energy_mj": row.get("rival_usable_energy_mj"),
        "rival_usable_p10_mj": p10,
        "rival_usable_p90_mj": p90,
        "bracket_width_mj": width,
        "own_usable_energy_mj": row.get("own_usable_energy_mj"),
        "own_mj": row.get("own_mj"),
        "rival_mj": row.get("rival_mj"),
        "metadata_usable_mean": meta.get("usable_mean"),
        "metadata_reserve_mean": meta.get("reserve_mean"),
        "metadata_opponent_state": meta.get("opponent_state"),
        "interpretation": (
            "rival_usable_energy_mj is a point summary of an interval whose "
            "reported spread is [p10, p90]. The point value alone is not a "
            "sufficient report of what is known about the rival's energy."),
    })


def gap_provenance(bundle: dict, t: float) -> dict:
    row = lap_row(bundle, t)
    return redact_outcome(_pick(row, (
        "gap_s", "gap_source", "gap_confidence", "gap_age_s", "gap_method",
        "gap_reference_s",
    )))


def zone_models(bundle: dict, t: float, zone: str | None = None) -> dict:
    row = lap_row(bundle, t)
    cand = row.get("all_zones") or []
    if zone:
        cand = [z for z in cand if z.get("zone") == zone]
        if not cand:
            raise EvidenceError(
                f"no zone {zone!r} at this decision point; "
                f"available: {[z.get('zone') for z in row.get('all_zones') or []]}")
    return redact_outcome({
        "requested_zone": row.get("requested_zone"),
        "chosen_zone": row.get("zone"),
        "all_zones": cand,
        "zone_candidates": row.get("zone_candidates"),
        "zone_shape": (bundle.get("decision") or {}).get("zone_models"),
    })


def pass_model_metadata(bundle: dict, t: float) -> dict:
    row = lap_row(bundle, t)
    meta = (bundle.get("decision") or {}).get("metadata") or {}
    return redact_outcome({
        "pass_model": row.get("pass_model"),
        "model_calibration": row.get("model_calibration"),
        "metadata_pass_model": meta.get("pass_model"),
        "pass_probability": row.get("pass_probability"),
        "attack_threshold": row.get("attack_threshold"),
    })


def regulation_variant(race: str) -> dict:
    return redact_outcome(load_race_meta(race))


def pair_status(bundle: dict, race: str, car: str, rival: str) -> dict:
    idx = read_decisions_index()
    sits = bundle.get("situations") or []
    p2 = bundle.get("p2") or {}
    return redact_outcome({
        "race": race, "car": car, "rival": rival,
        "refusal": bundle_refusal(bundle, race, car, rival, idx),
        "n_situations": len(sits),
        "horizon_s": bundle.get("horizon_s"),
        "artefact_mtime_ns": bundle.get("artefact_mtime_ns"),
        "p2_error": p2.get("error"),
        "has_race_level_call": (bundle.get("decision") or {}).get("call") is not None,
    })


def forecast_and_context(bundle: dict, t: float) -> dict:
    row = lap_row(bundle, t)
    forecast = row.get("future_energy_forecast_mj") or []
    return redact_outcome({
        "forecast_type": row.get("forecast_type"),
        "future_energy_model": row.get("future_energy_model"),
        "future_energy_model_class": row.get("future_energy_model_class"),
        "future_energy_forecast_method": row.get("future_energy_forecast_method"),
        "forecast_confidence": row.get("forecast_confidence"),
        # Eight laps is enough to see the shape and the saturation; the full
        # array is one entry per remaining lap and is mostly a flat 4.0 tail.
        "future_energy_forecast_mj_head": forecast[:8],
        "future_energy_forecast_n": len(forecast),
        "environment": row.get("environment"),
        "weather_source": row.get("weather_source"),
        "weather_age_s": row.get("weather_age_s"),
        "wetness_source": row.get("wetness_source"),
        "track_wetness_index": row.get("track_wetness_index"),
        "tyre_model": row.get("tyre_model"),
        "tyre_calibration": row.get("tyre_calibration"),
        "own_tyre": row.get("own_tyre"),
        "rival_tyre": row.get("rival_tyre"),
        "wear_fraction": row.get("wear_fraction"),
        "rival_wear_fraction": row.get("rival_wear_fraction"),
        "pit_source": row.get("pit_source"),
        "pit_confidence": row.get("pit_confidence"),
        "pit_resets_next_lap": row.get("pit_resets_next_lap"),
    })


def policy_fan(bundle: dict) -> dict:
    fan = (bundle.get("decision") or {}).get("fan") or {}
    return redact_outcome({
        "consensus_lap": fan.get("consensus_lap"),
        "consensus_fraction": fan.get("consensus_fraction"),
        "n_policies": fan.get("n_policies"),
        "n_curves": len(fan.get("curves") or []),
        "interpretation": (
            "n_policies 0 and consensus_fraction null mean the policy fan was "
            "not populated for this pair. That is an absence of evidence about "
            "robustness, not evidence of agreement."),
    })


# --------------------------------------------------------------------------
# Repo doctrine -- fixed strings, no model involvement
# --------------------------------------------------------------------------

DOCTRINE = {
    "pass_model": (
        "xray/overtake.py holds the only invented constants in the codebase: "
        "b0=-7.6650, b1=0.55, b2=2.10, b3=2.4324. b1 and b2 come from a design "
        "brief; b0 and b3 were solved from two shape anchors. They are NOT fit "
        "to any real race data. Every lap row in every bundle therefore carries "
        "pass_model.calibration='placeholder' and "
        "dataset_version='synthetic-design-anchors'. A pass_probability is a "
        "design anchor's output, not a measured pass rate."),
    "polytope_vs_posterior": (
        "The assumption-free identified set and the policy-tilted posterior are "
        "always reported as two numbers together. The tilted posterior rests on "
        "a behavioural assumption about how drivers deploy which cannot be "
        "checked on real data. In a decision row, the reported spread is "
        "[rival_usable_p10_mj, rival_usable_p90_mj]; rival_usable_energy_mj "
        "alone is the point summary and does not carry the width."),
    "confidence_cap": (
        "xray/decision_service.py caps decision confidence at 0.35, and the "
        "reason is the pass model: its coefficients are invented, and every "
        "decision claim inherits that. A row's own `confidence` field is "
        "frequently 0.2-0.25. A call presented with more certainty than the "
        "row's own confidence field supports is over-claiming."),
    "refusal_discipline": (
        "Declining is a correct output, not an error path. Silverstone and "
        "Monaco legitimately return 0% identifiability on real 2026 data; a car "
        "under tow is legitimately unidentifiable; three of ten races open on a "
        "battle whose first lap has no causal trace sample. A refusal that is "
        "stated is right. A number produced where the evidence does not support "
        "one is wrong even if it happens to be close."),
    "stack_identity": (
        "These bundles were produced by the real stack: FastF1 -> "
        "xray/data/ingest.py -> xray/realfit.py -> xray/analysis.py -> "
        "xray/decision_service.py. The Stage 1 estimator and the "
        "set-membership research group produced none of these numbers and must "
        "not be reasoned about here."),
    "unavailable_is_not_zero": (
        "Unavailable is not zero. value_attack null is the DP's -inf mapped at "
        "the serialisation boundary: the action is not available at any price. "
        "Likewise an absent wind or weather reading is available=False with a "
        "None value, never 0.0, so 'unknown' cannot be read as 'calm'."),
    "gap_provenance": (
        "gap_confidence and gap_source say how the gap was obtained. "
        "'explicit_low_confidence_fallback' with gap_confidence 0.2 is the "
        "engine stating it does not really know the gap. gap_age_s is how stale "
        "the reading is. A call that turns on a gap threshold while the gap is "
        "a low-confidence fallback is resting on a number the engine itself "
        "flagged as weak."),
}


def repo_doctrine(topic: str) -> dict:
    """Fixed repo-authored text. No LLM wrote any of this and none of it is
    derived from the bundle -- it is the doctrine the judge cites instead of
    inventing its own."""
    if topic not in DOCTRINE:
        # An unknown topic returns the menu rather than an error: a wasted turn
        # on a typo'd enum costs a request against a daily cap.
        return {"error": f"unknown topic {topic!r}",
                "valid_topics": sorted(DOCTRINE)}
    return {"topic": topic, "text": DOCTRINE[topic]}
