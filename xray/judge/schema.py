"""The verdict shape, and the contradictions the code refuses to accept.

Standard library only. The response schema is expressed as a plain dict in
OpenAPI-subset form, which is what `google.genai` accepts as `response_schema`;
building it here rather than in `gemini.py` keeps the SDK import confined to one
file and lets every validation test run with no SDK installed.
"""
from __future__ import annotations

from . import RUBRIC_VERSION, VERDICT_LABEL


class VerdictInvalid(Exception):
    """The model returned something the rubric forbids. Carries the message the
    agent feeds back on its one retry."""


VERDICTS = (
    "JUSTIFIED",
    "JUSTIFIED_WITH_CAVEATS",
    "OVERCLAIMED",
    "UNSUPPORTED",
    "SHOULD_HAVE_REFUSED",
    "CORRECTLY_REFUSED",
    "INSUFFICIENT_EVIDENCE_TO_JUDGE",
)

# A verdict that is not an endorsement. Used by the contradiction rule below.
NON_ENDORSING = frozenset({
    "OVERCLAIMED", "UNSUPPORTED", "SHOULD_HAVE_REFUSED",
    "CORRECTLY_REFUSED", "INSUFFICIENT_EVIDENCE_TO_JUDGE",
})

CONFIDENCES = ("LOW", "MEDIUM", "HIGH")

AXES = (
    "bracket_reporting",
    "pass_model_honesty",
    "gap_provenance",
    "affordability_and_value",
    "confidence_calibration",
    "refusal_discipline",
)

# An axis scored 0 on either of these is the judge saying the call rests on a
# number that does not support it. Returning JUSTIFIED in the same breath is a
# self-contradiction, and it is also the cheapest available route to
# sycophancy: flag everything, endorse everything, look thorough. Enforced in
# code rather than asked for in the prompt, because a rubric rule a model can
# decline to follow is a suggestion.
HARD_FAIL_AXES = frozenset({"bracket_reporting", "pass_model_honesty"})

TOOL_NAMES = (
    "get_rival_energy_bracket",
    "get_gap_provenance",
    "get_zone_models",
    "get_pass_model_metadata",
    "get_regulation_variant",
    "get_pair_status",
    "get_forecast_and_context",
    "get_policy_fan",
    "get_repo_doctrine",
    "headline",
)

# Provenance is filled by code and is absent from the schema the model sees. A
# model that supplies any of it is rejected outright: `is_measurement: false` is
# the one field standing between a verdict and being mistaken for physics, and
# it must never be a value the model had an opportunity to set.
RESERVED_KEYS = frozenset({
    "label", "is_measurement", "model_id", "prompt_version", "rubric_version",
    "judge_version", "stack", "situation_key", "race", "car", "rival",
    "decision_time_s", "lap", "artefact_mtime_ns", "tool_calls", "n_turns",
    "n_requests", "generated_at", "judge_error", "verdict_label",
    "verdict_tone",
})

_AXIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "score": {"type": "INTEGER",
                  "description": "2 = met, 1 = partially met, 0 = violated"},
        "note": {"type": "STRING",
                 "description": "at most 240 characters, citing the field names "
                                "that decided it"},
        "evidence_refs": {"type": "ARRAY", "items": {"type": "STRING"},
                          "description": "names of the tools whose results "
                                         "support this score"},
    },
    "required": ["score", "note", "evidence_refs"],
}

VERDICT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdict": {"type": "STRING", "enum": list(VERDICTS)},
        "summary": {"type": "STRING",
                    "description": "at most 60 words, no number that did not "
                                   "come from a tool result"},
        "judge_confidence": {"type": "STRING", "enum": list(CONFIDENCES)},
        "abstained": {"type": "BOOLEAN"},
        "abstain_reason": {"type": "STRING"},
        "axes": {
            "type": "OBJECT",
            "properties": {a: _AXIS_SCHEMA for a in AXES},
            "required": list(AXES),
        },
        "overclaims": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "claim": {"type": "STRING"},
                    "field": {"type": "STRING"},
                    "why_wrong": {"type": "STRING"},
                },
                "required": ["claim", "field", "why_wrong"],
            },
        },
    },
    "required": ["verdict", "summary", "judge_confidence", "abstained", "axes",
                 "overclaims"],
}

# Display strings and tones live server-side. The frontend must not hold an
# enum -> label or enum -> colour map: a map in the UI is a second opinion about
# what a verdict means, which is the drift `tests/test_frontend_p2.py` and
# `test_frontend_p4.py` exist to prevent.
VERDICT_LABELS = {
    "JUSTIFIED": "justified by the evidence",
    "JUSTIFIED_WITH_CAVEATS": "justified, caveat omitted",
    "OVERCLAIMED": "over-claimed",
    "UNSUPPORTED": "not supported by the evidence",
    "SHOULD_HAVE_REFUSED": "should have refused",
    "CORRECTLY_REFUSED": "correctly refused",
    "INSUFFICIENT_EVIDENCE_TO_JUDGE": "judge abstained",
}

VERDICT_TONES = {
    "JUSTIFIED": "ok",
    "JUSTIFIED_WITH_CAVEATS": "ok",
    "OVERCLAIMED": "warn",
    "UNSUPPORTED": "bad",
    "SHOULD_HAVE_REFUSED": "bad",
    "CORRECTLY_REFUSED": "ok",
    "INSUFFICIENT_EVIDENCE_TO_JUDGE": "neutral",
}


def validate_verdict(obj) -> dict:
    """Return the model's verdict, normalised, or raise `VerdictInvalid`."""
    if not isinstance(obj, dict):
        raise VerdictInvalid(f"expected a JSON object, got {type(obj).__name__}")

    reserved = RESERVED_KEYS & set(obj)
    if reserved:
        raise VerdictInvalid(
            f"these keys are filled by the harness and must not appear in your "
            f"output: {sorted(reserved)}")

    verdict = obj.get("verdict")
    if verdict not in VERDICTS:
        raise VerdictInvalid(f"verdict must be one of {list(VERDICTS)}, "
                             f"got {verdict!r}")

    conf = obj.get("judge_confidence")
    if conf not in CONFIDENCES:
        raise VerdictInvalid(f"judge_confidence must be one of "
                             f"{list(CONFIDENCES)}, got {conf!r}")

    axes = obj.get("axes")
    if not isinstance(axes, dict):
        raise VerdictInvalid("axes must be an object with all six rubric axes")
    missing = [a for a in AXES if a not in axes]
    if missing:
        raise VerdictInvalid(f"axes is missing {missing}")

    clean_axes = {}
    for name in AXES:
        ax = axes[name]
        if not isinstance(ax, dict):
            raise VerdictInvalid(f"axes.{name} must be an object")
        score = ax.get("score")
        if score not in (0, 1, 2):
            raise VerdictInvalid(f"axes.{name}.score must be 0, 1 or 2, "
                                 f"got {score!r}")
        refs = ax.get("evidence_refs") or []
        if not isinstance(refs, list):
            raise VerdictInvalid(f"axes.{name}.evidence_refs must be a list")
        bad = [r for r in refs if r not in TOOL_NAMES]
        if bad:
            raise VerdictInvalid(
                f"axes.{name}.evidence_refs names {bad}, which are not tools. "
                f"Valid: {list(TOOL_NAMES)}")
        clean_axes[name] = {"score": int(score),
                            "note": str(ax.get("note") or "")[:240],
                            "evidence_refs": [str(r) for r in refs]}

    # The contradiction rule.
    failed = [a for a in HARD_FAIL_AXES if clean_axes[a]["score"] == 0]
    if failed and verdict not in NON_ENDORSING:
        raise VerdictInvalid(
            f"you scored {failed} at 0, meaning the call rests on a number that "
            f"does not support it, but returned verdict {verdict!r}. A 0 on "
            f"{sorted(HARD_FAIL_AXES)} requires one of "
            f"{sorted(NON_ENDORSING)}.")

    abstained = bool(obj.get("abstained"))
    if verdict == "INSUFFICIENT_EVIDENCE_TO_JUDGE" and not abstained:
        abstained = True

    overclaims = []
    for oc in obj.get("overclaims") or []:
        if isinstance(oc, dict):
            overclaims.append({"claim": str(oc.get("claim") or ""),
                               "field": str(oc.get("field") or ""),
                               "why_wrong": str(oc.get("why_wrong") or "")})

    return {
        "verdict": verdict,
        "verdict_label": VERDICT_LABELS[verdict],
        "verdict_tone": VERDICT_TONES[verdict],
        "summary": str(obj.get("summary") or ""),
        "judge_confidence": conf,
        "abstained": abstained,
        "abstain_reason": (str(obj["abstain_reason"])
                           if obj.get("abstain_reason") else None),
        "axes": clean_axes,
        "overclaims": overclaims,
    }


def abstention(reason: str) -> dict:
    """The verdict recorded when the judge could not produce a valid one.

    Reported, never dropped -- the same discipline as `replay_error` in
    `_situations_rows`: a filter over a silently shortened list is a filter that
    lies about how many situations there were.
    """
    return {
        "verdict": "INSUFFICIENT_EVIDENCE_TO_JUDGE",
        "verdict_label": VERDICT_LABELS["INSUFFICIENT_EVIDENCE_TO_JUDGE"],
        "verdict_tone": VERDICT_TONES["INSUFFICIENT_EVIDENCE_TO_JUDGE"],
        "summary": "",
        "judge_confidence": "LOW",
        "abstained": True,
        "abstain_reason": reason,
        "axes": {a: {"score": 0, "note": "not assessed", "evidence_refs": []}
                 for a in AXES},
        "overclaims": [],
    }


def with_provenance(verdict: dict, **fields) -> dict:
    """Attach the harness-filled provenance. `is_measurement: false` and the
    label go on here and nowhere else, so there is exactly one place where a
    verdict acquires its "this is not physics" marking."""
    out = dict(verdict)
    out.update({
        "label": VERDICT_LABEL,
        "is_measurement": False,
        "rubric_version": RUBRIC_VERSION,
    })
    out.update(fields)
    return out
