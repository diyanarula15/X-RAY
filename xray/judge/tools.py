"""The tools the judge may call, and the dispatch that binds them to one situation.

Standard library only. The declarations are plain dicts in the OpenAPI subset
`google.genai` accepts as function declarations, so the whole registry can be
built, dispatched and tested with no SDK installed.

Every tool binds `(race, car, rival, t)` from the run context. The model never
supplies identity arguments, which is what stops it asking about a different
situation, a different pair, or a later lap whose outcome it could read off. The
only model-supplied arguments in the whole surface are `zone` and `topic`, and
both are validated against an enum before dispatch.
"""
from __future__ import annotations

from . import evidence as ev
from .evidence import DOCTRINE, EvidenceError

TOOL_DECLARATIONS = [
    {
        "name": "get_rival_energy_bracket",
        "description": (
            "The rival's usable-energy point value with its reported [p10, p90] "
            "spread and the bracket width, plus this car's own usable energy "
            "and the run-level usable/reserve means."),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "get_gap_provenance",
        "description": (
            "How the gap to the rival was obtained: value, source, confidence, "
            "age in seconds, and method."),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "get_zone_models",
        "description": (
            "The candidate deployment zones at this decision point, each with "
            "its pass probability, predicted delta-v, attack value and "
            "affordability. Omit `zone` for all of them."),
        "parameters": {
            "type": "OBJECT",
            "properties": {"zone": {
                "type": "STRING",
                "description": "optional single zone name, e.g. 'A'"}},
        },
    },
    {
        "name": "get_pass_model_metadata",
        "description": (
            "The pass model behind this row's pass_probability: its type, its "
            "calibration status, the dataset it was fitted on, and its "
            "coefficients."),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "get_regulation_variant",
        "description": (
            "Which 2026 regulation variant this race was analysed under, with "
            "the power caps that variant sets, plus the circuit and date."),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "get_pair_status",
        "description": (
            "Pair-level status: whether the engine refused this pair, any P2 "
            "error, the evaluation horizon, and how many decision points the "
            "race contained."),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "get_forecast_and_context",
        "description": (
            "The future-energy forecast with its model class and confidence, "
            "plus environment, weather provenance, tyre state and calibration, "
            "and pit context."),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "get_policy_fan",
        "description": (
            "Robustness across policies: how many policies were run and what "
            "fraction agreed."),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "get_repo_doctrine",
        "description": (
            "Fixed text from this project's own documentation about how a "
            "quantity must be reported. Use it to ground a judgement in the "
            "project's stated rules rather than in your own assumptions."),
        "parameters": {
            "type": "OBJECT",
            "properties": {"topic": {
                "type": "STRING", "enum": sorted(DOCTRINE),
                "description": "which rule to quote"}},
            "required": ["topic"],
        },
    },
]

TOOL_NAMES = tuple(d["name"] for d in TOOL_DECLARATIONS)


class Context:
    """One situation, and the nine tools bound to it."""

    def __init__(self, bundle: dict, race: str, car: str, rival: str,
                 t: float | None):
        self.bundle = bundle
        self.race, self.car, self.rival, self.t = race, car, rival, t

    # A refused pair has no lap rows at all, so the per-lap tools have nothing
    # to answer with. They say so explicitly rather than raising something the
    # model has to guess at -- and `get_pair_status` still works, which is the
    # tool that matters for judging whether the refusal was right.
    def _require_t(self) -> float:
        if self.t is None:
            raise EvidenceError(
                "this is a pair-level refusal, so there is no decision point "
                "and no lap row. Use get_pair_status and get_repo_doctrine.")
        return self.t

    def dispatch(self, name: str, args: dict | None = None) -> dict:
        """Run one tool. Returns `{"error": ...}` instead of raising.

        A raising tool would kill a batch mid-run over one malformed bundle,
        after the run has already spent hours of a daily request cap getting
        there. `scripts/16.precompute_decisions.py` guards the same way at pair
        level with its bare `except Exception`.
        """
        args = args or {}
        try:
            return self._dispatch(name, args)
        except EvidenceError as exc:
            return {"error": f"EvidenceError: {exc}"}
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _dispatch(self, name: str, args: dict) -> dict:
        b = self.bundle
        if name == "get_rival_energy_bracket":
            return ev.rival_energy_bracket(b, self._require_t())
        if name == "get_gap_provenance":
            return ev.gap_provenance(b, self._require_t())
        if name == "get_zone_models":
            zone = args.get("zone")
            return ev.zone_models(b, self._require_t(),
                                  str(zone) if zone else None)
        if name == "get_pass_model_metadata":
            return ev.pass_model_metadata(b, self._require_t())
        if name == "get_regulation_variant":
            return ev.regulation_variant(self.race)
        if name == "get_pair_status":
            return ev.pair_status(b, self.race, self.car, self.rival)
        if name == "get_forecast_and_context":
            return ev.forecast_and_context(b, self._require_t())
        if name == "get_policy_fan":
            return ev.policy_fan(b)
        if name == "get_repo_doctrine":
            return ev.repo_doctrine(str(args.get("topic") or ""))
        return {"error": f"unknown tool {name!r}", "valid_tools": list(TOOL_NAMES)}

    def opening_evidence(self) -> dict:
        """What goes in the first user message, so the agent never has to spend
        a request fetching what it needs every time."""
        status = ev.pair_status(self.bundle, self.race, self.car, self.rival)
        if self.t is None:
            return {"pair_status": status, "situation_kind": "pair_refusal"}
        return {"pair_status": status, "situation_kind": "decision_point",
                "headline": ev.headline(self.bundle, self.t)}
