"""The tool-calling loop. Standard library only -- the SDK lives behind `client`.

Two phases, deliberately:

  A. a tool loop, with the tool declarations attached and no response schema
  B. one extraction call, with the tools removed and the response schema attached

They are separate because Gemini's function-calling configuration and a
`response_schema` do not compose reliably: with both attached, the model
frequently emits its verdict as one more function call, which then never
validates against the schema and burns a retry. Splitting them costs one extra
request per situation and buys a schema-shaped object every time.

The client is injected rather than constructed here, which is what lets the
whole loop -- termination, tool errors, turn caps, validation, retry, abstention
-- be tested offline against a stub with no API key and no network.
"""
from __future__ import annotations

import time

from . import JUDGE_VERSION, RUBRIC_VERSION, STACK
from . import cache as cache_mod
from . import evidence as ev
from . import prompt as prompt_mod
from . import schema as schema_mod
from .tools import Context

MAX_TURNS = 6
MAX_TOOL_CALLS = 12

# Three tool errors in a row means the bundle cannot answer what the model is
# asking, and more turns will not change that. Cut to the verdict rather than
# spending the rest of the turn budget on the same failure.
MAX_CONSECUTIVE_TOOL_ERRORS = 3


def judge_situation(client, race: str, car: str, rival: str,
                    t: float | None, *, bundle: dict | None = None,
                    max_turns: int = MAX_TURNS,
                    max_tool_calls: int = MAX_TOOL_CALLS) -> dict:
    """Judge one situation. Returns a cache entry (verdict + transcript + usage).

    `client` must expose `model_id`, `tool_turn(...)` and `extract(...)`; see
    `gemini.GeminiClient` for the real one and `tests/test_judge.py` for the
    stub.
    """
    t0 = time.time()
    bundle = ev.load_bundle(race, car, rival) if bundle is None else bundle
    ctx = Context(bundle, race, car, rival, t)
    opening = ctx.opening_evidence()
    key = ev.key_of(race, car, rival, t)

    transcript: list[dict] = [
        {"role": "user", "text": prompt_mod.opening_message(opening)}]
    tool_calls: list[str] = []
    n_requests = 0
    turns = 0
    consecutive_errors = 0
    loop_error = None

    while turns < max_turns:
        turns += 1
        try:
            calls, text = client.tool_turn(transcript)
        except Exception as exc:                      # noqa: BLE001
            loop_error = f"{type(exc).__name__}: {exc}"
            break
        n_requests += 1

        if not calls:
            if text:
                transcript.append({"role": "model", "text": text})
            break

        if len(tool_calls) + len(calls) > max_tool_calls:
            calls = calls[:max(0, max_tool_calls - len(tool_calls))]
            transcript.append({"role": "user",
                               "text": "Tool-call budget reached. Give your "
                                       "verdict from what you already have."})
            if not calls:
                break

        transcript.append({"role": "model", "calls": calls})
        results = []
        for call in calls:
            name = call.get("name") or ""
            out = ctx.dispatch(name, call.get("args") or {})
            tool_calls.append(name)
            results.append({"name": name, "result": out})
            consecutive_errors = consecutive_errors + 1 if "error" in out else 0
        transcript.append({"role": "tool", "results": results})

        if consecutive_errors >= MAX_CONSECUTIVE_TOOL_ERRORS:
            transcript.append({
                "role": "user",
                "text": "Those tools cannot answer for this situation. Judge on "
                        "what you have, or abstain and say why."})
            break

        if len(tool_calls) >= max_tool_calls:
            transcript.append({"role": "user",
                               "text": "Tool-call budget reached. Give your "
                                       "verdict from what you already have."})
            break

    verdict, judge_error = _extract(client, transcript, loop_error)
    n_requests += verdict.pop("_n_requests", 0)

    full = schema_mod.with_provenance(
        verdict,
        model_id=client.model_id,
        prompt_version=prompt_mod.PROMPT_VERSION,
        rubric_version=RUBRIC_VERSION,
        judge_version=JUDGE_VERSION,
        stack=STACK,
        situation_key=key,
        race=race, car=car, rival=rival,
        decision_time_s=t,
        lap=(opening.get("headline") or {}).get("lap"),
        artefact_mtime_ns=bundle.get("artefact_mtime_ns"),
        tool_calls=tool_calls,
        n_turns=turns,
        n_requests=n_requests,
        generated_at=_utcnow(),
        judge_error=judge_error,
    )
    return {
        "situation_key": key,
        "verdict": full,
        "transcript": transcript,
        "n_requests": n_requests,
        "wall_s": round(time.time() - t0, 2),
        "evidence_fp": cache_mod.evidence_fingerprint(opening),
    }


def _extract(client, transcript: list[dict], loop_error):
    """Phase B, with one retry on a validation failure.

    A second failure is recorded as an abstention, not dropped. A run that
    silently skipped the situations its judge could not answer would report a
    coverage number that was quietly about a different, easier corpus.
    """
    if loop_error is not None:
        v = schema_mod.abstention(f"tool loop failed: {loop_error}")
        v["_n_requests"] = 0
        return v, loop_error

    used = 0
    last = None
    for attempt in (0, 1):
        try:
            raw = client.extract(transcript)
            used += 1
        except Exception as exc:                      # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            break
        try:
            v = schema_mod.validate_verdict(raw)
            v["_n_requests"] = used
            return v, None
        except schema_mod.VerdictInvalid as exc:
            last = str(exc)
            if attempt == 0:
                transcript.append({
                    "role": "user",
                    "text": f"That verdict was rejected: {exc}\nReturn a "
                            f"corrected verdict in the same JSON shape."})

    v = schema_mod.abstention(f"no valid verdict after retry: {last}")
    v["_n_requests"] = used
    return v, last


def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
