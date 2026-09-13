"""The only file in this package that touches the SDK.

Confined here so that `xray.judge.store` -- what the API imports -- needs
nothing but the standard library, and so `pytest` runs with no key, no network
and no `google-genai` installed. The import is guarded rather than assumed:
`tests/test_judge.py::test_absent_sdk_does_not_break_the_import` asserts the
module still imports when the SDK is missing, because a hard import here would
make `uvicorn` startup depend on a package the API never calls.
"""
from __future__ import annotations

import json
import os

from .limiter import Limiter, backoff_delay
from .prompt import system_instruction
from .schema import VERDICT_SCHEMA
from .tools import TOOL_DECLARATIONS

try:                                             # pragma: no cover - import guard
    from google import genai
    from google.genai import types as genai_types
except ImportError:                              # pragma: no cover
    genai = None
    genai_types = None

DEFAULT_MODEL = "gemini-2.5-flash-lite"

# 429 gets the long jittered backoff; 5xx gets a short one; every other 4xx is
# a request the same request will not fix, so it is recorded against the
# situation and the batch moves on.
MAX_429_ATTEMPTS = 5
MAX_5XX_ATTEMPTS = 3


class JudgeUnavailable(Exception):
    """No key, or no SDK. A typed refusal, not a traceback: `scripts/02` raising
    on a correct refusal is a known wart in this repo and is not repeated."""


def make_client(model: str | None = None, limiter: Limiter | None = None):
    if genai is None:
        raise JudgeUnavailable(
            "google-genai is not installed. pip install -r requirements.txt")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise JudgeUnavailable(
            "GEMINI_API_KEY is not set. The judge declines rather than "
            "guessing -- set the key, or run --dry-run / --materialize / "
            "--report, which need no network.")
    model_id = model or os.environ.get("XRAY_JUDGE_MODEL") or DEFAULT_MODEL
    return GeminiClient(genai.Client(api_key=key), model_id,
                        limiter or Limiter())


class GeminiClient:
    """Two calls, deliberately kept apart.

    `tool_turn` attaches the tool declarations and no response schema;
    `extract` attaches the response schema and no tools. With both attached the
    model routinely emits its verdict as one more function call, which never
    validates and costs a retry. One extra request per situation buys a
    schema-shaped object every time.
    """

    def __init__(self, client, model_id: str, limiter: Limiter):
        self._client = client
        self.model_id = model_id
        self.limiter = limiter

    # -- transcript -> SDK contents ---------------------------------------
    def _contents(self, transcript: list[dict]) -> list:
        T = genai_types
        out = []
        for msg in transcript:
            role = msg.get("role")
            if role == "user":
                out.append(T.Content(role="user",
                                     parts=[T.Part(text=msg["text"])]))
            elif role == "model" and msg.get("calls"):
                out.append(T.Content(role="model", parts=[
                    T.Part(function_call=T.FunctionCall(
                        name=c["name"], args=c.get("args") or {}))
                    for c in msg["calls"]]))
            elif role == "model":
                out.append(T.Content(role="model",
                                     parts=[T.Part(text=msg.get("text") or "")]))
            elif role == "tool":
                out.append(T.Content(role="user", parts=[
                    T.Part.from_function_response(
                        name=r["name"], response=r["result"])
                    for r in msg["results"]]))
        return out

    def _generate(self, contents, config):
        """One request, rate-limited and retried. Raises on final failure; the
        agent turns that into a recorded abstention."""
        import time
        last = None
        for attempt in range(MAX_429_ATTEMPTS):
            self.limiter.acquire(1)
            try:
                return self._client.models.generate_content(
                    model=self.model_id, contents=contents, config=config)
            except Exception as exc:                  # noqa: BLE001
                last = exc
                status = getattr(exc, "code", None) or getattr(
                    exc, "status_code", None)
                text = str(exc)
                if status == 429 or "RESOURCE_EXHAUSTED" in text or "429" in text:
                    time.sleep(backoff_delay(attempt))
                    continue
                if (status and 500 <= int(status) < 600) or "INTERNAL" in text:
                    if attempt >= MAX_5XX_ATTEMPTS - 1:
                        raise
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise
        raise last

    def tool_turn(self, transcript: list[dict]):
        T = genai_types
        cfg = T.GenerateContentConfig(
            system_instruction=system_instruction(),
            temperature=0.0,
            tools=[T.Tool(function_declarations=TOOL_DECLARATIONS)],
            # Manual dispatch: the SDK's automatic mode would call the tools
            # itself and hide the transcript, and the transcript is the artefact
            # the report's unattributed-number check reads.
            automatic_function_calling=T.AutomaticFunctionCallingConfig(
                disable=True),
        )
        resp = self._generate(self._contents(transcript), cfg)
        calls = [{"name": c.name, "args": dict(c.args or {})}
                 for c in (resp.function_calls or [])]
        return calls, (None if calls else _text_of(resp))

    def extract(self, transcript: list[dict]) -> dict:
        T = genai_types
        contents = self._contents(transcript) + [
            T.Content(role="user", parts=[T.Part(text=(
                "Now return your verdict as JSON matching the required schema. "
                "Do not call any tool."))])]
        cfg = T.GenerateContentConfig(
            system_instruction=system_instruction(),
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=VERDICT_SCHEMA,
        )
        resp = self._generate(contents, cfg)
        return json.loads(_text_of(resp) or "{}")


def _text_of(resp) -> str:
    text = getattr(resp, "text", None)
    if text:
        return text
    parts = []
    for cand in getattr(resp, "candidates", None) or []:
        for part in getattr(getattr(cand, "content", None), "parts", None) or []:
            if getattr(part, "text", None):
                parts.append(part.text)
    return "".join(parts)
