"""The LLM judge, tested entirely offline.

No API key, no network, no SDK required. The client is injected into
`agent.judge_situation`, so every branch of the loop -- termination, tool
errors, the turn cap, validation, the retry, abstention -- is reachable from a
stub.

Every file read here passes `encoding="utf-8"`. Omitting it has broken this
suite three times (`docs/dev_readme.md` section 7 item 1); on Windows the
default is cp1252.
"""
from __future__ import annotations

import ast
import json
import pathlib

import pytest

from xray.judge import cache, evidence, prompt, report, sampler, schema, tools
from xray.judge.agent import judge_situation
from xray.judge.tools import Context

JUDGE_DIR = pathlib.Path(__file__).resolve().parent.parent / "xray" / "judge"

# The layers that compute. `xray/judge/` may not import any of them.
FORBIDDEN = {
    # SIMULATION
    "sim", "track", "policy", "config",
    # INFERENCE
    "estimator", "realfit", "analysis", "balance", "setmem", "modes",
    "pipeline", "rbpf", "deadband", "strategy", "pooling",
    # DECISION
    "decision", "overtake", "qmdp", "opportunity", "stint",
    "decision_service", "passmodel",
    # SHARED PHYSICS / SCORING
    "vehicle", "tyres", "environment", "physics_context", "constants", "regs",
    "metrics", "closedloop", "observe", "ingest", "circuits",
}


def _judge_files():
    return sorted(JUDGE_DIR.glob("*.py"))


def _imported_names(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[-1])
        elif isinstance(node, ast.Import):
            names.update(a.name.split(".")[-1] for a in node.names)
    return names


@pytest.mark.parametrize("path", _judge_files(), ids=lambda p: p.name)
def test_judge_package_imports_only_stdlib_and_the_sdk(path):
    """The judge cannot compute, because it cannot reach anything that computes.

    Stronger than `test_layers.py`'s INFERENCE-must-not-import-SIMULATION rule,
    and deliberately so: a judge that could call `overtake.p_pass` or
    `vehicle.step` would eventually be asked to, and a verdict would quietly
    become a second, unvalidated physics path that the rest of the stack might
    believe.
    """
    leaked = _imported_names(path) & FORBIDDEN
    assert not leaked, (
        f"xray/judge/{path.name} imports {sorted(leaked)}. The judge reads "
        f"serialized results and computes nothing: every number in a verdict "
        f"must have come out of a JSON file on disk.")


@pytest.mark.parametrize("path", _judge_files(), ids=lambda p: p.name)
def test_the_judge_never_imports_the_api(path):
    src = path.read_text(encoding="utf-8")
    assert "simulation" not in _imported_names(path), (
        f"xray/judge/{path.name} imports from simulation/. The dependency "
        f"direction in this repo is API -> xray; inverting it makes uvicorn "
        f"and the judge circularly dependent.")
    assert "from .." not in src, f"{path.name} escapes the package"


def test_only_the_client_wrapper_imports_the_sdk():
    """If the SDK leaks past `gemini.py`, importing `xray.judge.store` from the
    API starts requiring `google-genai` to be installed to serve a request."""
    for path in _judge_files():
        names = _imported_names(path)
        if path.name == "gemini.py":
            continue
        assert "genai" not in names and "google" not in names, (
            f"xray/judge/{path.name} imports the SDK; only gemini.py may")


def test_judge_paths_agree_with_the_api():
    """`paths.py` re-derives what `simulation/api/main.py` already defines. The
    duplication is deliberate (see paths.py); this is what stops it drifting."""
    main = pytest.importorskip("simulation.api.main")
    from xray.judge import paths
    assert paths.DECISIONS == main.DECISIONS
    assert paths.RACES == main.RACES
    assert paths.DECISIONS_INDEX == main.INDEX


# --------------------------------------------------------------------------
# Redaction -- the most important test in the file
# --------------------------------------------------------------------------

def _bundles_on_disk(limit=None):
    pairs = evidence.pairs_on_disk()
    if not pairs:
        pytest.skip("no bundles in out/decisions/")
    return pairs[:limit] if limit else pairs


def _walk_strings_and_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield ("key", k)
            yield from _walk_strings_and_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings_and_keys(v)


def test_evidence_tools_never_expose_the_outcome():
    """The judge grades whether the evidence supported the call, not whether the
    driver agreed.

    `matches_recommendation` is already computed by
    `decision_service.historical_replay` with no model involved. A judge that
    can see it will grade it, and this evaluator collapses into a worse copy of
    a function that already exists. `actual_gap_s` is in the banned set too: it
    reads like an input and is the observed gap, not the estimated one the
    engine acted on.
    """
    checked = 0
    for race, car, rival in _bundles_on_disk(limit=12):
        bundle = evidence.load_bundle(race, car, rival)
        sits = bundle.get("situations") or []
        times = [s["decision_time_s"] for s in sits[:3]] or [None]
        for t in times:
            ctx = Context(bundle, race, car, rival, t)
            payloads = [ctx.opening_evidence()]
            payloads += [ctx.dispatch(n) for n in tools.TOOL_NAMES]
            payloads.append(ctx.dispatch("get_repo_doctrine",
                                         {"topic": "pass_model"}))
            for p in payloads:
                for kind, value in _walk_strings_and_keys(p):
                    assert value not in evidence.OUTCOME_KEYS, (
                        f"{race} {car}->{rival}@{t}: a tool returned the "
                        f"outcome field {value!r} ({kind}). The judge must "
                        f"never see what happened after the decision point.")
            checked += 1
    assert checked > 0


def test_redact_outcome_is_recursive():
    nested = {"a": {"b": [{"actual_action": "attacked", "keep": 1}]},
              "matches_recommendation": True, "keep": 2}
    out = evidence.redact_outcome(nested)
    assert out == {"a": {"b": [{"keep": 1}]}, "keep": 2}


def test_every_tool_survives_every_bundle_on_disk():
    """Including the 99 bundles with no `refusal` key and the one with a literal
    NaN token (`2026_r10_R__ANT__LEC.json`, gap_age_s).

    Every return must round-trip through `allow_nan=False`, which is what the
    verdict writer uses -- a NaN surviving to that point fails the write after
    the request has already been spent.
    """
    for race, car, rival in _bundles_on_disk():
        bundle = evidence.load_bundle(race, car, rival)
        sits = bundle.get("situations") or []
        t = sits[0]["decision_time_s"] if sits else None
        ctx = Context(bundle, race, car, rival, t)
        json.dumps(ctx.opening_evidence(), allow_nan=False)
        for name in tools.TOOL_NAMES:
            out = ctx.dispatch(name)
            assert isinstance(out, dict)
            json.dumps(out, allow_nan=False)


def test_a_bundle_without_a_refusal_key_does_not_raise():
    """99 of 102 bundles predate the `refusal` key. A `bundle["refusal"]` here
    would crash the loader on 97% of the corpus."""
    assert evidence.bundle_refusal({}, "nope", "A", "B", index={}) is None
    assert evidence.bundle_refusal(
        {}, "r", "A", "B", index={"r__A__B": {"refusal": "declined"}}) == "declined"
    assert evidence.bundle_refusal({"refusal": None}, "r", "A", "B",
                                   index={"r__A__B": {"refusal": "x"}}) is None


def test_literal_nan_in_a_bundle_becomes_none():
    assert evidence._loads('{"gap_age_s": NaN}')["gap_age_s"] is None


def test_key_round_trips_including_the_refusal_sentinel():
    assert evidence.parse_key(evidence.key_of("r1", "HAM", "LEC", 3442.55)) == \
        ("r1", "HAM", "LEC", 3442.55)
    assert evidence.parse_key(evidence.key_of("r1", "HAD", "LAW", None)) == \
        ("r1", "HAD", "LAW", None)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _ok_verdict(**over):
    v = {
        "verdict": "JUSTIFIED",
        "summary": "fine",
        "judge_confidence": "MEDIUM",
        "abstained": False,
        "axes": {a: {"score": 2, "note": "n", "evidence_refs": []}
                 for a in schema.AXES},
        "overclaims": [],
    }
    v.update(over)
    return v


def test_validate_accepts_a_well_formed_verdict():
    out = schema.validate_verdict(_ok_verdict())
    assert out["verdict_label"] and out["verdict_tone"] == "ok"


@pytest.mark.parametrize("axis", sorted(schema.HARD_FAIL_AXES))
def test_validate_rejects_a_contradiction(axis):
    """Scoring an axis 0 says the call rests on a number that does not support
    it. Returning JUSTIFIED in the same breath is self-contradictory, and it is
    the cheapest route to sycophancy: flag everything, endorse everything."""
    v = _ok_verdict()
    v["axes"][axis] = {"score": 0, "note": "n", "evidence_refs": []}
    with pytest.raises(schema.VerdictInvalid, match="requires one of"):
        schema.validate_verdict(v)
    v["verdict"] = "OVERCLAIMED"
    assert schema.validate_verdict(v)["verdict"] == "OVERCLAIMED"


def test_validate_rejects_model_supplied_provenance():
    """`is_measurement: false` is the field standing between a verdict and being
    mistaken for physics. It must never be something the model could set."""
    with pytest.raises(schema.VerdictInvalid, match="filled by the harness"):
        schema.validate_verdict(_ok_verdict(is_measurement=True))
    with pytest.raises(schema.VerdictInvalid, match="filled by the harness"):
        schema.validate_verdict(_ok_verdict(model_id="something-else"))


def test_validate_rejects_unknown_enums_and_missing_axes():
    with pytest.raises(schema.VerdictInvalid):
        schema.validate_verdict(_ok_verdict(verdict="LOOKS_GOOD"))
    with pytest.raises(schema.VerdictInvalid):
        schema.validate_verdict(_ok_verdict(judge_confidence="VERY"))
    v = _ok_verdict()
    v["axes"].pop("gap_provenance")
    with pytest.raises(schema.VerdictInvalid, match="missing"):
        schema.validate_verdict(v)


def test_validate_rejects_an_invented_evidence_reference():
    v = _ok_verdict()
    v["axes"]["bracket_reporting"]["evidence_refs"] = ["get_the_truth"]
    with pytest.raises(schema.VerdictInvalid, match="not tools"):
        schema.validate_verdict(v)


def test_verdicts_are_labelled_generated_commentary():
    v = schema.with_provenance(schema.validate_verdict(_ok_verdict()),
                               model_id="m")
    assert v["is_measurement"] is False
    assert "GENERATED COMMENTARY" in v["label"]


def test_no_verdict_field_can_be_mistaken_for_a_measurement():
    """A verdict must not be confusable with a physics payload by shape alone."""
    v = schema.with_provenance(schema.validate_verdict(_ok_verdict()),
                               model_id="m")
    physics = {"usable_energy_mj", "rival_usable_energy_mj", "cda", "CdA",
               "pass_probability", "deployed_lap", "value_attack", "gap_s"}
    assert not (set(v) & physics)


def test_abstention_is_a_valid_recorded_outcome():
    a = schema.abstention("because")
    assert a["verdict"] == "INSUFFICIENT_EVIDENCE_TO_JUDGE"
    assert a["abstained"] is True and a["abstain_reason"] == "because"


# --------------------------------------------------------------------------
# The agent loop, against a stub client
# --------------------------------------------------------------------------

class StubClient:
    """Scripted tool turns, then a verdict. Counts every request it serves."""

    model_id = "stub-model"

    def __init__(self, turns, verdicts):
        self.turns = list(turns)
        self.verdicts = list(verdicts)
        self.n_tool_turns = 0
        self.n_extracts = 0

    def tool_turn(self, transcript):
        self.n_tool_turns += 1
        if self.turns:
            return self.turns.pop(0), None
        return [], "done"

    def extract(self, transcript):
        self.n_extracts += 1
        if self.verdicts:
            return self.verdicts.pop(0)
        return _ok_verdict()


def _a_real_situation():
    for race, car, rival in _bundles_on_disk():
        bundle = evidence.load_bundle(race, car, rival)
        sits = bundle.get("situations") or []
        if sits:
            return race, car, rival, bundle, sits[0]["decision_time_s"]
    pytest.skip("no bundle with situations")


def test_the_agent_runs_offline_against_a_stub_client():
    race, car, rival, bundle, t = _a_real_situation()
    client = StubClient(
        turns=[[{"name": "get_rival_energy_bracket", "args": {}}],
               [{"name": "get_pass_model_metadata", "args": {}}]],
        verdicts=[_ok_verdict()])
    entry = judge_situation(client, race, car, rival, t, bundle=bundle)
    v = entry["verdict"]

    assert v["verdict"] == "JUSTIFIED"
    assert v["tool_calls"] == ["get_rival_energy_bracket",
                               "get_pass_model_metadata"]
    assert v["is_measurement"] is False
    assert v["model_id"] == "stub-model"
    assert v["prompt_version"] == prompt.PROMPT_VERSION
    assert v["situation_key"] == evidence.key_of(race, car, rival, t)
    assert v["judge_error"] is None
    assert entry["n_requests"] == client.n_tool_turns + client.n_extracts
    json.dumps(entry, allow_nan=False)


def test_tool_error_is_returned_not_raised():
    """One malformed bundle must not kill a batch that has spent a day of quota
    getting there."""
    race, car, rival, bundle, t = _a_real_situation()
    client = StubClient(turns=[[{"name": "no_such_tool", "args": {}}]],
                        verdicts=[_ok_verdict()])
    entry = judge_situation(client, race, car, rival, t, bundle=bundle)
    assert entry["verdict"]["verdict"] == "JUSTIFIED"
    results = [m for m in entry["transcript"] if m.get("role") == "tool"]
    assert "error" in results[0]["results"][0]["result"]


def test_three_consecutive_tool_errors_stop_the_loop_early():
    race, car, rival, bundle, t = _a_real_situation()
    client = StubClient(turns=[[{"name": "nope", "args": {}}]] * 6,
                        verdicts=[_ok_verdict()])
    entry = judge_situation(client, race, car, rival, t, bundle=bundle)
    assert client.n_tool_turns == 3


def test_max_turns_is_enforced():
    race, car, rival, bundle, t = _a_real_situation()
    client = StubClient(
        turns=[[{"name": "get_policy_fan", "args": {}}]] * 50,
        verdicts=[_ok_verdict()])
    entry = judge_situation(client, race, car, rival, t, bundle=bundle,
                            max_turns=3)
    assert client.n_tool_turns == 3
    assert entry["verdict"]["n_turns"] == 3


def test_the_tool_call_budget_is_enforced():
    race, car, rival, bundle, t = _a_real_situation()
    client = StubClient(
        turns=[[{"name": "get_policy_fan", "args": {}},
                {"name": "get_pair_status", "args": {}}]] * 10,
        verdicts=[_ok_verdict()])
    entry = judge_situation(client, race, car, rival, t, bundle=bundle,
                            max_turns=10, max_tool_calls=3)
    assert len(entry["verdict"]["tool_calls"]) <= 3


def test_an_invalid_verdict_is_retried_once_then_abstains():
    race, car, rival, bundle, t = _a_real_situation()
    bad = _ok_verdict(verdict="SPLENDID")
    client = StubClient(turns=[], verdicts=[bad, bad])
    entry = judge_situation(client, race, car, rival, t, bundle=bundle)
    v = entry["verdict"]
    assert client.n_extracts == 2
    assert v["verdict"] == "INSUFFICIENT_EVIDENCE_TO_JUDGE"
    assert v["abstained"] is True
    assert v["judge_error"]


def test_a_retry_that_succeeds_is_recorded_as_the_verdict():
    race, car, rival, bundle, t = _a_real_situation()
    client = StubClient(turns=[],
                        verdicts=[_ok_verdict(verdict="NOPE"),
                                  _ok_verdict(verdict="OVERCLAIMED")])
    v = judge_situation(client, race, car, rival, t, bundle=bundle)["verdict"]
    assert v["verdict"] == "OVERCLAIMED" and v["judge_error"] is None


def test_a_client_exception_becomes_an_abstention_not_a_crash():
    race, car, rival, bundle, t = _a_real_situation()

    class Boom(StubClient):
        def tool_turn(self, transcript):
            raise RuntimeError("network down")

    v = judge_situation(Boom([], []), race, car, rival, t,
                        bundle=bundle)["verdict"]
    assert v["verdict"] == "INSUFFICIENT_EVIDENCE_TO_JUDGE"
    assert "network down" in v["judge_error"]


def test_a_refused_pair_is_judged_at_pair_level():
    """A refusal is a correct output and gets judged, not skipped."""
    refused = None
    for race, car, rival in _bundles_on_disk():
        b = evidence.load_bundle(race, car, rival)
        if evidence.bundle_refusal(b, race, car, rival):
            refused = (race, car, rival, b)
            break
    if refused is None:
        pytest.skip("no refused bundle on disk")
    race, car, rival, bundle = refused
    client = StubClient(turns=[[{"name": "get_pair_status", "args": {}}],
                               [{"name": "get_gap_provenance", "args": {}}]],
                        verdicts=[_ok_verdict(verdict="CORRECTLY_REFUSED")])
    entry = judge_situation(client, race, car, rival, None, bundle=bundle)
    assert entry["verdict"]["verdict"] == "CORRECTLY_REFUSED"
    assert entry["situation_key"].endswith("@refusal")
    # The per-lap tools have nothing to answer with and must say so rather than
    # inventing a row.
    results = [m for m in entry["transcript"] if m.get("role") == "tool"]
    assert "error" in results[1]["results"][0]["result"]


# --------------------------------------------------------------------------
# Cache and versioning
# --------------------------------------------------------------------------

def _key(**over):
    base = dict(situation_key="k", artefact_mtime_ns=1, evidence_fp="fp",
                model_id="m", prompt_version="p", rubric_version="r",
                judge_version="j")
    base.update(over)
    return cache.cache_key(**base)


@pytest.mark.parametrize("field", ["situation_key", "artefact_mtime_ns",
                                   "evidence_fp", "model_id", "prompt_version",
                                   "rubric_version", "judge_version"])
def test_cache_key_changes_with_every_component(field):
    assert _key() != _key(**{field: "CHANGED"})


def test_cache_key_is_stable_for_identical_inputs():
    assert _key() == _key()


def test_prompt_version_is_the_hash_of_the_current_prompt():
    """Editing the prompt without this moving would pool verdicts from two
    rubrics under one headline number, with nothing recording that it had
    happened."""
    assert prompt.PROMPT_VERSION == "p" + prompt._hash()
    assert prompt.PROMPT_VERSION.startswith("p") and len(prompt.PROMPT_VERSION) == 13


def test_the_prompt_states_what_is_not_being_judged():
    text = prompt.system_instruction()
    assert "NOT whether the driver did the same thing" in text
    for axis in schema.AXES:
        assert axis in text, f"rubric does not define {axis}"


def test_doctrine_topics_are_all_declared_in_the_tool_enum():
    decl = next(d for d in tools.TOOL_DECLARATIONS
                if d["name"] == "get_repo_doctrine")
    assert decl["parameters"]["properties"]["topic"]["enum"] == sorted(
        evidence.DOCTRINE)


def test_an_unknown_doctrine_topic_returns_the_menu_not_an_error_state():
    out = evidence.repo_doctrine("nonsense")
    assert "valid_topics" in out and out["valid_topics"]


def test_tool_declarations_match_the_dispatch_and_the_validator():
    assert set(tools.TOOL_NAMES) <= set(schema.TOOL_NAMES)
    assert "headline" in schema.TOOL_NAMES


# --------------------------------------------------------------------------
# Degradation with no key, no SDK, no network
# --------------------------------------------------------------------------

def test_absent_sdk_does_not_break_the_import():
    """A hard SDK import here would make `uvicorn` startup and `pytest` depend
    on `google-genai` being installed, on checkouts that never run the judge."""
    from xray.judge import gemini
    assert hasattr(gemini, "JudgeUnavailable")
    assert gemini.DEFAULT_MODEL == "gemini-2.5-flash-lite"


def test_missing_api_key_degrades_to_a_typed_refusal(monkeypatch):
    """`scripts/02.run_estimator.py` raising an unhandled traceback on a correct
    refusal is a known wart in this repo. Not repeated: no key is a stated
    refusal with the thing to do next in it."""
    from xray.judge import gemini
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(gemini.JudgeUnavailable) as exc:
        gemini.make_client()
    msg = str(exc.value)
    assert "GEMINI_API_KEY" in msg or "google-genai" in msg
    assert "--dry-run" in msg or "not installed" in msg


def test_the_limiter_stops_cleanly_instead_of_overrunning_a_budget(tmp_path,
                                                                   monkeypatch):
    from xray.judge import limiter as lim
    monkeypatch.setattr(lim, "QUOTA", tmp_path / "_quota.json")
    L = lim.Limiter(rpm=1000, rpd=1000, max_requests=3, sleep=lambda s: None)
    for _ in range(3):
        L.acquire()
    assert L.remaining_this_run == 0
    with pytest.raises(lim.QuotaExhausted):
        L.acquire()


def test_the_daily_counter_persists_across_limiters(tmp_path, monkeypatch):
    """Two runs on the same day share one budget instead of each believing it
    has the whole thing."""
    from xray.judge import limiter as lim
    monkeypatch.setattr(lim, "QUOTA", tmp_path / "_quota.json")
    a = lim.Limiter(rpm=1000, rpd=10, sleep=lambda s: None)
    for _ in range(4):
        a.acquire()
    b = lim.Limiter(rpm=1000, rpd=10, sleep=lambda s: None)
    assert b.remaining_today == 6


# --------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------

def test_sampler_is_reproducible_under_a_fixed_seed():
    a = sampler.build_sample(n=40, seed=7)
    b = sampler.build_sample(n=40, seed=7)
    assert a["sample_id"] == b["sample_id"]
    assert a["keys"] == b["keys"]


def test_a_different_seed_draws_a_different_sample():
    a = sampler.build_sample(n=40, seed=7)
    b = sampler.build_sample(n=40, seed=8)
    assert a["keys"] != b["keys"]


def test_refused_pairs_are_censused_not_sampled():
    """A refusal is a correct output. Sampling them out would bias the corpus
    toward the cases where the engine was willing to speak."""
    m = sampler.build_sample(n=20, seed=42)
    refusal_keys = [k for k in m["keys"] if k.endswith("@refusal")]
    assert len(refusal_keys) == m["n_refusals_censused"]
    big = sampler.build_sample(n=400, seed=42)
    assert len([k for k in big["keys"] if k.endswith("@refusal")]) == \
        m["n_refusals_censused"]


def test_every_race_and_both_calls_appear_in_a_default_sample():
    m = sampler.build_sample(n=240, seed=42)
    races = {c.split("|")[0] for c in m["counts"]}
    calls = {c.split("|")[1] for c in m["counts"]}
    agreements = {c.split("|")[2] for c in m["counts"]}
    assert len(races) >= 10
    assert calls == {"ATTACK", "HOLD"}
    assert {"matched", "diverged", "unresolved"} <= agreements


def test_sample_id_changes_when_the_decisions_index_changes(monkeypatch):
    a = sampler.build_sample(n=20, seed=42)
    monkeypatch.setattr(sampler, "_index_fingerprint", lambda: "deadbeef")
    b = sampler.build_sample(n=20, seed=42)
    assert a["sample_id"] != b["sample_id"]


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _entry(key, verdict="JUSTIFIED", summary="", transcript=None):
    v = schema.with_provenance(
        schema.validate_verdict(_ok_verdict(verdict=verdict, summary=summary)),
        model_id="m", judge_version="j", stack="real", situation_key=key,
        judge_error=None, generated_at="2026-09-13T00:00:00+00:00",
        artefact_mtime_ns=1, tool_calls=[], n_turns=1)
    return {"situation_key": key, "verdict": v,
            "transcript": transcript or [], "n_requests": 1}


def test_report_names_the_stack_and_labels_the_diagnostic():
    m = sampler.build_sample(n=8, seed=42)
    entries = [_entry(k) for k in m["keys"][:5]]
    r = report.build_report(entries, m, "run-1", model_id="m")
    md = report.render_markdown(r)
    assert "realfit" in md and "decision_service" in md
    assert report.DIAGNOSTIC_HEADING in md
    assert "not a score" in md
    # And the expectation that low scores on these two axes are the healthy
    # outcome, not a failure of the engine.
    assert "rubber-stamping" in md


def test_the_report_flags_numbers_the_transcript_never_contained():
    e = _entry("2026_r1_R__HAM__LEC@3780.14",
               summary="the bracket was 4.71 MJ wide")
    bad = report._unattributed_numbers(e)
    assert "4.71" in bad


def test_a_number_present_in_the_transcript_is_not_flagged():
    e = _entry("2026_r1_R__HAM__LEC@3780.14", summary="bracket 1.84 MJ",
               transcript=[{"role": "tool", "results": [
                   {"name": "get_rival_energy_bracket",
                    "result": {"rival_usable_p90_mj": 1.84}}]}])
    assert "1.84" not in report._unattributed_numbers(e)
