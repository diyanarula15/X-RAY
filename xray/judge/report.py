"""Aggregation. Standard library only.

Two rules shape this file:

1. The headline names the stack that produced the numbers. Stage 1, the
   set-membership group and the real path are three different baselines and
   have been quoted interchangeably in this repo before.
2. Agreement with `matches_recommendation` is printed as a DIAGNOSTIC under a
   heading that says so. The judge never saw that field. "Did the evidence
   justify the call" and "did the driver do the same thing" are different
   questions, and optimising the prompt toward the second would convert this
   evaluator into a second, worse copy of `decision_service.historical_replay`.
"""
from __future__ import annotations

import json
import re

from . import STACK
from . import evidence as ev
from .paths import LATEST_REPORT, REPORTS, write_atomic
from .schema import AXES

DIAGNOSTIC_HEADING = "Diagnostic, not a target"

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _outcome_lookup(keys) -> dict:
    """`matches_recommendation` per key, read here and never handed to a model."""
    out = {}
    wanted: dict[tuple[str, str, str], list] = {}
    for k in keys:
        race, car, rival, t = ev.parse_key(k)
        wanted.setdefault((race, car, rival), []).append((k, t))
    for (race, car, rival), items in wanted.items():
        try:
            bundle = ev.load_bundle(race, car, rival)
        except ev.EvidenceError:
            continue
        by_t = {round(float(s["decision_time_s"]), 2): s
                for s in bundle.get("situations") or []
                if s.get("decision_time_s") is not None}
        for k, t in items:
            if t is None:
                out[k] = "refused"
                continue
            s = by_t.get(round(t, 2))
            m = s.get("matches_recommendation") if s else None
            out[k] = ("matched" if m is True
                      else "diverged" if m is False else "unresolved")
    return out


def _unattributed_numbers(entry: dict) -> list[str]:
    """Numbers in the prose that appear nowhere in the tool transcript.

    A cheap regex, no model involved. A non-zero count is a defect in the judge
    and is reported as one rather than quietly dropped -- it is the one
    automatic check on whether the verdict's prose is grounded in the evidence
    it was shown.
    """
    seen = json.dumps(entry.get("transcript") or [], allow_nan=False)
    pool = set(_NUM.findall(seen))
    v = entry.get("verdict") or {}
    prose = [v.get("summary") or ""]
    prose += [a.get("note") or "" for a in (v.get("axes") or {}).values()]
    prose += [o.get("why_wrong") or "" for o in v.get("overclaims") or []]
    bad = []
    for text in prose:
        for tok in _NUM.findall(text):
            # A bare small integer is almost always a score or a lap count, not
            # a claimed measurement; flagging those would drown the signal.
            if tok in pool or (tok.isdigit() and len(tok) <= 2):
                continue
            bad.append(tok)
    return bad


def build_report(entries: list[dict], manifest: dict, run_id: str,
                 *, model_id=None, requests_used=0, cache_hits=0,
                 wall_s=0.0) -> dict:
    keys = [e.get("situation_key") for e in entries if e.get("situation_key")]
    outcomes = _outcome_lookup(keys)

    verdict_hist: dict[str, int] = {}
    axis_dist = {a: {0: 0, 1: 0, 2: 0} for a in AXES}
    overclaim_fields: dict[str, int] = {}
    crosstab: dict[str, dict[str, int]] = {}
    variants: dict[str, int] = {}
    refusals = []
    unattributed = 0
    unattributed_examples: list[str] = []
    n_abstained = n_errored = 0

    for e in entries:
        v = e.get("verdict") or {}
        name = v.get("verdict") or "MISSING"
        verdict_hist[name] = verdict_hist.get(name, 0) + 1
        if v.get("abstained"):
            n_abstained += 1
        if v.get("judge_error"):
            n_errored += 1
        for axis, ax in (v.get("axes") or {}).items():
            if axis in axis_dist and ax.get("score") in (0, 1, 2):
                axis_dist[axis][ax["score"]] += 1
        for oc in v.get("overclaims") or []:
            f = oc.get("field") or "?"
            overclaim_fields[f] = overclaim_fields.get(f, 0) + 1
        key = e.get("situation_key") or ""
        bucket = outcomes.get(key, "unresolved")
        crosstab.setdefault(name, {})
        crosstab[name][bucket] = crosstab[name].get(bucket, 0) + 1
        if bucket == "refused":
            refusals.append({"key": key, "verdict": name})
        race = key.split("__")[0] if key else ""
        try:
            var = (ev.load_race_meta(race).get("regulation") or {}).get("variant")
        except ev.EvidenceError:
            var = None
        variants[var or "unknown"] = variants.get(var or "unknown", 0) + 1
        bad = _unattributed_numbers(e)
        unattributed += len(bad)
        unattributed_examples += bad[:2]

    axis_means = {
        a: (round(sum(k * n for k, n in d.items()) / max(1, sum(d.values())), 2)
            if sum(d.values()) else None)
        for a, d in axis_dist.items()}

    n_correct_refusals = sum(1 for r in refusals
                             if r["verdict"] == "CORRECTLY_REFUSED")

    return {
        "run_id": run_id,
        "stack": STACK,
        "sample_id": manifest.get("sample_id"),
        "seed": manifest.get("seed"),
        "model_id": model_id,
        "prompt_version": manifest.get("prompt_version"),
        "judge_version": manifest.get("judge_version"),
        "requests_used": requests_used,
        "cache_hits": cache_hits,
        "cache_hit_rate": (round(cache_hits / len(entries), 3)
                           if entries else None),
        "wall_s": round(wall_s, 1),
        "n_sampled": manifest.get("n_selected"),
        "n_judged": len(entries),
        "n_abstained": n_abstained,
        "n_errored": n_errored,
        "regulation_variants": variants,
        "verdict_histogram": verdict_hist,
        "axis_distribution": {a: {str(k): v for k, v in d.items()}
                              for a, d in axis_dist.items()},
        "axis_means": axis_means,
        "overclaim_fields": dict(sorted(overclaim_fields.items(),
                                        key=lambda kv: -kv[1])),
        "refusals": {"n": len(refusals),
                     "n_correctly_refused": n_correct_refusals,
                     "items": refusals},
        "diagnostic_crosstab": crosstab,
        "unattributed_numeric_claims": {
            "n": unattributed,
            "examples": unattributed_examples[:10]},
        "strata_sampled": manifest.get("counts"),
        "strata_judged": _strata_judged(keys, manifest),
    }


def _strata_judged(keys, manifest) -> dict:
    """Per-stratum judged counts, so a stratum the quota ran out on is reported
    as under-covered rather than quietly dropped."""
    sampled = set(manifest.get("keys") or [])
    judged = set(keys)
    return {"n_sampled": len(sampled), "n_judged": len(judged & sampled),
            "n_unjudged": len(sampled - judged)}


def render_markdown(r: dict) -> str:
    L = []
    L.append(f"# LLM-judge report {r['run_id']}\n")
    L.append(f"**Scored: {r['stack']}**, via precomputed `out/decisions/` "
             f"bundles. Not the Stage 1 estimator, not the set-membership "
             f"research group -- neither produced any of these numbers.\n")
    L.append(f"Model `{r.get('model_id')}` | prompt `{r.get('prompt_version')}` "
             f"| judge `{r.get('judge_version')}` | sample "
             f"`{r.get('sample_id')}` seed {r.get('seed')}\n")
    L.append(f"{r['n_judged']} judged of {r['n_sampled']} sampled "
             f"({r['n_abstained']} abstained, {r['n_errored']} errored). "
             f"{r['requests_used']} requests, {r['cache_hits']} cache hits, "
             f"{r['wall_s']} s.\n")
    L.append("Regulation variants in the sample: "
             + ", ".join(f"{k} {v}" for k, v in sorted(
                 r["regulation_variants"].items())) + "\n")

    L.append("\n## Verdicts\n")
    for k, v in sorted(r["verdict_histogram"].items(), key=lambda kv: -kv[1]):
        L.append(f"- {k}: {v}")

    L.append("\n## Rubric axes\n")
    L.append("`pass_model_honesty` and `bracket_reporting` are EXPECTED to "
             "score low often: every lap row in this corpus carries "
             "`calibration: \"placeholder\"`, and the rival energy bracket is "
             "routinely wider than the margin a call turns on. A high mean on "
             "these two is evidence the judge is rubber-stamping, not evidence "
             "the decision engine is good.\n")
    L.append("| axis | mean | 0 | 1 | 2 |")
    L.append("|---|---|---|---|---|")
    for a in AXES:
        d = r["axis_distribution"][a]
        L.append(f"| {a} | {r['axis_means'][a]} | {d['0']} | {d['1']} | {d['2']} |")

    if r["overclaim_fields"]:
        L.append("\n## Most-flagged fields\n")
        for f, n in list(r["overclaim_fields"].items())[:10]:
            L.append(f"- `{f}`: {n}")

    ref = r["refusals"]
    L.append(f"\n## Refusals\n\n{ref['n']} refused pairs judged, "
             f"{ref['n_correctly_refused']} as CORRECTLY_REFUSED. A refusal is "
             f"a correct output; these are censused, never sampled out.\n")

    L.append(f"\n## {DIAGNOSTIC_HEADING}\n")
    L.append("The judge never saw `matches_recommendation`; it was withheld "
             "from every tool and every prompt "
             "(`tests/test_judge.py::test_evidence_tools_never_expose_the_outcome`). "
             "This table asks only whether \"the evidence justified the call\" "
             "and \"the driver did the same thing\" happen to move together. "
             "They are different questions. A high agreement rate is not a "
             "score, and tuning the prompt toward it would turn this into a "
             "second, worse copy of `decision_service.historical_replay`.\n")
    buckets = sorted({b for row in r["diagnostic_crosstab"].values()
                      for b in row})
    L.append("| verdict | " + " | ".join(buckets) + " |")
    L.append("|---" * (len(buckets) + 1) + "|")
    for name, row in sorted(r["diagnostic_crosstab"].items()):
        L.append(f"| {name} | " + " | ".join(str(row.get(b, 0))
                                             for b in buckets) + " |")

    u = r["unattributed_numeric_claims"]
    L.append(f"\n## Judge defects\n\n{u['n']} numeric tokens appear in verdict "
             f"prose but nowhere in the tool transcript that produced it.")
    if u["examples"]:
        L.append(f"Examples: {', '.join(u['examples'])}")
    L.append("\nThis is a defect in the judge, reported rather than hidden.\n")
    return "\n".join(L)


def write_report(report: dict) -> tuple:
    blob = json.dumps(report, indent=1, allow_nan=False)
    j = REPORTS / f"{report['run_id']}.json"
    m = REPORTS / f"{report['run_id']}.md"
    write_atomic(j, blob)
    write_atomic(m, render_markdown(report))
    write_atomic(LATEST_REPORT, blob)
    return j, m
