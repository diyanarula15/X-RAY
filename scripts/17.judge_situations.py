#!/usr/bin/env python3
"""Judge the precomputed decision situations with an LLM, into `out/judge/`.

Why this exists: `out/decisions/` holds 5,759 causal decision points across ten
real 2026 races, each carrying X-RAY's own ATTACK/HOLD call and the evidence
that produced it -- energy brackets with p10/p90, an uncalibrated pass
probability, a gap whose confidence is often 0.2, DP values for attack against
wait. Nothing in the repo measures whether those calls were JUSTIFIED BY THAT
EVIDENCE. `matches_recommendation` only says whether the driver happened to do
the same thing, which is a different question and a poor target: a driver can
ignore a sound call, and a coin-flip call can match by luck.

This runs a tool-calling agent over a stratified sample and records a structured
verdict per situation. The judge is never shown what the driver did.

Unlike `scripts/16`, this is single-worker on purpose. A process pool against a
per-minute request cap manufactures 429s and nothing else.

    python scripts/17.judge_situations.py --dry-run     # cost projection, no calls
    python scripts/17.judge_situations.py --race 2026_r10_R --n 20
    python scripts/17.judge_situations.py --materialize  # rebuild views from cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray.judge import JUDGE_VERSION, RUBRIC_VERSION  # noqa: E402
from xray.judge import cache, evidence, report, sampler, store  # noqa: E402
from xray.judge.agent import judge_situation  # noqa: E402
from xray.judge.limiter import (DEFAULT_RPD, DEFAULT_RPM, Limiter,  # noqa: E402
                                QuotaExhausted)
from xray.judge.paths import REPORTS  # noqa: E402
from xray.judge.prompt import PROMPT_VERSION  # noqa: E402

# Measured on the agentic loop: two to four tool turns plus one extraction.
REQUESTS_PER_SITUATION = 4


def _dry_run(manifest: dict, rpd: int) -> int:
    print(f"sample {manifest['sample_id']}  seed {manifest['seed']}  "
          f"{manifest['n_selected']} of {manifest['population_size']} judgeable "
          f"items ({manifest['n_refusals_censused']} refused pairs censused)")
    print(f"prompt {manifest['prompt_version']}  judge {manifest['judge_version']}")
    print("\nper-stratum allocation (race|call|agreement|refused):")
    for cell, n in sorted(manifest["counts"].items()):
        print(f"  {cell:<46} {n}")
    est = manifest["n_selected"] * REQUESTS_PER_SITUATION
    print(f"\nprojected ~{est} requests at ~{REQUESTS_PER_SITUATION}/situation")
    print(f"at {rpd} requests/day that is ~{est / max(1, rpd):.1f} day(s)")
    print("no API calls made")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--race", action="append", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--rpm", type=int, default=DEFAULT_RPM)
    ap.add_argument("--rpd", type=int, default=DEFAULT_RPD)
    ap.add_argument("--max-requests", type=int, default=900)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true",
                    help="write the sample manifest and project the cost; no "
                         "API calls")
    ap.add_argument("--force", action="store_true", help="ignore cached verdicts")
    ap.add_argument("--resume", default=None, metavar="SAMPLE_ID")
    ap.add_argument("--materialize", action="store_true",
                    help="rebuild out/judge/verdicts/ from the cache, no network")
    ap.add_argument("--report", default=None, metavar="RUN_ID",
                    help="rebuild a report from the cache, no network")
    ap.add_argument("--full", action="store_true",
                    help="every situation on disk, not a sample")
    ap.add_argument("--yes", action="store_true", help="confirm --full")
    a = ap.parse_args()

    if a.materialize:
        idx = store.materialize(cache.entries())
        n = sum(v["n_verdicts"] for v in idx.values())
        print(f"materialised {n} verdicts over {len(idx)} pairs")
        return 0

    if a.report:
        entries = list(cache.entries())
        manifest = (sampler.load_sample(a.resume) if a.resume
                    else {"sample_id": None, "seed": a.seed,
                          "n_selected": len(entries),
                          "prompt_version": PROMPT_VERSION,
                          "judge_version": JUDGE_VERSION, "counts": {},
                          "keys": [e["situation_key"] for e in entries]})
        r = report.build_report(entries, manifest, a.report,
                                model_id=(entries[0]["verdict"]["model_id"]
                                          if entries else None),
                                cache_hits=len(entries))
        j, m = report.write_report(r)
        print(f"wrote {j.name} and {m.name} from {len(entries)} cached verdicts")
        return 0

    if a.resume:
        manifest = sampler.load_sample(a.resume)
    else:
        manifest = sampler.build_sample(n=a.n, seed=a.seed, races=a.race,
                                        full=a.full)
        sampler.write_sample(manifest)

    if a.full and not a.yes and not a.dry_run:
        est = manifest["n_selected"] * REQUESTS_PER_SITUATION
        print(f"--full selects {manifest['n_selected']} items, ~{est} requests, "
              f"~{est / max(1, a.rpd):.0f} days at {a.rpd}/day. "
              f"Re-run with --yes to proceed.")
        return 2

    if a.dry_run:
        return _dry_run(manifest, a.rpd)

    # The client is constructed only now, so --dry-run, --materialize and
    # --report never need a key, an SDK or a network.
    from xray.judge.gemini import JudgeUnavailable, make_client
    limiter = Limiter(rpm=a.rpm, rpd=a.rpd, max_requests=a.max_requests)
    try:
        client = make_client(a.model, limiter)
    except JudgeUnavailable as exc:
        print(exc)
        return 2

    keys = manifest["keys"]
    print(f"{len(keys)} items, sample {manifest['sample_id']}, "
          f"model {client.model_id}, budget {limiter.remaining_this_run} requests")

    t0 = time.time()
    entries, done, failed, hits = [], 0, 0, 0
    stopped = None
    for key in keys:
        done += 1
        race, car, rival, t = evidence.parse_key(key)
        try:
            bundle = evidence.load_bundle(race, car, rival)
        except evidence.EvidenceError as exc:
            failed += 1
            print(f"  [{done}/{len(keys)}] {key:<44}    0.0s  FAILED {exc}")
            continue

        ctx_open = None
        ck = None
        try:
            from xray.judge.tools import Context
            ctx_open = Context(bundle, race, car, rival, t).opening_evidence()
            ck = cache.cache_key(
                situation_key=key,
                artefact_mtime_ns=bundle.get("artefact_mtime_ns"),
                evidence_fp=cache.evidence_fingerprint(ctx_open),
                model_id=client.model_id, prompt_version=PROMPT_VERSION,
                rubric_version=RUBRIC_VERSION, judge_version=JUDGE_VERSION)
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print(f"  [{done}/{len(keys)}] {key:<44}    0.0s  "
                  f"FAILED {type(exc).__name__}: {exc}")
            continue

        if not a.force:
            hit = cache.get(ck)
            if hit is not None:
                hits += 1
                entries.append(hit)
                print(f"  [{done}/{len(keys)}] {key:<44}    0.0s  cached "
                      f"{hit['verdict']['verdict']}")
                continue

        t1 = time.time()
        try:
            entry = judge_situation(client, race, car, rival, t, bundle=bundle,
                                    max_turns=a.max_turns)
        except QuotaExhausted as exc:
            stopped = str(exc)
            done -= 1
            break
        except Exception as exc:                      # noqa: BLE001
            # One bad item must not kill a batch that has spent hours of a daily
            # cap getting here.
            failed += 1
            print(f"  [{done}/{len(keys)}] {key:<44} {time.time() - t1:6.1f}s  "
                  f"FAILED {type(exc).__name__}: {exc}")
            continue

        cache.put(ck, entry)
        entries.append(entry)
        v = entry["verdict"]
        print(f"  [{done}/{len(keys)}] {key:<44} {time.time() - t1:6.1f}s  "
              f"{v['verdict']} ({entry['n_requests']} req)")

    store.materialize(cache.entries())
    run_id = f"{manifest['sample_id']}-{int(t0)}"
    r = report.build_report(entries, manifest, run_id,
                            model_id=client.model_id,
                            requests_used=limiter.used_this_run,
                            cache_hits=hits, wall_s=time.time() - t0)
    j, m = report.write_report(r)

    print(f"\n{len(entries)} judged ({hits} cached, {failed} failed) in "
          f"{time.time() - t0:.0f}s, {limiter.used_this_run} requests")
    if stopped:
        left = len(keys) - done
        print(f"stopped on quota: {stopped}. {left} items remain -- re-run with "
              f"--resume {manifest['sample_id']} to continue where this stopped.")
    print(f"report: {j}")
    for k, v in sorted(r["verdict_histogram"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<32} {v}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
