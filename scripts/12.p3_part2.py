#!/usr/bin/env python3
"""P3 Part 2 real predictive validation artifact."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from xray.corpus import fingerprint_json  # noqa: E402
from xray.p3_validation import build_real_validation, load_payload_paths  # noqa: E402
from xray.passmodel import (audit_dataset, build_dataset, dataset_fingerprint,
                            fit as fit_pass_model)  # noqa: E402


def _load_corpus(path: Path) -> dict:
    return json.loads(path.read_text())


def _pass_audit(payloads: list[dict]) -> dict:
    samples = []
    for p in payloads:
        samples.extend(build_dataset(p))
    audit = audit_dataset(samples, payloads)
    result = {
        "feature_schema": "p2-pass-features-v1",
        "dataset_fingerprint": dataset_fingerprint(samples),
        "raw_opportunities": audit.n_raw,
        "excluded_opportunities": audit.n_excluded,
        "usable_opportunities": audit.n_usable,
        "positive_passes": audit.n_positive,
        "positive_rate": audit.positive_rate,
        "race_count": len(audit.races),
        "races": audit.races,
        "circuits": audit.circuits,
        "exclusions": audit.exclusions,
        "available_features": audit.available_features,
        "missing_signals": audit.missing_signals,
        "blocking_reasons": audit.blocking_reasons,
        "contamination": {
            "can_remove": [
                "compound changes",
                "TyreLife resets/decreases",
                "missing confirmation-window position data",
            ],
            "cannot_remove": [
                "SC/VSC/yellow effects because track status is unavailable",
                "same-compound pit cycles with stale/missing TyreLife",
                "pit timing because pit data is unavailable in current payloads",
            ],
        },
    }
    try:
        fit_pass_model(samples, audit)
        result["pass_model_result"] = "EMPIRICAL — ACTIVATED"
    except RuntimeError as exc:
        result["pass_model_result"] = "SYNTHETIC — EMPIRICAL FIT REFUSED"
        result["fit_refusal"] = str(exc)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "out" / "p3" / "corpus"
                                            / "20260912T200657Z" / "manifest.json"))
    ap.add_argument("--out-root", default=str(ROOT / "out" / "p3" / "part2"))
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    corpus = _load_corpus(corpus_path)
    payload_paths = [s["artifact_path"] for s in corpus["sessions"]]
    payloads = load_payload_paths(payload_paths)

    validation = build_real_validation(payloads)
    pass_audit = _pass_audit(payloads)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_manifest": str(corpus_path),
        "corpus_fingerprint": corpus["manifest_fingerprint"],
        "snapshot_schema_version": validation["snapshot_schema_version"],
        "split_definition": validation["split"],
        "real_prediction": validation,
        "energy_ablation": {
            "models": {
                "xray": "canonical realfit-derived usable energy at cutoff",
                "fixed_energy": "mean inferred energy by target over training races only",
                "energy_neutral": "zero stored deployable energy",
            },
            "paired_metrics": {
                k: v["paired"] for k, v in validation["metrics"].items()
            },
        },
        "nuisance_sensitivity": validation["nuisance_sensitivity"],
        "pass_dataset_audit": pass_audit,
        "pass_model_result": pass_audit["pass_model_result"],
        "physical_calibration_status": {
            "tyres": {
                "result": "TYRES REMAIN SYNTHETIC",
                "reason": ("compound and TyreLife are present, but stint and FreshTyre "
                           "are unavailable; TyreLife is age, not physical wear, so "
                           "the five-race corpus cannot isolate a reduced wear subset"),
            },
            "cda": {
                "result": "identified sets retained; not relabelled calibrated",
                "reason": "Part 1 synthetic canonical path is CdA-unidentifiable",
            },
            "cla": {
                "result": "ClA remains disabled",
                "reason": "held-out corner/braking improvement was not established in Part 2",
            },
            "wake": {
                "result": "wake remains synthetic/assumed",
                "reason": "following-gap examples lack enough clean status/pit context to separate tow and dirty air",
            },
            "wetness_wind": {
                "result": "not calibrated",
                "reason": "weather trace and verified circuit orientation are unavailable",
            },
            "rival_pit": {
                "result": "RIVAL PIT = UNKNOWN; DATA CURRENTLY INSUFFICIENT",
                "reason": "pit data unavailable in the current corpus",
            },
        },
    }
    summary["summary_fingerprint"] = fingerprint_json(summary)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_root) / version
    out_dir.mkdir(parents=True, exist_ok=False)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(path)
    print(json.dumps({
        "summary_fingerprint": summary["summary_fingerprint"],
        "corpus_fingerprint": summary["corpus_fingerprint"],
        "examples": validation["dataset"]["n_examples"],
        "pass_model_result": summary["pass_model_result"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
