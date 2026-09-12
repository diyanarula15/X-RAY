#!/usr/bin/env python3
"""P3.5 Part 3: capped independent baseline, final Gate D, summary artifact."""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray.hardened_eval import (FIXED_E_GRID_J, FIXED_E_MAX_TRAINING_EXAMPLES,
                                HARDENED, HARDENED_EVAL_VERSION,
                                run_hardened_real_validation,
                                score_with_independent_baseline)
from xray.p3_validation import (TARGETS, build_validation_examples, fingerprint,
                                load_payload_paths)

OUT = Path("out/p35")
FINAL_ESTIMATOR = {"centre": "evidence", "boundary": "conditional"}

OLD_XRAY_MAE = {
    "future_speed_5s": 22.39,
    "straight_speed_3s": 5.90,
    "braking_point_speed": 15.78,
}

PART2_REJECTED_MAE = {
    "future_speed_5s": 23.043,
    "straight_speed_3s": 6.227,
    "braking_point_speed": 16.313,
}

NEUTRAL_E = {
    "future_speed_5s": {"mae": 21.051910665560754},
    "straight_speed_3s": {"mae": 5.684472804347788},
    "braking_point_speed": {"mae": 15.117602549279304},
}

GATE_A_RECORDED = {
    "old": {"degenerate_frac": 0.226, "mean_width_mj": 0.498, "empty_frac": 0.327},
    "part2_truncation": {"degenerate_frac": 0.093, "mean_width_mj": 0.796,
                         "empty_frac": 0.169},
    "final_conditional_candidate": {"degenerate_frac": 0.101,
                                    "mean_width_mj": 0.729,
                                    "empty_frac": 0.190},
    "status": "reused_recorded_result_not_rerun",
}

GATE_B_RECORDED = {
    "old": {"mae_mj": 0.186, "bias_mj": -0.046, "containment": 0.453,
            "width_mj": 0.507, "degenerate_frac": 0.451, "corr": 0.844},
    "part2_rejected": {"mae_mj": 0.694, "bias_mj": 0.653,
                       "containment": 0.369, "width_mj": 1.313,
                       "degenerate_frac": 0.0, "corr": 0.683},
    "final_conditional_candidate": {"mae_mj": 0.188, "bias_mj": -0.034,
                                    "containment": 0.483, "width_mj": 0.545,
                                    "degenerate_frac": 0.391, "corr": 0.839},
    "status": "reused_recorded_result_not_rerun",
}

GATE_C_RECORDED = {
    "artifact": "out/p35/mismatch_final.json",
    "final_verdict": "0 of 5 mismatch cases produced larger energy error without a wider band",
    "status": "reused_recorded_result_not_rerun",
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _payload_fingerprint(paths: list[Path]) -> str:
    return fingerprint([{"path": str(p), "sha256": _sha256_file(p)}
                        for p in sorted(paths)])


def _build_examples(payloads: list[dict]) -> tuple[list, dict]:
    examples, audit = [], {}
    for pl in payloads:
        ex, a = build_validation_examples(pl)
        examples.extend(ex)
        audit[str(pl.get("id") or pl.get("event"))] = a
    return examples, audit


def _baseline_rows(chosen: dict) -> list[dict]:
    rows = []
    for fold in sorted(chosen):
        for target in TARGETS:
            fit = chosen.get(fold, {}).get(target)
            if not fit:
                continue
            rows.append({
                "fold": fold,
                "target": target,
                "selected_fixed_energy_j": fit["energy_j"],
                "selected_fixed_energy_mj": None if fit["energy_j"] is None
                    else fit["energy_j"] / 1e6,
                "training_mae": fit["train_mae"],
                "training_examples_available": fit["n_train_available"],
                "training_examples_used": fit["n_train_used"],
                "race_counts_used": fit["race_counts_used"],
                "subset_fingerprint": fit["subset_fingerprint"],
            })
    return rows


def _final_real_rows(final_real: dict, independent: dict) -> list[dict]:
    rows = []
    final_metrics = ((final_real.get("metrics") or {}).get("metrics") or {})
    independent_metrics = independent.get("metrics") or {}
    for target in TARGETS:
        fm = ((final_metrics.get(target) or {}).get("aggregate") or {})
        im = independent_metrics.get(target) or {}
        rows.append({
            "target": target,
            "final": (fm.get("xray") or {}),
            "old_xray": {"mae": OLD_XRAY_MAE[target]},
            "part2_rejected": {"mae": PART2_REJECTED_MAE[target]},
            "train_optimal_fixed_e": im.get("train_optimal_fixed") or {},
            "neutral_e": im.get("neutral") or NEUTRAL_E[target],
        })
    return rows


def main() -> int:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    payload_paths = [Path(p) for p in sorted(glob.glob("out/races/*.json"))]
    if not payload_paths:
        raise SystemExit("no payloads found under out/races")

    data_fp = _payload_fingerprint(payload_paths)
    payloads = load_payload_paths(payload_paths)
    examples, audit = _build_examples(payloads)
    rounds = {str(p.get("id") or p.get("event")): int(p["round"]) for p in payloads}

    baseline_config = {
        "baseline": "train-optimal-fixed-E",
        "grid_j": list(map(float, FIXED_E_GRID_J)),
        "max_training_examples_per_target_fold": FIXED_E_MAX_TRAINING_EXAMPLES,
        "targets": list(TARGETS),
        "data_fingerprint": data_fp,
    }
    baseline_fp = fingerprint(baseline_config)
    baseline_started = time.perf_counter()
    independent = score_with_independent_baseline(
        payloads, examples, cfg=None,
        max_examples=FIXED_E_MAX_TRAINING_EXAMPLES)
    baseline_runtime_s = time.perf_counter() - baseline_started
    independent["config"] = baseline_config
    independent["configuration_fingerprint"] = baseline_fp
    independent["data_fingerprint"] = data_fp
    independent["examples"] = {
        "n": len(examples),
        "by_target": {t: sum(1 for e in examples if e.target == t) for t in TARGETS},
        "audit": audit,
    }
    independent["runtime_s"] = baseline_runtime_s
    independent["methodology"] = (
        "For each leave-one-race-out fold and target, fit one constant energy on "
        "training races only. Use at most 1200 examples, balanced across training "
        "races with evenly spaced deterministic samples within each race. The "
        "subset key excludes held-out rows and estimator outputs."
    )
    baseline_path = OUT / f"train_optimal_fixed_e__{baseline_fp[:16]}.json"
    baseline_path.write_text(json.dumps(independent, indent=1, default=str))

    gate_d_config = {
        "version": HARDENED_EVAL_VERSION,
        "estimator": FINAL_ESTIMATOR,
        "n_particles": 200,
        "data_fingerprint": data_fp,
    }
    gate_d_fp = fingerprint(gate_d_config)
    gate_d_started = time.perf_counter()
    final_real = run_hardened_real_validation(
        [str(p) for p in payload_paths], rounds,
        estimator=FINAL_ESTIMATOR, n_particles=200)
    gate_d_runtime_s = time.perf_counter() - gate_d_started
    final_real["configuration_fingerprint"] = gate_d_fp
    final_real["data_fingerprint"] = data_fp
    final_real["runtime_s"] = gate_d_runtime_s
    final_path = OUT / f"real_final_conditional__{gate_d_fp[:16]}.json"
    final_path.write_text(json.dumps(final_real, indent=1, default=str))

    final_rows = _final_real_rows(final_real, independent)
    improves_old = [
        (r["final"].get("mae") is not None
         and r["final"]["mae"] < r["old_xray"]["mae"])
        for r in final_rows
    ]
    activated = bool(final_rows and all(improves_old))
    decision = ("FINAL HARDENED ESTIMATOR — ACTIVATED" if activated
                else "FINAL HARDENED ESTIMATOR — NOT ACTIVATED")
    blocker = None if activated else (
        "FINAL conditional candidate does not show consistent real held-out MAE "
        "improvement over OLD X-RAY across all three targets."
    )

    summary = {
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "gate_a_recorded": GATE_A_RECORDED,
        "gate_b_recorded": GATE_B_RECORDED,
        "gate_c_recorded": GATE_C_RECORDED,
        "final_candidate_configuration": FINAL_ESTIMATOR,
        "hardened_eval_default": HARDENED,
        "train_optimal_fixed_e_methodology": independent["methodology"],
        "train_optimal_fixed_e_artifact": str(baseline_path),
        "train_optimal_fixed_e_configuration_fingerprint": baseline_fp,
        "chosen_fixed_e_per_fold_target": _baseline_rows(
            independent["chosen_fixed_energy_by_fold"]),
        "gate_d_final_artifact": str(final_path),
        "gate_d_configuration_fingerprint": gate_d_fp,
        "gate_d_real_metrics": final_rows,
        "old_metrics_reference": {
            "values": OLD_XRAY_MAE,
            "artifact_reference": "recorded P3 real held-out metrics",
        },
        "part2_metrics_reference": {
            "values": PART2_REJECTED_MAE,
            "artifact_reference": "out/p35/part2/20260912T215427Z/summary.json",
        },
        "neutral_baseline": NEUTRAL_E,
        "activation_decision": decision,
        "activation_blocker": blocker,
        "p2_status": ("P2 integration unchanged" if not activated
                      else "P2 would consume existing compatible usable-energy fields"),
        "production_estimator_status": ("OLD estimator remains production"
                                        if not activated
                                        else "conditional/evidence estimator activated"),
        "files_changed_resumed_session": [
            "xray/hardened_eval.py",
            "xray/realfit.py",
            "tests/test_p35_part3.py",
            "scripts/15.p35_part3.py",
        ],
        "new_tests_resumed_session": [{
            "command": "python3 -m pytest -q tests/test_p35_part3.py",
            "result": "8 passed",
            "runtime_s": 0.08,
        }],
        "runtime_new_work_s": {
            "baseline": baseline_runtime_s,
            "gate_d_final_conditional": gate_d_runtime_s,
            "script_total": time.perf_counter() - started,
        },
        "data_fingerprint": data_fp,
        "safe_claims": [
            "The fixed-E baseline is estimator-independent.",
            "The fixed-E selector uses training races only for each held-out fold.",
            "Conditional deployment penalizes only empty support intersections.",
        ],
        "unsafe_claims": [
            "True real usable energy is known.",
            "The reserve level is identified from public telemetry.",
            "One target alone validates the hardened estimator.",
        ],
        "remaining_scientific_limitations": [
            "No real rival usable-energy ground truth.",
            "Real Gate D is future-observable predictive validation, not direct energy validation.",
        ],
    }
    summary["summary_fingerprint"] = fingerprint(summary)
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT / "part3" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1, default=str))

    print(f"baseline {baseline_path}")
    print(f"gate_d   {final_path}")
    print(f"summary  {summary_path}")
    print(f"fingerprint {summary['summary_fingerprint']}")
    print(decision)
    if blocker:
        print(blocker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
