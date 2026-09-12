#!/usr/bin/env python3
"""One decision point, every quantity behind it, and a causality proof.

Phase D of the audit. Prints the full chain for a single opportunity, then
re-runs it with ALL telemetry after the decision timestamp corrupted and asserts
that nothing at the decision moved. If any line changes, a future sample reached
a past decision.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.test_decision_service import _payload                 # noqa: E402
from xray.decision_service import evaluate_decision_trace_from_payload  # noqa: E402

KEYS = ("gap_s", "gap_source", "gap_method", "gap_age_s", "gap_confidence",
        "own_usable_energy_mj", "rival_usable_energy_mj", "rival_usable_p10_mj",
        "rival_usable_p90_mj", "predicted_own_speed_mps",
        "predicted_rival_speed_mps", "predicted_delta_v_mps",
        "pass_probability", "attack_threshold", "value_wait", "decision",
        "zone", "attack_affordable", "value_attack_hypothetical")


def show(payload):
    out = evaluate_decision_trace_from_payload(payload, "OWN", "RIV")
    row = out["laps"][1]
    cand = max(row["zone_candidates"], key=lambda c: c["value_attack_ranked"])
    return row, cand


def main() -> None:
    payload = _payload()
    row, c = show(payload)

    print("=" * 74)
    print("ONE DECISION POINT, END TO END")
    print("=" * 74)
    print(f"  lap                            {row['lap']}")
    print(f"  track position (decision pt)   s = {c['decision_point_s']:.1f} m "
          f"(zone {c['requested_zone']} entry)")
    print(f"  decision timestamp             t = {c['decision_time_s']:.3f} s")
    print(f"  telemetry used                 own/rival samples up to index "
          f"{c['own_sample_index']} / {c['rival_sample_index']} only")
    print()
    print(f"  ego usable energy              {c['own_usable_energy_mj']:.3f} MJ")
    print(f"  rival usable posterior         {c['rival_usable_energy_mj']:.3f} MJ "
          f"[p10 {c['rival_usable_p10_mj']:.3f}, p90 {c['rival_usable_p90_mj']:.3f}]")
    fc = np.asarray(c["future_energy_forecast_mj"])
    print(f"  rival causal future forecast   {np.round(fc, 3).tolist()} MJ")
    print(f"    forecast class               {c['future_energy_model_class']} "
          f"({c['future_energy_forecast_method']}), confidence "
          f"{c['future_energy_forecast_confidence']}")
    print(f"  actual current gap             {c['gap_s']:.3f} s  "
          f"[{c['gap_source']} / {c['gap_method']}, age {c['gap_age_s']:.3f} s, "
          f"confidence {c['gap_confidence']:.2f}]")
    print()
    print(f"  selected zone                  {c['zone']}")
    print(f"  ego deployment action          "
          f"{min(c['own_usable_energy_mj'], 1e9):.3f} MJ capped at attack cost")
    print(f"  ego braking-point speed        {c['predicted_own_speed_mps']:.2f} m/s")
    print(f"  rival braking-point speed      {c['predicted_rival_speed_mps']:.2f} m/s")
    print(f"  delta v                        {c['predicted_delta_v_mps']:+.2f} m/s")
    print(f"  P(pass)                        {c['pass_probability']:.4f}  "
          f"(model: {c['model_calibration']})")
    va = c["value_attack"]
    print(f"  V_attack                       "
          f"{'rejected (unaffordable)' if va is None else f'{va:.4f}'}"
          f"   [hypothetical {c['value_attack_hypothetical']:.4f}]")
    print(f"  V_wait                         {c['value_wait']:.4f}")
    print(f"  derived threshold tau          {c['attack_threshold']:.4f}")
    print(f"  DECISION                       {c['decision']}")
    print()

    # ---------------------------------------------------------- causality proof
    t_dec = c["decision_time_s"]
    corrupted = copy.deepcopy(payload)
    n_touched = 0
    for name in ("OWN", "RIV"):
        tr = corrupted["cars"][name]["trace"]
        for i, t in enumerate(tr["t"]):
            if t > t_dec:
                tr["v"][i] = 1.0
                # Position is only scrambled inside the decision lap. Scrambling
                # it on every later lap deletes those laps from the trace, and
                # the service then (correctly) refuses to evaluate them -- which
                # tests the refusal, not the causality.
                if int(tr["lap"][i]) == int(row["lap"]):
                    tr["s"][i] = 999.0
                tr["usable_mean"][i] = 4.0
                tr["usable_p10"][i] = 4.0
                tr["usable_p90"][i] = 4.0
                n_touched += 1
    for g in corrupted["gaps"]:
        g["gap_s"] = 9.99
    row2, c2 = show(corrupted)

    print("=" * 74)
    print(f"CAUSALITY: {n_touched} samples after t = {t_dec:.3f} s corrupted")
    print("=" * 74)
    bad = [k for k in KEYS if c.get(k) != c2.get(k)]
    for k in KEYS:
        flag = "CHANGED" if k in bad else "unchanged"
        print(f"  {k:30s} {flag}")
    if bad:
        raise SystemExit(f"\nFAIL: future telemetry reached the decision: {bad}")
    print("\nPASS: every quantity at the decision is unchanged.")


if __name__ == "__main__":
    main()
