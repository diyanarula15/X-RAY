#!/usr/bin/env python3
"""P2 worked trace: every candidate action, not just the winner.

Reports the full value decomposition at one opportunity, then mutates all future
data and shows the recommendation is unchanged.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.test_p1_service import _payload                            # noqa: E402
from xray.decision_service import evaluate_opportunity_decision       # noqa: E402

PLAN = {"OWN": {"planned_pit_lap": 4, "source": "team_plan", "confidence": 0.9}}
KEYS = ("decision", "zone", "opportunity_id", "deployment_budget_mj",
        "actual_deployed_mj", "predicted_own_speed_mps",
        "predicted_rival_speed_mps", "predicted_delta_v_mps",
        "pass_probability", "value_action", "value_hold", "decision_margin",
        "action_consensus", "expected_regret")


def show(r):
    ic = r["input_confidence"]
    h = r["horizon"]
    print("=" * 92)
    print("P2 DECISION -- FULL CANDIDATE DECOMPOSITION")
    print("=" * 92)
    print(f"  opportunity_id            {r['opportunity_id']}")
    print(f"  lap / zone                {r['lap']} / {r['zone']}")
    print(f"  decision_point_s          {r['decision_point_s']:.1f} m")
    print(f"  decision_time_s           {r['decision_time_s']:.3f} s")
    print(f"  horizon                   {len(h)} opportunities "
          f"(1 observed + {len(h)-1} causal forecast)")
    print()
    print("  CAUSAL SOURCES")
    for k, v in ic.items():
        print(f"    {k:24s} {v}")
    print()
    o0 = h[0]
    print("  STATE AT THE DECISION POINT")
    print(f"    own usable energy       {o0['own_usable_energy_mj']:.4f} MJ")
    print(f"    rival usable energy     {o0['rival_usable_energy_mj']:.4f} MJ")
    print(f"    gap                     {o0['gap_s']:.3f} s")
    print(f"    own / rival wear        {o0['own_wear_fraction']:.5f} / "
          f"{o0['rival_wear_fraction']:.5f}")
    print(f"    pit reset index         {r['pit_reset_index']}")
    print()
    print("  RIVAL-ENERGY SCENARIOS")
    for p in r["policy_posterior"]:
        print(f"    w={p['weight']:.2f}  E_riv={p['rival_usable_energy_mj']:.3f} MJ"
              f"  optimal={p['optimal_action']:<24s} "
              f"V*={p['optimal_value']:.6f}  V(chosen)={p['chosen_action_value']:.6f}")
    print()
    print("  CANDIDATE ACTIONS")
    hdr = (f"    {'action':<24s} {'req MJ':>7s} {'dep MJ':>7s} {'sat':>4s} "
           f"{'v_own':>7s} {'v_riv':>7s} {'dv':>7s} {'P(pass)':>8s} {'value':>10s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for c in sorted(r["candidate_actions"],
                    key=lambda x: (x["value"] is None, -(x["value"] or 0))):
        v = "  n/a" if c["value"] is None else f"{c['value']:10.6f}"
        mark = " <= CHOSEN" if (c["kind"] == r["decision"]
                                and abs(c["requested_budget_mj"]
                                        - r["deployment_budget_mj"]) < 1e-9) else ""
        print(f"    {c['action']:<24s} {c['requested_budget_mj']:7.3f} "
              f"{c['actual_deployed_mj']:7.3f} {str(c['saturated']):>4s} "
              f"{c['own_speed_mps']:7.2f} {c['rival_speed_mps']:7.2f} "
              f"{c['delta_v_mps']:+7.2f} {c['pass_probability']:8.5f} {v}{mark}")
        if not c["feasible"]:
            print(f"        REJECTED: {c['infeasible_reason']}")
    print()
    nb = r["next_best_action"]
    print("  RESULT")
    print(f"    chosen                  {r['decision']} zone {r['zone']} "
          f"{r['deployment_budget_mj']:.3f} MJ "
          f"(deployed {r['actual_deployed_mj']:.3f} MJ)")
    print(f"    next best               {nb['kind']} {nb['zone']} "
          f"{nb['deployment_budget_mj']:.3f} MJ  value {nb['value']:.6f}")
    print(f"    value_action / hold     {r['value_action']:.6f} / {r['value_hold']:.6f}")
    print(f"    decision_margin         {r['decision_margin']:.6f}")
    print(f"    action_consensus        {r['action_consensus']:.3f}")
    print(f"    expected_regret         {r['expected_regret']:.6f}")
    print(f"    pass model              {r['pass_model_calibration'].upper()}")
    print()


def main() -> None:
    base = _payload()
    r = evaluate_opportunity_decision(base, "OWN", "RIV", plan=PLAN)
    show(r)

    bad = copy.deepcopy(base)
    t, lap0, n = r["decision_time_s"], r["lap"], 0
    for w in bad["weather_trace"]:
        if w["t"] > t:
            w.update(air_temp_c=48.0, track_temp_c=72.0, rho=0.83, rainfall=True)
            n += 1
    for name in ("OWN", "RIV"):
        tr = bad["cars"][name]["trace"]
        for i, tt in enumerate(tr["t"]):
            if tt > t:
                tr["v"][i] = 2.0
                tr["usable_mean"][i] = 3.95
                n += 1
    for row in bad["laps"]:
        if row["lap"] > lap0:
            row.update(compound="WET", tyre_life=49, fresh_tyre=True, stint=9,
                       pit_in_time_s=1.0)
            n += 1
        row["pass_success"] = True
    for g in bad["gaps"]:
        if int(g["lap"]) > lap0:
            g["gap_s"] = 9.9
            n += 1
    r2 = evaluate_opportunity_decision(bad, "OWN", "RIV", plan=PLAN)

    print("=" * 92)
    print(f"CAUSALITY -- {n} future telemetry/weather/tyre/pit/label records corrupted")
    print("=" * 92)
    bad_keys = [k for k in KEYS if r[k] != r2[k]]
    for k in KEYS:
        print(f"    {k:28s} {'CHANGED' if k in bad_keys else 'unchanged'}")
    print(f"    {'candidate_actions':28s} "
          f"{'CHANGED' if r['candidate_actions'] != r2['candidate_actions'] else 'unchanged'}")
    if bad_keys or r["candidate_actions"] != r2["candidate_actions"]:
        raise SystemExit(f"\nFAIL: future data reached the past: {bad_keys}")
    print("\nPASS: the recommendation and every candidate value are unchanged.")


if __name__ == "__main__":
    main()
