#!/usr/bin/env python3
"""One P1 decision end to end, then proof the past cannot be moved.

Prints every quantity the P1 spec asks for at a single opportunity, then
corrupts ALL telemetry, weather, tyre metadata and oracle pit data after that
timestamp and asserts nothing at the decision changed.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.test_p1_service import _payload                       # noqa: E402
from xray.decision_service import (evaluate_decision_trace_from_payload,
                                   environment_at_opportunity)   # noqa: E402

KEYS = ("gap_s", "gap_source", "gap_method", "own_usable_energy_mj",
        "rival_usable_energy_mj", "predicted_own_speed_mps",
        "predicted_rival_speed_mps", "predicted_delta_v_mps", "pass_probability",
        "attack_threshold", "value_wait", "decision", "zone", "wear_fraction",
        "attack_wear_continuation_penalty", "pit_resets_next_lap",
        "weather_source", "track_wetness_index")


def run(payload, plan):
    out = evaluate_decision_trace_from_payload(payload, "OWN", "RIV",
                                               explicit_pit_plan=plan)
    row = out["laps"][1]
    cand = max(row["zone_candidates"], key=lambda c: c["value_attack_ranked"])
    return row, cand


def main() -> None:
    payload = _payload()
    plan = {"OWN": {"planned_pit_lap": 4, "source": "team_plan", "confidence": 0.9}}
    row, c = run(payload, plan)
    env, own, riv = c["environment"], c["own_tyre"], c["rival_tyre"]

    print("=" * 76)
    print("P1 DECISION, END TO END")
    print("=" * 76)
    print(f"  lap / zone / position        {row['lap']} / {c['requested_zone']} / "
          f"s = {c['decision_point_s']:.1f} m")
    print(f"  decision timestamp           t = {c['decision_time_s']:.3f} s")
    print()
    print(f"  causal weather               {env['source']} (age {env['age_s']:.1f} s)")
    print(f"    rho                        {env['rho']:.4f} kg/m^3")
    print(f"    air / track temp           {env['air_temp_c']} C / {env['track_temp_c']} C")
    print(f"    rainfall / wetness         {env['rainfall']} / "
          f"{c['track_wetness_index']:.3f}  [{c['wetness_source']}]")
    print(f"  wind_parallel                unavailable "
          f"(no track-frame heading; payload has no physical x/y)")
    print()
    print(f"  own tyre                     {own['compound']}  age "
          f"{own['tyre_life_laps']:.0f} laps  wear {own['wear_fraction']*100:.1f}%  "
          f"T {own['estimated_temp_c']:.0f} C  grip x{own['grip_scale']:.3f}")
    print(f"    source                     {own['source']}  "
          f"(calibration: {c['tyre_calibration']})")
    print(f"  rival tyre                   {riv['compound']}  age "
          f"{riv['tyre_life_laps']:.0f} laps  wear {riv['wear_fraction']*100:.1f}%")
    pc = c["pit_context"]
    print(f"  pit context                  {pc['source']} conf {pc['confidence']:.2f}  "
          f"laps_to_pit {pc['laps_to_pit_mean']}")
    print(f"    resets next lap            {c['pit_resets_next_lap']}")
    tm = c["tyre_model"]
    if tm:
        print(f"    wear/lap hold vs attack    {tm['wear_per_lap_hold']:.5f} vs "
              f"{tm['wear_per_lap_attack']:.5f} (extra "
              f"{tm['attack_extra_wear_per_lap']:.5f})")
    print()
    print(f"  own / rival usable energy    {c['own_usable_energy_mj']:.3f} / "
          f"{c['rival_usable_energy_mj']:.3f} MJ")
    print(f"  actual gap                   {c['gap_s']:.3f} s  [{c['gap_source']}]")
    print(f"  own braking-point speed      {c['predicted_own_speed_mps']:.2f} m/s")
    print(f"  rival braking-point speed    {c['predicted_rival_speed_mps']:.2f} m/s")
    print(f"  delta v                      {c['predicted_delta_v_mps']:+.2f} m/s")
    print(f"  P(pass)                      {c['pass_probability']:.4f} "
          f"[{c['model_calibration']}]")
    va = c["value_attack"]
    print(f"  V_attack                     "
          f"{'rejected (unaffordable)' if va is None else f'{va:.4f}'}")
    print(f"  V_wait                       {c['value_wait']:.4f}")
    print(f"  attack wear cost             {c['attack_wear_continuation_penalty']:.5f}")
    print(f"  threshold tau                {c['attack_threshold']:.4f}")
    print(f"  DECISION                     {c['decision']}")
    print()

    # ------------------------------------------------------ causality
    t_dec = c["decision_time_s"]
    bad = copy.deepcopy(payload)
    n = 0
    for w in bad["weather_trace"]:
        if w["t"] > t_dec:
            w.update(air_temp_c=45.0, track_temp_c=65.0, rho=0.88,
                     rainfall=True, wind_speed_ms=30.0, wind_dir_deg=270.0)
            n += 1
    for name in ("OWN", "RIV"):
        tr = bad["cars"][name]["trace"]
        for i, t in enumerate(tr["t"]):
            if t > t_dec:
                tr["v"][i] = 2.0
                if int(tr["lap"][i]) == int(row["lap"]):
                    tr["s"][i] = 999.0
                tr["usable_mean"][i] = 3.9
                tr["usable_p10"][i] = 3.9
                tr["usable_p90"][i] = 3.9
                n += 1
    for r in bad["laps"]:
        if r["lap"] > row["lap"]:
            r.update(compound="WET", tyre_life=44, fresh_tyre=True,
                     pit_in_time_s=1.0, pit_out_time_s=2.0)
            n += 1
    for r in bad["laps"]:
        r["pit_in_time_s"] = 7.0        # oracle, everywhere
    row2, c2 = run(bad, plan)

    print("=" * 76)
    print(f"CAUSALITY: {n} future telemetry/weather/tyre/pit records corrupted")
    print("=" * 76)
    changed = [k for k in KEYS if c.get(k) != c2.get(k)]
    for k in KEYS:
        print(f"  {k:34s} {'CHANGED' if k in changed else 'unchanged'}")
    if changed:
        raise SystemExit(f"\nFAIL: the future reached the past: {changed}")
    print("\nPASS: every quantity at the decision is unchanged.")


if __name__ == "__main__":
    main()
