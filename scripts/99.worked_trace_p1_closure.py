#!/usr/bin/env python3
"""P1 closure trace: current tyre condition through the production path.

Shows the physical speed lookup explicitly -- which surface, at which wear, for
which car -- then reruns the SAME opportunity with only the rival's tyre worse
and shows the physics change.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.test_p1_service import _payload                          # noqa: E402
from xray.decision_service import (evaluate_decision_trace_from_payload,
                                   _cached_zone_models)             # noqa: E402

PLAN = {"OWN": {"planned_pit_lap": 4, "source": "team_plan", "confidence": 0.9}}


def opportunity(payload):
    out = evaluate_decision_trace_from_payload(payload, "OWN", "RIV",
                                               explicit_pit_plan=PLAN)
    row = out["laps"][1]
    return row, max(row["zone_candidates"], key=lambda c: c["value_attack_ranked"])


def show(tag, c):
    env, own, riv = c["environment"], c["own_tyre"], c["rival_tyre"]
    pc, tm = c["pit_context"], c["tyre_model"]
    print("=" * 78)
    print(tag)
    print("=" * 78)
    print(f"  lap / zone                   {c['lap']} / {c['requested_zone']}")
    print(f"  decision_point_s             {c['decision_point_s']:.1f} m")
    print(f"  decision_time_s              {c['decision_time_s']:.3f} s")
    print()
    print("  ENVIRONMENT")
    print(f"    rho                        {env['rho']:.4f} kg/m^3")
    print(f"    track temp                 {env['track_temp_c']} C")
    print(f"    wetness                    {c['track_wetness_index']:.3f} "
          f"[{c['wetness_source']}]")
    print(f"    weather source             {c['weather_source']} "
          f"(age {c['weather_age_s']:.1f} s)")
    print(f"    wind                       unavailable -- no track-frame heading")
    print()
    print("  TYRES")
    for who, t in (("own  ", own), ("rival", riv)):
        print(f"    {who}                    {t['compound']}  age "
              f"{t['tyre_life_laps']:.0f} laps  wear {t['wear_fraction']*100:.3f}%  "
              f"T {t['estimated_temp_c']:.1f} C  grip x{t['grip_scale']:.4f}")
    print()
    print("  PIT CONTEXT")
    print(f"    source / causal            {pc['source']} / "
          f"{pc['source'] != 'oracle_eval'}")
    print(f"    laps_to_pit_mean           {pc['laps_to_pit_mean']}")
    print(f"    confidence                 {pc['confidence']:.2f}")
    print(f"    reset inside DP horizon    {c['pit_resets_next_lap']}")
    print()
    print("  ENERGY")
    print(f"    own usable                 {c['own_usable_energy_mj']:.4f} MJ")
    print(f"    rival usable (p10..p90)    {c['rival_usable_energy_mj']:.4f} "
          f"({c['rival_usable_p10_mj']:.3f}..{c['rival_usable_p90_mj']:.3f}) MJ")
    print()
    print("  PHYSICAL SPEED LOOKUP (energy x wear surface)")
    print(f"    own:   E = {c['own_usable_energy_mj']*1e6:.0f} J   "
          f"wear = {c['wear_fraction']:.5f}   -> v = "
          f"{c['predicted_own_speed_mps']:.4f} m/s")
    print(f"    rival: E = {c['rival_usable_energy_mj']*1e6:.0f} J   "
          f"wear = {c['rival_wear_fraction']:.5f}   -> v = "
          f"{c['predicted_rival_speed_mps']:.4f} m/s")
    print()
    print("  RESULT")
    print(f"    delta_v                    {c['predicted_delta_v_mps']:+.4f} m/s")
    print(f"    gap                        {c['gap_s']:.3f} s")
    print(f"    P(pass)                    {c['pass_probability']:.6f} "
          f"[{c['model_calibration']}]")
    if tm:
        print(f"    wear/lap hold              {tm['wear_per_lap_hold']:.6f}")
        print(f"    wear/lap attack            {tm['wear_per_lap_attack']:.6f}")
        print(f"    attack extra               {tm['attack_extra_wear_per_lap']:.6f}")
    print(f"    attack_wear_cont_penalty   {c['attack_wear_continuation_penalty']:.6f}")
    va = c["value_attack"]
    print(f"    V_attack                   "
          f"{'rejected (unaffordable)' if va is None else f'{va:.6f}'}")
    print(f"    V_wait                     {c['value_wait']:.6f}")
    print(f"    threshold tau              {c['attack_threshold']:.6f}")
    print(f"    DECISION                   {c['decision']}")
    print()


def main() -> None:
    base = _payload()
    _cached_zone_models.cache_clear()
    _, c1 = opportunity(base)
    show("BASELINE OPPORTUNITY", c1)

    worse = copy.deepcopy(base)
    for r in worse["laps"]:
        if r["driver"] == "RIV":
            r["fresh_tyre"] = False
            r["tyre_life"] = r["lap"] + 30
    _, c2 = opportunity(worse)
    show("SAME OPPORTUNITY, RIVAL TYRE WORSE (nothing else changed)", c2)

    print("=" * 78)
    print("PHYSICAL PATH CHANGED BY THE RIVAL TYRE ALONE")
    print("=" * 78)
    for k, fmt in (("rival_wear_fraction", "{:.5f}"),
                   ("predicted_rival_speed_mps", "{:.4f}"),
                   ("predicted_own_speed_mps", "{:.4f}"),
                   ("predicted_delta_v_mps", "{:+.4f}"),
                   ("pass_probability", "{:.6f}")):
        a, b = c1[k], c2[k]
        mark = "CHANGED" if a != b else "unchanged"
        print(f"  {k:28s} {fmt.format(a):>12s} -> {fmt.format(b):>12s}   {mark}")
    assert c2["rival_wear_fraction"] > c1["rival_wear_fraction"]
    assert c2["predicted_rival_speed_mps"] < c1["predicted_rival_speed_mps"], \
        "a worse rival tyre must lower its braking-point speed"
    assert c2["predicted_own_speed_mps"] == c1["predicted_own_speed_mps"], \
        "our own speed must not move when only the rival tyre changed"
    assert c2["predicted_delta_v_mps"] > c1["predicted_delta_v_mps"]
    print("\nPASS: rival tyre -> rival speed -> delta_v -> P(pass), own speed unmoved.")


if __name__ == "__main__":
    main()
