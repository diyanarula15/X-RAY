#!/usr/bin/env python3
"""One closed-loop episode, decision by decision, with the state it changed.

Shows that P2's choice is executed and that executing it moves the world the
next decision is made in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xray.closedloop import (AttackPlan, blind_rival_belief, p2_plan,  # noqa: E402
                             policy_for, run_episode, world)
from xray.config import load_config                                    # noqa: E402
from xray.policy import get_policy                                     # noqa: E402
from xray.sim import FOLLOWER, LEADER, Simulator                       # noqa: E402

SEED = 7


def main() -> None:
    cfg = world(load_config(), SEED)
    print("=" * 88)
    print(f"CLOSED-LOOP EPISODE -- seed {SEED}")
    print("=" * 88)
    print(f"  start gap        {cfg['sim']['start_gap_s']:.3f} s")
    print(f"  start energy     {cfg['sim']['e_start_frac']:.3f} of store, both cars")
    print(f"  rival policy     {cfg['sim']['leader_policy']}")
    print(f"  feed noise       {cfg['observe']['speed_noise_ms']:.3f} m/s")
    print()

    baseline = Simulator(cfg, seed=SEED).run()
    print("  STEP 1 -- observe the rival publicly, reconstruct blind")
    belief = blind_rival_belief(baseline, LEADER, cfg, SEED)
    print(f"    source           {belief.source}")
    truth = baseline.cars[LEADER].e_lap_open
    print(f"    {'lap':>4s} {'belief MJ':>10s} {'p10':>7s} {'p90':>7s} "
          f"{'TRUTH MJ':>9s}  (truth is for this printout only)")
    for lap in range(min(belief.n_laps, 8)):
        print(f"    {lap:4d} {belief.usable_by_lap_j[lap]/1e6:10.3f} "
              f"{belief.p10_by_lap_j[lap]/1e6:7.3f} {belief.p90_by_lap_j[lap]/1e6:7.3f} "
              f"{truth[min(lap, len(truth)-1)]/1e6:9.3f}")
    print()

    print("  STEP 2 -- P2 chooses opportunity, zone and budget from the BELIEF")
    plan = p2_plan(belief, baseline, cfg, FOLLOWER)
    print(f"    decision         {'ATTACK' if plan.attacks else 'HOLD'}")
    if plan.attacks:
        d = plan.detail
        ceiling = d.get("ceiling_j", 0.0)
        print(f"    lap / zone       {plan.lap} / {plan.zone}")
        print(f"    budget           {plan.budget_j/1e6:.3f} MJ "
              f"of a {ceiling/1e6:.3f} MJ zone ceiling "
              f"({100*plan.budget_j/max(ceiling,1):.0f}% -- "
              f"{'interior' if plan.budget_j < 0.999*ceiling else 'maximum'})")
        print(f"    V(action)        {d.get('value_action'):.6f}")
        print(f"    V(hold)          {d.get('value_hold'):.6f}")
        print(f"    margin           {d.get('decision_margin'):.6f}")
        print(f"    consensus        {d.get('action_consensus'):.3f}")
        print(f"    regret           {d.get('expected_regret'):.6f}")
        print("    candidate values:")
        for lbl, v in sorted(d.get("action_values", {}).items(),
                             key=lambda kv: -kv[1]):
            mark = "  <= chosen" if abs(v - d["value_action"]) < 1e-12 else ""
            print(f"        {lbl:26s} {v:.6f}{mark}")
    print()

    print("  STEP 3 -- the simulator executes it, and the world changes")
    hold = run_episode(cfg, SEED, AttackPlan("hold"), FOLLOWER)
    p2 = run_episode(cfg, SEED, plan, FOLLOWER)
    base = get_policy(cfg["sim"]["follower_policy"])
    g_hold = Simulator(cfg, policies={FOLLOWER: base,
                                      LEADER: get_policy(cfg["sim"]["leader_policy"])},
                       seed=SEED).run()
    g_p2 = Simulator(cfg, policies={FOLLOWER: policy_for(base, plan),
                                    LEADER: get_policy(cfg["sim"]["leader_policy"])},
                     seed=SEED).run()

    th, tp = g_hold.cars[FOLLOWER], g_p2.cars[FOLLOWER]
    print(f"    {'lap':>4s} {'store@open HOLD':>16s} {'store@open P2':>14s} "
          f"{'delta MJ':>9s} {'peak v HOLD':>12s} {'peak v P2':>10s}")
    for lap in range(min(g_hold.n_laps, 8)):
        mh, mp = th.lap == lap, tp.lap == lap
        vh = th.v[mh].max() * 3.6 if mh.any() else float("nan")
        vp = tp.v[mp].max() * 3.6 if mp.any() else float("nan")
        eh = th.e_lap_open[lap] / 1e6
        ep = tp.e_lap_open[lap] / 1e6
        star = "  <= attack lap" if plan.attacks and lap == plan.lap else ""
        print(f"    {lap:4d} {eh:16.4f} {ep:14.4f} {ep-eh:+9.4f} "
              f"{vh:12.1f} {vp:10.1f}{star}")
    print()
    print("  STEP 4 -- realised outcome (same seed, so the same pass dice)")
    for tag, r in (("HOLD", hold), ("P2", p2)):
        print(f"    {tag:5s} attempts={r.attempts} successes={r.successes} "
              f"ahead={r.ahead_fraction:.4f} finished_ahead={r.finished_ahead} "
              f"deployed={r.energy_deployed_j/1e6:.2f} MJ "
              f"unused={r.energy_unused_j/1e6:.3f} MJ")
    changed = not np.array_equal(th.v, tp.v)
    print()
    print(f"  The executed action changed the trajectory: {changed}")
    if not changed and plan.attacks:
        raise SystemExit("FAIL: the plan was not executed")
    print("  Pass model: SYNTHETIC -- P(pass) here is a model output, not a "
          "measured probability.")


if __name__ == "__main__":
    main()
