#!/usr/bin/env python3
"""Closed loop: read the rival blind, call the attack, then actually try it.

Everything up to now scored the *decision*. This scores the *outcome*.

For each seeded race:

  1. Two cars race. We know our own car completely. The rival's energy is hidden:
     we get its speed trace, downsampled and noised, and nothing else.
  2. X-RAY reconstructs the rival's deployable energy from that trace alone.
  3. The decision engine picks a lap and a zone to attack.
  4. The race is re-run with our car actually executing that attack -- saving
     until the called lap, then emptying the store in the called zone. The
     overtake is resolved by the simulator's own physics and pass model.

So the recommendation changes what the car does, and the pass either happens or
it does not. Compared against:

  ORACLE   the same procedure, but allowed to see the rival's true energy
  BLIND    attack on the first lap the battery allows, no read on the rival
  PASSIVE  never attack deliberately -- the control
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray.config import load_config, merge
from xray.constants import E_STORE_MAX
from xray.decision import (build_model, delta_v, rival_energy_at_zone,
                           solve_exogenous)
from xray.estimator import RESERVE_RELEASE_LAPS, estimate
from xray.observe import observe
from xray.overtake import p_pass
from xray.policy import get_policy
from xray.sim import FOLLOWER, LEADER, Simulator
from xray.vehicle import VehicleParams


def race(cfg, seed, follower_policy, leader_policy):
    return Simulator(cfg, seed=seed,
                     policies={LEADER: leader_policy, FOLLOWER: follower_policy}).run()


def outcome(gt) -> dict:
    """Did we get past, and when?"""
    passes = [e for e in gt.events
              if e.kind == "overtake" and e.detail["success"] and e.car == FOLLOWER]
    attempts = [e for e in gt.events if e.kind == "overtake" and e.car == FOLLOWER]
    ahead_at_end = bool(gt.order[-1] == 1)
    return {
        "passed": bool(passes), "lap_passed": passes[0].lap if passes else None,
        "attempts": len(attempts),
        "best_p": max((e.detail["p"] for e in attempts), default=0.0),
        "ahead_at_end": ahead_at_end,
    }


def true_rival_track(gt, n_laps: int) -> np.ndarray:
    """Ground truth, used only for the oracle arm and never by X-RAY."""
    tr = gt.cars[LEADER]
    zone = gt.track.zone_by_name("A")
    reserve = get_policy(gt.policies[LEADER]).reserve * E_STORE_MAX
    out = np.zeros(n_laps)
    for L in range(n_laps):
        m = trace_mask = (tr.lap == L)
        if not m.any():
            out[L] = out[L - 1] if L else 0.0
            continue
        idx = np.flatnonzero(m)
        j = idx[int(np.argmin(np.abs(tr.s[idx] - zone.s_straight_start)))]
        floor = reserve * min(1.0, max(n_laps - L, 0) / RESERVE_RELEASE_LAPS)
        out[L] = max(float(tr.E[j] - floor), 0.0)
    return out


def call_attack(model, sol, own: np.ndarray, n_laps: int):
    """First lap the engine says go, and the zone it picks."""
    for L in range(n_laps):
        z = sol.action(n_laps - L, own[L])
        if z is not None:
            return L, z
    return None, None


def run_seed(cfg, seed: int, rate: float, particles: int, verbose: bool) -> dict:
    base_f = get_policy(cfg["sim"]["follower_policy"])
    base_l = get_policy(cfg["sim"]["leader_policy"])

    # --- 1. the race as it would happen, and the rival's speed trace
    gt0 = race(cfg, seed, base_f, base_l)
    n_laps = gt0.n_laps

    # --- 2. blind the rival: speed only
    obs = observe(gt0, LEADER, rate_hz=rate,
                  speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=seed + 1)
    belief = estimate(obs, gt0.track, n_particles=particles, seed=seed + 2)

    params = VehicleParams.from_config(cfg)
    model = build_model(
        gt0.track, params, n_laps=n_laps,
        recharge_per_lap=float(np.mean(belief.harvested_lap[belief.harvested_lap > 0])),
        rival_spend_per_lap=float(np.mean(belief.deployed_lap[belief.deployed_lap > 0])),
        own_spend_per_lap=float(np.mean(gt0.cars[FOLLOWER].deployed_lap)) * 0.55)

    own = np.array([float(gt0.cars[FOLLOWER].E[gt0.cars[FOLLOWER].lap == L].max())
                    if (gt0.cars[FOLLOWER].lap == L).any() else 0.0
                    for L in range(n_laps)])

    seen = rival_energy_at_zone(belief, obs, gt0.track, n_laps)
    truth = true_rival_track(gt0, n_laps)

    lap_x, zone_x = call_attack(model, solve_exogenous(model, seen), own, n_laps)
    lap_o, zone_o = call_attack(model, solve_exogenous(model, truth), own, n_laps)
    lap_b = next((L for L in range(n_laps) if own[L] >= model.attack_cost * 0.75), 0)

    # --- 3. execute each plan in the same race and see what actually happens
    def plan(L, Z):
        if L is None:
            return base_f
        return dataclasses.replace(base_f, attack_lap=L, attack_zone=Z, save_until=L)

    arms = {
        "xray": race(cfg, seed, plan(lap_x, zone_x), base_l),
        "oracle": race(cfg, seed, plan(lap_o, zone_o), base_l),
        "blind": race(cfg, seed, plan(lap_b, "A"), base_l),
        "passive": gt0,
    }
    res = {k: outcome(v) for k, v in arms.items()}

    if verbose:
        print(f"  seed {seed}: X-RAY calls lap {(lap_x or 0)+1} zone {zone_x} -> "
              f"{'PASSED lap ' + str(res['xray']['lap_passed']+1) if res['xray']['passed'] else 'no pass'}"
              f"   | oracle lap {(lap_o or 0)+1} -> "
              f"{'pass' if res['oracle']['passed'] else 'no'}"
              f"   | blind lap {lap_b+1} -> "
              f"{'pass' if res['blind']['passed'] else 'no'}")

    return {"seed": seed, "lap_x": lap_x, "zone_x": zone_x, "lap_o": lap_o,
            "lap_b": lap_b, "rival_err": float(np.mean(np.abs(seen - truth)) / 1e6),
            **{f"{k}_{m}": res[k][m] for k in res for m in
               ("passed", "attempts", "best_p", "ahead_at_end")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--rate", type=float, default=4.17)
    ap.add_argument("--particles", type=int, default=250)
    ap.add_argument("--leader", default="AGGRESSIVE")
    ap.add_argument("--follower", default="CONSERVATIVE")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = merge(load_config(), sim={"leader_policy": args.leader,
                                    "follower_policy": args.follower})
    rows = []
    for i in range(args.seeds):
        try:
            rows.append(run_seed(cfg, 200 + i, args.rate, args.particles, args.verbose))
        except Exception as exc:  # noqa: BLE001
            print(f"  seed {200+i}: skipped ({exc})")
    if not rows:
        raise SystemExit("no usable seeds")

    def rate_of(k):
        return float(np.mean([r[f"{k}_passed"] for r in rows]))

    def bestp(k):
        return float(np.mean([r[f"{k}_best_p"] for r in rows]))

    def ahead(k):
        return float(np.mean([r[f"{k}_ahead_at_end"] for r in rows]))

    print(f"\n{'arm':10s}{'passed':>9s}{'ahead at flag':>15s}"
          f"{'best chance seen':>19s}   what it knew")
    print("-" * 74)
    for k, note in (("oracle", "the rival's true energy"),
                    ("xray", "only the rival's speed trace"),
                    ("blind", "nothing — attacks at the first chance"),
                    ("passive", "nothing — never attacks deliberately")):
        print(f"{k:10s}{rate_of(k)*100:8.0f}%{ahead(k)*100:14.0f}%"
              f"{bestp(k):19.3f}   {note}")

    x = np.array([r["xray_passed"] for r in rows], dtype=float)
    b = np.array([r["blind_passed"] for r in rows], dtype=float)
    p = np.array([r["passive_passed"] for r in rows], dtype=float)
    for name, other in (("blind", b), ("passive", p)):
        d = x - other
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        print(f"\nX-RAY minus {name:8s}: {d.mean()*100:+.0f} percentage points, "
              f"95% CI [{(d.mean()-1.96*se)*100:+.0f}, {(d.mean()+1.96*se)*100:+.0f}]"
              f"  {'SIGNIFICANT' if d.mean()-1.96*se > 0 else 'not significant'}")

    agree = float(np.mean([r["lap_x"] == r["lap_o"] for r in rows]))
    print(f"\nX-RAY called the oracle's exact lap in {agree*100:.0f}% of races")
    print(f"Mean error in the rival's deployable energy: "
          f"{np.mean([r['rival_err'] for r in rows]):.2f} MJ")
    print(f"n = {len(rows)} closed-loop races at {args.rate} Hz "
          f"({args.follower} chasing {args.leader})")


if __name__ == "__main__":
    main()
