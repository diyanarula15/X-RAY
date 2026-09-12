#!/usr/bin/env python3
"""Does the decision engine actually find the right lap to attack?

The honest way to ask this is to build a case where the true answer is known,
then hide it. For each seeded race we compute three answers:

  ORACLE  the DP solved against the rival's TRUE deployable energy, taken from
          the simulator. This is the best any method could do. It is the number
          to beat and it is not achievable in practice.
  X-RAY   the same DP solved against the rival energy RECONSTRUCTED from a
          noisy, downsampled speed trace. This is what we actually ship.
  BLIND   attack on the first lap the battery allows. This is what you do when
          the energy channel does not exist.

All three are then scored on the SAME ground truth: what was the real chance of
the pass on the lap each one chose? The gap between X-RAY and ORACLE is the cost
of not being able to see; the gap between X-RAY and BLIND is what the estimator
buys you.
"""
from __future__ import annotations

import argparse
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
from xray.sim import FOLLOWER, LEADER, run_sim
from xray.vehicle import VehicleParams


def true_rival_track(gt, model, n_laps: int) -> np.ndarray:
    """The rival's REAL deployable energy at the zone where a pass would happen.

    Ground truth. Used only to build the oracle and to score, never by X-RAY.
    """
    trace = gt.cars[LEADER]
    zone = gt.track.zone_by_name("A")
    reserve = get_policy(gt.policies[LEADER]).reserve * E_STORE_MAX
    out = np.zeros(n_laps)
    for L in range(n_laps):
        m = (trace.lap == L)
        if not m.any():
            out[L] = out[L - 1] if L else 0.0
            continue
        idx = np.flatnonzero(m)
        j = idx[int(np.argmin(np.abs(trace.s[idx] - zone.s_straight_start)))]
        laps_left = max(gt.n_laps - L, 0)
        floor = reserve * min(1.0, laps_left / RESERVE_RELEASE_LAPS)
        out[L] = max(float(trace.E[j] - floor), 0.0)
    return out


def own_energy(gt, n_laps: int) -> np.ndarray:
    trace = gt.cars[FOLLOWER]
    return np.array([float(trace.E[trace.lap == L].max()) if (trace.lap == L).any()
                     else 0.0 for L in range(n_laps)])


def q_curve(model, own: np.ndarray, rival: np.ndarray) -> np.ndarray:
    """Real pass probability on each lap, given both energy states."""
    out = np.zeros(len(own))
    for i in range(len(own)):
        out[i] = max(p_pass(delta_v(zm, min(own[i], model.attack_cost),
                                    min(rival[i], model.attack_cost)),
                            model.gap_s, zm) for zm in model.zones)
    return out


def first_affordable(model, own: np.ndarray) -> int:
    for i, e in enumerate(own):
        if e >= model.attack_cost * 0.75:
            return i
    return 0


def run_seed(cfg, seed: int, rate_hz: float, particles: int) -> dict:
    gt = run_sim(cfg, seed=seed)
    n_laps = gt.n_laps
    params = VehicleParams.from_config(cfg)

    obs = observe(gt, LEADER, rate_hz=rate_hz,
                  speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=seed + 1)
    belief = estimate(obs, gt.track, n_particles=particles, seed=seed + 2)

    recharge = float(np.mean(belief.harvested_lap[belief.harvested_lap > 0]))
    spend = float(np.mean(belief.deployed_lap[belief.deployed_lap > 0]))
    own_spend = float(np.mean(gt.cars[FOLLOWER].deployed_lap)) * 0.55
    model = build_model(gt.track, params, n_laps=n_laps, recharge_per_lap=recharge,
                        rival_spend_per_lap=spend, own_spend_per_lap=own_spend)

    own = own_energy(gt, n_laps)
    truth = true_rival_track(gt, model, n_laps)
    seen = rival_energy_at_zone(belief, obs, gt.track, n_laps)

    # the real chance of a pass on each lap -- the yardstick for all three
    q_true = q_curve(model, own, truth)

    oracle_sol = solve_exogenous(model, truth)
    xray_sol = solve_exogenous(model, seen)

    def first_call(sol) -> int | None:
        for L in range(n_laps):
            if sol.action(n_laps - L, own[L]) is not None:
                return L
        return None

    lap_oracle = first_call(oracle_sol)
    lap_xray = first_call(xray_sol)
    lap_blind = first_affordable(model, own)

    # The engine does not maximise p(pass); it maximises the share of the stint
    # spent in front, so a 0.53 chance on lap 2 beats a 0.63 chance on lap 9.
    # Scoring on p(pass) alone would mark the engine down for being right.
    reward = np.array([(n_laps - L) / n_laps for L in range(n_laps)])
    ev_true = q_true * reward
    lap_best_q = int(np.argmax(q_true))
    lap_best_ev = int(np.argmax(ev_true))

    # A fairer blind baseline than "attack on lap 1": a driver with no read on
    # the rival's energy, picking a lap they can afford, at random.
    rng = np.random.default_rng(seed)
    afford = [L for L in range(n_laps) if own[L] >= model.attack_cost * 0.75]
    lap_rand = int(rng.choice(afford)) if afford else 0

    def q(L):
        return float(q_true[L]) if L is not None else 0.0

    def ev(L):
        return float(ev_true[L]) if L is not None else 0.0

    return {
        "seed": seed,
        "lap_best_q": lap_best_q, "q_best": float(q_true.max()),
        "lap_best_ev": lap_best_ev, "ev_best": float(ev_true.max()),
        "lap_oracle": lap_oracle, "q_oracle": q(lap_oracle), "ev_oracle": ev(lap_oracle),
        "lap_xray": lap_xray, "q_xray": q(lap_xray), "ev_xray": ev(lap_xray),
        "lap_blind": lap_blind, "q_blind": q(lap_blind), "ev_blind": ev(lap_blind),
        "lap_rand": lap_rand, "q_rand": q(lap_rand), "ev_rand": ev(lap_rand),
        "rival_err_mj": float(np.mean(np.abs(seen - truth)) / 1e6),
        "q_true": q_true, "truth": truth, "seen": seen,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--rate", type=float, default=3.7)
    ap.add_argument("--particles", type=int, default=300)
    ap.add_argument("--leader", default="AGGRESSIVE")
    ap.add_argument("--follower", default="CONSERVATIVE")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = merge(load_config(), sim={"leader_policy": args.leader,
                                    "follower_policy": args.follower})
    rows = []
    for i in range(args.seeds):
        try:
            r = run_seed(cfg, 100 + i, args.rate, args.particles)
        except Exception as exc:  # noqa: BLE001
            print(f"  seed {100+i}: skipped ({exc})")
            continue
        rows.append(r)
        if args.verbose:
            print(f"  seed {r['seed']}: optimum lap {r['lap_best_ev']+1} "
                  f"(EV {r['ev_best']:.3f}) | oracle {(r['lap_oracle'] or 0)+1} "
                  f"EV {r['ev_oracle']:.3f} | X-RAY {(r['lap_xray'] or 0)+1} "
                  f"EV {r['ev_xray']:.3f} | blind {r['lap_blind']+1} "
                  f"EV {r['ev_blind']:.3f}")

    if not rows:
        raise SystemExit("no usable seeds")

    def arr(k):
        return np.array([r[k] for r in rows], dtype=float)

    print(f"\n{'':24s}{'EV':>9s}{'p(pass)':>10s}{'lap':>7s}   what it is")
    print("-" * 76)
    print(f"{'best possible (EV)':24s}{arr('ev_best').mean():9.3f}"
          f"{arr('q_best').mean():10.3f}{arr('lap_best_ev').mean()+1:7.1f}"
          f"   unreachable ceiling")
    print(f"{'ORACLE (true energy)':24s}{arr('ev_oracle').mean():9.3f}"
          f"{arr('q_oracle').mean():10.3f}{arr('lap_oracle').mean()+1:7.1f}"
          f"   the DP with nothing hidden")
    print(f"{'X-RAY (from speed)':24s}{arr('ev_xray').mean():9.3f}"
          f"{arr('q_xray').mean():10.3f}{arr('lap_xray').mean()+1:7.1f}"
          f"   what we ship")
    print(f"{'BLIND (first chance)':24s}{arr('ev_blind').mean():9.3f}"
          f"{arr('q_blind').mean():10.3f}{arr('lap_blind').mean()+1:7.1f}"
          f"   attack as soon as able")
    print(f"{'BLIND (random lap)':24s}{arr('ev_rand').mean():9.3f}"
          f"{arr('q_rand').mean():10.3f}{arr('lap_rand').mean()+1:7.1f}"
          f"   fairer baseline")

    lap_or = np.array([r["lap_oracle"] if r["lap_oracle"] is not None else -1 for r in rows])
    lap_xr = np.array([r["lap_xray"] if r["lap_xray"] is not None else -1 for r in rows])
    same = float(np.mean(lap_xr == lap_or))
    within1 = float(np.mean(np.abs(lap_xr - lap_or) <= 1))
    print(f"\nAgreement with the oracle: exact lap {same*100:.0f}%, "
          f"within one lap {within1*100:.0f}%")

    for name in ("blind", "rand"):
        d = arr("ev_xray") - arr(f"ev_{name}")
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        label = "first-chance" if name == "blind" else "random-lap"
        print(f"X-RAY minus {label:12s}: {d.mean():+.3f} EV, 95% CI "
              f"[{d.mean()-1.96*se:+.3f}, {d.mean()+1.96*se:+.3f}]"
              f"  {'SIGNIFICANT' if d.mean()-1.96*se > 0 else 'not significant'}")

    gap = arr("ev_oracle").mean() - arr("ev_rand").mean()
    got = arr("ev_xray").mean() - arr("ev_rand").mean()
    if gap > 1e-9:
        print(f"X-RAY closes {100*got/gap:.0f}% of the distance from a random "
              f"guess to the oracle")
    print(f"\nMean error in the rival's deployable energy: "
          f"{arr('rival_err_mj').mean():.2f} MJ")
    print(f"n = {len(rows)} seeded races at {args.rate} Hz. Note: X-RAY can score "
          f"above the oracle on\nindividual seeds -- that is estimation error "
          f"landing favourably, not skill.")


if __name__ == "__main__":
    main()
