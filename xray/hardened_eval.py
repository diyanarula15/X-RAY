"""Real held-out evaluation of the HARDENED estimator, on P3's exact methodology.

P3's `build_validation_examples` takes the X-RAY energy feature from the persisted
payload's `usable_mean` trace -- i.e. from the OLD estimator. Scoring a hardened
estimator therefore means re-inferring that one feature and substituting it, while
the example set, the leave-one-race-out split, the target definitions, the causal
cutoff rule and the rollout all stay byte-for-byte P3's.

Nothing here re-derives a target, a split or a baseline. `fixed_energy` and
`energy_neutral` are recomputed by P3's own `score_predictions` from the examples
it built, so the comparison is against the same baselines P3 recorded rather than
against new ones.
"""
from __future__ import annotations

import warnings
from dataclasses import replace
from collections import defaultdict

import numpy as np

from .p3_validation import (TARGETS, build_validation_examples, fingerprint,
                            score_predictions)
from .realfit import (belief_from_deployment, build_kin, deployment_trace,
                      fit_nuisance_real)
from .regs import regs_for

HARDENED_EVAL_VERSION = "p35-hardened-real-eval-v1"

# The configuration under test. Kept here as data so the artifact records exactly
# what was scored.
HARDENED = {"boundary": "conditional", "centre": "evidence"}


def hardened_energy_by_driver(year: int, rnd: int, session: str = "R",
                              drivers: list[str] | None = None,
                              estimator: dict | None = None,
                              n_particles: int = 200,
                              mass_kg: float = 790.0) -> dict:
    """Re-infer `usable_mean(t)` per driver with the hardened estimator.

    Uses the existing ingestion path and the existing canonical functions. Returns
    time-indexed arrays so a cutoff can be looked up by interpolation, which
    avoids depending on the payload and the rerun sharing a sample grid.
    """
    est = {**HARDENED, **(estimator or {})}
    import fastf1

    from .analysis import geometry_from_session
    from .data.circuits import RealTrack
    from .data.ingest import CACHE, ingest_session

    fastf1.Cache.enable_cache(str(CACHE))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ses = fastf1.get_session(year, rnd, session)
        ses.load(telemetry=True, laps=True, weather=True)
        track = RealTrack(geometry_from_session(ses))
        sd = ingest_session(year, rnd, session, drivers=drivers, verbose=False)

    rho = float(sd.weather.get("rho", 1.20))
    regs = regs_for(sd.date)
    out = {}
    for drv, df in sd.frames.items():
        d = df[df["usable"]] if "usable" in df else df
        if len(d) < 500:
            continue
        try:
            kin = build_kin(d, track, mass_kg, rho, regs=regs)
            fit = fit_nuisance_real(kin, rho, regs=regs)
            tr = deployment_trace(kin, fit, regs=regs, centre=est["centre"])
            bel = belief_from_deployment(kin, tr, fit, n_particles=n_particles,
                                         boundary=est["boundary"],
                                         reserve_obs=est.get("reserve_obs", "point"))
        except (ValueError, KeyError) as exc:
            out[drv] = {"refused": f"{type(exc).__name__}: {exc}"}
            continue
        t = np.asarray(kin.t, dtype=float)
        u = np.asarray(bel["usable_mean"], dtype=float)
        m = np.isfinite(t) & np.isfinite(u)
        if m.sum() < 100:
            out[drv] = {"refused": "too few finite belief samples"}
            continue
        order = np.argsort(t[m])
        out[drv] = {"t": t[m][order], "usable_j": u[m][order],
                    "identifiability": float(fit.identifiability),
                    "cda_hat": float(fit.cda_hat)}
    return {"race": f"{sd.year}_r{sd.round}_{sd.session}", "drivers": out,
            "estimator": est}


def substitute_energy(examples: list, hardened: dict) -> tuple[list, dict]:
    """Swap the X-RAY feature for the hardened one, at each example's own cutoff.

    An example whose driver could not be re-inferred is DROPPED rather than left
    carrying the old estimator's number: a silent mix of two estimators in one
    column would make the comparison meaningless in a way no metric would reveal.
    """
    per_race = {r["race"]: r["drivers"] for r in hardened}
    out, dropped = [], {}
    for ex in examples:
        drv = per_race.get(ex.race_id, {}).get(ex.driver)
        if not drv or "refused" in drv:
            dropped[ex.race_id] = dropped.get(ex.race_id, 0) + 1
            continue
        e = float(np.interp(ex.cutoff_t, drv["t"], drv["usable_j"]))
        out.append(replace(ex, xray_energy_j=max(e, 0.0)))
    return out, dropped


def run_hardened_real_validation(payload_paths: list, rounds: dict,
                                 estimator: dict | None = None,
                                 n_particles: int = 200) -> dict:
    """P3's real validation, with only the X-RAY energy column replaced."""
    import json
    from pathlib import Path

    payloads = []
    for p in payload_paths:
        with open(p) as fh:
            payloads.append(json.load(fh))

    examples, audit = [], {}
    for pl in payloads:
        ex, a = build_validation_examples(pl)
        examples.extend(ex)
        audit[pl.get("id")] = a

    # Re-inferring 105 full-race driver beliefs is the expensive step, so it is
    # cached to disk keyed by the estimator configuration. A scoring bug should
    # not cost another hour of inference.
    cache_dir = Path("out/p35/hardened_energy")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join(f"{k}-{v}" for k, v in sorted(
        {**HARDENED, **(estimator or {})}.items()))
    hardened = []
    for pl in payloads:
        rid = pl.get("id")
        cp = cache_dir / f"{rid}__{tag}__np{n_particles}.npz"
        if cp.exists():
            z = np.load(cp, allow_pickle=True)
            hardened.append({"race": rid,
                             "drivers": json.loads(str(z["drivers_json"])),
                             "estimator": {**HARDENED, **(estimator or {})}})
            for drv, v in hardened[-1]["drivers"].items():
                if "t" in v:
                    v["t"] = np.asarray(v["t"], dtype=float)
                    v["usable_j"] = np.asarray(v["usable_j"], dtype=float)
            continue
        r = hardened_energy_by_driver(
            int(pl["year"]), int(rounds[rid]), str(pl.get("session", "R")),
            estimator=estimator, n_particles=n_particles)
        r["race"] = rid
        serial = {d: ({**v, "t": list(map(float, v["t"])),
                       "usable_j": list(map(float, v["usable_j"]))}
                      if "t" in v else v) for d, v in r["drivers"].items()}
        np.savez_compressed(cp, drivers_json=json.dumps(serial))
        hardened.append(r)

    new_examples, dropped = substitute_energy(examples, hardened)
    # `score_predictions` takes cfg third and computes the leave-one-race-out
    # split itself. Passing the split there made it the config and cost a full
    # re-inference run to discover.
    scored = score_predictions(payloads, new_examples)
    return {"version": HARDENED_EVAL_VERSION,
            "estimator": {**HARDENED, **(estimator or {})},
            "n_examples_original": len(examples),
            "n_examples_scored": len(new_examples),
            "dropped_by_race": dropped,
            "metrics": scored,
            "targets": list(TARGETS)}


# ============================== estimator-independent baseline (Part 3 §8)
# A grid over the physically possible store, coarse on purpose: the baseline is
# meant to be a fair constant, not a tuned competitor, and a fine grid searched
# on training races would start fitting their noise.
FIXED_E_GRID_J = tuple(np.linspace(0.0, 4.0e6, 21))
FIXED_E_MAX_TRAINING_EXAMPLES = 1200


def _example_key(ex) -> tuple:
    return (
        str(ex.race_id), str(ex.driver), str(ex.target),
        float(ex.cutoff_t), float(ex.target_t), int(ex.lap),
        float(ex.s0), float(ex.target_s),
    )


def _evenly_spaced(rows: list, n: int) -> list:
    if n >= len(rows):
        return list(rows)
    idx = np.floor((np.arange(n, dtype=float) + 0.5) * len(rows) / n).astype(int)
    idx = np.clip(idx, 0, len(rows) - 1)
    return [rows[int(i)] for i in idx]


def deterministic_training_subset(examples: list,
                                  max_examples: int = FIXED_E_MAX_TRAINING_EXAMPLES
                                  ) -> tuple[list, dict]:
    """Balanced deterministic training subset for one target/fold.

    The caller has already removed the held-out race. This function reads only
    identity/time fields, never estimator energy or targets, so the selected
    constant is independent of the estimator candidate being compared.
    """
    grouped = defaultdict(list)
    for ex in examples:
        grouped[str(ex.race_id)].append(ex)
    for race in grouped:
        grouped[race] = sorted(grouped[race], key=_example_key)

    races = sorted(grouped)
    available = {r: len(grouped[r]) for r in races}
    n_available = int(sum(available.values()))
    cap = int(max(0, min(max_examples, n_available)))
    used = {r: 0 for r in races}
    for _ in range(cap):
        candidates = [r for r in races if used[r] < available[r]]
        if not candidates:
            break
        r = min(candidates, key=lambda x: (used[x], x))
        used[r] += 1

    subset = []
    for race in races:
        subset.extend(_evenly_spaced(grouped[race], used[race]))
    subset = sorted(subset, key=_example_key)
    meta = {
        "n_available": n_available,
        "n_used": len(subset),
        "race_counts_available": available,
        "race_counts_used": {r: int(used[r]) for r in races if used[r] > 0},
        "sampling_rule": (
            f"max {max_examples} examples per target/fold; quota balanced across "
            "training races, then midpoint/evenly-spaced deterministic samples "
            "within each race sorted by race/driver/target/cutoff/target/lap/s"
        ),
        "subset_fingerprint": fingerprint([_example_key(ex) for ex in subset]),
    }
    return subset, meta


def train_optimal_fixed_energy(train_examples: list, payloads_by_id: dict,
                               cfg=None, grid=FIXED_E_GRID_J,
                               max_examples: int = FIXED_E_MAX_TRAINING_EXAMPLES
                               ) -> dict:
    """The constant energy that best predicts the TRAINING races, per target.

    P3's `fixed_energy` baseline is the mean of the X-RAY column over the training
    races, so it MOVES whenever the estimator under test changes -- which made
    "new vs fixed-E" uninterpretable across estimator revisions (Part 2 saw it
    shift 21.95 -> 22.62 m/s on future_speed_5s for that reason alone).

    This baseline never reads an estimator. It scores a grid of constant energies
    on the training fold's own targets and freezes the winner, so the number is a
    property of the data and the rollout, identical for every candidate.
    """
    from .p3_validation import rollout_speed
    cfg = cfg or __import__("xray.config", fromlist=["load_config"]).load_config()

    out = {}
    by_target = {}
    for ex in train_examples:
        by_target.setdefault(ex.target, []).append(ex)
    for target, exs in by_target.items():
        sampled, sampling = deterministic_training_subset(exs, max_examples=max_examples)
        best, best_mae = None, float("inf")
        for e in grid:
            errs = []
            for ex in sampled:
                pl = payloads_by_id.get(ex.race_id)
                if pl is None:
                    continue
                errs.append(abs(rollout_speed(pl, ex, float(e), cfg) - ex.target_v))
            if not errs:
                continue
            mae = float(np.mean(errs))
            if mae < best_mae:
                best, best_mae = float(e), mae
        out[target] = {"energy_j": best, "train_mae": best_mae,
                       "n_train": len(exs),
                       "n_train_available": sampling["n_available"],
                       "n_train_used": sampling["n_used"],
                       "race_counts_available": sampling["race_counts_available"],
                       "race_counts_used": sampling["race_counts_used"],
                       "sampling_rule": sampling["sampling_rule"],
                       "subset_fingerprint": sampling["subset_fingerprint"]}
    return out


def score_with_independent_baseline(payloads: list, examples: list,
                                   cfg=None,
                                   max_examples: int = FIXED_E_MAX_TRAINING_EXAMPLES
                                   ) -> dict:
    """Leave-one-race-out scoring of the candidate against two fair baselines.

    `train-optimal-fixed-E` is fitted on the training fold only; `neutral-E` is a
    hard zero and needs no fitting. The candidate's own energy column is whatever
    is already on the examples, so the caller controls which estimator is scored.
    """
    from .p3_validation import leave_one_race_splits, rollout_speed
    cfg = cfg or __import__("xray.config", fromlist=["load_config"]).load_config()
    by_id = {str(p.get("id") or p.get("event")): p for p in payloads}

    rows, chosen = [], {}
    for race, split in leave_one_race_splits(examples).items():
        fixed = train_optimal_fixed_energy(
            split["train"], by_id, cfg, max_examples=max_examples)
        chosen[race] = fixed
        for ex in split["test"]:
            pl = by_id.get(ex.race_id)
            if pl is None:
                continue
            fe = fixed.get(ex.target, {}).get("energy_j")
            if fe is None:
                continue
            rows.append({
                "race": race, "target": ex.target, "truth": ex.target_v,
                "candidate": rollout_speed(pl, ex, ex.xray_energy_j, cfg),
                "train_optimal_fixed": rollout_speed(pl, ex, fe, cfg),
                "neutral": rollout_speed(pl, ex, 0.0, cfg)})
    metrics = {}
    for t in sorted({r["target"] for r in rows}):
        sub = [r for r in rows if r["target"] == t]
        metrics[t] = {"n": len(sub)}
        for name in ("candidate", "train_optimal_fixed", "neutral"):
            err = np.abs([r[name] - r["truth"] for r in sub])
            metrics[t][name] = {"mae": float(np.mean(err)),
                                "rmse": float(np.sqrt(np.mean(err ** 2)))}
        metrics[t]["candidate_minus_train_optimal_fixed"] = (
            metrics[t]["candidate"]["mae"] - metrics[t]["train_optimal_fixed"]["mae"])
        metrics[t]["candidate_minus_neutral"] = (
            metrics[t]["candidate"]["mae"] - metrics[t]["neutral"]["mae"])
    return {"metrics": metrics, "chosen_fixed_energy_by_fold": chosen,
            "n_rows": len(rows),
            "baseline_note": "train-optimal-fixed-E is selected on training races "
                             "only and never reads an estimator output"}
