"""The stratified sample, and the manifest that makes it reproducible.

Standard library only.

`matches_recommendation` is used HERE and only here, to build the strata, and is
stripped from everything the model sees (`evidence.redact_outcome`). Stratifying
on it keeps the sample balanced across agree and diverge cases so the report's
diagnostic cross-tab has cells to fill; it is not a target and the judge never
learns it.
"""
from __future__ import annotations

import hashlib
import json
import math
import random

from . import JUDGE_VERSION
from . import evidence as ev
from .paths import DECISIONS_INDEX, SAMPLES, write_atomic
from .prompt import PROMPT_VERSION

STRATA_SPEC = ("race", "call", "agreement", "refused")


def _cell(race: str, attack: bool, matches, refused: bool) -> str:
    call = "ATTACK" if attack else "HOLD"
    agreement = ("matched" if matches is True
                 else "diverged" if matches is False else "unresolved")
    return f"{race}|{call}|{agreement}|{'refused' if refused else 'solved'}"


def _cell_rng(seed: int, cell: str) -> random.Random:
    """Per-cell seed derivation, so adding a race changes that race's draw and
    leaves every other cell's selection bit-identical. A single global RNG
    reshuffles everything the moment the corpus grows, which makes two runs
    incomparable for a reason that has nothing to do with the judge."""
    h = hashlib.sha256(f"{seed}|{cell}".encode("utf-8")).hexdigest()[:8]
    return random.Random(int(h, 16))


def _index_fingerprint() -> str:
    try:
        blob = DECISIONS_INDEX.read_text(encoding="utf-8")
    except OSError:
        blob = ""
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def enumerate_population(races: list[str] | None = None) -> tuple[dict, list]:
    """Every judgeable item on disk, bucketed by stratum.

    A refused pair contributes exactly one pair-level item (key `...@refusal`);
    a solved pair contributes one item per situation.
    """
    index = ev.read_decisions_index()
    cells: dict[str, list[str]] = {}
    refusals: list[str] = []
    for race, car, rival in ev.pairs_on_disk():
        if races and race not in races:
            continue
        try:
            bundle = ev.load_bundle(race, car, rival)
        except ev.EvidenceError:
            continue
        refusal = ev.bundle_refusal(bundle, race, car, rival, index)
        if refusal:
            key = ev.key_of(race, car, rival, None)
            refusals.append(key)
            cells.setdefault(_cell(race, False, None, True), []).append(key)
            continue
        for s in bundle.get("situations") or []:
            t = s.get("decision_time_s")
            if t is None:
                continue
            cell = _cell(race, bool(s.get("attack")),
                         s.get("matches_recommendation"), False)
            cells.setdefault(cell, []).append(ev.key_of(race, car, rival, t))
    return cells, refusals


def build_sample(n: int = 240, seed: int = 42,
                 races: list[str] | None = None,
                 full: bool = False) -> dict:
    cells, refusals = enumerate_population(races)
    n_races = len({c.split("|")[0] for c in cells}) or 1

    if full:
        selected = sorted(k for keys in cells.values() for k in keys)
    else:
        # 1. Census the refusals. They are rare (3 of 102 pairs) and they are
        #    the cases where declining was the output, which is exactly what
        #    this rubric's refusal_discipline axis exists to check. Sampling
        #    them out would bias the corpus toward situations where the engine
        #    was willing to speak.
        picked: list[str] = list(refusals)
        remaining = max(0, n - len(picked))

        solved = {c: ks for c, ks in cells.items()
                  if not c.endswith("|refused")}

        # 2. Floor: one from every non-empty cell, so every race, both calls and
        #    all three agreement states appear even when one dominates by count.
        pool: dict[str, list[str]] = {}
        for cell, keys in sorted(solved.items()):
            shuffled = list(keys)
            _cell_rng(seed, cell).shuffle(shuffled)
            pool[cell] = shuffled
        for cell in sorted(pool):
            if remaining <= 0:
                break
            if pool[cell]:
                picked.append(pool[cell].pop(0))
                remaining -= 1

        # 3. Proportional remainder, capped per race so one long race cannot
        #    dominate a sample meant to describe ten of them.
        per_race_cap = math.ceil(1.5 * n / n_races)
        by_race: dict[str, int] = {}
        for k in picked:
            by_race[k.split("__")[0]] = by_race.get(k.split("__")[0], 0) + 1

        total_left = sum(len(v) for v in pool.values())
        order = sorted(pool, key=lambda c: (-len(pool[c]), c))
        while remaining > 0 and total_left > 0:
            progressed = False
            for cell in order:
                if remaining <= 0:
                    break
                race = cell.split("|")[0]
                if not pool[cell] or by_race.get(race, 0) >= per_race_cap:
                    continue
                picked.append(pool[cell].pop(0))
                by_race[race] = by_race.get(race, 0) + 1
                remaining -= 1
                total_left -= 1
                progressed = True
            if not progressed:
                break
        selected = picked

    selected = sorted(set(selected))
    counts: dict[str, int] = {}
    lookup = {k: c for c, ks in cells.items() for k in ks}
    for k in selected:
        c = lookup.get(k, "?")
        counts[c] = counts.get(c, 0) + 1

    fingerprint = _index_fingerprint()
    sid = hashlib.sha256(json.dumps(
        {"seed": seed, "spec": STRATA_SPEC, "n": n, "full": full,
         "races": sorted(races or []), "fp": fingerprint},
        sort_keys=True).encode("utf-8")).hexdigest()[:12]

    return {
        "sample_id": sid,
        "seed": seed,
        "n_requested": n,
        "n_selected": len(selected),
        "full": full,
        "races": sorted(races or []),
        "strata_spec": list(STRATA_SPEC),
        "decisions_index_fingerprint": fingerprint,
        "population_size": sum(len(v) for v in cells.values()),
        "n_refusals_censused": len(refusals),
        "judge_version": JUDGE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "counts": counts,
        "keys": selected,
    }


def write_sample(manifest: dict):
    p = SAMPLES / f"{manifest['sample_id']}.json"
    write_atomic(p, json.dumps(manifest, indent=1, allow_nan=False))
    return p


def load_sample(sample_id: str) -> dict:
    p = SAMPLES / f"{sample_id}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ev.EvidenceError(f"no sample manifest {sample_id}") from exc
