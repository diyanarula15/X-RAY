"""Content-addressed disk cache. Standard library only.

The cache is not an optimisation here, it is the resumability mechanism. A run
against a daily request cap stops part-way by design; the next run has to pick
up exactly where it left off, and re-judging a situation that was already judged
would spend quota to overwrite an artefact with a slightly different one.

A cache entry is the source of truth: it holds the transcript, the verdict, the
usage and the exception if there was one. `out/judge/verdicts/` is a
materialised view that can be rebuilt from here at any time with no network.
"""
from __future__ import annotations

import hashlib
import json

from .paths import CACHE, write_atomic


def evidence_fingerprint(opening: dict) -> str:
    """Hash of the opening evidence, so a bundle rebuilt with different numbers
    under the same mtime still misses the cache.

    `artefact_mtime_ns` alone would not catch it: a bundle can be rebuilt by
    `scripts/16 --force` from an unchanged race artefact, and if the decision
    code changed in between, the numbers move while the stamp does not.
    """
    blob = json.dumps(opening, sort_keys=True, allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def cache_key(*, situation_key: str, artefact_mtime_ns, evidence_fp: str,
              model_id: str, prompt_version: str, rubric_version: str,
              judge_version: str) -> str:
    """The seven things that make a verdict what it is.

    The API key is deliberately absent: it is not part of what determines the
    answer, and a key in a cache key is a key on disk.
    """
    blob = json.dumps({
        "situation_key": situation_key,
        "artefact_mtime_ns": artefact_mtime_ns,
        "evidence_fp": evidence_fp,
        "model_id": model_id,
        "prompt_version": prompt_version,
        "rubric_version": rubric_version,
        "judge_version": judge_version,
    }, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def path_for(key: str):
    return CACHE / f"{key}.json"


def get(key: str) -> dict | None:
    try:
        return json.loads(path_for(key).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None          # a truncated entry is re-judged, never served


def put(key: str, entry: dict) -> None:
    write_atomic(path_for(key), json.dumps(entry, allow_nan=False))


def entries():
    """Every cache entry on disk, for `--materialize` and `--report`."""
    for p in sorted(CACHE.glob("*.json")):
        try:
            yield json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
