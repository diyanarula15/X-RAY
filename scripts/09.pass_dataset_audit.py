#!/usr/bin/env python3
"""Audit the real-data pass dataset BEFORE any fitting (P2 step 6).

Reports what the available payloads can and cannot support. Exits non-zero only
on an error; a blocked audit is a successful, informative run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xray.passmodel import (CONFIRMATION_LAPS, audit_dataset, build_dataset,
                            dataset_fingerprint, load_payloads)  # noqa: E402

RACES = Path(__file__).resolve().parent.parent / "out" / "races"


def main() -> None:
    payloads = load_payloads(RACES)
    if not payloads:
        print(f"no analysed race payloads in {RACES}")
        return
    samples = []
    for p in payloads:
        samples += build_dataset(p)
    a = audit_dataset(samples, payloads)

    print("=" * 74)
    print("P2 PASS-MODEL DATASET AUDIT")
    print("=" * 74)
    print(f"  payloads                 {len(payloads)}")
    print(f"  label                    attacker behind at lap L, ahead at L+1,")
    print(f"                           still ahead through L+{CONFIRMATION_LAPS}")
    print(f"  raw candidate rows       {a.n_raw}")
    print(f"  excluded                 {a.n_excluded}")
    for why, n in sorted(a.exclusions.items(), key=lambda kv: -kv[1]):
        print(f"      {n:5d}  {why}")
    print(f"  usable samples           {a.n_usable}")
    print(f"  positive labels          {a.n_positive} "
          f"({100 * a.positive_rate:.1f}%)")
    print(f"  race groups              {len(a.races)}  {a.races}")
    print(f"  circuits                 {len(a.circuits)}  {a.circuits}")
    print(f"  fingerprint              {dataset_fingerprint(samples)}")
    print()
    print("  FEATURES AVAILABLE (decision-time only)")
    for f in a.available_features:
        print(f"      {f}")
    print()
    print("  SIGNALS NOT AVAILABLE")
    for m in a.missing_signals:
        print(f"      {m}")
    print()
    print("=" * 74)
    if a.fit_permitted:
        print("AUDIT PASSED -- fitting is permitted")
    else:
        print("AUDIT BLOCKED -- p_pass REMAINS SYNTHETIC")
        for r in a.blocking_reasons:
            print(f"  - {r}")
    print("=" * 74)


if __name__ == "__main__":
    main()
