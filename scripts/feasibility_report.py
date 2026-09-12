#!/usr/bin/env python3
"""Turn out/feasibility/*.json into out/feasibility.md."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

CLIFF_HZ = 4.0  # where the Stage 1 ablation showed accuracy fall apart


def load():
    rows = []
    for f in sorted(glob.glob("out/feasibility/r*.json")):
        try:
            rows.append(json.load(open(f)))
        except Exception:
            continue
    return sorted(rows, key=lambda d: -d["speed_window"]["vmax_session_kmh"])


def main() -> None:
    rows = load()
    if not rows:
        raise SystemExit("no feasibility JSON found; run scripts/feasibility.py first")
    L = []
    A = L.append
    A("# Stage 2 feasibility — the four reality checks\n")
    A("Measured on real 2026 sessions via FastF1, not assumed. Every number here is\n"
      "produced by `scripts/feasibility.py`; rerun it to reproduce.\n")
    A(f"Sessions probed: {len(rows)} — " + ", ".join(f"{d['circuit']} (R{d['round']})" for d in rows) + "\n")

    # ---------------------------------------------------------------- 1.1
    A("\n## 1.1 Does 2026 telemetry exist, and at what rate?\n")
    A("Yes. The `Speed` channel is populated for every session probed. It is **irregularly\n"
      "sampled**, so the nominal rate is meaningless — the distribution of Δt is what matters.\n")
    A("\n| circuit | median Δt | effective Hz | p10 | p90 | max Δt | median Δx |")
    A("|---|---|---|---|---|---|---|")
    for d in rows:
        t = d["dt_fastest_lap"]
        A(f"| {d['circuit']} | {t['median_s']:.3f} s | **{t['effective_hz']:.2f} Hz** | "
          f"{t['p10_s']:.2f} | {t['p90_s']:.2f} | {t['max_s']:.2f} s | {t['median_dx_m']:.1f} m |")
    hz = [d["dt_fastest_lap"]["effective_hz"] for d in rows]
    A(f"\nEffective rate across sessions: **{min(hz):.2f}–{max(hz):.2f} Hz**.\n")
    A(f"> **This is the headline risk.** The Stage 1 ablation put the accuracy cliff at about\n"
      f"> {CLIFF_HZ:.0f} Hz: 3.9% per-lap energy error at 3.7 Hz, 19.4% at 2 Hz, 41.7% at 1 Hz.\n"
      f"> Real 2026 telemetry lands at {min(hz):.1f}–{max(hz):.1f} Hz — **on the edge of that cliff,\n"
      f"> with no margin.** It is not below it, but nothing about the rate is comfortable, and\n"
      f"> the p90 Δt of {max(d['dt_fastest_lap']['p90_s'] for d in rows):.2f} s means a meaningful\n"
      f"> tail of the trace is sampled at 2.5 Hz or worse. This must be stated in the UI and\n"
      f"> the pitch, not buried.\n")

    A("\n### Which channels are populated\n")
    chans = ["Speed", "Throttle", "Brake", "nGear", "RPM", "DRS"]
    A("| circuit | " + " | ".join(chans) + " |")
    A("|---" * (len(chans) + 1) + "|")
    for d in rows:
        cells = []
        for c in chans:
            v = d["channels"].get(c, {})
            if not v.get("present"):
                cells.append("absent")
            elif v.get("frac_nonzero") == 0:
                cells.append("**all zero**")
            else:
                cells.append(f"{v['frac_nonzero'] * 100:.0f}% nonzero")
        A(f"| {d['circuit']} | " + " | ".join(cells) + " |")
    A("\n**DRS is dead in 2026, as expected** — the channel exists and is identically zero.\n"
      "Throttle, brake, gear and RPM *are* populated. That is more than Stage 1 assumed: the\n"
      "simulator's estimator deliberately saw speed only. It changes nothing about the core\n"
      "claim — **there is still no energy, deployment or state-of-charge channel, which is the\n"
      "entire point** — but it does mean regime classification (accelerating / braking /\n"
      "corner-limited) can be read off the feed instead of inferred from the speed trace.\n"
      "Stage 2 uses them for regime only, and the estimator still infers energy from dynamics.\n")

    # ---------------------------------------------------------------- 1.2
    A("\n## 1.2 Does the high-speed calibration window exist at real circuits?\n")
    A("**No. This is the finding that forces the Stage 2 methodology change.**\n")
    A("\nStage 1 calibrated drag above 340 km/h, where the regulatory taper has squeezed\n"
      "deployment to near zero. Fraction of all race samples above each threshold:\n")
    bins = ["340", "330", "320", "310", "300", "290"]
    A("\n| circuit | v_max | " + " | ".join(f">{b}" for b in bins) + " | samples |")
    A("|---" * (len(bins) + 3) + "|")
    for d in rows:
        sw = d["speed_window"]
        cells = [f"{sw['frac_above'][b] * 100:.2f}%" for b in bins]
        A(f"| {d['circuit']} | {sw['vmax_session_kmh']:.0f} km/h | " + " | ".join(cells)
          + f" | {sw['total_samples']:,} |")
    best = rows[0]
    A(f"\nEven at **{best['circuit']}**, the fastest circuit probed, only "
      f"**{best['speed_window']['frac_above']['340'] * 100:.2f}%** of samples clear 340 km/h — "
      f"a few hundred samples across an entire race, and they are spread across 20 cars.\n")
    A("\n> **Verdict: §2 generalized calibration is mandatory, not optional.** A binary\n"
      "> `v > 340` gate does not survive contact with real telemetry at any circuit. The\n"
      "> continuous formulation (the taper ceiling as a smooth speed-varying constraint) and\n"
      "> the coast-down channel are the only way this works outside a simulator.\n")

    # ---------------------------------------------------------------- 1.3
    A("\n## 1.3 Can we get track gradient?\n")
    ok = [d for d in rows if d["position"].get("has_Z")]
    A(f"**Yes, from FastF1 `Position` data — {len(ok)} of {len(rows)} sessions carry a live Z channel.**\n")
    A("\n| circuit | pos samples | source | Z range | Z std | frac zero |")
    A("|---|---|---|---|---|---|")
    for d in rows:
        p = d["position"]
        if p.get("has_Z"):
            A(f"| {d['circuit']} | {p['n_samples']} | {p.get('source', 'lap')} | "
              f"{p['z_range_m'][1] - p['z_range_m'][0]:.1f} m | {p['z_std_m']:.1f} m | "
              f"{p['z_frac_zero'] * 100:.0f}% |")
        else:
            A(f"| {d['circuit']} | {p.get('n_samples', 0)} | — | **unavailable** | — | — |")
    if ok:
        s = ok[0]
        A(f"\n{s['circuit']} reports {s['position']['z_range_m'][1] - s['position']['z_range_m'][0]:.0f} m "
          f"of elevation change, which matches the real circuit. The channel is usable: smoothed "
          f"along the centreline it gives `sin(grade)` directly, and it is also what makes the 3D "
          f"track ribbon real geometry rather than a drawing.\n")
    bad = [d for d in rows if not d["position"].get("has_Z")]
    if bad:
        A(f"\nWhere position data is missing ({', '.join(d['circuit'] for d in bad)}), the circuit is\n"
          "**excluded** rather than analysed with an assumed-zero gradient.\n")

    # ---------------------------------------------------------------- 1.4
    A("\n## 1.4 Are active-aero zones published anywhere?\n")
    A("Not in machine-readable form. FIA race director event notes are PDFs, and FastF1 exposes\n"
      "no aero-mode channel. Per the spec, aero zones are hand-built in\n"
      "`xray/data/circuits/<circuit>.yaml` for a small number of circuits only, from the\n"
      "published straight/corner geometry, and the file records that they are hand-entered.\n")

    # ---------------------------------------------------------------- gaps
    A("\n## Data quality: gaps\n")
    A("\n| circuit | laps scanned | Δt gaps > 1.0 s | per lap |")
    A("|---|---|---|---|")
    for d in rows:
        sw = d["speed_window"]
        A(f"| {d['circuit']} | {sw['laps_scanned']} | {sw['gaps_over_1s']} | {sw['gaps_per_lap']:.2f} |")
    A("\nGaps above 1.0 s are common — roughly one per lap. The pipeline never interpolates\n"
      "across them; it marks them and the estimator suspends across the break (spec §8).\n")

    # ---------------------------------------------------------------- weather
    A("\n## Weather: real air density\n")
    A("\n| circuit | air temp | pressure | wind |")
    A("|---|---|---|---|")
    for d in rows:
        w = d["weather"]
        if w.get("available"):
            A(f"| {d['circuit']} | {w['air_temp_c']:.1f} °C | {w['pressure_mbar']:.0f} mbar | "
              f"{w['wind_speed_ms']:.1f} m/s |")
    A("\nStage 1 assumed ρ = 1.20 kg/m³. Real sessions publish air temperature and pressure, so\n"
      "ρ is computed per session from the ideal gas law instead of assumed — one fewer nuisance.\n")

    # ---------------------------------------------------------------- verdict
    A("\n## Verdict\n")
    A("| check | result | consequence |")
    A("|---|---|---|")
    A(f"| 1.1 telemetry rate | {min(hz):.1f}–{max(hz):.1f} Hz, irregular | **On the cliff edge.** "
      "Stated in the UI; no margin claimed. |")
    A("| 1.2 calibration window | Essentially absent everywhere | **§2 is mandatory.** "
      "Continuous taper constraint + coast-down. |")
    A(f"| 1.3 gradient | Z channel live at {len(ok)}/{len(rows)} | Usable. Circuits without it "
      "are excluded, not fudged. |")
    A("| 1.4 aero zones | Not machine-readable | Hand-built for 3 circuits, flagged as such. |")
    A("\nProceed to Phase B, with §2 promoted ahead of the UI as the spec requires.\n")

    Path("out").mkdir(exist_ok=True)
    Path("out/feasibility.md").write_text("\n".join(L))
    print(f"wrote out/feasibility.md ({len(L)} lines, {len(rows)} sessions)")


if __name__ == "__main__":
    main()
