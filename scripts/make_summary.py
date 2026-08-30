#!/usr/bin/env python3
"""One-page summary figure: the whole claim in a single image.

Colour assignment follows the deck's palette, checked with the dataviz
validator against the #0B0B0F surface:

  RED  #E10600  the rival, and our belief about them   ) categorical identity
  AMBER #FFC300 us, and measured accuracy              ) CVD dE 27.1 - PASS
  WHITE #FFFFFF ground truth. Not a categorical slot: a reference ink, always
                direct-labelled, dE 42.5 from red under deuteranopia.
  GREEN #2ECC71 status only (attack), never colour-alone - always carries a
                text label, because green/amber is dE 7.1 under protanopia.
  GRAY  #9AA0A6 recessive ink only. It fails the chroma floor, so it never
                carries series identity - only "held fire", which is labelled.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from _common import base_parser, config_from
from xray.decision import rival_energy_at_zone
from xray.metrics import band_width_by_regime, score_estimate
from xray.render.style import (AMBER, BG, CARD, DIM, GRAY, GREEN, GRIDLINE, RED,
                               WHITE, apply_style)
from xray.render.video import build_scene
from xray.sim import LEADER

MJ = 1e6


def _hairline(ax, axis="y"):
    """Recessive solid grid, one shade off the surface. Never dashed."""
    ax.set_facecolor(BG)
    ax.grid(True, axis=axis, color=GRIDLINE, lw=0.8, ls="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRIDLINE)
        ax.spines[side].set_linewidth(0.8)


def _title(ax, text, sub=None):
    """Title above, subtitle between title and plot. They must not collide."""
    ax.text(0.0, 1.175, text, transform=ax.transAxes, color=WHITE, fontsize=15,
            weight="bold", va="bottom")
    if sub:
        ax.text(0.0, 1.055, sub, transform=ax.transAxes, color=GRAY, fontsize=10.5,
                va="bottom")


# ------------------------------------------------------------------- panels
def panel_reveal(ax, sc) -> None:
    """Hero: what we inferred, against what was actually there."""
    _hairline(ax)
    t = sc.belief.t / 60.0
    belief_lo = sc.belief.usable_p10 / MJ
    belief_hi = sc.belief.usable_p90 / MJ
    ax.fill_between(t, belief_lo, belief_hi, color=RED, alpha=0.30, lw=0, zorder=2)
    ax.plot(t, sc.belief.usable_mean / MJ, color=RED, lw=2.0, zorder=3)
    ax.plot(t, sc.true_usable_rival / MJ, color=WHITE, lw=2.0, zorder=4)

    top = max(belief_hi.max(), (sc.true_usable_rival / MJ).max())
    ax.set_ylim(-0.05, top * 1.42)
    ax.set_xlim(t[0], t[-1] * 1.16)   # right margin reserved for direct labels
    ax.set_ylabel("rival's deployable energy (MJ)", color=GRAY, fontsize=11)
    ax.set_xlabel("race time (minutes)", color=GRAY, fontsize=11)

    # direct labels at the right-hand end, so identity is never colour-alone
    ax.text(t[-1] * 1.02, sc.belief.usable_mean[-1] / MJ + top * 0.16,
            "X-RAY belief", color=RED, fontsize=11.5, va="center", weight="bold")
    ax.text(t[-1] * 1.02, max(sc.true_usable_rival[-1] / MJ - top * 0.02, top * 0.04),
            "ground truth", color=WHITE, fontsize=11.5, va="center", weight="bold")
    ax.legend(handles=[
        Line2D([0], [0], color=RED, lw=7, alpha=0.45, label="belief, p10-p90"),
        Line2D([0], [0], color=RED, lw=2, label="belief, mean"),
        Line2D([0], [0], color=WHITE, lw=2, label="ground truth (never shown to the estimator)"),
    ], loc="lower right", fontsize=10.5, labelcolor=GRAY, ncol=3,
        bbox_to_anchor=(1.0, -0.30))

    # zoomed inset: at full-stint scale the band's breathing is invisible
    lap = np.asarray(sc.obs.lap)
    sel = np.flatnonzero((lap >= 4) & (lap <= 5))
    if len(sel) > 20:
        i0, i1 = int(sel[0]), int(sel[-1])
        axi = ax.inset_axes([0.055, 0.50, 0.32, 0.46])
        axi.set_facecolor(CARD)
        axi.fill_between(t[i0:i1], belief_lo[i0:i1], belief_hi[i0:i1], color=RED,
                         alpha=0.34, lw=0)
        axi.plot(t[i0:i1], sc.belief.usable_mean[i0:i1] / MJ, color=RED, lw=1.8)
        axi.plot(t[i0:i1], sc.true_usable_rival[i0:i1] / MJ, color=WHITE, lw=1.8)
        axi.set_xticks([])
        axi.tick_params(labelsize=9, colors=GRAY)
        axi.grid(True, axis="y", color=GRIDLINE, lw=0.7)
        axi.set_axisbelow(True)
        for sp in axi.spines.values():
            sp.set_color(GRIDLINE)
        axi.text(0.02, 0.90, "two laps, close up", transform=axi.transAxes,
                 color=GRAY, fontsize=10.5, va="top")
        ax.indicate_inset_zoom(axi, edgecolor=DIM, lw=1.0, alpha=0.9)

    dry = np.flatnonzero(sc.belief.dry_events)
    if len(dry):
        k = int(dry[max(len(dry) - 4, 0)])
        ax.annotate("deployment cuts out — the store has hit\nits floor, and the band collapses onto it",
                    xy=(t[k], belief_hi[k]), xytext=(t[k] - 0.4, top * 1.38),
                    color=AMBER, fontsize=10.5, ha="center", va="top",
                    arrowprops=dict(arrowstyle="-", color=AMBER, lw=1.2,
                                    shrinkA=2, shrinkB=4))


def panel_regime(ax, widths: dict) -> None:
    """Nominal categories -> one hue. Bar length already encodes magnitude."""
    _hairline(ax, axis="x")
    order = [("hard acceleration", "accel"), ("braking", "brake"),
             ("steady-state corner", "corner")]
    labels = [lab for lab, _ in order]
    vals = [widths[k] for _, k in order]
    y = np.arange(len(vals))[::-1]
    for yi, v in zip(y, vals):
        ax.add_patch(FancyBboxPatch((0, yi - 0.17), v, 0.34,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=RED, ec="none", alpha=0.85,
                                    mutation_aspect=0.30))
        ax.text(v + 0.035, yi, f"{v:.2f}", color=WHITE, fontsize=12.5,
                va="center", weight="bold")
    ax.set_yticks(y, labels, color=GRAY, fontsize=11.5)
    ax.set_xlim(0, max(vals) * 1.32)
    ax.set_ylim(-0.6, len(vals) - 0.4)
    ax.set_xlabel("belief band width, p10-p90 (MJ)", color=GRAY, fontsize=11)
    ax.tick_params(axis="y", length=0)


def panel_ablation(ax, rates, mape) -> None:
    _hairline(ax)
    ax.plot(rates, mape, color=AMBER, lw=2.0, marker="o", ms=8,
            markerfacecolor=AMBER, markeredgecolor=BG, markeredgewidth=2, zorder=4)
    for y, lab, col in ((8.0, "8% target", GREEN), (15.0, "15% target", RED)):
        ax.axhline(y, color=col, ls="--", lw=1.2, zorder=2)
        ax.text(max(rates) * 1.12, y, lab, color=col, fontsize=9.5, ha="left",
                va="center", bbox=dict(facecolor=BG, edgecolor="none", pad=2.0))
    ax.axvspan(0.85, 2.7, color=RED, alpha=0.12, lw=0, zorder=1)
    ax.text(1.5, max(mape) * 0.52, "breaks\ndown", color=RED, fontsize=11,
            ha="center", weight="bold")
    k = int(np.argmin(np.abs(np.array(rates) - 3.7)))
    ax.annotate(f"3.7 Hz public feed · {mape[k]:.1f}%",
                xy=(rates[k], mape[k]), xytext=(rates[k] * 1.25, max(mape) * 0.66),
                color=WHITE, fontsize=10.5, weight="bold", ha="left",
                arrowprops=dict(arrowstyle="-", color=WHITE, lw=1.2, shrinkB=6))
    ax.set_xscale("log")
    ax.set_xticks(rates)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("telemetry sample rate (Hz)", color=GRAY, fontsize=11)
    ax.set_ylabel("per-lap energy error (%)", color=GRAY, fontsize=11)
    ax.set_ylim(0, max(mape) * 1.18)
    ax.set_xlim(min(rates) * 0.80, max(rates) * 2.6)
    ax.tick_params(labelsize=9.5)
    for lab in ax.get_xticklabels():
        lab.set_rotation(0)


def panel_decision(ax, sc) -> None:
    _hairline(ax)
    laps = np.arange(1, len(sc.tau) + 1)
    ax.plot(laps, sc.tau, color=WHITE, lw=2.0, zorder=3)
    mid = len(laps) // 2
    ax.text(laps[mid], sc.tau[mid] + 0.075, "attack threshold  τ", color=WHITE,
            fontsize=10.5, ha="center", va="bottom", weight="bold")
    took = [o for o in sc.opportunities if o[3]]
    held = [o for o in sc.opportunities if not o[3]]
    if held:
        ax.scatter([o[0] + 1 for o in held], [o[1] for o in held], s=70, c=GRAY,
                   edgecolors=BG, linewidths=2, zorder=4)
    if took:
        ax.scatter([o[0] + 1 for o in took], [o[1] for o in took], s=90, c=GREEN,
                   edgecolors=BG, linewidths=2, zorder=4)
    # Labels sit beside their marks. No leader lines across the plot: a
    # straight line drawn over a scatter reads as a trend series.
    if sc.attack:
        ax.scatter([sc.attack["lap"] + 1], [sc.attack["q"]], s=420,
                   facecolors="none", edgecolors=GREEN, lw=2.5, zorder=5)
        ax.text(sc.attack["lap"] + 1.5, sc.attack["q"] - 0.10,
                f"the call: lap {sc.attack['lap'] + 1}, zone {sc.attack['zone']}",
                color=GREEN, fontsize=10.5, weight="bold", ha="left", va="center")
    if held:
        h = held[0]
        ax.text(h[0] + 1.18, h[1], "held fire", color=GRAY, fontsize=10,
                ha="left", va="center")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, markersize=8,
               label="held fire"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markersize=9,
               label="attack"),
    ], loc="upper right", fontsize=10.5, labelcolor=GRAY, ncol=2)
    ax.set_xlim(0.4, len(sc.tau) + 0.6)
    ax.set_ylim(-0.02, max(max(o[1] for o in sc.opportunities), sc.tau.max()) * 1.62)
    ax.set_xlabel("lap", color=GRAY, fontsize=11)
    ax.set_ylabel("pass probability", color=GRAY, fontsize=11)


def panel_tiles(fig, y, values) -> None:
    """The headline numbers are numbers, not charts."""
    n = len(values)
    for i, (val, label, sub) in enumerate(values):
        x = 0.045 + (0.91 / n) * (i + 0.5)
        fig.text(x, y + 0.036, val, color=AMBER, fontsize=38, ha="center",
                 va="center", weight="bold")
        fig.text(x, y - 0.004, label, color=WHITE, fontsize=12, ha="center", va="center")
        fig.text(x, y - 0.028, sub, color=DIM, fontsize=10, ha="center", va="center")


# --------------------------------------------------------------------- main
def main() -> None:
    ap = base_parser(__doc__)
    args = ap.parse_args()
    cfg = config_from(args)
    out = args.out or "out/xray_summary.png"

    print("building scene ...")
    sc = build_scene(cfg, seed=args.seed, verbose=False)
    score = sc.metrics["_score"]
    comp = sc.metrics["_compare"]
    widths = band_width_by_regime(sc.gt, sc.rival, sc.obs, sc.belief)

    abl = Path("out/ablation.json")
    if abl.exists():
        d = json.loads(abl.read_text())
        rates, mape = d["rates"], d["mape"]
    else:
        raise SystemExit("run scripts/run_ablation.py first (needs out/ablation.json)")

    apply_style()
    fig = plt.figure(figsize=(16, 11.2), dpi=200, facecolor=BG)
    gs = fig.add_gridspec(2, 3, left=0.105, right=0.955, top=0.795, bottom=0.215,
                          hspace=0.70, wspace=0.34, height_ratios=(1.30, 1.0))

    fig.text(0.045, 0.955, "X-RAY", color=WHITE, fontsize=30, weight="bold")
    fig.text(0.045, 0.915,
             "A rival's hidden electrical energy, reconstructed from its speed trace alone — "
             "and used to time an overtake.",
             color=GRAY, fontsize=14)
    fig.text(0.045, 0.884,
             f"Circuit Sigma · {sc.gt.n_laps} laps · {sc.obs.sample_rate_hz:g} Hz public-feed "
             f"telemetry · the estimator sees speed, distance and lap. Nothing else.",
             color=DIM, fontsize=11.5)

    ax_hero = fig.add_subplot(gs[0, :])
    panel_reveal(ax_hero, sc)
    _title(ax_hero, "What we inferred, against what was actually there",
           "the white line was never shown to the estimator")

    ax1 = fig.add_subplot(gs[1, 0])
    panel_regime(ax1, widths)
    _title(ax1, "The band is a belief, not decoration",
           "tightest where the trace is informative")

    ax2 = fig.add_subplot(gs[1, 1])
    panel_ablation(ax2, rates, mape)
    _title(ax2, "Does it survive real data rates?",
           "flat to ~4 Hz, then a cliff")

    ax3 = fig.add_subplot(gs[1, 2])
    panel_decision(ax3, sc)
    riv0 = rival_energy_at_zone(sc.belief, sc.obs, sc.gt.track, sc.gt.n_laps)[0]
    _title(ax3, "When to attack",
           f"lap 1 held — the rival still had {riv0 / MJ:.1f} MJ")

    panel_tiles(fig, 0.098, [
        (f"{score.deployed_mape:.1f}%", "per-lap energy error", "target 8% · at 3.7 Hz"),
        (f"{abs(score.cda_error_pct):.1f}%", "drag area recovered to", "target 6%"),
        (f"{comp['xray_pass_rate']:.0%} vs {comp['blind_pass_rate']:.0%}",
         "pass rate, X-RAY vs blind", f"{comp['n_races']} seeded stints"),
        (f"+{comp['mean_gain']:.2f}", "positions gained",
         f"95% CI [{comp['ci95'][0]:.2f}, {comp['ci95'][1]:.2f}]"),
    ])
    fig.text(0.5, 0.030,
             "All data is generated by our own physics simulator. The estimator receives a noisy, "
             "downsampled speed trace and no energy channel —\nenforced by module boundaries and "
             "verified by an automated test. Ground truth is used only to score the estimator, "
             "after the fact.",
             color=DIM, fontsize=10, ha="center", va="center", linespacing=1.6)

    fig.savefig(out, facecolor=BG)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
