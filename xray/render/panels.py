"""Individual plot panels. Each takes an axis and draws one frame.

Every number a panel displays comes from the pipeline. Nothing here is a
caption typed in by hand.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from ..constants import E_STORE_MAX, TAPER_V_END, TAPER_V_START
from .style import AMBER, BG, CARD, DIM, GRAY, GREEN, GRIDLINE, RED, WHITE, strip

KMH = 3.6


@dataclass
class Scene:
    """Everything the panels draw, assembled once before rendering."""
    gt: object
    obs: object
    belief: object
    ours: str
    rival: str
    true_soc_rival: np.ndarray     # at obs sample times
    true_usable_rival: np.ndarray
    our_soc: np.ndarray            # at obs sample times
    idx_gt: np.ndarray             # obs sample -> gt index
    tau: np.ndarray                # threshold per lap
    opportunities: list            # (lap, q, zone, taken)
    attack: dict | None
    metrics: dict
    counterfactual: dict


# ------------------------------------------------------------------- battery
def panel_battery_bars(ax, sc: Scene, k: int, reveal_band: bool = True,
                       show_question: bool = False) -> None:
    """The hero visual: our exact bar, and the rival's breathing belief band."""
    strip(ax)
    ax.set_xlim(-0.06, 1.12)
    ax.set_ylim(0, 1)

    def bar_bg(y):
        ax.add_patch(FancyBboxPatch((0.0, y), 1.0, 0.17, boxstyle="round,pad=0.004",
                                    fc=CARD, ec=GRIDLINE, lw=1.0, zorder=1))

    ours = float(np.clip(sc.our_soc[k] / E_STORE_MAX, 0, 1))
    bar_bg(0.60)
    ax.add_patch(Rectangle((0.0, 0.60), ours, 0.17, fc=AMBER, ec="none", zorder=2))
    ax.text(0.0, 0.80, "YOUR CAR", color=GRAY, fontsize=13, weight="bold")
    ax.text(1.02, 0.665, f"{sc.our_soc[k] / 1e6:.2f} MJ", color=AMBER, fontsize=15,
            va="center", weight="bold")

    bar_bg(0.28)
    ax.text(0.0, 0.48, "RIVAL", color=GRAY, fontsize=13, weight="bold")
    if show_question:
        ax.text(0.5, 0.365, "?", color=RED, fontsize=44, ha="center", va="center",
                weight="bold")
        ax.text(1.02, 0.345, "no channel", color=DIM, fontsize=13, va="center")
        return

    if reveal_band:
        p10 = float(np.clip(sc.belief.usable_p10[k] / E_STORE_MAX, 0, 1))
        p90 = float(np.clip(sc.belief.usable_p90[k] / E_STORE_MAX, 0, 1))
        mean = float(np.clip(sc.belief.usable_mean[k] / E_STORE_MAX, 0, 1))
        ax.add_patch(Rectangle((0.0, 0.28), p90, 0.17, fc=RED, ec="none",
                               alpha=0.28, zorder=2))
        ax.add_patch(Rectangle((0.0, 0.28), max(p10, 0.0), 0.17, fc=RED, ec="none",
                               alpha=0.95, zorder=3))
        ax.plot([mean, mean], [0.28, 0.45], color=WHITE, lw=2.4, zorder=4)
        ax.text(1.02, 0.345, f"{sc.belief.usable_mean[k] / 1e6:.2f} MJ",
                color=WHITE, fontsize=15, va="center", weight="bold")
        ax.text(1.02, 0.29, f"±{(sc.belief.usable_p90[k] - sc.belief.usable_p10[k]) / 2e6:.2f}",
                color=DIM, fontsize=11, va="center")


def panel_truth_reveal(ax, sc: Scene, k: int, err_pct: float | None = None) -> None:
    """Act 3: drop the true store on top of the band we built blind."""
    ax.set_facecolor(BG)
    ax.grid(True, axis="y", color=GRIDLINE, lw=0.8)
    ax.set_xticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    t = sc.belief.t
    lo, hi = max(0, k - 900), k + 1
    ax.fill_between(t[lo:hi], sc.belief.usable_p10[lo:hi] / 1e6,
                    sc.belief.usable_p90[lo:hi] / 1e6, color=RED, alpha=0.28, lw=0)
    ax.plot(t[lo:hi], sc.belief.usable_mean[lo:hi] / 1e6, color=RED, lw=1.6, alpha=0.9)
    ax.plot(t[lo:hi], sc.true_usable_rival[lo:hi] / 1e6, color=WHITE, lw=2.0)
    ax.set_ylim(-0.1, E_STORE_MAX / 1e6 * 0.92)
    ax.set_ylabel("rival deployable energy (MJ)", color=GRAY, fontsize=11)
    ax.legend(handles=[
        _proxy(RED, "X-RAY belief (p10-p90)", alpha=0.35),
        _proxy(WHITE, "ground truth"),
    ], loc="upper right", fontsize=11, labelcolor=GRAY)
    if err_pct is not None:
        ax.text(0.015, 0.90, f"per-lap energy error: {err_pct:.1f}%", transform=ax.transAxes,
                color=AMBER, fontsize=20, weight="bold", va="top")


def _proxy(color, label, alpha=1.0):
    from matplotlib.lines import Line2D
    return Line2D([0], [0], color=color, lw=6 if alpha < 1 else 2.4, alpha=alpha,
                  label=label)


# --------------------------------------------------------------- speed trace
def panel_speed_traces(ax, sc: Scene, k: int, window_s: float = 26.0) -> None:
    ax.set_facecolor(BG)
    ax.grid(True, axis="y", color=GRIDLINE, lw=0.8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    t = sc.belief.t
    n = max(int(window_s * sc.obs.sample_rate_hz), 8)
    lo, hi = max(0, k - n), k + 1
    gi = sc.idx_gt[lo:hi]
    ax.plot(t[lo:hi], sc.gt.cars[sc.ours].v[gi] * KMH, color=AMBER, lw=1.8, label="you")
    ax.plot(t[lo:hi], sc.obs.v[lo:hi] * KMH, color=RED, lw=1.8, label="rival (public feed)")
    for v, lab in ((TAPER_V_START * KMH, "290 taper starts"),
                   (TAPER_V_END * KMH, "355 deployment zero")):
        ax.axhline(v, color=DIM, ls="--", lw=1.0)
        ax.text(t[lo] if hi > lo else 0, v + 3, lab, color=DIM, fontsize=9)
    ax.set_ylim(40, 385)
    ax.set_xlim(t[lo], max(t[hi - 1], t[lo] + 1))
    ax.set_ylabel("speed (km/h)", color=GRAY, fontsize=11)
    ax.set_xticks([])
    ax.legend(loc="lower right", fontsize=10, labelcolor=GRAY, ncol=2)


# ----------------------------------------------------------------- track map
def panel_track_map(ax, sc: Scene, k: int) -> None:
    strip(ax)
    track = sc.gt.track
    pts = track.xy()
    ax.plot(pts[:, 0], pts[:, 1], color=CARD, lw=13, solid_capstyle="round", zorder=1)
    ax.plot(pts[:, 0], pts[:, 1], color=GRIDLINE, lw=9, solid_capstyle="round", zorder=2)
    for z in track.zones:
        m = ((np.linspace(0, track.length, len(pts)) >= z.s_straight_start)
             & (np.linspace(0, track.length, len(pts)) <= z.s_end))
        ax.plot(pts[m, 0], pts[m, 1], color=DIM, lw=9, solid_capstyle="round", zorder=3)
        c = pts[m][len(pts[m]) // 2] if m.any() else pts[0]
        ax.text(c[0], c[1], z.name, color=WHITE, fontsize=13, weight="bold",
                ha="center", va="center", zorder=6)
    gi = sc.idx_gt[k]
    for car, color in ((sc.ours, AMBER), (sc.rival, RED)):
        p = track.xy_at(sc.gt.cars[car].s[gi])
        dep = sc.gt.cars[car].P_mguk[gi] / 350e3 if car == sc.ours else \
            float(np.clip(sc.belief.p_mguk_mean[k] / 350e3, 0, 1))
        ax.add_patch(Circle(p, 24 + 34 * float(np.clip(dep, 0, 1)), fc=color,
                            ec="none", alpha=0.25, zorder=7))
        ax.add_patch(Circle(p, 24, fc=color, ec=BG, lw=2, zorder=8))
    ax.set_aspect("equal")
    ax.margins(0.09)


# ----------------------------------------------------------------- threshold
def panel_threshold(ax, sc: Scene, k: int) -> None:
    ax.set_facecolor(BG)
    ax.grid(True, axis="y", color=GRIDLINE, lw=0.8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    laps = np.arange(1, len(sc.tau) + 1)
    ax.plot(laps, sc.tau, color=WHITE, lw=2.2, label=r"attack threshold $\tau$")
    lap_now = int(sc.obs.lap[k])
    shown = [o for o in sc.opportunities if o[0] <= lap_now]
    if shown:
        xs = [o[0] + 1 for o in shown]
        ys = [o[1] for o in shown]
        cols = [GREEN if o[3] else GRAY for o in shown]
        ax.scatter(xs, ys, c=cols, s=52, zorder=5, edgecolors="none",
                   label="opportunity quality")
    if sc.attack and lap_now >= sc.attack["lap"]:
        ax.add_patch(Circle((sc.attack["lap"] + 1, sc.attack["q"]), 0.0, fill=False))
        ax.scatter([sc.attack["lap"] + 1], [sc.attack["q"]], s=340, facecolors="none",
                   edgecolors=GREEN, lw=3.0, zorder=6)
        ax.annotate(f"ATTACK  P {sc.attack['q']:.2f}",
                    (sc.attack["lap"] + 1, sc.attack["q"]),
                    textcoords="offset points", xytext=(14, 16), color=GREEN,
                    fontsize=15, weight="bold")
    ax.set_xlim(0.5, len(sc.tau) + 0.5)
    ax.set_ylim(0, max(0.75, max([o[1] for o in sc.opportunities] + [0.4]) * 1.35))
    ax.set_xlabel("lap", color=GRAY, fontsize=11)
    ax.set_ylabel("pass probability", color=GRAY, fontsize=11)
    ax.legend(loc="upper left", fontsize=10, labelcolor=GRAY)


# ------------------------------------------------------------------- cards
def panel_metrics_card(ax, values: dict) -> None:
    strip(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    items = list(values.items())
    n = len(items)
    for i, (label, val) in enumerate(items):
        x = (i + 0.5) / n
        ax.text(x, 0.62, val, color=AMBER, fontsize=40, ha="center", va="center",
                weight="bold")
        ax.text(x, 0.30, label, color=GRAY, fontsize=13, ha="center", va="center")


def panel_counterfactual(ax, sc: Scene, k_frac: float) -> None:
    """Act 5: same energy, different timing."""
    strip(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cf = sc.counterfactual
    for i, (name, data, color) in enumerate((
            ("BLIND", cf["blind"], GRAY), ("X-RAY", cf["xray"], AMBER))):
        y = 0.72 - 0.42 * i
        ax.text(0.0, y + 0.16, name, color=color, fontsize=16, weight="bold")
        laps = data["laps"]
        for j, (lap, q, attacked, passed) in enumerate(laps):
            x = 0.06 + 0.88 * j / max(len(laps) - 1, 1)
            if attacked:
                col = GREEN if passed else RED
                ax.add_patch(Circle((x, y), 0.030, fc=col, ec="none"))
            else:
                ax.add_patch(Circle((x, y), 0.013, fc=DIM, ec="none"))
        result = "PASSED lap %d" % data["lap_passed"] if data["passed"] else "never passed"
        ax.text(0.0, y - 0.13, result, color=color if data["passed"] else DIM,
                fontsize=13)
    ax.text(0.0, 0.06, "attack   ", color=DIM, fontsize=11)
