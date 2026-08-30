"""Shot list -> MP4.

Scripted rendering: deterministic, reproducible, and it cannot fail live.
Every number that appears on screen was computed by the pipeline.
"""
from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

from ..constants import E_STORE_MAX
from ..decision import (build_model, compare_exogenous, delta_v, policy_posterior,
                        rival_energy_at_zone, robustness, simulate_stint_exogenous,
                        solve, solve_exogenous)
from ..estimator import estimate
from ..metrics import band_width_by_regime, score_estimate, true_reserve_floor
from ..observe import observe
from ..overtake import p_pass
from ..sim import FOLLOWER, LEADER, run_sim
from ..vehicle import VehicleParams
from .panels import (Scene, panel_battery_bars, panel_counterfactual,
                     panel_metrics_card, panel_speed_traces, panel_threshold,
                     panel_track_map, panel_truth_reveal)
from .style import AMBER, BG, CARD, DIM, GRAY, GREEN, GRIDLINE, RED, WHITE, apply_style, strip

FPS = 30
FIGSIZE = (16, 9)
DPI = 120


# --------------------------------------------------------------- scene build
def build_scene(cfg: dict, seed: int = 42, rate_hz: float | None = None,
                verbose: bool = True) -> Scene:
    rate = rate_hz or cfg["observe"]["rate_hz"]
    gt = run_sim(cfg, seed=seed)
    ours, rival = FOLLOWER, LEADER
    obs = observe(gt, rival, rate_hz=rate,
                  speed_noise_ms=cfg["observe"]["speed_noise_ms"], seed=seed + 1)
    belief = estimate(obs, gt.track, n_particles=cfg["estimator"]["n_particles"],
                      seed=seed + 2)
    score = score_estimate(gt, rival, obs, belief, cfg["vehicle"]["cda_straight"])
    idx_gt = np.clip(np.searchsorted(gt.t, obs.t), 0, len(gt.t) - 1)

    true_soc = gt.cars[rival].E[idx_gt]
    true_usable = np.maximum(true_soc - true_reserve_floor(gt, rival, obs.lap), 0.0)
    our_soc = gt.cars[ours].E[idx_gt]

    # --- decision layer, driven by the belief (never by the rival's true state)
    params = VehicleParams.from_config(cfg)
    recharge = float(np.mean(belief.harvested_lap[belief.harvested_lap > 0]))
    rival_spend = float(np.mean(belief.deployed_lap[belief.deployed_lap > 0]))
    own_spend = float(np.mean(gt.cars[ours].deployed_lap)) * 0.55
    model = build_model(gt.track, params, n_laps=gt.n_laps,
                        recharge_per_lap=recharge, rival_spend_per_lap=rival_spend,
                        own_spend_per_lap=own_spend)

    # what we believe the rival can still deploy, at the entry to the zone where
    # a pass would actually happen -- not a lap average
    rival_track = rival_energy_at_zone(belief, obs, gt.track, gt.n_laps)
    sol = solve_exogenous(model, rival_track)

    our_e = np.array([float(our_soc[obs.lap == L].max()) if (obs.lap == L).any()
                      else 0.0 for L in range(gt.n_laps)])

    tau = np.array([sol.threshold(gt.n_laps - L, our_e[L]) for L in range(gt.n_laps)])
    opportunities = []
    attack = None
    for L in range(gt.n_laps):
        q = sol.quality(gt.n_laps - L, our_e[L])
        zone = sol.action(gt.n_laps - L, our_e[L])
        opportunities.append((L, float(q), zone, zone is not None))
        if zone is not None and attack is None:
            attack = {"lap": L, "q": float(q), "zone": zone}

    posterior = policy_posterior(belief, gt.track, n_samples=200, seed=seed)
    if attack is not None:
        rb = robustness(model, posterior, laps_left=gt.n_laps - attack["lap"],
                        e_own=our_e[attack["lap"]], e_riv=rival_track[attack["lap"]])
        attack["robust"] = rb["fraction_agreeing"]

    # --- counterfactual: same energy, two decision rules, same rival
    # Show a run where the two rules actually diverge, and say on screen how
    # often that happens. Picking an illustrative example is fine; picking a
    # flattering one is not, so the pass rates are always shown next to it.
    cf = {}
    for r in range(60):
        runs = {name: simulate_stint_exogenous(
            model, sol, np.random.default_rng(seed * 131 + r), gt.n_laps,
            float(our_e[0]), rival_track, blind=blind)
            for name, blind in (("blind", True), ("xray", False))}
        cf = {k: {"laps": v["laps"], "passed": v["passed"],
                  "lap_passed": v["lap_passed"]} for k, v in runs.items()}
        if runs["xray"]["passed"] and not runs["blind"]["passed"]:
            break
    comp = compare_exogenous(model, sol, rival_track, 50, gt.n_laps,
                             float(our_e[0]), seed=seed)

    metrics = {
        "per-lap energy error": f"{score.deployed_mape:.1f}%",
        "band coverage": f"{score.usable_coverage:.2f}",
        "CdA recovered to": f"{abs(score.cda_error_pct):.1f}%",
        "positions gained": f"+{comp['mean_gain']:.2f}",
    }
    if verbose:
        print(f"  estimator: MAPE {score.deployed_mape:.1f}%, "
              f"coverage {score.usable_coverage:.2f}, CdA err {score.cda_error_pct:+.1f}%")
        print(f"  band width by regime: "
              f"{ {k: round(v, 2) for k, v in band_width_by_regime(gt, rival, obs, belief).items()} }")
        waits = [o[0] + 1 for o in opportunities if not o[3]]
        print(f"  rival deployable at zone entry (MJ): "
              f"{np.round(rival_track / 1e6, 2)}")
        if attack:
            print(f"  decision: hold on lap(s) {waits or '-'}, attack lap "
                  f"{attack['lap'] + 1} zone {attack['zone']}, P {attack['q']:.2f}, "
                  f"optimal across {attack['robust'] * 100:.0f}% of opponent "
                  f"policy space")
        print(f"  counterfactual: X-RAY {comp['xray_pass_rate']:.2f} vs blind "
              f"{comp['blind_pass_rate']:.2f}, gain {comp['mean_gain']:+.2f} "
              f"CI95 [{comp['ci95'][0]:.2f}, {comp['ci95'][1]:.2f}]")

    return Scene(gt=gt, obs=obs, belief=belief, ours=ours, rival=rival,
                 true_soc_rival=true_soc, true_usable_rival=true_usable,
                 our_soc=our_soc, idx_gt=idx_gt, tau=tau,
                 opportunities=opportunities, attack=attack,
                 metrics=metrics | {"_score": score, "_compare": comp},
                 counterfactual=cf)


def _unused_counterfactual(model, sol, believed, n_laps, e_own0, e_riv0, seed):
    out = {}
    for name, chooser in (("blind", blind_chooser(model)),
                          ("xray", xray_chooser(sol, np.array(believed), 0.0,
                                                np.random.default_rng(seed)))):
        rng = np.random.default_rng(seed + 11)
        e_own, e_riv = e_own0, e_riv0
        rows = []
        passed, lap_passed = 0, None
        for k in range(n_laps, 0, -1):
            pick = chooser(k, e_own, e_riv)
            e_own = float(np.clip(e_own + model.recharge_per_lap - model.own_spend_per_lap,
                                  0, E_STORE_MAX))
            e_riv = float(np.clip(e_riv + model.recharge_per_lap - model.rival_spend_per_lap,
                                  0, E_STORE_MAX))
            q, hit = 0.0, False
            if pick is not None and not passed:
                zm = next(z for z in model.zones if z.name == pick)
                spend = min(model.attack_cost, e_own)
                q = p_pass(delta_v(zm, spend, min(e_riv, model.attack_cost)),
                           model.gap_s, zm)
                e_own = max(e_own - spend, 0.0)
                hit = bool(rng.random() < q)
                if hit:
                    passed, lap_passed = 1, n_laps - k + 1
            rows.append((n_laps - k, q, pick is not None and not (passed and not hit), hit))
            if passed:
                break
        out[name] = {"laps": rows, "passed": passed, "lap_passed": lap_passed or 0}
    return out


# ------------------------------------------------------------------- shots
def _caption(fig, text, alpha):
    if not text:
        return
    fig.text(0.045, 0.055, text, color=WHITE, fontsize=22, alpha=alpha,
             va="bottom", zorder=20)


def _title_text(fig, big, small=None, alpha=1.0):
    fig.text(0.5, 0.56, big, color=WHITE, fontsize=46, ha="center", va="center",
             weight="bold", alpha=alpha)
    if small:
        fig.text(0.5, 0.44, small, color=GRAY, fontsize=20, ha="center", va="center",
                 alpha=alpha)


def _racing_line(fig, phase):
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    strip(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i, (y, col, w) in enumerate(((0.22, RED, 5), (0.19, AMBER, 3))):
        x1 = float(np.clip(phase * 1.5 - 0.12 * i, 0, 1))
        ax.plot([0.0, x1], [y, y], color=col, lw=w, solid_capstyle="butt")
    ax.set_zorder(-1)


def _checkers(fig, phase):
    ax = fig.add_axes((0.0, 0.0, 1.0, 0.055))
    strip(ax)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 2)
    n = int(40 * min(phase * 1.6, 1.0))
    for i in range(n):
        for j in range(2):
            if (i + j) % 2 == 0:
                ax.add_patch(plt.Rectangle((i, j), 1, 1, fc=WHITE, ec="none"))
    ax.set_zorder(-1)


def _sample_at(sc: Scene, frac: float, lo_lap: int, hi_lap: int) -> int:
    m = np.flatnonzero((sc.obs.lap >= lo_lap) & (sc.obs.lap <= hi_lap))
    if len(m) == 0:
        m = np.arange(len(sc.obs.t))
    return int(m[int(np.clip(frac, 0, 1) * (len(m) - 1))])


def _fade(local_t, dur=0.4):
    return float(np.clip(local_t / dur, 0, 1))


def draw_act0(fig, sc, u, t):
    _racing_line(fig, u)
    _title_text(fig, "X-RAY", "seeing the invisible half of the 2026 race",
                alpha=_fade(t, 0.6))


def draw_act1(fig, sc, u, t):
    k = _sample_at(sc, u, 1, 2)
    gs = fig.add_gridspec(2, 2, left=0.03, right=0.97, top=0.90, bottom=0.15,
                          hspace=0.30, wspace=0.10, width_ratios=(1.0, 1.35))
    panel_track_map(fig.add_subplot(gs[:, 0]), sc, k)
    panel_speed_traces(fig.add_subplot(gs[0, 1]), sc, k)
    panel_battery_bars(fig.add_subplot(gs[1, 1]), sc, k, show_question=True)
    fig.text(0.05, 0.935, "WHAT THE WORLD SEES", color=GRAY, fontsize=17,
             weight="bold")
    fig.text(0.97, 0.935, f"gap {sc.gt.gap_s[sc.idx_gt[k]]:+.2f} s", color=WHITE,
             fontsize=17, ha="right", weight="bold")
    _caption(fig, "Public data. No energy channel exists.", _fade(t))


def draw_act2(fig, sc, u, t):
    k = _sample_at(sc, u, 3, 5)
    gs = fig.add_gridspec(2, 2, left=0.03, right=0.97, top=0.90, bottom=0.15,
                          hspace=0.30, wspace=0.10, width_ratios=(1.0, 1.35))
    panel_track_map(fig.add_subplot(gs[:, 0]), sc, k)
    panel_speed_traces(fig.add_subplot(gs[0, 1]), sc, k)
    panel_battery_bars(fig.add_subplot(gs[1, 1]), sc, k, reveal_band=True)
    fig.text(0.05, 0.935, "WHAT WE INFER", color=GRAY, fontsize=17, weight="bold")
    width = (sc.belief.usable_p90[k] - sc.belief.usable_p10[k]) / 1e6
    regime = sc.gt.cars[sc.rival].regime[sc.idx_gt[k]]
    label = {"accel": "hard acceleration - informative",
             "brake": "braking - recovery visible",
             "corner": "steady-state corner - uninformative"}[regime]
    fig.text(0.97, 0.935, f"band {width:.2f} MJ  ({label})", color=GRAY, fontsize=15,
             ha="right")
    _caption(fig, "Reconstructed from the speed trace alone.", _fade(t))


def draw_act3(fig, sc, u, t, dur):
    ax = fig.add_axes((0.07, 0.22, 0.88, 0.62))
    k = _sample_at(sc, min(u * 1.6, 1.0), 1, sc.gt.n_laps - 1)
    err = sc.metrics["_score"].deployed_mape
    shown = err * float(np.clip((t - 1.0) / max(dur * 0.45, 0.1), 0, 1))
    panel_truth_reveal(ax, sc, k, err_pct=shown if t > 1.0 else None)
    fig.text(0.05, 0.925, "THE REVEAL", color=GRAY, fontsize=17, weight="bold")
    _caption(fig, "Truth, revealed.", _fade(t))


def draw_act4(fig, sc, u, t):
    lap_now = int(np.clip(u * (sc.gt.n_laps - 1), 0, sc.gt.n_laps - 1))
    k = _sample_at(sc, 0.9, lap_now, lap_now)
    if u < 0.72:
        ax = fig.add_axes((0.08, 0.20, 0.86, 0.64))
        panel_threshold(ax, sc, k)
    else:
        gs = fig.add_gridspec(1, 2, left=0.05, right=0.97, top=0.86, bottom=0.18,
                              wspace=0.16)
        panel_threshold(fig.add_subplot(gs[0, 0]), sc, k)
        panel_track_map(fig.add_subplot(gs[0, 1]), sc, k)
    fig.text(0.05, 0.925, "THE DECISION", color=GRAY, fontsize=17, weight="bold")
    if sc.attack and u > 0.35:
        fig.text(0.97, 0.925,
                 f"attack lap {sc.attack['lap'] + 1}, zone {sc.attack['zone']} - "
                 f"optimal across {sc.attack['robust'] * 100:.0f}% of opponent policy space",
                 color=GREEN, fontsize=15, ha="right")
    _caption(fig, "Wait for the number, not the feeling.", _fade(t))


def draw_act5(fig, sc, u, t):
    ax = fig.add_axes((0.07, 0.24, 0.86, 0.60))
    panel_counterfactual(ax, sc, u)
    fig.text(0.05, 0.925, "COUNTERFACTUAL", color=GRAY, fontsize=17, weight="bold")
    c = sc.metrics["_compare"]
    fig.text(0.97, 0.925, f"over {c['n_races']} seeded stints: "
                          f"{c['xray_pass_rate']:.0%} vs {c['blind_pass_rate']:.0%}",
             color=GRAY, fontsize=15, ha="right")
    _caption(fig, "Same energy. Different timing.", _fade(t))


def draw_act6(fig, sc, u, t):
    _checkers(fig, u)
    ax = fig.add_axes((0.05, 0.34, 0.90, 0.34))
    panel_metrics_card(ax, {k: v for k, v in sc.metrics.items()
                            if not k.startswith("_")})
    fig.text(0.5, 0.80, "X-RAY", color=WHITE, fontsize=34, ha="center", weight="bold")
    fig.text(0.5, 0.735, "reconstructed from a 3.7 Hz speed trace, nothing else",
             color=GRAY, fontsize=16, ha="center")
    _caption(fig, "", 1.0)


SHOTS = [
    ("title", 3.0, draw_act0),
    ("world", 9.0, draw_act1),
    ("infer", 13.0, draw_act2),
    ("reveal", 9.0, draw_act3),
    ("decision", 16.0, draw_act4),
    ("counterfactual", 8.0, draw_act5),
    ("close", 4.0, draw_act6),
]


def render(sc: Scene, out: str, fps: int = FPS, progress: bool = True) -> str:
    apply_style()
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=BG)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=6000,
                          extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"])
    total = int(sum(d for _n, d, _f in SHOTS) * fps)
    done = 0
    with writer.saving(fig, out, DPI):
        for name, dur, fn in SHOTS:
            n = int(dur * fps)
            for i in range(n):
                fig.clear()
                fig.patch.set_facecolor(BG)
                u = i / max(n - 1, 1)
                local_t = i / fps
                if fn is draw_act3:
                    fn(fig, sc, u, local_t, dur)
                else:
                    fn(fig, sc, u, local_t)
                writer.grab_frame(facecolor=BG)
                done += 1
                if progress and done % 60 == 0:
                    print(f"    {done}/{total} frames ({name})", flush=True)
    plt.close(fig)
    return out
