# X-RAY — developer README

Orientation for someone about to change code.

**2026-09-12 "folder cleanups" commit (`2a9861f`) deleted `README.md`,
`PROJECT_BRIEF.md`, and all of `docs/` (`status.md`, `model.md`,
`model_inventory.md`, `stage2.md`, `feasibility.md`) outright — they were not
moved. The claim, Stage 1 results, every number's derivation, and the six-view
Stage 2 argument that used to live there are gone from the repo; nothing
replaced them.** The one surviving normative document is:

| Read this | For |
|---|---|
| [`.claude/skills/x-ray/SKILL.md`](.claude/skills/x-ray/SKILL.md) | the invariants, in enforceable form |
| [`STARTUP.md`](STARTUP.md) | copy-pasteable commands per setup (Stage 1 only, Stage 2 only, with/without API, with/without frontend, tests, maintenance) |

Its metrics table still reproduces (§6); its test-count and file-layout
references are pre-cleanup and stale — this file and a directory listing are
the current truth for those. `scripts/08.a_feasibility_report.py` now writes
`out/feasibility.md` instead of `docs/feasibility.md` (§3), the one script-level
trace of where that doc used to land.

This file is the part none of those covered anyway: how to get it running,
where each module sits, which test guards which invariant, and what is
currently broken.

---

## 0. Folder overview

```
config/       one file, config/default.yaml — every Stage 1 script's defaults
xray/         the library: Stage 1 core, the v2 identification path, real-data
              ingest/analysis, and decision_service.py (§4)
scripts/      every CLI entry point, run against xray/ (§2, §3)
tests/        pytest suite, one file per invariant (§5)
simulation/   everything that turns an analysed race into something to look at:
                api/      FastAPI app (simulation/api/main.py), serves out/races/*.json
                app/      React + three.js viewer (six views under src/views/)
                sim2vid/  shot.mjs + its own package.json/requirements.txt —
                          Playwright screenshot capture, unrelated to xray/'s deps
prediction/   currently empty, untracked by git — reserved, nothing built here yet
```

`simulation/` used to be split across root-level `app/`, `api/`, and
`shot.mjs`/`package.json` (Playwright); the cleanup commit consolidated all of
it under one directory. `xray/`, `scripts/`, `tests/`, `config/` were untouched
by that move.

---

## 1. Environment

Python **3.11+** is a hard floor: `numpy==2.3.5` and `pandas==3.0.2` publish no
wheels for 3.10. `ffmpeg` on the path is needed only by `scripts/06.b_make_video.py`.
Node 20+ and `npm` are needed only for `simulation/app/` and `simulation/sim2vid/`.

**There is no root `requirements.txt` or `package.json` any more** — the
2026-09-12 "folder cleanups" commit deleted them along with `docs/` and did
not replace them (§7 item 1). `simulation/sim2vid/requirements.txt` and
`simulation/sim2vid/package.json` exist but only cover Playwright for
`shot.mjs`; nothing pins the actual Stage 2 stack (fastf1, pandas, fastapi,
pyarrow, ...). This machine's `.venv/` already has that stack installed
(below) from before the cleanup — do not delete it without saving a `pip
freeze` first, since there is currently no other way to reconstruct it.

```powershell
py -3.14 -m venv .venv                 # or any 3.11+
.\.venv\Scripts\python -m pip install numpy scipy matplotlib PyYAML pytest `
    fastf1 pandas==2.3.3 fastapi uvicorn pyarrow
.\.venv\Scripts\python -m pytest -q
```

Stage 1 alone needs only the light half: `numpy scipy matplotlib PyYAML pytest`.
`pandas==2.3.3`, not `3.0.2`, is the correct pin — it's what fastf1 accepts
(`pandas<3.0.0`) and the repo's own pandas usage (`pd.DataFrame`, `concat`,
`isna`, `to_parquet` in `xray/data/ingest.py` and `circuits.py`) doesn't care
about the difference either way.

**State of this machine, 2026-09-12.** `.venv/` exists on **Python 3.14.4** with
all of the above installed: fastf1 3.8.3, pandas 2.3.3, numpy 2.3.5, scipy
1.16.3, pyarrow 23.0.1, fastapi 0.121.2, uvicorn 0.41.0. Every Stage 2 module
imports (`xray.data.ingest`, `xray.data.circuits`, `xray.realfit`,
`xray.analysis`, `simulation.api.main`), and fastf1 reaches the live API —
`get_event_schedule(2026)` returns 25 rounds, with the cache seeded at
`xray/data/cache/`.

`python` on bare PATH is still 3.10.11 and will not do — always go through
`.venv\Scripts\python`. There is no `out/`, no `simulation/app/node_modules/`,
no `simulation/app/dist/`: no race has been analysed here, the web app has
never been built, and every API route 404s until `08.b_analyse_race.py` writes a
race JSON.

`pip` here fails intermittently with `NameResolutionError` on
`files.pythonhosted.org` part-way through a large download, leaving nothing
installed. Install in small batches and re-run; it succeeds on retry.

The two `package.json` files serve unrelated purposes: `simulation/sim2vid/`
carries Playwright (for `shot.mjs`); the web app's own dependencies are in
`simulation/app/package.json` and install from `simulation/app/`.

---

## 2. Running it

For copy-pasteable commands grouped by setup (Stage 1 only, Stage 2 only,
with/without the API, with/without the frontend, tests, maintenance), see
[`STARTUP.md`](STARTUP.md). This section explains what each stage does; §2a
below gives the pipeline order.

### Stage 1 — simulator, estimator, decision (no network)

```bash
python scripts/01.run_sim.py --seed 42                        # physics and the race
python scripts/02.run_estimator.py --seed 42 --car LEADER     # blindfold, estimate, score
python scripts/05.run_ablation.py                             # MAPE vs sample rate -> out/
python scripts/03.b_run_decision_eval.py --mode fast          # blind vs X-RAY, 50 stints
python scripts/03.a_validate_decision.py                      # oracle vs X-RAY vs blind
python scripts/04.simulate_attack.py                          # scores the outcome, not the call
python scripts/06.a_make_summary.py --seed 42                 # -> out/xray_summary.png
python scripts/06.b_make_video.py --seed 42                   # -> out/xray_demo.mp4 (needs ffmpeg)
```

`--car LEADER` is not optional on `02.run_estimator.py`; see §6. `--mode full`
on the decision eval re-runs the whole 200 Hz two-car sim plus a fresh
estimator pass per seed — minutes, not seconds. Seed 42 is the eventful race
used through the docs; seed 9 is the clean one-pass race.

All Stage 1 scripts share `scripts/_common.py`, so they all take
`--seed --config --laps --out`, and all read `config/default.yaml`.

### Stage 2 — real telemetry and the app (network, slow first run)

```bash
python scripts/07.a_feasibility.py --round 10        # -> out/feasibility/r10.json
python scripts/08.a_feasibility_report.py            # -> out/feasibility.md
python scripts/07.b_run_real.py --round 10           # Stage 1 core over a real session
python scripts/08.b_analyse_race.py --round 10       # -> out/races/<id>.json  (the app's input)
python -m uvicorn simulation.api.main:app --port 8011

cd simulation/app && npm install && npm run build   # the API then serves it at /
```

Then <http://127.0.0.1:8011/>, `?lite=1` on a weak GPU. FastF1 downloads session
telemetry on first touch and caches it in `xray/data/cache/` (gitignored) —
leave that cache alone, it is the difference between seconds and minutes.

`simulation/api/main.py`'s `ROOT` resolves to `simulation/`, so it serves
`simulation/app/dist/` once built. The API is thin by design: it serves
precomputed `out/races/*.json` and computes exactly one thing live, the RDD,
because the judge drags the cutoff. No particle filter runs on request. Routes:

```
GET /api/races                         GET /api/race/{rid}/observability
GET /api/race/{rid}/summary            GET /api/race/{rid}/decision
GET /api/race/{rid}/car/{drv}          GET /api/race/{rid}/counterfactual
GET /api/race/{rid}/battle/{a}/{b}     GET /api/rdd
```

Every one of them 404s until `08.b_analyse_race.py` has written a race JSON.

---

## 2a. Script pipeline order

`scripts/` filenames are numbered to show the pipeline order. The convention:

- `NN.name.py` for a serial pipeline stage.
- `NN.a_name.py`, `NN.b_name.py` for stages that sit at the same level and can
  be run independently after the previous numbered step is available.
- `99.name.py` for maintenance or audit helpers that are not part of the
  normal production flow.

**Synthetic pipeline**

```
01.run_sim.py
   Builds and optionally saves synthetic two-car ground truth.

02.run_estimator.py
   Runs the simulator, observes the public trace, estimates hidden energy,
   and scores estimator quality.

03.a_validate_decision.py         03.b_run_decision_eval.py
   Decision-service diagnostic —      Monte Carlo decision evaluation —
   checks one decision point end      compares X-RAY against blind and
   to end and proves future           oracle decisions using the
   telemetry does not affect the      estimator output.
   past decision.
   (parallel siblings, both need 02's output)

04.simulate_attack.py
   Closed-loop attack simulation. Executes the chosen plan in the simulator
   and scores whether the pass actually happens.

05.run_ablation.py
   Telemetry-rate ablation. Writes out/ablation.json and the MAPE plot used
   by later summary artifacts.

06.a_make_summary.py              06.b_make_video.py
   One-page summary figure.           Demo video. Independent artifact
   Requires out/ablation.json         builder, no dependency on 06.a.
   from step 05.
   (parallel artifact builders)
```

**Real-data branch**

```
07.a_feasibility.py               07.b_run_real.py
   Probes one real FastF1             Runs the Stage 1 real-session core
   session, writes                    end to end.
   out/feasibility/r<round>.json.
   (real-data siblings)

08.a_feasibility_report.py            08.b_analyse_race.py
   Aggregates feasibility JSON           Precomputes one real race into
   files into out/feasibility.md.        app-ready JSON.
   Depends on 07.a's output.             (the app-data path)
```

**Maintenance and audit**

```
99.make_golden.py     Regenerates golden estimator outputs for tests.
                       Use only when a numerics change is intentional.
99.worked_trace.py    Prints one worked decision trace and its causality proof.
_common.py            Shared CLI plumbing. Unnumbered — imported by the other
                       scripts rather than run as a pipeline stage.
```

---

## 3. Script reference: every script's inputs, outputs, and data

All Stage 1 scripts below share `scripts/_common.py`'s parser: `--seed`
(default 42), `--config` (default `config/default.yaml`), `--laps` (override
`sim.n_laps`), `--out` (override the default output path). Stage 2 scripts
take their own `argparse` flags — no shared base — because they key off
`--round`/`--year`/`--session`, not a seed. "Data" below means what each
script reads beyond its CLI flags: a config file, a cache, or another
script's output.

### Stage 1 — simulator, estimator, decision

| Script | In | Out | Data |
|---|---|---|---|
| `01.run_sim.py` | `--seed --config --laps --out --save` | stdout: lap times, energy balance, overtakes, final order. With `--save`, `out/ground_truth.npz` (`t`, `gap`, per-car `s/v/E/P_mguk`) | `config/default.yaml`; nothing on disk |
| `02.run_estimator.py` | `--seed --config --laps --rate (Hz list, default 100 3.7) --car (default LEADER FOLLOWER) --particles` | stdout only: per car/rate Stage A (CdA fit) and Stage C (deployment, reserve, band width) blocks, a score table, and an aggregated JSON | `config/default.yaml`; no files read or written. `--car LEADER` is required in practice — FOLLOWER is tow-bound and raises (§7 item 7) |
| `05.run_ablation.py` | `--seed --config --laps --out --rates (Hz list) --seeds (default 42 7 13) --cars (default LEADER)` | `out/ablation_mape_vs_rate.png` (or `--out`); always also `out/ablation.json` (`rates`, `mape`, `coverage`, `cda_abs_err_pct`, `failures`) | `config/default.yaml`. `out/ablation.json` is consumed by `06.a_make_summary.py` |
| `03.b_run_decision_eval.py` | `--seed --config --laps --mode {fast,full} (default fast) --races (default 50)` | stdout: pass rates and positions-gained CI for X-RAY vs blind | `config/default.yaml`; nothing on disk. `--mode full` reruns a full 200 Hz sim per race — minutes, not seconds |
| `03.a_validate_decision.py` | `--seeds (default 30) --rate (default 3.7) --particles (default 300) --leader/--follower (policy names) --verbose` | stdout: EV/p(pass)/lap table for best-possible, ORACLE, X-RAY, BLIND(first-chance), BLIND(random); agreement-with-oracle and significance stats | `config/default.yaml`; nothing on disk. Uses `xray.decision.solve_exogenous` against ground-truth vs reconstructed rival energy — ground truth only to build ORACLE and to score |
| `04.simulate_attack.py` | `--seeds (default 40) --rate (default 4.17) --particles (default 250) --leader/--follower --verbose` | stdout: outcome table (passed / ahead-at-flag / best chance seen) for oracle, xray, blind, passive arms, plus significance vs blind and passive | `config/default.yaml`. Closed-loop: re-runs the race with the called attack actually executed, so outcomes come from the simulator's own pass model, not a score function |
| `06.a_make_summary.py` | `--seed --config --laps --out` | `out/xray_summary.png` (6-panel deck figure) | `config/default.yaml` **and** `out/ablation.json` — exits with `SystemExit` if that file is missing, so run `05.run_ablation.py` first |
| `06.b_make_video.py` | `--seed --config --laps --out --fps --rate` | `out/xray_demo.mp4` | `config/default.yaml`; needs `ffmpeg` on PATH (not a Python dependency) |
| `99.make_golden.py` | `--seed --config --laps --out` (only `--seed` should ever be changed — see its docstring) | `tests/golden/belief_seed<seed>_3.7hz.npz` (scalar + array belief outputs) | `config/default.yaml`. Pinned to `tests/conftest.py`'s exact configuration (seed 42, LEADER, 3.7 Hz, obs seed+1, estimate seed+2); run only when a numerics change is intended (§4, §7 item 3) |

### Stage 2 — real telemetry and the app

| Script | In | Out | Data |
|---|---|---|---|
| `07.a_feasibility.py` | `--year (default 2026) --round (required) --session (default R) --max-drivers (default 20)` | `out/feasibility/r<round>.json` — telemetry rate histogram, channel presence, position/elevation availability, high-speed-window fractions, weather | Downloads one FastF1 session (network, first run slow); caches to `xray/data/cache/` |
| `08.a_feasibility_report.py` | none (no CLI args) | `out/feasibility.md` (moved from `docs/feasibility.md` in the folder-cleanups commit, since `docs/` no longer exists) | Reads every `out/feasibility/r*.json`; exits with `SystemExit` if none exist — run `07.a_feasibility.py` first, once per round of interest |
| `07.b_run_real.py` | `--round (default 10) --year (default 2026) --session (default R) --drivers --max-laps --mass (default 790.0)` | stdout only: per-car CdA fit, identified set, identifiability, field pooling, cross-car common mode, one car's deployment/harvest per lap | Downloads a FastF1 session (cached in `xray/data/cache/`); no files written |
| `08.b_analyse_race.py` | `--round (required) --year (default 2026) --session (default R) --max-laps --drivers --particles (default 400)` | one `out/races/<id>.json` — the app's and API's sole input; stdout summary (cars analysed, refusals, telemetry rate, pooled CdA, RDD row count) | Downloads a FastF1 session (cached); every `simulation/api/main.py` route 404s until this has run for a given round |
| `simulation/api/main.py` (not a `scripts/` file, run via `uvicorn simulation.api.main:app`) | none — serves `GET /api/*` | HTTP responses over `out/races/*.json`; computes the RDD live, nothing else; also exposes the decision service (`xray/decision_service.py`) used by `test_decision_service.py` | Reads whatever `08.b_analyse_race.py` has written to `out/races/` |

---

## 4. Module map

Stage 1 core — the original path, untouched by the `math-model` merge:

```
xray/constants.py   regulation constants and public knowledge
xray/track.py       Circuit Sigma (public — the estimator may read it)
xray/vehicle.py     longitudinal dynamics and the energy store
xray/policy.py      the "driver", four interpretable parameters
xray/sim.py         two-car simulator -> GroundTruth
xray/observe.py     the blindfold: GroundTruth -> Observation
xray/estimator.py   Observation -> BeliefTrace        <- the contribution
xray/overtake.py    P(pass); the invented coefficients (§6)
xray/decision.py    value iteration, thresholds, robustness
xray/metrics.py     scoring; the only module allowed to read ground truth
xray/config.py      YAML loading; the estimator never imports it
xray/render/        style, panels, shot list -> MP4
xray/decision_service.py   causal state extraction from an analysed race
                    payload -> the ATTACK/HOLD decision the web app renders;
                    calls xray/decision.py's solver, does not reimplement it
```

The v2 identification path, one module per build step (steps numbered as in
`docs/status.md`):

```
1  xray/regs.py      the 2026 rulebook as bounds; pre/post-Miami by date
1  xray/balance.py   interval energy balance -> linear constraints on theta
2  xray/setmem.py    the feasible set as a polytope; LP projections; L1 alarm
3  xray/modes.py     aero mode and powertrain regime as an HMM
4  xray/pipeline.py  composes 1-3; owns no numerics of its own
5  xray/rbpf.py      store, reserve and theta by Rao-Blackwellised particle filter
6  xray/strategy.py  the PMP prior: three interpretable parameters per lap
7  xray/pooling.py   partial pooling car -> team -> field, closed form
8  xray/qmdp.py      deciding under the belief, sensitivity reported
14 xray/deadband.py  v_cut / v_harv by changepoint, BEFORE the filter runs
```

`theta = (CdA_X, CdA_Z, F_rr, dm)` — the two aero states, rolling resistance as a
force at a reference mass, and dry mass minus the published weight.

Real data and delivery:

```
xray/data/ingest.py         FastF1 -> canonical parquet on a common distance grid
xray/data/circuits.py       centreline, elevation, curvature from real telemetry
xray/realfit.py             generalized calibration for real telemetry
xray/analysis.py            whole pipeline over one session -> one JSON
simulation/api/main.py      serves those JSONs; computes only the RDD live;
                            calls xray/decision_service.py for the decision routes
simulation/app/             React + three.js, six views (src/views/)
simulation/sim2vid/shot.mjs Playwright screenshot capture (§7 item 9)
```

---

## 5. Tests, and which invariant each one holds

```
test_estimator.py      the blindfold (test_estimator_is_blind parses the file with ast),
                       refusal on a tow-bound car, Stage 1 acceptance
test_live_readiness.py kernel purity and look-ahead: batch == reduce(update, .)
test_golden.py         fixed-seed .npz baseline every optimisation must reproduce
test_physics.py        the simulator's own dynamics
test_balance.py        step 1 — the balance must CONTAIN the truth
test_setmem.py         step 2 — what the polytope can and cannot pin down
test_modes.py          step 3 — HMM, not a per-sample classifier
test_pipeline.py       step 4 — mass drift, the tow bound, the wiring
test_rbpf.py           step 5 — store vs reserve, and which one actually works
test_strategy.py       step 6 — the PMP strategy prior
test_pooling.py        step 7 — closed-form partial pooling
test_qmdp.py           step 8 — belief, not point estimate
test_deadband.py       step 14 — the band measured before the filter
test_decision.py       decision-engine acceptance
test_decision_service.py  the web-facing decision service: causal state
                       extraction from an analysed payload, and that
                       simulation/api/main.py's decision route agrees with it
```

`tests/conftest.py` builds the expensive things once per session: three seeded
races (42/7/13) and their beliefs at 3.7 Hz, LEADER only. Use those fixtures — a
test that calls `run_sim` itself pays for a whole race.

`tests/golden/belief_seed42_3.7hz.npz` is the baseline. `scripts/99.make_golden.py`
regenerates it and is pinned to `conftest`'s exact configuration. Run it **only**
when a numerics change is intended, and say in the commit message which headline
metric moved and why — regenerating silently converts a regression into a new
baseline, which is the one failure mode that harness exists to prevent.

There is currently **no `slow` marker**, so there is no fast test path. Adding
one (multi-seed coverage marked slow, `-m "not slow"` under ~30 s, the blindfold
test always in the fast path) is an open item from the skill's efficiency rules.

---

## 6. Two baselines — know which one you are quoting

Both are real and they measure different paths. Quoting the wrong one at a
reviewer is how a regression gets waved through.

**Stage 1 estimator** (`xray/estimator.py`, unchanged). Reproduced here at HEAD
with `run_estimator.py --seed 42 --car LEADER`, and it matches the skill's table
exactly:

| | @ 100 Hz | @ 3.7 Hz |
|---|---|---|
| deployed MAPE | 4.6% | 3.5% |
| harvested MAPE | 6.3% | 22.5% |
| usable coverage | 0.939 | 0.971 |
| CdA error | −0.21% | +1.77% (0.99% mean over the run) |

Band width by regime at 3.7 Hz: accel 0.665 < brake 0.753 < corner 0.915 MJ.
Recovered energy is rate-sensitive because it is a braking-*duration*
measurement; deployment is not, which is why 3.7 Hz is no worse than 100 Hz.

**v2 identification path** (`regs` → `deadband`, from `docs/status.md`, not
re-derived here): deployable-band coverage 0.67 / 0.72 / 0.80 on seeds 42/7/13;
per-lap deployed 9.9 / 15.2 / 14.0% MAPE; CdA_X 0.597 / 0.571 / 0.494 against a
true 0.660; identified set for CdA_X on seed 42 `[0.264, 0.744]`, two-sided.

**Suite, measured here:** `pytest -q` is **111 tests in 126 s** — 109 passed,
2 failed, both environmental rather than numerical (§7 items 2 and 3). The
126 s is against the 52 s in `docs/status.md`; different machine, no conclusion
drawn.

The v2 numbers are worse and that is the point: they are produced under honest
priors and reported with the assumption-free identified set beside them. Do not
"improve" them by narrowing a band. **A band is never narrowed to hit a MAPE
number.**

The skill file's test count (26, ~195 s) is the pre-merge Stage 1 figure and is
stale; its *metrics* table still reproduces exactly, as above. `docs/status.md`
is the post-merge truth for the count. Reproduce whichever applies to what you
touched, before and after, and paste the table into your summary.

---

## 7. Known broken / open, in priority order

Items 1–3 are environment breakage found while building this venv on
2026-09-12; items 4 onward are the project's own open work.

1. **Root `requirements.txt` and `package.json` no longer exist.** They were
   deleted (not moved) in the 2026-09-12 "folder cleanups" commit along with
   `README.md`, `PROJECT_BRIEF.md`, and all of `docs/`. Before that commit,
   `requirements.txt` had been repinned to `pandas==2.3.3` (from `3.0.2`, which
   conflicted with fastf1's `pandas<3.0.0`) — that fix is now undocumented
   anywhere except this file and this machine's already-installed `.venv`
   (§1). There is no manifest to reinstall the Stage 2 stack from; the pinned
   package list above (§1) is reconstructed from `pip list` on this machine,
   not from a committed source of truth. Restoring a real manifest (a root
   `requirements.txt` or a `pyproject.toml`) is open work, not done.
2. **`tests/test_live_readiness.py::test_kernels_are_pure` fails on Windows**
   with `UnicodeDecodeError: 'charmap' codec can't decode byte 0x81`. It calls
   `path.read_text()` with no `encoding=`, so it gets cp1252; `xray/realfit.py`
   has a UTF-8 `§` in its docstring. A one-word fix (`encoding="utf-8"`), but it
   is an invariant-guarding test — the look-ahead/kernel-purity check — so change
   it deliberately, not in passing. `test_estimator_is_blind` in
   `test_estimator.py` reads the same way and will hit this the moment a non-ASCII
   character lands in `estimator.py`.
3. **`tests/test_golden.py::test_belief_matches_golden` fails on a clean
   checkout here** — 51 of 3972 `soc_p10` samples off by up to 2.6e-4 J, a
   relative 9.1e-8 against `rtol=1e-9`. No code changed (`git status` clean), so
   this is the golden `.npz` being platform-specific at that tolerance, not a
   regression. **Do not run `scripts/99.make_golden.py` to make it go away** — that
   is precisely the silent-rebaseline this harness exists to prevent. The right
   fix is either a tolerance that reflects cross-platform float64 accumulation,
   argued in the commit, or a golden regenerated on a named reference platform.
4. **`overtake.py` holds the one invented number in the codebase** — `b0..b3`,
   with `b1`/`b2` from the design brief and `b0`/`b3` solved to hit its two shape
   anchors. They propagate into every decision claim. They cannot be fitted until
   the estimates feeding them are validated on real data, which is item 3 of
   `docs/status.md`. If you touch that file, say explicitly which number is still
   assumed.
5. **Band calibration.** Deployable coverage 0.67–0.80 where it should be ≥ 0.85.
   The missing variance is named rather than tuned away: the CdA posterior is not
   propagated into the flows, and the per-lap drift nuisance is uniform where the
   real thing is a strategy.
6. **The fixed-policy fixture is the largest unfalsified assumption in the
   stack.** The shape likelihood is trivially well specified against a driver
   whose v_cut never moves. A drifting-v_cut simulator variant is the next test;
   if step 14's gain does not survive it, the polytope becomes the headline.
7. **`scripts/02.run_estimator.py --seed 42` raises an unhandled traceback** on a
   *correct* refusal (the tow-bound FOLLOWER). Pass `--car LEADER`. The fix is to
   catch `EstimatorError` and print the refusal — not to make the estimator
   return a number.
8. **Seed 42 band coverage runs 0.94–0.97** on the Stage 1 path, above both the
   stated 0.86–0.90 and the test's own 0.92 ceiling; it passes only because the
   test averages three seeds. Understand the over-coverage; do not widen the
   ceiling.
9. **`shot.mjs` hardcodes `/Users/apple/Desktop/x-ray/out/shots`**, the original
   author's macOS path. It writes nowhere useful on Windows or Linux.
10. **`_decision_cached` in `api/main.py` is dead** — `decision()` calls
   `_decision_payload` directly, and its local imports are unused.
11. **`docs/status.md` item 3 names the wrong races.** It asks for "one pre-Miami
    race (Bahrain or Jeddah)". In the 2026 schedule fastf1 returns, Miami is
    round 4 on 2026-05-03, so the only pre-Miami rounds are **1 Melbourne,
    2 Shanghai, 3 Suzuka**. Bahrain is round 16 on 2026-10-04 — post-Miami, and
    therefore under the 350 kW super-clip where the local ICE floor is dead — and
    there is no Jeddah round at all. That schedule also reports Bahrain's
    `Location` as "Kuala Lumpur", which is wrong and matters because
    `xray/data/circuits.py` keys geometry off the circuit. Check the schedule
    before picking a round, and pick from 1–3 for the pre-Miami check.
12. **Five races is not enough to power the RDD**; roughly ten would be. The RDD
   running variable is also the gap at the start/finish line rather than at each
   Manual Override detection point, which adds noise and costs power.

---

## 8. Before you open a PR

- `pytest -q` passes, `test_estimator_is_blind` included.
- Headline metrics reproduced — the right baseline from §5 — and pasted into the
  summary. A change that moves one outside noise is a **finding to report, not a
  test to relax**.
- No new speed threshold, no new "clean sample" filter, no fallback guess. CdA is
  bracketed; the width of the surviving interval is the identifiability score.
- No new invented constants. If one is unavoidable it is named `ASSUMED_*` with
  its derivation in the comment beside it.
- Any printed effect estimate prints its MDE next to it. Nulls stay nulls.
- Refusals are caught and reported — never an unhandled traceback, never replaced
  by a number.
- Optimisations: profile first, report before/after wall time, and match the
  golden `.npz` to a stated tolerance.

---

## 9. House style

Comments explain **why the wrong version was wrong**, with the measured
consequence attached — not what the code does.

```python
# No speed threshold here. The 340 km/h "pure engine" window looked clean in the
# simulator but 0.00-0.11% of real samples reach it (Monaco: none, max 292 km/h),
# so a thresholded estimator returned nothing on five of six races. Every sample
# brackets CdA instead; the interval width is the identifiability score.
```

Cite the number that killed the alternative. Refusals get a comment saying what
would have to be true for the estimate to exist. No hedging adjectives — state
the measurement. Docstrings short; the argument goes in the comment nearest the
decision. Targets that were missed are argued, not tuned. Bugs are recorded, not
erased. Limitations lead.
