---
name: x-ray
description: Working rules for the X-RAY codebase — reconstructing a rival F1 car's hidden electrical energy state (deployable energy, CdA) from public speed telemetry alone, across a simulator, a set-membership research stack, and the real-data pipeline that actually ships (FastF1 ingest → `xray/data/ingest.py` → `xray/realfit.py` → `xray/analysis.py` → `simulation/api/`, `simulation/app/`). Use this whenever touching any `xray/*.py` module or `xray/data/`, any numbered script in `scripts/`, the tests, the API, or the frontend; whenever asked to make the simulation or filter faster; and whenever writing or editing comments in this repo. Consult it even for small edits — the repo has hard invariants (blindfold, layer discipline, bracket-not-threshold, decline-not-guess, regulation-variant agreement, quote-the-stack-that-produced-the-number) that a well-meaning change can silently break.
---

# X-RAY

One claim: a rival's hidden electrical energy state can be reconstructed from its
public speed trace alone and used to time an overtake. Public telemetry has
speed/throttle/brake/gear/RPM/position and no energy channel. The absence is the
premise; everything below defends it.

Two quantities matter and they are not equally identifiable:
- **Deployable energy** — how much the driver can still spend. A flow integral,
  measured well.
- **Drag area (CdA)** — needed to convert speed into power at all. Bounded by the
  regulation from both sides, never thresholded.

The reporting discipline this implies is the project's spine: the
**assumption-free identified set** and the **policy-tilted posterior** are always
reported as two numbers together. On real data the tilted posterior rests on a
behavioural assumption about how drivers deploy, and that cannot be checked —
never collapse it to one number.

## Read this first: there are three inference implementations, not one

| Stack | Modules | Reached by | Input struct |
|---|---|---|---|
| **Stage 1** | `estimator.py` | scripts `02`, `03.a`, `03.b`, `04`, `05`, `99.make_golden`, `render/video.py`, `closedloop.py` | `observe.Observation` |
| **Set-membership group** | `regs`, `balance`, `setmem`, `modes`, `pipeline`, `rbpf`, `deadband`, `strategy`, `pooling`, `qmdp` | **`tests/` only** | `pipeline.Feed` |
| **Real** | `realfit.py`, `analysis.py` | scripts `07.b`, `08.b`, the API, `decision_service` | `realfit.Kin` |

Only the third ships. A knob in the second cannot affect a real race — tuning
`balance.N_SIGMA_DEFAULT`, `setmem.OUTLIER_FRAC`, `rbpf.CLOSURE_SIGMA_J`,
`deadband.BIC_MARGIN` changes no shipped output, and the metrics they produce
describe code no user reaches. The keep-or-retire decision for that group is
open; what is **not** open is that its numbers must not be quoted as if they
described the product. See `docs/pipeline_layers.md` §4 and §10.

The two entry points are unified from `grid_lap` onward — the same files, not
two modules calling a shared helper:

```
FastF1 session ──► ingest.frame_from_fastf1 ──────┐
                                                  ├──► ingest.grid_lap ──► realfit ──► belief ──► decision_service
GroundTruth ──► observe ──► frame_from_observation ┘
```

`xray/simfeed.py` existed briefly as a separate synthetic adapter and was
deleted: it reimplemented `_lap_frame` instead of reusing it, and the copy
drifted within a day (its `usable` column meant per-cell where the real one
means per-lap, a difference `analysis.py` acts on). Don't recreate it.

## Architecture — every file belongs to exactly one layer

| Layer | Rule |
|---|---|
| **REGULATION** | One source. Never restated in another module. |
| **SHARED PHYSICS** | Identical for both data sources. No source-specific branch. |
| **SIMULATION** | May read `config/default.yaml`. **Never imported by an inference module.** |
| **SOURCE ADAPTER** | The *only* layer allowed to know where data came from. |
| **INFERENCE** | **Must be source-agnostic.** One body of code behind two adapters. |
| **DECISION** | Consumes belief, never ground truth. |
| **SCORING** | Ground truth allowed here and nowhere else. |
| **PRESENTATION** | Reads results. Computes no physics. |

| Layer | Files |
|---|---|
| REGULATION | `constants.py` (2026 power unit, taper curve, store size); `regs.py` (date-dispatched `PRE_MIAMI`/`POST_MIAMI` variants) |
| SHARED PHYSICS | `vehicle.py` (the single authoritative force/energy integrator); `tyres.py`; `environment.py` (weather, air density, wind projection); `physics_context.py` |
| SIMULATION | `sim.py` (two-car sim → `GroundTruth`); `track.py` (Circuit Sigma); `policy.py` (the driver model); `config.py` |
| SOURCE ADAPTER | `observe.py` (`GroundTruth` → `Observation`, plus `public_channels`); `data/ingest.py` (the one gridding module); `data/circuits.py` (`RealTrack`: centreline, elevation, curvature, zones) |
| INFERENCE | `estimator.py` (Stage 1, locked to `Observation`); `realfit.py` (the only inference a real race executes); `analysis.py` (session → `out/races/<id>.json`); `balance` `setmem` `modes` `pipeline` `rbpf` `deadband` `strategy` `pooling` (tests only) |
| DECISION | `decision.py` (zone calibration, DP solvers); `overtake.py` (`p_pass`, synthetic coefficients); `qmdp.py`; `opportunity.py` (opportunity-horizon solver, wired); `stint.py` (pit context, causality gate); `decision_service.py` (canonical web/core entry point); `passmodel.py` (real overtake labels — **not** wired, deliberately) |
| SCORING | `metrics.py` (the only module allowed to read ground truth); `closedloop.py` (closed-loop policy evaluation, synthetic only) |
| PRESENTATION | `render/`; `simulation/api/main.py`; `simulation/app/` (React/Three.js) |

`theta = (CdA_X, CdA_Z, F_rr, dm)`: the two aero states, rolling resistance as a
force at a reference mass, and dry mass minus published weight.

## Invariants — do not break these, and add a test if you touch them

1. **Inference is blind to the simulator.** No INFERENCE module imports a
   SIMULATION module. Enforced twice:
   `tests/test_estimator.py::test_estimator_is_blind` parses `estimator.py` with
   `ast`, and `tests/test_layers.py::test_no_inference_module_imports_the_simulator`
   runs the same scan across all eleven inference modules including `realfit.py`.
   All eleven are currently clean — it is a guard, not a to-do list. Do not
   "helpfully" share a constants module either; if a number lives in the
   simulator, inference must derive it or take it as a public-regulation input.
2. **CdA is bracketed, never thresholded.** Wheel power is linear in CdA and the
   2026 regulation bounds it from both sides, so *every* sample brackets CdA.
   Real telemetry across six 2026 races clears 340 km/h on 0.00–0.11% of samples
   (Monaco: none, tops out at 292 km/h) — a speed-threshold "clean sample"
   estimator returns nothing on five of six real circuits. Never reintroduce a
   speed threshold or a fallback guess. Interval width is the identifiability
   score; drag is otherwise identified only in the dead band `v_cut < v < v_harv`
   (`xray/deadband.py`), where the policy sets P_K = 0 and the residual is pure
   `delta_CdA * 0.5 * rho * v^3`.
3. **Decline rather than guess.** Silverstone and Monaco legitimately return 0%
   identifiability on real 2026 data. A `FOLLOWER` under tow, or a car with no
   dead-band samples, is legitimately unidentifiable. Refusal is a correct
   output, not an error path to paper over — but it must be *caught and
   reported*, never an unhandled traceback (see Gotchas).
4. **Report deployable energy, not raw store.** A speed trace measures flows
   exactly and absolute level only up to a constant: `E_k = c + F_k` with `c`
   unidentified. If the reserve has been touched at least once,
   `D_k = F_k - min_{j<=k} F_j` exactly; before that it's a lower bound. Raw
   store is reported as the *bracket* `c` admits, never as a point estimate with
   a chosen `c` — an earlier version centred `c` in the interval and it read as a
   real number. Do not add a raw-store headline or try to "fix" its coverage;
   that width is correct.
5. **No cut-out detector for the reserve.** A detector ("deployment stops while
   throttle is full") cannot distinguish a genuine reserve hit from a policy
   cut-off without a policy, and its false positives at a near-full store
   collapsed the deployable band to 0.04 coverage before the running-minimum
   form replaced it. If a detector is reintroduced, gate it on a *sustained*
   deviation from the policy prior, not a single sample.
6. **Store closure is solved, not guessed, and per-lap drift is bounded, not
   forced to zero.** Deployed ≈ harvested within one store over the *race*, but
   forcing exact per-lap closure is wrong when a lap genuinely starts on a full
   store and dumps energy (coverage collapses to 0.01 if you do). Each particle
   carries a per-lap drift nuisance bounded by `E_STORE_MAX / n_laps` — derived
   from the store size, not invented — and the store box does the rejecting.
7. **Nulls are reported as nulls, with power.** The real-data RDD at the 1.000 s
   Manual Override boundary is −0.20 MJ, p = 0.41, MDE 0.68 MJ vs a 0.5 MJ
   allocation, over 5 races / 1,851 car-laps. Any code path that prints an
   effect estimate prints the MDE next to it. Never reframe a null as evidence
   for the regulation working or not working.
8. **One rulebook object, and every consumer must be on the same variant.** This
   has now bitten twice, both times at the same 250/350 = 0.71 ratio:
   - The simulator once ran 350 kW (post-Miami) harvest while fixtures assumed
     250 kW for a full iteration; every measurement came out 0.71× on harvest.
   - `regs_for(race_date)` was called at exactly one site — `pipeline.py:72`,
     inside the group only tests reach — while `realfit.py` took `P_MGUK_MAX`
     from static `constants` unconditionally. **Every real race was analysed at
     a 350 kW harvest cap**, wrong for pre-Miami rounds 1 Melbourne, 2 Shanghai,
     3 Suzuka. Fixed: `analysis.analyse` resolves `regs_for(sd.date)` and threads
     a `RegSet` through; `realfit._p_harv_max` is the one place the cap is read;
     the variant is published in the race artefact under `regulation`.

   The blindfold forbids inference *importing* the simulator; it does not forbid
   a test asserting `sim.rules == fixture.rules`. A variant mismatch must fail
   loudly, not silently scale results. **Still open:** `p_mguk_ceiling(v)` remains
   the unconditional taper because `Kin` carries no zone mask;
   `regs.p_dep_max(v_ms, in_zone, regs)` is the call to make once `RealTrack`
   supplies one.
9. **Assumption-free set and policy-tilted posterior are always two numbers.**
   `Belief.theta_polytope` carries the assumption-free interval; the RBPF
   posterior is tilted by the strategy-shape likelihood, which is a behavioural
   assumption unverifiable on real data. A test asserts the posterior lies
   inside the polytope. Never present only the tilted number.
10. **Don't treat a heuristic or a calibration placeholder as physical ground
    truth.** See "Model inventory" below before adding presentation code that
    reads a value as if it were measured physics.
11. **Quote a metric with the stack that produced it.** Stage 1, the
    set-membership group and real are three different baselines and have been
    quoted interchangeably before. `docs/dev_readme.md` §6 holds the two live
    baselines; `docs/pipeline_layers.md` §7 says which belief producer each
    evaluation harness uses. Every evaluation in the repo scores the Stage 1
    belief; every real race uses a different one, and nothing has measured the
    difference — so closed-loop results do not characterise real decision quality.
12. **Tune against the stack that ships.** The only knobs that reach a real race
    are the ones in `docs/pipeline_layers.md` §10: `realfit.RESERVE_SIGMA_REAL`,
    `COAST_THROTTLE`, `COAST_MIN_DECEL`/`COAST_MAX_DECEL`, `BRAKE_DECEL`,
    `build_kin`'s `smooth_m`, the wind grid and robust quantiles in
    `fit_nuisance_real`, `analysis.n_particles`, `analysis.MIN_IDENT`, and the
    `identifiability >= 0.25` / `n_binding >= 50` gate in
    `RealNuisanceFit.usable`. They are also the least test-covered code in the
    repo, which is exactly where tuning effort belongs.
13. **A fix validated on synthetic can be a straight loss on real.** The worked
    case (`docs/pipeline_layers.md` §11): dilating the brake mask by the
    Savitzky-Golay half-window was quantitative, reproducible, pointed at a
    genuine code asymmetry, tightened the synthetic identified set from
    `[-2.19, 2.15]` to `[-0.67, 0.62]`, and left the whole suite passing. On real
    telemetry it cut identifiable cars 7 → 2, halved median `n_binding`, and made
    the pooled CdA refuse. The pathology was the simulator's step-function brake
    onset; real braking ramps. Reverted. **Repairing the shipping path to
    accommodate a simulator limitation is the wrong direction every time**, and
    no test will stop you — nothing scores `realfit` against real ground truth.
14. **Any test that scans source files passes `encoding="utf-8"`.** Omitting it
    reads cp1252 on Windows and dies on the `§` in `realfit.py`'s docstring. This
    has broken the suite three times.
15. **`tyre_life` is observed age in laps and is never wear.** `wear_fraction` is
    the modelled quantity. `tests/test_tyres.py::test_tyre_life_is_age_and_is_never_wear`
    asserts the shortcut is absent from the source text, not just from behaviour.
16. **Unavailable is not zero.** Wind and weather absence is `available=False`
    with value `None`, never `0.0`, so "unknown" cannot be read as "calm" and no
    headwind/tailwind is inferred. Same discipline in `TrackFrame.oriented`,
    which rejects `"unknown"`/`"assumed"` as a provenance so an undocumented
    heading cannot enable the wind projection.

## Numbers live in `docs/`, not here

This file once carried headline-metric tables. They went stale for four commits
and, worse, quoted set-membership-group figures as though they described the
product. Both failures are structural, so the tables are gone. Before and after
any change, reproduce the baseline for the stack you touched and paste it into
your summary, **naming the stack**:

- `docs/dev_readme.md` §6 — the two live baselines, and which applies to what.
- `docs/dev_readme.md` §7 — known broken / open, in priority order.
- `docs/pipeline_layers.md` §4–§7 — which stack produced which number, the
  divergence register (mass, fuel burn, reserve sigma, regulation source), and
  which harness scores which belief.
- `docs/model_inventory.md` — what each value *is*.
- `docs/SCRIPT_PIPELINE_ORDER.md`, `docs/STARTUP.md` — entry points and setup paths.

Re-measure rather than trusting a written-down count; a count in a document is
evidence about the day it was written. If a change moves a metric outside noise,
that is a finding to report, not a test to relax.

## Gotchas (found, not read — verify they are still true before relying on them)

- `python scripts/02.run_estimator.py --seed 42` raises an unhandled traceback on
  a *correct* refusal (the tow-bound `FOLLOWER`). Use `--car LEADER`. The fix is
  to catch `EstimatorError` and print the refusal, not to make the estimator
  return a number.
- `tests/test_golden.py::test_belief_matches_golden` fails on a clean checkout
  here: 51 of 3972 `soc_p10` samples off by up to 2.6e-4 J, relative 9.1e-8
  against `rtol=1e-9`. Nothing changed, so this is the golden `.npz` being
  platform-specific at that tolerance. **Do not run `scripts/99.make_golden.py`
  to clear it** — that is the silent rebaseline the harness exists to prevent.
  Either argue a tolerance that reflects cross-platform float64 accumulation, or
  regenerate golden on a named reference platform.
- Two tests fail on Windows with `UnicodeDecodeError` —
  `test_live_readiness.py::test_kernels_are_pure` and
  `test_physics.py::test_one_regulation_curve_and_every_module_uses_it`. One word
  each (invariant 14), but both are invariant-guarding, so change them
  deliberately rather than in passing.
- Two bugs that produced *plausible-looking* wrong numbers before being caught,
  worth re-checking for the pattern whenever a metric looks slightly-off-but-
  reasonable: (a) `sim.step()` mutates state then records, so `trace.v[k]` is
  post-step while `trace.P_mguk[k]` is the power applied during it — always
  close energy against index `k+1`, not `k`; (b) after a particle-filter
  resample, per-particle flow bands (`e_wheel`, `d_lo`, `d_hi`, `harvest`) must
  be recomputed from the *current* theta, not carried over from the initial
  draw — otherwise a resample silently scores each particle's dead-band
  residual against a different particle's drag.
- Deployment-floor terms need 5-sigma measurement slack when accumulated over an
  interval; a bare `max(0, e_wheel - e_ice_max)` with no slack rectifies noise
  one-sidedly and can inflate a lap's claimed deployment several-fold, driving
  `range(F)` far past the store size and making the store-box reject the truth.
- **Mass on the real path is constant for a whole race and measurably too low.**
  790 kg ships; it sits under the 768 kg dry minimum plus any realistic fuel, so
  it is an end-of-race figure applied to a whole race, and the one-sided bias
  follows (too little mass understates work done, so deployment comes out low).
  Measured with CdA pinned to truth: 790 kg → 8.2% MAPE / −8.2% bias, 821 kg →
  5.4% / −1.8%. **A fuel model is not the fix** — a linear burn loses even handed
  the simulator's own exact fuel numbers — and 821 kg does not transplant,
  because it is 768 plus the *simulator's* end-of-race fuel. FastF1 publishes no
  fuel signal at all. `realfit.mass_series` carries the measurement and
  `build_kin` accepts a per-sample mass array, so don't re-derive this.
- `overtake.py` holds the one *invented* constants in the codebase — `b0 =
  -7.6650`, `b1 = 0.55`, `b2 = 2.10`, `b3 = 2.4324`, with `b1`/`b2` from the
  design brief and `b0`/`b3` solved from two shape anchors — and they propagate
  into every decision claim. `opportunity.py` takes every pass probability from
  `overtake.p_pass`, and `decision_service.py` caps decision confidence at `0.35`
  because of it. Fitting them is gated behind validating the estimator on real
  data first. Do not add more invented constants; if you must, name them
  `ASSUMED_*` and comment the derivation (e.g. `ASSUMED_MIAMI_2026` for the
  pre/post-Miami changeover date — the single most worth re-checking).
- `passmodel.py` is deliberately **not** wired into the decision path. It builds
  a real labelled overtake dataset, excludes pit cycles via tyre-life resets and
  compound changes, and bounds safety-car contamination rather than ignoring it
  (there is no track-status channel in the payload). Its `fit()` refuses until
  `audit_dataset` passes, and it cannot: `MIN_RACE_GROUPS = 4`,
  `MIN_USABLE_SAMPLES = 300`, `MIN_POSITIVES = 40`, against exactly one race
  artefact on disk. Wiring it early is the failure mode the audit exists to
  prevent.
- `closedloop.WORLD_VARIATION` is the right mechanism on the wrong axes. It
  varies `sim.start_gap_s`, `sim.e_start_frac`, `sim.leader_policy` and
  `observe.speed_noise_ms` — four synthetic *scenario* parameters, no real-world
  nuisance. `closedloop.py:141` reads `cfg["observe"]["rate_hz"]` but `world()`
  never varies it, so every world runs at a clean regular 3.7 Hz while real
  telemetry is 4.17 Hz median and irregular (p90 gap 0.36–0.40 s), sitting on the
  accuracy cliff the rate ablation found near 4 Hz. The axis most likely to break
  transfer is the one held fixed. Any new axis must have its range measured and
  cited from feasibility output: "a randomisation chosen to make P2 look good is
  not a randomisation, it is a fixture."
- The synthetic `Feed` is *easier* than the real one in some channels and the
  real trace is richer in others. Synthetic `throttle`/`brake` come straight off
  the simulator's `regime` label (noiseless binary), `rpm` is pinned at redline,
  `in_zone` is `True` everywhere, sampling is regular. Meanwhile
  `observe.Observation` carries only `t, s, v, lap, gap_to_leader` — speed is
  "THE ONLY SIGNAL" — while FastF1 also supplies throttle, brake, gear and RPM.
  Information content differs in both directions, a third reason a knob tuned on
  one stack does not transfer to the other.
- Stage 2 requires `pip install -r requirements.txt` (adds `fastf1`, `fastapi`,
  `uvicorn`), `python scripts/08.b_analyse_race.py --round N` (downloads real
  telemetry via FastF1 — slow, network-bound, cache it), and
  `cd simulation/app && npm install && npm run build` before the API/frontend serve
  anything real. Check `.venv/` contents and whether `out/`/`simulation/app/dist` exist
  before assuming Stage 2 has been run locally.
- An earlier version rejected a whole lap on a single telemetry gap, discarding
  ~⅔ of every real race. Gaps are handled at sample level (grid cells spanning a
  gap are blanked, never interpolated across); don't reintroduce lap-level
  rejection on top of that.
- Before concluding a file was deleted, run `git log --follow` / `git diff-tree -M`.
  A claim that `requirements.txt` had been dropped cost real time; git had logged
  it as `R100`, a 100%-identical rename, tracked the whole time.

## Model inventory — don't treat these as more certain than they are

When writing presentation code (UI, reports, decision explanations), match the
claim strength to the category. The full labelling is
[docs/model_inventory.md](../../../docs/model_inventory.md), which covers
physics, regulation (single source), Manual Override semantics, P1 environment
and wind frames, P1 aerodynamics, and the tyre/stint models. Read it before
adding presentation code that reads a value as if it were measured physics.

The two most often mistaken for physics:

- `xray.overtake.p_pass` — coefficients are synthetic design anchors, *not* fit
  to real race data (see Gotchas). Everything downstream inherits this.
- `xray.estimator.RESERVE_SIGMA` — a synthetic observation-noise calibration
  knob, not a battery parameter. Its own comment records being moved
  `2.5e5 → 1.0e5` to shift two `test_band_coverage` criteria, i.e. to hit a
  coverage target. Real independently uses 5× that value (`realfit.py:40`).
  Synthetic cannot say which is right for real data, because real data has no
  `E_true`.

## The generalized calibration (the replacement for the 340 km/h window)

Every sample brackets CdA — no threshold anywhere:

```
(-P_MGUK_MAX - A)/B  <=  CdA  <=  ((P_ICE_MAX + ceiling(v))*eta - A)/B
```

Both bounds come from the rulebook, not an engine model — an earlier version
inferred ICE output from throttle and got `CdA > 13` lower bounds, because a
real car at 30 m/s is traction-limited, not power-limited, and no
throttle-to-power curve survives that. Sharpen with the feed: brakes-on drops
the lower bound (brake force isn't modelled); throttle-off drops the 400 kW ICE
term, tightening the upper bound severalfold (the coast-down channel — exists
at every circuit, unlike a high-speed window). Pool identified sets across the
field (20 cars, same regs, same air); a car with no clean running inherits the
field constraint and must be flagged as such in the UI.

Deployment is a *band*, not a point: lower bound = whatever exceeds the 400 kW
ICE cap must be electrical; upper bound = the MGU-K ceiling at that speed. Where
in the band the truth sits is not visible in the trace — it's pinned by store
closure (deployment ≈ recovery within one 4 MJ store over the race). A single
global split spreads deployment evenly around the lap where a real car bursts it
out of slow corners — visible in the app as the reconstructed store reading
empty for much of the race with an over-narrow belief band. That's a known,
stated gap, not a bug to quietly patch by widening the band to look better.

## Efficiency rules for the simulator and filter

1. **Measure first.** `python -m cProfile -o prof.out <entry>` then
   `python -c "import pstats; pstats.Stats('prof.out').sort_stats('cumtime').print_stats(25)"`.
   For a single hot function, `line_profiler` (`kernprof -l -v`). Don't optimise
   anything outside the top of the profile.
2. **Vectorise the time axis.** Fixed-rate traces: replace
   `for i in range(len(speed))` with numpy array ops (`np.diff`, `np.cumsum`,
   boolean masks, `np.where`). Preallocate outputs; never append inside the
   sample loop.
3. **Keep the integrator's order of operations.** Check an explicit time-step
   recurrence is actually vectorisable (no dependence on its own previous
   output) before vectorising it. If not, keep the loop and JIT it
   (`numba.njit`, optional behind an import guard) rather than rewriting the
   physics.
4. **Pandas out of hot paths.** FastF1 returns DataFrames; convert to numpy once
   at the boundary. No per-row `.iloc`, no `.apply` with a Python lambda.
5. **Cache expensive inputs.** FastF1 sessions, generated reference laps, and
   per-track regulation tables should be cached (FastF1's own cache dir,
   `functools.lru_cache`, or `.npz` for generated traces). Seed all randomness
   explicitly and store the seed with the cache key.
6. **Keep the test suite cheap.** Move generated traces into session-scoped
   pytest fixtures; mark multi-seed coverage tests `@pytest.mark.slow`; keep a
   `-m "not slow"` path fast. The blindfold and layer scans must stay in the
   fast path.
7. **Prove it.** Every optimisation PR reports before/after wall time *and* the
   baseline for the stack it touched, unchanged to stated precision. A speedup
   that shifts MAPE by 0.1 pp is a bug until explained.

## Design for live (a later phase — shape the code for it now, don't build it)

1. **Kernels are pure.** Every numerical step (wheel power, CdA bracket update,
   deployable-energy accumulation, overtake decision) is a function of
   contiguous `float64` numpy arrays → arrays/scalars. No FastF1 objects,
   pandas, I/O, or plotting inside a kernel. The eventual C++/Rust port moves
   kernels only. Enforced by `test_live_readiness.py::test_kernels_are_pure`.
2. **Batch == streaming, tested.** Expose `state = update(state, sample)`
   alongside `batch(trace)`, and assert `batch(trace) == reduce(update, trace)`
   bit-for-bit on fixed seeds. This also catches hidden look-ahead (a filter
   peeking at future samples) — treat it as seriously as the blindfold.
3. **Golden outputs define correctness.** Keep a fixed-seed `.npz` of estimator
   outputs per test trace (`scripts/99.make_golden.py`, `tests/test_golden.py`).
   Every optimisation must match golden to a stated tolerance before it merges.
4. **No live-only code paths yet.** No threads, no ring buffers, no sockets. A
   change only useful for live operation waits.

## House style (the codebase has a voice — keep it)

Comments explain *why the wrong version was wrong*, with the measured
consequence attached. Not what the code does.

**Bad:**
```python
# Clip deployment above 340 km/h
```

**Good:**
```python
# No speed threshold here. The 340 km/h "pure engine" window looked clean in the
# simulator but 0.00-0.11% of real samples reach it (Monaco: none, max 292
# km/h), so a thresholded estimator returned nothing on five of six real
# races. Every sample brackets CdA instead; interval width is the
# identifiability score.
```

Rules:
- Cite the number that killed the alternative (percent of samples, MJ,
  p-value).
- Refusals get a comment saying what would have to be true for the estimate to
  exist.
- No hedging adjectives ("robust", "careful") — state the measurement.
- Keep docstrings short; put the argument in the comment nearest the decision.

## Before you finish any task

- `pytest -q` passes, including the blindfold and layer scans.
- Reproduce the baseline for the stack you touched (`docs/dev_readme.md` §6) and
  paste it into your summary, **naming the stack**. Do not quote a
  set-membership-group figure as a product number.
- Any new test that scans source files passes `encoding="utf-8"`.
- No new speed thresholds, no new fallback guesses, no new invented constants
  outside `ASSUMED_*` with a stated derivation.
- The assumption-free polytope and the policy-tilted posterior are both
  reported if either changed — never just the tilted number.
- If you changed anything a real race executes (`data/ingest.py`, `realfit.py`,
  `analysis.py`, the knobs in invariant 12), say so explicitly — it is the least
  test-covered code in the repo and synthetic evidence does not vouch for it.
- If you touched `overtake.py`, say explicitly which numbers are still
  invented/assumed.
- If you touched anything in `docs/model_inventory.md`'s heuristic or
  statistical categories, say so — those are the values a reviewer should not
  mistake for physics.
