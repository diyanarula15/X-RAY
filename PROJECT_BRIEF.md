# X-RAY — project brief

Repo: https://github.com/diyanarula15/X-RAY · local: `~/Desktop/repos/X-RAY` · main @ `cb09916`
Stack: Python 3.11 (numpy/scipy) + FastAPI backend; React 19 / TypeScript / Vite / react-three-fiber / D3 / zustand frontend.

## The one claim

**A rival Formula 1 car's hidden electrical energy state can be reconstructed from its public speed
trace alone — and that reconstruction is good enough to time an overtake better than a driver acting
blind.**

Public F1 telemetry publishes speed, throttle, brake, gear, RPM and position. It publishes **no
energy, deployment or state-of-charge channel**. That absence is the entire premise: the project
infers the hidden electrical state from dynamics, then uses it to make a strategic call.

There are two stages, both in the repo:

- **Stage 1** — the method proved inside a simulator the authors control (synthetic "Circuit Sigma"),
  where ground truth exists and the estimator can be scored exactly.
- **Stage 2** — the same method run on **real 2026 F1 telemetry via FastF1**, shipped as a six-view
  interactive web instrument instead of a rendered video.

## Why it is hard (first principles)

A speed trace gives you total force at the wheels. The car makes that force from two sources at the
same axle — a 400 kW internal combustion engine and a 350 kW electric motor (MGU-K) — and the trace
cannot see which is which. To recover the electrical part you must first know the drag area (CdA),
which is also unknown, and drag dominates the power balance at speed. So the problem is:

1. **Identify the nuisances** (drag area, wind) without ever being told them.
2. **Split wheel power** into engine vs motor, using only the regulation's own bounds.
3. **Integrate the flows into a level** (state of charge), which is only weakly identified.
4. **Act on it** — decide which lap and which corner to attack.

### Stage 1's key insight, and why it died
The 2026 regulation *tapers* MGU-K deployment to zero above 355 km/h. So above ~340 km/h the car is
almost a pure engine-vs-drag balance and CdA falls out. Stage 1 used that window (lowered to
320 km/h, because at 340 the residual ceiling is too small to separate from drag).

Stage 2's feasibility study **killed that window**: across six real 2026 races, **0.00%–0.11% of
samples clear 340 km/h**, and Monaco tops out at 292 km/h. A speed gate does not survive real data.

### Stage 2's replacement — interval identification
For a given wind offset, observed wheel power is *linear* in CdA, and the regulation bounds total
wheel power from both sides. Rearranged, **every single sample brackets CdA**:

```
(-P_MGUK_MAX - A(k)) / B(k)  <=  CdA  <=  ((P_ICE_MAX + ceiling(v_k)) * eta - A(k)) / B(k)
```

No threshold anywhere. A 340 km/h sample has near-zero ceiling and pins CdA tightly; a 120 km/h
sample barely constrains it. Intersect all of them; **the width of the surviving interval IS the
identifiability score.** Sharpened by two channels the feed provides: brakes-on drops the lower
bound, throttle-off removes the 400 kW term and tightens the upper bound severalfold (the
coast-down channel — which exists at every circuit on earth, unlike a high-speed straight).
Finally, twenty cars run the same rules through the same air, so identified sets are **pooled across
the field**; a car with no clean high-speed running inherits the field constraint and is flagged.

Measured: Zandvoort 97% identifiable, Spa 78% (pooled CdA 0.87 m² — a real F1 figure), Melbourne
72%, **Silverstone 0%, Monaco 0%**. The last two return nothing, and the project treats that as the
result rather than a failure: a circuit that never reaches high speed cannot pin drag from a speed
trace, so the method **declines instead of guessing**, with no per-circuit tuning.

### The reserve degeneracy (the honest core limitation)
A speed trace measures energy **flows** exactly and the absolute **level** only up to a constant. A
deployment cut-out does not mean the store is empty — it means the store hit whatever buffer that
driver refuses to spend. A particle offset high can explain every observation by claiming a large
buffer. Partial identification comes from the fact that a buffer held all stint gets spent at the
end of it. So the estimator reports two things and headlines the second:

- `soc_*` — belief about the raw store. Honest, but weakly identified (coverage 0.09–0.42).
- `usable_*` — **deployable energy** = store minus buffer. Identified (bias ≈ 0.05 MJ), and it is
  also the only quantity that decides anything.

## Architecture

```
xray/constants.py   2026 regulation constants + public knowledge (estimator MAY import)
xray/track.py       synthetic Circuit Sigma; public to everyone
xray/vehicle.py     ground-truth longitudinal dynamics + energy store   <- estimator MUST NOT import
xray/policy.py      the "driver": 4 interpretable params (front_loading, reserve, gap_sensitivity, zone_preference)
xray/sim.py         two-car simulator -> GroundTruth
xray/observe.py     THE BLINDFOLD: GroundTruth -> Observation (7 fields, none of them energy)
xray/estimator.py   Observation -> BeliefTrace   <-- the contribution (Stage 1)
xray/realfit.py     generalized interval calibration for real telemetry (Stage 2)
xray/analysis.py    whole pipeline over a real session -> one precomputed JSON per race
xray/data/          FastF1 ingest onto a common distance grid; circuit geometry from position stream
xray/overtake.py    P(pass) logistic — PLACEHOLDER coefficients, not fitted to anything real
xray/decision.py    zone calibration, backward-induction DP, robustness sweep
xray/metrics.py     scoring; the ONLY module allowed to read ground truth, strictly after the fact
api/main.py         FastAPI: serves precomputed artefacts; computes only the live RDD
app/                React instrument, six views
```

**The blindfold is enforced by a test, not by convention.** `tests/test_estimator.py::test_estimator_is_blind`
parses `estimator.py` with `ast` and fails the build if it ever imports `sim`, `policy` or `vehicle`.
A second test asserts `Observation` has exactly seven fields and none is an energy/throttle/brake/
gear/deployment channel.

### Estimator, three stages
- **A — nuisance fit.** CdA + wind in the calibration band. Wind is *profiled out*, not jointly
  fitted (over a narrow band drag and wind are collinear and a joint fit walks the ridge).
  Deployment in the band is treated as binary (off, or hard on the ceiling), not as an average —
  that took CdA bias from −1.8% to +0.06%.
- **B — deployment reconstruction.** Savitzky-Golay smoothing plus its *analytic* derivative (never
  `np.diff` — differentiating a quantised noisy speed trace puts ~100 kW of noise on every sample),
  then a power balance, then the residual above what the engine can produce. Recovery under braking
  is cap-limited throughout, so recovered energy is just the cap times braking *duration*;
  thresholding each event at half its own peak deceleration recovers the true duration and took
  recovered-energy error from +45% to ~5%.
- **C — SoC continuity.** 400-particle filter, each particle carrying its own drag area, wind, mass,
  tow strength, deployment scale and driver buffer; store clipped at 0 and 4 MJ; systematic
  resampling once per lap. The informative event is a **deployment cut-out**. The band's width is
  **not a setting** — it is computed from the estimator's own Stage A residuals (a white term
  integrating as √N and a lap-to-lap drift term integrating as N), which is why it is correctly
  wider at low sample rates without anyone telling it to be.

### Decision engine
Backward induction over the remaining stint. Three things make it a strategy rather than
"attack always": the reward is the **share of the stint spent in front** (not a flat 1 for passing,
which saturates the value function); an attack costs energy **on top of** a normal lap; and a failed
lunge costs something real (the simulator drops a failed attacker 10 m, and the penalty 0.083 is
read off the pass model, not chosen).

Zone energy→speed maps are measured by running the actual vehicle model down each zone straight.
The counterintuitive result nobody typed in: **Zone A, the obvious overtaking place — 1,100 m into a
65 km/h hairpin — is the *worst* place to spend an energy advantage (1.7 m/s per MJ), because the car
is already near terminal speed. Zone C, a 240 km/h kink, gives 6.8 m/s per MJ.** When X-RAY says you
hold a big advantage, it sends you to Zone C.

## What is verified (I ran this locally, 2026-09-06)

- `pytest -q` → **26 passed in 195 s**. Exactly the 26 acceptance tests the README claims.
- `scripts/run_estimator.py --seed 42 --car LEADER` reproduces the headline numbers:
  per-lap deployed-energy MAPE **4.6% @ 100 Hz / 3.5% @ 3.7 Hz**; CdA error **0.99% mean, 1.77%
  worst**; recovered-energy error **6.3% @ 100 Hz vs 22.5% @ 3.7 Hz** (recovery is a
  braking-duration measurement, so it *is* rate-sensitive where deployment is not); raw-SoC coverage
  **0.09–0.42** vs deployable-energy coverage **0.94–0.97** — the reserve degeneracy, measured.
- Band width by regime at 3.7 Hz: **accel 0.665 MJ < braking 0.753 < corner 0.915** — tightest where
  the trace is informative, and asserted by a test so it cannot regress into decoration.

Two caveats I found rather than read:
1. `python scripts/run_estimator.py --seed 42` — the exact command in the README — **crashes with an
   unhandled `EstimatorError` traceback** before printing its score table. The cause is correct
   behaviour (the FOLLOWER is permanently in a tow, so its drag cannot be calibrated and the
   estimator refuses by design, which is a passing test) but the script does not catch it. Use
   `--car LEADER`.
2. Deployable-energy band coverage on seed 42 alone runs **0.94–0.97**, above both the README's
   stated 0.86–0.90 and the test's own 0.92 ceiling; the assertion passes because it averages over
   seeds 42/7/13. The coverage claim is seed-sensitive.

## Real-data validation is a null, and reported as one

There is no public ground-truth energy channel for a real car, so estimates cannot be checked
directly. What *can* be checked is a prediction the regulation makes: a car within 1.000 s of the car
ahead at the detection point becomes eligible for extra deployment (Manual Override, 0.5 MJ); a car
at 1.001 s does not. So: **regression discontinuity at the 1.000 s boundary.**

Over five races and 1,851 car-laps: **−0.20 MJ, SE 0.242, p = 0.41.** A null. The design is sound
(McCrary density shows no manipulation; position and lap number are balanced across the boundary),
but the **minimum detectable effect is 0.68 MJ against a 0.5 MJ allocation** — the study is
underpowered, so the null is a statement about how many races were analysed, not about the
regulation. Roughly twice the data would resolve it. The app reports the power calculation next to
the estimate every time, and a significance-vs-cutoff scan strip shows the scan is flat with one
spurious p = 0.04 at 1.30 s (thirteen tests, one false positive, exactly as expected).

Does it pick the right lap? `scripts/validate_decision.py` solves the same problem three ways on
seeded races where truth is known — oracle (given true energy), X-RAY (given only a noisy speed
trace), and blind. At full rate X-RAY picks **the oracle's exact lap in 100% of races**; at the real
4.17 Hz rate it is exact 40% of the time and **within one lap 100%** of the time, at nil
expected-value cost. X-RAY occasionally scores *above* the oracle on single seeds — that is
estimation error landing favourably, not skill, which is why it is reported over 25 seeds.

## The interface — six views

One zustand store holds **race time in seconds of session time**; the 3D scene, every overlay and
every chart subscribe to it, and nothing keeps its own timer. The 3D scene reads the clock
imperatively inside its own render loop; 2D overlays sample it at 12 Hz via `useThrottledTime`, so
React does not reconcile the Canvas subtree sixty times a second.

1. **Race theatre** — scrubbable replay on real circuit geometry (real `Z` channel, so Eau Rouge
   actually climbs), with the belief cloud beside the rival.
2. **Observability map** — the circuit coloured by what the estimator can learn where. La Source and
   the Bus Stop come out red; the Kemmel straight comes out green. Partial observability drawn as a
   picture of a racetrack.
3. **RDD explorer** — the falsification instrument. Drag the cutoff and hunt for an effect where the
   rules do not put one. The **only** endpoint that computes live.
4. **Decision explorer** — threshold, opportunity quality, and a fan of 200 re-solved opponent policies.
5. **Opponent fingerprint** — deployment style per driver, tightening as evidence accumulates. On
   Spa the two drivers separate: 75% aggression vs 45%, with a visibly larger held buffer on the second.
6. **Method & limits** — what the system cannot do, stated first.

Design discipline worth noting: spectacle only where the spectacle *is* the data (deployment painted
on real geometry, a particle filter rendered as actual particles). The statistical views are flat 2D
with no effects — their authority comes from looking as though they were not art-directed. The
signature visual is 400 points on a vertical energy axis beside the rival's car, contracting where
the trace is informative and dispersing where it is not. `?lite=1` drops postprocessing and cuts the
cloud to 120 particles. System font stack, all assets bundled, runs fully offline. Cars are extruded
geometry built in code — no liveried model, so no licensing question.

## Known limits (the authors state these; they are not hidden)

- **The pass-probability coefficients are invented.** `b1`/`b2` come from the design brief; `b0`/`b3`
  are solved to hit two shape anchors. Nothing in `overtake.py` is fitted to real data. Everything
  downstream of it inherits that.
- Real 2026 telemetry arrives irregularly at a **median 4.17 Hz**, and Stage 1's own ablation puts
  the accuracy cliff at **about 4 Hz**. There is no margin. The app says so on its front page.
- A single global ICE/MGU-K split spreads deployment evenly around the lap where a real car deploys
  in bursts out of slow corners. Consequence: the reconstructed store reads empty 14–45% of the race
  with a narrow 0.04–0.26 MJ band — more confidence than the method has earned, and the clearest
  remaining gap between the simulator result and the real-data one.
- Five races is not enough to power the RDD; roughly ten would be.
- The RDD running variable is the gap at the start/finish line, not at each Manual Override detection
  point. That adds noise and costs power.
- Car mass is a constant 790 kg on real data; fuel burn over a stint is real and unmodelled, which
  biases early-lap estimates.
- Aero-mode zones are hand-entered per circuit (the FIA publishes them as PDFs) and flagged as such.
- Stage 1 scope: two cars, no tyres, pit stops, safety cars or weather. Rolling resistance, air
  density and drivetrain efficiency are assumed known — only drag area and a wind offset are fitted.
- The estimator assumes the ICE runs wide open whenever accelerating and not corner-limited. True in
  the simulator; a real car lifts and coasts.
- The Stage 2 decision endpoint in `api/main.py` uses a closed-form heuristic threshold
  (`0.10 + 0.35*(k/n)^2`) and an analytic dv-per-MJ approximation rather than the full DP in
  `xray/decision.py`. Worth knowing before quoting real-data decision numbers.

## Repo hygiene notes

- `shot.mjs` (Playwright screenshot script) hardcodes `/Users/apple/Desktop/x-ray/out/shots` — the
  original author's macOS path. It will not write anywhere useful on Linux.
- `_decision_cached` in `api/main.py` is defined but never called; `decision()` calls
  `_decision_payload` directly and its local imports are unused.
- Git history is unusual: the newest commit is titled "Initial commit" sitting on top of nine
  substantive checkpoint commits, all dated 2026-08-30 to 2026-09-06.

## How to run it

Clean clone needs everything installed. Stage 1 only needs the light half of `requirements.txt`.

```bash
# Stage 1 (verified working)
python -m venv .venv && ./.venv/bin/pip install numpy scipy matplotlib PyYAML pytest
./.venv/bin/python -m pytest -q                                    # 26 tests, ~3 min
./.venv/bin/python scripts/run_sim.py --seed 42                    # physics and the race
./.venv/bin/python scripts/run_estimator.py --seed 42 --car LEADER # blindfold, estimate, score
./.venv/bin/python scripts/run_ablation.py                         # MAPE vs sample rate
./.venv/bin/python scripts/run_decision_eval.py --mode fast        # blind vs X-RAY, 50 stints
./.venv/bin/python scripts/validate_decision.py                    # oracle vs X-RAY vs blind
./.venv/bin/python scripts/make_summary.py --seed 42               # -> out/xray_summary.png

# Stage 2 (needs fastf1, fastapi, uvicorn, pandas, pyarrow; downloads race data on first run)
pip install -r requirements.txt
python scripts/feasibility.py --round 10 && python scripts/feasibility_report.py
python scripts/analyse_race.py --round 10                          # -> out/races/*.json
python -m uvicorn api.main:app --port 8011
cd app && npm install && npm run build                             # API then serves it at /
# open http://127.0.0.1:8011/   (add ?lite=1 for a weak GPU)
```

Seed 9 gives the cleanest single race (one decisive pass, lap 4); seed 42 is the eventful one used
throughout the docs.

**Current local state:** `.venv/` exists with the Stage 1 dependencies installed and the test suite
passing. `out/`, `app/node_modules/` and `app/dist/` do not exist — Stage 2 has never been run on
this machine, so there are no race JSONs and the web app has not been built. Anything touching the
six views or the FastAPI endpoints requires the Stage 2 install plus at least one
`analyse_race.py` run first (which downloads real session telemetry through FastF1).

## Where outside help is actually worth having

Ranked by how much the project's credibility rests on it:

1. **Fit `overtake.py` on real 2026 race data.** It is the one place where an invented number
   propagates into every headline decision claim.
2. **Analyse ~5 more races to power the RDD**, and move the running variable to the actual Manual
   Override detection points instead of the start/finish line.
3. **Replace the global ICE/MGU-K split with a per-corner or per-sector one**, which is what would
   stop the real-data store from reading empty 14–45% of the race with an unearned narrow band.
4. **Frontend/QA on the six views** — none of it has been built or exercised on this machine. Needs
   `npm install && npm run build`, a race JSON, and a browser pass over playback, scrubbing, the
   3D scene on a weak GPU (`?lite=1`), and the live RDD cutoff drag.
5. **Wire the real-data decision endpoint to the actual DP** in `xray/decision.py`.
6. Small cleanups: catch `EstimatorError` in `scripts/run_estimator.py`, fix `shot.mjs`'s hardcoded
   macOS path, drop the dead `_decision_cached`.

## House style, if you contribute

The codebase has a distinctive and consistent voice, and matching it matters more than usual here.
Comments explain *why a wrong version was wrong*, with the measured consequence attached — "an
earlier version inferred ICE output from the throttle trace and produced lower bounds of CdA > 13,
because a real car at 30 m/s is traction-limited rather than power-limited"; "the spring
interpolation used `cur[i] || target`, which treats a legitimate 0.0 as unset, so every particle
stayed pinned at the world origin". Targets that were missed are argued rather than tuned. Bugs are
recorded, not erased. Limitations lead. Numbers in prose are always numbers the pipeline computed.
Match that, or the seams will show.
