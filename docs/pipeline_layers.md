# Pipeline layers: what is simulation, what is inference, what is decision, what is real

Every file in `xray/` belongs to exactly one layer. This document says which, and says
where the code does not yet match.

It exists because of one claim that turned out to be false: *"what we tune on synthetic is
what real runs."* It is not. There are **three separate inference implementations**, only
one of them reaches a real race, and it is the one that has never been scored against
ground truth. Several parameters currently being tuned cannot affect a real result at all,
because the code containing them does not execute on real data.

Verified against HEAD `2f54d64`.

---

## 1. The layers

| Layer | Owns | Rule |
|---|---|---|
| **REGULATION** | the 2026 rulebook as numbers | One source. Never restated in another module. |
| **SHARED PHYSICS** | forces, energy integration, tyres, air | Identical for both data sources. No source-specific branch. |
| **SIMULATION** | generating synthetic ground truth, including hidden energy | May read `config/default.yaml`. **Never imported by an inference module.** |
| **SOURCE ADAPTER** | one data origin → the common feed struct | The *only* layer allowed to know where data came from. |
| **INFERENCE** | feed → belief about hidden energy | **Must be source-agnostic.** This is the layer that should be one body of code. |
| **DECISION** | belief → the call | Consumes belief, never ground truth. |
| **SCORING** | comparing a belief to truth | Ground truth allowed here and nowhere else. |
| **PRESENTATION** | API, frontend, video | Reads results. Computes no physics. |

The four names asked for map onto this as: *simulation* = SIMULATION; *synthetic* =
INFERENCE running on simulator-sourced data; *real* = INFERENCE running on FastF1-sourced
data; *decision* = DECISION. **"Synthetic" and "real" are not supposed to be different
files.** They are supposed to be the same INFERENCE layer behind two SOURCE ADAPTERs.

---

## 2. File table

`target` differs from `current` only where the code is in the wrong place today.

| File | Current layer | Target | Note |
|---|---|---|---|
| `constants.py` | REGULATION | same | 2026 power unit, taper curve, store size |
| `regs.py` | REGULATION | same | Date-dispatched variants. **Reached only by the set-membership group, i.e. only by tests** — see §4 |
| `vehicle.py` | SHARED PHYSICS | same | The single authoritative force/energy integrator |
| `tyres.py` | SHARED PHYSICS | same | Used by simulation and decision |
| `environment.py` | SHARED PHYSICS | same | Weather, air density, wind projection |
| `physics_context.py` | SHARED PHYSICS | same | Environment + optional tyre bundle |
| `sim.py` | SIMULATION | same | Two-car simulator → `GroundTruth` |
| `track.py` | SIMULATION | same | Circuit Sigma |
| `policy.py` | SIMULATION | same | The driver model |
| `config.py` | SIMULATION | same | "The estimator never touches this module" (`config.py:1`) |
| `observe.py` | SOURCE ADAPTER (synthetic) | same | `GroundTruth` → `Observation`, plus `public_channels` for the published throttle/brake the real feed also carries |
| `data/ingest.py` | SOURCE ADAPTER (**both**) | same | The one gridding module. `grid_lap` is shared; `frame_from_fastf1` and `frame_from_observation` are the two thin entry points |
| `data/circuits.py` | SOURCE ADAPTER (real) | same | Real centreline, elevation, curvature, zones |
| `estimator.py` | INFERENCE (synthetic only) | INFERENCE | Locked to `Observation` by its import at `:30` |
| `realfit.py` | INFERENCE (real only) | INFERENCE | The only inference code that reaches a real race |
| `analysis.py` | INFERENCE (real orchestration) | same | Session → `out/races/<id>.json` |
| `balance.py` `setmem.py` `modes.py` `pipeline.py` `rbpf.py` `deadband.py` `strategy.py` `pooling.py` | INFERENCE (**tests only**) | undecided | See §4 |
| `decision.py` | DECISION | same | Zone calibration, DP solvers |
| `overtake.py` | DECISION | same | `p_pass`. Coefficients synthetic — see §8 |
| `qmdp.py` | DECISION | same | Deciding under the belief |
| `opportunity.py` | DECISION | same | Opportunity-horizon solver, wired into the service |
| `stint.py` | DECISION | same | Pit context, causality gate |
| `decision_service.py` | DECISION | same | Canonical web/core entry point |
| `passmodel.py` | DECISION (calibration) | same | Real overtake labels. **Not wired** — see §8 |
| `metrics.py` | SCORING | same | The only module allowed to read ground truth |
| `closedloop.py` | SCORING | same | Closed-loop policy evaluation. Synthetic only — see §7 |
| `render/` | PRESENTATION | same | |
| `simulation/api/`, `simulation/app/` | PRESENTATION | same | |

---

## 3. The layer rules, and what enforces them

`tests/test_estimator.py::test_estimator_is_blind` parses `estimator.py` with `ast` and
fails on any simulator import. That was the right mechanism applied to exactly one file;
two rules this document depends on had nothing behind them. Both now do, in
`tests/test_layers.py`:

- **`test_no_inference_module_imports_the_simulator`** — the same `ast` scan across all
  eleven INFERENCE modules, including `realfit.py`, the only one a real race executes. All
  eleven are currently clean, so it is a guard rather than a to-do list.
- **`test_registered_constants_still_hold_their_registered_values`** and
  **`test_fuel_burn_defaults_are_the_registered_ones`** — a registry of the divergences in
  §5. It does not fix them; it stops them growing and makes removing one a deliberate act.
  An entry there is an admission, not an approval.
- **`test_config_wind_prior_sigma_matches_the_code`** — closes the second instance of the
  dead-config-key failure (see §5).

Every one of them reads source with `encoding="utf-8"`. Omitting it has broken this suite
three times (`docs/dev_readme.md` §7 item 1).

---

## 4. Three inference implementations, not one

| Stack | Modules | Reached by | Input struct |
|---|---|---|---|
| Stage 1 | `estimator.py` | scripts `02`, `03.a`, `03.b`, `04`, `05`, `99.make_golden`, `render/video.py`, `closedloop.py` | `observe.Observation` |
| Set-membership group | `regs`, `balance`, `setmem`, `modes`, `pipeline`, `rbpf`, `deadband`, `strategy`, `pooling`, `qmdp` | **`tests/` only** | `pipeline.Feed` |
| Real | `realfit.py`, `analysis.py` | scripts `07.b`, `08.b`, the API, `decision_service` | `realfit.Kin` |

How this was established: `analysis.py:14-17` imports only `constants`, `data.circuits`,
`data.ingest`, `realfit`. `realfit.py:37` imports only `.constants`. Grepping every import
across `xray/`, `scripts/`, `simulation/`: no module outside the set-membership group
imports any of it.

Two consequences, and they point in opposite directions.

**Knobs in the set-membership group cannot affect a real race.**
`balance.N_SIGMA_DEFAULT`, `setmem.OUTLIER_FRAC`, `rbpf.CLOSURE_SIGMA_J`,
`rbpf.THETA_JITTER_FRAC`, `deadband.BIC_MARGIN` and the rest are measured only by the test
suite. Tuning them changes no shipped output. Equally, the headline metrics they produce —
deployable coverage 0.67/0.72/0.80, CdA_X 0.597/0.571/0.494 — describe code that ships to
nobody. The keep-or-retire decision for this group is **open**; what is not open is that
its numbers must not be quoted as though they describe the product.

**`realfit.py` had never been scored against ground truth.** It is the only inference code
a real race touches, and no number existed for how accurately it recovers hidden energy,
because ground truth lives only in the simulator and nothing bridged the two.
`ingest.frame_from_observation` now bridges it: the simulator's blinded trace runs the same
gridding, the same `build_kin`, and the same inference as a real session.

That bridge immediately paid for itself and immediately drew blood — see §11, where the
first thing it found was a real-looking bug whose "fix" cost the shipping path five
identifiable cars.

---

## 5. Divergence register: the same quantity, different values

| Quantity | Stage 1 | Set-membership group | Real |
|---|---|---|---|
| Reference mass | `768.0` (`estimator.py:87`) | `790.0` (`pipeline.py:68`, `M_REF_KG`) | `790.0` (`analysis.py:73`) |
| Fuel burn per lap | `1.4` kg (`constants.FUEL_BURN_PER_LAP_KG`) | `1.15` kg (`balance.py:135`, `rbpf.py:227`, `deadband.py:127`) | **not modelled** |
| Cut-out reserve sigma | `1.0e5` J (`estimator.py:49`) | `5.0e5` J (`rbpf.py:40`) | `5.0e5` J (`realfit.py:40`) |
| Regulation source | static `constants` | `regs.RegSet`, date-dispatched | static `constants` |

**Mass is constant for a whole race on the real path, and it is measurably too low.**
`realfit.build_kin` holds mass fixed while rolling resistance and the inertial term in
`_terms` both scale with it. Measured on a simulator trace with CdA pinned to truth, so
this isolates mass from the identification problem — per-lap deployed-energy error:

| mass model | MAPE | bias |
|---|---|---|
| constant 790 kg (what ships) | 8.2% | −8.2% |
| constant 768 kg (dry minimum) | 8.1% | −8.1% |
| constant 838 kg (dry + full fuel) | 11.5% | +8.7% |
| constant 821 kg (dry + end-of-race fuel) | **5.4%** | **−1.8%** |
| fuel model, simulator's *true* 70 → 53.2 kg | 6.6% | +3.6% |
| fuel model, ASSUMED 100 → 0 kg | 9.7% | −2.5% |

So mass is worth about 3 points of MAPE and most of the bias, and 790 kg is too low — it
sits under the 768 kg dry minimum plus any realistic fuel load, which makes it an
end-of-race figure applied to a whole race. The one-sided bias follows: too little mass
understates the work done, so deployment comes out low.

**A fuel model is not the fix**, which is the surprise. A linear burn is more physical than
a constant and still loses, even handed the simulator's own exact fuel numbers. And the
winning constant does not transplant: 821 kg is 768 plus the *simulator's* end-of-race
fuel, off its own 70 kg start and 1.4 kg/lap burn, neither of which is a real 2026 figure.

`analysis.py` therefore still passes 790. What has changed is that the cost is on record
and the obvious repair has been ruled out. Settling it for real data needs a real fuel
load, and no public channel publishes one — FastF1 has no fuel signal at all.
`realfit.mass_series` carries the measurement and `build_kin` now accepts a per-sample mass
array, so the experiment is repeatable without re-deriving it.

**`RESERVE_SIGMA` is the clearest case of the problem this document is about.** Its comment
in `estimator.py:49-61` records it being moved `2.5e5 → 1.0e5` to shift two
`test_band_coverage` criteria — a synthetic coverage target. Real independently uses 5×
that value. Synthetic cannot say which is right for real data, because real data has no
`E_true`.

**Dead config key — FIXED.** `config/default.yaml` set `cda_prior_wind_sigma: 3.0` while
`estimator.fit_nuisance` defaulted `wind_prior_sigma = 1.0`, and nothing read the yaml key,
so 1.0 was what ran and 3.0 was decoration. The same failure is already documented one
block below for `dry_event_e_scale` ("It was edited alone once") and pinned by
`test_config_dry_event_scale_matches_the_code`; this second instance was unpinned.

Corrected on the **yaml** side, to the value that actually runs. Changing the code to 3.0
would have moved the estimator and invalidated the golden baseline in order to make a
stale config file right. Now pinned by `test_config_wind_prior_sigma_matches_the_code`.

---

## 6. The regulation variant did not reach the real path — FIXED

`regs.py:125-141`:

| | `p_harv_max` | `p_dep_max_elsewhere` |
|---|---|---|
| `PRE_MIAMI` | 250 kW | 350 kW |
| `POST_MIAMI` | 350 kW | 250 kW |

**The bug.** `regs_for(race_date)` was called at exactly one site, `pipeline.py:72` — inside
the group that only tests reach. `realfit.py` took `P_MGUK_MAX` from `constants` and used
it unconditionally for the harvest side of the CdA bracket and for harvest reconstruction,
which feeds store closure. So **every real race was analysed at a 350 kW harvest cap.** For
a pre-Miami round — 1 Melbourne, 2 Shanghai, 3 Suzuka — the true cap is 250 kW.
250/350 = 0.71, the same ratio and the same cause as the simulator/fixture mismatch that
invariant 8 exists to prevent. Round 10, the only race analysed at the time, is post-Miami
and therefore happened to be correct.

**The fix.** `analysis.analyse` resolves `regs_for(sd.date)` — the date was already in
scope — and threads a `RegSet` into `fit_nuisance_real` and `deployment_trace`.
`realfit._p_harv_max` is the one place the cap is read. The variant is now published in the
race artefact under `regulation`, so a consumer can see which rulebook produced the
numbers instead of assuming one.

**Verified both directions.** Post-Miami round 10 is unchanged: 0 per-car `cda_hat`
differences, identical pooled CdA, identical `deployed_lap`, identical refusals — correct,
because it was already on the 350 kW curve. Pre-Miami round 1 moves, and not uniformly:

| driver | identified set width, 350 kW | at the correct 250 kW |
|---|---|---|
| VER | 1.224 | 0.377 (identifiability 0.000 → 0.148) |
| LEC | 1.180 | 0.530 |
| NOR | 0.184 | 0.859 (identifiability 0.670 → 0.000) |

Two tighten sharply and one loosens, because the wind scan re-optimises onto a different
solution once the bounds move. **The justification is correctness, not the metric.** On
2026-03-08 the cap was 250 kW, so the old numbers came from the wrong rulebook however
good they looked — and a narrower interval computed from a rule that was not in force is
worse than a wide one that was.

Still open: the *deployment* ceiling. `p_mguk_ceiling(v)` remains the unconditional taper,
because `POST_MIAMI` caps deployment outside a zone at 250 kW and `Kin` carries no zone
mask. `regs.p_dep_max(v_ms, in_zone, regs)` is the call to make once `RealTrack` supplies
one.

---

## 7. Which belief producer each evaluation harness uses

This matters more than it looks, because it decides what an evaluation number means.

| Harness | Belief from | Describes |
|---|---|---|
| `scripts/02`, `05` (`metrics.score_estimate`) | `estimator.py` | Stage 1 |
| `scripts/03.a`, `03.b`, `04` | `estimator.py` | Stage 1 |
| `closedloop.py` / `scripts/10` | `estimator.py` (`closedloop.py:43`) | Stage 1 |
| A real race (`08.b` → API → `decision_service`) | `realfit.belief_from_deployment` | the shipping path |

Every evaluation in the repo scores the Stage 1 belief. Every real race uses a different
one. Closed-loop P2 results therefore do not characterise real decision quality — the two
paths have different error characteristics and nothing has measured the difference.

`closedloop.py` itself is built correctly otherwise: the belief comes from `observe()` →
`estimate()` and nothing else, `oracle_p2_plan` is labelled `oracle` in its own source
field and never shares a row with an X-RAY result, seeds are paired so a difference in
outcome is strategy rather than luck, and an `EstimatorError` is recorded as a `REFUSED`
row instead of a number.

**Its world randomisation is the right mechanism on the wrong axes.** `WORLD_VARIATION`
(`closedloop.py:397-402`) varies `sim.start_gap_s`, `sim.e_start_frac`,
`sim.leader_policy` and `observe.speed_noise_ms`. All four are synthetic *scenario*
parameters; none is a real-world nuisance. `closedloop.py:141` reads
`cfg["observe"]["rate_hz"]` but `world()` never varies it, so every world runs at a clean
regular 3.7 Hz — while real telemetry is 4.17 Hz median and irregular with p90 gaps of
0.36–0.40 s, sitting on the accuracy cliff the rate ablation found near 4 Hz. The axis most
likely to break transfer is the one held fixed.

The comment at `closedloop.py:393-396` already states the governing rule and it is the
right one: "a randomisation chosen to make P2 look good is not a randomisation, it is a
fixture." Any new axis must have its range measured and cited from feasibility output.

---

## 8. Which pass model reaches a decision

**The synthetic one.** `overtake.COEFFS` — `b0 = -7.6650`, `b1 = 0.55`, `b2 = 2.10`,
`b3 = 2.4324`, with `b1`/`b2` from the design brief and `b0`/`b3` solved from two shape
anchors. `opportunity.py` states in its own docstring that every pass probability comes
from `xray.overtake.p_pass`, and `decision_service.py` caps decision confidence at `0.35`
because of it.

`passmodel.py` builds a *real* labelled overtake dataset from race payloads, excludes pit
cycles via tyre-life resets and compound changes, and reports safety-car contamination as
a bound rather than ignoring it (there is no track-status channel in the payload). Its
`fit()` refuses to run until `audit_dataset` passes.

It is not wired into the decision path, and **that is correct, not an oversight**:
`MIN_RACE_GROUPS = 4`, `MIN_USABLE_SAMPLES = 300`, `MIN_POSITIVES = 40`, and exactly one
race artefact exists (`out/races/2026_r10_R.json`). The audit cannot pass yet. Wiring it
early would be the failure mode the audit was written to prevent.

---

## 9. Feed fidelity: the synthetic feed is easier than the real one

The set-membership group's synthetic `Feed` is built in `tests/test_pipeline.py:24-30`.

| Channel | Synthetic `Feed` | Real feed (`ingest.py:163-169`) |
|---|---|---|
| `throttle` | `where(regime=="brake", 0, 1)` — noiseless binary, taken from the simulator's `regime` field | FastF1 `Throttle`, 0–100 %, ~4 Hz, interpolated onto a 10 m grid |
| `brake` | `(regime=="brake")` — same ground-truth label | FastF1 `Brake` |
| `rpm` | `full(n, N_RPM_MAX)` — constant at redline | FastF1 `RPM`, varies; `regs.ice_work_from_fuel` is RPM-dependent |
| `z` | exact `z(s)` from the track model | position data / 10, smoothed; `has_elevation` sometimes `False` |
| `in_zone` | `ones(n, bool)` — everywhere | hand-entered per circuit |
| sampling | regular 3.7 Hz | 4.17 Hz median, irregular, holes > 1.0 s blanked |

`throttle` and `brake` here come from a channel `observe.py` explicitly refuses to pass:
"There is no energy, throttle, brake, gear or deployment field here, and there never will
be."

In the other direction, Stage 1 is solving a **harder** problem than the real pipeline
faces. `observe.Observation` carries `t`, `s`, `v`, `lap`, `gap_to_leader` — speed is
annotated "THE ONLY SIGNAL" — while FastF1 supplies throttle, brake, gear and RPM as well.
Information content differs both ways, which is a third reason a knob tuned on one does not
transfer to the other.

---

## 10. Which knobs actually reach a real race

Only these. They are also the least test-covered code in the repo, and they are where
tuning effort belongs.

| Knob | Where | Controls |
|---|---|---|
| `RESERVE_SIGMA_REAL` | `realfit.py:40` | how tightly a cut-out pins the store to the buffer |
| `COAST_THROTTLE` | `realfit.py:41` | throttle % below which the ICE is treated as off |
| `COAST_MIN_DECEL` / `COAST_MAX_DECEL` | `realfit.py:42-43` | what counts as a real coast-down |
| `BRAKE_DECEL` | `realfit.py:44` | brake detection |
| `smooth_m` | `realfit.build_kin:114` | Savitzky-Golay window, in metres |
| wind grid, robust quantiles | `realfit.fit_nuisance_real` | the CdA interval intersection |
| `n_particles` | `analysis.py:73` | store filter resolution |
| `MIN_IDENT` | `analysis.py:22` | refuse-or-report boundary |
| `identifiability >= 0.25`, `n_binding >= 50` | `realfit.RealNuisanceFit.usable:72-74` | whether a car is reported at all |

Everything in the set-membership group, and everything in `estimator.py`, is outside this
list.

---

## 11. Worked example: a synthetic finding that damaged the real product

This happened while writing this document, and it is the clearest case for everything above.

Running `realfit` against simulator data for the first time produced an absurd result: a
`CdA >= 13.2 m^2` lower bound and an empty interval intersection. The cause looked
structural. `build_kin` smooths `a` with a Savitzky-Golay derivative over a 60 m window but
passes the brake channel through unsmoothed, so the window's leading edge reports braking
deceleration on samples whose brake flag still reads 0, and `interval_bounds` then has to
explain that deceleration with drag alone. The evidence was strong: 153 grid samples
not-braking with `a < -10 m/s^2`, worst `-44.3` against a true non-brake floor of `-24.7`,
and **100% of them within 3 cells of a braking cell against a +/-3 cell half-window**.

Dilating the brake mask by the half-window fixed it. The identified set went from
`[-2.19, 2.15]` to `[-0.67, 0.62]`, and the full test suite was unchanged at
6 failed / 322 passed.

**On real telemetry the same change was a straight loss:**

| | before | after |
|---|---|---|
| identifiable cars (of 21) | 7 | 2 |
| mean identifiability | 0.153 | 0.054 |
| median `n_binding` | 4531 | 1475 |
| pooled CdA | available | refused, "only 2 identifiable cars" |

Because the pathology is not a real one. Measured on three cars at Spa: samples that are
not-braking with `a < -10 m/s^2` are **0.000-0.036%** of the trace, and the non-braking
floor is `-9.9` to `-13.5 m/s^2` -- lift-off and downshift, not a contradiction. The
simulator's `vehicle._regime` switches braking on as a *step* to `brake_decel_max`, and
smoothing a derivative across a step manufactures the contradiction. Real braking ramps.

The change was reverted. Four things this demonstrates, in order of how much they cost:

1. **The test suite did not catch it.** All 322 passing tests stayed passing, because no
   test scores `realfit` on real data — §4's "never scored against ground truth" cuts both
   ways.
2. **A fix validated on synthetic can be negative on real**, even when the synthetic
   evidence is quantitative, reproducible and points at a genuine asymmetry in the code.
3. **Shared code amplifies this.** The change was correct for the caller it was tested
   against and wrong for the caller that ships, and one file served both.
4. **The artefact belonged to the simulator, and so does the fix.** Repairing the shipping
   path to accommodate a simulator limitation is the wrong direction every time.

The comment at `realfit.build_kin` records this so the reconciliation is not re-attempted
on the same synthetic evidence.

## 12. The architecture, as it now stands

```
FastF1 session ──► ingest.frame_from_fastf1 ──────┐
                                                  ├──► ingest.grid_lap ──► realfit ──► belief ──► decision_service
GroundTruth ──► observe ──► frame_from_observation ┘
```

Not two modules calling a shared helper — **the same files**. From `grid_lap` onward every
stage is one implementation with no branch on origin: one gridding step, one `build_kin`,
one `fit_nuisance_real`, one `deployment_trace`, one `belief_from_deployment`.
Source-specific code survives only at the two entry points, which is unavoidable, and both
live in the modules that already own that job.

`realfit.Kin` is the common struct rather than `pipeline.Feed`, for three reasons: it is
what the shipping path already uses; it is on a distance grid, which survives irregular
sampling where a time grid does not; and it already carries `w_along_mps`, `rho_series`,
`wind_source` and `rho_source`, so provenance travels with the data.

`xray/simfeed.py` existed briefly as a separate synthetic adapter and was deleted. It was a
reimplementation of `_lap_frame` rather than a reuse of it, and the copy drifted within a
day — its `usable` column meant per-cell where the real one means per-lap (a difference
`analysis.py` acts on), and it had no hole handling at all. §11 is what the drift cost.

**Proof it is behaviour-neutral on the shipping path:** `08.b_analyse_race.py --round 10`
produces a byte-identical artefact before and after the unification —
`ad0fbbcd3559822623911a917ac1cac6`. Same code, one fewer hop.

---

## 13. Rules

1. An INFERENCE module may not import a SIMULATION module. Enforced for the whole layer by
   `tests/test_layers.py::test_no_inference_module_imports_the_simulator`.
2. A physical quantity has one definition. If two modules need different values, the
   difference is documented with the measurement that justifies it, or it is a bug.
3. The SOURCE ADAPTER layer is the only place allowed to know where data came from.
4. Ground truth appears in SCORING and nowhere else.
5. Quote a metric with the stack that produced it. Stage 1 and the set-membership group are
   different baselines and have been quoted interchangeably before.
6. A knob is tuned against the stack that ships. Tuning one the real path never executes
   changes nothing real.
7. Any source-file-scanning test passes `encoding="utf-8"` — that omission has bitten three
   times (`docs/dev_readme.md` §7 item 1).

---

## 14. P3 final state: one canonical path, and an honest verdict on it

§4 opened with the finding that there were **three inference implementations, only one of
which reached a real race, and it was the one never scored against ground truth.** P3
closed the first half of that and measured the second. Both results are recorded here
because the second one is negative.

### The canonical path, after P3

```
SIM  ──► observe()  (blinded public feed) ─┐
                                           ├──►  ingest.grid_lap  ──►  build_kin
FASTF1 ──► data/ingest.frame_from_fastf1 ──┘                              │
                                                                         ▼
                                                                      realfit
                                                                         │
                                                              canonical inference
                                                                         │
                                                                         ▼
                                                                 decision_service
                                                                         │
                                                                    P1  /  P2

SIM hidden truth ───────────────────────────────────────────────► SCORING only
```

The simulator's hidden energy reaches the scoring layer and nothing else. That is what
makes the synthetic result in §14.2 a *direct* validation rather than a circular one.

### 14.1 What is still RESEARCH_ONLY

| Component | Status | Why it is not production |
|---|---|---|
| `estimator.py` | `RESEARCH_ONLY` | locked to the synthetic `Observation` type by its import; never executes on a real race |
| `balance.py` `setmem.py` `modes.py` `pipeline.py` `rbpf.py` `deadband.py` `strategy.py` `pooling.py` | `RESEARCH_ONLY` | reached by tests only; not on the shipping post-ingestion path |

Their metrics are **not evidence about real-race inference**, and §13 rule 5 applies: quote
a metric with the stack that produced it. `xray/registry.py` carries these statuses so the
API and UI cannot quietly promote them.

### 14.2 Direct synthetic validation — mixed

Canonical shipping inference, scored against hidden simulator truth (mean over three
seeds, `out/p3/part1/*/summary.json`):

| Quantity | Result | Reading |
|---|---|---|
| deployment lap MAPE | 5.71% | flows are measured well |
| deployment bias | −5.32% | small, consistent under-read |
| deployment band containment | 93.4% | the band does its job |
| SOC mean MAE | 0.180 MJ | level is useful |
| SOC band containment | 43.8% (35.7–55.0% across seeds) | **uncertainty is not calibrated** |
| CdA truth containment | 3/3 | the interval contains the truth |
| CdA interval width | 658.6% of truth | **CdA is not identified on this path** |
| CdA identifiability | 0.0 | — |

The CdA rows are a containment result, not an accuracy one — and barely even that: the
interval spans **negative drag area** ([−2.19, +2.16], centre −0.018 on the first seed),
which is physically impossible, so an interval that wide would contain almost any truth. The cause is in the coverage
map: **the simulator has no genuine lift-and-coast regime**, so the dead-band drag channel
that identifies CdA on real data is never exercised. That is a synthetic-coverage gap, not
an estimator bug, and it is why a better synthetic CdA number would not have meant much.

### 14.3 Indirect real validation — no robust improvement

48,920 strict-future held-out examples across Australian, Monaco, British, Belgian and
Dutch GPs, leave-one-race-out (`out/p3/part2/*/summary.json`). Held-out MAE, lower better:

| Target | X-RAY | fixed-E | neutral-E | X-RAY − fixed |
|---|---|---|---|---|
| future speed 5 s | 22.39 | 21.95 | 21.05 | **+0.437** |
| straight speed 3 s | 5.90 | 5.62 | 5.68 | **+0.279** |
| braking-point speed | 15.78 | 16.11 | 15.12 | −0.325 |

**A positive difference is X-RAY losing.** It loses on two of three targets and wins on
one, and the energy-neutral baseline is the best of the three on two targets. The
authoritative conclusion:

> The current canonical inferred-energy state does **not** add robust predictive value over
> the tested baselines on held-out real F1 data.

Reporting only braking-point speed would be cherry-picking, and tuning until the sign
flips would be fitting the held-out set. Neither is allowed.

### 14.4 Why this is a P3 *success*

P3's deliverable was never "make the estimator work". It was to build a path on which that
question can be asked and answered. Before P3 the question was unanswerable: the code that
ran on real races had never been scored, and the code that had been scored never ran on a
real race. The negative result is the first real measurement the project has of its central
claim, and it is preserved rather than papered over — in the registry, the API, the UI and
this document.

### 14.5 What this does not license

Latent energy is **not** uniquely identified. Exact re-inference sensitivity could not be
measured, because the persisted race payloads do not keep the raw canonical throttle/brake
inputs `realfit` needs to rerun. Downstream prediction at *fixed* inferred energy moves
0.15–0.81 m/s under small nuisance perturbations, 1.38–2.95 m/s under moderate ones, and
6.96–11.85 m/s in the fragile cases. Predictions can therefore be materially sensitive to
physical assumptions that are not themselves calibrated.

### 14.6 The deployment-zone ceiling (P3 Step 1)

P2 sized its deployment budget from `max(ZoneModel.energy_grid)`. That axis is swept by
`calibrate_zone` over **run-up plus straight** — 1700 m from the previous corner exit at
200 km/h on Circuit Sigma zone A — so it offered budgets up to 1.602 MJ for a zone that can
execute about 0.45 in race conditions. Three distinct causes, all now documented in
`tests/test_deployment_budget.py`:

1. **Window.** The attack spans the 1100 m straight, not the 1700 m calibration run.
2. **Entry speed.** The car arrives at 314 km/h and crosses 345 km/h after 387 m, beyond
   which the normal MGU-K curve is exactly 0 kW. Zone A absorbs 1.603 MJ entered at
   200 km/h and 0.448 MJ entered at 314 km/h — a per-zone constant cannot say both.
3. **Accounting (a bug).** `closedloop.BudgetedAttack` charged the budget against store
   *drawdown*. A deployment straight ends in a braking zone, so the store refills while the
   attack is still running: drawdown runs 0 → 0.4281 → 0.4492 → 0.4492 → **−0.1353** MJ,
   peaking mid-straight and ending negative, while delivered energy rises monotonically to
   0.4564 MJ. A budget compared against a non-monotone, eventually-negative quantity never
   binds, and the "0.449 MJ executable" figure in the P2 audit was the peak of that curve
   rather than a deployment measurement.

The ceiling is now **measured through the shared integrator** per zone per entry speed
(`decision.executable_attack_ceiling`), not integrated in closed form — integrating the
taper over the window gives 0.981 MJ against 0.448 delivered, because it counts the braking
stretch and ignores that deploying harder raises `v` and lowers the next step's ceiling.
Budget accounting now uses delivered MGU-K energy, fed back from the integrator via
`DeploymentPolicy.note_deployed`.

Fixing the ceiling also exposed a coupling bug: `evaluate_action` clipped the *rival's*
energy with *our* deployment ceiling, so our arrival speed moved the rival's predicted
speed. Tightening zone A from 1.422 to 0.354 MJ moved HOLD's `delta_v` from −10.55 to
−2.31 m/s — on an action that deploys nothing. Each car now carries its own measured
ceiling from its own observed entry speed.
