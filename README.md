# ⚡ X-RAY — See the Other Car's Battery

**Team: Power Buff Girls**

> *Once we can see the other car's battery, the game changes.*

---

## 2026: The Battery Decides the Race

The 2026 Formula 1 regulations make the MGU-K half the powertrain:

| Fact | Number |
|---|---|
| MGU-K peak power | **350 kW** |
| Share of total power that is electric | **~50%** |
| Override boost, deployable inside 1.0 s | **0.5 MJ** |

Every overtake is decided in this energy window.

The asymmetry is brutal:

```
YOU     ████████████████░░░   2.6 MJ   ← exact, from your own telemetry
RIVAL   ░░░░░░░░░░░░░░░░░░░     ?      ← invisible — no public channel exists
```

**Teams see their own battery perfectly. Nobody can see their rival's.**

X-RAY is the attempt to close that gap using only what is legally observable.

---

## The Rulebook Is Our Natural Experiment

The 2026 MGU-K deployment ceiling tapers with speed — not as a performance choice, but as a regulation. That taper is our signal.

```
MGU-K ceiling (kW)
 350 ─────────────┐
                   \
                    \
                     \
                      └──────── 0
       ← full ─────── taper ─── dead →
     <290 km/h              >340 km/h
```

| Speed window | What the regulation does | What we learn |
|---|---|---|
| **> 340 km/h** | Deployment ≈ 0 by rule | Calibrate drag, wind, air, tow — nuisances only, no energy noise |
| **< 290 km/h** | Channel fully open | Read deployment directly — ≈47% of measured power |

**The taper gates the signal by speed while nuisances stay on.**  
Identify nuisances up high, read the battery down low.

This is not a clever trick. It is what the regulation enforces on every car, every lap, whether or not anyone is watching.

---

## How We Attack It

### 1 · Energy, Not Power

We estimate per-lap **energy** (joules) and state-of-charge, not instantaneous power (watts). Continuity of the energy balance regularises the estimate — noise that spikes at one instant averages out over an entire straight. A 3.7 Hz speed sample is enough.

### 2 · Cross-Car Differencing

Wind, air density, and track state hit every car the same way at the same track position. We take each car's deviation from the **field's median** at each position. Shared nuisances cancel. What remains is the car-specific signal: deployment strategy.

### 3 · Two-Tier Simulation

- **F1 25 UDP** — the game's physics engine as sensor ground truth; gives us true store energy to validate against
- **Our own two-car simulator** — policy lab, thousands of races per second; the estimator never sees its output (enforced by test)

### 4 · We Report What's Identifiable

A speed trace fixes energy **flows** exactly; it fixes the absolute level only to a constant. We do not invent the constant. We report **deployable energy** — the usable margin above the driver's personal reserve — with a bias of **0.05 MJ**. The width of the identified set is the identifiability score.

---

## What's Built

- ✅ **Two-car 2026 energy simulator** — deterministic, energy balance closes
- ✅ **Virtual sensor** ("the blindfold") — blind to ground truth by design, enforced by automated test; fails the build if the estimator ever imports the simulator
- ✅ **Particle-filter belief over deployable energy** — 400 particles, Rao-Blackwellised; discrete latents (aero mode, powertrain regime) sampled, continuous store carried in closed form
- ✅ **Optimal-stopping decision engine** — finite-horizon Bellman recursion; outputs ATTACK / HOLD with expected regret
- ✅ **Ablation across 100 → 1 Hz** — validates that 3.7 Hz public telemetry is sufficient

**Key numbers so far:**

| Metric | Value |
|---|---|
| Deployable-energy bias | **0.05 MJ** |
| Belief band coverage | **0.75 – 0.92** |

---

## What's Next

- 🟡 **OpenF1 2026 ingest** — real speed traces at 3.7 Hz into the same inference pipeline
- 🟡 **Fuzzy RDD at the 1.000 s Override boundary** — causal estimate of decision value at the energy cutoff
- 🟡 **Overtake model fitted on real 2026 outcomes** — currently synthetic design anchors; the dataset is built, fit blocked pending track-status channel
- 🟡 **F1 25 UDP cross-check** — true store energy vs. X-RAY estimate, apples-to-apples
- 🟡 **Live dashboard** — six-view React / Three.js application for the final demo

---

## Why This Changes the Game

Self-play analysis: if one team can see the rival's battery, the rational response is to make your own battery unreadable — vary your strategy, use the override unpredictably. **The equilibrium goes mixed. Unpredictability becomes optimal.**

That is the game theory consequence of a working estimator.

X-RAY is built for the **public data regime** — broadcast feeds, fan analytics, any context where no engine channel exists. The same physics that hides the battery also constrains it. We use the constraint.

---

*Power Buff Girls · Hackathon ideation submission · 2026*
