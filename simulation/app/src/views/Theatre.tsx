import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useMemo } from 'react';
import type { Car, RaceDetail } from '../lib/api';
import { sampleCar, timeRange } from '../lib/carState';
import { C } from '../lib/theme';
import { usePlayback } from '../store/playback';
import { useThrottledTime } from '../store/clock';
import { EnergyBar, Panel, Refusal } from '../components/Readouts';
import { ScrubBar } from '../components/ScrubBar';
import { SpeedTrace } from '../components/SpeedTrace';
import { SceneLegend } from '../components/Explain';
import { Scene } from '../three/Scene';

// The right rail and the gutter it sits in, named once. These were hand-tuned
// magic numbers (`right: 366` = 330 + 18 + 18) repeated at three call sites, so
// changing the rail width silently broke the overlays that abutted it.
const RAIL_W = 330;
const GUTTER = 18;
const RAIL_EDGE = RAIL_W + GUTTER * 2;

// ROLE is fixed by the pairing; POSITION is the thing that moves, and they are
// two separate readouts here because the old header had only one: it printed
// "<driver> ahead" off the sign of `d`, so the only identity statement on screen
// was recomputed every frame and flipped the moment a pass completed. A battle
// where NORRIS=CHASER 0.4 s behind becomes NORRIS=CHASER 0.2 s ahead is one
// battle; renaming the target into the chaser makes it read as two. The subject
// of the session is the CHASER for every frame of the replay, the estimated car
// is the TARGET, and no code path below writes either.
const ROLE_SUBJECT = 'CHASER';
const ROLE_RIVAL = 'TARGET';

// Published bodywork geometry (2026 regulation caps overall length at 5.6 m),
// not an estimate and not a regulation call: "side by side" means the cars
// overlap along the lap. Nothing downstream of this is a strategy or an
// eligibility decision — see the eligibility note in the gap readout.
const CAR_LENGTH_M = 5.6;

export function Theatre({ race, subject, rival, obs }: {
  race: RaceDetail; subject: Car | null; rival: Car | null; obs: any;
}) {
  // The 3D scene reads the clock itself, inside its own loop. The overlays
  // sample it at a readable rate instead of forcing a React render per frame.
  const raceTime = useThrottledTime(12);
  const setTime = usePlayback((s) => s.setTime);
  const setView = usePlayback((s) => s.setView);
  const [t0, t1] = usePlayback((s) => s.tRange);
  const playing = usePlayback((s) => s.playing);
  const speed = usePlayback((s) => s.speed);
  const paint = usePlayback((s) => s.paintMode);
  const setPaint = usePlayback((s) => s.setPaint);
  const cam = usePlayback((s) => s.camera);
  const setCam = usePlayback((s) => s.setCamera);
  const showCloud = usePlayback((s) => s.showCloud);
  const setShowCloud = usePlayback((s) => s.setShowCloud);
  const lite = usePlayback((s) => s.lite);

  const L = race.circuit_geometry.length;
  const sS = useMemo(() => sampleCar(subject, raceTime, L), [subject, raceTime, L]);
  const rS = useMemo(() => sampleCar(rival, raceTime, L), [rival, raceTime, L]);

  // Real on-track gap: distance between the two cars along the lap, divided by
  // the speed of the one behind. Both are sampled at the same race time, which
  // is the whole reason this number means anything.
  const gap = useMemo(() => {
    if (!sS || !rS || !sS.onTrack || !rS.onTrack) return null;
    // d = rival.s - subject.s, wrap-corrected across the start line. d > 0 is
    // the RIVAL ahead and the subject behind. The field used to be called
    // `subjectAhead` and set to `d > 0`, i.e. the exact inverse: at
    // subject.s=100 / rival.s=110 the header read "<subject> ahead" while the
    // 3D scene, which places each car at `poseAt(geo, s)` (Scene.tsx), drew the
    // subject 10 m behind. Same frame, same numbers, opposite stories.
    // The name now states the sign convention so the two cannot disagree.
    let d = rS.s - sS.s;
    if (d > L / 2) d -= L; else if (d < -L / 2) d += L;
    const behind = d > 0 ? sS : rS;
    return { seconds: Math.abs(d) / Math.max(behind.v, 5), metres: Math.abs(d),
             rivalAhead: d > 0 };
  }, [sS, rS, L]);

  // Physical position, derived per frame, kept separate from the roles above.
  // `state` is a description of the track situation and is NOT a strategy call:
  // the canonical strategy vocabulary is ATTACK / HOLD and lives in Cockpit,
  // fed by the solver. Nothing here ever produces a third strategy word.
  const position = useMemo(() => {
    if (!gap) return null;
    if (gap.metres <= CAR_LENGTH_M) {
      return { subject: 'SIDE-BY-SIDE', rival: 'SIDE-BY-SIDE', state: 'SIDE-BY-SIDE' };
    }
    return gap.rivalAhead
      ? { subject: 'BEHIND', rival: 'AHEAD', state: 'CHASING' }
      : { subject: 'AHEAD', rival: 'BEHIND', state: 'DEFENDING POSITION' };
  }, [gap]);

  // Both cars have telemetry, but not necessarily over the same stretch of the
  // session. The shared clock is the intersection, so when it is empty there is
  // no moment at which the two can be compared at all — said out loud rather
  // than rendered as a frozen car with a null gap.
  const windowsOverlap = useMemo(() => {
    const a = timeRange(subject), b = timeRange(rival);
    if (!a || !b) return null;
    return Math.min(a[1], b[1]) > Math.max(a[0], b[0]);
  }, [subject, rival]);

  // the one clock, advancing in seconds of race time
  useEffect(() => {
    if (!playing) return;
    let raf = 0, last = performance.now();
    const tick = () => {
      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.1); last = now;
      // read the authoritative clock, not the throttled copy
      const next = usePlayback.getState().raceTime + dt * speed;
      if (next >= t1) { usePlayback.getState().pause(); setTime(t1); return; }
      setTime(next);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, t1]);

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.code === 'Space') { e.preventDefault(); usePlayback.getState().toggle(); }
      const now = usePlayback.getState().raceTime;
      if (e.code === 'ArrowRight') setTime(Math.min(now + 5, t1));
      if (e.code === 'ArrowLeft') setTime(Math.max(now - 5, t0));
    };
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [t0, t1]);

  const refusal = rival ? null : Object.entries(race.refusals)[0];

  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <Scene geo={race.circuit_geometry} subject={subject} rival={rival} obs={obs} />

      <div style={{ position: 'absolute', top: 18, left: 18, pointerEvents: 'none' }}>
        <div className="display" style={{ fontSize: 26, lineHeight: 1 }}>{race.event}</div>
        <div style={{ color: C.gray, fontSize: 12.5, marginTop: 5 }}>
          {race.circuit} · lap <b className="num" style={{ color: C.white }}>{sS?.lap ?? '—'}</b>
          {' '}· <span className="num">{(sS?.v ? sS.v * 3.6 : 0).toFixed(0)}</span> km/h
          {' '}· {race.telemetry.median_hz} Hz public telemetry
        </div>
        {/* HISTORICAL REPLAY, said at the top of the viewport. Without it this
            screen is indistinguishable from a live or simulated run, and the
            P2 recommendation rendered beside it reads as an action being taken
            — it is not: the recorded telemetry is the only thing moving here,
            and no counterfactual branch is executed anywhere in this view. */}
        <div style={{ marginTop: 9, display: 'inline-block', padding: '5px 9px',
                      border: `1px solid ${C.amber}`, borderRadius: 6,
                      background: 'rgba(255,195,0,.08)' }}>
          <b style={{ color: C.amber, fontSize: 11, letterSpacing: '0.09em' }}>
            HISTORICAL REPLAY
          </b>
          <span style={{ color: C.gray, fontSize: 11 }}> · Recorded telemetry</span>
          <div style={{ color: C.dim, fontSize: 10.5, marginTop: 3, maxWidth: 420,
                        lineHeight: 1.5 }}>
            X-RAY recommendation is advisory here. This replay does not execute
            the counterfactual branch.
          </div>
        </div>
        {gap && position && (
          <div style={{ marginTop: 9, display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <span className="num" style={{ fontSize: 30, fontWeight: 800, color: C.white }}>
              {gap.seconds.toFixed(2)}s
            </span>
            <span style={{ color: C.gray, fontSize: 12 }}>
              {gap.metres.toFixed(0)} m
              {' · '}{ROLE_SUBJECT} {subject?.driver} <b style={{ color: C.amber }}>
                {position.subject}</b>
              {' · '}{ROLE_RIVAL} {rival?.driver} <b style={{ color: C.red }}>
                {position.rival}</b>
              {' · '}<span style={{ color: C.dim }}>track state {position.state}</span>
            </span>
          </div>
        )}
        {/* The 1.000 s Manual Override rule is decided at a zone detection point
            against the regulation in force, not by whatever the instantaneous
            on-track gap happens to be in this frame. The old badge lit
            "Override eligible" off `gap.seconds < 1` computed right here, which
            is the frontend issuing a legal ruling from one interpolated sample;
            the car payload carries no Manual Override eligibility field
            (`regulation.zone_eligibility` is deployment-ZONE geometry, a
            different rule), so there is nothing to render and the definitive
            badge is gone. */}
        {gap && (
          <div style={{ color: C.dim, fontSize: 10.5, marginTop: 5 }}>
            Manual Override: eligibility not established — not inferred from this gap.
          </div>
        )}
      </div>

      <div style={{ position: 'absolute', top: GUTTER, right: GUTTER, width: RAIL_W,
                    maxHeight: `calc(100% - ${GUTTER * 2}px)`, overflowY: 'auto',
                    display: 'flex', flexDirection: 'column', gap: 12 }}>
        <Panel>
          {/* Neither side is known here. This panel used to read "Yours is
              known; theirs is reconstructed from speed", which is true of the
              simulator and false of every session this view can open: these are
              historical PUBLIC-data races, the payload marks no car as
              internally known, and both bars are the same particle-filter
              output off the same public speed trace. The asymmetric wording made
              the amber bar look like a measurement and invited reading its width
              as instrument noise rather than as an identified band. */}
          <div style={{ color: C.dim, fontSize: 10.5, marginBottom: 8, lineHeight: 1.5 }}>
            Estimated usable energy — both cars. Telemetry-derived from public
            speed alone; no car publishes its energy state. Bands are p10–p90.
          </div>
          <EnergyBar label={`${ROLE_SUBJECT} — ${subject?.driver ?? '—'}`}
            mean={sS?.usable ?? null} p10={sS?.p10} p90={sS?.p90} colour={C.amber}
            sub={sS ? (sS.usable < 0.02
              ? `estimated deployable band at its floor · recovering ${Math.round(sS.harvest)} kW`
              : `deploying ${Math.round(sS.deploy)} kW`) : undefined} />
          <EnergyBar label={`${ROLE_RIVAL} — ${rival?.driver ?? '—'}`}
            mean={rS?.usable ?? null} p10={rS?.p10} p90={rS?.p90} colour={C.red}
            unknown={!showCloud}
            sub={rS ? (rS.stale
              ? 'telemetry gap — estimate suspended'
              : rS.usable < 0.02
                ? 'estimated deployable band at its floor'
                : `band ±${((rS.p90 - rS.p10) / 2).toFixed(2)} MJ · telemetry-derived`)
              : undefined} />
          {rival && (
            <div style={{ borderTop: `1px solid ${C.panelBorder}`, paddingTop: 9,
                          color: C.dim, fontSize: 11, lineHeight: 1.6 }}>
              CdA {rival.cda.toFixed(2)} m² [{rival.cda_lo.toFixed(2)}–{rival.cda_hi.toFixed(2)}]
              {rival.inherited_pooled && <span style={{ color: C.amber }}> · pooled from the field</span>}
              <br />identifiability {(rival.identifiability * 100).toFixed(0)}%
              {' · '}{rival.n_coast_samples} coast samples
            </div>
          )}
        </Panel>

        <AnimatePresence>
          {rS?.dry && (
            <motion.div key="dry" initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 24 }}>
              {/* "The store HAS reached the floor" was a claim the trace cannot
                  support: a speed trace measures flows exactly and absolute
                  level only up to an unidentified constant, so a cut-out is
                  evidence consistent with a floor, not a measurement of one —
                  and a store/reserve split is not separately identifiable from
                  public telemetry at all. No estimator value changed with this
                  wording; only the strength of the claim. */}
              <Refusal tone="amber" title="DEPLOYMENT CUT-OUT"
                message="Deployment cut-out detected. Observed behaviour is consistent with the car being near its deployable-energy floor. Public telemetry does not uniquely identify store and reserve separately." />
            </motion.div>
          )}
        </AnimatePresence>

        {!rival && refusal && (
          <Refusal title={`CANNOT ESTIMATE — ${refusal[0]}`} message={refusal[1].message} />
        )}

        {/* Replay is the one tab not guarded on `subject && rival` upstream, so
            an unavailable car used to land here as a scene with one car missing,
            a null gap and an em-dash energy bar — indistinguishable from a
            loading frame. Say which side is missing instead. */}
        {(!subject || !rival) && (
          <Refusal title="PAIR NOT PLAYABLE"
            message={`The ${!subject ? ROLE_SUBJECT : ROLE_RIVAL} side has no `
              + 'analysed trace in this race, so there is no pair to replay. '
              + 'Pick another pairing in the bar above.'} />
        )}
        {windowsOverlap === false && (
          <Refusal title="NO SHARED TELEMETRY WINDOW"
            message="These two cars have no overlapping telemetry window in this race. There is no moment at which both can be sampled, so no gap and no comparison exist — the clock has nothing to span." />
        )}

        <SceneLegend />

        <Panel style={{ padding: 11 }}>
          <div style={{ color: C.gray, fontSize: 10, letterSpacing: '0.09em',
                        marginBottom: 3, fontWeight: 700 }}>TRACK PAINT</div>
          <div style={{ color: C.dim, fontSize: 10.5, marginBottom: 7 }}>
            what to colour the circuit by
          </div>
          <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
            {(['neutral', 'deployment', 'observability', 'refusal'] as const).map((m) => (
              <button key={m} onClick={() => setPaint(m)}
                style={{ padding: '5px 9px', fontSize: 11, borderRadius: 6,
                         border: `1px solid ${paint === m ? C.amber : C.panelBorder}`,
                         background: 'transparent', color: paint === m ? C.amber : C.gray }}>
                {m}
              </button>
            ))}
          </div>
          <div style={{ color: C.gray, fontSize: 10, letterSpacing: '0.09em',
                        margin: '11px 0 3px', fontWeight: 700 }}>CAMERA</div>
          <div style={{ color: C.dim, fontSize: 10.5, marginBottom: 7 }}>
            chase follows you · duel frames both · tactical shows the lap
          </div>
          <div style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
            {(['chase', 'duel', 'tactical'] as const).map((m) => (
              <button key={m} onClick={() => setCam(m)}
                style={{ padding: '5px 9px', fontSize: 11, borderRadius: 6,
                         border: `1px solid ${cam === m ? C.amber : C.panelBorder}`,
                         background: 'transparent', color: cam === m ? C.amber : C.gray }}>
                {m}
              </button>
            ))}
            <button onClick={() => setShowCloud(!showCloud)}
              style={{ marginLeft: 'auto', padding: '5px 9px', fontSize: 11, borderRadius: 6,
                       border: `1px solid ${showCloud ? C.red : C.panelBorder}`,
                       background: 'transparent', color: showCloud ? C.red : C.gray }}>
              cloud
            </button>
          </div>
        </Panel>
      </div>

      <div style={{ position: 'absolute', bottom: 16, left: GUTTER, right: RAIL_EDGE,
                    display: 'flex', flexDirection: 'column', gap: 10 }}>
        <SpeedTrace subject={subject} rival={rival} raceTime={raceTime} />
        <ScrubBar car={subject} rival={rival} />
      </div>

      {lite && (
        <div style={{ position: 'absolute', bottom: 16, right: 18, color: C.dim,
                      fontSize: 11, textAlign: 'right', lineHeight: 1.5 }}>
          lite mode
          {usePlayback.getState().autoLite && (
            <><br /><span style={{ color: C.amber }}>
              dropped automatically — this machine could not hold 30 fps
            </span></>
          )}
        </div>
      )}

      {/* The scenario walkthrough that used to hang here as a collapsed overlay
          is now the Situations tab. It was wedged between the 3D scene and the
          right rail at a hardcoded offset, easy to miss entirely, and its
          "find the next divergent instance" walk fired up to 40 serial requests
          with every control disabled meanwhile. */}
      {subject && rival && (
        <button onClick={() => setView('situations')}
          style={{ position: 'absolute', top: GUTTER, right: RAIL_EDGE,
                   padding: '7px 12px', fontSize: 12, borderRadius: 7,
                   border: `1px solid ${C.panelBorder}`, background: 'rgba(11,11,15,.85)',
                   color: C.gray }}>
          ▸ pick a situation
        </button>
      )}
    </div>
  );
}
