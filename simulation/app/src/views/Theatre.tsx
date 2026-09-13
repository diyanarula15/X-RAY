import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useMemo } from 'react';
import type { Car, RaceDetail } from '../lib/api';
import { sampleCar } from '../lib/carState';
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
    let d = rS.s - sS.s;
    if (d > L / 2) d -= L; else if (d < -L / 2) d += L;
    const behind = d > 0 ? sS : rS;
    return { seconds: Math.abs(d) / Math.max(behind.v, 5), metres: Math.abs(d),
             subjectAhead: d > 0 };
  }, [sS, rS, L]);

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
        {gap && (
          <div style={{ marginTop: 9, display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <span className="num" style={{ fontSize: 30, fontWeight: 800,
              color: gap.seconds < 1 ? C.green : C.white }}>
              {gap.seconds.toFixed(2)}s
            </span>
            <span style={{ color: C.gray, fontSize: 12 }}>
              {gap.metres.toFixed(0)} m · {gap.subjectAhead ? `${subject?.driver} ahead`
                : `${rival?.driver} ahead`}
              {gap.seconds < 1 && <b style={{ color: C.green }}> · Override eligible</b>}
            </span>
          </div>
        )}
      </div>

      <div style={{ position: 'absolute', top: GUTTER, right: GUTTER, width: RAIL_W,
                    maxHeight: `calc(100% - ${GUTTER * 2}px)`, overflowY: 'auto',
                    display: 'flex', flexDirection: 'column', gap: 12 }}>
        <Panel>
          <div style={{ color: C.dim, fontSize: 10.5, marginBottom: 8, lineHeight: 1.5 }}>
            Deployable energy — how much each car can still throw at the other.
            Yours is known; theirs is reconstructed from speed.
          </div>
          <EnergyBar label={`YOU — ${subject?.driver ?? '—'}`}
            mean={sS?.usable ?? null} p10={sS?.p10} p90={sS?.p90} colour={C.amber}
            sub={sS ? (sS.usable < 0.02
              ? `store spent · recovering ${Math.round(sS.harvest)} kW`
              : `deploying ${Math.round(sS.deploy)} kW`) : undefined} />
          <EnergyBar label={`RIVAL — ${rival?.driver ?? '—'}`}
            mean={rS?.usable ?? null} p10={rS?.p10} p90={rS?.p90} colour={C.red}
            unknown={!showCloud}
            sub={rS ? (rS.stale
              ? 'telemetry gap — estimate suspended'
              : rS.usable < 0.02
                ? 'store spent · nothing left to deploy at you'
                : `band ±${((rS.p90 - rS.p10) / 2).toFixed(2)} MJ · reconstructed`)
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
              <Refusal tone="amber" title="DEPLOYMENT CUT-OUT"
                message="The rival stopped deploying while still on the throttle below the taper. The store has reached the floor this driver refuses to spend — the band collapses onto it." />
            </motion.div>
          )}
        </AnimatePresence>

        {!rival && refusal && (
          <Refusal title={`CANNOT ESTIMATE — ${refusal[0]}`} message={refusal[1].message} />
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
