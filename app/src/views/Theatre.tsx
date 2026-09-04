import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useRef } from 'react';
import type { Car, RaceDetail } from '../lib/api';
import { C } from '../lib/theme';
import { usePlayback } from '../store/playback';
import { EnergyBar, Panel, Refusal } from '../components/Readouts';
import { ScrubBar } from '../components/ScrubBar';
import { Scene } from '../three/Scene';
import { SpeedTrace } from '../components/SpeedTrace';

export function Theatre({ race, subject, rival, obs }: {
  race: RaceDetail; subject: Car | null; rival: Car | null; obs: any;
}) {
  const frame = usePlayback((s) => s.frame);
  const setFrame = usePlayback((s) => s.setFrame);
  const playing = usePlayback((s) => s.playing);
  const speed = usePlayback((s) => s.speed);
  const paint = usePlayback((s) => s.paintMode);
  const setPaint = usePlayback((s) => s.setPaint);
  const cam = usePlayback((s) => s.camera);
  const setCam = usePlayback((s) => s.setCamera);
  const lite = usePlayback((s) => s.lite);
  const n = subject?.trace.s.length ?? 0;

  // the one clock: everything else reads `frame`
  const raf = useRef(0); const last = useRef(performance.now());
  useEffect(() => {
    if (!playing || !n) return;
    const tick = () => {
      const now = performance.now();
      const dt = Math.min(now - last.current, 60); last.current = now;
      const next = frameRef.current + (dt / 1000) * 24 * speed;
      if (next >= n - 1) { usePlayback.getState().pause(); setFrame(n - 1); return; }
      setFrame(next);
      raf.current = requestAnimationFrame(tick);
    };
    last.current = performance.now();
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, n, speed]);
  const frameRef = useRef(frame); frameRef.current = frame;

  const i = Math.min(Math.round(frame), Math.max(n - 1, 0));
  const t = subject?.trace, rt = rival?.trace;
  const lap = t?.lap[i] ?? 0;
  const refusal = rival ? null : Object.entries(race.refusals)[0];

  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.code === 'Space') { e.preventDefault(); usePlayback.getState().toggle(); }
      if (e.code === 'ArrowRight') setFrame(Math.min(frameRef.current + 12, n - 1));
      if (e.code === 'ArrowLeft') setFrame(Math.max(frameRef.current - 12, 0));
    };
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [n]);

  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <Scene geo={race.circuit_geometry} subject={subject} rival={rival} obs={obs} />

      {/* top-left: where we are */}
      <div style={{ position: 'absolute', top: 18, left: 18, pointerEvents: 'none' }}>
        <div className="display" style={{ fontSize: 26, lineHeight: 1 }}>{race.event}</div>
        <div style={{ color: C.gray, fontSize: 12.5, marginTop: 5 }}>
          {race.circuit} · {race.date} · lap <b className="num" style={{ color: C.white }}>{lap}</b>
          {' '}· {race.telemetry.median_hz} Hz public telemetry
        </div>
      </div>

      {/* right rail: the instrument */}
      <div style={{ position: 'absolute', top: 18, right: 18, width: 330,
                    display: 'flex', flexDirection: 'column', gap: 12 }}>
        <Panel>
          <EnergyBar label={`YOU — ${subject?.driver ?? '—'}`}
            mean={t ? t.usable_mean[i] : null} p10={t ? t.usable_p10[i] : null}
            p90={t ? t.usable_p90[i] : null} colour={C.amber}
            sub={subject ? `deploying ${Math.round(t!.deploy_kw[i] ?? 0)} kW` : undefined} />
          <EnergyBar label={`RIVAL — ${rival?.driver ?? '—'}`}
            mean={rt ? rt.usable_mean[i] : null} p10={rt ? rt.usable_p10[i] : null}
            p90={rt ? rt.usable_p90[i] : null} colour={C.red}
            unknown={paint === 'neutral' && !usePlayback.getState().showCloud}
            sub={rt ? `band ±${(((rt.usable_p90[i] ?? 0) - (rt.usable_p10[i] ?? 0)) / 2).toFixed(2)} MJ · reconstructed` : undefined} />
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

        {rival && rt?.dry[i] && (
          <AnimatePresence>
            <motion.div key="dry" initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}>
              <Refusal tone="amber" title="DEPLOYMENT CUT-OUT"
                message="The rival stopped deploying while still on the throttle below the taper. The store has reached the floor this driver refuses to spend — the band collapses onto it." />
            </motion.div>
          </AnimatePresence>
        )}

        {!rival && refusal && (
          <Refusal title={`CANNOT ESTIMATE — ${refusal[0]}`} message={refusal[1].message} />
        )}

        <Panel style={{ padding: 11 }}>
          <div style={{ color: C.gray, fontSize: 10, letterSpacing: '0.09em',
                        marginBottom: 7, fontWeight: 700 }}>TRACK PAINT</div>
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
                        margin: '11px 0 7px', fontWeight: 700 }}>CAMERA</div>
          <div style={{ display: 'flex', gap: 5 }}>
            {(['chase', 'duel', 'tactical'] as const).map((m) => (
              <button key={m} onClick={() => setCam(m)}
                style={{ padding: '5px 9px', fontSize: 11, borderRadius: 6,
                         border: `1px solid ${cam === m ? C.amber : C.panelBorder}`,
                         background: 'transparent', color: cam === m ? C.amber : C.gray }}>
                {m}
              </button>
            ))}
          </div>
        </Panel>
      </div>

      {/* bottom: speed traces and the scrub */}
      <div style={{ position: 'absolute', bottom: 16, left: 18, right: 366,
                    display: 'flex', flexDirection: 'column', gap: 10 }}>
        <SpeedTrace subject={subject} rival={rival} frame={i} />
        <ScrubBar car={subject} />
      </div>

      {lite && (
        <div style={{ position: 'absolute', bottom: 16, right: 18, color: C.dim,
                      fontSize: 11 }}>lite mode</div>
      )}
    </div>
  );
}
