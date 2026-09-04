import { useMemo } from 'react';
import type { Car } from '../lib/api';
import { C } from '../lib/theme';
import { usePlayback } from '../store/playback';

/** Scrub over race distance, with lap ticks, Override windows and cut-outs.
 *  When the user scrubs, everything moves together and instantly. */
export function ScrubBar({ car }: { car: Car | null }) {
  const frame = usePlayback((s) => s.frame);
  const setFrame = usePlayback((s) => s.setFrame);
  const playing = usePlayback((s) => s.playing);
  const toggle = usePlayback((s) => s.toggle);
  const speed = usePlayback((s) => s.speed);
  const setSpeed = usePlayback((s) => s.setSpeed);
  const n = car?.trace.s.length ?? 1;

  const marks = useMemo(() => {
    if (!car) return { laps: [], dry: [], coast: [] };
    const laps: number[] = [];
    let prev = -1;
    car.trace.lap.forEach((l, i) => { if (l !== prev) { laps.push(i); prev = l; } });
    const dry = car.trace.dry.map((d, i) => (d ? i : -1)).filter((i) => i >= 0);
    const coast = car.trace.coast.map((d, i) => (d ? i : -1)).filter((i) => i >= 0);
    return { laps, dry, coast };
  }, [car]);

  const pc = (i: number) => `${(i / Math.max(n - 1, 1)) * 100}%`;

  return (
    <div className="panel" style={{ padding: '10px 14px 12px', display: 'flex',
                                    gap: 14, alignItems: 'center' }}>
      <button onClick={toggle}
        style={{ width: 40, height: 40, borderRadius: 20, border: `1px solid ${C.panelBorder}`,
                 background: playing ? C.amber : 'transparent',
                 color: playing ? C.bg : C.white, fontSize: 14, flexShrink: 0 }}>
        {playing ? '❚❚' : '▶'}
      </button>
      <div style={{ flex: 1 }}>
        <div style={{ position: 'relative', height: 30 }}>
          <div style={{ position: 'absolute', left: 0, right: 0, top: 13, height: 4,
                        background: '#0E0E13', borderRadius: 2 }} />
          {marks.coast.map((i) => (
            <div key={`c${i}`} style={{ position: 'absolute', left: pc(i), top: 13,
              width: 1, height: 4, background: '#39C6E0', opacity: 0.5 }} />
          ))}
          {marks.laps.map((i) => (
            <div key={`l${i}`} style={{ position: 'absolute', left: pc(i), top: 6,
              width: 1, height: 18, background: C.dim, opacity: 0.75 }}
              title={`lap ${car?.trace.lap[i]}`} />
          ))}
          {marks.dry.map((i) => (
            <div key={`d${i}`} style={{ position: 'absolute', left: pc(i), top: 4,
              width: 2, height: 22, background: C.amber }} title="deployment cut-out" />
          ))}
          <div style={{ position: 'absolute', left: pc(frame), top: 2, width: 2,
                        height: 26, background: C.white, boxShadow: `0 0 8px ${C.white}` }} />
          <input type="range" min={0} max={Math.max(n - 1, 1)} value={frame}
            onChange={(e) => setFrame(+e.target.value)}
            style={{ position: 'absolute', inset: 0, width: '100%', opacity: 0,
                     cursor: 'ew-resize' }} />
        </div>
        <div style={{ display: 'flex', gap: 16, color: C.dim, fontSize: 10,
                      marginTop: 2 }}>
          <span><i style={{ background: C.dim, width: 8, height: 2, display: 'inline-block',
            verticalAlign: 'middle', marginRight: 4 }} />lap</span>
          <span><i style={{ background: C.amber, width: 8, height: 2, display: 'inline-block',
            verticalAlign: 'middle', marginRight: 4 }} />deployment cut-out</span>
          <span><i style={{ background: '#39C6E0', width: 8, height: 2, display: 'inline-block',
            verticalAlign: 'middle', marginRight: 4 }} />coast (calibration)</span>
        </div>
      </div>
      <div style={{ display: 'flex', gap: 4 }}>
        {[0.5, 1, 2, 4].map((s) => (
          <button key={s} onClick={() => setSpeed(s)}
            style={{ padding: '5px 9px', borderRadius: 6, fontSize: 11,
                     border: `1px solid ${speed === s ? C.amber : C.panelBorder}`,
                     background: 'transparent', color: speed === s ? C.amber : C.gray }}>
            {s}×
          </button>
        ))}
      </div>
    </div>
  );
}
