import { useMemo } from 'react';
import type { Car } from '../lib/api';
import { C } from '../lib/theme';
import { usePlayback } from '../store/playback';
import { useThrottledTime } from '../store/clock';

/** Scrub over RACE TIME, with lap ticks, cut-outs and coast phases.
 *  When the user scrubs, everything moves together and instantly. */
export function ScrubBar({ car, rival }: { car: Car | null; rival: Car | null }) {
  const raceTime = useThrottledTime(12);
  const setTime = usePlayback((s) => s.setTime);
  const [t0, t1] = usePlayback((s) => s.tRange);
  const playing = usePlayback((s) => s.playing);
  const toggle = usePlayback((s) => s.toggle);
  const speed = usePlayback((s) => s.speed);
  const setSpeed = usePlayback((s) => s.setSpeed);

  const marks = useMemo(() => {
    const tr = car?.trace as any;
    if (!tr?.t) return { laps: [], dry: [], coast: [] };
    const laps: { t: number; lap: number }[] = [];
    let prev = -1;
    tr.lap.forEach((l: number, i: number) => {
      if (l !== prev) { laps.push({ t: tr.t[i], lap: l }); prev = l; }
    });
    const rt = (rival?.trace as any);
    const dry = rt?.t ? rt.dry.map((d: boolean, i: number) => (d ? rt.t[i] : -1))
      .filter((x: number) => x >= 0) : [];
    const coast = tr.coast.map((d: boolean, i: number) => (d ? tr.t[i] : -1))
      .filter((x: number) => x >= 0);
    return { laps, dry, coast };
  }, [car, rival]);

  const pc = (t: number) => `${((t - t0) / Math.max(t1 - t0, 1)) * 100}%`;
  const mmss = (t: number) => {
    const s = Math.max(t - t0, 0);
    return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  };

  return (
    <div className="panel" style={{ padding: '10px 14px 12px', display: 'flex',
                                    gap: 14, alignItems: 'center' }}>
      <button onClick={toggle}
        style={{ width: 40, height: 40, borderRadius: 20, border: `1px solid ${C.panelBorder}`,
                 background: playing ? C.amber : 'transparent',
                 color: playing ? C.bg : C.white, fontSize: 14, flexShrink: 0 }}>
        {playing ? '❚❚' : '▶'}
      </button>
      <div className="num" style={{ color: C.gray, fontSize: 12, width: 46 }}>
        {mmss(raceTime)}
      </div>
      <div style={{ flex: 1 }}>
        <div style={{ position: 'relative', height: 30 }}>
          <div style={{ position: 'absolute', left: 0, right: 0, top: 13, height: 4,
                        background: '#0E0E13', borderRadius: 2 }} />
          {marks.coast.map((t: number, i: number) => (
            <div key={`c${i}`} style={{ position: 'absolute', left: pc(t), top: 13,
              width: 1, height: 4, background: '#39C6E0', opacity: 0.45 }} />
          ))}
          {marks.laps.map((m) => (
            <div key={`l${m.t}`} style={{ position: 'absolute', left: pc(m.t), top: 6,
              width: 1, height: 18, background: C.dim, opacity: 0.75 }}
              title={`lap ${m.lap}`} />
          ))}
          {marks.dry.map((t: number, i: number) => (
            <div key={`d${i}`} style={{ position: 'absolute', left: pc(t), top: 4,
              width: 2, height: 22, background: C.amber }} title="rival deployment cut-out" />
          ))}
          <div style={{ position: 'absolute', left: pc(raceTime), top: 2, width: 2,
                        height: 26, background: C.white, boxShadow: `0 0 8px ${C.white}` }} />
          <input type="range" min={t0} max={t1} step={0.05} value={raceTime}
            onChange={(e) => setTime(+e.target.value)}
            style={{ position: 'absolute', inset: 0, width: '100%', opacity: 0,
                     cursor: 'ew-resize' }} />
        </div>
        <div style={{ display: 'flex', gap: 16, color: C.dim, fontSize: 10, marginTop: 2 }}>
          <span><i style={{ background: C.dim, width: 8, height: 2, display: 'inline-block',
            verticalAlign: 'middle', marginRight: 4 }} />lap</span>
          <span><i style={{ background: C.amber, width: 8, height: 2, display: 'inline-block',
            verticalAlign: 'middle', marginRight: 4 }} />rival cut-out</span>
          <span><i style={{ background: '#39C6E0', width: 8, height: 2, display: 'inline-block',
            verticalAlign: 'middle', marginRight: 4 }} />coast (calibration)</span>
        </div>
      </div>
      <div style={{ display: 'flex', gap: 4 }}>
        {[0.5, 1, 2, 5, 10].map((s) => (
          <button key={s} onClick={() => setSpeed(s)}
            style={{ padding: '5px 8px', borderRadius: 6, fontSize: 11,
                     border: `1px solid ${speed === s ? C.amber : C.panelBorder}`,
                     background: 'transparent', color: speed === s ? C.amber : C.gray }}>
            {s}×
          </button>
        ))}
      </div>
    </div>
  );
}
