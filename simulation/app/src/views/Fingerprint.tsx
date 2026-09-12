import type { Car } from '../lib/api';
import { C } from '../lib/theme';

/** View 5 — the deployment style of a driver, as it accumulates evidence. */
function radar(car: Car | null, upto: number) {
  if (!car) return null;
  const all = Object.keys(car.deployed_lap).map(Number).sort((a, b) => a - b);
  if (all.length < 2) return null;
  // At the start of a replay there is not yet any evidence. Show the first few
  // laps rather than an empty chart, and let it grow from there.
  let laps = all.filter((l) => l <= upto);
  if (laps.length < 2) laps = all.slice(0, Math.min(3, all.length));
  const dep = laps.map((l) => car.deployed_lap[String(l)] ?? 0);
  const har = laps.map((l) => car.harvested_lap[String(l)] ?? 0);
  const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / Math.max(a.length, 1);
  const sd = (a: number[]) => {
    const m = mean(a);
    return Math.sqrt(mean(a.map((x) => (x - m) ** 2)));
  };
  const half = Math.max(Math.floor(laps.length / 2), 1);
  const front = mean(dep.slice(0, half)), back = mean(dep.slice(half));
  return {
    n: laps.length,
    axes: [
      { key: 'aggression', label: 'aggression', v: Math.min(mean(dep) / 4, 1) },
      { key: 'front', label: 'front-loading',
        v: Math.min(Math.max(0.5 + (front - back) / Math.max(front + back, 0.01), 0), 1) },
      { key: 'reserve', label: 'buffer held', v: Math.min(car.reserve_mean / 1.4e6, 1) },
      { key: 'consistency', label: 'consistency',
        v: Math.min(Math.max(1 - sd(dep) / Math.max(mean(dep), 0.01), 0), 1) },
      { key: 'recovery', label: 'recovery',
        v: Math.min(mean(har) / Math.max(mean(dep), 0.01), 1) },
    ],
    // uncertainty shrinks as evidence accumulates
    spread: 1 / Math.sqrt(laps.length),
  };
}

export function Fingerprint({ cars, upto }: { cars: (Car | null)[]; upto: number }) {
  const R = 112, cx = 165, cy = 168;
  const fps = cars.map((c) => radar(c, upto));
  const axes = fps.find(Boolean)?.axes ?? [];
  const pt = (i: number, v: number) => {
    const a = (i / Math.max(axes.length, 1)) * Math.PI * 2 - Math.PI / 2;
    return [cx + Math.cos(a) * R * v, cy + Math.sin(a) * R * v];
  };
  const colours = [C.amber, C.red];
  return (
    <div style={{ padding: '26px 34px', height: '100%', overflow: 'auto' }}>
      <h2 className="display" style={{ fontSize: 26, margin: 0 }}>Opponent fingerprint</h2>
      <p style={{ color: C.gray, maxWidth: 820, fontSize: 13.5, lineHeight: 1.65 }}>
        Deployment style, estimated from the speed trace and nothing else. The
        shaded ring is the uncertainty, and it tightens as the race supplies more
        laps — drag the scrub bar in Race Theatre and watch it close.
      </p>
      <div style={{ display: 'flex', gap: 34, marginTop: 18, flexWrap: 'wrap' }}>
        <div className="panel" style={{ padding: 16 }}>
          <svg width={330} height={330} viewBox="0 0 330 330">
            {[0.25, 0.5, 0.75, 1].map((r) => (
              <circle key={r} cx={cx} cy={cy} r={R * r} fill="none"
                stroke={C.grid} strokeWidth={1} />
            ))}
            {axes.map((a, i) => {
              const [x, y] = pt(i, 1.0);
              const [lx, ly] = pt(i, 1.27);
              return (
                <g key={a.key}>
                  <line x1={cx} y1={cy} x2={x} y2={y} stroke={C.grid} />
                  <text x={lx} y={ly} fill={C.gray} fontSize={10.5}
                    textAnchor={lx < cx - 12 ? 'end' : lx > cx + 12 ? 'start' : 'middle'}
                    dominantBaseline="middle">{a.label}</text>
                </g>
              );
            })}
            {fps.map((fp, k) => fp && (
              <g key={k}>
                <polygon
                  points={fp.axes.map((a, i) =>
                    pt(i, Math.min(a.v + fp.spread, 1)).join(',')).join(' ')}
                  fill={colours[k]} opacity={0.13} />
                <polygon
                  points={fp.axes.map((a, i) =>
                    pt(i, Math.max(a.v - fp.spread, 0)).join(',')).join(' ')}
                  fill={colours[k]} opacity={0.2} />
                <polygon points={fp.axes.map((a, i) => pt(i, a.v).join(',')).join(' ')}
                  fill="none" stroke={colours[k]} strokeWidth={2.2} />
              </g>
            ))}
          </svg>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, minWidth: 300 }}>
          {cars.map((c, k) => c && fps[k] && (
            <div key={c.driver} className="panel" style={{ padding: 15 }}>
              <div style={{ color: colours[k], fontWeight: 800, fontSize: 14,
                            marginBottom: 8 }}>{c.driver}</div>
              <table className="num" style={{ width: '100%', fontSize: 12.5, color: C.gray }}>
                <tbody>
                  {fps[k]!.axes.map((a) => (
                    <tr key={a.key}>
                      <td style={{ padding: '3px 0' }}>{a.label}</td>
                      <td style={{ textAlign: 'right', color: C.white }}>
                        {(a.v * 100).toFixed(0)}%</td>
                    </tr>
                  ))}
                  <tr><td style={{ paddingTop: 7, color: C.dim }}>laps of evidence</td>
                    <td style={{ textAlign: 'right', paddingTop: 7, color: C.dim }}>
                      {fps[k]!.n}</td></tr>
                </tbody>
              </table>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
