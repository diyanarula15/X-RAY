import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { api } from '../lib/api';
import { C } from '../lib/theme';

/**
 * View 3 — the falsification instrument.
 *
 * Flat 2D, no effects, no bloom. Its authority comes from looking as though it
 * was not art-directed. Hand the judge the slider and let them hunt for
 * spurious effects at 0.7, 0.9, 1.3. Watching someone fail to break your result
 * is worth more than any claim you can make — and if they DO break it, the
 * honest thing is that they can see that too.
 */
export function RDD() {
  const [cutoff, setCutoff] = useState(1.0);
  const [bw, setBw] = useState(0.6);
  const [data, setData] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [show, setShow] = useState({ mccrary: false, covariates: false });
  const scatter = useRef<SVGSVGElement>(null);
  const strip = useRef<SVGSVGElement>(null);

  useEffect(() => {
    let dead = false;
    setBusy(true);
    api.rdd(cutoff, bw).then((d) => { if (!dead) { setData(d); setBusy(false); } })
      .catch(() => setBusy(false));
    return () => { dead = true; };
  }, [cutoff, bw]);

  useEffect(() => {
    if (!data?.points?.length) return;
    const svg = d3.select(scatter.current!); svg.selectAll('*').remove();
    const W = 780, H = 380, m = { l: 62, r: 18, t: 16, b: 46 };
    svg.attr('viewBox', `0 0 ${W} ${H}`);
    // clamping piles every long gap onto the right-hand edge as a fake stripe
    const pts = (data.points as any[]).filter((d) => d.gap_s <= 6);
    const x = d3.scaleLinear().domain([0, 6]).range([m.l, W - m.r]);
    const y = d3.scaleLinear()
      .domain([0, (d3.quantile(pts.map((p) => p.next_lap_mj).sort(d3.ascending), 0.98) ?? 6) * 1.1])
      .range([H - m.b, m.t]).nice();

    svg.append('g').attr('transform', `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).ticks(7)).call((g) => {
        g.selectAll('line,path').attr('stroke', C.grid);
        g.selectAll('text').attr('fill', C.gray).attr('font-size', 11);
      });
    svg.append('g').attr('transform', `translate(${m.l},0)`)
      .call(d3.axisLeft(y).ticks(6)).call((g) => {
        g.selectAll('line,path').attr('stroke', C.grid);
        g.selectAll('text').attr('fill', C.gray).attr('font-size', 11);
      });
    svg.append('text').attr('x', W / 2).attr('y', H - 8).attr('fill', C.gray)
      .attr('font-size', 11.5).attr('text-anchor', 'middle')
      .text('gap to the car ahead at the line (s)');
    svg.append('text').attr('transform', `rotate(-90)`).attr('x', -H / 2).attr('y', 15)
      .attr('fill', C.gray).attr('font-size', 11.5).attr('text-anchor', 'middle')
      .text('estimated next-lap deployment (MJ)');

    // bandwidth shading — what the fit can actually see
    svg.append('rect').attr('x', x(Math.max(cutoff - bw, 0))).attr('y', m.t)
      .attr('width', Math.max(x(cutoff + bw) - x(Math.max(cutoff - bw, 0)), 0))
      .attr('height', H - m.b - m.t).attr('fill', C.white).attr('opacity', 0.035);

    svg.append('g').selectAll('circle').data(pts).join('circle')
      .attr('cx', (d: any) => x(d.gap_s)).attr('cy', (d: any) => y(d.next_lap_mj))
      .attr('r', 2.1)
      .attr('fill', (d: any) => (d.gap_s < cutoff ? C.amber : C.gray))
      .attr('opacity', (d: any) => (Math.abs(d.gap_s - cutoff) <= bw ? 0.62 : 0.14));

    // fitted lines either side
    if (data.left && data.right) {
      const seg = (side: 'left' | 'right') => {
        const f = data[side];
        const x0 = side === 'left' ? Math.max(cutoff - bw, 0) : cutoff;
        const x1 = side === 'left' ? cutoff : cutoff + bw;
        return [[x0, f.intercept + f.slope * (x0 - cutoff)],
                [x1, f.intercept + f.slope * (x1 - cutoff)]] as [number, number][];
      };
      (['left', 'right'] as const).forEach((s) => {
        svg.append('path').datum(seg(s))
          .attr('d', d3.line<[number, number]>().x((d) => x(d[0])).y((d) => y(d[1])) as any)
          .attr('stroke', s === 'left' ? C.amber : C.gray).attr('stroke-width', 2.4)
          .attr('fill', 'none');
      });
    }
    svg.append('line').attr('x1', x(cutoff)).attr('x2', x(cutoff))
      .attr('y1', m.t).attr('y2', H - m.b).attr('stroke', C.white)
      .attr('stroke-width', 2).attr('stroke-dasharray', '6 4');
    svg.append('text').attr('x', x(cutoff) + 7).attr('y', m.t + 13).attr('fill', C.white)
      .attr('font-size', 11.5).text(`cutoff ${cutoff.toFixed(2)} s`);
  }, [data, cutoff, bw]);

  useEffect(() => {
    if (!data?.scan?.length) return;
    const svg = d3.select(strip.current!); svg.selectAll('*').remove();
    const W = 780, H = 132, m = { l: 62, r: 18, t: 12, b: 32 };
    svg.attr('viewBox', `0 0 ${W} ${H}`);
    const rows = (data.scan as any[]).filter((r) => r.p_value != null);
    const x = d3.scaleLinear().domain([0.4, 2.2]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);
    svg.append('g').attr('transform', `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).ticks(8)).call((g) => {
        g.selectAll('line,path').attr('stroke', C.grid);
        g.selectAll('text').attr('fill', C.gray).attr('font-size', 10.5);
      });
    svg.append('g').attr('transform', `translate(${m.l},0)`)
      .call(d3.axisLeft(y).ticks(3)).call((g) => {
        g.selectAll('line,path').attr('stroke', C.grid);
        g.selectAll('text').attr('fill', C.gray).attr('font-size', 10.5);
      });
    svg.append('line').attr('x1', m.l).attr('x2', W - m.r)
      .attr('y1', y(0.05)).attr('y2', y(0.05)).attr('stroke', C.red)
      .attr('stroke-dasharray', '4 4').attr('stroke-width', 1);
    svg.append('text').attr('x', W - m.r).attr('y', y(0.05) - 5).attr('fill', C.red)
      .attr('font-size', 10).attr('text-anchor', 'end').text('p = 0.05');
    svg.append('path').datum(rows)
      .attr('d', d3.line<any>().x((d) => x(d.cutoff)).y((d) => y(d.p_value)) as any)
      .attr('stroke', C.white).attr('stroke-width', 2).attr('fill', 'none');
    svg.append('g').selectAll('circle').data(rows).join('circle')
      .attr('cx', (d: any) => x(d.cutoff)).attr('cy', (d: any) => y(d.p_value))
      .attr('r', (d: any) => (d.p_value < 0.05 ? 4 : 2.4))
      .attr('fill', (d: any) => (d.p_value < 0.05 ? C.red : C.gray));
    svg.append('line').attr('x1', x(1.0)).attr('x2', x(1.0)).attr('y1', m.t)
      .attr('y2', H - m.b).attr('stroke', C.amber).attr('stroke-width', 1.4);
    svg.append('text').attr('x', x(1.0) + 5).attr('y', m.t + 11).attr('fill', C.amber)
      .attr('font-size', 10).text('1.000 s — the regulation boundary');
    svg.append('text').attr('x', 6).attr('y', H / 2).attr('fill', C.gray)
      .attr('font-size', 10.5).text('p-value');
  }, [data]);

  const p = data?.power;
  return (
    <div style={{ padding: '26px 34px', height: '100%', overflow: 'auto' }}>
      <h2 className="display" style={{ fontSize: 26, margin: 0 }}>RDD explorer</h2>
      <p style={{ color: C.gray, maxWidth: 860, fontSize: 13.5, lineHeight: 1.65 }}>
        There is no public ground-truth energy channel for a real car, so the
        estimates cannot be checked directly. What can be checked is a prediction
        the <i>regulation</i> makes: a car within 1.000 s of the car ahead becomes
        eligible for extra deployment and a car at 1.001 s does not. If we are
        measuring deployment, it should jump there and nowhere else.
        <b style={{ color: C.white }}> Drag the cutoff and try to find an effect
        somewhere the rules do not put one.</b>
      </p>

      <div className="panel" style={{ padding: '14px 18px', margin: '14px 0',
        display: 'flex', gap: 26, alignItems: 'center', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 340px' }}>
          <div style={{ color: C.gray, fontSize: 11, marginBottom: 6 }}>
            cutoff <b className="num" style={{ color: C.white }}>{cutoff.toFixed(2)} s</b>
          </div>
          <input type="range" min={0.4} max={2.2} step={0.01} value={cutoff}
            onChange={(e) => setCutoff(+e.target.value)}
            style={{ width: '100%', accentColor: C.amber }} />
        </div>
        <div style={{ flex: '0 1 200px' }}>
          <div style={{ color: C.gray, fontSize: 11, marginBottom: 6 }}>
            bandwidth <b className="num" style={{ color: C.white }}>{bw.toFixed(2)}</b>
          </div>
          <input type="range" min={0.2} max={1.5} step={0.05} value={bw}
            onChange={(e) => setBw(+e.target.value)}
            style={{ width: '100%', accentColor: C.gray }} />
        </div>
        <button onClick={() => setCutoff(1.0)}
          style={{ padding: '7px 12px', fontSize: 11.5, borderRadius: 7,
            border: `1px solid ${C.amber}`, background: 'transparent', color: C.amber }}>
          reset to 1.000 s
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 20 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div className="panel" style={{ padding: 12 }}>
            <svg ref={scatter} style={{ width: '100%', height: 'auto', display: 'block' }} />
          </div>
          <div className="panel" style={{ padding: 12 }}>
            <div style={{ color: C.gray, fontSize: 11, marginBottom: 4 }}>
              significance against cutoff position — a real effect spikes at 1.000 s
              and is flat elsewhere
            </div>
            <svg ref={strip} style={{ width: '100%', height: 'auto', display: 'block' }} />
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div className="panel" style={{ padding: 15 }}>
            <div style={{ color: C.gray, fontSize: 10.5, letterSpacing: '0.09em',
                          fontWeight: 700, marginBottom: 10 }}>
              ESTIMATE {busy && <span style={{ color: C.dim }}>· …</span>}
            </div>
            {data?.effect != null ? (
              <>
                <div className="num" style={{ fontSize: 30, fontWeight: 800,
                  color: data.p_value < 0.05 ? C.green : C.white }}>
                  {data.effect > 0 ? '+' : ''}{data.effect.toFixed(3)} <span style={{ fontSize: 14 }}>MJ</span>
                </div>
                <div className="num" style={{ color: C.gray, fontSize: 12.5, marginTop: 6,
                  lineHeight: 1.7 }}>
                  se {data.se?.toFixed(3)} · z {data.z?.toFixed(2)}<br />
                  <b style={{ color: data.p_value < 0.05 ? C.green : C.gray }}>
                    p = {data.p_value?.toFixed(4)}</b><br />
                  n = {data.n} car-laps · {data.n_left}/{data.n_right} either side
                </div>
              </>
            ) : <div style={{ color: C.dim, fontSize: 12 }}>{data?.note ?? 'loading…'}</div>}
          </div>

          {p?.available && (
            <div className="panel" style={{ padding: 15,
              borderColor: p.powered ? C.panelBorder : C.amber }}>
              <div style={{ color: p.powered ? C.gray : C.amber, fontSize: 10.5,
                letterSpacing: '0.09em', fontWeight: 700, marginBottom: 9 }}>
                {p.powered ? 'STATISTICAL POWER' : 'UNDERPOWERED'}
              </div>
              <div className="num" style={{ fontSize: 20, color: C.white }}>
                {p.mde_mj.toFixed(2)} MJ
              </div>
              <div style={{ color: C.gray, fontSize: 12, marginTop: 6, lineHeight: 1.6 }}>
                smallest effect this many races can resolve, against a Manual
                Override allocation of {p.target_effect_mj} MJ.
              </div>
              <div style={{ color: C.amber, fontSize: 12, marginTop: 9, lineHeight: 1.6 }}>
                {p.note}
              </div>
            </div>
          )}

          {(['mccrary', 'covariates'] as const).map((k) => (
            <div key={k} className="panel" style={{ padding: 13 }}>
              <button onClick={() => setShow((s) => ({ ...s, [k]: !s[k] }))}
                style={{ background: 'none', border: 'none', padding: 0, width: '100%',
                  textAlign: 'left', color: C.gray, fontSize: 10.5,
                  letterSpacing: '0.09em', fontWeight: 700 }}>
                {show[k] ? '▾' : '▸'} {k === 'mccrary' ? 'MCCRARY DENSITY' : 'COVARIATE CONTINUITY'}
              </button>
              {show[k] && k === 'mccrary' && data?.mccrary?.available && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ display: 'flex', alignItems: 'flex-end', gap: 2, height: 54 }}>
                    {data.mccrary.counts.map((c: number, i: number) => {
                      const mx = Math.max(...data.mccrary.counts, 1);
                      const left = i < data.mccrary.counts.length / 2;
                      return <div key={i} style={{ flex: 1, height: `${(c / mx) * 100}%`,
                        background: left ? C.amber : C.gray, opacity: 0.75 }} />;
                    })}
                  </div>
                  <div style={{ color: data.mccrary.suspicious ? C.red : C.gray,
                    fontSize: 11.5, marginTop: 8, lineHeight: 1.55 }}>
                    {data.mccrary.note} (log jump {data.mccrary.log_jump})
                  </div>
                </div>
              )}
              {show[k] && k === 'covariates' && (
                <table className="num" style={{ width: '100%', marginTop: 10,
                  fontSize: 11.5, color: C.gray, borderCollapse: 'collapse' }}>
                  <tbody>
                    {(data?.covariates ?? []).map((c: any) => (
                      <tr key={c.covariate}>
                        <td style={{ padding: '4px 0' }}>{c.covariate}</td>
                        <td style={{ textAlign: 'right' }}>z {c.z}</td>
                        <td style={{ textAlign: 'right',
                          color: c.balanced ? C.green : C.red, paddingLeft: 8 }}>
                          {c.balanced ? 'balanced' : 'IMBALANCED'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
