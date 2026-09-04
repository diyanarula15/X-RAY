import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { api } from '../lib/api';
import { C } from '../lib/theme';

/** View 4 — click any moment, see why the engine says what it says. */
export function Decision({ raceId, car, rival }:
  { raceId: string; car: string; rival: string }) {
  const [d, setD] = useState<any>(null);
  const [sel, setSel] = useState<any>(null);
  const ref = useRef<SVGSVGElement>(null);

  useEffect(() => {
    api.decision(raceId, car, rival).then(setD).catch(() => setD(null));
  }, [raceId, car, rival]);

  useEffect(() => {
    if (!d?.laps?.length) return;
    const svg = d3.select(ref.current!); svg.selectAll('*').remove();
    const W = 820, H = 400, m = { l: 58, r: 20, t: 18, b: 44 };
    svg.attr('viewBox', `0 0 ${W} ${H}`);
    const laps = d.laps as any[];
    const x = d3.scaleLinear().domain(d3.extent(laps, (r) => r.lap) as [number, number])
      .range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);

    svg.append('g').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(10))
      .call((g) => { g.selectAll('line,path').attr('stroke', C.grid);
                     g.selectAll('text').attr('fill', C.gray).attr('font-size', 11); });
    svg.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5))
      .call((g) => { g.selectAll('line,path').attr('stroke', C.grid);
                     g.selectAll('text').attr('fill', C.gray).attr('font-size', 11); });
    svg.append('text').attr('x', W / 2).attr('y', H - 8).attr('fill', C.gray)
      .attr('font-size', 11.5).attr('text-anchor', 'middle').text('lap');
    svg.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', 14)
      .attr('fill', C.gray).attr('font-size', 11.5).attr('text-anchor', 'middle')
      .text('pass probability');

    // the sensitivity fan: the recommendation across sampled opponent policies
    (d.fan?.curves ?? []).forEach((row: number[]) => {
      svg.append('path').datum(row.map((q, i) => [laps[i]?.lap ?? i, q]))
        .attr('d', d3.line<any>().x((p) => x(p[0])).y((p) => y(p[1])) as any)
        .attr('stroke', C.gray).attr('stroke-width', 1).attr('fill', 'none')
        .attr('opacity', 0.07);
    });

    svg.append('path').datum(laps)
      .attr('d', d3.line<any>().x((r) => x(r.lap)).y((r) => y(r.tau)) as any)
      .attr('stroke', C.white).attr('stroke-width', 2.2).attr('fill', 'none');
    svg.append('text').attr('x', x(laps[Math.floor(laps.length * 0.62)].lap))
      .attr('y', y(laps[Math.floor(laps.length * 0.62)].tau) - 9)
      .attr('fill', C.white).attr('font-size', 11.5).attr('font-weight', 700)
      .text('attack threshold τ');

    svg.append('g').selectAll('circle').data(laps).join('circle')
      .attr('cx', (r: any) => x(r.lap)).attr('cy', (r: any) => y(r.q))
      .attr('r', (r: any) => (r.attack ? 6 : 4))
      .attr('fill', (r: any) => (r.attack ? C.green : C.gray))
      .attr('stroke', C.bg).attr('stroke-width', 2).style('cursor', 'pointer')
      .on('click', (_e, r: any) => setSel(r));

    if (d.call) {
      svg.append('circle').attr('cx', x(d.call.lap)).attr('cy', y(d.call.q)).attr('r', 13)
        .attr('fill', 'none').attr('stroke', C.green).attr('stroke-width', 2.4);
      svg.append('text').attr('x', x(d.call.lap) + 20).attr('y', y(d.call.q) - 18)
        .attr('fill', C.green).attr('font-size', 12.5).attr('font-weight', 700)
        .text(`the call — lap ${d.call.lap}, zone ${d.call.zone}`);
    }
  }, [d]);

  if (!d) return <div style={{ padding: 34, color: C.dim }}>No decision trace for this pair.</div>;
  return (
    <div style={{ padding: '26px 34px', height: '100%', overflow: 'auto' }}>
      <h2 className="display" style={{ fontSize: 26, margin: 0 }}>Decision explorer</h2>
      <p style={{ color: C.gray, maxWidth: 820, fontSize: 13.5, lineHeight: 1.65 }}>
        {car} attacking {rival}. Every number here decomposes — click a lap.
        The faint fan behind the threshold is the same recommendation re-solved
        across {d.fan?.n_policies ?? 200} sampled opponent policies.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 310px', gap: 20,
                    marginTop: 14 }}>
        <div className="panel" style={{ padding: 12 }}>
          <svg ref={ref} style={{ width: '100%', height: 'auto', display: 'block' }} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div className="panel" style={{ padding: 15 }}>
            <div style={{ color: C.gray, fontSize: 10.5, letterSpacing: '0.09em',
                          fontWeight: 700, marginBottom: 9 }}>ROBUSTNESS</div>
            <div className="num" style={{ fontSize: 30, fontWeight: 800, color: C.green }}>
              {((d.fan?.consensus_fraction ?? 0) * 100).toFixed(0)}%
            </div>
            <div style={{ color: C.gray, fontSize: 12.5, marginTop: 6, lineHeight: 1.6 }}>
              of {d.fan?.n_policies} sampled opponent policies give the same call
              {d.fan?.consensus_lap ? ` (lap ${d.fan.consensus_lap})` : ''}.
              Computed by re-solving, not asserted.
            </div>
          </div>
          <div className="panel" style={{ padding: 15 }}>
            <div style={{ color: C.gray, fontSize: 10.5, letterSpacing: '0.09em',
                          fontWeight: 700, marginBottom: 9 }}>ZONES ON THIS CIRCUIT</div>
            <table className="num" style={{ width: '100%', fontSize: 12, color: C.gray }}>
              <tbody>
                {(d.zone_models ?? []).map((z: any) => (
                  <tr key={z.name}>
                    <td style={{ padding: '3px 0' }}>zone {z.name}</td>
                    <td style={{ textAlign: 'right' }}>severity {z.severity}</td>
                    <td style={{ textAlign: 'right', color: C.white }}>
                      {z.dv_per_mj.toFixed(1)} m/s per MJ</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {sel && (
            <div className="panel" style={{ padding: 15 }}>
              <div style={{ color: C.gray, fontSize: 10.5, letterSpacing: '0.09em',
                            fontWeight: 700, marginBottom: 9 }}>LAP {sel.lap}</div>
              <table className="num" style={{ width: '100%', fontSize: 12.5, color: C.gray }}>
                <tbody>
                  <tr><td>opportunity q</td><td style={{ textAlign: 'right', color: C.white }}>{sel.q.toFixed(3)}</td></tr>
                  <tr><td>threshold τ</td><td style={{ textAlign: 'right', color: C.white }}>{sel.tau.toFixed(3)}</td></tr>
                  <tr><td>your energy</td><td style={{ textAlign: 'right', color: C.amber }}>{sel.own_mj.toFixed(2)} MJ</td></tr>
                  <tr><td>rival deployable</td><td style={{ textAlign: 'right', color: C.red }}>{sel.rival_mj.toFixed(2)} MJ</td></tr>
                  <tr><td>best zone</td><td style={{ textAlign: 'right', color: C.white }}>{sel.zone}</td></tr>
                </tbody>
              </table>
              <div style={{ marginTop: 10, fontSize: 12.5,
                            color: sel.attack ? C.green : C.gray }}>
                {sel.attack ? 'ATTACK — q clears the threshold' : 'HOLD — q is below the threshold'}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
