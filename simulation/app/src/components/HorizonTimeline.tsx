import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import type { P2Decision } from '../lib/api';
import { C } from '../lib/theme';

/**
 * The P2 opportunity horizon, visualized — DISPLAY ONLY.
 *
 * `p2.horizon[]` is the only source. Opportunity 0 is causally observed (it
 * has a real `decision_time_s`); every later entry is a forecast
 * (`decision_time_s === null`) and is drawn distinctly so a forecast is never
 * mistaken for an observed fact. No value here is computed — only laid out.
 */
export function HorizonTimeline({ horizon, pitResetIndex }: {
  horizon: P2Decision['horizon']; pitResetIndex: number | null;
}) {
  const ref = useRef<SVGSVGElement>(null);
  const rows = horizon ?? [];

  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current); svg.selectAll('*').remove();
    if (!rows.length) return;
    const W = 820, H = 150, m = { l: 50, r: 20, t: 14, b: 30 };
    svg.attr('viewBox', `0 0 ${W} ${H}`);
    const x = d3.scalePoint<number>()
      .domain(rows.map((_, i) => i)).range([m.l, W - m.r]).padding(0.5);
    const maxE = d3.max(rows, (r) => Math.max(r.own_usable_energy_mj, r.rival_usable_energy_mj)) ?? 1;
    const y = d3.scaleLinear().domain([0, maxE * 1.1]).range([H - m.b, m.t]);

    svg.append('g').attr('transform', `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).tickFormat((_, i) => `L${rows[i]?.lap ?? ''}`))
      .call((g) => { g.selectAll('line,path').attr('stroke', C.grid);
                     g.selectAll('text').attr('fill', C.gray).attr('font-size', 10); });
    svg.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(4))
      .call((g) => { g.selectAll('line,path').attr('stroke', C.grid);
                     g.selectAll('text').attr('fill', C.gray).attr('font-size', 10); });
    svg.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', 12)
      .attr('fill', C.gray).attr('font-size', 10.5).attr('text-anchor', 'middle')
      .text('MJ');

    if (pitResetIndex != null && rows[pitResetIndex]) {
      const px = x(pitResetIndex)!;
      svg.append('line').attr('x1', px).attr('x2', px).attr('y1', m.t).attr('y2', H - m.b)
        .attr('stroke', C.amber).attr('stroke-width', 1.5).attr('stroke-dasharray', '4 3');
      svg.append('text').attr('x', px + 4).attr('y', m.t + 10).attr('fill', C.amber)
        .attr('font-size', 9.5).text('pit reset');
    }

    const line = (key: 'own_usable_energy_mj' | 'rival_usable_energy_mj', colour: string) =>
      svg.append('path').datum(rows)
        .attr('d', d3.line<any>().x((_, i) => x(i)!).y((r) => y(r[key])) as any)
        .attr('stroke', colour).attr('stroke-width', 2).attr('fill', 'none');
    line('own_usable_energy_mj', C.amber);
    line('rival_usable_energy_mj', C.red);

    svg.append('g').selectAll('circle').data(rows).join('circle')
      .attr('cx', (_, i) => x(i)!).attr('cy', (r: any) => y(r.own_usable_energy_mj))
      .attr('r', (_, i) => (i === 0 ? 5 : 3.5))
      .attr('fill', C.amber).attr('stroke', C.bg).attr('stroke-width', 1.5);
    svg.append('g').selectAll('circle').data(rows).join('circle')
      .attr('cx', (_, i) => x(i)!).attr('cy', (r: any) => y(r.rival_usable_energy_mj))
      .attr('r', (_, i) => (i === 0 ? 5 : 3.5))
      .attr('fill', C.red).attr('stroke', C.bg).attr('stroke-width', 1.5)
      // forecasts (no real decision_time_s) are drawn hollow -- a forecast
      // must never look identical to the one observed opportunity.
      .attr('fill-opacity', (r: any) => (r.decision_time_s == null ? 0.35 : 1));
  }, [rows, pitResetIndex]);

  if (!rows.length) {
    return (
      <div style={{ color: C.gray, fontSize: 11.5, padding: '8px 0' }}>
        single-opportunity horizon; no forward view
      </div>
    );
  }
  return (
    <div>
      <svg ref={ref} style={{ width: '100%', height: 150, display: 'block' }} />
      <div style={{ color: C.gray, fontSize: 10.5, marginTop: 4, lineHeight: 1.5 }}>
        solid dot · opportunity 0, observed. faint dot · later opportunities,
        forecast — not yet decided or observed.
      </div>
    </div>
  );
}
