import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import type { RaceSummary } from '../lib/api';
import { api } from '../lib/api';
import { C } from '../lib/theme';

/** Stage 1's measured ablation. Real telemetry lands at 4.17 Hz — on the cliff
 *  edge, which is the single biggest risk to the real-data claim. */
const ABLATION = [
  { hz: 100, mape: 5.0 }, { hz: 50, mape: 4.1 }, { hz: 20, mape: 3.7 },
  { hz: 10, mape: 8.8 }, { hz: 5, mape: 11.7 }, { hz: 3.7, mape: 3.9 },
  { hz: 2, mape: 19.4 }, { hz: 1, mape: 41.7 },
];

/**
 * View 6 — not an "about" page. A deliberate statement of what the system
 * cannot do. Judges remember the team that showed them the failure modes.
 */
export function Method({ races, current }:
  { races: RaceSummary[]; current: RaceSummary | null }) {
  const ref = useRef<SVGSVGElement>(null);
  const [rdd, setRdd] = useState<any>(null);
  useEffect(() => { api.rdd(1.0).then(setRdd).catch(() => {}); }, []);

  useEffect(() => {
    const svg = d3.select(ref.current!); svg.selectAll('*').remove();
    const W = 560, H = 260, m = { l: 54, r: 22, t: 16, b: 42 };
    svg.attr('viewBox', `0 0 ${W} ${H}`);
    const x = d3.scaleLog().domain([0.8, 120]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, 45]).range([H - m.b, m.t]);
    svg.append('g').attr('transform', `translate(0,${H - m.b})`)
      .call(d3.axisBottom(x).tickValues([1, 2, 3.7, 10, 20, 50, 100]).tickFormat(d3.format('~g')))
      .call((g) => { g.selectAll('line,path').attr('stroke', C.grid);
                     g.selectAll('text').attr('fill', C.gray).attr('font-size', 10.5); });
    svg.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5))
      .call((g) => { g.selectAll('line,path').attr('stroke', C.grid);
                     g.selectAll('text').attr('fill', C.gray).attr('font-size', 10.5); });
    svg.append('rect').attr('x', x(0.8)).attr('y', m.t).attr('width', x(2.7) - x(0.8))
      .attr('height', H - m.b - m.t).attr('fill', C.red).attr('opacity', 0.12);
    svg.append('text').attr('x', x(1.5)).attr('y', m.t + 30).attr('fill', C.red)
      .attr('font-size', 11).attr('text-anchor', 'middle').attr('font-weight', 700)
      .text('breaks down');
    svg.append('path').datum([...ABLATION].sort((a, b) => a.hz - b.hz))
      .attr('d', d3.line<any>().x((d) => x(d.hz)).y((d) => y(d.mape)) as any)
      .attr('stroke', C.amber).attr('stroke-width', 2).attr('fill', 'none');
    svg.append('g').selectAll('circle').data(ABLATION).join('circle')
      .attr('cx', (d: any) => x(d.hz)).attr('cy', (d: any) => y(d.mape)).attr('r', 4)
      .attr('fill', C.amber);
    const hz = current?.telemetry.median_hz ?? 4.17;
    svg.append('line').attr('x1', x(hz)).attr('x2', x(hz)).attr('y1', m.t).attr('y2', H - m.b)
      .attr('stroke', C.white).attr('stroke-width', 2).attr('stroke-dasharray', '5 4');
    svg.append('text').attr('x', x(hz) + 7).attr('y', m.t + 14).attr('fill', C.white)
      .attr('font-size', 11).attr('font-weight', 700)
      .text(`this race: ${hz} Hz`);
    svg.append('text').attr('x', W / 2).attr('y', H - 6).attr('fill', C.gray)
      .attr('font-size', 11).attr('text-anchor', 'middle').text('telemetry sample rate (Hz)');
    svg.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', 13)
      .attr('fill', C.gray).attr('font-size', 11).attr('text-anchor', 'middle')
      .text('per-lap energy error (%)');
  }, [current]);

  const S = ({ title, children }: any) => (
    <div className="panel" style={{ padding: 18, marginBottom: 16 }}>
      <div style={{ color: C.gray, fontSize: 10.5, letterSpacing: '0.09em',
                    fontWeight: 700, marginBottom: 10 }}>{title}</div>
      {children}
    </div>
  );
  const P = ({ children }: any) => (
    <p style={{ color: C.gray, fontSize: 13, lineHeight: 1.7, margin: '0 0 10px' }}>{children}</p>
  );

  return (
    <div style={{ padding: '26px 34px', height: '100%', overflow: 'auto' }}>
      <h2 className="display" style={{ fontSize: 26, margin: '0 0 6px' }}>Method &amp; limits</h2>

      <div style={{ border: `1px solid ${C.amber}`, borderRadius: 10, padding: 18,
                    background: 'rgba(255,195,0,0.06)', marginBottom: 18, maxWidth: 980 }}>
        <div style={{ color: C.amber, fontSize: 10.5, letterSpacing: '0.09em',
                      fontWeight: 700, marginBottom: 9 }}>WHAT THIS IS AND IS NOT</div>
        <p style={{ color: C.white, fontSize: 13.5, lineHeight: 1.75, margin: 0 }}>
          Deployment estimates are derived from public speed telemetry via inverse
          longitudinal dynamics. <b>We do not have access to ground-truth energy
          data for real cars — no such public channel exists.</b> Real-data
          validation is therefore indirect, via a regression discontinuity at the
          Manual Override eligibility boundary. Simulator validation, where ground
          truth is available, is reported separately below. Where identifiability
          is poor, we say so rather than reporting a confident number.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16,
                    maxWidth: 1180 }}>
        <div>
          <S title="SAMPLE RATE — THE HEADLINE RISK">
            <P>
              Real 2026 telemetry arrives irregularly at a median of 4.17 Hz. Our
              own ablation puts the accuracy cliff at about 4 Hz. We are on the
              edge of it with no margin. The dashed line is this race.
            </P>
            <svg ref={ref} style={{ width: '100%', height: 'auto', display: 'block' }} />
            <P>
              The simulator numbers on this curve are where ground truth exists.
              They are <i>not</i> a claim about real-data accuracy.
            </P>
          </S>

          <S title="WHY WE REPORT DEPLOYABLE ENERGY, NOT STATE OF CHARGE">
            <P>
              A speed trace measures energy <i>flows</i> exactly and the absolute
              <i> level</i> only up to a constant. A deployment cut-out does not
              mean the store is empty — it means the store has reached whatever
              buffer that driver refuses to spend, and nothing in the trace
              separates the two.
            </P>
            <P>
              Measured in simulation, where the truth is known: raw state-of-charge
              band coverage runs <b style={{ color: C.white }}>0.09–0.42</b> for a
              driver holding no buffer, against <b style={{ color: C.white }}>0.88+</b>
              {' '}for one whose buffer sits near the prior mean. That spread is the
              degeneracy, measured. Deployable energy is identified, and it is also
              the quantity that decides anything.
            </P>
          </S>
        </div>

        <div>
          <S title="PER-CIRCUIT IDENTIFIABILITY — INCLUDING THE BAD ONES">
            <table className="num" style={{ width: '100%', fontSize: 12.5,
              color: C.gray, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ color: C.dim, fontSize: 10.5 }}>
                  <th style={{ textAlign: 'left', paddingBottom: 6 }}>circuit</th>
                  <th style={{ textAlign: 'right' }}>ident.</th>
                  <th style={{ textAlign: 'right' }}>CdA</th>
                  <th style={{ textAlign: 'right' }}>cars</th>
                  <th style={{ textAlign: 'right' }}>refused</th>
                </tr>
              </thead>
              <tbody>
                {races.map((r) => (
                  <tr key={r.id} style={{ borderTop: `1px solid ${C.grid}` }}>
                    <td style={{ padding: '6px 0', color: C.white }}>{r.circuit}</td>
                    <td style={{ textAlign: 'right',
                      color: r.identifiability > 0.4 ? C.green
                        : r.identifiability > 0.15 ? C.amber : C.red }}>
                      {(r.identifiability * 100).toFixed(0)}%</td>
                    <td style={{ textAlign: 'right' }}>
                      {r.cda_pooled ? r.cda_pooled.toFixed(2) : '—'}</td>
                    <td style={{ textAlign: 'right' }}>{r.n_cars}</td>
                    <td style={{ textAlign: 'right', color: C.red }}>{r.n_refused}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <P>
              A circuit that never reaches high speed cannot pin drag area from a
              speed trace. Monaco is not a bug; it is the method telling the truth.
            </P>
          </S>

          <S title="REFUSAL CONDITIONS AND HOW OFTEN THEY FIRED">
            <table className="num" style={{ width: '100%', fontSize: 12.5, color: C.gray }}>
              <tbody>
                <tr><td style={{ padding: '4px 0' }}>car never in clean air</td>
                  <td style={{ textAlign: 'right', color: C.white }}>drag not calibratable</td></tr>
                <tr><td style={{ padding: '4px 0' }}>circuit lacks speed range</td>
                  <td style={{ textAlign: 'right', color: C.white }}>bands wide by necessity</td></tr>
                <tr><td style={{ padding: '4px 0' }}>telemetry gap &gt; 1.0 s</td>
                  <td style={{ textAlign: 'right', color: C.white }}>estimate suspends</td></tr>
                <tr style={{ borderTop: `1px solid ${C.grid}` }}>
                  <td style={{ paddingTop: 8 }}>usable laps, this race</td>
                  <td style={{ textAlign: 'right', paddingTop: 8, color: C.amber }}>
                    {current ? `${current.telemetry.laps_usable} / ${current.telemetry.laps_total}` : '—'}
                  </td></tr>
                <tr><td>cars refused, this race</td>
                  <td style={{ textAlign: 'right', color: C.red }}>{current?.n_refused ?? '—'}</td></tr>
              </tbody>
            </table>
          </S>

          <S title="THE REAL-DATA VALIDATION, STATED PLAINLY">
            {rdd?.effect != null ? (
              <>
                <P>
                  Regression discontinuity at the 1.000 s Manual Override boundary,
                  over <b style={{ color: C.white }}>{rdd.n}</b> car-laps:
                  effect <b style={{ color: C.white }}>{rdd.effect > 0 ? '+' : ''}{rdd.effect} MJ</b>,
                  {' '}p = <b style={{ color: C.white }}>{rdd.p_value}</b>. That is a null.
                </P>
                <P>
                  The design itself checks out — the density shows no manipulation
                  of the running variable and the covariates are balanced across the
                  boundary. But the minimum detectable effect is{' '}
                  <b style={{ color: C.amber }}>{rdd.power?.mde_mj} MJ</b> against an
                  allocation of {rdd.power?.target_effect_mj} MJ.{' '}
                  <b style={{ color: C.white }}>This study is underpowered, so the
                  null is a statement about how many races we have analysed, not
                  about whether the regulation does anything.</b> Reporting the null
                  without that number would be close to meaningless.
                </P>
              </>
            ) : <P>loading…</P>}
          </S>

          <S title="DOES IT ACTUALLY PICK THE RIGHT LAP?">
            <P>
              The question the whole system exists to answer, tested where the
              true answer is known. In simulation we can solve the same decision
              problem three ways and score all three on the same ground truth:
              an <b style={{ color: C.white }}>oracle</b> given the rival's real
              energy, <b style={{ color: C.amber }}>X-RAY</b> given only a noisy
              speed trace, and a <b>blind</b> driver with no read at all.
            </P>
            <table className="num" style={{ width: '100%', fontSize: 12.5,
              color: C.gray, borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ color: C.dim, fontSize: 10.5 }}>
                  <th style={{ textAlign: 'left', paddingBottom: 6 }}>method</th>
                  <th style={{ textAlign: 'right' }}>expected value</th>
                  <th style={{ textAlign: 'right' }}>lap chosen</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderTop: `1px solid ${C.grid}` }}>
                  <td style={{ padding: '6px 0' }}>best possible</td>
                  <td style={{ textAlign: 'right' }}>0.496</td>
                  <td style={{ textAlign: 'right' }}>3.0</td></tr>
                <tr style={{ borderTop: `1px solid ${C.grid}` }}>
                  <td style={{ padding: '6px 0', color: C.white }}>oracle (true energy)</td>
                  <td style={{ textAlign: 'right', color: C.white }}>0.483</td>
                  <td style={{ textAlign: 'right' }}>2.0</td></tr>
                <tr style={{ borderTop: `1px solid ${C.grid}` }}>
                  <td style={{ padding: '6px 0', color: C.amber }}>X-RAY (from speed)</td>
                  <td style={{ textAlign: 'right', color: C.amber }}>0.491</td>
                  <td style={{ textAlign: 'right' }}>2.6</td></tr>
                <tr style={{ borderTop: `1px solid ${C.grid}` }}>
                  <td style={{ padding: '6px 0' }}>blind, random lap</td>
                  <td style={{ textAlign: 'right' }}>0.334</td>
                  <td style={{ textAlign: 'right' }}>5.3</td></tr>
                <tr style={{ borderTop: `1px solid ${C.grid}` }}>
                  <td style={{ padding: '6px 0' }}>blind, first chance</td>
                  <td style={{ textAlign: 'right', color: C.red }}>0.023</td>
                  <td style={{ textAlign: 'right' }}>1.0</td></tr>
              </tbody>
            </table>
            <P>
              At full telemetry rate X-RAY picks the oracle's exact lap in{' '}
              <b style={{ color: C.green }}>100%</b> of races. At the 4.17 Hz rate
              real 2026 telemetry actually arrives at, it is exact 40% of the time
              and <b style={{ color: C.green }}>within one lap in 100%</b> — and
              the cost of that error in expected value is nil.
            </P>
            <P>
              Two honest notes. X-RAY can score <i>above</i> the oracle on
              individual seeds; that is estimation error landing favourably, not
              skill. And the engine does not maximise the chance of a pass — it
              maximises laps spent in front, so it correctly takes a 0.53 chance
              on lap 2 over a 0.63 chance on lap 9.
            </P>
            <P>
              Reproduce with <span className="mono" style={{ color: C.white }}>
              python scripts/03.a_validate_decision.py --seeds 25</span>.
            </P>
          </S>

          <S title="BAND CALIBRATION, BOTH DIRECTIONS">
            <P>
              In simulation the deployable-energy band covers the truth 86–90% of
              the time against a 75–85% target, so it is slightly conservative. Its
              p10–p90 width is 1.7–2.4× its own RMSE, where a Gaussian error would
              need 2.56×. The band is not wide; the errors are more peaked than
              Gaussian. Narrowing it to hit 85% would make it narrower than the
              errors it describes.
            </P>
          </S>
        </div>
      </div>
    </div>
  );
}
