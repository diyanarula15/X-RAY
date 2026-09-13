import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { C } from '../lib/theme';

/**
 * View 2 — the novel one. Colour the circuit by how much the estimator can
 * learn at each point, so partial observability stops being an abstraction and
 * becomes a picture of a racetrack.
 *
 * Two learnable things, and they are not the same thing:
 *   deployment information — where the trace tells you about energy
 *   nuisance information   — where it tells you about drag instead
 * High-speed running informs the nuisances, not the deployment, so it gets its
 * own colour rather than being called "good".
 */
export function Observability({ data }: { data: any }) {
  const ref = useRef<SVGSVGElement>(null);
  const [layer, setLayer] = useState<'deployment' | 'nuisance'>('deployment');
  const [hover, setHover] = useState<any>(null);

  useEffect(() => {
    if (!data?.geometry?.x?.length) return;
    const svg = d3.select(ref.current!);
    svg.selectAll('*').remove();
    const W = 900, H = 560;
    svg.attr('viewBox', `0 0 ${W} ${H}`)
       .attr('preserveAspectRatio', 'xMidYMid meet');

    const gx = data.geometry.x as number[], gy = data.geometry.y as number[];
    // keep the circuit's true aspect ratio: a squashed Spa is a different circuit
    const [x0, x1] = d3.extent(gx) as [number, number];
    const [y0, y1] = d3.extent(gy) as [number, number];
    const pad = 46;
    const k = Math.min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (y1 - y0));
    const cxm = (x0 + x1) / 2, cym = (y0 + y1) / 2;
    const sx = (v: number) => W / 2 + (v - cxm) * k;
    const sy = (v: number) => H / 2 - (v - cym) * k;
    const L = data.geometry.length as number;

    const field: number[] = layer === 'deployment' ? data.deployment_info : data.nuisance_info;
    const os = data.s as number[];
    const colour = layer === 'deployment'
      ? d3.scaleLinear<string>().domain([0, 0.35, 1]).range([C.red, '#C9A227', C.green]).clamp(true)
      : d3.scaleLinear<string>().domain([0, 1]).range(['#1B2B3A', '#5AC8FA']).clamp(true);

    const refusals: any[] = data.refusal_zones ?? [];
    const g = svg.append('g');
    for (let i = 0; i < gx.length - 1; i++) {
      const s = (i / gx.length) * L;
      const k = Math.min(Math.floor((s / L) * os.length), os.length - 1);
      const val = field?.[k] ?? 0;
      const refused = refusals.some((r) => s >= r.s0 && s <= r.s1);
      g.append('line')
        .attr('x1', sx(gx[i])).attr('y1', sy(gy[i]))
        .attr('x2', sx(gx[i + 1])).attr('y2', sy(gy[i + 1]))
        .attr('stroke', refused ? C.red : colour(val))
        .attr('stroke-width', refused ? 8 : 6.5)
        .attr('stroke-linecap', 'round')
        .attr('opacity', refused ? 0.95 : 1)
        .attr('stroke-dasharray', refused ? '5 5' : null)
        .on('mouseenter', () => setHover({
          s: Math.round(s), val, refused,
          reason: refusals.find((r) => s >= r.s0 && s <= r.s1)?.reason,
          speed: data.speed?.[k],
        }))
        .on('mouseleave', () => setHover(null));
    }
    svg.append('circle').attr('cx', sx(gx[0])).attr('cy', sy(gy[0])).attr('r', 5)
      .attr('fill', C.white);
    svg.append('text').attr('x', sx(gx[0]) + 9).attr('y', sy(gy[0]) + 4)
      .attr('fill', C.gray).attr('font-size', 11).text('start / finish');
  }, [data, layer]);

  // Standalone tab now that "Race context" lost its RDD sub-tab, so a missing
  // payload has to say so. It previously rendered its full chrome around an
  // empty SVG, which reads as "this circuit has no observability" rather than
  // "the request failed".
  if (!data?.geometry?.x?.length) {
    return (
      <div style={{ padding: 34, color: C.amber, fontSize: 13, lineHeight: 1.7,
                    maxWidth: 640 }}>
        No observability map for this race — the circuit geometry or the
        per-point information field is missing from the artefact.
      </div>
    );
  }

  return (
    <div style={{ padding: '26px 34px', height: '100%', overflow: 'auto' }}>
      <h2 className="display" style={{ fontSize: 26, margin: 0 }}>Observability map</h2>
      <p style={{ color: C.gray, maxWidth: 780, fontSize: 13.5, lineHeight: 1.65 }}>
        Where on this circuit can the estimator actually learn something? Green is
        informative about deployment, red is not. This is the same fact as the
        breathing band, drawn on the ground: the band tightens where the track is
        green and goes slack where it is red.
      </p>
      <div style={{ display: 'flex', gap: 8, margin: '14px 0' }}>
        {(['deployment', 'nuisance'] as const).map((l) => (
          <button key={l} onClick={() => setLayer(l)}
            style={{ padding: '7px 13px', borderRadius: 7, fontSize: 12,
                     border: `1px solid ${layer === l ? C.amber : C.panelBorder}`,
                     background: 'transparent', color: layer === l ? C.amber : C.gray }}>
            {l === 'deployment' ? 'deployment information' : 'nuisance (drag) information'}
          </button>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 300px', gap: 20,
                    alignItems: 'start' }}>
        <div className="panel" style={{ padding: 10 }}>
          <svg ref={ref} style={{ width: '100%', height: 'auto', display: 'block' }} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div className="panel" style={{ padding: 14 }}>
            <div style={{ color: C.gray, fontSize: 10.5, letterSpacing: '0.09em',
                          fontWeight: 700, marginBottom: 9 }}>READING THE MAP</div>
            {layer === 'deployment' ? (
              <div style={{ fontSize: 12.5, lineHeight: 1.65, color: C.gray }}>
                <b style={{ color: C.green }}>Green</b> — the car is on power with
                headroom under the regulatory ceiling, so a change in deployment
                moves the speed trace and we can see it.<br /><br />
                <b style={{ color: C.red }}>Red</b> — corner-limited or off the
                power. Deployment could be anything and the trace would look the
                same.<br /><br />
                <b style={{ color: C.red }}>Red hatching</b> — the estimator
                refuses here.
              </div>
            ) : (
              <div style={{ fontSize: 12.5, lineHeight: 1.65, color: C.gray }}>
                Brightness is <b>|∂P/∂CdA|</b>, which grows as v³. This is what
                high-speed running actually buys you: it pins the nuisances, not
                the deployment. The two maps are almost complementary, which is
                the whole difficulty of the problem in one picture.
              </div>
            )}
          </div>
          {hover && (
            <div className="panel" style={{ padding: 13 }}>
              <div className="num" style={{ fontSize: 12.5, color: C.white }}>
                {hover.s} m into the lap
              </div>
              <div className="num" style={{ color: C.gray, fontSize: 12, marginTop: 4 }}>
                score {(hover.val ?? 0).toFixed(3)}
                {hover.speed ? ` · ${(hover.speed * 3.6).toFixed(0)} km/h` : ''}
              </div>
              {hover.refused && (
                <div style={{ color: C.red, fontSize: 12, marginTop: 8, lineHeight: 1.5 }}>
                  {hover.reason}
                </div>
              )}
            </div>
          )}
          <div className="panel" style={{ padding: 14 }}>
            <div style={{ color: C.gray, fontSize: 10.5, letterSpacing: '0.09em',
                          fontWeight: 700, marginBottom: 8 }}>REFUSAL ZONES</div>
            <div className="num" style={{ fontSize: 22, color: C.red }}>
              {(data?.refusal_zones ?? []).length}
            </div>
            <div style={{ color: C.dim, fontSize: 11.5, marginTop: 4 }}>
              stretches where the estimator declines to answer
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
