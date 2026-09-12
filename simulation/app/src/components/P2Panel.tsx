import type { P2Decision } from '../lib/api';
import { C } from '../lib/theme';

/**
 * P2 strategic recommendation — DISPLAY ONLY.
 *
 * Every number rendered here is read verbatim from the backend response. This
 * file contains no logistic, no threshold, no expected value, no Bellman step
 * and no ranking formula; `tests/test_frontend_p2.py` greps it to keep that
 * true. The only arithmetic below is unit presentation — x3.6 for km/h and
 * x100 for a percentage — and a UI-only sort of the candidate table, which
 * never changes a value, only the order rows appear in.
 */

const mj = (x: number | null | undefined, d = 3) =>
  x == null ? '—' : `${x.toFixed(d)} MJ`;
const ms = (x: number | null | undefined) =>
  x == null ? '—' : `${x.toFixed(2)} m/s (${(x * 3.6).toFixed(1)} km/h)`;
const pct = (x: number | null | undefined, d = 1) =>
  x == null ? '—' : `${(x * 100).toFixed(d)}%`;
const num = (x: number | null | undefined, d = 4) =>
  x == null ? '—' : x.toFixed(d);

function Row({ k, v, color }: { k: string; v: any; color?: string }) {
  return (
    <tr>
      <td style={{ color: C.gray, paddingRight: 10 }}>{k}</td>
      <td style={{ textAlign: 'right', color: color ?? C.white }}>{v}</td>
    </tr>
  );
}

function Section({ title, children }: { title: string; children: any }) {
  return (
    <div className="panel" style={{ padding: 15 }}>
      <div style={{ color: C.gray, fontSize: 10.5, letterSpacing: '0.09em',
                    fontWeight: 700, marginBottom: 9 }}>{title}</div>
      {children}
    </div>
  );
}

export function P2Panel({ p2 }: { p2: P2Decision | null }) {
  // P1-only responses, a disabled P2 path, or a failed fetch all land here.
  // The view must degrade, not crash: P2 fields are never assumed to exist.
  if (!p2) {
    return (
      <Section title="STRATEGIC OPTIMISER (P2)">
        <div style={{ color: C.gray, fontSize: 12.5, lineHeight: 1.6 }}>
          No P2 recommendation for this selection. The P1 decision trace above is
          unaffected.
        </div>
      </Section>
    );
  }

  const attack = p2.decision === 'ATTACK';
  const nb = p2.next_best_action;
  const synthetic = p2.pass_model_calibration !== 'empirical';

  // UI-ONLY sort: highest backend `value` first, nulls (infeasible) last. The
  // values themselves are the backend's; this only decides row order.
  const rows = [...(p2.candidate_actions ?? [])].sort(
    (a, b) => (b.value ?? -Infinity) - (a.value ?? -Infinity));
  const chosenLabel = attack
    ? `ATTACK(${p2.zone}, ${p2.deployment_budget_mj.toFixed(3)} MJ)`
    : 'HOLD';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <Section title="RECOMMENDATION">
        <div className="num" style={{ fontSize: 30, fontWeight: 800,
                                      color: attack ? C.green : C.amber }}>
          {p2.decision}{attack ? ` — ZONE ${p2.zone}` : ''}
        </div>
        <table className="num" style={{ width: '100%', fontSize: 12.5, marginTop: 8 }}>
          <tbody>
            <Row k="opportunity" v={p2.opportunity_id} />
            <Row k="lap" v={p2.lap} />
            <Row k="decision point" v={`${p2.decision_point_s.toFixed(1)} m`} />
            <Row k="decision time"
                 v={p2.decision_time_s == null ? '—' : `${p2.decision_time_s.toFixed(3)} s`} />
            <Row k="deployment budget" v={mj(p2.deployment_budget_mj)} color={C.amber} />
            <Row k="actually deployable" v={mj(p2.actual_deployed_mj)} color={C.amber} />
            <Row k="saturated"
                 v={p2.deployment_saturated ? `yes — ${p2.saturation_reason}` : 'no'} />
          </tbody>
        </table>
      </Section>

      <Section title="PREDICTED STATE">
        <table className="num" style={{ width: '100%', fontSize: 12.5 }}>
          <tbody>
            <Row k="own braking-point speed" v={ms(p2.predicted_own_speed_mps)} />
            <Row k="rival braking-point speed" v={ms(p2.predicted_rival_speed_mps)} />
            <Row k="delta v" v={ms(p2.predicted_delta_v_mps)} color={C.green} />
            <Row k="gap" v={p2.horizon?.[0]
              ? `${p2.horizon[0].gap_s.toFixed(3)} s` : '—'} />
            <Row k="P(pass)" v={pct(p2.pass_probability, 2)} />
          </tbody>
        </table>
        {/* Calibration status is rendered inline and unconditionally -- never
            behind a hover. A synthetic probability shown as a bare percentage
            reads as a measured one. */}
        <div style={{ marginTop: 10, padding: '7px 9px', borderRadius: 4,
                      background: synthetic ? 'rgba(201,162,39,0.14)' : 'rgba(0,160,90,0.14)',
                      color: synthetic ? C.amber : C.green,
                      fontSize: 11.5, fontWeight: 700, letterSpacing: '0.04em' }}>
          PASS MODEL: {p2.pass_model_calibration.toUpperCase()}
          {synthetic ? ' — NOT EMPIRICALLY CALIBRATED' : ''}
        </div>
        {synthetic && (
          <div style={{ color: C.gray, fontSize: 11.5, marginTop: 6, lineHeight: 1.55 }}>
            The logistic coefficients are synthetic design anchors, not a fit to
            race data. Treat this as a model output, not a real-world
            probability.
          </div>
        )}
      </Section>

      <Section title="STRATEGIC VALUE">
        <table className="num" style={{ width: '100%', fontSize: 12.5 }}>
          <tbody>
            <Row k="value of chosen action" v={num(p2.value_action)} />
            <Row k="value of holding" v={num(p2.value_hold)} />
            <Row k="decision margin" v={num(p2.decision_margin)}
                 color={p2.decision_margin < 0.01 ? C.amber : C.white} />
            <Row k="action consensus" v={pct(p2.action_consensus, 0)} />
            <Row k="expected regret" v={num(p2.expected_regret)} />
          </tbody>
        </table>
        {nb && (
          <div style={{ color: C.gray, fontSize: 12.5, marginTop: 9, lineHeight: 1.6 }}>
            Next best: <span style={{ color: C.white }}>
              {nb.kind}{nb.zone ? ` zone ${nb.zone}` : ''}
              {nb.kind === 'ATTACK' ? ` at ${nb.deployment_budget_mj.toFixed(3)} MJ` : ''}
            </span> — value {num(nb.value)}
            {p2.decision_margin < 0.01
              ? ' (a close call: the margin is small)' : ''}
          </div>
        )}
      </Section>

      <Section title="CANDIDATE ACTIONS">
        <div style={{ overflowX: 'auto' }}>
          <table className="num" style={{ width: '100%', fontSize: 11.5,
                                          borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: C.gray }}>
                <th style={{ textAlign: 'left' }}>action</th>
                <th style={{ textAlign: 'right' }}>req</th>
                <th style={{ textAlign: 'right' }}>deployed</th>
                <th style={{ textAlign: 'right' }}>sat</th>
                <th style={{ textAlign: 'right' }}>v own</th>
                <th style={{ textAlign: 'right' }}>v rival</th>
                <th style={{ textAlign: 'right' }}>Δv</th>
                <th style={{ textAlign: 'right' }}>P(pass)</th>
                <th style={{ textAlign: 'right' }}>value</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const isChosen = r.action === chosenLabel;
                return (
                  <tr key={r.action}
                      style={{ color: isChosen ? C.white : C.gray,
                               fontWeight: isChosen ? 700 : 400,
                               background: isChosen ? 'rgba(0,160,90,0.12)' : undefined }}>
                    <td style={{ textAlign: 'left' }}>
                      {isChosen ? '▸ ' : ''}{r.action}</td>
                    <td style={{ textAlign: 'right' }}>{r.requested_budget_mj.toFixed(3)}</td>
                    <td style={{ textAlign: 'right' }}>{r.actual_deployed_mj.toFixed(3)}</td>
                    <td style={{ textAlign: 'right' }}>{r.saturated ? 'yes' : '—'}</td>
                    <td style={{ textAlign: 'right' }}>{r.own_speed_mps.toFixed(2)}</td>
                    <td style={{ textAlign: 'right' }}>{r.rival_speed_mps.toFixed(2)}</td>
                    <td style={{ textAlign: 'right' }}>{r.delta_v_mps.toFixed(2)}</td>
                    <td style={{ textAlign: 'right' }}>{(r.pass_probability * 100).toFixed(2)}%</td>
                    <td style={{ textAlign: 'right' }}>
                      {r.value == null ? 'rejected' : r.value.toFixed(5)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div style={{ color: C.gray, fontSize: 11, marginTop: 7 }}>
          Energy in MJ, speeds in m/s. Rows ordered by backend value; the values
          themselves are the solver's.
        </div>
      </Section>

      <Section title="RIVAL-ENERGY SCENARIOS">
        <table className="num" style={{ width: '100%', fontSize: 11.5 }}>
          <thead>
            <tr style={{ color: C.gray }}>
              <th style={{ textAlign: 'left' }}>rival energy</th>
              <th style={{ textAlign: 'right' }}>weight</th>
              <th style={{ textAlign: 'left' }}>scenario optimum</th>
              <th style={{ textAlign: 'right' }}>V*</th>
              <th style={{ textAlign: 'right' }}>V(chosen)</th>
            </tr>
          </thead>
          <tbody>
            {(p2.policy_posterior ?? []).map((s) => (
              <tr key={s.scenario_index}
                  style={{ color: s.optimal_action === chosenLabel ? C.white : C.gray }}>
                <td style={{ textAlign: 'left' }}>{s.rival_usable_energy_mj.toFixed(3)} MJ</td>
                <td style={{ textAlign: 'right' }}>{(s.weight * 100).toFixed(0)}%</td>
                <td style={{ textAlign: 'left' }}>{s.optimal_action}</td>
                <td style={{ textAlign: 'right' }}>{s.optimal_value.toFixed(5)}</td>
                <td style={{ textAlign: 'right' }}>{s.chosen_action_value.toFixed(5)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ color: C.gray, fontSize: 11.5, marginTop: 8, lineHeight: 1.55 }}>
          Consensus {pct(p2.action_consensus, 0)} of belief mass · regret{' '}
          {num(p2.expected_regret)}. Consensus is a robustness statistic over the
          rival-energy belief, not a probability that the call is correct.
        </div>
      </Section>

      <Section title="INPUT PROVENANCE">
        <table className="num" style={{ width: '100%', fontSize: 11.5 }}>
          <tbody>
            {Object.entries(p2.input_confidence ?? {}).map(([k, v]) => (
              <Row key={k} k={k.replace(/_/g, ' ')}
                   v={typeof v === 'number' ? v.toFixed(3) : String(v ?? '—')} />
            ))}
            <Row k="horizon" v={`${p2.horizon?.length ?? 0} opportunities`} />
            <Row k="pit reset index"
                 v={p2.pit_reset_index == null ? 'none in horizon' : p2.pit_reset_index} />
            <Row k="version" v={p2.p2_version} />
          </tbody>
        </table>
      </Section>
    </div>
  );
}
