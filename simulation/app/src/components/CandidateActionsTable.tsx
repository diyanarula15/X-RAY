import type { P2Decision } from '../lib/api';
import { C } from '../lib/theme';

/**
 * The P2 candidate-action comparison — DISPLAY ONLY, extracted from
 * `P2Panel.tsx` so `Cockpit.tsx` can show the same table without a second
 * copy of it. Every number is read verbatim from the backend; the only
 * client-side choice is row order (`tests/test_frontend_p2.py` already
 * documents this sort as UI-only).
 */
export function CandidateActionsTable({ rows, chosenLabel }:
  { rows: P2Decision['candidate_actions']; chosenLabel: string }) {
  // UI-ONLY sort: highest backend `value` first, nulls (infeasible) last. The
  // values themselves are the backend's; this only decides row order.
  const sorted = [...(rows ?? [])].sort(
    (a, b) => (b.value ?? -Infinity) - (a.value ?? -Infinity));
  if (!sorted.length) {
    return (
      <div style={{ color: C.gray, fontSize: 11.5 }}>
        no candidate actions returned for this opportunity
      </div>
    );
  }
  return (
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
          {sorted.map((r) => {
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
      <div style={{ color: C.gray, fontSize: 11, marginTop: 7 }}>
        Energy in MJ, speeds in m/s. Rows ordered by backend value; the values
        themselves are the solver's.
      </div>
    </div>
  );
}
