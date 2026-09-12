import { useEffect, useState } from 'react';
import type { P3Status } from '../lib/api';
import { api } from '../lib/api';
import { ValidationPanel, ModelStatusPanel, DataQualityPanel } from '../components/P3Panel';
import { C } from '../lib/theme';

/**
 * View — the model/evidence registry. Extracted from "Method & limits" so the
 * per-component status table (energy inference, pass model, tyres, CdA, ClA,
 * wake, wetness, wind, rival pit, regulation variant, mass model, deployment
 * ceiling) and the direct/indirect validation results have their own tab,
 * rather than sharing one long scroll with the sample-rate/oracle-vs-blind
 * narrative that now lives in Method & limits.
 */
export function Evidence() {
  const [p3, setP3] = useState<P3Status | null>(null);
  useEffect(() => { api.p3Status().then(setP3).catch(() => setP3(null)); }, []);

  return (
    <div style={{ padding: '26px 34px', height: '100%', overflow: 'auto' }}>
      <h2 className="display" style={{ fontSize: 26, margin: '0 0 6px' }}>Evidence</h2>
      <p style={{ color: C.gray, maxWidth: 900, fontSize: 13.5, lineHeight: 1.65,
                  marginBottom: 18 }}>
        What each model is, and what is known about it. The negative held-out
        result is the strongest statement the project can make about itself,
        so it sits first, at the same weight as everything else.
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, maxWidth: 1180 }}>
        <DataQualityPanel p3={p3} />
        <ValidationPanel p3={p3} />
        <ModelStatusPanel p3={p3} />
      </div>
    </div>
  );
}
