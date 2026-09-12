import { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { api, type Car, type RaceDetail, type RaceSummary } from './lib/api';
import { C } from './lib/theme';
import { usePlayback, type View } from './store/playback';
import { bestBattleTime, sampleCar, timeRange } from './lib/carState';
import { Theatre } from './views/Theatre';
import { Observability } from './views/Observability';
import { RDD } from './views/RDD';
import { Decision } from './views/Decision';
import { Fingerprint } from './views/Fingerprint';
import { Method } from './views/Method';
import { GUIDE, HelpButton, Onboarding } from './components/Explain';
import { useDemo, DEMO_BEATS } from './lib/demo';

const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: 'theatre', label: 'Race theatre', hint: 'the battle, replayed' },
  { id: 'observability', label: 'Observability', hint: 'what can be known where' },
  { id: 'rdd', label: 'RDD explorer', hint: 'try to break it' },
  { id: 'decision', label: 'Decision', hint: 'why the engine says what it says' },
  { id: 'fingerprint', label: 'Fingerprint', hint: 'deployment style' },
  { id: 'method', label: 'Method & limits', hint: 'what it cannot do' },
];

export default function App() {
  const [races, setRaces] = useState<RaceSummary[]>([]);
  const [race, setRace] = useState<RaceDetail | null>(null);
  const [subject, setSubject] = useState<Car | null>(null);
  const [rival, setRival] = useState<Car | null>(null);
  const [obs, setObs] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const view = usePlayback((s) => s.view);
  const setView = usePlayback((s) => s.setView);
  const lite = usePlayback((s) => s.lite);
  const setLite = usePlayback((s) => s.setLite);
  const raceTime = usePlayback((s) => s.raceTime);
  const demo = usePlayback((s) => s.demo);
  const beat = usePlayback((s) => s.demoBeat);
  useDemo();

  useEffect(() => {
    api.races().then((r) => {
      setRaces(r);
      // preload the hero race on mount: no loading spinner in the demo path
      const hero = r.find((x) => x.calibratable && x.identifiability > 0.3) ?? r[0];
      if (hero) load(hero.id);
    }).catch((e) => setErr(String(e)));
  }, []);

  async function load(id: string) {
    setErr(null);
    const d = await api.race(id);
    setRace(d);
    usePlayback.getState().setRace(id);
    // Open on a pair that actually raced each other, not on whoever sorts
    // first alphabetically half a lap apart.
    const b = (d.battles ?? [])[0];
    const ranked = d.drivers.slice().sort((a, b2) => a.localeCompare(b2));
    const [s, r] = b ? [b.car, b.ahead] : [ranked[0], ranked[1]];
    usePlayback.getState().setCars(s, r);
    const [sc, rc, ob] = await Promise.all([
      api.car(id, s).catch(() => null),
      api.car(id, r).catch(() => null),
      api.observability(id).catch(() => null),
    ]);
    setSubject(sc); setRival(rc); setObs(ob);
    // The shared clock spans the window where BOTH cars have telemetry, so the
    // two are always sampled at the same moment of the same race.
    const ra = timeRange(sc), rb = timeRange(rc);
    const range: [number, number] | null = ra && rb
      ? [Math.max(ra[0], rb[0]), Math.min(ra[1], rb[1])] : (ra ?? null);
    if (range) {
      usePlayback.getState().setRange(range);
      // open on the moment they were actually closest, not on a lap boundary
      const t = bestBattleTime(sc, rc, d.circuit_geometry.length, range);
      if (t != null) usePlayback.getState().setTime(t);
    }
  }

  async function pick(which: 'subject' | 'rival', drv: string) {
    if (!race) return;
    const c = await api.car(race.id, drv).catch(() => null);
    if (which === 'subject') { setSubject(c); usePlayback.getState().setCars(drv, rival?.driver ?? ''); }
    else { setRival(c); usePlayback.getState().setCars(subject?.driver ?? '', drv); }
  }

  const current = races.find((r) => r.id === race?.id) ?? null;

  if (err) return <Center>API unreachable — start it with
    <code style={{ color: C.amber }}> uvicorn api.main:app --port 8011</code>. {err}</Center>;
  if (!race) return <Center>loading X-RAY…</Center>;

  return (
    <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column' }}>
      <Onboarding />
      {/* nav */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '9px 16px',
        borderBottom: `1px solid ${C.panelBorder}`, background: 'rgba(11,11,15,.9)',
        zIndex: 20, flexShrink: 0 }}>
        <div className="display" style={{ fontSize: 19, marginRight: 16, letterSpacing: '-0.03em' }}>
          X-RAY
        </div>
        {VIEWS.map((v) => (
          <button key={v.id} onClick={() => setView(v.id)} title={v.hint}
            style={{ padding: '7px 12px', fontSize: 12.5, borderRadius: 7,
              border: '1px solid transparent', background: view === v.id ? C.panel : 'transparent',
              borderColor: view === v.id ? C.panelBorder : 'transparent',
              color: view === v.id ? C.white : C.gray }}>
            {v.label}
          </button>
        ))}
        <div style={{ flex: 1 }} />
        <select value={race.id} onChange={(e) => load(e.target.value)}
          style={{ background: C.panel, color: C.white, border: `1px solid ${C.panelBorder}`,
            borderRadius: 7, padding: '6px 9px', fontSize: 12 }}>
          {races.map((r) => (
            <option key={r.id} value={r.id}>
              {r.circuit} — {(r.identifiability * 100).toFixed(0)}% identifiable
            </option>
          ))}
        </select>
        {['subject', 'rival'].map((w) => (
          <select key={w} value={(w === 'subject' ? subject : rival)?.driver ?? ''}
            onChange={(e) => pick(w as any, e.target.value)}
            style={{ background: C.panel, color: w === 'subject' ? C.amber : C.red,
              border: `1px solid ${C.panelBorder}`, borderRadius: 7, padding: '6px 9px',
              fontSize: 12 }}>
            {race.drivers.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        ))}
        <button onClick={() => usePlayback.getState().startDemo(
          bestBattleTime(subject, rival, race.circuit_geometry.length,
            usePlayback.getState().tRange) ?? undefined)}
          style={{ padding: '7px 12px', fontSize: 12, borderRadius: 7,
            border: `1px solid ${demo ? C.green : C.panelBorder}`, background: 'transparent',
            color: demo ? C.green : C.gray }}>
          {demo ? `demo · ${DEMO_BEATS[beat]?.label ?? ''}` : 'run demo'}
        </button>
        <HelpButton view={view} />
        <button onClick={() => setLite(!lite)}
          style={{ padding: '7px 10px', fontSize: 11.5, borderRadius: 7,
            border: `1px solid ${lite ? C.amber : C.panelBorder}`, background: 'transparent',
            color: lite ? C.amber : C.dim }}>lite</button>
      </div>

      <div style={{ padding: '7px 16px', borderBottom: `1px solid ${C.panelBorder}`,
        background: 'rgba(11,11,15,.75)', color: C.gray, fontSize: 12, zIndex: 19,
        display: 'flex', gap: 10, alignItems: 'baseline' }}>
        <b style={{ color: C.white }}>{GUIDE[view].title}</b>
        <span style={{ opacity: 0.85 }}>{GUIDE[view].what.split('. ')[0]}.</span>
        <span style={{ marginLeft: 'auto', color: C.dim }}>
          press <b style={{ color: C.gray }}>?</b> in the bar for what to try here
        </span>
      </div>

      {/* low-identifiability banner: loud, not buried */}
      {current && !current.calibratable && (
        <div style={{ padding: '9px 16px', background: 'rgba(255,195,0,.1)',
          borderBottom: `1px solid ${C.amber}`, color: C.amber, fontSize: 12.5, zIndex: 20 }}>
          Low identifiability at {current.circuit} — no car ran fast enough to pin
          drag area from its own trace. Bands here are wide by necessity, and the
          estimator says so rather than guessing.
        </div>
      )}

      <div style={{ position: 'relative', flex: 1, minHeight: 0 }}>
        <AnimatePresence mode="wait">
          <motion.div key={view} initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            exit={{ opacity: 0 }} transition={{ duration: 0.18 }}
            style={{ position: 'absolute', inset: 0 }}>
            {view === 'theatre' && <Theatre race={race} subject={subject} rival={rival} obs={obs} />}
            {view === 'observability' && <Observability data={obs} />}
            {view === 'rdd' && <RDD />}
            {view === 'decision' && subject && rival &&
              <Decision raceId={race.id} car={subject.driver} rival={rival.driver} />}
            {view === 'fingerprint' &&
              <Fingerprint cars={[subject, rival]} upto={
                subject ? (sampleCar(subject, raceTime,
                  race.circuit_geometry.length)?.lap ?? 9999) : 9999} />}
            {view === 'method' && <Method races={races} current={current} />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

function Center({ children }: any) {
  return (
    <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center',
      color: C.gray, fontSize: 14, textAlign: 'center', padding: 40 }}>
      <div>{children}</div>
    </div>
  );
}
