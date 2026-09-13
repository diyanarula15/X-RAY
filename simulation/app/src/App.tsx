import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { api, type Car, type RaceDetail, type RaceSummary } from './lib/api';
import { C } from './lib/theme';
import { usePlayback, type View } from './store/playback';
import { bestBattleTime, timeRange } from './lib/carState';
import { Theatre } from './views/Theatre';
import { Observability } from './views/Observability';
import { Situations } from './views/Situations';
import { Cockpit } from './views/Cockpit';
import { Evidence } from './views/Evidence';
import { GUIDE, HelpButton, Onboarding } from './components/Explain';
import { useDemo, DEMO_BEATS } from './lib/demo';

const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: 'cockpit', label: 'Cockpit', hint: 'attack or hold, and the horizon behind it' },
  { id: 'situations', label: 'Situations', hint: 'pick a real decision point' },
  { id: 'replay', label: 'Replay', hint: 'the battle, replayed in 3D' },
  { id: 'observability', label: 'Observability', hint: 'where the estimator can learn' },
  { id: 'evidence', label: 'Evidence & limits', hint: 'what it cannot do' },
];

export default function App() {
  const [races, setRaces] = useState<RaceSummary[]>([]);
  const [race, setRace] = useState<RaceDetail | null>(null);
  const [subject, setSubject] = useState<Car | null>(null);
  const [rival, setRival] = useState<Car | null>(null);
  const [obs, setObs] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [pairErr, setPairErr] = useState<string | null>(null);
  const [pairBusy, setPairBusy] = useState(false);
  const [showAnalyse, setShowAnalyse] = useState(false);
  const [analyzeRound, setAnalyzeRound] = useState('');
  const [analyzeYear, setAnalyzeYear] = useState('2026');
  const [analyzeJob, setAnalyzeJob] =
    useState<{ id: string; status: string; message?: string } | null>(null);
  const view = usePlayback((s) => s.view);
  const setView = usePlayback((s) => s.setView);
  const lite = usePlayback((s) => s.lite);
  const setLite = usePlayback((s) => s.setLite);
  const demo = usePlayback((s) => s.demo);
  const beat = usePlayback((s) => s.demoBeat);
  // One monotonic token guards every async pair swap. Without it two quick
  // driver changes could resolve out of order and leave the losing pair on
  // screen with the winning pair's labels in the picker.
  const pairToken = useRef(0);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useDemo();

  useEffect(() => {
    api.races().then((r) => {
      setRaces(r);
      // preload the hero race on mount: no loading spinner in the demo path
      const hero = r.find((x) => x.calibratable && x.identifiability > 0.3) ?? r[0];
      if (hero) load(hero.id).catch((e) => setErr(String(e)));
    }).catch((e) => setErr(String(e)));
    // The analyse poller is a chained setTimeout, so it survives unmount unless
    // it is explicitly cancelled. It used to leak a chain per run.
    return () => { if (pollTimer.current) clearTimeout(pollTimer.current); };
  }, []);

  /**
   * Load both cars of a pair and re-derive everything that depends on them.
   *
   * This exists because `pick()` used to set the car objects and stop there,
   * leaving `tRange` and `raceTime` belonging to the PREVIOUS pair. `sampleCar`
   * reports `onTrack: false` outside a car's own observed window, so after a
   * swap the newly-chosen car sat frozen or absent in Replay, the gap readout
   * went null and the energy bar read "—". That is what "choosing drivers
   * doesn't work" was.
   */
  async function applyPair(raceId: string, geoLength: number,
                           subjectCode: string, rivalCode: string) {
    const mine = ++pairToken.current;
    setPairBusy(true); setPairErr(null);
    try {
      const [sc, rc] = await Promise.all([
        api.car(raceId, subjectCode).catch(() => null),
        api.car(raceId, rivalCode).catch(() => null),
      ]);
      if (pairToken.current !== mine) return;
      setSubject(sc); setRival(rc);
      usePlayback.getState().setCars(subjectCode, rivalCode);
      if (!sc || !rc) {
        // A car the estimator refused is a legitimate outcome, not an error to
        // swallow into a blank screen -- which is what the old `.catch(() =>
        // null)` produced, because every tab was guarded on `subject && rival`.
        const missing = [!sc && subjectCode, !rc && rivalCode].filter(Boolean).join(' and ');
        setPairErr(`${missing} not available for this race.`);
        return;
      }
      // The shared clock spans the window where BOTH cars have telemetry, so the
      // two are always sampled at the same moment of the same race.
      const ra = timeRange(sc), rb = timeRange(rc);
      const range: [number, number] | null = ra && rb
        ? [Math.max(ra[0], rb[0]), Math.min(ra[1], rb[1])] : (ra ?? null);
      if (range && range[1] > range[0]) {
        usePlayback.getState().setRange(range);
        // open on the moment they were actually closest, not on a lap boundary
        const t = bestBattleTime(sc, rc, geoLength, range);
        if (t != null) usePlayback.getState().setTime(t);
      } else {
        setPairErr(`${subjectCode} and ${rivalCode} have no overlapping telemetry `
                   + 'window in this race — they cannot be replayed against each other.');
      }
    } finally {
      if (pairToken.current === mine) setPairBusy(false);
    }
  }

  async function load(id: string) {
    setErr(null);
    const d = await api.race(id);
    setRace(d);
    usePlayback.getState().setRace(id);
    api.observability(id).then(setObs).catch(() => setObs(null));
    // Open on a pair that actually raced each other, not on whoever sorts
    // first alphabetically half a lap apart.
    const b = (d.battles ?? [])[0];
    const ranked = d.drivers.slice().sort((a, b2) => a.localeCompare(b2));
    const [s, r] = b ? [b.car, b.ahead] : [ranked[0], ranked[1]];
    if (!s || !r) { setPairErr('This race has fewer than two analysed cars.'); return; }
    await applyPair(id, d.circuit_geometry.length, s, r);
  }

  function pick(which: 'subject' | 'rival', drv: string) {
    if (!race || !drv) return;
    const s = which === 'subject' ? drv : (subject?.driver ?? '');
    const r = which === 'rival' ? drv : (rival?.driver ?? '');
    if (!s || !r || s === r) return;
    void applyPair(race.id, race.circuit_geometry.length, s, r);
  }

  async function runAnalysis() {
    const round = parseInt(analyzeRound, 10);
    const year = parseInt(analyzeYear, 10);
    if (!Number.isFinite(round)) {
      setAnalyzeJob({ id: '', status: 'error', message: 'enter a round number' }); return;
    }
    if (!Number.isFinite(year)) {
      setAnalyzeJob({ id: '', status: 'error', message: 'enter a season year' }); return;
    }
    let job_id: string;
    try {
      ({ job_id } = await api.analyzeRace(round, year));
    } catch (e) {
      setAnalyzeJob({ id: '', status: 'error', message: String(e) });
      return;
    }
    setAnalyzeJob({ id: job_id, status: 'running' });
    const poll = async () => {
      const j = await api.analyzeStatus(job_id)
        .catch((e) => ({ status: 'error' as const, message: String(e) }));
      if (j.status === 'running') {
        pollTimer.current = setTimeout(poll, 3000);
        return;
      }
      if (j.status === 'done') {
        setAnalyzeJob({ id: job_id, status: 'done', message: j.race_id });
        setRaces(await api.races().catch(() => races));
        await load(j.race_id).catch((e) => setErr(String(e)));
      } else {
        setAnalyzeJob({ id: job_id, status: 'error', message: (j as any).message });
      }
    };
    pollTimer.current = setTimeout(poll, 3000);
  }

  const current = races.find((r) => r.id === race?.id) ?? null;

  if (err) return <Center>API unreachable — start it with
    <code style={{ color: C.amber }}> uvicorn simulation.api.main:app --port 8011</code>. {err}</Center>;
  if (!race) return <Center>loading X-RAY…</Center>;

  const btn = (on: boolean) => ({
    padding: '7px 12px', fontSize: 12, borderRadius: 7,
    border: `1px solid ${on ? C.amber : C.panelBorder}`,
    background: 'transparent', color: on ? C.amber : C.gray,
  });
  const lowIdent = current && !current.calibratable;

  return (
    <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column' }}>
      <Onboarding />

      {/* nav — five tabs and the two pickers, nothing else. The analyse
          controls moved into a popover off the race picker; at 1366 px this row
          used to hold fourteen controls in a non-wrapping flex and overran. */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '9px 16px',
        borderBottom: `1px solid ${C.panelBorder}`, background: 'rgba(11,11,15,.9)',
        zIndex: 20, flexShrink: 0, flexWrap: 'wrap' }}>
        <div className="display" style={{ fontSize: 19, marginRight: 14,
                                          letterSpacing: '-0.03em' }}>X-RAY</div>
        {VIEWS.map((v) => (
          <button key={v.id} onClick={() => setView(v.id)} title={v.hint}
            style={{ padding: '7px 12px', fontSize: 12, borderRadius: 7,
              border: '1px solid transparent', background: view === v.id ? C.panel : 'transparent',
              borderColor: view === v.id ? C.panelBorder : 'transparent',
              color: view === v.id ? C.white : C.gray }}>
            {v.label}
          </button>
        ))}
        <div style={{ flex: 1, minWidth: 12 }} />

        <RacePicker races={races} race={race}
          onLoad={(id: string) => load(id).catch((e) => setErr(String(e)))}
          open={showAnalyse} setOpen={setShowAnalyse}
          round={analyzeRound} setRound={setAnalyzeRound}
          year={analyzeYear} setYear={setAnalyzeYear}
          job={analyzeJob} onRun={runAnalysis} />

        <PairPicker race={race} subject={subject} rival={rival}
          busy={pairBusy} onPick={pick}
          onBattle={(a, b) => void applyPair(race.id, race.circuit_geometry.length, a, b)} />

        <button onClick={() => usePlayback.getState().startDemo(
          bestBattleTime(subject, rival, race.circuit_geometry.length,
            usePlayback.getState().tRange) ?? undefined)}
          style={btn(demo)}>
          {demo ? `demo · ${DEMO_BEATS[beat]?.label ?? ''}` : 'run demo'}
        </button>
        <HelpButton view={view} />
        <button onClick={() => setLite(!lite)} style={{ ...btn(lite), padding: '7px 10px' }}>
          lite
        </button>
      </div>

      {/* One context strip, not four stacked bands. Circuit, regulation variant,
          harvest cap, identifiability and the guide line all live here; the
          low-identifiability warning is an amber treatment of this same strip
          rather than a fifth row pushing the content down. */}
      <div style={{ padding: '7px 16px', zIndex: 19, flexShrink: 0,
        borderBottom: `1px solid ${lowIdent ? C.amber : C.panelBorder}`,
        background: lowIdent ? 'rgba(255,195,0,.08)' : 'rgba(11,11,15,.75)',
        display: 'flex', gap: 10, alignItems: 'baseline', flexWrap: 'wrap',
        fontSize: 12 }}>
        <b style={{ color: C.white }}>{GUIDE[view].title}</b>
        <span style={{ color: C.gray, opacity: 0.85 }}>
          {GUIDE[view].what.split('. ')[0]}.
        </span>
        <span style={{ marginLeft: 'auto', color: lowIdent ? C.amber : C.dim,
                       display: 'flex', gap: 8, alignItems: 'baseline' }}>
          <span>{race.circuit}</span>
          <span>· {race.regulation?.variant} · {race.regulation?.p_harv_max_kw} kW harvest</span>
          <span>· {((current?.identifiability ?? 0) * 100).toFixed(0)}% identifiable</span>
          {lowIdent && <b title="No car ran fast enough at this circuit to pin drag area
 from its own trace. Bands here are wide by necessity, and the estimator says so
 rather than guessing.">· bands wide by necessity</b>}
          <span style={{ color: C.dim }}>· press <b style={{ color: C.gray }}>?</b> for help</span>
        </span>
      </div>

      {analyzeJob && analyzeJob.status !== 'done' && (
        <div style={{ padding: '8px 16px', fontSize: 12.5, zIndex: 20, flexShrink: 0,
          borderBottom: `1px solid ${analyzeJob.status === 'error' ? C.red : C.panelBorder}`,
          background: analyzeJob.status === 'error' ? 'rgba(255,80,80,.1)' : 'rgba(255,255,255,.06)',
          color: analyzeJob.status === 'error' ? C.red : C.gray }}>
          {analyzeJob.status === 'error'
            ? `Analysis failed: ${analyzeJob.message}`
            : `Running the real pipeline for round ${analyzeRound} — FastF1 download `
              + 'plus particle filter, this can take a few minutes.'}
          <button onClick={() => setAnalyzeJob(null)}
            style={{ marginLeft: 12, background: 'transparent', border: 'none',
                     color: 'inherit', fontSize: 12, textDecoration: 'underline' }}>
            dismiss
          </button>
        </div>
      )}

      {pairErr && (
        <div style={{ padding: '8px 16px', background: 'rgba(255,195,0,.1)',
          borderBottom: `1px solid ${C.amber}`, color: C.amber, fontSize: 12.5,
          zIndex: 20, flexShrink: 0 }}>
          {pairErr}
        </div>
      )}

      <div style={{ position: 'relative', flex: 1, minHeight: 0 }}>
        <AnimatePresence mode="wait">
          <motion.div key={view} initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            exit={{ opacity: 0 }} transition={{ duration: 0.18 }}
            style={{ position: 'absolute', inset: 0 }}>
            {view === 'cockpit' && (subject && rival
              ? <Cockpit raceId={race.id} car={subject.driver} rival={rival.driver} />
              : <NoPair busy={pairBusy} />)}
            {view === 'situations' && (subject && rival
              ? <Situations raceId={race.id} car={subject.driver} rival={rival.driver} />
              : <NoPair busy={pairBusy} />)}
            {view === 'replay' &&
              <Theatre race={race} subject={subject} rival={rival} obs={obs} />}
            {view === 'observability' && <Observability data={obs} />}
            {view === 'evidence' && <Evidence races={races} current={current} />}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}

/** Every tab that needs two cars says so, rather than rendering nothing. The
 *  old view switch was guarded on `subject && rival` with no else branch, so a
 *  failed car fetch produced an empty viewport under a populated nav bar. */
function NoPair({ busy }: { busy: boolean }) {
  return (
    <div style={{ padding: 34, color: busy ? C.dim : C.amber, fontSize: 13,
                  lineHeight: 1.7, maxWidth: 640 }}>
      {busy ? 'loading the pair…'
        : 'This view needs two cars. Pick a pairing in the bar above — the '
          + 'battles listed first are pairs that actually raced each other.'}
    </div>
  );
}

function RacePicker({ races, race, onLoad, open, setOpen, round, setRound,
                      year, setYear, job, onRun }: any) {
  return (
    <div style={{ position: 'relative', display: 'flex', gap: 4 }}>
      <select value={race.id} onChange={(e) => onLoad(e.target.value)}
        title="which analysed race to open"
        style={{ background: C.panel, color: C.white, border: `1px solid ${C.panelBorder}`,
          borderRadius: 7, padding: '6px 9px', fontSize: 12, maxWidth: 260 }}>
        {races.map((r: RaceSummary) => (
          <option key={r.id} value={r.id}>
            {r.circuit} — {(r.identifiability * 100).toFixed(0)}% identifiable
          </option>
        ))}
      </select>
      <button onClick={() => setOpen(!open)} title="analyse a round not yet on disk"
        style={{ padding: '6px 9px', fontSize: 12, borderRadius: 7,
          border: `1px solid ${open ? C.amber : C.panelBorder}`,
          background: 'transparent', color: open ? C.amber : C.dim }}>＋</button>
      {open && (
        <div className="panel" style={{ position: 'absolute', top: 36, right: 0,
          zIndex: 40, padding: 13, width: 262 }}>
          <div style={{ color: C.gray, fontSize: 11, letterSpacing: '0.09em',
                        fontWeight: 700, marginBottom: 8 }}>ANALYSE A ROUND</div>
          <div style={{ color: C.dim, fontSize: 11.5, lineHeight: 1.55, marginBottom: 9 }}>
            Runs the real pipeline — FastF1 download plus particle filter. Minutes,
            not seconds, and it needs a network.
          </div>
          <div style={{ display: 'flex', gap: 6 }}>
            <input value={round} onChange={(e) => setRound(e.target.value)}
              placeholder="round" inputMode="numeric"
              style={{ width: 70, background: C.panel, color: C.white,
                border: `1px solid ${C.panelBorder}`, borderRadius: 7,
                padding: '6px 7px', fontSize: 12 }} />
            <input value={year} onChange={(e) => setYear(e.target.value)}
              placeholder="year" inputMode="numeric"
              style={{ width: 74, background: C.panel, color: C.white,
                border: `1px solid ${C.panelBorder}`, borderRadius: 7,
                padding: '6px 7px', fontSize: 12 }} />
            <button onClick={onRun} disabled={job?.status === 'running'}
              style={{ flex: 1, padding: '6px 9px', fontSize: 12, borderRadius: 7,
                border: `1px solid ${C.panelBorder}`, background: 'transparent',
                color: job?.status === 'running' ? C.dim : C.gray }}>
              {job?.status === 'running' ? '…' : 'run'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Pick who is attacking whom.
 *
 * The old control was two unlabelled selects over the raw driver list, which
 * hid three things the payload already knows: which pairs actually raced
 * (`battles`, previously read once at load and then discarded), which cars the
 * estimator refused (`refusals` — 16 of 20 on Melbourne), and that picking the
 * same driver on both sides produced a self-battle the backend happily served
 * with gap 0.
 */
function PairPicker({ race, subject, rival, busy, onPick, onBattle }: {
  race: RaceDetail; subject: Car | null; rival: Car | null; busy: boolean;
  onPick: (w: 'subject' | 'rival', d: string) => void;
  onBattle: (a: string, b: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const battles = race.battles ?? [];
  const label = subject && rival ? `${subject.driver} vs ${rival.driver}` : 'pick a pairing';

  const sel = (which: 'subject' | 'rival', value: string, other: string, colour: string) => (
    <select value={value} disabled={busy}
      onChange={(e) => onPick(which, e.target.value)}
      style={{ flex: 1, background: C.panel, color: colour,
        border: `1px solid ${C.panelBorder}`, borderRadius: 7, padding: '6px 8px',
        fontSize: 12 }}>
      {/* An explicit empty option. With `value=''` and no matching option the
          browser displays the first driver while React holds '', so selecting
          that driver fired no change event and it could not be picked at all. */}
      <option value="">— pick —</option>
      {race.drivers.map((d) => {
        const refused = race.refusals?.[d];
        return (
          <option key={d} value={d} disabled={d === other}
            title={refused?.message ?? undefined}>
            {d}{refused ? ' ·' : ''}{d === other ? ' (other car)' : ''}
          </option>
        );
      })}
    </select>
  );

  return (
    <div style={{ position: 'relative' }}>
      <button onClick={() => setOpen(!open)} title="choose which two cars to compare"
        style={{ padding: '6px 11px', fontSize: 12, borderRadius: 7,
          border: `1px solid ${open ? C.amber : C.panelBorder}`, background: C.panel,
          color: C.white, minWidth: 132 }}>
        <span style={{ color: C.amber }}>{subject?.driver ?? '—'}</span>
        <span style={{ color: C.dim }}> vs </span>
        <span style={{ color: C.red }}>{rival?.driver ?? '—'}</span>
        {busy && <span style={{ color: C.dim }}> …</span>}
      </button>
      {open && (
        <div className="panel" style={{ position: 'absolute', top: 36, right: 0,
          zIndex: 40, padding: 13, width: 330 }}>
          <div style={{ color: C.gray, fontSize: 11, letterSpacing: '0.09em',
                        fontWeight: 700, marginBottom: 4 }}>BATTLES IN THIS RACE</div>
          <div style={{ color: C.dim, fontSize: 11.5, marginBottom: 8 }}>
            Pairs that actually raced each other, closest first.
          </div>
          {battles.length ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4,
                          maxHeight: 190, overflowY: 'auto' }}>
              {battles.map((b) => {
                const on = subject?.driver === b.car && rival?.driver === b.ahead;
                return (
                  <button key={`${b.car}-${b.ahead}`} disabled={busy}
                    onClick={() => { onBattle(b.car, b.ahead); setOpen(false); }}
                    style={{ textAlign: 'left', padding: '6px 9px', fontSize: 12,
                      borderRadius: 6, border: `1px solid ${on ? C.amber : C.panelBorder}`,
                      background: 'transparent', color: on ? C.amber : C.gray }}>
                    <b style={{ color: on ? C.amber : C.white }}>{b.car} vs {b.ahead}</b>
                    <span style={{ color: C.dim }}>
                      {' '}— {b.laps_close} laps within {race.telemetry.gap_limit_s ?? 1} s,
                      median {b.median_gap.toFixed(2)} s
                    </span>
                  </button>
                );
              })}
            </div>
          ) : (
            <div style={{ color: C.amber, fontSize: 11.5, lineHeight: 1.55 }}>
              No two cars in this race spent long enough close together to count
              as a battle. Pick a pair by hand below — the decision trace may
              have nothing to say about them.
            </div>
          )}

          <div style={{ color: C.gray, fontSize: 11, letterSpacing: '0.09em',
                        fontWeight: 700, margin: '13px 0 4px' }}>OR PICK ANY TWO</div>
          <div style={{ color: C.dim, fontSize: 11.5, marginBottom: 8 }}>
            A <b>·</b> marks a car the estimator refused; hover it for the reason.
            A new pair is solved once and takes about a minute.
          </div>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            {sel('subject', subject?.driver ?? '', rival?.driver ?? '', C.amber)}
            <span style={{ color: C.dim, fontSize: 11 }}>vs</span>
            {sel('rival', rival?.driver ?? '', subject?.driver ?? '', C.red)}
          </div>
          <div style={{ color: C.dim, fontSize: 11, marginTop: 8 }}>{label}</div>
        </div>
      )}
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
