import { useEffect, useRef } from 'react';
import { usePlayback } from '../store/playback';

/**
 * Scripted demo run. Every beat is driven by real precomputed data — nothing is
 * a scripted fake. Manual override available at any point: any keypress or
 * click on a control stops it.
 */
export const DEMO_BEATS = [
  { label: 'intro orbit', s: 4, apply: () => set({ camera: 'chase', paintMode: 'neutral', showCloud: false }) },
  { label: 'just a race', s: 6, apply: () => set({ camera: 'chase', showCloud: false }) },
  { label: 'cloud materialises', s: 3, apply: () => set({ showCloud: true }) },
  { label: 'convergence', s: 8, apply: () => set({ camera: 'duel', showCloud: true }) },
  { label: 'trails ignite', s: 5, apply: () => set({ paintMode: 'deployment' }) },
  { label: 'decision', s: 6, apply: () => { set({ camera: 'tactical' }); usePlayback.getState().setView('decision'); } },
  { label: 'execution', s: 5, apply: () => { usePlayback.getState().setView('theatre'); set({ camera: 'duel' }); } },
  { label: 'observability', s: 5, apply: () => { set({ paintMode: 'observability' }); usePlayback.getState().setView('observability'); } },
] as const;

function set(p: any) { usePlayback.setState(p); }

export function useDemo() {
  const demo = usePlayback((s) => s.demo);
  const timer = useRef<any>(null);
  useEffect(() => {
    if (!demo) { clearTimeout(timer.current); return; }
    let i = 0;
    const step = () => {
      if (i >= DEMO_BEATS.length) { usePlayback.getState().stopDemo(); return; }
      usePlayback.getState().setBeat(i);
      DEMO_BEATS[i].apply();
      timer.current = setTimeout(() => { i++; step(); }, DEMO_BEATS[i].s * 1000);
    };
    usePlayback.getState().play();
    step();
    const bail = () => usePlayback.getState().stopDemo();
    window.addEventListener('keydown', bail, { once: true });
    return () => { clearTimeout(timer.current); window.removeEventListener('keydown', bail); };
  }, [demo]);
}
