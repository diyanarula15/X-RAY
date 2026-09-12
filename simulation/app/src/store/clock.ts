import { useEffect, useState } from 'react';
import { usePlayback } from './playback';

/**
 * The 2D layer does not need 60 Hz.
 *
 * Race time advances every animation frame. Feeding that straight into React
 * re-renders the whole tree -- including the Canvas subtree -- sixty times a
 * second, and the result is a scene that stutters while the browser is busy
 * reconciling panels that only change in the second decimal place.
 *
 * The 3D scene reads the clock imperatively inside its own render loop. Overlays
 * use this hook, which samples the same clock at a rate a human can read.
 */
export function useThrottledTime(hz = 12): number {
  const [t, setT] = useState(() => usePlayback.getState().raceTime);
  useEffect(() => {
    let last = 0;
    const period = 1000 / hz;
    const unsub = usePlayback.subscribe((s) => {
      const now = performance.now();
      if (now - last >= period) { last = now; setT(s.raceTime); }
    });
    return unsub;
  }, [hz]);
  return t;
}
