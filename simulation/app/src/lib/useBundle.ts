import { useEffect, useRef, useState } from 'react';
import { api, type BundleJob } from './api';

/**
 * Wait for one pair's decision trace, reporting progress while it solves.
 *
 * Every view that needs `/decision`, `/p2` or `/situations` gates on this
 * first. A pair nobody has solved before costs ~112 s (measured, one 22-lap
 * pair), and that solve used to happen inline on the request: the tab sat in a
 * spinner and, because the work is GIL-bound pure Python in the API process,
 * so did every other endpoint — switching race or driver pair did nothing
 * until it finished. The solve is now a background job and this polls it, so
 * the rest of the app stays interactive and the waiting tab can say how long
 * it has been rather than claiming "about a minute".
 *
 * The poll is a chained `setTimeout`, not `setInterval`: it cannot overlap
 * itself on a slow response, and the cleanup below cancels it. `App.tsx`'s
 * analyse poller had to learn the same thing — it leaked a chain per run.
 */
const POLL_MS = 2000;

export function useBundle(raceId: string, car: string, rival: string): BundleJob {
  const [job, setJob] = useState<BundleJob>({ status: 'building', elapsed_s: 0 });
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Guards against a poll for the previously-selected pair resolving after a
  // swap and reporting its progress against the new one.
  const token = useRef(0);

  useEffect(() => {
    const mine = ++token.current;
    setJob({ status: 'building', elapsed_s: 0 });

    const poll = async () => {
      let next: BundleJob;
      try {
        next = await api.bundleStatus(raceId, car, rival);
      } catch (e) {
        // A failed poll is reported as an error rather than retried forever:
        // an unreachable API is not a pair that is still solving.
        next = { status: 'error', message: String(e) };
      }
      if (token.current !== mine) return;
      setJob(next);
      if (next.status === 'building') timer.current = setTimeout(poll, POLL_MS);
    };
    void poll();

    return () => {
      token.current++;
      if (timer.current) clearTimeout(timer.current);
    };
  }, [raceId, car, rival]);

  return job;
}

/** Shared copy for a tab that is waiting on a solve. One wording, so Cockpit
 *  and Situations cannot drift into quoting different durations. */
export function solveProgressText(job: BundleJob): string {
  const s = Math.round(job.elapsed_s ?? 0);
  return `Solving this pair's decision trace — ${s} s elapsed. `
    + 'A pair nobody has opened before takes roughly two minutes and is on '
    + 'disk from then on. The rest of the app stays usable meanwhile.';
}
