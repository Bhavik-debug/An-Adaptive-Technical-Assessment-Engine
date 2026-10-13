import * as React from 'react';

import { ApiError } from '@/api/client';

export type PollPhase = 'idle' | 'polling' | 'done' | 'timeout' | 'error';

export interface PollOptions<T> {
  /** Fetch the latest value. */
  fetcher: (signal: AbortSignal) => Promise<T>;
  /** True when no more polling is needed (e.g. status is ready or failed). */
  isDone: (value: T) => boolean;
  intervalMs?: number;
  timeoutMs?: number;
}

/**
 * Poll a background job until it settles, times out, or errors. Polling stops
 * on unmount, and a timeout is reported rather than silently treated as failure:
 * the job may still finish, and `restart` checks again.
 */
export function usePolling<T>(key: string | null, options: PollOptions<T>) {
  const { intervalMs = 1500, timeoutMs = 90_000 } = options;
  const [value, setValue] = React.useState<T | undefined>(undefined);
  const [phase, setPhase] = React.useState<PollPhase>('idle');
  const [error, setError] = React.useState<ApiError | null>(null);
  const [round, setRound] = React.useState(0);

  // Keep the latest callbacks without restarting the loop on every render.
  const fetcherRef = React.useRef(options.fetcher);
  const isDoneRef = React.useRef(options.isDone);
  fetcherRef.current = options.fetcher;
  isDoneRef.current = options.isDone;

  React.useEffect(() => {
    if (!key) {
      setPhase('idle');
      setValue(undefined);
      return;
    }
    const controller = new AbortController();
    const deadline = Date.now() + timeoutMs;
    let timer: ReturnType<typeof setTimeout> | undefined;
    setPhase('polling');
    setError(null);

    const tick = async () => {
      try {
        const next = await fetcherRef.current(controller.signal);
        if (controller.signal.aborted) return;
        setValue(next);
        if (isDoneRef.current(next)) {
          setPhase('done');
          return;
        }
        if (Date.now() >= deadline) {
          setPhase('timeout');
          return;
        }
        timer = setTimeout(tick, intervalMs);
      } catch (err) {
        if (controller.signal.aborted) return;
        setError(err instanceof ApiError ? err : new ApiError(-1, 'Something went wrong.'));
        setPhase('error');
      }
    };
    void tick();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [key, round, intervalMs, timeoutMs]);

  const restart = React.useCallback(() => setRound((n) => n + 1), []);
  return { value, phase, error, restart, setValue };
}
