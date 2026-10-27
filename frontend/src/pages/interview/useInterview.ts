import * as React from 'react';

import { ApiError } from '@/api/client';
import { getSession, getTurn } from '@/api/sessions';
import { followSessionStream, StreamFailure } from '@/api/stream';
import type { StreamEvent } from '@/api/types';

import {
  applyEvent,
  fromSnapshot,
  type InterviewView,
  snapshotIsCurrent,
  withTurnStatus,
} from './interviewModel';

export type LoadState =
  | { kind: 'loading' }
  | { kind: 'not-found' }
  | { kind: 'error'; message: string }
  | { kind: 'ready' };

/** "connecting" until the first open, "live" while open, "reconnecting" between tries. */
export type Connection = 'connecting' | 'live' | 'reconnecting' | 'closed' | 'failed';

/** Reconnect backoff: 1 s, 2 s, 4 s ... capped at 15 s. */
export function reconnectDelayMs(attempt: number): number {
  return Math.min(15_000, 1000 * 2 ** Math.max(0, attempt - 1));
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        resolve();
      },
      { once: true },
    );
  });
}

/**
 * The interview's authoritative state: a snapshot, then the event stream from
 * the snapshot's seq (Last-Event-ID), reconnecting from the last seq applied.
 *
 * The stream is aborted on unmount - which includes sign-out and a lost
 * session, since the route is protected - and whenever the snapshot is
 * re-read, so at most one stream is ever open per screen.
 */
export function useInterview(sessionId: string) {
  const [load, setLoad] = React.useState<LoadState>({ kind: 'loading' });
  const [view, setView] = React.useState<InterviewView | null>(null);
  const [connection, setConnection] = React.useState<Connection>('connecting');
  const [streamError, setStreamError] = React.useState<string | null>(null);
  // The applied view, synchronously: events are deduplicated against it.
  const viewRef = React.useRef<InterviewView | null>(null);
  const [generation, setGeneration] = React.useState(0);

  const commit = React.useCallback((next: InterviewView | null) => {
    viewRef.current = next;
    setView(next);
  }, []);

  // 1. The snapshot (re-read on demand: a seq gap, a conflict, "Try again").
  React.useEffect(() => {
    const controller = new AbortController();
    setLoad((current) => (current.kind === 'ready' ? current : { kind: 'loading' }));
    (async () => {
      try {
        let snapshot = fromSnapshot(await getSession(sessionId, controller.signal));
        if (snapshot.state === 'GRADING' && snapshot.currentTurnId) {
          // Was this answer's review already given up on? Only the turn knows.
          try {
            const turn = await getTurn(sessionId, snapshot.currentTurnId, controller.signal);
            snapshot = withTurnStatus(snapshot, turn);
          } catch {
            // Not fatal: the stream reports a failure too.
          }
        }
        if (controller.signal.aborted) return;
        if (snapshotIsCurrent(viewRef.current, snapshot)) commit(snapshot);
        setLoad({ kind: 'ready' });
      } catch (error) {
        if (controller.signal.aborted) return;
        if (error instanceof ApiError && (error.status === 404 || error.status === 422)) {
          setLoad({ kind: 'not-found' });
        } else if (error instanceof ApiError && error.status === 401) {
          // The auth provider has already been told; this screen is unmounting.
          return;
        } else {
          setLoad({
            kind: 'error',
            message:
              error instanceof ApiError ? error.message : 'The interview could not be loaded.',
          });
        }
      }
    })();
    return () => controller.abort();
  }, [sessionId, generation, commit]);

  const reload = React.useCallback(() => setGeneration((n) => n + 1), []);

  const onEvent = React.useCallback(
    (event: StreamEvent) => {
      const current = viewRef.current;
      if (!current) return;
      const { view: next, gap } = applyEvent(current, event);
      if (gap) {
        // Missed something: the snapshot is the truth, not a guess.
        reload();
        return;
      }
      if (next !== current) commit(next);
    },
    [commit, reload],
  );

  // 2. The stream, from whatever seq has been applied.
  const ready = load.kind === 'ready';
  const finished = view?.state === 'COMPLETED';
  React.useEffect(() => {
    if (!ready || finished) {
      if (finished) setConnection('closed');
      return;
    }
    const controller = new AbortController();
    const { signal } = controller;
    setStreamError(null);
    (async () => {
      let attempt = 0;
      while (!signal.aborted) {
        setConnection(attempt === 0 ? 'connecting' : 'reconnecting');
        try {
          const result = await followSessionStream(sessionId, {
            after: viewRef.current?.seq ?? -1,
            signal,
            onOpen: () => {
              attempt = 0;
              setConnection('live');
            },
            onEvent,
          });
          if (result === 'complete') {
            setConnection('closed');
            return;
          }
        } catch (error) {
          if (signal.aborted) return;
          if (error instanceof StreamFailure) {
            setConnection('failed');
            setStreamError('Live updates for this interview are unavailable right now.');
            return;
          }
          if (error instanceof ApiError && (error.status === 401 || error.status === 404)) {
            // 401: signed out (the auth provider handles it). 404: gone.
            setConnection('failed');
            if (error.status === 404) setLoad({ kind: 'not-found' });
            return;
          }
          // Network drop, proxy restart, 5xx: retry from the last applied seq.
        }
        attempt += 1;
        setConnection('reconnecting');
        await sleep(reconnectDelayMs(attempt), signal);
      }
    })();
    return () => controller.abort();
  }, [sessionId, ready, finished, generation, onEvent]);

  return { load, view, connection, streamError, reload, commit };
}
