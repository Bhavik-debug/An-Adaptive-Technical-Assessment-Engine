/**
 * GET /api/sessions/{id}/stream - the session's state transitions as SSE
 * (backend/app/interview/stream.py).
 *
 * Contract this code relies on:
 *  - every message's `id` is the event's per-session `seq`, and its JSON
 *    `data` carries `seq` and the `state` after the event;
 *  - `Last-Event-ID: n` replays every event with seq > n, in order, exactly
 *    once - so reconnecting with the last seq seen misses nothing;
 *  - the server ends the stream after `session_complete`;
 *  - `event: error` (no id) means the session's log could not be read.
 */

import { apiStream } from './client';
import type { StreamEvent } from './types';
import { SseParser } from '@/lib/sse';

const KNOWN = new Set([
  'grading_started',
  'grade_ready',
  'grading_failed',
  'follow_up',
  'follow_up_abandoned',
  'next_question',
  'session_complete',
  'state_changed',
]);

export class StreamFailure extends Error {
  readonly code: string;
  constructor(code: string) {
    super(`The interview stream reported an error (${code}).`);
    this.name = 'StreamFailure';
    this.code = code;
  }
}

export interface StreamOptions {
  /** The last seq already applied; events after it are delivered. */
  after: number;
  signal: AbortSignal;
  onOpen?: () => void;
  onEvent: (event: StreamEvent) => void;
}

/** Parse one SSE message into a typed event, or null if it is not one we can use. */
export function toStreamEvent(name: string, data: string): StreamEvent | null {
  let body: unknown;
  try {
    body = JSON.parse(data);
  } catch {
    return null;
  }
  if (!body || typeof body !== 'object') return null;
  const record = body as Record<string, unknown>;
  if (typeof record.seq !== 'number' || typeof record.state !== 'string') return null;
  // An event name added server-side later is still a state transition.
  const type = KNOWN.has(name) ? name : 'state_changed';
  return { ...record, type } as StreamEvent;
}

/**
 * Follow the stream until the server ends it or `signal` aborts.
 * Resolves `'complete'` after `session_complete`, `'ended'` if the server
 * closed the stream early (the caller reconnects). Rejects with an ApiError
 * (HTTP/network) or a StreamFailure (`event: error`).
 */
export async function followSessionStream(
  sessionId: string,
  { after, signal, onOpen, onEvent }: StreamOptions,
): Promise<'complete' | 'ended'> {
  const res = await apiStream(`/api/sessions/${encodeURIComponent(sessionId)}/stream`, {
    headers: { 'Last-Event-ID': String(after) },
    signal,
  });
  onOpen?.();
  if (!res.body) return 'ended';
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return 'ended';
      for (const message of parser.push(decoder.decode(value, { stream: true }))) {
        if (message.event === 'error') {
          let code = 'stream_error';
          try {
            code = (JSON.parse(message.data) as { code?: string }).code ?? code;
          } catch {
            // keep the generic code
          }
          throw new StreamFailure(code);
        }
        const event = toStreamEvent(message.event, message.data);
        if (!event) continue;
        onEvent(event);
        if (event.type === 'session_complete') return 'complete';
      }
    }
  } finally {
    // Releases the connection: an unmount or sign-out must not leave it open.
    reader.cancel().catch(() => undefined);
  }
}
