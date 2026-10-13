import { apiRequest, apiRequestFull } from './client';
import type {
  Report,
  SessionCreate,
  SessionCreated,
  SessionStarted,
  SessionSummary,
} from './types';

/** GET /api/sessions -> the caller's sessions, newest first. */
export async function listSessions(signal?: AbortSignal): Promise<SessionSummary[]> {
  const body = await apiRequest<{ sessions: SessionSummary[] }>('/api/sessions', { signal });
  return body.sessions;
}

/**
 * POST /api/sessions -> 201. Not idempotent: callers must guard against
 * double submission and must not call this again once a session id exists.
 */
export function createSession(spec: SessionCreate): Promise<SessionCreated> {
  return apiRequest<SessionCreated>('/api/sessions', { method: 'POST', json: spec });
}

/** POST /api/sessions/{id}/start -> 200. Idempotent: a repeat replays the same first question. */
export function startSession(sessionId: string): Promise<SessionStarted> {
  return apiRequest<SessionStarted>(`/api/sessions/${encodeURIComponent(sessionId)}/start`, {
    method: 'POST',
  });
}

/** GET /api/sessions/{id}/report -> 200 ready, 202 being built, 409 not finished. */
export async function getReport(sessionId: string, signal?: AbortSignal): Promise<Report> {
  const res = await apiRequestFull<Report>(
    `/api/sessions/${encodeURIComponent(sessionId)}/report`,
    { signal },
  );
  return res.data;
}
