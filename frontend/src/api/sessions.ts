import { apiRequest, apiRequestFull } from './client';
import type {
  Report,
  SessionCreate,
  SessionCreated,
  SessionDetail,
  SessionStarted,
  SessionSummary,
  TurnStatus,
  TurnSubmission,
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

/** backend/app/grading/grader.py: MAX_ANSWER_CHARS (TurnSubmission.answer's max_length). */
export const MAX_ANSWER_CHARS = 8000;

/** GET /api/sessions/{id} -> the session's current state and question (404 if not yours). */
export function getSession(sessionId: string, signal?: AbortSignal): Promise<SessionDetail> {
  return apiRequest<SessionDetail>(`/api/sessions/${encodeURIComponent(sessionId)}`, { signal });
}

/**
 * POST /api/sessions/{id}/turns -> 202 accepted, or 200 for a retry of the same
 * turn. Idempotent on `turn_id` **with the exact same answer text**: resending
 * after an unknown outcome is safe only with both unchanged.
 */
export function submitTurn(
  sessionId: string,
  submission: TurnSubmission,
  signal?: AbortSignal,
): Promise<TurnStatus> {
  return apiRequest<TurnStatus>(`/api/sessions/${encodeURIComponent(sessionId)}/turns`, {
    method: 'POST',
    json: submission,
    signal,
  });
}

/** GET /api/sessions/{id}/turns/{turn_id} -> where one answer's grading stands. */
export function getTurn(sessionId: string, turnId: string, signal?: AbortSignal) {
  return apiRequest<TurnStatus>(
    `/api/sessions/${encodeURIComponent(sessionId)}/turns/${encodeURIComponent(turnId)}`,
    { signal },
  );
}

/** GET /api/sessions/{id}/report -> 200 ready, 202 being built, 409 not finished. */
export async function getReport(sessionId: string, signal?: AbortSignal): Promise<Report> {
  const res = await apiRequestFull<Report>(
    `/api/sessions/${encodeURIComponent(sessionId)}/report`,
    { signal },
  );
  return res.data;
}
