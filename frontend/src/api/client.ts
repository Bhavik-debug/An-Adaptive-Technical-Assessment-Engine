/**
 * The one place the frontend talks HTTP.
 *
 * Authentication follows the backend's design (backend/app/api/auth.py):
 *
 * - The **access token** (15 min) lives only in this module's memory. It is
 *   never written to localStorage/sessionStorage and never logged.
 * - The **refresh token** is an httpOnly cookie scoped to /api/auth. This code
 *   cannot read it; it only asks the browser to send it to /api/auth/refresh.
 * - Refresh tokens are **single-use with replay detection**: presenting a spent
 *   one revokes the whole family. So refreshes must never overlap. Within a tab
 *   every caller shares one in-flight promise; across tabs the Web Locks API
 *   serialises them (each waiting tab then sends the newest cookie).
 */

import type { TokenResponse } from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '');

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/** FastAPI's request-validation error item. */
interface ValidationItem {
  loc?: (string | number)[];
  msg?: string;
  type?: string;
}

export class ApiError extends Error {
  /** HTTP status; 0 when the server could not be reached at all. */
  readonly status: number;
  /** The backend's machine-readable code, when it sends one (`{code, message}` details). */
  readonly code: string | undefined;
  /** Validation messages keyed by the offending body field (FastAPI 422 lists). */
  readonly fieldErrors: Record<string, string>;

  constructor(
    status: number,
    message: string,
    options: { code?: string; fieldErrors?: Record<string, string> } = {},
  ) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = options.code;
    this.fieldErrors = options.fieldErrors ?? {};
  }

  get isNetworkError(): boolean {
    return this.status === 0;
  }
}

const FALLBACK_MESSAGES: Record<number, string> = {
  400: 'The request was not valid.',
  401: 'Your session has ended. Please sign in again.',
  403: 'You do not have access to this.',
  404: 'We could not find what you were looking for.',
  409: 'That conflicts with the current state. Refresh and try again.',
  413: 'That is larger than the allowed limit.',
  415: 'That file type is not supported.',
  422: 'Some of the information provided is not valid.',
  429: 'Too many attempts. Please wait and try again.',
  500: 'Something went wrong on the server. Please try again.',
  502: 'The server is unavailable right now. Please try again shortly.',
  503: 'The server is unavailable right now. Please try again shortly.',
};

function fieldName(loc: (string | number)[] | undefined): string | null {
  if (!loc || loc.length === 0) return null;
  // ["body", "email"] -> "email"; ["body", "skills", 0, "name"] -> "skills.0.name"
  const parts = loc[0] === 'body' || loc[0] === 'query' ? loc.slice(1) : loc;
  return parts.length ? parts.join('.') : null;
}

function cleanValidationMessage(msg: string): string {
  // Pydantic prefixes custom errors with "Value error, ".
  return msg.replace(/^Value error, /, '');
}

/** Build an ApiError from any backend error body without discarding detail. */
export async function errorFromResponse(res: Response): Promise<ApiError> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // Not JSON (e.g. a proxy error page). Fall through to the status message.
  }
  const fallback = FALLBACK_MESSAGES[res.status] ?? `Request failed (${res.status}).`;
  const detail = (body as { detail?: unknown } | null)?.detail;

  if (typeof detail === 'string' && detail.trim()) {
    return new ApiError(res.status, detail);
  }
  if (Array.isArray(detail)) {
    const fieldErrors: Record<string, string> = {};
    const messages: string[] = [];
    for (const item of detail as ValidationItem[]) {
      const msg = cleanValidationMessage(item.msg ?? 'Invalid value');
      const field = fieldName(item.loc);
      if (field && !(field in fieldErrors)) fieldErrors[field] = msg;
      messages.push(field ? `${field}: ${msg}` : msg);
    }
    return new ApiError(res.status, messages.length ? messages.join('; ') : fallback, {
      fieldErrors,
    });
  }
  if (detail && typeof detail === 'object') {
    const { code, message } = detail as { code?: unknown; message?: unknown };
    return new ApiError(res.status, typeof message === 'string' ? message : fallback, {
      code: typeof code === 'string' ? code : undefined,
    });
  }
  return new ApiError(res.status, fallback);
}

function networkError(): ApiError {
  return new ApiError(0, 'Could not reach the server. Check your connection and try again.');
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

// ---------------------------------------------------------------------------
// Access token (memory only) and refresh
// ---------------------------------------------------------------------------

let accessToken: string | null = null;
let refreshInFlight: Promise<string | null> | null = null;
let authLostHandler: (() => void) | null = null;

export function setAccessToken(token: string | null): void {
  accessToken = token;
}

export function hasAccessToken(): boolean {
  return accessToken !== null;
}

/** Called when a request proves the session is gone (refresh rejected). */
export function setAuthLostHandler(handler: (() => void) | null): void {
  authLostHandler = handler;
}

const REFRESH_LOCK = 'adaptive-interviewer:auth-refresh';

async function withRefreshLock<T>(fn: () => Promise<T>): Promise<T> {
  const locks = typeof navigator !== 'undefined' ? navigator.locks : undefined;
  if (locks?.request) {
    return locks.request(REFRESH_LOCK, fn) as Promise<T>;
  }
  return fn();
}

async function performRefresh(): Promise<string | null> {
  let res: Response;
  try {
    res = await fetch(apiUrl('/api/auth/refresh'), { method: 'POST', credentials: 'include' });
  } catch {
    throw networkError();
  }
  if (res.status === 401) {
    accessToken = null;
    return null;
  }
  if (!res.ok) throw await errorFromResponse(res);
  const body = (await res.json()) as TokenResponse;
  accessToken = body.access_token;
  return accessToken;
}

/**
 * Exchange the refresh cookie for a new access token.
 * Resolves to `null` when there is no valid session (the user must sign in);
 * rejects with a network ApiError when the server cannot be reached.
 */
export function refreshAccessToken(): Promise<string | null> {
  if (!refreshInFlight) {
    refreshInFlight = withRefreshLock(performRefresh).finally(() => {
      refreshInFlight = null;
    });
  }
  return refreshInFlight;
}

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

export interface RequestOptions {
  method?: 'GET' | 'POST' | 'PATCH' | 'PUT' | 'DELETE';
  /** Serialised as JSON with the matching Content-Type. */
  json?: unknown;
  /** A raw body (e.g. a File), sent as-is; set Content-Type via `headers`. */
  body?: BodyInit;
  headers?: Record<string, string>;
  signal?: AbortSignal;
  /** Attach the bearer token and recover from one expired-token 401. Default true. */
  auth?: boolean;
  /** Send cookies (only the /api/auth routes need them). */
  withCredentials?: boolean;
}

export interface ApiResponse<T> {
  data: T;
  status: number;
  headers: Headers;
}

async function send(path: string, options: RequestOptions, token: string | null) {
  const headers: Record<string, string> = { Accept: 'application/json', ...options.headers };
  if (options.json !== undefined) headers['Content-Type'] = 'application/json';
  if (token) headers.Authorization = `Bearer ${token}`;
  try {
    return await fetch(apiUrl(path), {
      method: options.method ?? 'GET',
      headers,
      body: options.json !== undefined ? JSON.stringify(options.json) : options.body,
      credentials: options.withCredentials ? 'include' : 'same-origin',
      signal: options.signal,
    });
  } catch (error) {
    if (isAbort(error)) throw error;
    throw networkError();
  }
}

async function parseBody<T>(res: Response): Promise<T> {
  if (res.status === 204) return undefined as T;
  const type = res.headers.get('content-type') ?? '';
  if (type.includes('application/json')) return (await res.json()) as T;
  return undefined as T;
}

/** Like `apiRequest`, but also returns the status and headers (e.g. 202 vs 200). */
export async function apiRequestFull<T>(
  path: string,
  options: RequestOptions = {},
): Promise<ApiResponse<T>> {
  const useAuth = options.auth ?? true;
  const tokenUsed = useAuth ? accessToken : null;
  let res = await send(path, options, tokenUsed);

  if (res.status === 401 && useAuth) {
    // Another request may already have refreshed while this one was in flight.
    const fresh =
      accessToken && accessToken !== tokenUsed ? accessToken : await refreshAccessToken();
    if (!fresh) {
      authLostHandler?.();
      throw await errorFromResponse(res);
    }
    res = await send(path, options, fresh);
    if (res.status === 401) {
      accessToken = null;
      authLostHandler?.();
    }
  }

  if (!res.ok) throw await errorFromResponse(res);
  return { data: await parseBody<T>(res), status: res.status, headers: res.headers };
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  return (await apiRequestFull<T>(path, options)).data;
}
