import { render } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router';
import { vi } from 'vitest';

import type { SessionSummary, SetupOptions, User } from '@/api/types';
import { AppRoutes } from '@/App';
import { AuthProvider } from '@/auth/AuthProvider';

export interface MockResponse {
  status?: number;
  body?: unknown;
}

type Handler = (request: {
  body: unknown;
  init: RequestInit;
}) => MockResponse | Promise<MockResponse>;

export interface RecordedCall {
  method: string;
  path: string;
  body: unknown;
  headers: Record<string, string>;
}

/**
 * Route `fetch` by "METHOD /path". Unmatched requests fail the test loudly
 * rather than hanging. A handler array is consumed in order (last one repeats).
 */
export function mockApi(routes: Record<string, Handler | MockResponse | MockResponse[]>) {
  const calls: RecordedCall[] = [];
  const counters = new Map<string, number>();
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = new URL(String(input), 'http://localhost');
    const method = (init.method ?? 'GET').toUpperCase();
    const key = `${method} ${url.pathname}`;
    let body: unknown = init.body;
    if (typeof init.body === 'string') {
      try {
        body = JSON.parse(init.body);
      } catch {
        body = init.body;
      }
    }
    calls.push({
      method,
      path: url.pathname,
      body,
      headers: (init.headers ?? {}) as Record<string, string>,
    });
    const route = routes[key];
    if (!route) throw new Error(`Unmocked request: ${key}`);
    let spec: MockResponse;
    if (typeof route === 'function') spec = await route({ body, init });
    else if (Array.isArray(route)) {
      const n = counters.get(key) ?? 0;
      counters.set(key, n + 1);
      spec = route[Math.min(n, route.length - 1)] as MockResponse;
    } else spec = route;
    const status = spec.status ?? 200;
    return new Response(
      status === 204 || spec.body === undefined ? null : JSON.stringify(spec.body),
      {
        status,
        headers: { 'Content-Type': 'application/json' },
      },
    );
  });
  vi.stubGlobal('fetch', fetchMock);
  return {
    calls,
    fetchMock,
    callsTo: (method: string, path: string) =>
      calls.filter((c) => c.method === method && c.path === path),
  };
}

export const USER: User = {
  id: '11111111-1111-1111-1111-111111111111',
  email: 'candidate@example.com',
  created_at: '2026-10-01T10:00:00Z',
};

export const TOKEN = { access_token: 'access-1', token_type: 'bearer', expires_in: 900 };

export const OPTIONS: SetupOptions = {
  topics: [
    {
      key: 'dsa',
      display_name: 'Data structures & algorithms',
      domain: 'cs_fundamentals',
      subtopics: [
        { key: 'arrays', display_name: 'Arrays & two pointers' },
        { key: 'graphs', display_name: 'Graphs' },
      ],
    },
    {
      key: 'databases',
      display_name: 'Databases',
      domain: 'backend',
      subtopics: [{ key: 'indexing', display_name: 'Indexing' }],
    },
  ],
  levels: ['junior', 'mid', 'senior', 'staff'],
  modes: ['screening', 'standard', 'deep_dive'],
  default_level: 'mid',
  default_mode: 'standard',
  default_duration_minutes: 45,
  min_duration_minutes: 5,
  max_duration_minutes: 360,
};

export function session(overrides: Partial<SessionSummary> = {}): SessionSummary {
  return {
    session_id: '22222222-2222-2222-2222-222222222222',
    state: 'COMPLETED',
    role: 'Backend engineer',
    level: 'mid',
    mode: 'standard',
    topic_keys: ['dsa'],
    resume_id: null,
    jd_id: null,
    started_at: '2026-10-02T09:00:00Z',
    ended_at: '2026-10-02T09:40:00Z',
    report_status: 'ready',
    ...overrides,
  };
}

/** A signed-in user's standard backend: restore succeeds, nothing else is assumed. */
export function signedInRoutes() {
  return {
    'POST /api/auth/refresh': { body: TOKEN },
    'GET /api/auth/me': { body: USER },
    'GET /api/setup-options': { body: OPTIONS },
  } satisfies Record<string, MockResponse>;
}

function LocationProbe() {
  const location = useLocation();
  return <div data-testid="location">{location.pathname}</div>;
}

export function renderApp(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AuthProvider>
        <AppRoutes />
        <LocationProbe />
      </AuthProvider>
    </MemoryRouter>,
  );
}
