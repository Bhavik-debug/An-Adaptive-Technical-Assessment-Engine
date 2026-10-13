import { describe, expect, it, vi } from 'vitest';

import { mockApi, TOKEN } from '@/test/utils';

import {
  ApiError,
  apiRequest,
  refreshAccessToken,
  setAccessToken,
  setAuthLostHandler,
} from './client';

describe('error parsing', () => {
  it('keeps a plain string detail as the message', async () => {
    mockApi({ 'GET /api/x': { status: 409, body: { detail: 'Not allowed right now' } } });
    const error = await apiRequest('/api/x', { auth: false }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 409, message: 'Not allowed right now' });
  });

  it('maps FastAPI validation lists to field errors', async () => {
    mockApi({
      'POST /api/x': {
        status: 422,
        body: {
          detail: [
            { loc: ['body', 'email'], msg: 'value is not a valid email address', type: 'x' },
            { loc: ['body', 'skills', 0, 'name'], msg: 'Value error, must not be blank' },
          ],
        },
      },
    });
    const error = (await apiRequest('/api/x', { method: 'POST', json: {}, auth: false }).catch(
      (e: unknown) => e,
    )) as ApiError;
    expect(error.fieldErrors).toEqual({
      email: 'value is not a valid email address',
      'skills.0.name': 'must not be blank',
    });
  });

  it('keeps the backend code from {code, message} details', async () => {
    mockApi({
      'POST /api/x': {
        status: 413,
        body: { detail: { code: 'file_too_large', message: 'Too big.' } },
      },
    });
    const error = (await apiRequest('/api/x', { method: 'POST', auth: false }).catch(
      (e: unknown) => e,
    )) as ApiError;
    expect(error).toMatchObject({ status: 413, code: 'file_too_large', message: 'Too big.' });
  });

  it('reports an unreachable server as a network error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    const error = (await apiRequest('/api/x', { auth: false }).catch(
      (e: unknown) => e,
    )) as ApiError;
    expect(error.isNetworkError).toBe(true);
  });
});

describe('authentication', () => {
  it('sends the in-memory bearer token', async () => {
    const api = mockApi({ 'GET /api/x': { body: { ok: true } } });
    setAccessToken('abc');
    await apiRequest('/api/x');
    expect(api.calls[0]?.headers.Authorization).toBe('Bearer abc');
  });

  it('refreshes once on 401 and retries with the new token', async () => {
    const api = mockApi({
      'GET /api/x': ({ init }) =>
        (init.headers as Record<string, string>).Authorization === `Bearer ${TOKEN.access_token}`
          ? { body: { ok: true } }
          : { status: 401, body: { detail: 'Not authenticated' } },
      'POST /api/auth/refresh': { body: TOKEN },
    });
    setAccessToken('expired');
    await expect(apiRequest('/api/x')).resolves.toEqual({ ok: true });
    expect(api.callsTo('POST', '/api/auth/refresh')).toHaveLength(1);
  });

  it('never overlaps refreshes: concurrent 401s share one refresh call', async () => {
    // A second, overlapping refresh would replay a spent single-use token and
    // make the backend revoke the whole session.
    const api = mockApi({
      'GET /api/x': ({ init }) =>
        (init.headers as Record<string, string>).Authorization === `Bearer ${TOKEN.access_token}`
          ? { body: { ok: true } }
          : { status: 401, body: { detail: 'Not authenticated' } },
      'POST /api/auth/refresh': async () => {
        await new Promise((r) => setTimeout(r, 20));
        return { body: TOKEN };
      },
    });
    setAccessToken('expired');
    await Promise.all([apiRequest('/api/x'), apiRequest('/api/x'), apiRequest('/api/x')]);
    expect(api.callsTo('POST', '/api/auth/refresh')).toHaveLength(1);
  });

  it('signals a lost session when the refresh token is rejected', async () => {
    mockApi({
      'GET /api/x': { status: 401, body: { detail: 'Not authenticated' } },
      'POST /api/auth/refresh': {
        status: 401,
        body: { detail: 'Invalid or expired refresh token' },
      },
    });
    const lost = vi.fn();
    setAuthLostHandler(lost);
    setAccessToken('expired');
    await expect(apiRequest('/api/x')).rejects.toMatchObject({ status: 401 });
    expect(lost).toHaveBeenCalledTimes(1);
  });

  it('refresh resolves to null without a session and uses cookies', async () => {
    const api = mockApi({ 'POST /api/auth/refresh': { status: 401, body: { detail: 'x' } } });
    await expect(refreshAccessToken()).resolves.toBeNull();
    const init = api.fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.credentials).toBe('include');
  });

  it('does not attempt a refresh for unauthenticated calls such as login', async () => {
    const api = mockApi({
      'POST /api/auth/login': { status: 401, body: { detail: 'Invalid email or password' } },
    });
    await expect(
      apiRequest('/api/auth/login', { method: 'POST', json: {}, auth: false }),
    ).rejects.toMatchObject({ status: 401 });
    expect(api.calls).toHaveLength(1);
  });
});
