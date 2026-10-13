import { apiRequest, setAccessToken } from './client';
import type { TokenResponse, User } from './types';

/** backend/app/security.py */
export const MIN_PASSWORD_LENGTH = 8;
export const MAX_PASSWORD_BYTES = 1024;

export interface Credentials {
  email: string;
  password: string;
}

/** POST /api/auth/register -> 201 TokenResponse (+ refresh cookie). 409 if the email exists. */
export async function register(credentials: Credentials): Promise<void> {
  const tokens = await apiRequest<TokenResponse>('/api/auth/register', {
    method: 'POST',
    json: credentials,
    auth: false,
    withCredentials: true,
  });
  setAccessToken(tokens.access_token);
}

/** POST /api/auth/login -> TokenResponse (+ refresh cookie). 401 bad credentials, 429 lockout. */
export async function login(credentials: Credentials): Promise<void> {
  const tokens = await apiRequest<TokenResponse>('/api/auth/login', {
    method: 'POST',
    json: credentials,
    auth: false,
    withCredentials: true,
  });
  setAccessToken(tokens.access_token);
}

/** POST /api/auth/logout -> 204 always. Local state is cleared even if the call fails. */
export async function logout(): Promise<void> {
  try {
    await apiRequest<void>('/api/auth/logout', {
      method: 'POST',
      auth: false,
      withCredentials: true,
    });
  } finally {
    setAccessToken(null);
  }
}

/** GET /api/auth/me */
export function fetchMe(signal?: AbortSignal): Promise<User> {
  return apiRequest<User>('/api/auth/me', { signal });
}
