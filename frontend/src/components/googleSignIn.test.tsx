import { act, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import * as gis from '@/lib/googleIdentity';
import { mockApi, OPTIONS, renderApp, TOKEN, USER } from '@/test/utils';

// Google's script is never loaded in tests: `loadGoogleIdentity` is replaced
// per test with a fake `google.accounts.id`, or with a failure.
vi.mock('@/lib/googleIdentity', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/googleIdentity')>();
  return { ...actual, loadGoogleIdentity: vi.fn() };
});

const CLIENT_ID = '123-test.apps.googleusercontent.com';
const anonymous = { 'POST /api/auth/refresh': { status: 401, body: { detail: 'x' } } };
const googleOn = {
  'GET /api/auth/providers': { body: { password: true, google_client_id: CLIENT_ID } },
};
const dashboard = {
  'GET /api/sessions': { body: { sessions: [] } },
  'GET /api/setup-options': { body: OPTIONS },
};

function fakeGoogle() {
  let callback: ((r: { credential?: string }) => void) | undefined;
  const id = {
    initialize: vi.fn((config: { callback: (r: { credential?: string }) => void }) => {
      callback = config.callback;
    }),
    renderButton: vi.fn((parent: HTMLElement) => {
      // Stands in for Google's iframe button.
      const button = document.createElement('div');
      button.textContent = 'Google button';
      parent.appendChild(button);
    }),
    disableAutoSelect: vi.fn(),
  };
  vi.mocked(gis.loadGoogleIdentity).mockResolvedValue(id);
  return {
    id,
    /** What Google does when the user completes its flow. */
    complete: (credential = 'google-id-token') => act(() => callback?.({ credential })),
  };
}

const location = () => screen.getByTestId('location').textContent;

beforeEach(() => {
  gis.resetGoogleIdentityForTests();
  vi.mocked(gis.loadGoogleIdentity).mockReset();
});

describe('Continue with Google', () => {
  it('is not shown at all when the server has it switched off', async () => {
    mockApi({
      ...anonymous,
      'GET /api/auth/providers': { body: { password: true, google_client_id: null } },
    });
    renderApp('/login');
    await screen.findByRole('heading', { name: 'Sign in' });
    await waitFor(() => expect(gis.loadGoogleIdentity).not.toHaveBeenCalled());
    expect(screen.queryByTestId('google-button')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeEnabled();
  });

  it("renders Google's own button with the server's client id", async () => {
    const google = fakeGoogle();
    mockApi({ ...anonymous, ...googleOn });
    renderApp('/register');
    expect(await screen.findByText('Google button')).toBeInTheDocument();
    expect(google.id.initialize).toHaveBeenCalledWith(
      expect.objectContaining({ client_id: CLIENT_ID, ux_mode: 'popup', auto_select: false }),
    );
    expect(google.id.renderButton).toHaveBeenCalledWith(
      screen.getByTestId('google-button'),
      expect.objectContaining({ text: 'signup_with' }),
    );
  });

  it('exchanges the credential for a normal session and goes to the dashboard', async () => {
    const google = fakeGoogle();
    const api = mockApi({
      ...anonymous,
      ...googleOn,
      ...dashboard,
      'POST /api/auth/google': { status: 201, body: TOKEN },
      'GET /api/auth/me': { body: USER },
    });
    renderApp('/login');
    await screen.findByText('Google button');
    await google.complete('the-credential');

    expect(await screen.findByRole('heading', { name: 'Welcome' })).toBeInTheDocument();
    expect(location()).toBe('/dashboard');
    expect(api.callsTo('POST', '/api/auth/google')[0]?.body).toEqual({
      credential: 'the-credential',
    });
    // The app session is the ordinary one: the bearer token from the response.
    expect(api.callsTo('GET', '/api/auth/me')[0]?.headers.Authorization).toBe('Bearer access-1');
    // Nothing is persisted in the browser.
    expect(window.localStorage.length).toBe(0);
    expect(window.sessionStorage.length).toBe(0);
  });

  it('shows progress, blocks the email form, and ignores a second credential', async () => {
    const google = fakeGoogle();
    let release!: () => void;
    const api = mockApi({
      ...anonymous,
      ...googleOn,
      ...dashboard,
      'POST /api/auth/google': () =>
        new Promise((resolve) => {
          release = () => resolve({ body: TOKEN });
        }),
      'GET /api/auth/me': { body: USER },
    });
    renderApp('/login');
    await screen.findByText('Google button');
    await google.complete('one');
    expect(await screen.findByText('Signing in with Google…')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeDisabled();
    expect(screen.getByLabelText('Email')).toBeDisabled();
    expect(screen.getByTestId('google-button')).toHaveAttribute('inert');
    await google.complete('two');
    expect(api.callsTo('POST', '/api/auth/google')).toHaveLength(1);
    release();
    await screen.findByRole('heading', { name: 'Welcome' });
  });

  it('explains an existing password account instead of linking it', async () => {
    const google = fakeGoogle();
    mockApi({
      ...anonymous,
      ...googleOn,
      'POST /api/auth/google': {
        status: 409,
        body: {
          detail: {
            code: 'account_exists',
            message: 'An account with this email already exists. Sign in the way you created it.',
          },
        },
      },
    });
    renderApp('/login');
    await screen.findByText('Google button');
    await google.complete();
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'An account with this email already exists',
    );
    expect(location()).toBe('/login');
    // The password form is still usable.
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeEnabled();
  });

  it('gives one generic message for a rejected credential', async () => {
    const google = fakeGoogle();
    mockApi({
      ...anonymous,
      ...googleOn,
      'POST /api/auth/google': {
        status: 401,
        body: { detail: { code: 'google_token_invalid', message: 'Google sign-in failed.' } },
      },
    });
    renderApp('/login');
    await screen.findByText('Google button');
    await google.complete();
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Google sign-in did not complete. Please try again.',
    );
  });

  it('cancelling (closing Google’s window) changes nothing', async () => {
    fakeGoogle(); // Google sends no callback when its window is closed.
    const api = mockApi({ ...anonymous, ...googleOn });
    renderApp('/login');
    await screen.findByText('Google button');
    expect(api.callsTo('POST', '/api/auth/google')).toHaveLength(0);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.queryByText('Signing in with Google…')).not.toBeInTheDocument();
  });

  it('degrades to a note, never a fake button, when Google’s script is blocked', async () => {
    vi.mocked(gis.loadGoogleIdentity).mockRejectedValue(new Error('blocked'));
    const api = mockApi({
      ...anonymous,
      ...googleOn,
      ...dashboard,
      'POST /api/auth/login': { body: TOKEN },
      'GET /api/auth/me': { body: USER },
    });
    renderApp('/login');
    expect(await screen.findByText(/Google sign-in is unavailable right now/)).toBeInTheDocument();
    expect(screen.queryByText('Google button')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /google/i })).not.toBeInTheDocument();

    const user = userEvent.setup();
    await user.type(screen.getByLabelText('Email'), 'candidate@example.com');
    await user.type(screen.getByLabelText('Password'), 'correct-horse');
    await user.click(screen.getByRole('button', { name: 'Sign in' }));
    await screen.findByRole('heading', { name: 'Welcome' });
    expect(api.callsTo('POST', '/api/auth/login')).toHaveLength(1);
  });

  it('keeps the internal redirect guard after a Google sign-in', async () => {
    const google = fakeGoogle();
    mockApi({
      ...anonymous,
      ...googleOn,
      ...dashboard,
      'POST /api/auth/google': { body: TOKEN },
      'GET /api/auth/me': { body: USER },
    });
    // A protected deep link sends the visitor to sign-in with `from` set.
    renderApp('/interviews/new');
    await screen.findByRole('heading', { name: 'Sign in' });
    await screen.findByText('Google button');
    await google.complete();
    await waitFor(() => expect(location()).toBe('/interviews/new'));
  });

  it('sign-out stops Google from silently re-selecting the account', async () => {
    const google = fakeGoogle();
    mockApi({
      ...anonymous,
      ...googleOn,
      ...dashboard,
      'POST /api/auth/google': { body: TOKEN },
      'GET /api/auth/me': { body: USER },
      'POST /api/auth/logout': { status: 204 },
    });
    renderApp('/login');
    await screen.findByText('Google button');
    await google.complete();
    await screen.findByRole('heading', { name: 'Welcome' });
    // The script is loaded at this point, so the real helper finds it on window.
    window.google = { accounts: { id: google.id } };
    await userEvent.setup().click(screen.getByRole('button', { name: /sign out/i }));
    await screen.findByRole('heading', { name: 'Sign in' });
    expect(google.id.disableAutoSelect).toHaveBeenCalled();
    delete window.google;
  });
});
