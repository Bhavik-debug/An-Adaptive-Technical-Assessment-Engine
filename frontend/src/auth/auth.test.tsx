import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import { mockApi, OPTIONS, renderApp, signedInRoutes, TOKEN, USER } from '@/test/utils';

const anonymous = { 'POST /api/auth/refresh': { status: 401, body: { detail: 'x' } } };
const dashboardData = {
  'GET /api/sessions': { body: { sessions: [] } },
  'GET /api/setup-options': { body: OPTIONS },
};

const location = () => screen.getByTestId('location').textContent;

describe('protected routes', () => {
  it('sends an unauthenticated visitor to the sign-in page', async () => {
    mockApi(anonymous);
    renderApp('/dashboard');
    expect(await screen.findByRole('heading', { name: 'Sign in' })).toBeInTheDocument();
    expect(location()).toBe('/login');
  });

  it('protects the setup wizard and placeholder routes too', async () => {
    mockApi(anonymous);
    renderApp('/interviews/new');
    await screen.findByRole('heading', { name: 'Sign in' });
    expect(location()).toBe('/login');
  });

  it('restores a session from the refresh cookie without a sign-in', async () => {
    const api = mockApi({ ...signedInRoutes(), ...dashboardData });
    renderApp('/dashboard');
    expect(await screen.findByRole('heading', { name: 'Welcome' })).toBeInTheDocument();
    expect(api.callsTo('POST', '/api/auth/refresh')).toHaveLength(1);
  });

  it('sends a signed-in user away from the sign-in page', async () => {
    mockApi({ ...signedInRoutes(), ...dashboardData });
    renderApp('/login');
    await screen.findByRole('heading', { name: 'Welcome' });
    expect(location()).toBe('/dashboard');
  });

  it('returns to sign-in with a notice when the session is revoked mid-use', async () => {
    mockApi({
      'POST /api/auth/refresh': [{ body: TOKEN }, { status: 401, body: { detail: 'x' } }],
      'GET /api/auth/me': { body: USER },
      'GET /api/setup-options': { body: OPTIONS },
      'GET /api/sessions': { status: 401, body: { detail: 'Not authenticated' } },
    });
    renderApp('/dashboard');
    expect(await screen.findByText('Your session has ended')).toBeInTheDocument();
    expect(location()).toBe('/login');
  });
});

describe('sign in', () => {
  it('validates fields before calling the API', async () => {
    const api = mockApi(anonymous);
    renderApp('/login');
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: 'Sign in' }));
    expect(screen.getByText('Enter your email address.')).toBeInTheDocument();
    expect(screen.getByText('Enter your password.')).toBeInTheDocument();
    expect(screen.getByLabelText('Email')).toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByLabelText('Email')).toHaveFocus();
    expect(api.callsTo('POST', '/api/auth/login')).toHaveLength(0);
  });

  it('shows one message for wrong credentials and clears the password', async () => {
    mockApi({
      ...anonymous,
      'POST /api/auth/login': { status: 401, body: { detail: 'Invalid email or password' } },
    });
    renderApp('/login');
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText('Email'), 'a@b.co');
    await user.type(screen.getByLabelText('Password'), 'wrong-password');
    await user.click(screen.getByRole('button', { name: 'Sign in' }));
    expect(await screen.findByText('The email or password is incorrect.')).toBeInTheDocument();
    expect(screen.getByLabelText('Password')).toHaveValue('');
  });

  it('shows the lockout message from the server', async () => {
    mockApi({
      ...anonymous,
      'POST /api/auth/login': {
        status: 429,
        body: { detail: 'Too many failed attempts. Try again in 15 minutes.' },
      },
    });
    renderApp('/login');
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText('Email'), 'a@b.co');
    await user.type(screen.getByLabelText('Password'), 'whatever');
    await user.click(screen.getByRole('button', { name: 'Sign in' }));
    expect(await screen.findByText(/Too many failed attempts/)).toBeInTheDocument();
  });

  it('signs in and returns to the page that was requested', async () => {
    const api = mockApi({
      ...anonymous,
      ...dashboardData,
      'POST /api/auth/login': { body: TOKEN },
      'GET /api/auth/me': { body: USER },
    });
    renderApp('/interviews/new');
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText('Email'), USER.email);
    await user.type(screen.getByLabelText('Password'), 'correct horse 9');
    await user.click(screen.getByRole('button', { name: 'Sign in' }));
    await waitFor(() => expect(location()).toBe('/interviews/new'));
    const login = api.callsTo('POST', '/api/auth/login')[0];
    expect(login?.body).toEqual({ email: USER.email, password: 'correct horse 9' });
    const me = api.callsTo('GET', '/api/auth/me')[0];
    expect(me?.headers.Authorization).toBe(`Bearer ${TOKEN.access_token}`);
  });
});

describe('sign up', () => {
  it('checks password length and confirmation', async () => {
    const api = mockApi(anonymous);
    renderApp('/register');
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText('Email'), 'new@example.com');
    await user.type(screen.getByLabelText('Password'), 'short');
    await user.type(screen.getByLabelText('Confirm password'), 'different');
    await user.click(screen.getByRole('button', { name: 'Create account' }));
    expect(screen.getByText('Use at least 8 characters.')).toBeInTheDocument();
    expect(screen.getByText('The passwords do not match.')).toBeInTheDocument();
    expect(api.callsTo('POST', '/api/auth/register')).toHaveLength(0);
  });

  it('reports an existing account on the email field', async () => {
    mockApi({
      ...anonymous,
      'POST /api/auth/register': {
        status: 409,
        body: { detail: 'An account with that email already exists' },
      },
    });
    renderApp('/register');
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText('Email'), 'taken@example.com');
    await user.type(screen.getByLabelText('Password'), 'long enough pw');
    await user.type(screen.getByLabelText('Confirm password'), 'long enough pw');
    await user.click(screen.getByRole('button', { name: 'Create account' }));
    expect(
      await screen.findByText('An account with this email already exists.'),
    ).toBeInTheDocument();
  });

  it('creates the account and opens the dashboard', async () => {
    mockApi({
      ...anonymous,
      ...dashboardData,
      'POST /api/auth/register': { status: 201, body: TOKEN },
      'GET /api/auth/me': { body: USER },
    });
    renderApp('/register');
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText('Email'), USER.email);
    await user.type(screen.getByLabelText('Password'), 'long enough pw');
    await user.type(screen.getByLabelText('Confirm password'), 'long enough pw');
    await user.click(screen.getByRole('button', { name: 'Create account' }));
    expect(await screen.findByRole('heading', { name: 'Welcome' })).toBeInTheDocument();
  });
});

describe('sign out', () => {
  it('calls logout and returns to sign-in', async () => {
    const api = mockApi({
      ...signedInRoutes(),
      ...dashboardData,
      'POST /api/auth/logout': { status: 204 },
    });
    renderApp('/dashboard');
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: 'Sign out' }));
    await screen.findByRole('heading', { name: 'Sign in' });
    expect(api.callsTo('POST', '/api/auth/logout')).toHaveLength(1);
    expect(screen.queryByText('Your session has ended')).not.toBeInTheDocument();
  });
});
