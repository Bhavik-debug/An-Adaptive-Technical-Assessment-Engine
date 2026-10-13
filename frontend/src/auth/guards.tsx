import { Navigate, Outlet, useLocation } from 'react-router';

import { FullPageSpinner } from '@/components/FullPageSpinner';
import { safeRedirectPath } from '@/lib/redirect';

import { useAuth } from './AuthProvider';

export interface LoginRedirectState {
  from?: string;
}

/** Protected screens: render only for a signed-in user, else go to /login. */
export function RequireAuth() {
  const { status } = useAuth();
  const location = useLocation();
  if (status === 'loading') return <FullPageSpinner label="Restoring your session" />;
  if (status !== 'authenticated') {
    const state: LoginRedirectState = { from: location.pathname + location.search };
    return <Navigate to="/login" replace state={state} />;
  }
  return <Outlet />;
}

/** Sign-in/sign-up screens: a signed-in user goes straight to the app. */
export function PublicOnly() {
  const { status } = useAuth();
  const location = useLocation();
  if (status === 'loading') return <FullPageSpinner label="Restoring your session" />;
  if (status === 'authenticated') {
    const from = (location.state as LoginRedirectState | null)?.from;
    return <Navigate to={safeRedirectPath(from)} replace />;
  }
  return <Outlet />;
}
