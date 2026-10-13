import * as React from 'react';

import * as authApi from '@/api/auth';
import { ApiError, refreshAccessToken, setAccessToken, setAuthLostHandler } from '@/api/client';
import type { User } from '@/api/types';

export type AuthStatus = 'loading' | 'authenticated' | 'anonymous';

export interface AuthContextValue {
  status: AuthStatus;
  user: User | null;
  /** True after a signed-in session was lost (expired or revoked refresh token). */
  sessionExpired: boolean;
  /** Set when the session could not be restored because the server was unreachable. */
  restoreError: string | null;
  signIn: (credentials: authApi.Credentials) => Promise<void>;
  signUp: (credentials: authApi.Credentials) => Promise<void>;
  signOut: () => Promise<void>;
  retryRestore: () => void;
}

const AuthContext = React.createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = React.useState<AuthStatus>('loading');
  const [user, setUser] = React.useState<User | null>(null);
  const [sessionExpired, setSessionExpired] = React.useState(false);
  const [restoreError, setRestoreError] = React.useState<string | null>(null);
  const [restoreAttempt, setRestoreAttempt] = React.useState(0);

  // Restore a session from the refresh cookie on load. Concurrent calls (React
  // StrictMode mounts effects twice in development) share one refresh request.
  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const token = await refreshAccessToken();
        if (!token) {
          if (!cancelled) setStatus('anonymous');
          return;
        }
        const me = await authApi.fetchMe();
        if (cancelled) return;
        setUser(me);
        setRestoreError(null);
        setStatus('authenticated');
      } catch (error) {
        if (cancelled) return;
        setAccessToken(null);
        setUser(null);
        setStatus('anonymous');
        setRestoreError(
          error instanceof ApiError && error.isNetworkError
            ? error.message
            : 'We could not restore your session. Please sign in.',
        );
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [restoreAttempt]);

  React.useEffect(() => {
    setAuthLostHandler(() => {
      setAccessToken(null);
      setUser(null);
      setSessionExpired(true);
      setStatus('anonymous');
    });
    return () => setAuthLostHandler(null);
  }, []);

  const finishSignIn = React.useCallback(async () => {
    const me = await authApi.fetchMe();
    setUser(me);
    setSessionExpired(false);
    setRestoreError(null);
    setStatus('authenticated');
  }, []);

  const signIn = React.useCallback(
    async (credentials: authApi.Credentials) => {
      await authApi.login(credentials);
      await finishSignIn();
    },
    [finishSignIn],
  );

  const signUp = React.useCallback(
    async (credentials: authApi.Credentials) => {
      await authApi.register(credentials);
      await finishSignIn();
    },
    [finishSignIn],
  );

  const signOut = React.useCallback(async () => {
    try {
      await authApi.logout();
    } catch {
      // The local session is cleared regardless (authApi.logout's finally).
    }
    setUser(null);
    setSessionExpired(false);
    setStatus('anonymous');
  }, []);

  const retryRestore = React.useCallback(() => {
    setStatus('loading');
    setRestoreError(null);
    setRestoreAttempt((n) => n + 1);
  }, []);

  const value = React.useMemo<AuthContextValue>(
    () => ({
      status,
      user,
      sessionExpired,
      restoreError,
      signIn,
      signUp,
      signOut,
      retryRestore,
    }),
    [status, user, sessionExpired, restoreError, signIn, signUp, signOut, retryRestore],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = React.useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used inside <AuthProvider>');
  return context;
}
