import { Loader2 } from 'lucide-react';
import * as React from 'react';

import { fetchProviders } from '@/api/auth';
import { ApiError } from '@/api/client';
import { useAuth } from '@/auth/AuthProvider';
import { Alert } from '@/components/ui/alert';
import { attachGoogleHandler, loadGoogleIdentity } from '@/lib/googleIdentity';

type Phase =
  /** Asking the server whether Google sign-in is on. */
  | { kind: 'checking' }
  /** Not configured on this server: render nothing at all. */
  | { kind: 'off' }
  | { kind: 'loading'; clientId: string }
  | { kind: 'ready'; clientId: string }
  /** Google's script was blocked or failed: say so; email sign-in still works. */
  | { kind: 'unavailable' }
  /** The server could not be asked (unreachable or erroring): say so, offer no button. */
  | { kind: 'unknown' };

function messageFor(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.isNetworkError) return error.message;
    switch (error.code) {
      case 'account_exists':
      case 'google_email_unverified':
      case 'google_unavailable':
      case 'google_not_configured':
        return error.message;
      case 'google_token_invalid':
        return 'Google sign-in did not complete. Please try again.';
    }
    if (error.status === 429) return error.message;
  }
  return 'Google sign-in did not complete. Please try again.';
}

/**
 * "Continue with Google": Google's own button (rendered by Google's script),
 * shown only when the server reports a Google client id - never a look-alike.
 *
 * Google reports only a completed sign-in. Closing Google's window sends
 * nothing, so cancelling simply leaves this idle and ready to try again.
 */
export function GoogleSignIn({
  mode,
  disabled = false,
  onBusyChange,
  onSuccess,
}: {
  mode: 'signin' | 'signup';
  /** The email form is submitting: block Google until it settles. */
  disabled?: boolean;
  onBusyChange?: (busy: boolean) => void;
  onSuccess: () => void;
}) {
  const { signInWithGoogle } = useAuth();
  const [phase, setPhase] = React.useState<Phase>({ kind: 'checking' });
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const container = React.useRef<HTMLDivElement>(null);
  const inFlight = React.useRef(false);
  const mounted = React.useRef(true);
  // Latest props, read at callback time, so Google's button is rendered once
  // rather than on every parent render.
  const disabledRef = React.useRef(disabled);
  const onSuccessRef = React.useRef(onSuccess);
  const onBusyChangeRef = React.useRef(onBusyChange);
  disabledRef.current = disabled;
  onSuccessRef.current = onSuccess;
  onBusyChangeRef.current = onBusyChange;

  React.useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  // 1. Is Google sign-in configured on the server?
  React.useEffect(() => {
    const controller = new AbortController();
    fetchProviders(controller.signal)
      .then((providers) => {
        if (controller.signal.aborted) return;
        setPhase(
          providers.google_client_id
            ? { kind: 'loading', clientId: providers.google_client_id }
            : { kind: 'off' },
        );
      })
      .catch(() => {
        // Cannot tell. Offer no button that may not work, but say why it is
        // missing rather than failing silently.
        if (!controller.signal.aborted) setPhase({ kind: 'unknown' });
      });
    return () => controller.abort();
  }, []);

  const handleCredential = React.useCallback(
    async (credential: string) => {
      // One exchange at a time, and never while the email form is submitting.
      if (inFlight.current || disabledRef.current) return;
      inFlight.current = true;
      setBusy(true);
      setError(null);
      onBusyChangeRef.current?.(true);
      try {
        await signInWithGoogle(credential);
        if (mounted.current) onSuccessRef.current();
      } catch (err) {
        if (mounted.current) setError(messageFor(err));
      } finally {
        inFlight.current = false;
        if (mounted.current) {
          setBusy(false);
          onBusyChangeRef.current?.(false);
        }
      }
    },
    [signInWithGoogle],
  );

  // 2. Load Google's script and render its button.
  const clientId = phase.kind === 'loading' || phase.kind === 'ready' ? phase.clientId : null;
  React.useEffect(() => {
    if (!clientId) return;
    let cancelled = false;
    let detach: (() => void) | undefined;
    loadGoogleIdentity()
      .then((id) => {
        if (cancelled || !container.current) return;
        detach = attachGoogleHandler(
          id,
          clientId,
          (credential) => void handleCredential(credential),
        );
        const width = container.current.clientWidth;
        id.renderButton(container.current, {
          type: 'standard',
          theme: 'outline',
          size: 'large',
          text: mode === 'signup' ? 'signup_with' : 'continue_with',
          shape: 'rectangular',
          logo_alignment: 'center',
          // Google accepts 200-400 px; omitted when the width is unknown.
          ...(width > 0 ? { width: Math.max(200, Math.min(400, Math.floor(width))) } : {}),
        });
        setPhase({ kind: 'ready', clientId });
      })
      .catch(() => {
        if (!cancelled) setPhase({ kind: 'unavailable' });
      });
    return () => {
      cancelled = true;
      detach?.();
    };
  }, [clientId, mode, handleCredential]);

  // Block keyboard and pointer access to Google's button while locked.
  const locked = busy || disabled;
  React.useEffect(() => {
    container.current?.toggleAttribute('inert', locked);
  }, [locked]);

  if (phase.kind === 'off' || phase.kind === 'checking') return null;

  if (phase.kind === 'unknown') {
    return (
      <p className="text-xs text-muted-foreground">
        We could not check whether Google sign-in is available because the server did not respond.
        Reload the page to try again.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      {error && (
        <Alert variant="error" title="Google sign-in failed">
          {error}
        </Alert>
      )}
      {phase.kind === 'unavailable' ? (
        <p className="text-xs text-muted-foreground">
          Google sign-in is unavailable right now (it may be blocked by your browser or network).
          You can still use your email and password.
        </p>
      ) : (
        <div className="relative">
          {phase.kind === 'loading' && (
            <p role="status" className="text-xs text-muted-foreground">
              Loading Google sign-in…
            </p>
          )}
          <div
            ref={container}
            aria-disabled={locked || undefined}
            className={
              locked ? 'pointer-events-none flex justify-center opacity-60' : 'flex justify-center'
            }
            data-testid="google-button"
          />
          {busy && (
            <p
              role="status"
              className="mt-2 flex items-center justify-center gap-2 text-sm text-muted-foreground"
            >
              <Loader2 className="size-4 animate-spin" aria-hidden="true" />
              Signing in with Google…
            </p>
          )}
        </div>
      )}
      <div className="flex items-center gap-3" aria-hidden="true">
        <span className="h-px flex-1 bg-border" />
        <span className="text-xs uppercase tracking-wide text-muted-foreground">or</span>
        <span className="h-px flex-1 bg-border" />
      </div>
    </div>
  );
}
