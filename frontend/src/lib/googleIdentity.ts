/**
 * Google Identity Services (GIS), loaded on demand from Google.
 *
 * GIS renders Google's own "Continue with Google" button and, when the user
 * completes Google's flow, hands the page a *credential* (an ID token). The
 * page never verifies or reads it: it goes to POST /api/auth/google, and the
 * backend verifies it with Google's library. No client secret exists in this
 * flow, and nothing Google returns is stored in the browser by this code.
 *
 * GIS must be initialised once per page; a second `initialize` overrides the
 * first and logs a warning. So it is initialised once per client id, and the
 * credential is routed to whichever button is currently mounted.
 */

export const GIS_SCRIPT_URL = 'https://accounts.google.com/gsi/client';
const LOAD_TIMEOUT_MS = 10_000;

interface CredentialResponse {
  credential?: string;
}

interface GsiButtonOptions {
  type: 'standard';
  theme: 'outline';
  size: 'large';
  text: 'continue_with' | 'signup_with' | 'signin_with';
  shape: 'rectangular';
  logo_alignment: 'left' | 'center';
  width?: number;
}

export interface GoogleAccountsId {
  initialize(config: {
    client_id: string;
    callback: (response: CredentialResponse) => void;
    ux_mode?: 'popup';
    auto_select?: boolean;
    cancel_on_tap_outside?: boolean;
    itp_support?: boolean;
  }): void;
  renderButton(parent: HTMLElement, options: GsiButtonOptions): void;
  disableAutoSelect(): void;
}

declare global {
  interface Window {
    google?: { accounts?: { id?: GoogleAccountsId } };
  }
}

let loading: Promise<GoogleAccountsId> | null = null;
let initialisedFor: string | null = null;
let handler: ((credential: string) => void) | null = null;

function loaded(): GoogleAccountsId | undefined {
  return typeof window !== 'undefined' ? window.google?.accounts?.id : undefined;
}

/** Load Google's script once. Rejects if it is blocked, fails, or takes too long. */
export function loadGoogleIdentity(): Promise<GoogleAccountsId> {
  const ready = loaded();
  if (ready) return Promise.resolve(ready);
  if (loading) return loading;
  loading = new Promise<GoogleAccountsId>((resolve, reject) => {
    const script = document.createElement('script');
    script.src = GIS_SCRIPT_URL;
    script.async = true;
    script.defer = true;
    const fail = (reason: string) => {
      clearTimeout(timer);
      script.remove();
      loading = null; // allow a later retry (e.g. after a network blip)
      reject(new Error(reason));
    };
    const timer = setTimeout(() => fail('Google sign-in took too long to load.'), LOAD_TIMEOUT_MS);
    script.onload = () => {
      const id = loaded();
      if (id) {
        clearTimeout(timer);
        resolve(id);
      } else fail('Google sign-in did not initialise.');
    };
    script.onerror = () => fail('Google sign-in could not be loaded.');
    document.head.appendChild(script);
  });
  return loading;
}

/** Initialise GIS for `clientId` (once) and route credentials to `onCredential`. */
export function attachGoogleHandler(
  id: GoogleAccountsId,
  clientId: string,
  onCredential: (credential: string) => void,
): () => void {
  handler = onCredential;
  if (initialisedFor !== clientId) {
    id.initialize({
      client_id: clientId,
      callback: (response) => {
        if (response.credential) handler?.(response.credential);
      },
      ux_mode: 'popup',
      auto_select: false,
      cancel_on_tap_outside: true,
      itp_support: true,
    });
    initialisedFor = clientId;
  }
  return () => {
    if (handler === onCredential) handler = null;
  };
}

/** After sign-out, stop Google from silently re-selecting the account. */
export function disableGoogleAutoSelect(): void {
  try {
    loaded()?.disableAutoSelect();
  } catch {
    // Google's script is optional; sign-out never depends on it.
  }
}

/** Test-only: forget module state between tests. */
export function resetGoogleIdentityForTests(): void {
  loading = null;
  initialisedFor = null;
  handler = null;
}
