# "Continue with Google" (Phase 6, Day 27)

Google sign-in and sign-up on the login and registration pages. A Google
sign-in ends in **exactly the same application session** as a password login:
the same 15-minute access token (kept in memory by the SPA), the same single-use
rotating refresh cookie (`httpOnly`, `SameSite=Strict`, `Path=/api/auth`), the
same logout and replay detection. Nothing about the existing session design
changed.

**Status:** implemented and tested with Google's real verification library
against locally generated keys. **Live sign-in against Google has not been
verified**: no Google OAuth client was available for this work. See
"Verifying with a real Google client" below.

---

## 1. The flow

```
Browser                         Google                        API
   │  GET /api/auth/providers  ─────────────────────────────▶ │  {google_client_id}
   │  load accounts.google.com/gsi/client                     │
   │  user clicks Google's button ─▶ Google popup             │
   │  ◀── credential (an ID token, signed by Google) ──       │
   │  POST /api/auth/google {credential} ────────────────────▶│  verify (google-auth):
   │                                                          │   signature (Google's keys),
   │                                                          │   iss, aud = our client id,
   │                                                          │   exp/iat, email_verified
   │  ◀── 200/201 {access_token} + Set-Cookie refresh_token ──│  single-use (jti in Redis)
```

* **Google Identity Services (GIS)**, the ID-token ("Sign In With Google")
  flow. The browser receives an ID token from Google's own script and button.
  There is **no client secret** in this flow, and none is configured anywhere.
* The backend verifies the token with **`google-auth`**, Google's official
  Python library (`google.oauth2.id_token.verify_oauth2_token`): signature
  against Google's published keys, issuer `accounts.google.com`, audience equal
  to `GOOGLE_CLIENT_ID`, expiry and issue time (10 s skew). Then, in
  `app/google_auth.py`: a non-empty `sub` and `email`, and `email_verified`
  **exactly `true`**. The payload is never read any other way.
* Google's signing keys are fetched over HTTPS with a 5 s timeout and cached
  for their own `Cache-Control: max-age`.
* **Single use.** Each ID token's `jti` is claimed in Redis (`SET NX`) until the
  token's own expiry. A replayed credential is rejected; a token without a
  `jti` is refused rather than left replayable.
* The verified identity then gets the normal tokens from the same
  `_issue_tokens` helper `/login` and `/register` use.

## 2. Account policy (no silent linking)

| Situation | Result |
|---|---|
| An account already holds this Google `sub` | Signed in to it (`200`). The email Google reports now is not used to look it up: `sub` is the stable identity. |
| No account has this `sub`, and the email is free | A new account (`201`) with `google_sub` set and **no password** (`password_hash` is `NULL`; none is invented). |
| The email belongs to an existing account (a password account, or one tied to a different Google account) | `409 account_exists`. **A matching email is not treated as proof of owning that account.** The owner signs in with the method they already have. There is no linking flow yet. |
| Two first sign-ins with the same Google account at once | One account. The unique index on `users.google_sub` decides; the loser signs in to the winner's row. |
| A Google sign-up and a password sign-up race for one email | One wins (`201`); the other gets `409`. The unique index on `users.email` (CITEXT, case-insensitive) decides. |
| Password login to a Google-only account | `401 Invalid email or password`, with the same timing as an unknown email (verified against the decoy hash, then refused because the account has no password). |
| Password sign-up with a Google-only account's email | `409`, as for any existing email. |

Schema (migration `0008`): `users.google_sub TEXT UNIQUE NULL`,
`users.password_hash` becomes nullable, and
`CHECK (password_hash IS NOT NULL OR google_sub IS NOT NULL)` so no account can
exist without a way to sign in. Downgrading refuses while any Google-only
account exists, rather than deleting accounts or inventing passwords.

A note on disclosure: `409 account_exists` tells the caller that an account
uses that email. `/register` already reveals the same thing (it returns 409 for
an existing email), so this adds nothing new, and it requires a valid Google
token for that email.

## 3. Endpoints

### `GET /api/auth/providers`

Unauthenticated. `{"password": true, "google_client_id": "<id>" | null}`. The
client id is public by design (it is the `aud` of every token and the browser
needs it to render Google's button). `null` means Google sign-in is off.

### `POST /api/auth/google`

Body: `{"credential": "<ID token>"}` (1–4096 characters; anything else is 422).

| Status | `detail.code` | Meaning |
|---|---|---|
| `201` | — | New account; `TokenResponse` + refresh cookie |
| `200` | — | Existing account; `TokenResponse` + refresh cookie |
| `401` | `google_token_invalid` | Not a valid, unused Google ID token for this app. One generic message (`Google sign-in failed.`) for every reason: bad signature, unknown key, wrong issuer or audience, expired, malformed, replayed, no `jti`. |
| `403` | `google_email_unverified` | Google has not verified the account's email |
| `409` | `account_exists` | The email belongs to another account (see §2) |
| `422` | — | Malformed body |
| `503` | `google_not_configured` | `GOOGLE_CLIENT_ID` is not set |
| `503` | `google_unavailable` | Google's signing keys could not be fetched |

Errors use the `{code, message}` detail shape the SPA already parses. The
token is never logged. google-auth's own error messages can quote the token
(`MalformedError: Wrong number of segments in token: b'eyJ…'`), so the logged
reason is scrubbed of the credential and each of its segments first
(`app/google_auth.py::_scrubbed`, with tests).

Rate limiting: the endpoint has no password to guess, so the per-email login
lockout does not apply. General request rate limiting is still the plan's
Day-30 item, as for every other route.

## 4. Configuration

| Variable | Where | Value |
|---|---|---|
| `GOOGLE_CLIENT_ID` | backend (`.env` / environment) | The OAuth **client ID** of a Google Cloud "Web application" client, e.g. `1234-abc.apps.googleusercontent.com`. Unset or empty = Google sign-in off. A value that does not end in `.apps.googleusercontent.com` refuses to boot. |

There is deliberately **no** frontend variable: the SPA asks the backend
(`/api/auth/providers`), so the two can never disagree, and no `VITE_*`
variable is involved. **There is no client secret in this flow. Never put one
in `.env`, in a `VITE_*` variable, or in the repository.**

### Creating a Google OAuth client (manual, owner only)

1. In Google Cloud Console, choose or create a project.
2. **Google Auth Platform → Branding** (the OAuth consent screen): set the app
   name, support email and developer contact. While it is in *Testing*, add the
   Google accounts that may sign in as **test users**.
3. **Google Auth Platform → Clients → Create client**, type **Web application**.
4. **Authorized JavaScript origins.** Add the exact origin the SPA is served
   from. For local development add both:
   * `http://localhost:5173` (Vite dev server)
   * `http://localhost` (Google recommends it alongside the port-specific origin)

   and `http://localhost:4173` if you use `npm run preview`. Use `localhost`,
   not `127.0.0.1`, in the browser to match. Authorized redirect URIs are
   **not** needed: the popup flow does not redirect.
5. Copy the **Client ID** (not the secret; the secret is not used) into the
   backend environment as `GOOGLE_CLIENT_ID`, and restart the API.

### Verifying with a real Google client (not yet done)

With `GOOGLE_CLIENT_ID` set and the stack running (README, frontend §2):

1. Open `http://localhost:5173/register`. Google's own button should appear
   above the email form. If it shows "Google sign-in is unavailable right
   now", the script was blocked (an extension or a network filter).
2. Sign up with a test-user Google account. Expect `POST /api/auth/google`
   → `201`, a `Set-Cookie: refresh_token=…; HttpOnly; SameSite=Strict;
   Path=/api/auth`, then `GET /api/auth/me` → 200 and the dashboard.
3. Reload: the session restores via `/api/auth/refresh` like any other.
4. Sign out, then sign in again with Google: `200`, the same account.
5. Register a *password* account with another Google account's email, then try
   "Continue with Google" with that Google account: expect the 409 message, and
   check that the password account still signs in.
6. In the console, a GIS warning such as "origin is not allowed" means step 4's
   origin list does not match the address in the browser bar exactly.

## 5. Frontend

`frontend/src/components/GoogleSignIn.tsx` and `src/lib/googleIdentity.ts`.

* Rendered **only** when the server reports a client id. Otherwise nothing is
  shown, never a look-alike button.
* Google's script is loaded on demand from `accounts.google.com/gsi/client`.
  If it is blocked, fails, or takes longer than 10 s, a one-line note replaces
  it and the email form works as before.
* GIS is initialised once per page (a second `initialize` overrides the first),
  with `ux_mode: 'popup'` and `auto_select: false`; sign-out calls
  `disableAutoSelect()`.
* While the credential is being exchanged: "Signing in with Google…"
  (`role=status`), the email form is disabled, and Google's button is made
  `inert`, so a second credential is ignored. The email form's own submission
  likewise blocks the Google button.
* **Cancellation:** closing Google's popup sends the page nothing (GIS reports
  only completed sign-ins), so the UI stays idle and ready to try again. There
  is no "cancelled" message because there is no event to base one on.
* Errors: the server's message for `account_exists`, `google_email_unverified`
  and `google_unavailable`; one generic message otherwise.
* After success: the same internal-only redirect guard as password login
  (`lib/redirect.ts`).
* Nothing from Google is stored in the browser by this code; the access token
  stays in memory as before.
* Keyboard and focus: Google's button is Google's own iframe, with its own
  focus handling. That has **not** been checked in a real browser.

## 6. Tests

* `backend/tests/unit/test_google_auth.py`: **real** google-auth verification
  of tokens signed with a local RSA key, served by a fake key endpoint. Covers
  valid tokens; both issuer spellings; a foreign key, an unknown `kid`,
  expired, future-dated, wrong audience, wrong issuer, malformed and truncated
  tokens; a tampered payload; `alg: none`; `email_verified` false, missing,
  `"true"` or `1`; missing `sub` or `email`; no `jti`; size limits; a missing
  client id; unreachable keys; key caching by `max-age` and no caching of
  failures; reasons never quoting the credential.
* `backend/tests/integration/test_google_auth_api.py` (real Postgres + Redis +
  app; only the key fetch is faked): providers on and off; malformed client id
  refuses to boot; new account with a normal session and cookie flags and no
  password; repeat sign-in; identity keyed by `sub`; single use; no `jti`;
  wrong audience, issuer and expiry with one generic answer; bad signature;
  unverified email; malformed bodies; Google unreachable; the credential never
  logged; no linking to a password account (which still works); Google-only
  accounts refusing password login (including the decoy's plaintext);
  register conflict; 5 concurrent first sign-ins → one account; Google and
  password sign-up racing; refresh rotation and replay detection; logout;
  protected routes; password auth unchanged.
* `backend/tests/integration/test_migration_0008.py`: round trip, downgrade
  refusal, the credential CHECK, `google_sub` uniqueness.
* `frontend/src/components/googleSignIn.test.tsx`: hidden when off, Google's
  button with the server's client id, success → normal session → dashboard
  with no browser storage, busy state blocking both forms and a second
  credential, 409 explained, generic invalid message, cancellation changing
  nothing, blocked script → note and password login still working, the
  redirect guard, and sign-out disabling auto-select.
