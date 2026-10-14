# Adaptive Interviewer — frontend

React 18 + Vite + TypeScript + Tailwind CSS (shadcn/ui conventions).
Day 26 scope: sign-in/sign-up, dashboard, and the three-step interview setup
wizard. The interview and report screens (Days 27–28) are placeholders.

Full documentation: [`../docs/frontend.md`](../docs/frontend.md).

## Run locally

Start the backend first (see the root README), then:

```bash
npm ci
npm run dev        # http://localhost:5173 — /api is proxied to the backend
```

If the backend is not on `http://localhost:8000`, set `API_PROXY_TARGET`
(copy `.env.example` to `.env.local`). The SPA must be served from the API's
origin: the refresh cookie is `SameSite=Strict` and the backend has no CORS.

## Checks

```bash
npm run typecheck
npm run lint
npm run format:check
npm test
npm run build      # local dist/ only — nothing is deployed
```

## Rules for this codebase

- All HTTP goes through `src/api/client.ts`.
- The access token stays in memory. Never store it, never log it, and never
  log resume or profile content (`no-console` is a lint error).
- Every `VITE_*` variable is public. Never put a secret in one.
- Show only options the backend accepts (`GET /api/setup-options`), and never
  show sample data as if it were a user's results.
