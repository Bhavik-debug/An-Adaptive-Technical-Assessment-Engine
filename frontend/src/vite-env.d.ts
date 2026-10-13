/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Public. Empty means same origin (the Vite proxy). Never put a secret in a VITE_* variable. */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
