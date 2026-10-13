/// <reference types="vitest/config" />
import path from 'node:path';

import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';

// The dev server proxies /api to the FastAPI backend so the SPA and the API
// share one origin. That is required, not a convenience: the backend sets its
// refresh cookie with SameSite=Strict and Path=/api/auth, and has no CORS
// middleware. API_PROXY_TARGET has no VITE_ prefix, so it never reaches the
// browser bundle.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.API_PROXY_TARGET || 'http://localhost:8000';
  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: { '@': path.resolve(__dirname, './src') },
    },
    server: {
      port: 5173,
      strictPort: true,
      proxy: { '/api': { target, changeOrigin: false } },
    },
    preview: {
      port: 4173,
      proxy: { '/api': { target, changeOrigin: false } },
    },
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: ['./src/test/setup.ts'],
      css: false,
      restoreMocks: true,
    },
  };
});
