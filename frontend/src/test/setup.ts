import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

import { setAccessToken, setAuthLostHandler } from '@/api/client';

afterEach(() => {
  cleanup();
  setAccessToken(null);
  setAuthLostHandler(null);
  vi.unstubAllGlobals();
});
