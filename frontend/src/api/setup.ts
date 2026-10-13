import { apiRequest } from './client';
import type { SetupOptions } from './types';

/** GET /api/setup-options -> topics, levels, modes and duration bounds the backend accepts. */
export function getSetupOptions(signal?: AbortSignal): Promise<SetupOptions> {
  return apiRequest<SetupOptions>('/api/setup-options', { signal });
}
