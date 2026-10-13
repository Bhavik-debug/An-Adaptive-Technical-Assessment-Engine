import * as React from 'react';

import { ApiError } from '@/api/client';

export interface ApiData<T> {
  data: T | undefined;
  error: ApiError | null;
  loading: boolean;
  reload: () => void;
}

function toApiError(error: unknown): ApiError {
  return error instanceof ApiError ? error : new ApiError(-1, 'Something went wrong.');
}

/**
 * Load data on mount (and on `reload`). The request is aborted when the
 * component unmounts or a newer load starts, so a slow, stale response can
 * never overwrite a fresh one.
 */
export function useApiData<T>(
  load: (signal: AbortSignal) => Promise<T>,
  deps: React.DependencyList,
): ApiData<T> {
  const [data, setData] = React.useState<T | undefined>(undefined);
  const [error, setError] = React.useState<ApiError | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [attempt, setAttempt] = React.useState(0);

  React.useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    load(controller.signal)
      .then((value) => {
        if (!controller.signal.aborted) setData(value);
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted) setError(toApiError(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- callers own `deps`
  }, [...deps, attempt]);

  const reload = React.useCallback(() => setAttempt((n) => n + 1), []);
  return { data, error, loading, reload };
}
