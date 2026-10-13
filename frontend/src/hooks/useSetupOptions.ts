import * as React from 'react';

import { getSetupOptions } from '@/api/setup';
import type { SetupOptions } from '@/api/types';

import { useApiData } from './useApiData';

export function useSetupOptions() {
  const result = useApiData<SetupOptions>((signal) => getSetupOptions(signal), []);
  const options = result.data;

  const topicNames = React.useMemo(
    () => new Map(options?.topics.map((t) => [t.key, t.display_name]) ?? []),
    [options],
  );
  const subtopicNames = React.useMemo(
    () =>
      new Map(
        options?.topics.flatMap((t) => t.subtopics.map((s) => [s.key, s.display_name] as const)) ??
          [],
      ),
    [options],
  );

  return { ...result, topicNames, subtopicNames };
}
