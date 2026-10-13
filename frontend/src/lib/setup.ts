import type { CandidateProfile, JobDescription, ProfileSkill, SetupOptions } from '@/api/types';

const DURATION_CHOICES = [15, 30, 45, 60, 90];

/** The JD's topics with a positive weight, most emphasised first ([] unless ready). */
export function jdTopicKeys(jd: JobDescription | null): string[] {
  if (!jd || jd.status !== 'ready') return [];
  return Object.entries(jd.topic_weights)
    .filter(([, weight]) => weight > 0)
    .sort(([, a], [, b]) => b - a)
    .map(([key]) => key);
}

/** Common lengths within the backend's bounds, plus its default. */
export function durationChoices(options: SetupOptions): number[] {
  const set = new Set(
    [...DURATION_CHOICES, options.default_duration_minutes].filter(
      (m) => m >= options.min_duration_minutes && m <= options.max_duration_minutes,
    ),
  );
  return [...set].sort((a, b) => a - b);
}

/** The PATCH body must match CandidateProfile exactly (extra fields are rejected). */
export function initialProfile(
  profile: CandidateProfile | null,
  suggestion: Partial<CandidateProfile> | null,
): CandidateProfile {
  const source = profile ?? suggestion ?? {};
  return {
    headline: source.headline ?? null,
    skills: (source.skills ?? []) as ProfileSkill[],
    years_experience: source.years_experience ?? null,
  };
}
