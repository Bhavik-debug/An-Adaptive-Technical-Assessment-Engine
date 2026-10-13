import { describe, expect, it } from 'vitest';

import type { JobDescription } from '@/api/types';
import { OPTIONS } from '@/test/utils';

import { safeRedirectPath } from './redirect';
import { sessionStatus } from './session';
import { durationChoices, initialProfile, jdTopicKeys } from './setup';
import { validateEmail, validateLoginPassword, validateNewPassword } from './validation';

describe('validation', () => {
  it('validates email', () => {
    expect(validateEmail('')).toMatch(/enter your email/i);
    expect(validateEmail('not-an-email')).toMatch(/valid email/i);
    expect(validateEmail(' a@b.co ')).toBeNull();
  });

  it('enforces the backend password rules', () => {
    expect(validateNewPassword('')).toMatch(/choose/i);
    expect(validateNewPassword('short')).toMatch(/at least 8/);
    expect(validateNewPassword('long enough')).toBeNull();
    expect(validateNewPassword('x'.repeat(1025))).toMatch(/too long/);
    // Login never rejects a short password: that would leak what is "too short".
    expect(validateLoginPassword('abc')).toBeNull();
    expect(validateLoginPassword('')).toMatch(/enter your password/i);
  });
});

describe('safeRedirectPath', () => {
  it.each([
    ['/interviews/new', '/interviews/new'],
    ['https://evil.example', '/dashboard'],
    ['//evil.example', '/dashboard'],
    ['/\\evil.example', '/dashboard'],
    ['/login', '/dashboard'],
    [undefined, '/dashboard'],
  ])('%s -> %s', (input, expected) => {
    expect(safeRedirectPath(input)).toBe(expected);
  });
});

describe('sessionStatus', () => {
  it('collapses internal engine states into "In progress"', () => {
    for (const state of ['PREPARING', 'ASKING', 'GRADING', 'PROBING', 'SELECTING']) {
      expect(sessionStatus(state).label).toBe('In progress');
    }
    expect(sessionStatus('CREATED').label).toBe('Not started');
    expect(sessionStatus('COMPLETED').label).toBe('Completed');
    expect(sessionStatus('REPORTING').label).toBe('Preparing report');
  });
});

describe('setup helpers', () => {
  const jd = (overrides: Partial<JobDescription>): JobDescription => ({
    jd_id: 'jd',
    status: 'ready',
    error: null,
    redactions: {},
    title: null,
    requirements: [],
    topic_weights: {},
    min_years_experience: null,
    created_at: '',
    parsed_at: null,
    ...overrides,
  });

  it('orders JD topics by weight and ignores unready JDs', () => {
    expect(jdTopicKeys(jd({ topic_weights: { dsa: 0.3, databases: 0.7, caching: 0 } }))).toEqual([
      'databases',
      'dsa',
    ]);
    expect(jdTopicKeys(jd({ status: 'pending', topic_weights: { dsa: 1 } }))).toEqual([]);
    expect(jdTopicKeys(null)).toEqual([]);
  });

  it('offers only durations within the backend bounds', () => {
    expect(durationChoices(OPTIONS)).toEqual([15, 30, 45, 60, 90]);
    expect(
      durationChoices({ ...OPTIONS, min_duration_minutes: 20, max_duration_minutes: 50 }),
    ).toEqual([30, 45]);
  });

  it('builds a PATCH body with only CandidateProfile fields', () => {
    const suggestion = {
      headline: 'Engineer',
      skills: [{ name: 'SQL', evidence: 'professional' as const, subtopic_keys: ['indexing'] }],
      years_experience: 4,
      verification: { dropped: 1 },
    };
    expect(initialProfile(null, suggestion)).toEqual({
      headline: 'Engineer',
      skills: suggestion.skills,
      years_experience: 4,
    });
    expect(initialProfile(null, null)).toEqual({
      headline: null,
      skills: [],
      years_experience: null,
    });
  });
});
