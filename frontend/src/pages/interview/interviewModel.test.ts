import { describe, expect, it } from 'vitest';

import type { SessionDetail, StreamEvent } from '@/api/types';

import { answerLength, splitFences, validateAnswer } from './content';
import {
  acceptsAnswer,
  applyEvent,
  fromSnapshot,
  questionLabel,
  snapshotIsCurrent,
  withTurnStatus,
  type InterviewView,
} from './interviewModel';

function detail(overrides: Partial<SessionDetail> = {}): SessionDetail {
  return {
    session_id: 's1',
    state: 'ASKING',
    seq: 2,
    role: 'Backend engineer',
    level: 'mid',
    mode: 'standard',
    topic_keys: ['dsa'],
    started_at: '2026-10-14T09:00:00Z',
    ended_at: null,
    report_status: null,
    question: { question_id: 'q1', is_follow_up: false, text: 'First?' },
    questions_asked: 1,
    follow_ups_asked: 0,
    current_turn_id: null,
    stream_url: '/api/sessions/s1/stream',
    turns_url: '/api/sessions/s1/turns',
    ...overrides,
  };
}

const ev = (e: Record<string, unknown>) => e as unknown as StreamEvent;

function run(view: InterviewView, events: StreamEvent[]) {
  return events.reduce((v, e) => applyEvent(v, e).view, view);
}

describe('applyEvent', () => {
  it('follows a full turn: submitted, graded, next question', () => {
    const start = fromSnapshot(detail());
    const after = run(start, [
      ev({ type: 'grading_started', seq: 3, state: 'GRADING', turn_id: 't1' }),
      ev({ type: 'grade_ready', seq: 4, state: 'GRADING', turn_id: 't1', score: 0.9 }),
      ev({ type: 'state_changed', seq: 5, state: 'SELECTING', event_type: 'SELECTION_STARTED' }),
      ev({
        type: 'next_question',
        seq: 6,
        state: 'ASKING',
        question_id: 'q2',
        is_follow_up: false,
        text: 'Second?',
      }),
    ]);
    expect(after.seq).toBe(6);
    expect(after.state).toBe('ASKING');
    expect(after.question?.text).toBe('Second?');
    expect(after.questionSeq).toBe(6);
    expect(after.questionsAsked).toBe(2);
    expect(after.currentTurnId).toBeNull();
    expect(acceptsAnswer(after)).toBe(true);
    // The grade never enters the view.
    expect(JSON.stringify(after)).not.toContain('0.9');
  });

  it('ignores events already applied (a replay after reconnecting)', () => {
    const start = fromSnapshot(detail({ seq: 5 }));
    const { view, gap } = applyEvent(
      start,
      ev({ type: 'grading_started', seq: 5, state: 'GRADING', turn_id: 't' }),
    );
    expect(view).toBe(start);
    expect(gap).toBe(false);
  });

  it('reports a gap instead of guessing', () => {
    const start = fromSnapshot(detail());
    const { view, gap } = applyEvent(
      start,
      ev({ type: 'grade_ready', seq: 9, state: 'GRADING', turn_id: 't' }),
    );
    expect(gap).toBe(true);
    expect(view).toBe(start);
  });

  it('counts follow-ups separately and labels them', () => {
    const start = fromSnapshot(detail());
    const after = run(start, [
      ev({ type: 'grading_started', seq: 3, state: 'GRADING', turn_id: 't1' }),
      ev({ type: 'follow_up', seq: 4, state: 'PROBING' }),
      ev({
        type: 'next_question',
        seq: 5,
        state: 'ASKING',
        question_id: null,
        is_follow_up: true,
        text: 'Why?',
      }),
    ]);
    expect(after.questionsAsked).toBe(1);
    expect(after.followUpsAsked).toBe(1);
    expect(questionLabel(after)).toBe('Follow-up on question 1');
  });

  it('never lets a late snapshot roll back events the stream already applied', () => {
    // A re-read races the restarted stream: the snapshot is read at seq 3
    // (GRADING), the stream then applies seq 4-5 (graded, next question).
    const stale = fromSnapshot(detail({ state: 'GRADING', seq: 3, current_turn_id: 't1' }));
    const live = run(stale, [
      ev({ type: 'grade_ready', seq: 4, state: 'SELECTING', turn_id: 't1', score: 0.5 }),
      ev({
        type: 'next_question',
        seq: 5,
        state: 'ASKING',
        question_id: 'q2',
        is_follow_up: false,
        text: 'Next?',
      }),
    ]);
    expect(live.state).toBe('ASKING');
    expect(snapshotIsCurrent(live, stale)).toBe(false);
    // Same or newer snapshots, a first load, or another session: accepted.
    expect(snapshotIsCurrent(live, { ...stale, seq: 5 })).toBe(true);
    expect(snapshotIsCurrent(live, { ...stale, seq: 9 })).toBe(true);
    expect(snapshotIsCurrent(null, stale)).toBe(true);
    expect(snapshotIsCurrent(live, { ...stale, sessionId: 'other' })).toBe(true);
  });

  it('treats every grading_failed as final, whatever retries_exhausted says', () => {
    // backend/app/interview/grading.py: GRADING_FAILED is appended only by
    // `_fail`; `retries_exhausted: false` means a permanent error, not "retrying".
    const grading = run(fromSnapshot(detail()), [
      ev({ type: 'grading_started', seq: 3, state: 'GRADING', turn_id: 't1' }),
    ]);
    for (const retries_exhausted of [false, true]) {
      const after = applyEvent(
        grading,
        ev({
          type: 'grading_failed',
          seq: 4,
          state: 'GRADING',
          turn_id: 't1',
          error_code: 'x',
          retries_exhausted,
        }),
      ).view;
      expect(after.review).toBe('failed');
    }
  });

  it('clears the question when the interview ends', () => {
    const done = run(fromSnapshot(detail({ state: 'GRADING', seq: 3, current_turn_id: 't' })), [
      ev({ type: 'state_changed', seq: 4, state: 'REPORTING', event_type: 'REPORTING_STARTED' }),
      ev({ type: 'session_complete', seq: 5, state: 'COMPLETED', report_id: 'r' }),
    ]);
    expect(done.state).toBe('COMPLETED');
    expect(done.question).toBeNull();
  });
});

describe('snapshot', () => {
  it('only an ASKING snapshot knows the question seq', () => {
    expect(fromSnapshot(detail()).questionSeq).toBe(2);
    expect(fromSnapshot(detail({ state: 'GRADING', seq: 3 })).questionSeq).toBe(-1);
  });

  it('a failed turn marks the review as given up', () => {
    const view = fromSnapshot(detail({ state: 'GRADING', seq: 3, current_turn_id: 't1' }));
    const turn = {
      session_id: 's1',
      turn_id: 't1',
      question_id: 'q1',
      status: 'failed',
      attempts: 3,
      score: null,
      error_code: 'provider_down',
      submitted_at: '',
      graded_at: null,
      status_url: '',
      stream_url: '',
    };
    expect(withTurnStatus(view, turn).review).toBe('failed');
    expect(withTurnStatus(view, { ...turn, status: 'pending' }).review).toBe('retrying');
    expect(withTurnStatus(view, { ...turn, turn_id: 'other' }).review).toBe('pending');
  });
});

describe('answer helpers', () => {
  it('counts code points like the backend', () => {
    expect(answerLength('😀😀')).toBe(2);
    expect(validateAnswer('😀'.repeat(8000))).toBeNull();
    expect(validateAnswer('a'.repeat(8001))).toMatch(/under 8,000/);
    expect(validateAnswer('   \n ')).toMatch(/Write an answer/);
  });

  it('splits fenced code from prose and keeps indentation', () => {
    const parts = splitFences('Read this:\n```python\n  x = 1\n    y = 2\n```\nThen explain.');
    expect(parts).toEqual([
      { code: false, text: 'Read this:\n' },
      { code: true, text: '  x = 1\n    y = 2' },
      { code: false, text: '\nThen explain.' },
    ]);
    expect(splitFences('no code')).toEqual([{ code: false, text: 'no code' }]);
  });
});
