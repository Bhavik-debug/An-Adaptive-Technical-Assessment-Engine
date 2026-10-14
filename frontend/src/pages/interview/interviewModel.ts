/**
 * The interview screen's model: the server's snapshot (GET /api/sessions/{id})
 * advanced by stream events, and nothing else.
 *
 * - Nothing here decides what happens next in the interview - the backend's
 *   engine does. This only mirrors what the server has recorded.
 * - Events are applied strictly in `seq` order. One already applied is ignored
 *   (a replay after reconnecting); one that skips a seq means something was
 *   missed, and the caller re-reads the snapshot instead of guessing.
 * - Grades arrive on the stream (`grade_ready.score`) and are deliberately
 *   dropped: the candidate sees no per-answer scoring during the interview.
 */

import type { Question, SessionDetail, StreamEvent, TurnStatus } from '@/api/types';

/** How the current answer's review stands, from the candidate's point of view. */
export type ReviewStatus = 'none' | 'pending' | 'retrying' | 'failed';

export interface InterviewView {
  sessionId: string;
  state: string;
  seq: number;
  role: string | null;
  level: string | null;
  mode: string | null;
  topicKeys: string[];
  question: Question | null;
  /** The seq of the event that asked the current question (-1 if not known). */
  questionSeq: number;
  questionsAsked: number;
  followUpsAsked: number;
  currentTurnId: string | null;
  review: ReviewStatus;
}

export function fromSnapshot(detail: SessionDetail): InterviewView {
  return {
    sessionId: detail.session_id,
    state: detail.state,
    seq: detail.seq,
    role: detail.role,
    level: detail.level,
    mode: detail.mode,
    topicKeys: detail.topic_keys,
    question: detail.question,
    // While ASKING, the last event is the one that asked the question.
    questionSeq: detail.state === 'ASKING' ? detail.seq : -1,
    questionsAsked: detail.questions_asked,
    followUpsAsked: detail.follow_ups_asked,
    currentTurnId: detail.current_turn_id,
    review: detail.state === 'GRADING' ? 'pending' : 'none',
  };
}

/** Refine `review` from a turn's status (used once, after loading a snapshot). */
export function withTurnStatus(view: InterviewView, turn: TurnStatus): InterviewView {
  if (view.currentTurnId !== turn.turn_id || view.state !== 'GRADING') return view;
  if (turn.status === 'failed') return { ...view, review: 'failed' };
  if (turn.status === 'graded') return { ...view, review: 'pending' };
  return { ...view, review: turn.error_code ? 'retrying' : 'pending' };
}

export interface Applied {
  view: InterviewView;
  /** True when the event skipped a seq: re-read the snapshot. */
  gap: boolean;
}

export function applyEvent(view: InterviewView, event: StreamEvent): Applied {
  if (event.seq <= view.seq) return { view, gap: false }; // already applied
  if (event.seq !== view.seq + 1) return { view, gap: true };

  const next: InterviewView = { ...view, seq: event.seq, state: event.state };
  switch (event.type) {
    case 'grading_started':
      return { view: { ...next, currentTurnId: event.turn_id, review: 'pending' }, gap: false };
    case 'grade_ready':
      // The score is not shown. The engine now decides what comes next.
      return { view: { ...next, review: 'pending' }, gap: false };
    case 'grading_failed':
      return {
        view: { ...next, review: event.retries_exhausted ? 'failed' : 'retrying' },
        gap: false,
      };
    case 'next_question':
      return {
        view: {
          ...next,
          question: {
            question_id: event.question_id,
            is_follow_up: event.is_follow_up,
            text: event.text,
          },
          questionSeq: event.seq,
          questionsAsked: next.questionsAsked + (event.is_follow_up ? 0 : 1),
          followUpsAsked: next.followUpsAsked + (event.is_follow_up ? 1 : 0),
          currentTurnId: null,
          review: 'none',
        },
        gap: false,
      };
    case 'session_complete':
      return { view: { ...next, question: null, currentTurnId: null, review: 'none' }, gap: false };
    default:
      // follow_up, follow_up_abandoned, state_changed: only the state moves.
      if (next.state === 'REPORTING' || next.state === 'COMPLETED') {
        return {
          view: { ...next, question: null, currentTurnId: null, review: 'none' },
          gap: false,
        };
      }
      return { view: next, gap: false };
  }
}

/** True when the candidate can type and submit an answer right now. */
export function acceptsAnswer(view: InterviewView): boolean {
  return view.state === 'ASKING' && view.question !== null;
}

export function isFinished(state: string): boolean {
  return state === 'REPORTING' || state === 'COMPLETED';
}

/** "Question 3", or "Follow-up on question 3" - counted, never a fabricated total. */
export function questionLabel(view: InterviewView): string {
  const n = Math.max(view.questionsAsked, 1);
  return view.question?.is_follow_up ? `Follow-up on question ${n}` : `Question ${n}`;
}
