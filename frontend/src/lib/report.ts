/**
 * The candidate-facing view of a report (backend/app/reporting/stats.py and
 * narrative.py; docs/reporting.md).
 *
 * `stats` arrives as an untyped JSON object, so it is read defensively here:
 * a missing or malformed field is dropped (shown as absent), never guessed.
 * Only what the candidate is meant to see is kept. The display score, its
 * interval and the evidence flag are the product's presentation of ability
 * (plan §9.3). The raw theta, RD, level target, question ids, stop reasons
 * and log position are internal, so they are not copied into this view at
 * all. Nothing is computed here: every number is the backend's.
 */

export interface AbilityDisplay {
  /** 0-100, `round(100 / (1 + exp(-1.1·θ)))`, computed by the backend. */
  display: number;
  /** The 95% interval on the same scale. */
  interval: [number, number];
  /** RD > 0.8: too uncertain to rate (plan §9.3). */
  insufficientEvidence: boolean;
  observations: number;
}

export interface TopicResult {
  topicKey: string;
  questionsAsked: number;
  observed: AbilityDisplay | null;
}

export interface SubtopicResult {
  subtopicKey: string;
  topicKey: string | null;
  observed: AbilityDisplay | null;
}

export interface QuestionResult {
  order: number;
  topicKey: string | null;
  subtopicKey: string | null;
  /** The grade, 0-1; null when the answer was never graded. */
  score: number | null;
  followUpScores: (number | null)[];
}

export interface ReportView {
  role: string | null;
  level: string | null;
  counts: {
    primaryQuestions: number | null;
    followUpsAsked: number | null;
    answersGraded: number | null;
    gradingFailures: number | null;
  };
  topics: TopicResult[];
  subtopics: SubtopicResult[];
  questions: QuestionResult[];
  /** Subtopic keys, ranked by the backend (weakest first). */
  gaps: string[];
  strengths: string[];
  needsMoreAssessment: string[];
  /** True when the starting points came from the candidate's resume. */
  resumePriors: boolean;
  priorsNote: string | null;
  methodologyNotes: string[];
}

export interface NarrativeView {
  summary: string | null;
  strengths: string[];
  growthAreas: string[];
  nextSteps: string[];
}

type Json = Record<string, unknown>;

function obj(value: unknown): Json | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Json)
    : null;
}

function str(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value : null;
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((v): v is string => !!str(v)) : [];
}

function ability(value: unknown): AbilityDisplay | null {
  const o = obj(value);
  if (!o) return null;
  const display = num(o.display);
  const interval = Array.isArray(o.interval) ? o.interval.map(num) : [];
  const [low, high] = interval;
  if (display === null || low == null || high == null || interval.length !== 2) return null;
  return {
    display,
    interval: [low, high],
    // Absent flag = do not claim precision the backend did not vouch for.
    insufficientEvidence: o.insufficient_evidence !== false,
    observations: num(o.n_observations) ?? 0,
  };
}

function list<T>(value: unknown, read: (item: Json) => T | null): T[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const o = obj(item);
    const parsed = o ? read(o) : null;
    return parsed === null ? [] : [parsed];
  });
}

/** Read `stats`; null when it is not an object at all. */
export function parseReportStats(raw: unknown): ReportView | null {
  const s = obj(raw);
  if (!s) return null;
  const counts = obj(s.counts) ?? {};
  const priors = obj(s.priors) ?? {};
  return {
    role: str(s.role),
    level: str(s.level),
    counts: {
      primaryQuestions: num(counts.primary_questions),
      followUpsAsked: num(counts.follow_ups_asked),
      answersGraded: num(counts.answers_graded),
      gradingFailures: num(counts.grading_failures),
    },
    topics: list(s.topics, (t) => {
      const topicKey = str(t.topic_key);
      if (!topicKey) return null;
      return {
        topicKey,
        questionsAsked: num(t.questions_asked) ?? 0,
        observed: ability(t.observed),
      };
    }),
    subtopics: list(s.subtopics, (t) => {
      const subtopicKey = str(t.subtopic_key);
      if (!subtopicKey) return null;
      return { subtopicKey, topicKey: str(t.topic_key), observed: ability(t.observed) };
    }),
    questions: list(s.questions, (q) => {
      const order = num(q.order);
      if (order === null) return null;
      return {
        order,
        topicKey: str(q.topic_key),
        subtopicKey: str(q.subtopic_key),
        score: num(q.score),
        followUpScores: list(q.follow_ups, (f) => ({ score: num(f.score) })).map((f) => f.score),
      };
    }),
    gaps: strings(s.gaps),
    strengths: strings(s.strengths),
    needsMoreAssessment: strings(s.needs_more_assessment),
    resumePriors: priors.source === 'resume_profile',
    priorsNote: str(priors.note),
    methodologyNotes: strings(s.methodology_notes),
  };
}

/** Read `narrative` (ReportNarrative); null when absent or unusable. */
export function parseNarrative(raw: unknown): NarrativeView | null {
  const n = obj(raw);
  if (!n) return null;
  const view = {
    summary: str(n.summary),
    strengths: strings(n.strengths),
    growthAreas: strings(n.growth_areas),
    nextSteps: strings(n.next_steps),
  };
  const empty =
    !view.summary && !view.strengths.length && !view.growthAreas.length && !view.nextSteps.length;
  return empty ? null : view;
}

/** A 0-1 grade as a whole number out of 100, for display only. */
export function scoreOutOf100(score: number): number {
  return Math.round(Math.min(1, Math.max(0, score)) * 100);
}
