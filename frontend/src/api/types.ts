/**
 * Response and request shapes, transcribed from the backend's Pydantic models.
 * Each type names its source so drift can be checked against one file.
 * Fields the frontend does not use are still listed when they are part of the
 * contract, but nothing here is invented.
 */

// backend/app/api/auth.py ---------------------------------------------------

/** TokenResponse */
export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

/** AuthProviders (GET /api/auth/providers). `google_client_id` is public; null = off. */
export interface AuthProviders {
  password: boolean;
  google_client_id: string | null;
}

/** UserResponse */
export interface User {
  id: string;
  email: string;
  created_at: string;
}

// backend/app/api/ingestion.py + backend/app/models/ingestion.py ------------

export type IngestStatus = 'pending' | 'processing' | 'ready' | 'failed';

/** backend/app/ingestion/profile.py: Evidence */
export type Evidence = 'listed' | 'project' | 'professional';

/** backend/app/ingestion/profile.py: ProfileSkill */
export interface ProfileSkill {
  name: string;
  evidence: Evidence;
  subtopic_keys: string[];
}

/** backend/app/ingestion/profile.py: CandidateProfile (PATCH body; extra fields forbidden) */
export interface CandidateProfile {
  headline: string | null;
  skills: ProfileSkill[];
  years_experience: number | null;
}

/** ResumeOut */
export interface Resume {
  resume_id: string;
  status: IngestStatus;
  error: string | null;
  document_kind: string;
  page_count: number | null;
  truncated: boolean;
  redactions: Record<string, number>;
  /** The parser's suggestion; seeds nothing until confirmed. Shape follows CandidateProfile. */
  suggestion: Partial<CandidateProfile> | null;
  profile: CandidateProfile | null;
  profile_confirmed: boolean;
  created_at: string;
  parsed_at: string | null;
  profile_updated_at: string | null;
}

/** backend/app/ingestion/extraction.py: verify_jd's requirement rows */
export interface JdRequirement {
  quote: string;
  importance: string;
  topic_key: string | null;
}

/** JobDescriptionOut */
export interface JobDescription {
  jd_id: string;
  status: IngestStatus;
  error: string | null;
  redactions: Record<string, number>;
  title: string | null;
  requirements: JdRequirement[];
  topic_weights: Record<string, number>;
  min_years_experience: number | null;
  created_at: string;
  parsed_at: string | null;
}

// backend/app/api/setup.py --------------------------------------------------

export interface SubtopicOption {
  key: string;
  display_name: string;
}

export interface TopicOption {
  key: string;
  display_name: string;
  domain: string;
  subtopics: SubtopicOption[];
}

/** SetupOptions */
export interface SetupOptions {
  topics: TopicOption[];
  levels: string[];
  modes: string[];
  default_level: string;
  default_mode: string;
  default_duration_minutes: number;
  min_duration_minutes: number;
  max_duration_minutes: number;
}

// backend/app/api/sessions.py -----------------------------------------------

/** backend/app/interview/fsm.py: InterviewState */
export type SessionState =
  | 'CREATED'
  | 'PREPARING'
  | 'ASKING'
  | 'GRADING'
  | 'PROBING'
  | 'SELECTING'
  | 'REPORTING'
  | 'COMPLETED';

export type ReportStatus = 'pending' | 'processing' | 'ready' | 'failed';

/** SessionSummary */
export interface SessionSummary {
  session_id: string;
  state: SessionState | string;
  role: string | null;
  level: string | null;
  mode: string | null;
  topic_keys: string[];
  resume_id: string | null;
  jd_id: string | null;
  started_at: string;
  ended_at: string | null;
  report_status: ReportStatus | string | null;
}

/** TopicIn */
export interface TopicIn {
  topic_key: string;
  jd_weight?: number;
  subtopic_keys?: string[];
  required?: boolean;
  min_items?: number;
  max_items?: number | null;
}

/** SessionCreate (extra fields forbidden) */
export interface SessionCreate {
  role: string;
  topics?: TopicIn[];
  resume_id?: string | null;
  jd_id?: string | null;
  level?: string | null;
  mode?: string | null;
  duration_minutes?: number | null;
  item_budget?: number | null;
}

/** SessionCreated. `blueprint` is part of the contract but the UI never reads it. */
export interface SessionCreated {
  session_id: string;
  state: string;
  seq: number;
  blueprint: Record<string, unknown>;
  resume_id: string | null;
  jd_id: string | null;
  start_url: string;
  stream_url: string;
}

/** QuestionOut */
export interface Question {
  question_id: string | null;
  is_follow_up: boolean;
  text: string | null;
}

/** SessionStarted */
export interface SessionStarted {
  session_id: string;
  state: string;
  seq: number;
  question: Question;
  turns_url: string;
  stream_url: string;
}

/** SessionDetail (GET /api/sessions/{id}, Day 27) */
export interface SessionDetail {
  session_id: string;
  state: SessionState | string;
  /** The last event's seq: the stream's Last-Event-ID. */
  seq: number;
  role: string | null;
  level: string | null;
  mode: string | null;
  topic_keys: string[];
  started_at: string;
  ended_at: string | null;
  report_status: ReportStatus | string | null;
  /** Being answered (ASKING) or just answered (GRADING/PROBING/SELECTING); else null. */
  question: Question | null;
  /** Bank questions asked so far, including the current one. There is no total. */
  questions_asked: number;
  follow_ups_asked: number;
  /** The answer being graded, while one is. */
  current_turn_id: string | null;
  stream_url: string;
  turns_url: string;
}

/** TurnSubmission (extra fields forbidden). `turn_id` is the idempotency key. */
export interface TurnSubmission {
  turn_id: string;
  answer: string;
}

export type GradingStatus = 'pending' | 'processing' | 'graded' | 'failed';

/** TurnStatus. `score` exists in the contract; the interview screen never shows it. */
export interface TurnStatus {
  session_id: string;
  turn_id: string;
  question_id: string | null;
  status: GradingStatus | string;
  attempts: number;
  score: number | null;
  error_code: string | null;
  submitted_at: string;
  graded_at: string | null;
  status_url: string;
  stream_url: string;
}

// backend/app/interview/stream.py: SSE event names and their whitelisted data --

/** Every SSE message carries the event's seq and the state after it. */
export interface StreamEventBase {
  seq: number;
  state: SessionState | string;
}

export type StreamEvent =
  | ({ type: 'grading_started'; turn_id: string } & StreamEventBase)
  | ({ type: 'grade_ready'; turn_id: string } & StreamEventBase)
  | ({
      type: 'grading_failed';
      turn_id: string;
      error_code: string | null;
      retries_exhausted: boolean | null;
    } & StreamEventBase)
  | ({ type: 'follow_up' } & StreamEventBase)
  | ({
      type: 'follow_up_abandoned';
      error_code: string | null;
      retries_exhausted: boolean | null;
    } & StreamEventBase)
  | ({
      type: 'next_question';
      question_id: string | null;
      is_follow_up: boolean;
      text: string | null;
    } & StreamEventBase)
  | ({ type: 'session_complete'; report_id: string | null } & StreamEventBase)
  | ({ type: 'state_changed'; event_type: string } & StreamEventBase);

// backend/app/reporting/stats.py (the subset the dashboard reads) -------------

export interface AbilityView {
  display: number;
  interval: [number, number];
  insufficient_evidence: boolean;
  n_observations: number;
}

export interface ReportTopic {
  topic_key: string;
  questions_asked: number;
  observed: AbilityView | null;
}

export interface ReportStats {
  role?: string;
  topics?: ReportTopic[];
}

/** ReportOut */
export interface Report {
  report_id: string;
  session_id: string;
  status: ReportStatus | string;
  stats: ReportStats | null;
  narrative: Record<string, unknown> | null;
  narrative_source: string | null;
  narrative_error: string | null;
  error: string | null;
  generated_at: string | null;
}
