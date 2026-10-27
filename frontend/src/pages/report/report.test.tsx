import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import type { Report, SessionDetail } from '@/api/types';
import { parseReportStats } from '@/lib/report';
import { mockApi, renderApp, session, signedInRoutes } from '@/test/utils';

const SID = session().session_id;
const REPORT_PATH = `GET /api/sessions/${SID}/report`;
const SESSION_PATH = `GET /api/sessions/${SID}`;

function detail(overrides: Partial<SessionDetail> = {}): SessionDetail {
  return {
    session_id: SID,
    state: 'COMPLETED',
    seq: 40,
    role: 'Backend engineer',
    level: 'mid',
    mode: 'standard',
    topic_keys: ['dsa', 'databases'],
    started_at: '2026-10-02T09:00:00Z',
    ended_at: '2026-10-02T09:40:00Z',
    report_status: 'ready',
    question: null,
    questions_asked: 3,
    follow_ups_asked: 2,
    current_turn_id: null,
    stream_url: `/api/sessions/${SID}/stream`,
    turns_url: `/api/sessions/${SID}/turns`,
    ...overrides,
  };
}

const ability = (display: number, low: number, high: number, insufficient = false, n = 2) => ({
  // theta, rd are in the real payload; the screen must never show them.
  theta: 0.1234,
  rd: 0.4321,
  n_observations: n,
  display,
  interval: [low, high],
  insufficient_evidence: insufficient,
});

/** Shaped like backend/app/reporting/stats.py output. */
const STATS = {
  version: 1,
  role: 'Backend engineer',
  level: 'mid',
  target_theta: 0.0,
  stop_reasons: ['item_budget_reached'],
  counts: {
    primary_questions: 3,
    follow_ups_asked: 2,
    follow_ups_decided: 2,
    follow_ups_abandoned: 0,
    answers_graded: 5,
    grading_failures: 1,
    item_budget: 3,
  },
  questions: [
    {
      order: 1,
      question_id: 'q-internal-1',
      topic_key: 'dsa',
      subtopic_key: 'arrays',
      score: 0.875,
      follow_ups: [{ score: 0.4 }, { score: null }],
    },
    {
      order: 2,
      question_id: 'q-internal-2',
      topic_key: 'dsa',
      subtopic_key: 'graphs',
      score: 0.2,
      follow_ups: [],
    },
    {
      order: 3,
      question_id: 'q-internal-3',
      topic_key: 'databases',
      subtopic_key: 'indexing',
      score: null,
      follow_ups: [],
    },
  ],
  topics: [
    { topic_key: 'dsa', questions_asked: 2, observed: ability(62, 48, 75) },
    { topic_key: 'databases', questions_asked: 1, observed: null },
  ],
  subtopics: [
    { subtopic_key: 'arrays', topic_key: 'dsa', observed: ability(71, 55, 84) },
    { subtopic_key: 'graphs', topic_key: 'dsa', observed: ability(30, 18, 45, false, 1) },
    { subtopic_key: 'indexing', topic_key: 'databases', observed: null },
  ],
  gaps: ['graphs'],
  strengths: ['arrays'],
  needs_more_assessment: ['indexing'],
  priors: { source: 'default', seeded_subtopics: [], note: 'Starting points only.' },
  methodology_notes: ['Scores and ability estimates are computed in code.'],
  log_seq: 40,
};

function report(overrides: Partial<Report> = {}): Report {
  return {
    report_id: 'r1',
    session_id: SID,
    status: 'ready',
    stats: STATS as unknown as Report['stats'],
    narrative: {
      summary: 'You answered questions across data structures and databases with mixed results.',
      strengths: ['Clear explanation of two-pointer techniques'],
      growth_areas: ['Graph traversal trade-offs'],
      next_steps: ['Practise breadth-first search problems'],
    },
    narrative_source: 'model',
    narrative_error: null,
    error: null,
    generated_at: '2026-10-02T09:41:00Z',
    ...overrides,
  };
}

function routes(extra: Record<string, unknown> = {}) {
  return {
    ...signedInRoutes(),
    [SESSION_PATH]: { body: detail() },
    [REPORT_PATH]: { body: report() },
    ...extra,
  } as Parameters<typeof mockApi>[0];
}

describe('report screen', () => {
  it('renders the real report: summary, topic bars with ranges, lists, questions', async () => {
    mockApi(routes());
    renderApp(`/interviews/${SID}/report`);
    expect(
      await screen.findByRole('heading', { name: 'Backend engineer', level: 1 }),
    ).toBeVisible();
    expect(screen.getByText('Completed')).toBeInTheDocument();
    const topics = screen.getByRole('list', { name: 'Topics in this interview' });
    expect(within(topics).getByText('Data structures & algorithms')).toBeInTheDocument();

    expect(
      await screen.findByText(/You answered questions across data structures/),
    ).toBeInTheDocument();
    expect(screen.getByText(/3 questions and 2 follow-ups/)).toBeInTheDocument();
    expect(screen.getByText(/1 answer could not be graded/)).toBeInTheDocument();

    // Bars always carry their interval (plan §9.3), with setup-options names.
    expect(
      screen.getByRole('img', {
        name: 'Data structures & algorithms: 62 out of 100, likely range 48 to 75',
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('img', {
        name: 'Arrays & two pointers: 71 out of 100, likely range 55 to 84',
      }),
    ).toBeInTheDocument();
    expect(screen.getByText('Not assessed')).toBeInTheDocument(); // databases: observed null

    const improve = screen.getByRole('heading', { name: 'Areas to improve' }).parentElement!;
    expect(within(improve).getByText('Graphs')).toBeInTheDocument();
    const strengths = screen.getByRole('heading', { name: 'Strengths' }).parentElement!;
    expect(within(strengths).getByText('Arrays & two pointers')).toBeInTheDocument();
    const more = screen.getByRole('heading', { name: 'Needs more assessment' }).parentElement!;
    expect(within(more).getByText('Indexing')).toBeInTheDocument();

    // Model-written feedback is labelled as such.
    expect(screen.getByRole('heading', { name: 'Feedback' })).toBeInTheDocument();
    expect(screen.getByText('Practise breadth-first search problems')).toBeInTheDocument();
    expect(screen.getAllByText(/Written by an AI model/).length).toBeGreaterThan(0);

    // Per-question grades: 0.875 -> 88/100; ungraded shown as such.
    const questions = screen.getByRole('list', { name: 'Answer grades' });
    expect(within(questions).getByText('88')).toBeInTheDocument();
    // The column chart names each column with the same real grades.
    const chart = screen.getByRole('list', { name: 'Answer grades chart' });
    expect(
      within(chart).getByLabelText('Question 1: 88 out of 100; follow-ups 40, not graded'),
    ).toBeInTheDocument();
    expect(within(chart).getByLabelText('Question 3: not graded')).toBeInTheDocument();
    expect(within(questions).getByText('Follow-ups: 40/100, not graded')).toBeInTheDocument();
    expect(within(questions).getByText('Not graded')).toBeInTheDocument();
    expect(within(questions).getByText('Databases › Indexing')).toBeInTheDocument();

    expect(screen.getByText(/still being validated against human reviewers/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Back to dashboard' })).toHaveAttribute(
      'href',
      '/dashboard',
    );
  });

  it('charts only measured skills, classified by the backend, with a matching table', async () => {
    mockApi(routes());
    renderApp(`/interviews/${SID}/report`);
    await screen.findByRole('heading', { name: 'Skill estimates' });

    // Stat tiles: the backend's counts, plus the number of skills it rated.
    const tile = (label: string) => screen.getByText(label, { selector: 'dt' }).nextSibling;
    expect(tile('Questions')).toHaveTextContent('3');
    expect(tile('Follow-ups')).toHaveTextContent('2');
    expect(tile('Answers graded')).toHaveTextContent('5');
    expect(tile('Skills measured')).toHaveTextContent('2');

    const chart = screen.getByRole('list', { name: 'Skill estimates' });
    const rows = within(chart).getAllByRole('listitem');
    // Sorted by estimate; Indexing (never observed) is not plotted.
    expect(rows.map((r) => r.querySelector('p')?.textContent)).toEqual([
      'Arrays & two pointers',
      'Graphs',
    ]);
    expect(within(rows[0]!).getByText('Strength')).toBeInTheDocument();
    expect(within(rows[1]!).getByText('Area to improve')).toBeInTheDocument();
    expect(
      within(rows[1]!).getByRole('img', { name: 'Graphs: 30 out of 100, likely range 18 to 45' }),
    ).toBeInTheDocument();

    // The table view carries the same values.
    const user = userEvent.setup();
    await user.click(screen.getByText('Show as a table'));
    const table = screen.getByRole('table', { name: 'Skill estimates' });
    const graphs = within(table).getByRole('row', { name: /Graphs/ });
    expect(
      within(graphs)
        .getAllByRole('cell')
        .map((c) => c.textContent),
    ).toEqual(['Data structures & algorithms', '30', '18–45', '1', 'Area to improve']);
  });

  it('never shows internal parameters (theta, RD, target, question ids, stop reasons)', async () => {
    mockApi(routes());
    renderApp(`/interviews/${SID}/report`);
    await screen.findByRole('heading', { name: 'Question by question' });
    const text = document.body.textContent ?? '';
    for (const hidden of ['0.1234', '0.4321', 'q-internal-1', 'item_budget_reached', 'theta']) {
      expect(text).not.toContain(hidden);
    }
    expect(text).not.toMatch(/\bRD\b/);
  });

  it('shows only the feedback sections the model actually wrote', async () => {
    mockApi(
      routes({
        [REPORT_PATH]: {
          body: report({
            narrative: {
              summary: 'You answered questions across data structures with mixed results.',
              strengths: [],
              growth_areas: [],
              next_steps: ['Practise breadth-first search problems'],
            },
          }),
        },
      }),
    );
    renderApp(`/interviews/${SID}/report`);
    expect(await screen.findByRole('heading', { name: 'Next steps' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'What went well' })).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'To work on' })).not.toBeInTheDocument();
  });

  it('does not repeat template lists as "feedback"; labels the standard summary', async () => {
    mockApi(
      routes({
        [REPORT_PATH]: {
          body: report({
            narrative_source: 'template',
            narrative: {
              summary: 'This assessment asked 3 questions and 2 follow-ups across dsa, databases.',
              strengths: ['Comparatively strong: arrays'],
              growth_areas: [],
              next_steps: [],
            },
          }),
        },
      }),
    );
    renderApp(`/interviews/${SID}/report`);
    expect(await screen.findByText(/A standard summary generated/)).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Feedback' })).not.toBeInTheDocument();
    expect(screen.queryByText('Comparatively strong: arrays')).not.toBeInTheDocument();
  });

  it('shows an honest pending state, then the report once it is ready', async () => {
    const api = mockApi(
      routes({
        [SESSION_PATH]: [
          { body: detail({ state: 'REPORTING', report_status: 'processing', ended_at: null }) },
          { body: detail() },
        ],
        [REPORT_PATH]: [
          { status: 202, body: report({ status: 'processing', stats: null, narrative: null }) },
          { body: report() },
        ],
      }),
    );
    renderApp(`/interviews/${SID}/report`);
    expect(await screen.findByText('Preparing your report…')).toBeInTheDocument();
    expect(screen.getByText('Finishing')).toBeInTheDocument();
    expect(
      await screen.findByRole('heading', { name: 'Results by topic' }, { timeout: 5000 }),
    ).toBeInTheDocument();
    expect(api.callsTo('GET', `/api/sessions/${SID}/report`)).toHaveLength(2);
    // Ready implies completed: the header is re-read, not left at "Finishing".
    expect(await screen.findByText('Completed')).toBeInTheDocument();
    expect(screen.queryByText('Finishing')).not.toBeInTheDocument();
    expect(api.callsTo('GET', `/api/sessions/${SID}`)).toHaveLength(2);
  });

  it('explains a failed report without inventing results', async () => {
    mockApi(
      routes({
        [REPORT_PATH]: {
          status: 500,
          body: report({ status: 'failed', stats: null, narrative: null, error: 'stats_failed' }),
        },
      }),
    );
    renderApp(`/interviews/${SID}/report`);
    expect(await screen.findByText('This report could not be prepared')).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'Results by topic' })).not.toBeInTheDocument();
    expect(document.body.textContent).not.toContain('stats_failed');
  });

  it('sends an unfinished interview back to the interview, without asking for a report', async () => {
    const api = mockApi(
      routes({ [SESSION_PATH]: { body: detail({ state: 'ASKING', ended_at: null }) } }),
    );
    renderApp(`/interviews/${SID}/report`);
    expect(await screen.findByText('This interview is not finished yet')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Continue the interview' })).toHaveAttribute(
      'href',
      `/interviews/${SID}`,
    );
    expect(api.callsTo('GET', `/api/sessions/${SID}/report`)).toHaveLength(0);
  });

  it('treats another account’s (or a missing) session as not found', async () => {
    mockApi(routes({ [SESSION_PATH]: { status: 404, body: { detail: 'Session not found' } } }));
    renderApp(`/interviews/${SID}/report`);
    expect(await screen.findByText('Report not found')).toBeInTheDocument();
    expect(screen.getByText(/belongs to another account/)).toBeInTheDocument();
  });

  it('shows a retryable error when the report cannot be fetched', async () => {
    const api = mockApi(
      routes({
        [REPORT_PATH]: [
          { status: 503, body: { detail: 'Service Unavailable' } },
          { body: report() },
        ],
      }),
    );
    renderApp(`/interviews/${SID}/report`);
    expect(await screen.findByText('We could not load this report')).toBeInTheDocument();
    await userEvent.setup().click(screen.getByRole('button', { name: 'Try again' }));
    expect(await screen.findByRole('heading', { name: 'Results by topic' })).toBeInTheDocument();
    expect(api.callsTo('GET', `/api/sessions/${SID}/report`)).toHaveLength(2);
  });

  it('refuses to display unusable stats rather than guessing', async () => {
    mockApi(
      routes({
        [REPORT_PATH]: { body: report({ stats: 'garbage' as unknown as Report['stats'] }) },
      }),
    );
    renderApp(`/interviews/${SID}/report`);
    expect(await screen.findByText('This report could not be displayed')).toBeInTheDocument();
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
  });

  it('is reachable from the dashboard', async () => {
    mockApi(
      routes({
        'GET /api/sessions': { body: { sessions: [session()] } },
      }),
    );
    renderApp('/dashboard');
    const user = userEvent.setup();
    await user.click(await screen.findByRole('link', { name: /view report/i }));
    expect(screen.getByTestId('location').textContent).toBe(`/interviews/${SID}/report`);
    expect(await screen.findByRole('heading', { name: 'Results by topic' })).toBeInTheDocument();
  });

  it('keeps keyboard access to the way back', async () => {
    mockApi(routes());
    renderApp(`/interviews/${SID}/report`);
    await screen.findByRole('heading', { name: 'Question by question' });
    const user = userEvent.setup();
    const back = screen.getByRole('link', { name: 'Back to dashboard' });
    for (let i = 0; i < 30 && document.activeElement !== back; i++) await user.tab();
    expect(back).toHaveFocus();
    await user.keyboard('{Enter}');
    expect(screen.getByTestId('location').textContent).toBe('/dashboard');
  });
});

describe('parseReportStats', () => {
  it('drops malformed entries instead of guessing values', () => {
    const view = parseReportStats({
      topics: [
        { topic_key: 'dsa', questions_asked: 2, observed: { display: 'x', interval: [1, 2] } },
        { questions_asked: 1 },
        'nonsense',
        { topic_key: 'db', observed: { display: 40, interval: [10, 70] } },
      ],
      questions: [{ order: 1, score: 'high' }, { score: 0.5 }],
      gaps: ['graphs', 3, null],
      counts: 'none',
    });
    expect(view?.topics).toEqual([
      { topicKey: 'dsa', questionsAsked: 2, observed: null },
      {
        topicKey: 'db',
        questionsAsked: 0,
        // No evidence flag from the server: do not claim precision.
        observed: { display: 40, interval: [10, 70], insufficientEvidence: true, observations: 0 },
      },
    ]);
    expect(view?.questions).toEqual([
      { order: 1, topicKey: null, subtopicKey: null, score: null, followUpScores: [] },
    ]);
    expect(view?.gaps).toEqual(['graphs']);
    expect(view?.counts.primaryQuestions).toBeNull();
    expect(parseReportStats(null)).toBeNull();
    expect(parseReportStats([1, 2])).toBeNull();
  });
});
