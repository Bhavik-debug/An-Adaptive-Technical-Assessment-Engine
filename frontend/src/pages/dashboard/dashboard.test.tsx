import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import type { Report } from '@/api/types';
import { mockApi, renderApp, session, signedInRoutes } from '@/test/utils';

const REPORT: Report = {
  report_id: 'r1',
  session_id: session().session_id,
  status: 'ready',
  stats: {
    topics: [
      {
        topic_key: 'dsa',
        questions_asked: 4,
        observed: {
          display: 62,
          interval: [48, 75],
          insufficient_evidence: false,
          n_observations: 4,
        },
      },
      {
        topic_key: 'databases',
        questions_asked: 1,
        observed: {
          display: 50,
          interval: [20, 80],
          insufficient_evidence: true,
          n_observations: 1,
        },
      },
    ],
  },
  narrative: null,
  narrative_source: 'template',
  narrative_error: null,
  error: null,
  generated_at: '2026-10-02T09:41:00Z',
};

describe('dashboard', () => {
  it('explains an empty history and the absence of skill data', async () => {
    mockApi({ ...signedInRoutes(), 'GET /api/sessions': { body: { sessions: [] } } });
    renderApp('/dashboard');
    expect(await screen.findByText('No interviews yet')).toBeInTheDocument();
    expect(screen.getByText('No skill data yet')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /start an interview/i })).toHaveAttribute(
      'href',
      '/interviews/new',
    );
  });

  it('shows a retryable error when sessions cannot load', async () => {
    const api = mockApi({
      ...signedInRoutes(),
      'GET /api/sessions': [
        { status: 500, body: { detail: 'Internal Server Error' } },
        { body: { sessions: [] } },
      ],
    });
    renderApp('/dashboard');
    const alert = await screen.findByRole('alert');
    expect(within(alert).getByText('We could not load your interviews')).toBeInTheDocument();
    await userEvent.setup().click(within(alert).getByRole('button', { name: /try again/i }));
    expect(await screen.findByText('No interviews yet')).toBeInTheDocument();
    expect(api.callsTo('GET', '/api/sessions')).toHaveLength(2);
  });

  it('lists real sessions with candidate-facing statuses and dates', async () => {
    mockApi({
      ...signedInRoutes(),
      'GET /api/sessions': {
        body: {
          sessions: [
            session({
              session_id: 'a',
              state: 'GRADING',
              role: 'Data engineer',
              report_status: null,
              ended_at: null,
            }),
            session({ session_id: 'b', state: 'CREATED', report_status: null, ended_at: null }),
            session({ session_id: 'c', state: 'COMPLETED', report_status: 'pending' }),
          ],
        },
      },
    });
    renderApp('/dashboard');
    expect(await screen.findByText('Data engineer')).toBeInTheDocument();
    expect(screen.getByText('In progress')).toBeInTheDocument();
    expect(screen.queryByText(/grading/i)).not.toBeInTheDocument();
    expect(screen.getByText('Not started')).toBeInTheDocument();
    expect(screen.getByText('Completed')).toBeInTheDocument();
    expect(screen.getAllByText(/Data structures & algorithms/).length).toBeGreaterThan(0);
    // A completed session whose report is not ready yet is not used for skills.
    expect(screen.getByText('No skill data yet')).toBeInTheDocument();
  });

  it('shows skills from the latest ready report, flagging thin evidence', async () => {
    const api = mockApi({
      ...signedInRoutes(),
      'GET /api/sessions': { body: { sessions: [session()] } },
      [`GET /api/sessions/${session().session_id}/report`]: { body: REPORT },
    });
    renderApp('/dashboard');
    expect(
      await screen.findByRole('img', { name: /Data structures & algorithms: 62 out of 100/ }),
    ).toBeInTheDocument();
    expect(screen.getByText('Not enough evidence yet')).toBeInTheDocument();
    expect(screen.getByText(/still being validated/)).toBeInTheDocument();
    expect(api.callsTo('GET', `/api/sessions/${session().session_id}/report`)).toHaveLength(1);
  });

  it('starts a not-yet-started session from the list', async () => {
    const id = 'b';
    const api = mockApi({
      ...signedInRoutes(),
      'GET /api/sessions': {
        body: { sessions: [session({ session_id: id, state: 'CREATED', report_status: null })] },
      },
      [`POST /api/sessions/${id}/start`]: {
        body: {
          session_id: id,
          state: 'ASKING',
          seq: 3,
          question: { question_id: 'q', is_follow_up: false, text: 'Q?' },
          turns_url: '',
          stream_url: '',
        },
      },
    });
    renderApp('/dashboard');
    await userEvent.setup().click(await screen.findByRole('button', { name: 'Start' }));
    expect(await screen.findByText('Your interview has started')).toBeInTheDocument();
    expect(api.callsTo('POST', `/api/sessions/${id}/start`)).toHaveLength(1);
  });
});
