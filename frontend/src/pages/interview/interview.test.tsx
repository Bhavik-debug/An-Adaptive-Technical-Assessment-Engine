import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import type { SessionDetail } from '@/api/types';
import { mockApi, renderApp, signedInRoutes } from '@/test/utils';

const SID = '33333333-3333-3333-3333-333333333333';
const BASE = `/api/sessions/${SID}`;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

function detail(overrides: Partial<SessionDetail> = {}): SessionDetail {
  return {
    session_id: SID,
    state: 'ASKING',
    seq: 2,
    role: 'Backend engineer',
    level: 'mid',
    mode: 'standard',
    topic_keys: ['dsa', 'databases'],
    started_at: '2026-10-14T09:00:00Z',
    ended_at: null,
    report_status: null,
    question: {
      question_id: 'q1',
      is_follow_up: false,
      text: 'Explain this query:\n```sql\nSELECT *\n  FROM orders;\n```\nWhat would you index?',
    },
    questions_asked: 1,
    follow_ups_asked: 0,
    current_turn_id: null,
    stream_url: `${BASE}/stream`,
    turns_url: `${BASE}/turns`,
    ...overrides,
  };
}

/** A controllable SSE response per stream request, aborted like real fetch. */
function streams() {
  const opened: {
    lastEventId: string | undefined;
    send: (event: string, data: Record<string, unknown>) => void;
    close: () => void;
    aborted: () => boolean;
  }[] = [];
  const handler = ({ init }: { init: RequestInit }) => {
    let controller!: ReadableStreamDefaultController<Uint8Array>;
    let aborted = false;
    const body = new ReadableStream<Uint8Array>({
      start(c) {
        controller = c;
      },
    });
    init.signal?.addEventListener('abort', () => {
      aborted = true;
      try {
        controller.error(new DOMException('Aborted', 'AbortError'));
      } catch {
        // already closed
      }
    });
    const encoder = new TextEncoder();
    const headers = (init.headers ?? {}) as Record<string, string>;
    opened.push({
      lastEventId: headers['Last-Event-ID'],
      send: (event, data) =>
        controller.enqueue(
          encoder.encode(
            `id: ${String(data.seq)}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`,
          ),
        ),
      close: () => controller.close(),
      aborted: () => aborted,
    });
    return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
  };
  return { opened, handler };
}

function setup(extra: Record<string, unknown> = {}, snapshot = detail()) {
  const s = streams();
  const api = mockApi({
    ...signedInRoutes(),
    [`GET ${BASE}`]: { body: snapshot },
    [`GET ${BASE}/stream`]: s.handler,
    ...(extra as Record<string, never>),
  });
  return { api, streams: s.opened };
}

const answerBox = () => screen.getByLabelText('Your answer');

describe('interview screen: loading', () => {
  it('shows the server question verbatim, with code kept as code', async () => {
    setup();
    renderApp(`/interviews/${SID}`);
    expect(await screen.findByRole('heading', { name: 'Question 1' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { level: 1, name: 'Backend engineer' })).toBeInTheDocument();
    const code = document.querySelector('pre code');
    expect(code?.textContent).toBe('SELECT *\n  FROM orders;');
    expect(screen.getByText('What would you index?')).toBeInTheDocument();
    // Honest progress: a count, never a total or a percentage.
    expect(screen.getByText(/1 question asked so far/)).toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    const topics = screen.getByRole('list', { name: 'Topics in this interview' });
    expect(within(topics).getByText('Data structures & algorithms')).toBeInTheDocument();
  });

  it('follows the stream from the snapshot seq', async () => {
    const { streams } = setup();
    renderApp(`/interviews/${SID}`);
    await screen.findByRole('heading', { name: 'Question 1' });
    await waitFor(() => expect(streams).toHaveLength(1));
    expect(streams[0]?.lastEventId).toBe('2');
  });

  it('says so when the session does not exist or is not yours', async () => {
    setup({ [`GET ${BASE}`]: { status: 404, body: { detail: 'Session not found' } } });
    renderApp(`/interviews/${SID}`);
    expect(await screen.findByText('Interview not found')).toBeInTheDocument();
    expect(screen.queryByLabelText('Your answer')).not.toBeInTheDocument();
  });

  it('treats a malformed id like a missing session', async () => {
    mockApi({
      ...signedInRoutes(),
      'GET /api/sessions/not-a-uuid': { status: 422, body: { detail: [] } },
    });
    renderApp('/interviews/not-a-uuid');
    expect(await screen.findByText('Interview not found')).toBeInTheDocument();
  });

  it('offers a retry when the snapshot cannot be loaded', async () => {
    setup({
      [`GET ${BASE}`]: [{ status: 503, body: { detail: 'Down' } }, { body: detail() }],
    });
    renderApp(`/interviews/${SID}`);
    await userEvent.setup().click(await screen.findByRole('button', { name: 'Try again' }));
    expect(await screen.findByRole('heading', { name: 'Question 1' })).toBeInTheDocument();
  });

  it('a session that has not started can be started (idempotently)', async () => {
    const { api } = setup(
      {
        [`GET ${BASE}`]: [
          { body: detail({ state: 'CREATED', seq: 0, question: null, questions_asked: 0 }) },
          { body: detail() },
        ],
        [`POST ${BASE}/start`]: { body: {} },
      },
      undefined,
    );
    renderApp(`/interviews/${SID}`);
    await userEvent.setup().click(await screen.findByRole('button', { name: 'Start interview' }));
    expect(await screen.findByRole('heading', { name: 'Question 1' })).toBeInTheDocument();
    expect(api.callsTo('POST', `${BASE}/start`)).toHaveLength(1);
  });

  it('a finished interview links to the (Day 28) report', async () => {
    setup({}, detail({ state: 'COMPLETED', question: null, report_status: 'ready', seq: 40 }));
    renderApp(`/interviews/${SID}`);
    expect(await screen.findByText('Interview complete')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'View report' })).toHaveAttribute(
      'href',
      `/interviews/${SID}/report`,
    );
    expect(screen.queryByLabelText('Your answer')).not.toBeInTheDocument();
  });

  it('after a refresh mid-review it shows the review, not the editor', async () => {
    setup(
      {
        [`GET ${BASE}/turns/t-1`]: {
          body: { turn_id: 't-1', status: 'failed', error_code: 'x', score: null },
        },
      },
      detail({ state: 'GRADING', seq: 3, current_turn_id: 't-1' }),
    );
    renderApp(`/interviews/${SID}`);
    expect(await screen.findByText('We could not review this answer')).toBeInTheDocument();
    expect(screen.queryByLabelText('Your answer')).not.toBeInTheDocument();
  });
});

describe('interview screen: answering', () => {
  it('validates before sending anything', async () => {
    const { api } = setup({ [`POST ${BASE}/turns`]: { status: 202, body: {} } });
    renderApp(`/interviews/${SID}`);
    const user = userEvent.setup();
    await screen.findByRole('heading', { name: 'Question 1' });
    await user.type(answerBox(), '   ');
    await user.click(screen.getByRole('button', { name: 'Submit answer' }));
    expect(await screen.findByText('Write an answer before submitting.')).toBeInTheDocument();
    expect(answerBox()).toHaveFocus();
    expect(answerBox()).toHaveAttribute('aria-invalid', 'true');
    expect(api.callsTo('POST', `${BASE}/turns`)).toHaveLength(0);
  });

  it('submits once, waits for the server, then shows the next question', async () => {
    const { api, streams } = setup({
      [`POST ${BASE}/turns`]: async () => {
        await new Promise((r) => setTimeout(r, 30));
        return { status: 202, body: { status: 'pending' } };
      },
    });
    renderApp(`/interviews/${SID}`);
    const user = userEvent.setup();
    await screen.findByRole('heading', { name: 'Question 1' });
    await user.type(answerBox(), 'Index customer_id.');
    await user.dblClick(screen.getByRole('button', { name: 'Submit answer' }));

    expect(await screen.findByText(/Waiting for the next step/)).toBeInTheDocument();
    const posts = api.callsTo('POST', `${BASE}/turns`);
    expect(posts).toHaveLength(1);
    const sent = posts[0]?.body as { turn_id: string; answer: string };
    expect(sent.answer).toBe('Index customer_id.');
    expect(sent.turn_id).toMatch(UUID);
    // Not advanced locally: still question 1 until the server says otherwise.
    expect(screen.getByRole('heading', { name: 'Question 1' })).toBeInTheDocument();

    await waitFor(() => expect(streams).toHaveLength(1));
    const stream = streams[0]!;
    stream.send('grading_started', { seq: 3, state: 'GRADING', turn_id: sent.turn_id });
    expect(await screen.findByText('Answer received')).toBeInTheDocument();
    stream.send('grade_ready', { seq: 4, state: 'GRADING', turn_id: sent.turn_id, score: 0.83 });
    stream.send('state_changed', {
      seq: 5,
      state: 'SELECTING',
      event_type: 'SELECTION_STARTED',
    });
    stream.send('next_question', {
      seq: 6,
      state: 'ASKING',
      question_id: 'q2',
      is_follow_up: false,
      text: 'How does a B-tree index work?',
    });

    expect(await screen.findByRole('heading', { name: 'Question 2' })).toBeInTheDocument();
    expect(screen.getByText('How does a B-tree index work?')).toBeInTheDocument();
    expect(answerBox()).toHaveValue('');
    // The grade is never shown.
    expect(document.body.textContent).not.toMatch(/0\.83|83/);

    await user.type(answerBox(), 'Balanced tree.');
    await user.click(screen.getByRole('button', { name: 'Submit answer' }));
    await waitFor(() => expect(api.callsTo('POST', `${BASE}/turns`)).toHaveLength(2));
    const second = api.callsTo('POST', `${BASE}/turns`)[1]?.body as { turn_id: string };
    expect(second.turn_id).not.toBe(sent.turn_id);
  });

  it('an unknown outcome freezes the answer and resends it identically', async () => {
    let calls = 0;
    const { api } = setup({
      [`POST ${BASE}/turns`]: () => {
        calls += 1;
        if (calls === 1) throw new TypeError('Failed to fetch');
        return { status: 200, body: { status: 'pending' } };
      },
    });
    renderApp(`/interviews/${SID}`);
    const user = userEvent.setup();
    await screen.findByRole('heading', { name: 'Question 1' });
    await user.type(answerBox(), 'My answer');
    await user.click(screen.getByRole('button', { name: 'Submit answer' }));

    expect(await screen.findByText('Answer status unknown')).toBeInTheDocument();
    expect(answerBox()).toHaveAttribute('readonly');
    expect(answerBox()).toHaveValue('My answer');
    expect(screen.getByRole('button', { name: 'Submit answer' })).toBeDisabled();

    await user.click(screen.getByRole('button', { name: 'Resend the same answer' }));
    expect(await screen.findByText(/Waiting for the next step/)).toBeInTheDocument();
    const [first, retry] = api.callsTo('POST', `${BASE}/turns`);
    expect(retry?.body).toEqual(first?.body);
  });

  it('a server-side rejection keeps the draft editable', async () => {
    setup({
      [`POST ${BASE}/turns`]: {
        status: 422,
        body: {
          detail: [{ loc: ['body', 'answer'], msg: 'Value error, answer must not be blank' }],
        },
      },
    });
    renderApp(`/interviews/${SID}`);
    const user = userEvent.setup();
    await screen.findByRole('heading', { name: 'Question 1' });
    await user.type(answerBox(), 'x');
    await user.click(screen.getByRole('button', { name: 'Submit answer' }));
    expect(await screen.findByText('answer must not be blank')).toBeInTheDocument();
    expect(answerBox()).not.toHaveAttribute('readonly');
    expect(answerBox()).toHaveValue('x');
  });

  it('a state conflict re-reads the server state instead of guessing', async () => {
    const { api } = setup({
      [`GET ${BASE}`]: [
        { body: detail() },
        { body: detail({ state: 'GRADING', seq: 3, current_turn_id: 'other-tab' }) },
      ],
      [`GET ${BASE}/turns/other-tab`]: { body: { turn_id: 'other-tab', status: 'pending' } },
      [`POST ${BASE}/turns`]: { status: 409, body: { detail: 'Not allowed' } },
    });
    renderApp(`/interviews/${SID}`);
    const user = userEvent.setup();
    await screen.findByRole('heading', { name: 'Question 1' });
    await user.type(answerBox(), 'late');
    await user.click(screen.getByRole('button', { name: 'Submit answer' }));
    expect(await screen.findByText('Answer received')).toBeInTheDocument();
    expect(api.callsTo('GET', BASE)).toHaveLength(2);
  });
});

describe('interview screen: the stream', () => {
  it('ignores replayed events and re-reads the snapshot on a gap', async () => {
    const { api, streams } = setup({
      [`GET ${BASE}`]: [
        { body: detail() },
        { body: detail({ state: 'COMPLETED', seq: 30, question: null }) },
      ],
    });
    renderApp(`/interviews/${SID}`);
    await screen.findByRole('heading', { name: 'Question 1' });
    await waitFor(() => expect(streams).toHaveLength(1));
    streams[0]!.send('next_question', {
      seq: 2,
      state: 'ASKING',
      is_follow_up: false,
      text: 'dup',
    });
    streams[0]!.send('grade_ready', { seq: 9, state: 'GRADING', turn_id: 'x' });
    expect(await screen.findByText('Interview complete')).toBeInTheDocument();
    expect(screen.queryByText('dup')).not.toBeInTheDocument();
    expect(api.callsTo('GET', BASE)).toHaveLength(2);
  });

  it('reconnects from the last applied seq after a drop', async () => {
    const { streams } = setup();
    renderApp(`/interviews/${SID}`);
    await screen.findByRole('heading', { name: 'Question 1' });
    await waitFor(() => expect(streams).toHaveLength(1));
    streams[0]!.send('grading_started', { seq: 3, state: 'GRADING', turn_id: 't' });
    await screen.findByText('Answer received');
    streams[0]!.close();
    expect(await screen.findByText(/Reconnecting/)).toBeInTheDocument();
    await waitFor(() => expect(streams).toHaveLength(2), { timeout: 3000 });
    expect(streams[1]?.lastEventId).toBe('3');
  });

  it('closes the stream when leaving the screen', async () => {
    const { streams } = setup({ 'GET /api/sessions': { body: { sessions: [] } } });
    renderApp(`/interviews/${SID}`);
    await screen.findByRole('heading', { name: 'Question 1' });
    await waitFor(() => expect(streams).toHaveLength(1));
    await userEvent.setup().click(screen.getByRole('link', { name: 'return from your dashboard' }));
    await screen.findByRole('heading', { name: 'Welcome' });
    expect(streams[0]?.aborted()).toBe(true);
  });

  it('sends the bearer token on the stream request', async () => {
    const { api, streams } = setup();
    renderApp(`/interviews/${SID}`);
    await waitFor(() => expect(streams).toHaveLength(1));
    const call = api.callsTo('GET', `${BASE}/stream`)[0];
    expect(call?.headers.Authorization).toBe('Bearer access-1');
    expect(call?.headers.Accept).toBe('text/event-stream');
  });
});
