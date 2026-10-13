import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import type { JobDescription, Resume } from '@/api/types';
import { mockApi, renderApp, signedInRoutes } from '@/test/utils';

const SID = '33333333-3333-3333-3333-333333333333';
const CREATED = {
  status: 201,
  body: {
    session_id: SID,
    state: 'CREATED',
    seq: 0,
    blueprint: {},
    resume_id: null,
    jd_id: null,
    start_url: '',
    stream_url: '',
  },
};
const STARTED = {
  body: {
    session_id: SID,
    state: 'ASKING',
    seq: 3,
    question: { question_id: 'q1', is_follow_up: false, text: 'First question' },
    turns_url: '',
    stream_url: '',
  },
};

function resume(overrides: Partial<Resume> = {}): Resume {
  return {
    resume_id: 'res-1',
    status: 'pending',
    error: null,
    document_kind: 'text',
    page_count: null,
    truncated: false,
    redactions: {},
    suggestion: null,
    profile: null,
    profile_confirmed: false,
    created_at: '',
    parsed_at: null,
    profile_updated_at: null,
    ...overrides,
  };
}

function jd(overrides: Partial<JobDescription> = {}): JobDescription {
  return {
    jd_id: 'jd-1',
    status: 'pending',
    error: null,
    redactions: {},
    title: null,
    requirements: [],
    topic_weights: {},
    min_years_experience: null,
    created_at: '',
    parsed_at: null,
    ...overrides,
  };
}

async function skipToConfig(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole('button', { name: /continue without a resume/i }));
  await user.click(
    await screen.findByRole('button', { name: /continue without a job description/i }),
  );
  await screen.findByLabelText('Role');
}

describe('setup wizard — interview settings', () => {
  it('validates role and topics before creating anything', async () => {
    const api = mockApi(signedInRoutes());
    renderApp('/interviews/new');
    const user = userEvent.setup();
    await skipToConfig(user);
    await user.click(screen.getByRole('button', { name: 'Start interview' }));
    expect(screen.getByText('Enter the role you are preparing for.')).toBeInTheDocument();
    expect(screen.getByText('Choose at least one topic.')).toBeInTheDocument();
    expect(api.callsTo('POST', '/api/sessions')).toHaveLength(0);
  });

  it('offers only backend-supported options', async () => {
    mockApi(signedInRoutes());
    renderApp('/interviews/new');
    await skipToConfig(userEvent.setup());
    const level = screen.getByLabelText('Seniority');
    expect([...level.querySelectorAll('option')].map((o) => o.value)).toEqual([
      'junior',
      'mid',
      'senior',
      'staff',
    ]);
    expect(level).toHaveValue('mid');
    expect(screen.getAllByRole('radio').map((r) => (r as HTMLInputElement).value)).toEqual([
      'screening',
      'standard',
      'deep_dive',
    ]);
    expect(screen.getAllByRole('checkbox')).toHaveLength(2);
  });

  it('creates then starts, sends the exact spec, and hands off only after confirmation', async () => {
    const api = mockApi({
      ...signedInRoutes(),
      'POST /api/sessions': CREATED,
      [`POST /api/sessions/${SID}/start`]: STARTED,
    });
    renderApp('/interviews/new');
    const user = userEvent.setup();
    await skipToConfig(user);
    await user.type(screen.getByLabelText('Role'), '  Backend engineer ');
    await user.selectOptions(screen.getByLabelText('Seniority'), 'senior');
    await user.click(screen.getByRole('radio', { name: /screening/i }));
    await user.selectOptions(screen.getByLabelText('Length'), '30');
    await user.click(screen.getByRole('checkbox', { name: 'Databases' }));
    await user.click(screen.getByRole('button', { name: 'Start interview' }));

    expect(await screen.findByText('Your interview has started')).toBeInTheDocument();
    expect(screen.getByTestId('location').textContent).toBe(`/interviews/${SID}`);
    expect(api.callsTo('POST', '/api/sessions')[0]?.body).toEqual({
      role: 'Backend engineer',
      level: 'senior',
      mode: 'screening',
      duration_minutes: 30,
      resume_id: null,
      jd_id: null,
      topics: [{ topic_key: 'databases' }],
    });
    expect(api.callsTo('POST', `/api/sessions/${SID}/start`)).toHaveLength(1);
  });

  it('shows the backend reason when creation is rejected', async () => {
    mockApi({
      ...signedInRoutes(),
      'POST /api/sessions': {
        status: 422,
        body: { detail: 'duration_minutes must be between 5 and 360' },
      },
    });
    renderApp('/interviews/new');
    const user = userEvent.setup();
    await skipToConfig(user);
    await user.type(screen.getByLabelText('Role'), 'Engineer');
    await user.click(screen.getByRole('checkbox', { name: 'Databases' }));
    await user.click(screen.getByRole('button', { name: 'Start interview' }));
    expect(await screen.findByText('The interview was not created')).toBeInTheDocument();
    expect(screen.getByText(/duration_minutes must be between/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start interview' })).toBeEnabled();
  });

  it('never re-creates a session: a failed start is retried with /start only', async () => {
    const api = mockApi({
      ...signedInRoutes(),
      'POST /api/sessions': CREATED,
      [`POST /api/sessions/${SID}/start`]: [
        { status: 409, body: { detail: "No question in the bank fits this session's blueprint" } },
        STARTED,
      ],
    });
    renderApp('/interviews/new');
    const user = userEvent.setup();
    await skipToConfig(user);
    await user.type(screen.getByLabelText('Role'), 'Engineer');
    await user.click(screen.getByRole('checkbox', { name: 'Databases' }));
    await user.click(screen.getByRole('button', { name: 'Start interview' }));

    expect(
      await screen.findByText('The interview was created but could not be started'),
    ).toBeInTheDocument();
    expect(screen.queryByText('Your interview has started')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Role')).toBeDisabled();

    await user.click(screen.getByRole('button', { name: 'Try starting again' }));
    expect(await screen.findByText('Your interview has started')).toBeInTheDocument();
    expect(api.callsTo('POST', '/api/sessions')).toHaveLength(1);
    expect(api.callsTo('POST', `/api/sessions/${SID}/start`)).toHaveLength(2);
  });

  it('ignores a double click on start', async () => {
    const api = mockApi({
      ...signedInRoutes(),
      'POST /api/sessions': async () => {
        await new Promise((r) => setTimeout(r, 30));
        return CREATED;
      },
      [`POST /api/sessions/${SID}/start`]: STARTED,
    });
    renderApp('/interviews/new');
    const user = userEvent.setup();
    await skipToConfig(user);
    await user.type(screen.getByLabelText('Role'), 'Engineer');
    await user.click(screen.getByRole('checkbox', { name: 'Databases' }));
    const button = screen.getByRole('button', { name: 'Start interview' });
    await user.dblClick(button);
    await screen.findByText('Your interview has started');
    expect(api.callsTo('POST', '/api/sessions')).toHaveLength(1);
  });
});

describe('setup wizard — resume', () => {
  it('rejects unsupported files without uploading', async () => {
    const api = mockApi(signedInRoutes());
    renderApp('/interviews/new');
    const user = userEvent.setup({ applyAccept: false });
    const input = (await screen.findByRole('button', { name: 'Choose file' }))
      .previousElementSibling as HTMLInputElement;
    await user.upload(input, new File(['x'], 'photo.png', { type: 'image/png' }));
    expect(screen.getByText('Upload a PDF or a plain-text (.txt) file.')).toBeInTheDocument();
    expect(api.callsTo('POST', '/api/resumes')).toHaveLength(0);
  });

  it('uploads, waits for extraction, and confirms the reviewed profile', async () => {
    const suggestion = {
      headline: 'Backend engineer',
      skills: [
        { name: 'PostgreSQL', evidence: 'professional' as const, subtopic_keys: ['indexing'] },
      ],
      years_experience: 3,
    };
    const api = mockApi({
      ...signedInRoutes(),
      'POST /api/resumes': { status: 202, body: resume() },
      'GET /api/resumes/res-1': { body: resume({ status: 'ready', suggestion }) },
      'PATCH /api/resumes/res-1/profile': ({ body }) => ({
        body: resume({
          status: 'ready',
          suggestion,
          profile: body as Resume['profile'],
          profile_confirmed: true,
        }),
      }),
      'POST /api/sessions': CREATED,
      [`POST /api/sessions/${SID}/start`]: STARTED,
    });
    renderApp('/interviews/new');
    const user = userEvent.setup();
    const input = (await screen.findByRole('button', { name: 'Choose file' }))
      .previousElementSibling as HTMLInputElement;
    await user.upload(input, new File(['My resume text'], 'cv.txt', { type: 'text/plain' }));

    const skill = await screen.findByDisplayValue('PostgreSQL');
    expect(screen.getByDisplayValue('Backend engineer')).toBeInTheDocument();
    expect(screen.getByText('Indexing')).toBeInTheDocument();
    const upload = api.callsTo('POST', '/api/resumes')[0];
    expect(upload?.headers['Content-Type']).toBe('text/plain');

    await user.clear(skill);
    await user.type(skill, 'Postgres');
    await user.click(screen.getByRole('button', { name: 'Confirm profile' }));
    expect(await screen.findByText('Profile confirmed')).toBeInTheDocument();
    expect(api.callsTo('PATCH', '/api/resumes/res-1/profile')[0]?.body).toEqual({
      headline: 'Backend engineer',
      years_experience: 3,
      skills: [{ name: 'Postgres', evidence: 'professional', subtopic_keys: ['indexing'] }],
    });

    // The resume id is carried through to session creation.
    await user.click(screen.getByRole('button', { name: 'Continue' }));
    await user.click(
      await screen.findByRole('button', { name: /continue without a job description/i }),
    );
    expect(screen.getByText('Resume: attached, profile confirmed')).toBeInTheDocument();
    await user.type(screen.getByLabelText('Role'), 'Engineer');
    await user.click(screen.getByRole('checkbox', { name: 'Databases' }));
    await user.click(screen.getByRole('button', { name: 'Start interview' }));
    await screen.findByText('Your interview has started');
    expect(api.callsTo('POST', '/api/sessions')[0]?.body).toMatchObject({ resume_id: 'res-1' });
  });

  it('lets the candidate enter skills manually when extraction fails', async () => {
    mockApi({
      ...signedInRoutes(),
      'POST /api/resumes': { status: 202, body: resume() },
      'GET /api/resumes/res-1': { body: resume({ status: 'failed', error: 'extraction_failed' }) },
    });
    renderApp('/interviews/new');
    const user = userEvent.setup();
    const input = (await screen.findByRole('button', { name: 'Choose file' }))
      .previousElementSibling as HTMLInputElement;
    await user.upload(input, new File(['%PDF-1.7'], 'cv.pdf', { type: 'application/pdf' }));
    expect(
      await screen.findByText('We could not extract a profile automatically'),
    ).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Add skill' }));
    await user.click(screen.getByRole('button', { name: 'Confirm profile' }));
    expect(screen.getByText('Enter a skill name or remove this row.')).toBeInTheDocument();
  });

  it('explains server-side upload rejections', async () => {
    mockApi({
      ...signedInRoutes(),
      'POST /api/resumes': {
        status: 415,
        body: { detail: { code: 'unsupported_type', message: 'Unsupported.' } },
      },
    });
    renderApp('/interviews/new');
    const user = userEvent.setup();
    const input = (await screen.findByRole('button', { name: 'Choose file' }))
      .previousElementSibling as HTMLInputElement;
    await user.upload(input, new File(['%PDF'], 'cv.pdf', { type: 'application/pdf' }));
    expect(await screen.findByText(/That file type is not supported/)).toBeInTheDocument();
  });
});

describe('setup wizard — job description', () => {
  it('requires enough text before submitting', async () => {
    const api = mockApi(signedInRoutes());
    renderApp('/interviews/new');
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: /continue without a resume/i }));
    await user.type(await screen.findByLabelText('Job description'), 'Too short');
    await user.click(screen.getByRole('button', { name: 'Analyse job description' }));
    expect(screen.getByText(/too short to extract requirements/)).toBeInTheDocument();
    expect(api.callsTo('POST', '/api/job-descriptions')).toHaveLength(0);
  });

  it('uses JD topics by default, sending no explicit topics', async () => {
    const text = 'We need a backend engineer with strong SQL indexing and algorithms skills.';
    const ready = jd({
      status: 'ready',
      title: 'Backend Engineer',
      topic_weights: { databases: 0.7, dsa: 0.3 },
      requirements: [
        { quote: 'strong SQL indexing', importance: 'must_have', topic_key: 'databases' },
      ],
    });
    const api = mockApi({
      ...signedInRoutes(),
      'POST /api/job-descriptions': { status: 202, body: jd() },
      'GET /api/job-descriptions/jd-1': { body: ready },
      'POST /api/sessions': CREATED,
      [`POST /api/sessions/${SID}/start`]: STARTED,
    });
    renderApp('/interviews/new');
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: /continue without a resume/i }));
    await user.type(await screen.findByLabelText('Job description'), text);
    await user.click(screen.getByRole('button', { name: 'Analyse job description' }));
    expect(await screen.findByText('What we found for “Backend Engineer”')).toBeInTheDocument();
    expect(api.callsTo('POST', '/api/job-descriptions')[0]?.body).toEqual({ text });

    await user.click(screen.getByRole('button', { name: 'Continue' }));
    expect(await screen.findByLabelText('Role')).toHaveValue('Backend Engineer');
    expect(
      screen.getByRole('radio', { name: /use topics from the job description/i }),
    ).toBeChecked();
    await user.click(screen.getByRole('button', { name: 'Start interview' }));
    await screen.findByText('Your interview has started');
    const spec = api.callsTo('POST', '/api/sessions')[0]?.body as Record<string, unknown>;
    expect(spec.jd_id).toBe('jd-1');
    expect(spec).not.toHaveProperty('topics');
  });

  it('falls back to manual topics when the JD is still processing', async () => {
    const api = mockApi({
      ...signedInRoutes(),
      'POST /api/job-descriptions': { status: 202, body: jd() },
      'GET /api/job-descriptions/jd-1': { body: jd({ status: 'processing' }) },
      'POST /api/sessions': CREATED,
      [`POST /api/sessions/${SID}/start`]: STARTED,
    });
    renderApp('/interviews/new');
    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: /continue without a resume/i }));
    await user.type(
      await screen.findByLabelText('Job description'),
      'A long enough job description for a backend role with SQL.',
    );
    await user.click(screen.getByRole('button', { name: 'Analyse job description' }));
    await screen.findByText('Analysing the job description…');
    await user.click(screen.getByRole('button', { name: 'Continue' }));
    expect(
      await screen.findByText(/has no usable topics yet, so choose the topics/),
    ).toBeInTheDocument();
    await user.type(screen.getByLabelText('Role'), 'Engineer');
    await user.click(screen.getByRole('checkbox', { name: 'Databases' }));
    await user.click(screen.getByRole('button', { name: 'Start interview' }));
    await waitFor(() => expect(api.callsTo('POST', '/api/sessions')).toHaveLength(1));
    expect(api.callsTo('POST', '/api/sessions')[0]?.body).toMatchObject({
      jd_id: 'jd-1',
      topics: [{ topic_key: 'databases' }],
    });
  });
});
