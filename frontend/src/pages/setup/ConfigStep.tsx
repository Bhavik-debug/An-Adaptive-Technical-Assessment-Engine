import * as React from 'react';
import { useNavigate } from 'react-router';

import { ApiError } from '@/api/client';
import { createSession, startSession } from '@/api/sessions';
import type { JobDescription, Resume, SessionCreate, SetupOptions } from '@/api/types';
import { FormField } from '@/components/FormField';
import { Alert } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { keyLabel, levelLabel, modeLabel } from '@/lib/format';
import { durationChoices, jdTopicKeys } from '@/lib/setup';
import { cn } from '@/lib/utils';

const MAX_ROLE = 120;

const MODE_HINTS: Record<string, string> = {
  screening: 'Broad coverage with quicker questions.',
  standard: 'A balanced mix of breadth and depth.',
  deep_dive: 'Fewer, more involved questions.',
};

type TopicSource = 'jd' | 'manual';

interface Errors {
  role?: string;
  topics?: string;
}

/** Where the create/start sequence stands. A session id, once created, is never discarded. */
type Phase =
  | { kind: 'idle' }
  | { kind: 'creating' }
  | { kind: 'starting'; sessionId: string }
  | { kind: 'start-failed'; sessionId: string; message: string };

export function ConfigStep({
  options,
  resume,
  jd,
  topicNames,
  onBack,
}: {
  options: SetupOptions;
  resume: Resume | null;
  jd: JobDescription | null;
  topicNames: Map<string, string>;
  onBack: () => void;
}) {
  const navigate = useNavigate();
  const jdTopics = jdTopicKeys(jd);
  const jdUsable = jdTopics.length > 0;

  const [role, setRole] = React.useState(jd?.title ?? '');
  const [level, setLevel] = React.useState(options.default_level);
  const [mode, setMode] = React.useState(options.default_mode);
  const [duration, setDuration] = React.useState(options.default_duration_minutes);
  const [topicSource, setTopicSource] = React.useState<TopicSource>(jdUsable ? 'jd' : 'manual');
  const [selected, setSelected] = React.useState<Set<string>>(() => new Set(jdTopics));
  const [errors, setErrors] = React.useState<Errors>({});
  const [formError, setFormError] = React.useState<string | null>(null);
  const [phase, setPhase] = React.useState<Phase>({ kind: 'idle' });
  // A ref, not state: it blocks a second click before React re-renders.
  const inFlight = React.useRef(false);

  const locked = phase.kind !== 'idle';
  const busy = phase.kind === 'creating' || phase.kind === 'starting';

  function toggleTopic(key: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function validate(): Errors {
    const next: Errors = {};
    const trimmed = role.trim();
    if (!trimmed) next.role = 'Enter the role you are preparing for.';
    else if (trimmed.length > MAX_ROLE) next.role = `Keep the role under ${MAX_ROLE} characters.`;
    if (topicSource === 'manual' && selected.size === 0) next.topics = 'Choose at least one topic.';
    return next;
  }

  async function start(sessionId: string) {
    setPhase({ kind: 'starting', sessionId });
    try {
      await startSession(sessionId);
      navigate(`/interviews/${sessionId}`, { replace: true, state: { started: true } });
    } catch (error) {
      setPhase({
        kind: 'start-failed',
        sessionId,
        message: error instanceof ApiError ? error.message : 'The interview could not be started.',
      });
    } finally {
      inFlight.current = false;
    }
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (inFlight.current || locked) return;
    const next = validate();
    setErrors(next);
    setFormError(null);
    if (next.role || next.topics) {
      document.getElementById(next.role ? 'config-role' : 'config-topics')?.focus();
      return;
    }

    const spec: SessionCreate = {
      role: role.trim(),
      level,
      mode,
      duration_minutes: duration,
      resume_id: resume?.resume_id ?? null,
      jd_id: jd?.jd_id ?? null,
      // Omitted for 'jd': the backend derives topics from the JD's weights.
      topics:
        topicSource === 'manual'
          ? options.topics.filter((t) => selected.has(t.key)).map((t) => ({ topic_key: t.key }))
          : undefined,
    };

    inFlight.current = true;
    setPhase({ kind: 'creating' });
    let sessionId: string;
    try {
      sessionId = (await createSession(spec)).session_id;
    } catch (error) {
      inFlight.current = false;
      setPhase({ kind: 'idle' });
      if (error instanceof ApiError && error.status === 409 && topicSource === 'jd') {
        setTopicSource('manual');
        setFormError(
          'The job description is not ready yet, so its topics cannot be used. Choose topics below and try again.',
        );
      } else {
        setFormError(
          error instanceof ApiError ? error.message : 'The interview could not be created.',
        );
      }
      return;
    }
    await start(sessionId);
  }

  function retryStart() {
    if (phase.kind !== 'start-failed' || inFlight.current) return;
    inFlight.current = true;
    void start(phase.sessionId);
  }

  return (
    <form noValidate onSubmit={handleSubmit} className="space-y-6">
      <FormField
        id="config-role"
        label="Role"
        hint="The job title you are preparing for."
        error={errors.role}
      >
        {(a11y) => (
          <Input
            {...a11y}
            value={role}
            maxLength={MAX_ROLE}
            placeholder="e.g. Backend engineer"
            onChange={(e) => setRole(e.target.value)}
            disabled={locked}
          />
        )}
      </FormField>

      <div className="grid gap-4 sm:grid-cols-2">
        <FormField id="config-level" label="Seniority">
          {(a11y) => (
            <Select
              {...a11y}
              value={level}
              onChange={(e) => setLevel(e.target.value)}
              disabled={locked}
            >
              {options.levels.map((value) => (
                <option key={value} value={value}>
                  {levelLabel(value)}
                </option>
              ))}
            </Select>
          )}
        </FormField>
        <FormField id="config-duration" label="Length">
          {(a11y) => (
            <Select
              {...a11y}
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              disabled={locked}
            >
              {durationChoices(options).map((minutes) => (
                <option key={minutes} value={minutes}>
                  {minutes} minutes
                </option>
              ))}
            </Select>
          )}
        </FormField>
      </div>

      <fieldset className="space-y-2">
        <legend className="mb-1 text-sm font-medium">Interview style</legend>
        <div className="grid gap-2 sm:grid-cols-3">
          {options.modes.map((value) => (
            <label
              key={value}
              className={cn(
                'flex cursor-pointer gap-3 rounded-md border bg-card p-3 text-sm has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-ring',
                mode === value && 'border-primary bg-accent',
                locked && 'cursor-not-allowed opacity-60',
              )}
            >
              <input
                type="radio"
                name="mode"
                value={value}
                checked={mode === value}
                onChange={() => setMode(value)}
                disabled={locked}
                className="mt-0.5 accent-[var(--primary)]"
              />
              <span>
                <span className="block font-medium">{modeLabel(value)}</span>
                {MODE_HINTS[value] && (
                  <span className="block text-muted-foreground">{MODE_HINTS[value]}</span>
                )}
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      <fieldset className="space-y-3">
        <legend className="mb-1 text-sm font-medium">Topics</legend>
        {jdUsable ? (
          <div className="flex flex-col gap-2 sm:flex-row sm:gap-6">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="radio"
                name="topic-source"
                checked={topicSource === 'jd'}
                onChange={() => setTopicSource('jd')}
                disabled={locked}
                className="accent-[var(--primary)]"
              />
              Use topics from the job description
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="radio"
                name="topic-source"
                checked={topicSource === 'manual'}
                onChange={() => setTopicSource('manual')}
                disabled={locked}
                className="accent-[var(--primary)]"
              />
              Choose topics myself
            </label>
          </div>
        ) : (
          jd && (
            <p className="text-sm text-muted-foreground">
              The job description has no usable topics yet, so choose the topics to cover.
            </p>
          )
        )}
        {topicSource === 'jd' ? (
          <p className="text-sm text-muted-foreground">
            Covering: {jdTopics.map((key) => keyLabel(key, topicNames)).join(', ')}.
          </p>
        ) : (
          <div
            id="config-topics"
            tabIndex={-1}
            role="group"
            aria-label="Topics to cover"
            aria-describedby={errors.topics ? 'config-topics-error' : undefined}
            className="grid gap-2 rounded-md focus-visible:outline-none sm:grid-cols-2"
          >
            {options.topics.map((topic) => (
              <label
                key={topic.key}
                className={cn(
                  'flex cursor-pointer items-center gap-3 rounded-md border bg-card p-3 text-sm has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-ring',
                  selected.has(topic.key) && 'border-primary bg-accent',
                  locked && 'cursor-not-allowed opacity-60',
                )}
              >
                <input
                  type="checkbox"
                  checked={selected.has(topic.key)}
                  onChange={() => toggleTopic(topic.key)}
                  disabled={locked}
                  className="accent-[var(--primary)]"
                />
                {topic.display_name}
              </label>
            ))}
          </div>
        )}
        {errors.topics && (
          <p id="config-topics-error" className="text-xs font-medium text-destructive">
            {errors.topics}
          </p>
        )}
      </fieldset>

      <div className="rounded-md bg-muted/60 p-4 text-sm">
        <p className="font-medium">Summary</p>
        <ul className="mt-1 space-y-0.5 text-muted-foreground">
          <li>
            Resume:{' '}
            {resume
              ? resume.profile_confirmed
                ? 'attached, profile confirmed'
                : 'attached, profile not confirmed (not used)'
              : 'none'}
          </li>
          <li>Job description: {jd ? (jdUsable ? 'attached' : 'attached, no topics') : 'none'}</li>
        </ul>
      </div>

      {formError && (
        <Alert variant="error" title="The interview was not created">
          {formError}
        </Alert>
      )}
      {phase.kind === 'start-failed' && (
        <Alert variant="error" title="The interview was created but could not be started">
          <p>{phase.message}</p>
          <p className="mt-1">
            Your settings are saved. You can try starting again, or start it later from your
            dashboard.
          </p>
        </Alert>
      )}

      <div className="flex flex-col-reverse gap-3 border-t pt-6 sm:flex-row sm:justify-between">
        <Button variant="outline" onClick={onBack} disabled={locked}>
          Back
        </Button>
        {phase.kind === 'start-failed' ? (
          <div className="flex flex-col-reverse gap-3 sm:flex-row">
            <Button variant="outline" onClick={() => navigate('/dashboard')}>
              Go to dashboard
            </Button>
            <Button onClick={retryStart}>Try starting again</Button>
          </div>
        ) : (
          <Button type="submit" size="lg" loading={busy}>
            {phase.kind === 'creating'
              ? 'Creating interview…'
              : phase.kind === 'starting'
                ? 'Starting interview…'
                : 'Start interview'}
          </Button>
        )}
      </div>
    </form>
  );
}
