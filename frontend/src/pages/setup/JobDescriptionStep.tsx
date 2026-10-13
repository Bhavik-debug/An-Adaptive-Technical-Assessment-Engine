import * as React from 'react';

import { ApiError } from '@/api/client';
import {
  JD_MAX_CHARS,
  JD_MIN_CHARS,
  getJobDescription,
  submitJobDescription,
} from '@/api/ingestion';
import type { JobDescription } from '@/api/types';
import { FormField } from '@/components/FormField';
import { Alert } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { usePolling } from '@/hooks/usePolling';
import { keyLabel } from '@/lib/format';
import { jdTopicKeys } from '@/lib/setup';

const SETTLED = new Set(['ready', 'failed']);

function validateJd(text: string): string | null {
  const length = text.trim().length;
  if (length === 0) return 'Paste the job description, or skip this step.';
  if (length < JD_MIN_CHARS)
    return `That is too short to extract requirements from. Paste at least ${JD_MIN_CHARS} characters.`;
  if (length > JD_MAX_CHARS)
    return `That is longer than the ${JD_MAX_CHARS.toLocaleString()}-character limit.`;
  return null;
}

export function JobDescriptionStep({
  jd,
  onJdChange,
  text,
  onTextChange,
  topicNames,
  onBack,
  onContinue,
  onSkip,
}: {
  jd: JobDescription | null;
  onJdChange: (jd: JobDescription | null) => void;
  text: string;
  onTextChange: (text: string) => void;
  topicNames: Map<string, string>;
  onBack: () => void;
  onContinue: () => void;
  onSkip: () => void;
}) {
  const [error, setError] = React.useState<string | null>(null);
  const [submitting, setSubmitting] = React.useState(false);
  const [submittedText, setSubmittedText] = React.useState<string | null>(jd ? text : null);

  const jdId = jd?.jd_id ?? null;
  const settled = jd ? SETTLED.has(jd.status) : false;
  const poll = usePolling<JobDescription>(jd && !settled ? jdId : null, {
    fetcher: (signal) => getJobDescription(jdId as string, signal),
    isDone: (value) => SETTLED.has(value.status),
  });
  React.useEffect(() => {
    if (poll.value) onJdChange(poll.value);
  }, [poll.value, onJdChange]);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitting) return;
    const problem = validateJd(text);
    setError(problem);
    if (problem) {
      document.getElementById('jd-text')?.focus();
      return;
    }
    setSubmitting(true);
    try {
      const created = await submitJobDescription(text);
      setSubmittedText(text);
      onJdChange(created);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? (err.fieldErrors.text ?? err.message)
          : 'Could not submit the job description.',
      );
    } finally {
      setSubmitting(false);
    }
  }

  const edited = submittedText !== null && submittedText !== text;
  const processing = Boolean(jd && !settled && poll.phase === 'polling');
  const topics = jdTopicKeys(jd);
  const trimmedLength = text.trim().length;

  return (
    <div className="space-y-6">
      <form noValidate onSubmit={handleSubmit} className="space-y-3">
        <FormField
          id="jd-text"
          label="Job description"
          hint={`Paste the full posting. ${trimmedLength.toLocaleString()} / ${JD_MAX_CHARS.toLocaleString()} characters.`}
          error={error}
        >
          {(a11y) => (
            <Textarea
              {...a11y}
              rows={10}
              value={text}
              onChange={(e) => onTextChange(e.target.value)}
              disabled={submitting}
              placeholder="Paste the job description here…"
            />
          )}
        </FormField>
        <div className="flex flex-wrap items-center gap-3">
          <Button
            type="submit"
            variant={jd && !edited ? 'outline' : 'default'}
            loading={submitting}
          >
            {jd ? 'Submit updated description' : 'Analyse job description'}
          </Button>
          {edited && (
            <span className="text-xs text-muted-foreground">
              You have edited the text since it was analysed.
            </span>
          )}
        </div>
      </form>

      {processing && (
        <Alert variant="info" title="Analysing the job description…">
          This usually takes a few seconds.
        </Alert>
      )}
      {poll.phase === 'timeout' && (
        <Alert variant="warning" title="Still analysing">
          <p>
            This is taking longer than expected. You can check again, or continue and choose topics
            yourself in the next step.
          </p>
          <Button variant="outline" size="sm" className="mt-2" onClick={poll.restart}>
            Check again
          </Button>
        </Alert>
      )}
      {poll.phase === 'error' && poll.error && (
        <Alert variant="error" title="We could not check the analysis status">
          <p>{poll.error.message}</p>
          <Button variant="outline" size="sm" className="mt-2" onClick={poll.restart}>
            Try again
          </Button>
        </Alert>
      )}
      {jd?.status === 'failed' && (
        <Alert variant="warning" title="We could not analyse this job description">
          You can edit and resubmit it, or continue and choose topics yourself.
          {jd.error && <span className="block text-xs">Reason code: {jd.error}</span>}
        </Alert>
      )}

      {jd?.status === 'ready' && (
        <section aria-labelledby="jd-result-heading" className="space-y-3 rounded-md border p-4">
          <h3 id="jd-result-heading" className="font-medium">
            {jd.title ? `What we found for “${jd.title}”` : 'What we found'}
          </h3>
          {topics.length > 0 ? (
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Topics this role emphasises:</p>
              <ul className="flex flex-wrap gap-2">
                {topics.map((key) => (
                  <li key={key}>
                    <Badge variant="info">{keyLabel(key, topicNames)}</Badge>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              None of the requirements matched a topic we currently assess. You will choose topics
              in the next step.
            </p>
          )}
          {jd.requirements.length > 0 && (
            <details className="text-sm">
              <summary className="cursor-pointer rounded-sm font-medium text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                Requirements quoted from the posting ({jd.requirements.length})
              </summary>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-muted-foreground">
                {jd.requirements.map((req, index) => (
                  <li key={index}>
                    “{req.quote}”{' '}
                    <span className="text-xs">({req.importance.replace(/_/g, ' ')})</span>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </section>
      )}

      <div className="flex flex-col-reverse gap-3 border-t pt-6 sm:flex-row sm:justify-between">
        <div className="flex flex-col-reverse gap-3 sm:flex-row">
          <Button variant="outline" onClick={onBack} disabled={submitting}>
            Back
          </Button>
          <Button variant="ghost" onClick={onSkip} disabled={submitting}>
            Skip — continue without a job description
          </Button>
        </div>
        <Button onClick={onContinue} disabled={!jd || submitting}>
          Continue
        </Button>
      </div>
    </div>
  );
}
