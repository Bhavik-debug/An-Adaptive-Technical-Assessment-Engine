import * as React from 'react';

import { ApiError } from '@/api/client';
import { MAX_ANSWER_CHARS, submitTurn } from '@/api/sessions';
import { FormField } from '@/components/FormField';
import { Alert } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';

import { answerLength, SUBMIT_TIMEOUT_MS, validateAnswer } from './content';

function newTurnId(): string {
  return crypto.randomUUID();
}

type Phase =
  | { kind: 'editing' }
  | { kind: 'sending' }
  /** The request may or may not have reached the server: only an identical resend is safe. */
  | { kind: 'unknown'; message: string }
  | { kind: 'accepted' };

/**
 * One answer to one question.
 *
 * The submission is idempotent on `turn_id` **with byte-identical text**
 * (backend/app/interview/turns.py). So, once a request has been sent:
 *
 * - success (202, or 200 for a replay) -> accepted; the parent waits for the
 *   server's state to move on (never advanced locally);
 * - a response that proves nothing was written (422, a 409 state conflict) ->
 *   editable again, same turn id;
 * - no response (network failure, timeout, 5xx) -> **outcome unknown**: the
 *   text is frozen and only an identical resend with the same turn id is
 *   offered, so a retry can never become a second, different answer. If the
 *   lost request did land, the stream's `grading_started` moves the session
 *   to GRADING and the parent unmounts this form - the question is settled
 *   by the server either way.
 *
 * The parent mounts a fresh form (new key) per question, which is what
 * generates a new turn id and clears the draft.
 */
export function AnswerForm({
  sessionId,
  onAccepted,
  onConflict,
}: {
  sessionId: string;
  onAccepted: (turnId: string) => void;
  /** The session is no longer accepting this answer: re-read the server's state. */
  onConflict: () => void;
}) {
  const [turnId] = React.useState(newTurnId);
  const [draft, setDraft] = React.useState('');
  const [phase, setPhase] = React.useState<Phase>({ kind: 'editing' });
  const [error, setError] = React.useState<string | null>(null);
  const [formError, setFormError] = React.useState<string | null>(null);
  // The exact text sent with `turnId`; a resend must use it, not the textarea.
  const sent = React.useRef<string | null>(null);
  const inFlight = React.useRef(false);
  const unmounted = React.useRef(new AbortController());

  React.useEffect(() => {
    const controller = unmounted.current;
    return () => controller.abort();
  }, []);

  // Warn before leaving with an unsent answer (the draft is kept in memory only).
  const unsent = phase.kind !== 'accepted' && draft.trim() !== '';
  React.useEffect(() => {
    if (!unsent) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [unsent]);

  async function send(answer: string) {
    if (inFlight.current) return;
    inFlight.current = true;
    sent.current = answer;
    setPhase({ kind: 'sending' });
    setFormError(null);
    const timeout = new AbortController();
    const timer = setTimeout(() => timeout.abort(), SUBMIT_TIMEOUT_MS);
    const abortOnUnmount = () => timeout.abort();
    unmounted.current.signal.addEventListener('abort', abortOnUnmount);
    try {
      await submitTurn(sessionId, { turn_id: turnId, answer }, timeout.signal);
      if (unmounted.current.signal.aborted) return;
      setPhase({ kind: 'accepted' });
      onAccepted(turnId);
    } catch (err) {
      if (unmounted.current.signal.aborted) return;
      const status = err instanceof ApiError ? err.status : -1;
      if (status === 422) {
        // Nothing was written: the server refused the content.
        sent.current = null;
        setPhase({ kind: 'editing' });
        setError(
          (err as ApiError).fieldErrors.answer ?? (err as ApiError).message ?? 'Invalid answer.',
        );
        document.getElementById('answer')?.focus();
      } else if (status === 409 || status === 404) {
        // Not accepting answers (another tab answered, or the state moved on):
        // nothing of ours was written. The server's state decides what's next.
        sent.current = null;
        setPhase({ kind: 'editing' });
        setFormError(
          status === 404
            ? 'This interview could not be found.'
            : 'This question is no longer accepting answers. Showing the latest state.',
        );
        onConflict();
      } else if (status === 401) {
        // Signed out; the auth provider takes over. The outcome is unknown.
        setPhase({
          kind: 'unknown',
          message: 'Your session ended before we could confirm your answer.',
        });
      } else {
        // Network failure, timeout (AbortError), 5xx: it may have been recorded.
        setPhase({
          kind: 'unknown',
          message:
            'We could not confirm that your answer was received. It may already be saved. ' +
            'Resending is safe: the same answer is never recorded twice.',
        });
      }
    } finally {
      clearTimeout(timer);
      unmounted.current.signal.removeEventListener('abort', abortOnUnmount);
      inFlight.current = false;
    }
  }

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (phase.kind !== 'editing') return;
    const problem = validateAnswer(draft);
    setError(problem);
    if (problem) {
      document.getElementById('answer')?.focus();
      return;
    }
    void send(draft);
  }

  if (phase.kind === 'accepted') return null;

  const locked = phase.kind !== 'editing';
  const length = answerLength(draft);
  return (
    <form noValidate onSubmit={handleSubmit} className="space-y-4">
      {formError && (
        <Alert variant="warning" title="Answer not submitted">
          {formError}
        </Alert>
      )}
      {phase.kind === 'unknown' && (
        <Alert variant="warning" title="Answer status unknown">
          <p>{phase.message}</p>
          <Button
            variant="outline"
            size="sm"
            className="mt-2"
            onClick={() => sent.current !== null && void send(sent.current)}
          >
            Resend the same answer
          </Button>
        </Alert>
      )}
      <FormField
        id="answer"
        label="Your answer"
        error={error}
        hint={
          locked
            ? 'Your answer is locked while we confirm it was received.'
            : `Explain your reasoning; plain text. ${length.toLocaleString()} / ${MAX_ANSWER_CHARS.toLocaleString()} characters.`
        }
      >
        {(a11y) => (
          <Textarea
            {...a11y}
            rows={10}
            value={draft}
            onChange={(e) => {
              setDraft(e.target.value);
              if (error) setError(null);
            }}
            readOnly={locked}
            aria-readonly={locked || undefined}
            spellCheck
            className="font-mono text-sm"
          />
        )}
      </FormField>
      <div className="flex flex-wrap items-center gap-3">
        <Button type="submit" loading={phase.kind === 'sending'} disabled={locked}>
          {phase.kind === 'sending' ? 'Submitting…' : 'Submit answer'}
        </Button>
        <p className="text-xs text-muted-foreground">
          Answers cannot be changed after they are submitted.
        </p>
      </div>
    </form>
  );
}
