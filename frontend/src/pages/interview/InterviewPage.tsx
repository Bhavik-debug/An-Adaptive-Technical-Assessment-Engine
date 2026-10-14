import { CircleCheck, FileQuestion, Loader2 } from 'lucide-react';
import * as React from 'react';
import { Link, useLocation, useParams } from 'react-router';

import { ApiError } from '@/api/client';
import { startSession } from '@/api/sessions';
import { EmptyState } from '@/components/EmptyState';
import { Alert } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button, buttonVariants } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useSetupOptions } from '@/hooks/useSetupOptions';
import { keyLabel, levelLabel, modeLabel } from '@/lib/format';

import { AnswerForm } from './AnswerForm';
import { acceptsAnswer, isFinished, questionLabel, type InterviewView } from './interviewModel';
import { QuestionText } from './QuestionText';
import { useInterview, type Connection } from './useInterview';

const WAITING: Record<string, string> = {
  CREATED: 'This interview has not started yet.',
  PREPARING: 'Preparing your first question…',
  PROBING: 'Preparing a follow-up question…',
  SELECTING: 'Choosing your next question…',
};

function ConnectionNote({ connection }: { connection: Connection }) {
  if (connection === 'live' || connection === 'closed') return null;
  const text =
    connection === 'connecting'
      ? 'Connecting to live updates…'
      : connection === 'reconnecting'
        ? 'Live updates were interrupted. Reconnecting…'
        : 'Live updates are unavailable.';
  return (
    <p role="status" className="text-xs text-muted-foreground">
      {text}
    </p>
  );
}

function Waiting({ children }: { children: React.ReactNode }) {
  return (
    <div role="status" className="flex items-center gap-2 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" aria-hidden="true" />
      <span>{children}</span>
    </div>
  );
}

function ReviewNotice({ view }: { view: InterviewView }) {
  if (view.review === 'failed') {
    return (
      <Alert variant="warning" title="We could not review this answer">
        Your answer was saved, but it could not be reviewed after several attempts, so the interview
        is paused. Nothing is lost: it stays on your dashboard, and you can come back to it later.
      </Alert>
    );
  }
  return (
    <div className="space-y-2">
      <Alert variant="success" title="Answer received">
        Your answer has been saved and is being reviewed.
      </Alert>
      <Waiting>
        {view.review === 'retrying'
          ? 'The review is taking longer than usual and is being retried automatically…'
          : 'Reviewing your answer…'}
      </Waiting>
    </div>
  );
}

function Finished({ view }: { view: InterviewView }) {
  const done = view.state === 'COMPLETED';
  return (
    <div className="space-y-4">
      <Alert variant="success" title="Interview complete">
        {done
          ? 'Thank you. Your interview is finished and your report has been prepared.'
          : 'Thank you. Your interview is finished and your report is being prepared.'}
      </Alert>
      <div className="flex flex-wrap gap-3">
        <Link
          to={`/interviews/${encodeURIComponent(view.sessionId)}/report`}
          className={buttonVariants()}
        >
          View report
        </Link>
        <Link to="/dashboard" className={buttonVariants({ variant: 'outline' })}>
          Back to dashboard
        </Link>
      </div>
    </div>
  );
}

function StartPrompt({ sessionId, onStarted }: { sessionId: string; onStarted: () => void }) {
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  async function start() {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      // Idempotent: a repeat returns the same first question.
      await startSession(sessionId);
      onStarted();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'The interview could not be started.');
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">{WAITING.CREATED}</p>
      {error && (
        <Alert variant="error" title="Could not start">
          {error}
        </Alert>
      )}
      <Button onClick={start} loading={busy}>
        {busy ? 'Starting…' : 'Start interview'}
      </Button>
    </div>
  );
}

export function InterviewPage() {
  const { sessionId = '' } = useParams();
  const { load, view, connection, streamError, reload } = useInterview(sessionId);
  const location = useLocation();
  // Set by the setup wizard only after POST /start succeeded (server-confirmed).
  const handedOff = (location.state as { started?: boolean } | null)?.started === true;
  const options = useSetupOptions();
  const [acceptedFor, setAcceptedFor] = React.useState<string | null>(null);
  const headingRef = React.useRef<HTMLSpanElement>(null);

  // Move focus to a new question so keyboard and screen-reader users land on it.
  const askingKey = view && acceptsAnswer(view) ? `q${view.questionSeq}` : null;
  const lastAsked = React.useRef<string | null>(null);
  React.useEffect(() => {
    if (askingKey && lastAsked.current !== null && askingKey !== lastAsked.current) {
      headingRef.current?.focus();
    }
    if (askingKey) lastAsked.current = askingKey;
  }, [askingKey]);

  // Which question an answer was accepted for (by this screen), so the next
  // question's form is never hidden by an earlier acceptance.
  const onAccepted = React.useCallback(() => setAcceptedFor(askingKeyRef.current), []);
  const askingKeyRef = React.useRef<string | null>(null);
  askingKeyRef.current = askingKey;

  const startedBanner =
    handedOff && (!view || (view.state === 'ASKING' && view.questionsAsked <= 1)) ? (
      <Alert variant="success" title="Your interview has started">
        The server confirmed the session is running and your first question is ready.
      </Alert>
    ) : null;

  if (load.kind === 'loading' || (load.kind === 'ready' && !view)) {
    return (
      <div className="mx-auto max-w-3xl space-y-4" aria-busy="true">
        {startedBanner}
        <p role="status" className="sr-only">
          Loading your interview
        </p>
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  if (load.kind === 'not-found') {
    return (
      <div className="mx-auto max-w-2xl">
        <h1 className="sr-only">Interview</h1>
        <EmptyState
          icon={FileQuestion}
          title="Interview not found"
          action={
            <Link to="/dashboard" className={buttonVariants({ variant: 'outline' })}>
              Back to dashboard
            </Link>
          }
        >
          This interview does not exist, or it belongs to another account.
        </EmptyState>
      </div>
    );
  }

  if (load.kind === 'error' || !view) {
    return (
      <div className="mx-auto max-w-2xl space-y-4">
        {startedBanner}
        <h1 className="text-xl font-semibold">Interview</h1>
        <Alert variant="error" title="The interview could not be loaded">
          <p>{load.kind === 'error' ? load.message : 'Please try again.'}</p>
          <Button variant="outline" size="sm" className="mt-2" onClick={reload}>
            Try again
          </Button>
        </Alert>
      </div>
    );
  }

  const answering = acceptsAnswer(view);
  const finished = isFinished(view.state);
  // Answered here, confirmed by the server, before the stream caught up.
  const justAnswered = answering && askingKey !== null && acceptedFor === askingKey;
  const meta = [levelLabel(view.level), modeLabel(view.mode)].filter(Boolean).join(' · ');

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      {startedBanner}
      <header className="space-y-2">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h1 className="text-2xl font-semibold tracking-tight">{view.role ?? 'Interview'}</h1>
          <ConnectionNote connection={connection} />
        </div>
        {meta && <p className="text-sm text-muted-foreground">{meta}</p>}
        {view.topicKeys.length > 0 && (
          <div>
            <h2 className="sr-only">Topics in this interview</h2>
            <ul className="flex flex-wrap gap-1.5" aria-label="Topics in this interview">
              {view.topicKeys.map((key) => (
                <li key={key}>
                  <Badge>{keyLabel(key, options.topicNames)}</Badge>
                </li>
              ))}
            </ul>
          </div>
        )}
      </header>

      {streamError && (
        <Alert variant="warning" title="Live updates unavailable">
          <p>{streamError}</p>
          <Button variant="outline" size="sm" className="mt-2" onClick={reload}>
            Refresh
          </Button>
        </Alert>
      )}

      <Card>
        <CardHeader>
          {view.question && !finished ? (
            <>
              <CardTitle as="h2" className="text-lg">
                <span ref={headingRef} tabIndex={-1} className="outline-none">
                  {questionLabel(view)}
                </span>
              </CardTitle>
              <CardDescription>
                {view.questionsAsked} {view.questionsAsked === 1 ? 'question' : 'questions'} asked
                so far
                {view.followUpsAsked > 0 &&
                  `, plus ${view.followUpsAsked} follow-up${view.followUpsAsked === 1 ? '' : 's'}`}
                . The interview ends when enough has been measured, so there is no fixed count.
              </CardDescription>
            </>
          ) : (
            <CardTitle as="h2" className="text-lg">
              {finished ? (
                <span className="inline-flex items-center gap-2">
                  <CircleCheck className="size-5 text-success" aria-hidden="true" />
                  Finished
                </span>
              ) : (
                'Interview'
              )}
            </CardTitle>
          )}
        </CardHeader>
        <CardContent className="space-y-6">
          {finished ? (
            <Finished view={view} />
          ) : (
            <>
              {view.question &&
                (view.question.text ? (
                  <QuestionText text={view.question.text} />
                ) : (
                  <Alert variant="warning" title="Question wording unavailable">
                    This session was recorded before question wording was stored, so it cannot be
                    shown here.
                  </Alert>
                ))}

              {view.state === 'CREATED' && <StartPrompt sessionId={sessionId} onStarted={reload} />}
              {WAITING[view.state] && view.state !== 'CREATED' && (
                <Waiting>{WAITING[view.state]}</Waiting>
              )}
              {view.state === 'GRADING' && <ReviewNotice view={view} />}
              {justAnswered && <Waiting>Answer received. Waiting for the next step…</Waiting>}

              {answering && view.question?.text && !justAnswered && (
                <AnswerForm
                  key={askingKey ?? 'none'}
                  sessionId={sessionId}
                  onAccepted={onAccepted}
                  onConflict={reload}
                />
              )}
            </>
          )}
        </CardContent>
      </Card>

      {!finished && (
        <p className="text-xs text-muted-foreground">
          Your progress is saved on the server after every answer. You can leave and{' '}
          <Link to="/dashboard" className="underline underline-offset-4">
            return from your dashboard
          </Link>{' '}
          at any time.
        </p>
      )}
    </div>
  );
}
