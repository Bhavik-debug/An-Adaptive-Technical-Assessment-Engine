import { ClipboardList, RefreshCw } from 'lucide-react';
import * as React from 'react';
import { Link, useNavigate } from 'react-router';

import { ApiError } from '@/api/client';
import { startSession } from '@/api/sessions';
import type { SessionSummary } from '@/api/types';
import { EmptyState } from '@/components/EmptyState';
import { SessionStatusBadge } from '@/components/StatusBadge';
import { Alert } from '@/components/ui/alert';
import { Button, buttonVariants } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { formatDateTime, keyLabel, levelLabel, modeLabel } from '@/lib/format';

const IN_PROGRESS = new Set(['PREPARING', 'ASKING', 'GRADING', 'PROBING', 'SELECTING']);

function SessionAction({ session }: { session: SessionSummary }) {
  const navigate = useNavigate();
  const [starting, setStarting] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const id = session.session_id;

  async function handleStart() {
    if (starting) return;
    setStarting(true);
    setError(null);
    try {
      // /start is idempotent on the backend, so a retry here is safe.
      await startSession(id);
      navigate(`/interviews/${id}`, { state: { started: true } });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Could not start the interview.');
      setStarting(false);
    }
  }

  if (session.state === 'CREATED') {
    return (
      <div className="flex flex-col items-start gap-1 sm:items-end">
        <Button size="sm" onClick={handleStart} loading={starting}>
          Start
        </Button>
        {error && (
          <p role="alert" className="text-xs text-destructive">
            {error}
          </p>
        )}
      </div>
    );
  }
  if (IN_PROGRESS.has(session.state)) {
    return (
      <Link to={`/interviews/${id}`} className={buttonVariants({ variant: 'outline', size: 'sm' })}>
        Open
      </Link>
    );
  }
  if (session.state === 'COMPLETED' || session.state === 'REPORTING') {
    return (
      <Link
        to={`/interviews/${id}/report`}
        className={buttonVariants({ variant: 'outline', size: 'sm' })}
      >
        View report
      </Link>
    );
  }
  return null;
}

export function SessionHistory({
  sessions,
  loading,
  error,
  onRetry,
  topicNames,
}: {
  sessions: SessionSummary[] | undefined;
  loading: boolean;
  error: ApiError | null;
  onRetry: () => void;
  topicNames: Map<string, string>;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Past interviews</CardTitle>
        <CardDescription>Your interview sessions, most recent first.</CardDescription>
      </CardHeader>
      <CardContent>
        {loading && !sessions ? (
          <div className="space-y-3" aria-busy="true">
            <span className="sr-only" role="status">
              Loading your interviews
            </span>
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        ) : error ? (
          <Alert variant="error" title="We could not load your interviews">
            <p>{error.message}</p>
            <Button variant="outline" size="sm" className="mt-2" onClick={onRetry}>
              <RefreshCw aria-hidden="true" />
              Try again
            </Button>
          </Alert>
        ) : !sessions || sessions.length === 0 ? (
          <EmptyState
            icon={ClipboardList}
            title="No interviews yet"
            action={
              <Link to="/interviews/new" className={buttonVariants()}>
                Start your first interview
              </Link>
            }
          >
            When you complete an interview it will appear here with its status and report.
          </EmptyState>
        ) : (
          <ul className="divide-y rounded-md border">
            {sessions.map((session) => {
              const details = [levelLabel(session.level), modeLabel(session.mode)]
                .filter(Boolean)
                .join(' · ');
              const topics = session.topic_keys.map((k) => keyLabel(k, topicNames)).join(', ');
              return (
                <li
                  key={session.session_id}
                  className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between"
                >
                  <div className="min-w-0 space-y-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="truncate font-medium">{session.role || 'Interview'}</p>
                      <SessionStatusBadge state={session.state} />
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {[details, topics].filter(Boolean).join(' — ')}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Created {formatDateTime(session.started_at)}
                      {session.ended_at && ` · Finished ${formatDateTime(session.ended_at)}`}
                    </p>
                  </div>
                  <SessionAction session={session} />
                </li>
              );
            })}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
