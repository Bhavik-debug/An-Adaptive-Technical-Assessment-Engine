import { BarChart3 } from 'lucide-react';

import { getReport } from '@/api/sessions';
import type { ReportTopic, SessionSummary } from '@/api/types';
import { EmptyState } from '@/components/EmptyState';
import { Alert } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useApiData } from '@/hooks/useApiData';
import { formatDateTime, keyLabel } from '@/lib/format';
import { latestReportedSession } from '@/lib/session';

function TopicRow({ topic, name }: { topic: ReportTopic; name: string }) {
  const observed = topic.observed;
  if (!observed || observed.insufficient_evidence) {
    return (
      <li className="flex flex-wrap items-center justify-between gap-2 py-2.5">
        <span className="text-sm font-medium">{name}</span>
        <Badge variant="neutral">{observed ? 'Not enough evidence yet' : 'Not assessed'}</Badge>
      </li>
    );
  }
  const [low, high] = observed.interval;
  return (
    <li className="space-y-1.5 py-2.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-medium">{name}</span>
        <span className="text-sm tabular-nums">
          {observed.display}
          <span className="text-muted-foreground"> / 100</span>
        </span>
      </div>
      <div
        className="relative h-2 rounded-full bg-muted"
        role="img"
        aria-label={`${name}: ${observed.display} out of 100, likely range ${low} to ${high}`}
      >
        <div
          className="absolute inset-y-0 rounded-full bg-primary/25"
          style={{ left: `${low}%`, width: `${Math.max(high - low, 1)}%` }}
        />
        <div
          className="absolute inset-y-0 w-1 -translate-x-1/2 rounded-full bg-primary"
          style={{ left: `${observed.display}%` }}
        />
      </div>
      <p className="text-xs text-muted-foreground">
        Likely range {low}–{high} · {topic.questions_asked} question
        {topic.questions_asked === 1 ? '' : 's'}
      </p>
    </li>
  );
}

function ReportedSkills({
  session,
  topicNames,
}: {
  session: SessionSummary;
  topicNames: Map<string, string>;
}) {
  const report = useApiData(
    (signal) => getReport(session.session_id, signal),
    [session.session_id],
  );
  if (report.loading) {
    return (
      <div className="space-y-3" aria-busy="true">
        <span className="sr-only" role="status">
          Loading skill overview
        </span>
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
      </div>
    );
  }
  if (report.error) {
    return (
      <Alert variant="error" title="We could not load your latest report">
        {report.error.message}
      </Alert>
    );
  }
  const topics = report.data?.stats?.topics ?? [];
  if (report.data?.status !== 'ready' || topics.length === 0) {
    return <NoSkillData />;
  }
  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        From your interview for <span className="font-medium text-foreground">{session.role}</span>
        {session.ended_at && <> on {formatDateTime(session.ended_at)}</>}.
      </p>
      <ul className="divide-y">
        {topics.map((topic) => (
          <TopicRow
            key={topic.topic_key}
            topic={topic}
            name={keyLabel(topic.topic_key, topicNames)}
          />
        ))}
      </ul>
      <p className="text-xs text-muted-foreground">
        Scores come from an automated grader that is still being validated against human reviewers.
        Treat them as practice feedback, not a formal assessment.
      </p>
    </div>
  );
}

function NoSkillData() {
  return (
    <EmptyState icon={BarChart3} title="No skill data yet">
      Your skill overview appears after you complete an interview and its report is ready. We only
      show results measured from your answers — never estimates or sample data.
    </EmptyState>
  );
}

export function SkillOverview({
  sessions,
  loading,
  topicNames,
}: {
  sessions: SessionSummary[] | undefined;
  loading: boolean;
  topicNames: Map<string, string>;
}) {
  const latest = latestReportedSession(sessions);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Skill overview</CardTitle>
        <CardDescription>Results from your most recent completed interview.</CardDescription>
      </CardHeader>
      <CardContent>
        {loading && !sessions ? (
          <Skeleton className="h-24 w-full" />
        ) : latest ? (
          <ReportedSkills session={latest} topicNames={topicNames} />
        ) : (
          <NoSkillData />
        )}
      </CardContent>
    </Card>
  );
}
