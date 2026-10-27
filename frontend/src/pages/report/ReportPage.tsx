import {
  BarChart3,
  CalendarCheck,
  ChartColumn,
  CircleCheck,
  FileBarChart,
  FileQuestion,
  Gauge,
  Hourglass,
  Info,
  ListChecks,
  Loader2,
  type LucideIcon,
  MessageSquareQuote,
  Sparkles,
  Target,
  ThumbsUp,
  TrendingUp,
} from 'lucide-react';
import * as React from 'react';
import { Link, useParams } from 'react-router';

import { ApiError } from '@/api/client';
import { getReport, getSession } from '@/api/sessions';
import type { Report, SessionDetail } from '@/api/types';
import { EmptyState } from '@/components/EmptyState';
import { Alert } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button, buttonVariants } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useApiData } from '@/hooks/useApiData';
import { usePolling } from '@/hooks/usePolling';
import { useSetupOptions } from '@/hooks/useSetupOptions';
import { formatDateTime, keyLabel, levelLabel, modeLabel } from '@/lib/format';
import {
  type NarrativeView,
  parseNarrative,
  parseReportStats,
  type QuestionResult,
  type ReportView,
  scoreOutOf100,
} from '@/lib/report';

import {
  AnswerChart,
  GradeMeter,
  type SkillPoint,
  SkillRangeChart,
  StandingPanel,
  type StatTile,
  StatTiles,
  StatusLegend,
  TopicGauge,
} from './ReportVisuals';

const FINISHED_STATES = new Set(['REPORTING', 'COMPLETED']);

/** GET .../report, as the states this screen distinguishes. */
type ReportResult =
  | { kind: 'ready'; report: Report }
  | { kind: 'pending' }
  | { kind: 'failed' }
  | { kind: 'not-finished' };

async function loadReport(sessionId: string, signal: AbortSignal): Promise<ReportResult> {
  try {
    const report = await getReport(sessionId, signal);
    if (report.status === 'ready') return { kind: 'ready', report };
    if (report.status === 'failed') return { kind: 'failed' };
    return { kind: 'pending' }; // 202: pending or processing
  } catch (error) {
    // A failed report is a 500 whose body is the report itself (docs/reporting.md §4).
    if (error instanceof ApiError) {
      const status = (error.body as { status?: unknown } | null)?.status;
      if (error.status === 500 && status === 'failed') return { kind: 'failed' };
      if (error.status === 409) return { kind: 'not-finished' };
    }
    throw error;
  }
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function ReportPage() {
  const { sessionId = '' } = useParams();
  const session = useApiData((signal) => getSession(sessionId, signal), [sessionId]);
  const names = useSetupOptions();
  const finished = !!session.data && FINISHED_STATES.has(session.data.state);

  const poll = usePolling<ReportResult>(finished ? sessionId : null, {
    fetcher: (signal) => loadReport(sessionId, signal),
    isDone: (r) => r.kind !== 'pending',
    intervalMs: 2000,
  });

  // A ready report always belongs to a completed session (docs/reporting.md
  // section 1). If the details were read while it was still finishing, read
  // them once more so the header shows "Completed" and the finish time.
  const staleDetails = poll.value?.kind === 'ready' && session.data?.state !== 'COMPLETED';
  const reloadSession = session.reload;
  React.useEffect(() => {
    if (staleDetails) reloadSession();
  }, [staleDetails, reloadSession]);

  if (session.loading && !session.data) {
    return (
      <Shell>
        <p role="status" className="sr-only">
          Loading report
        </p>
        <Skeleton className="h-8 w-56" />
        <Skeleton className="h-40 w-full" />
        <Skeleton className="h-64 w-full" />
      </Shell>
    );
  }

  if (session.error?.status === 404 || poll.error?.status === 404) {
    return (
      <Shell>
        <h1 className="sr-only">Interview report</h1>
        <EmptyState icon={FileQuestion} title="Report not found" action={<BackLink />}>
          This interview does not exist, or it belongs to another account.
        </EmptyState>
      </Shell>
    );
  }

  if (session.error || !session.data) {
    return (
      <Shell>
        <h1 className="text-xl font-semibold">Interview report</h1>
        <Alert variant="error" title="We could not load this report">
          {session.error?.message ?? 'Something went wrong.'}
          <div className="mt-2">
            <Button variant="outline" size="sm" onClick={session.reload}>
              Try again
            </Button>
          </div>
        </Alert>
        <BackLink />
      </Shell>
    );
  }

  const detail = session.data;
  const header = <ReportHeader detail={detail} topicNames={names.topicNames} />;
  const result = poll.value;

  if (!finished || result?.kind === 'not-finished') {
    return (
      <Shell>
        {header}
        <EmptyState
          icon={Hourglass}
          title="This interview is not finished yet"
          action={
            <Link to={`/interviews/${encodeURIComponent(sessionId)}`} className={buttonVariants()}>
              Continue the interview
            </Link>
          }
        >
          The report is prepared once the interview ends.
        </EmptyState>
      </Shell>
    );
  }

  if (poll.phase === 'error') {
    return (
      <Shell>
        {header}
        <Alert variant="error" title="We could not load this report">
          {poll.error?.message ?? 'Something went wrong.'}
          <div className="mt-2">
            <Button variant="outline" size="sm" onClick={poll.restart}>
              Try again
            </Button>
          </div>
        </Alert>
        <BackLink />
      </Shell>
    );
  }

  if (result?.kind === 'failed') {
    return (
      <Shell>
        {header}
        <Alert variant="error" title="This report could not be prepared">
          Something went wrong while building the report. Your answers and their grades are saved,
          and the report can be prepared again once the problem is fixed. Please check back later.
        </Alert>
        <BackLink />
      </Shell>
    );
  }

  if (result?.kind !== 'ready') {
    const timedOut = poll.phase === 'timeout';
    return (
      <Shell>
        {header}
        <Card>
          <CardContent className="flex flex-col items-center gap-3 py-10 text-center">
            {timedOut ? (
              <>
                <p className="font-medium">Your report is still being prepared</p>
                <p className="max-w-md text-sm text-muted-foreground">
                  This is taking longer than usual. It will appear here when it is ready.
                </p>
                <Button variant="outline" onClick={poll.restart}>
                  Check again
                </Button>
              </>
            ) : (
              <p role="status" className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" aria-hidden="true" />
                Preparing your report…
              </p>
            )}
          </CardContent>
        </Card>
        <BackLink />
      </Shell>
    );
  }

  const view = parseReportStats(result.report.stats);
  if (!view) {
    return (
      <Shell>
        {header}
        <Alert variant="error" title="This report could not be displayed">
          The report was returned in a form this page does not understand. Nothing has been
          estimated in its place.
        </Alert>
        <BackLink />
      </Shell>
    );
  }

  return (
    <Shell>
      {header}
      <ReadyReport
        view={view}
        narrative={parseNarrative(result.report.narrative)}
        narrativeSource={result.report.narrative_source}
        generatedAt={result.report.generated_at}
        topicNames={names.topicNames}
        subtopicNames={names.subtopicNames}
      />
      <BackLink />
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="mx-auto max-w-5xl space-y-6">{children}</div>;
}

function BackLink() {
  return (
    <Link to="/dashboard" className={buttonVariants({ variant: 'outline' })}>
      Back to dashboard
    </Link>
  );
}

function ReportHeader({
  detail,
  topicNames,
}: {
  detail: SessionDetail;
  topicNames: Map<string, string>;
}) {
  const meta = [levelLabel(detail.level), modeLabel(detail.mode)].filter(Boolean).join(' · ');
  const done = detail.state === 'COMPLETED';
  return (
    <header className="relative overflow-hidden rounded-xl border bg-gradient-to-br from-primary/[0.09] via-accent/60 to-card p-5 shadow-xs sm:p-6">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -right-16 -top-20 size-56 rounded-full bg-primary/[0.07] blur-2xl"
      />
      <div className="relative flex items-start gap-4">
        <span className="hidden size-11 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm sm:flex">
          <FileBarChart className="size-5" aria-hidden="true" />
        </span>
        <div className="min-w-0 space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wider text-primary">
            Interview report
          </p>
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            {detail.role ?? 'Interview'}
          </h1>
          <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
            {meta && <span>{meta}</span>}
            <Badge variant={done ? 'success' : 'info'}>{done ? 'Completed' : 'Finishing'}</Badge>
            {detail.ended_at && (
              <span className="flex items-center gap-1">
                <CalendarCheck className="size-3.5" aria-hidden="true" />
                Finished {formatDateTime(detail.ended_at)}
              </span>
            )}
          </div>
          {detail.topic_keys.length > 0 && (
            <ul className="flex flex-wrap gap-1.5 pt-1" aria-label="Topics in this interview">
              {detail.topic_keys.map((key) => (
                <li key={key}>
                  <span className="inline-flex items-center rounded-full border bg-card/80 px-2.5 py-0.5 text-xs font-medium">
                    {keyLabel(key, topicNames)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </header>
  );
}

// ---------------------------------------------------------------------------
// The ready report
// ---------------------------------------------------------------------------

function SectionTitle({ icon: Icon, children }: { icon: LucideIcon; children: React.ReactNode }) {
  return (
    <CardTitle as="h2" className="flex items-center gap-2">
      <span className="flex size-7 items-center justify-center rounded-md bg-primary/10 text-primary">
        <Icon className="size-4" aria-hidden="true" />
      </span>
      {children}
    </CardTitle>
  );
}

function ReadyReport({
  view,
  narrative,
  narrativeSource,
  generatedAt,
  topicNames,
  subtopicNames,
}: {
  view: ReportView;
  narrative: NarrativeView | null;
  narrativeSource: string | null;
  generatedAt: string | null;
  topicNames: Map<string, string>;
  subtopicNames: Map<string, string>;
}) {
  const sub = (key: string) => keyLabel(key, subtopicNames);
  const { primaryQuestions, followUpsAsked, answersGraded, gradingFailures } = view.counts;

  // Measured skills, as the backend classified them. Too-uncertain or
  // unmeasured skills are not plotted; they are in "Needs more assessment".
  const strengths = new Set(view.strengths);
  const gaps = new Set(view.gaps);
  const skills: SkillPoint[] = view.subtopics
    .filter((s) => s.observed && !s.observed.insufficientEvidence)
    .map((s) => ({
      key: s.subtopicKey,
      name: sub(s.subtopicKey),
      topicName: s.topicKey ? keyLabel(s.topicKey, topicNames) : null,
      observed: s.observed!,
      status: strengths.has(s.subtopicKey)
        ? ('strength' as const)
        : gaps.has(s.subtopicKey)
          ? ('gap' as const)
          : ('measured' as const),
    }))
    .sort((a, b) => b.observed.display - a.observed.display);

  const tiles: StatTile[] = [
    primaryQuestions !== null && { label: 'Questions', value: primaryQuestions, icon: ListChecks },
    followUpsAsked !== null && {
      label: 'Follow-ups',
      value: followUpsAsked,
      icon: MessageSquareQuote,
    },
    answersGraded !== null && { label: 'Answers graded', value: answersGraded, icon: CircleCheck },
    { label: 'Skills measured', value: skills.length, icon: Gauge },
  ].filter((t): t is StatTile => !!t);

  return (
    <>
      <StatTiles tiles={tiles} />

      <Card className="overflow-hidden">
        <div
          className="h-1 bg-gradient-to-r from-primary via-primary/60 to-viz-strength/70"
          aria-hidden="true"
        />
        <CardHeader>
          <SectionTitle icon={Sparkles}>Summary</SectionTitle>
          {primaryQuestions !== null && (
            <CardDescription>
              {plural(primaryQuestions, 'question')}
              {followUpsAsked !== null && <> and {plural(followUpsAsked, 'follow-up')}</>}
              {generatedAt && <> · report prepared {formatDateTime(generatedAt)}</>}
            </CardDescription>
          )}
        </CardHeader>
        <CardContent className="space-y-3">
          {narrative?.summary ? (
            <p className="text-[15px] leading-relaxed">{narrative.summary}</p>
          ) : (
            <p className="text-sm text-muted-foreground">No written summary is available.</p>
          )}
          {narrative?.summary && (
            <p className="flex items-start gap-1.5 text-xs text-muted-foreground">
              <Info className="mt-px size-3.5 shrink-0" aria-hidden="true" />
              {narrativeSource === 'model'
                ? 'Written by an AI model from the computed results. It describes them; it did not produce any of the numbers.'
                : 'A standard summary generated from the computed results.'}
            </p>
          )}
          {!!gradingFailures && (
            <Alert variant="warning">
              {plural(gradingFailures, 'answer')} could not be graded and{' '}
              {gradingFailures === 1 ? 'is' : 'are'} not counted in these results.
            </Alert>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <SectionTitle icon={BarChart3}>Results by topic</SectionTitle>
          <CardDescription>
            Your estimated level in each topic on a 0–100 scale. The shaded arc is the likely range.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {view.topics.length === 0 ? (
            <p className="text-sm text-muted-foreground">No topic results were recorded.</p>
          ) : (
            <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {view.topics.map((topic) => (
                <TopicGauge
                  key={topic.topicKey}
                  name={keyLabel(topic.topicKey, topicNames)}
                  observed={topic.observed}
                  questionsAsked={topic.questionsAsked}
                  skillsMeasured={
                    view.subtopics.filter(
                      (s) =>
                        s.topicKey === topic.topicKey &&
                        s.observed &&
                        !s.observed.insufficientEvidence,
                    ).length
                  }
                />
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <SectionTitle icon={Gauge}>Skill estimates</SectionTitle>
          <CardDescription>
            Each measured skill: the dot is the estimate and the band its likely range. Hover a row
            for details, or show the table.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <SkillRangeChart skills={skills} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <SectionTitle icon={Target}>Where you stand</SectionTitle>
          <CardDescription>
            Compared with the level you chose. Only skills measured precisely enough are listed as
            strengths or gaps.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid items-start gap-3 md:grid-cols-3">
          <StandingPanel
            kind="gap"
            title="Areas to improve"
            hint="Measured as below your level, weakest first."
            items={view.gaps.map(sub)}
            empty="None identified."
            ordered
          />
          <StandingPanel
            kind="strength"
            title="Strengths"
            hint="Measured as above your level."
            items={view.strengths.map(sub)}
            empty="None identified."
          />
          <StandingPanel
            kind="more"
            title="Needs more assessment"
            hint="Not asked, or too few answers to tell."
            items={view.needsMoreAssessment.map(sub)}
            empty="None."
          />
        </CardContent>
      </Card>

      {narrativeSource === 'model' &&
        narrative &&
        (narrative.strengths.length > 0 ||
          narrative.growthAreas.length > 0 ||
          narrative.nextSteps.length > 0) && (
          <Card>
            <CardHeader>
              <SectionTitle icon={MessageSquareQuote}>Feedback</SectionTitle>
              <CardDescription>
                Written by an AI model from the results above. Treat it as suggestions.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid items-start gap-3 md:grid-cols-3">
              {(
                [
                  ['What went well', narrative.strengths, ThumbsUp],
                  ['To work on', narrative.growthAreas, TrendingUp],
                  ['Next steps', narrative.nextSteps, ListChecks],
                ] as const
              )
                .filter(([, items]) => items.length > 0)
                .map(([title, items, Icon]) => (
                  <section key={title} className="space-y-2 rounded-lg border bg-muted/30 p-4">
                    <h3 className="flex items-center gap-2 text-sm font-semibold">
                      <Icon className="size-4 text-primary" aria-hidden="true" />
                      {title}
                    </h3>
                    <ul className="space-y-1.5 text-sm">
                      {items.map((item, i) => (
                        <li key={`${item}-${i}`} className="flex items-start gap-2">
                          <span
                            className="mt-1.5 size-1.5 shrink-0 rounded-full bg-primary/60"
                            aria-hidden="true"
                          />
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </section>
                ))}
            </CardContent>
          </Card>
        )}

      <QuestionList questions={view.questions} topicNames={topicNames} subtopicName={sub} />

      <Card className="border-primary/15 bg-accent/40">
        <CardHeader>
          <SectionTitle icon={Info}>How to read this report</SectionTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-muted-foreground">
          <p>
            Each skill score is an estimate on a 0–100 scale, where higher means stronger. The bar
            shows the estimate and the shaded band its likely range (95%). A wide band means fewer
            or less consistent answers, so the estimate is less certain. Skills without enough
            evidence are labelled rather than given a number.
          </p>
          <StatusLegend />
          <p>
            Scores come from an automated grader that is still being validated against human
            reviewers. Treat them as practice feedback, not a formal assessment.
          </p>
          {view.resumePriors && (
            <p>
              This interview started from your resume.{' '}
              {view.priorsNote ?? 'Starting points are not evidence of skill.'} Results above come
              only from your answers.
            </p>
          )}
          {view.methodologyNotes.length > 0 && (
            <details className="rounded-md border bg-card p-3">
              <summary className="cursor-pointer font-medium text-foreground">Method notes</summary>
              <ul className="mt-2 list-disc space-y-1 pl-5">
                {view.methodologyNotes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            </details>
          )}
        </CardContent>
      </Card>
    </>
  );
}

function ScoreText({ score }: { score: number | null }) {
  return score === null ? (
    <span className="text-muted-foreground">Not graded</span>
  ) : (
    <span className="font-semibold tabular-nums">
      {scoreOutOf100(score)}
      <span className="font-normal text-muted-foreground"> / 100</span>
    </span>
  );
}

function QuestionList({
  questions,
  topicNames,
  subtopicName,
}: {
  questions: QuestionResult[];
  topicNames: Map<string, string>;
  subtopicName: (key: string) => string;
}) {
  if (questions.length === 0) return null;
  return (
    <Card>
      <CardHeader>
        <SectionTitle icon={ChartColumn}>Question by question</SectionTitle>
        <CardDescription>
          The grade for each answer: how much of what a strong answer covers, and how well.
          Follow-up answers are graded against the original question at half weight, so a correct
          but narrow follow-up can score low.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <AnswerChart questions={questions} />
        <ol className="divide-y rounded-lg border" aria-label="Answer grades">
          {questions.map((q) => (
            <li
              key={q.order}
              className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-3"
            >
              <div className="flex min-w-0 items-center gap-3">
                <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
                  Q{q.order}
                </span>
                <div className="min-w-0 space-y-0.5">
                  <p className="text-sm font-medium">Question {q.order}</p>
                  <p className="text-xs text-muted-foreground">
                    {[
                      q.topicKey && keyLabel(q.topicKey, topicNames),
                      q.subtopicKey && subtopicName(q.subtopicKey),
                    ]
                      .filter(Boolean)
                      .join(' › ')}
                  </p>
                </div>
              </div>
              <div className="space-y-1 text-sm sm:text-right">
                <div className="flex items-center gap-3 sm:justify-end">
                  {q.score !== null && <GradeMeter score={q.score} />}
                  <ScoreText score={q.score} />
                </div>
                {q.followUpScores.length > 0 && (
                  <p className="text-xs text-muted-foreground">
                    Follow-ups:{' '}
                    {q.followUpScores
                      .map((s) => (s === null ? 'not graded' : `${scoreOutOf100(s)}/100`))
                      .join(', ')}
                  </p>
                )}
              </div>
            </li>
          ))}
        </ol>
      </CardContent>
    </Card>
  );
}
