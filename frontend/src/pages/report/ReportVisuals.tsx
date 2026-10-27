/**
 * The report's charts and panels (Day 28). Presentation only: every value
 * drawn here comes from the parsed report (`lib/report.ts`); nothing is
 * computed beyond placing the backend's numbers on a 0-100 axis.
 *
 * Hover tooltips only add convenience. Every value is also printed, in each
 * mark's accessible name, and (skills) in the table view, so keyboard and
 * screen-reader users lose nothing.
 *
 * Colour carries one meaning, the backend's classification of a measured
 * skill: strength (`--viz-strength`), area to improve (`--viz-gap`), or
 * measured (`--viz-measured`). It is never the only channel. Each mark also
 * has a visible value, an icon and label, an accessible name, and a table
 * view. Text is always in text colours, never in a mark colour.
 */
import { Award, CircleHelp, Gauge, type LucideIcon, Table2, Target } from 'lucide-react';
import * as React from 'react';

import { Badge } from '@/components/ui/badge';
import type { AbilityDisplay, QuestionResult } from '@/lib/report';
import { scoreOutOf100 } from '@/lib/report';
import { cn } from '@/lib/utils';

// ---------------------------------------------------------------------------
// Classification styling
// ---------------------------------------------------------------------------

export type SkillStatus = 'strength' | 'gap' | 'measured';

const STATUS: Record<
  SkillStatus,
  {
    label: string;
    icon: LucideIcon;
    dot: string;
    band: string;
    badge: 'success' | 'warning' | 'info';
  }
> = {
  strength: {
    label: 'Strength',
    icon: Award,
    dot: 'bg-viz-strength',
    band: 'bg-viz-strength/20',
    badge: 'success',
  },
  gap: {
    label: 'Area to improve',
    icon: Target,
    dot: 'bg-viz-gap',
    band: 'bg-viz-gap/25',
    badge: 'warning',
  },
  measured: {
    label: 'Measured',
    icon: Gauge,
    dot: 'bg-viz-measured',
    band: 'bg-viz-measured/20',
    badge: 'info',
  },
};

export function StatusLegend() {
  return (
    <ul
      className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-muted-foreground"
      aria-label="Legend"
    >
      {(Object.keys(STATUS) as SkillStatus[]).map((key) => (
        <li key={key} className="flex items-center gap-1.5">
          <span
            className={cn('size-2.5 rounded-full ring-2 ring-card', STATUS[key].dot)}
            aria-hidden="true"
          />
          {STATUS[key].label}
        </li>
      ))}
      <li className="flex items-center gap-1.5">
        <span className="h-2 w-5 rounded-full bg-viz-measured/20" aria-hidden="true" />
        Likely range (95%)
      </li>
    </ul>
  );
}

const pct = (value: number) => `${Math.min(100, Math.max(0, value))}%`;

/** Keep a tooltip inside its chart near the axis ends. */
function edgeShift(position: number): string {
  if (position < 12) return 'translate-x-0';
  if (position > 88) return '-translate-x-full';
  return '-translate-x-1/2';
}

function Tooltip({ at, children }: { at: number; children: React.ReactNode }) {
  return (
    <span
      role="presentation"
      className={cn(
        'pointer-events-none absolute bottom-full z-10 mb-2 whitespace-nowrap rounded-md bg-foreground px-2 py-1 text-xs text-background shadow-md',
        edgeShift(at),
      )}
      style={{ left: pct(at) }}
    >
      {children}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Stat tiles
// ---------------------------------------------------------------------------

export interface StatTile {
  label: string;
  value: number;
  icon: LucideIcon;
}

export function StatTiles({ tiles }: { tiles: StatTile[] }) {
  if (tiles.length === 0) return null;
  return (
    <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {tiles.map(({ label, value, icon: Icon }) => (
        <div key={label} className="rounded-lg border bg-card/80 p-3.5 shadow-xs backdrop-blur-sm">
          <dt className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <Icon className="size-3.5 text-primary" aria-hidden="true" />
            {label}
          </dt>
          <dd className="mt-1 text-2xl font-semibold tracking-tight">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

// ---------------------------------------------------------------------------
// Topic gauges (a meter per topic: estimate dot on its likely-range band)
// ---------------------------------------------------------------------------

const R = 46;
const CX = 60;
const CY = 58;

function arcPoint(value: number): [number, number] {
  const a = Math.PI * (1 - Math.min(100, Math.max(0, value)) / 100);
  return [CX + R * Math.cos(a), CY - R * Math.sin(a)];
}

function arc(from: number, to: number): string {
  const [x1, y1] = arcPoint(from);
  const [x2, y2] = arcPoint(Math.max(to, from + 0.5));
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${R} ${R} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

export function TopicGauge({
  name,
  observed,
  questionsAsked,
  skillsMeasured,
}: {
  name: string;
  observed: AbilityDisplay | null;
  questionsAsked: number;
  skillsMeasured: number;
}) {
  const rated = observed && !observed.insufficientEvidence ? observed : null;
  const [low, high] = rated?.interval ?? [0, 0];
  const [mx, my] = arcPoint(rated?.display ?? 0);
  return (
    <li className="flex flex-col rounded-lg border bg-card p-4 shadow-xs">
      <h3 className="text-sm font-semibold leading-snug">{name}</h3>
      <p className="text-xs text-muted-foreground">
        {questionsAsked} question{questionsAsked === 1 ? '' : 's'}
        {skillsMeasured > 0 && (
          <>
            {' '}
            · {skillsMeasured} skill{skillsMeasured === 1 ? '' : 's'} measured
          </>
        )}
      </p>
      <div className="relative mx-auto mt-3 w-full max-w-[13rem]">
        <svg
          viewBox="0 0 120 66"
          className="w-full overflow-visible"
          {...(rated
            ? {
                role: 'img',
                'aria-label': `${name}: ${rated.display} out of 100, likely range ${low} to ${high}`,
              }
            : { 'aria-hidden': true })}
        >
          <path
            d={arc(0, 100)}
            fill="none"
            stroke="var(--border)"
            strokeWidth="9"
            strokeLinecap="round"
          />
          {rated && (
            <>
              <path
                d={arc(low, high)}
                fill="none"
                stroke="var(--viz-measured)"
                strokeOpacity="0.28"
                strokeWidth="9"
                strokeLinecap="round"
              />
              <circle
                cx={mx}
                cy={my}
                r="6"
                fill="var(--viz-measured)"
                stroke="var(--card)"
                strokeWidth="2.5"
              />
            </>
          )}
        </svg>
        <div className="absolute inset-x-0 bottom-0 text-center">
          {rated ? (
            <p className="text-3xl font-semibold leading-none tracking-tight">
              {rated.display}
              <span className="text-sm font-normal text-muted-foreground"> / 100</span>
            </p>
          ) : (
            <p
              className="text-2xl font-semibold leading-none text-muted-foreground"
              aria-hidden="true"
            >
              –
            </p>
          )}
        </div>
      </div>
      <div
        className="mt-2 flex justify-between px-1 text-[11px] text-muted-foreground"
        aria-hidden="true"
      >
        <span>0</span>
        <span>100</span>
      </div>
      <div className="mt-auto pt-2 text-center">
        {rated ? (
          <p className="text-xs text-muted-foreground">
            Likely range {low}–{high}
          </p>
        ) : (
          <Badge variant="neutral">{observed ? 'Not enough evidence yet' : 'Not assessed'}</Badge>
        )}
      </div>
    </li>
  );
}

// ---------------------------------------------------------------------------
// Skill range chart (estimate dot + likely-range band on one 0-100 axis)
// ---------------------------------------------------------------------------

export interface SkillPoint {
  key: string;
  name: string;
  topicName: string | null;
  observed: AbilityDisplay;
  status: SkillStatus;
}

const TICKS = [0, 25, 50, 75, 100];

function Gridlines() {
  return (
    <>
      {TICKS.map((t) => (
        <span
          key={t}
          aria-hidden="true"
          className="absolute inset-y-0 w-px bg-border"
          style={{ left: pct(t) }}
        />
      ))}
    </>
  );
}

export function SkillRangeChart({ skills }: { skills: SkillPoint[] }) {
  const [active, setActive] = React.useState<string | null>(null);
  if (skills.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No skill was measured precisely enough to place on the scale yet.
      </p>
    );
  }
  return (
    <div className="space-y-4">
      <ul className="space-y-1" aria-label="Skill estimates">
        {skills.map((s) => {
          const [low, high] = s.observed.interval;
          const st = STATUS[s.status];
          const Icon = st.icon;
          const label = `${s.name}: ${s.observed.display} out of 100, likely range ${low} to ${high}`;
          return (
            <li
              key={s.key}
              className="grid gap-1.5 rounded-md px-2 py-2 transition-colors hover:bg-muted/60 sm:grid-cols-[13rem_1fr] sm:items-center sm:gap-4"
              onMouseEnter={() => setActive(s.key)}
              onMouseLeave={() => setActive((k) => (k === s.key ? null : k))}
            >
              <div className="min-w-0">
                <p className="truncate text-sm font-medium" title={s.name}>
                  {s.name}
                </p>
                <p className="flex min-w-0 items-center gap-1 text-xs text-muted-foreground">
                  <Icon className="size-3.5 shrink-0" aria-hidden="true" />
                  <span className="shrink-0 whitespace-nowrap">{st.label}</span>
                  {s.topicName && <span className="min-w-0 truncate">· {s.topicName}</span>}
                </p>
              </div>
              <div className="flex items-center gap-3">
                <div className="relative h-7 flex-1" role="img" aria-label={label}>
                  <Gridlines />
                  <span
                    aria-hidden="true"
                    className={cn('absolute top-1/2 h-2.5 -translate-y-1/2 rounded-full', st.band)}
                    style={{ left: pct(low), width: pct(Math.max(high - low, 1)) }}
                  />
                  <span
                    aria-hidden="true"
                    className={cn(
                      'absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-card',
                      st.dot,
                    )}
                    style={{ left: pct(s.observed.display) }}
                  />
                  {active === s.key && (
                    <Tooltip at={s.observed.display}>
                      {s.observed.display} · likely {low}–{high} · {s.observed.observations} answer
                      {s.observed.observations === 1 ? '' : 's'}
                    </Tooltip>
                  )}
                </div>
                <span className="w-8 text-right text-sm font-semibold tabular-nums">
                  {s.observed.display}
                </span>
              </div>
            </li>
          );
        })}
      </ul>
      <div className="grid sm:grid-cols-[13rem_1fr] sm:gap-4 sm:px-2" aria-hidden="true">
        <span className="hidden sm:block" />
        <div className="relative mr-11 h-4 text-[11px] text-muted-foreground tabular-nums">
          {TICKS.map((t) => (
            <span
              key={t}
              className={cn(
                'absolute',
                t === 0 ? '' : t === 100 ? '-translate-x-full' : '-translate-x-1/2',
              )}
              style={{ left: pct(t) }}
            >
              {t}
            </span>
          ))}
        </div>
      </div>
      <StatusLegend />
      <details className="group rounded-md border">
        <summary className="flex cursor-pointer items-center gap-2 px-3 py-2 text-sm font-medium">
          <Table2 className="size-4 text-muted-foreground" aria-hidden="true" />
          Show as a table
        </summary>
        <div className="overflow-x-auto border-t">
          <table className="w-full text-left text-sm">
            <caption className="sr-only">Skill estimates</caption>
            <thead className="bg-muted/60 text-xs text-muted-foreground">
              <tr>
                <th scope="col" className="px-3 py-2 font-medium">
                  Skill
                </th>
                <th scope="col" className="px-3 py-2 font-medium">
                  Topic
                </th>
                <th scope="col" className="px-3 py-2 text-right font-medium">
                  Estimate
                </th>
                <th scope="col" className="px-3 py-2 font-medium">
                  Likely range
                </th>
                <th scope="col" className="px-3 py-2 text-right font-medium">
                  Answers
                </th>
                <th scope="col" className="px-3 py-2 font-medium">
                  Result
                </th>
              </tr>
            </thead>
            <tbody className="divide-y tabular-nums">
              {skills.map((s) => (
                <tr key={s.key}>
                  <th scope="row" className="px-3 py-2 font-medium">
                    {s.name}
                  </th>
                  <td className="px-3 py-2 text-muted-foreground">{s.topicName ?? ''}</td>
                  <td className="px-3 py-2 text-right">{s.observed.display}</td>
                  <td className="px-3 py-2">
                    {s.observed.interval[0]}–{s.observed.interval[1]}
                  </td>
                  <td className="px-3 py-2 text-right">{s.observed.observations}</td>
                  <td className="px-3 py-2">{STATUS[s.status].label}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Where you stand
// ---------------------------------------------------------------------------

const STANDING = {
  gap: {
    icon: Target,
    panel: 'border-viz-gap/30 bg-viz-gap/[0.06]',
    chip: 'bg-viz-gap',
  },
  strength: {
    icon: Award,
    panel: 'border-viz-strength/25 bg-viz-strength/[0.05]',
    chip: 'bg-viz-strength',
  },
  more: {
    icon: CircleHelp,
    panel: 'border-border bg-muted/50',
    chip: 'bg-muted-foreground/50',
  },
} as const;

export function StandingPanel({
  kind,
  title,
  hint,
  items,
  empty,
  ordered = false,
}: {
  kind: keyof typeof STANDING;
  title: string;
  hint: string;
  items: string[];
  empty: string;
  ordered?: boolean;
}) {
  const s = STANDING[kind];
  const Icon = s.icon;
  const List = ordered ? 'ol' : 'ul';
  return (
    <section className={cn('space-y-2 rounded-lg border p-4', s.panel)}>
      <h3 className="flex items-center gap-2 text-sm font-semibold">
        <span className="flex size-7 items-center justify-center rounded-md bg-card shadow-xs">
          <Icon className="size-4 text-foreground/70" aria-hidden="true" />
        </span>
        {title}
        <span
          aria-hidden="true"
          className="ml-auto rounded-full bg-card px-2 py-0.5 text-xs font-medium text-muted-foreground shadow-xs"
        >
          {items.length}
        </span>
      </h3>
      <p className="text-xs text-muted-foreground">{hint}</p>
      {items.length === 0 ? (
        <p className="text-sm text-muted-foreground">{empty}</p>
      ) : (
        <List className="space-y-1.5 text-sm">
          {items.map((item, i) => (
            <li key={`${item}-${i}`} className="flex items-start gap-2">
              {ordered ? (
                <span className="mt-0.5 w-4 shrink-0 text-right text-xs font-semibold text-muted-foreground tabular-nums">
                  {i + 1}.
                </span>
              ) : (
                <span
                  className={cn('mt-1.5 size-2 shrink-0 rounded-full', s.chip)}
                  aria-hidden="true"
                />
              )}
              <span>{item}</span>
            </li>
          ))}
        </List>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Answer grades: a column chart (answer + its follow-ups) and its list twin
// ---------------------------------------------------------------------------

function columnLabel(q: QuestionResult): string {
  const main = q.score === null ? 'not graded' : `${scoreOutOf100(q.score)} out of 100`;
  const follow = q.followUpScores.length
    ? `; follow-ups ${q.followUpScores
        .map((s) => (s === null ? 'not graded' : `${scoreOutOf100(s)}`))
        .join(', ')}`
    : '';
  return `Question ${q.order}: ${main}${follow}`;
}

function Column({ score, follow }: { score: number | null; follow?: boolean }) {
  if (score === null) {
    return (
      <span
        className="h-1 w-2.5 self-end rounded-sm bg-muted-foreground/30 sm:w-3.5"
        aria-hidden="true"
      />
    );
  }
  return (
    <span
      aria-hidden="true"
      className={cn(
        'w-2.5 self-end rounded-t-[4px] sm:w-3.5',
        follow ? 'bg-viz-measured/40' : 'bg-viz-measured',
      )}
      style={{ height: `max(${scoreOutOf100(score)}%, 2px)` }}
    />
  );
}

export function AnswerChart({ questions }: { questions: QuestionResult[] }) {
  const [active, setActive] = React.useState<number | null>(null);
  const hasFollowUps = questions.some((q) => q.followUpScores.length > 0);
  return (
    <figure className="space-y-3">
      <div className="overflow-x-auto pb-1">
        <div className="min-w-[18rem]">
          <div className="relative ml-8 mt-5 h-44">
            {[0, 50, 100].map((t) => (
              <React.Fragment key={t}>
                <span
                  aria-hidden="true"
                  className="absolute inset-x-0 h-px bg-border"
                  style={{ bottom: `${t}%` }}
                />
                <span
                  aria-hidden="true"
                  className="absolute -left-8 w-6 translate-y-1/2 text-right text-[11px] text-muted-foreground tabular-nums"
                  style={{ bottom: `${t}%` }}
                >
                  {t}
                </span>
              </React.Fragment>
            ))}
            <ol
              className="absolute inset-0 flex items-end justify-around gap-1 px-1"
              aria-label="Answer grades chart"
            >
              {questions.map((q, i) => {
                const at = ((i + 0.5) / questions.length) * 100;
                return (
                  <li
                    key={q.order}
                    className="relative flex h-full min-w-0 flex-1 justify-center"
                    aria-label={columnLabel(q)}
                    onMouseEnter={() => setActive(q.order)}
                    onMouseLeave={() => setActive((o) => (o === q.order ? null : o))}
                  >
                    <div className="relative flex h-full items-end gap-[2px]">
                      <div className="relative flex h-full items-end">
                        <Column score={q.score} />
                        {q.score !== null && (
                          <span
                            aria-hidden="true"
                            className="absolute left-1/2 -translate-x-1/2 pb-0.5 text-[11px] font-medium tabular-nums"
                            style={{ bottom: `${scoreOutOf100(q.score)}%` }}
                          >
                            {scoreOutOf100(q.score)}
                          </span>
                        )}
                      </div>
                      {q.followUpScores.map((s, j) => (
                        <Column key={j} score={s} follow />
                      ))}
                    </div>
                    {active === q.order && (
                      <span
                        role="presentation"
                        className={cn(
                          'pointer-events-none absolute bottom-full z-10 mb-1 whitespace-nowrap rounded-md bg-foreground px-2 py-1 text-xs text-background shadow-md',
                          edgeShift(at),
                          at < 12 ? 'left-0' : at > 88 ? 'right-0 translate-x-0' : 'left-1/2',
                        )}
                      >
                        {columnLabel(q)}
                      </span>
                    )}
                  </li>
                );
              })}
            </ol>
          </div>
          <div className="ml-8 flex justify-around gap-1 px-1 pt-1.5" aria-hidden="true">
            {questions.map((q) => (
              <span key={q.order} className="flex-1 text-center text-[11px] text-muted-foreground">
                Q{q.order}
              </span>
            ))}
          </div>
        </div>
      </div>
      <figcaption className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="h-3 w-2.5 rounded-t-[3px] bg-viz-measured" aria-hidden="true" />
          Answer grade (0–100)
        </span>
        {hasFollowUps && (
          <span className="flex items-center gap-1.5">
            <span className="h-3 w-2.5 rounded-t-[3px] bg-viz-measured/40" aria-hidden="true" />
            Follow-up grades
          </span>
        )}
        <span className="flex items-center gap-1.5">
          <span className="h-1 w-2.5 rounded-sm bg-muted-foreground/30" aria-hidden="true" />
          Not graded
        </span>
      </figcaption>
    </figure>
  );
}

/** A thin meter for one grade, same hue as the chart. */
export function GradeMeter({ score }: { score: number }) {
  return (
    <span
      className="block h-1.5 w-24 overflow-hidden rounded-full bg-viz-measured/15"
      aria-hidden="true"
    >
      <span
        className="block h-full rounded-full bg-viz-measured"
        style={{ width: `${scoreOutOf100(score)}%` }}
      />
    </span>
  );
}
