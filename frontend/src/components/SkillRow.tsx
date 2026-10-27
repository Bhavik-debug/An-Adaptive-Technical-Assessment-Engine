import { Badge } from '@/components/ui/badge';

export interface SkillEstimate {
  display: number;
  interval: [number, number];
  insufficient_evidence: boolean;
}

/**
 * One skill on the 0-100 display scale with its likely range (plan §9.3:
 * "always render the interval"). Too-uncertain or unmeasured skills get a
 * label instead of a number, never a bar.
 */
export function SkillRow({
  name,
  observed,
  detail,
}: {
  name: string;
  observed: SkillEstimate | null;
  /** Extra context under the bar, e.g. "4 questions". */
  detail?: string;
}) {
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
        Likely range {low}–{high}
        {detail && <> · {detail}</>}
      </p>
    </li>
  );
}
