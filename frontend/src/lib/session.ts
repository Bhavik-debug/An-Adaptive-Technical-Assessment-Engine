import type { BadgeProps } from '@/components/ui/badge';
import type { SessionSummary } from '@/api/types';

type Variant = NonNullable<BadgeProps['variant']>;

/**
 * Candidate-facing session status. The FSM's internal states (grading,
 * probing, selecting) are collapsed into "In progress": they describe how the
 * engine works, not anything the candidate needs to act on.
 */
export function sessionStatus(state: string): { label: string; variant: Variant } {
  switch (state) {
    case 'CREATED':
      return { label: 'Not started', variant: 'neutral' };
    case 'PREPARING':
    case 'ASKING':
    case 'GRADING':
    case 'PROBING':
    case 'SELECTING':
      return { label: 'In progress', variant: 'info' };
    case 'REPORTING':
      return { label: 'Preparing report', variant: 'warning' };
    case 'COMPLETED':
      return { label: 'Completed', variant: 'success' };
    default:
      return { label: 'Unknown', variant: 'neutral' };
  }
}

/** The newest session whose persisted report is ready, if any. */
export function latestReportedSession(sessions: SessionSummary[] | undefined) {
  return sessions?.find((s) => s.state === 'COMPLETED' && s.report_status === 'ready');
}
