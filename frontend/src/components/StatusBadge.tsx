import { Badge } from '@/components/ui/badge';
import { sessionStatus } from '@/lib/session';

export function SessionStatusBadge({ state }: { state: string }) {
  const { label, variant } = sessionStatus(state);
  return <Badge variant={variant}>{label}</Badge>;
}
