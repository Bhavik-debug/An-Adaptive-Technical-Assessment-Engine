import { Loader2 } from 'lucide-react';

export function FullPageSpinner({ label }: { label: string }) {
  return (
    <div className="flex min-h-svh items-center justify-center" role="status" aria-live="polite">
      <Loader2 className="size-6 animate-spin text-primary" aria-hidden="true" />
      <span className="sr-only">{label}</span>
    </div>
  );
}
