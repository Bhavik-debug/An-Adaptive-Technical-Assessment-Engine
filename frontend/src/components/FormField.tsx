import * as React from 'react';

import { Label } from '@/components/ui/label';
import { cn } from '@/lib/utils';

export interface FieldA11yProps {
  id: string;
  'aria-invalid': boolean | undefined;
  'aria-describedby': string | undefined;
}

/**
 * Label + control + hint + error, wired for assistive technology: the control
 * gets `aria-invalid` and `aria-describedby` pointing at the hint and error.
 */
export function FormField({
  id,
  label,
  hint,
  error,
  optional,
  className,
  children,
}: {
  id: string;
  label: React.ReactNode;
  hint?: React.ReactNode;
  error?: string | null;
  optional?: boolean;
  className?: string;
  children: (a11y: FieldA11yProps) => React.ReactNode;
}) {
  const hintId = hint ? `${id}-hint` : undefined;
  const errorId = error ? `${id}-error` : undefined;
  const describedBy = [hintId, errorId].filter(Boolean).join(' ') || undefined;
  return (
    <div className={cn('space-y-1.5', className)}>
      <Label htmlFor={id}>
        {label}
        {optional && <span className="ml-1 font-normal text-muted-foreground">(optional)</span>}
      </Label>
      {children({ id, 'aria-invalid': error ? true : undefined, 'aria-describedby': describedBy })}
      {hint && (
        <p id={hintId} className="text-xs text-muted-foreground">
          {hint}
        </p>
      )}
      {error && (
        <p id={errorId} className="text-xs font-medium text-destructive">
          {error}
        </p>
      )}
    </div>
  );
}
