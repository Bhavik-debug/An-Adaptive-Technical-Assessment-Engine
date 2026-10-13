import * as React from 'react';

import { cn } from '@/lib/utils';

/** Shared by Input, Textarea and Select so every field looks and focuses alike. */
export const fieldClasses =
  'w-full rounded-md border border-input bg-card px-3 text-sm text-foreground shadow-xs placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:border-ring disabled:cursor-not-allowed disabled:opacity-60 aria-[invalid=true]:border-destructive aria-[invalid=true]:focus-visible:ring-destructive';

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, type, ...props }, ref) => (
  <input ref={ref} type={type} className={cn(fieldClasses, 'h-10 py-2', className)} {...props} />
));
Input.displayName = 'Input';
