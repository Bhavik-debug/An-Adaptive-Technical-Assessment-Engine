import { Check } from 'lucide-react';

import { cn } from '@/lib/utils';

export function Stepper({ steps, current }: { steps: string[]; current: number }) {
  return (
    <nav aria-label="Interview setup progress">
      <ol className="flex flex-wrap items-center gap-x-2 gap-y-2 text-sm">
        {steps.map((label, index) => {
          const done = index < current;
          const active = index === current;
          return (
            <li
              key={label}
              className="flex items-center gap-2"
              aria-current={active ? 'step' : undefined}
            >
              <span
                className={cn(
                  'flex size-7 items-center justify-center rounded-full border text-xs font-semibold',
                  done && 'border-primary bg-primary text-primary-foreground',
                  active && 'border-primary text-primary',
                  !done && !active && 'text-muted-foreground',
                )}
                aria-hidden="true"
              >
                {done ? <Check className="size-3.5" /> : index + 1}
              </span>
              <span className={cn(active ? 'font-medium' : 'text-muted-foreground')}>
                <span className="sr-only">
                  Step {index + 1} of {steps.length}
                  {done ? ', completed' : ''}:{' '}
                </span>
                {label}
              </span>
              {index < steps.length - 1 && (
                <span aria-hidden="true" className="mx-1 hidden h-px w-8 bg-border sm:block" />
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
