import * as React from 'react';

import { cn } from '@/lib/utils';

export const Label = React.forwardRef<
  HTMLLabelElement,
  React.LabelHTMLAttributes<HTMLLabelElement>
>(({ className, ...props }, ref) => (
  // eslint-disable-next-line jsx-a11y/label-has-associated-control -- callers pass htmlFor
  <label
    ref={ref}
    className={cn('text-sm font-medium leading-none text-foreground', className)}
    {...props}
  />
));
Label.displayName = 'Label';
