import * as React from 'react';

import { cn } from '@/lib/utils';

import { fieldClasses } from './input';

/** A styled native <select>: keyboard and screen-reader behaviour for free. */
export const Select = React.forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(({ className, children, ...props }, ref) => (
  <select ref={ref} className={cn(fieldClasses, 'h-10 py-2', className)} {...props}>
    {children}
  </select>
));
Select.displayName = 'Select';
