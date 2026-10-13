import { cva, type VariantProps } from 'class-variance-authority';
import { AlertCircle, CheckCircle2, Info, TriangleAlert } from 'lucide-react';
import * as React from 'react';

import { cn } from '@/lib/utils';

const alertVariants = cva(
  'flex gap-3 rounded-md border p-3.5 text-sm [&>svg]:mt-0.5 [&>svg]:size-4 [&>svg]:shrink-0',
  {
    variants: {
      variant: {
        info: 'border-primary/20 bg-accent [&>svg]:text-primary',
        success: 'border-success/25 bg-success/5 [&>svg]:text-success',
        warning: 'border-warning/30 bg-warning/5 [&>svg]:text-warning',
        error: 'border-destructive/30 bg-destructive/5 [&>svg]:text-destructive',
      },
    },
    defaultVariants: { variant: 'info' },
  },
);

const icons = { info: Info, success: CheckCircle2, warning: TriangleAlert, error: AlertCircle };

export interface AlertProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, 'title'>, VariantProps<typeof alertVariants> {
  title?: React.ReactNode;
}

/** Errors are announced assertively (role=alert); everything else politely. */
export function Alert({ className, variant, title, children, ...props }: AlertProps) {
  const Icon = icons[variant ?? 'info'];
  return (
    <div
      role={variant === 'error' ? 'alert' : 'status'}
      className={cn(alertVariants({ variant }), className)}
      {...props}
    >
      <Icon aria-hidden="true" />
      <div className="min-w-0 space-y-1 text-foreground">
        {title && <p className="font-medium">{title}</p>}
        {children && <div className="text-foreground/80">{children}</div>}
      </div>
    </div>
  );
}
