import * as React from 'react';

import { cn } from '@/lib/utils';

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('rounded-lg border bg-card text-card-foreground shadow-xs', className)}
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('flex flex-col gap-1.5 p-5 sm:p-6', className)} {...props} />;
}

export function CardTitle({
  className,
  as: Tag = 'h2',
  ...props
}: React.HTMLAttributes<HTMLHeadingElement> & { as?: 'h1' | 'h2' | 'h3' }) {
  return <Tag className={cn('text-lg font-semibold leading-tight', className)} {...props} />;
}

export function CardDescription({
  className,
  ...props
}: React.HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn('text-sm text-muted-foreground', className)} {...props} />;
}

export function CardContent({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('p-5 pt-0 sm:p-6 sm:pt-0', className)} {...props} />;
}

export function CardFooter({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn('flex flex-wrap items-center gap-3 p-5 pt-0 sm:p-6 sm:pt-0', className)}
      {...props}
    />
  );
}
