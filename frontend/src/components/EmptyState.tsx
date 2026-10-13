import type { LucideIcon } from 'lucide-react';
import * as React from 'react';

export function EmptyState({
  icon: Icon,
  title,
  children,
  action,
}: {
  icon: LucideIcon;
  title: string;
  children?: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center rounded-md border border-dashed px-6 py-10 text-center">
      <Icon className="mb-3 size-8 text-muted-foreground" aria-hidden="true" />
      <p className="font-medium">{title}</p>
      {children && <div className="mt-1 max-w-md text-sm text-muted-foreground">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
