import * as React from 'react';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

import { BrandMark } from './BrandMark';

export function AuthLayout({
  title,
  description,
  children,
  footer,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
}) {
  return (
    <div className="flex min-h-svh flex-col items-center justify-center px-4 py-10">
      <div className="mb-6">
        <BrandMark to="/login" />
      </div>
      <main className="w-full max-w-md">
        <Card>
          <CardHeader>
            <CardTitle as="h1" className="text-xl">
              {title}
            </CardTitle>
            <CardDescription>{description}</CardDescription>
          </CardHeader>
          <CardContent>{children}</CardContent>
        </Card>
        {footer && <div className="mt-4 text-center text-sm text-muted-foreground">{footer}</div>}
      </main>
    </div>
  );
}
