import { Construction } from 'lucide-react';
import { Link, useLocation, useParams } from 'react-router';

import { Alert } from '@/components/ui/alert';
import { buttonVariants } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

function NotBuiltYet({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 rounded-md border border-dashed p-4 text-sm text-muted-foreground">
      <Construction className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
      <div>{children}</div>
    </div>
  );
}

/**
 * Hand-off target after the wizard starts a session. The interview screen
 * itself is Day 27; this page says only what the backend has confirmed.
 */
export function InterviewPlaceholderPage() {
  const { sessionId } = useParams();
  const location = useLocation();
  const started = (location.state as { started?: boolean } | null)?.started === true;
  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <Card>
        <CardHeader>
          <CardTitle as="h1" className="text-xl">
            Interview
          </CardTitle>
          <CardDescription>Session {sessionId}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {started && (
            <Alert variant="success" title="Your interview has started">
              The server confirmed the session is running and your first question is ready.
            </Alert>
          )}
          <NotBuiltYet>
            The interview screen is not available in this build yet. Your session is saved and
            appears on your dashboard.
          </NotBuiltYet>
          <Link to="/dashboard" className={buttonVariants({ variant: 'outline' })}>
            Back to dashboard
          </Link>
        </CardContent>
      </Card>
    </div>
  );
}

/** The report screen is Day 28. */
export function ReportPlaceholderPage() {
  const { sessionId } = useParams();
  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <Card>
        <CardHeader>
          <CardTitle as="h1" className="text-xl">
            Interview report
          </CardTitle>
          <CardDescription>Session {sessionId}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <NotBuiltYet>
            The full report view is not available in this build yet. A summary of your latest
            completed interview is shown on the dashboard.
          </NotBuiltYet>
          <Link to="/dashboard" className={buttonVariants({ variant: 'outline' })}>
            Back to dashboard
          </Link>
        </CardContent>
      </Card>
    </div>
  );
}

export function NotFoundPage() {
  return (
    <div className="flex min-h-svh flex-col items-center justify-center gap-4 px-4 text-center">
      <h1 className="text-2xl font-semibold">Page not found</h1>
      <p className="text-sm text-muted-foreground">
        The page you are looking for does not exist or has moved.
      </p>
      <Link to="/dashboard" className={buttonVariants()}>
        Go to dashboard
      </Link>
    </div>
  );
}
