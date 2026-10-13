import { Plus } from 'lucide-react';
import { Link } from 'react-router';

import { listSessions } from '@/api/sessions';
import { useAuth } from '@/auth/AuthProvider';
import { buttonVariants } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useApiData } from '@/hooks/useApiData';
import { useSetupOptions } from '@/hooks/useSetupOptions';

import { SessionHistory } from './SessionHistory';
import { SkillOverview } from './SkillOverview';

export function DashboardPage() {
  const { user } = useAuth();
  const sessions = useApiData((signal) => listSessions(signal), []);
  // Only for display names; the dashboard still works if this fails.
  const { topicNames } = useSetupOptions();

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="flex flex-col gap-4 pt-5 sm:flex-row sm:items-center sm:justify-between sm:pt-6">
          <div className="space-y-1">
            <h1 className="text-2xl font-semibold tracking-tight">Welcome</h1>
            <p className="text-sm text-muted-foreground">
              {user?.email ? `Signed in as ${user.email}. ` : ''}Set up a practice interview
              tailored to your resume and the role you are targeting.
            </p>
          </div>
          <Link to="/interviews/new" className={buttonVariants({ size: 'lg' })}>
            <Plus aria-hidden="true" />
            Start an interview
          </Link>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <SessionHistory
            sessions={sessions.data}
            loading={sessions.loading}
            error={sessions.error}
            onRetry={sessions.reload}
            topicNames={topicNames}
          />
        </div>
        <div className="lg:col-span-2">
          <SkillOverview
            sessions={sessions.error ? [] : sessions.data}
            loading={sessions.loading}
            topicNames={topicNames}
          />
        </div>
      </div>
    </div>
  );
}
