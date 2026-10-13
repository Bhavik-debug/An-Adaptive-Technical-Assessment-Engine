import { RefreshCw } from 'lucide-react';
import * as React from 'react';

import type { JobDescription, Resume } from '@/api/types';
import { Alert } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useSetupOptions } from '@/hooks/useSetupOptions';

import { ConfigStep } from './ConfigStep';
import { JobDescriptionStep } from './JobDescriptionStep';
import { ResumeStep } from './ResumeStep';
import { Stepper } from './Stepper';

const STEPS = ['Resume', 'Job description', 'Interview settings'];
const STEP_DESCRIPTIONS = [
  'Optional. Upload your resume so the interview can start from your background.',
  'Optional. Paste the job description so the interview focuses on what the role needs.',
  'Choose how the interview should run, then start.',
];

export function SetupWizardPage() {
  const setup = useSetupOptions();
  const [step, setStep] = React.useState(0);
  // The IDs (and latest server view) carried between steps.
  const [resume, setResume] = React.useState<Resume | null>(null);
  const [jd, setJd] = React.useState<JobDescription | null>(null);
  const [jdText, setJdText] = React.useState('');
  const headingRef = React.useRef<HTMLHeadingElement>(null);
  const firstRender = React.useRef(true);

  // Move focus to the new step's heading so keyboard and screen-reader users
  // land at the start of the step rather than on a now-missing button.
  React.useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    headingRef.current?.focus();
  }, [step]);

  const go = (next: number) => () => setStep(next);

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="space-y-3">
        <h1 className="text-2xl font-semibold tracking-tight">Set up your interview</h1>
        <Stepper steps={STEPS} current={step} />
      </div>
      <Card>
        <CardHeader>
          <CardTitle>
            <span ref={headingRef} tabIndex={-1} className="focus-visible:outline-none">
              {STEPS[step]}
            </span>
          </CardTitle>
          <CardDescription>{STEP_DESCRIPTIONS[step]}</CardDescription>
        </CardHeader>
        <CardContent>
          {setup.loading && !setup.data ? (
            <div className="space-y-3" aria-busy="true">
              <span className="sr-only" role="status">
                Loading setup options
              </span>
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-24 w-full" />
            </div>
          ) : setup.error || !setup.data ? (
            <Alert variant="error" title="We could not load the setup options">
              <p>{setup.error?.message ?? 'Please try again.'}</p>
              <Button variant="outline" size="sm" className="mt-2" onClick={setup.reload}>
                <RefreshCw aria-hidden="true" />
                Try again
              </Button>
            </Alert>
          ) : step === 0 ? (
            <ResumeStep
              resume={resume}
              onResumeChange={setResume}
              topics={setup.data.topics}
              onContinue={go(1)}
              onSkip={() => {
                setResume(null);
                setStep(1);
              }}
            />
          ) : step === 1 ? (
            <JobDescriptionStep
              jd={jd}
              onJdChange={setJd}
              text={jdText}
              onTextChange={setJdText}
              topicNames={setup.topicNames}
              onBack={go(0)}
              onContinue={go(2)}
              onSkip={() => {
                setJd(null);
                setStep(2);
              }}
            />
          ) : (
            <ConfigStep
              options={setup.data}
              resume={resume}
              jd={jd}
              topicNames={setup.topicNames}
              onBack={go(1)}
            />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
