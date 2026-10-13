import { FileText, Upload } from 'lucide-react';
import * as React from 'react';

import { ApiError } from '@/api/client';
import {
  RESUME_MAX_BYTES,
  confirmProfile,
  getResume,
  resumeKind,
  uploadResume,
} from '@/api/ingestion';
import type { Resume, TopicOption } from '@/api/types';
import { Alert } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { usePolling } from '@/hooks/usePolling';
import { formatBytes } from '@/lib/format';

import { initialProfile } from '@/lib/setup';

import { ProfileEditor } from './ProfileEditor';

const SETTLED = new Set(['ready', 'failed']);

function uploadErrorMessage(error: unknown): string {
  if (!(error instanceof ApiError)) return 'The upload failed. Please try again.';
  if (error.code === 'file_too_large' || error.status === 413)
    return `That file is larger than ${formatBytes(RESUME_MAX_BYTES)}. Please upload a smaller file.`;
  if (error.code === 'unsupported_type' || error.status === 415)
    return 'That file type is not supported. Upload a PDF or a plain-text (.txt) file.';
  return error.message;
}

export function ResumeStep({
  resume,
  onResumeChange,
  topics,
  onContinue,
  onSkip,
}: {
  resume: Resume | null;
  onResumeChange: (resume: Resume | null) => void;
  topics: TopicOption[];
  onContinue: () => void;
  onSkip: () => void;
}) {
  const inputRef = React.useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = React.useState<string | null>(null);
  const [fileError, setFileError] = React.useState<string | null>(null);
  const [uploading, setUploading] = React.useState(false);

  const resumeId = resume?.resume_id ?? null;
  const settled = resume ? SETTLED.has(resume.status) : false;
  const poll = usePolling<Resume>(resume && !settled ? resumeId : null, {
    fetcher: (signal) => getResume(resumeId as string, signal),
    isDone: (r) => SETTLED.has(r.status),
  });

  React.useEffect(() => {
    if (poll.value) onResumeChange(poll.value);
  }, [poll.value, onResumeChange]);

  async function handleFile(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = ''; // allow re-selecting the same file after an error
    if (!file || uploading) return;
    setFileError(null);
    const kind = resumeKind(file);
    if (!kind) {
      setFileError('Upload a PDF or a plain-text (.txt) file.');
      return;
    }
    if (file.size === 0) {
      setFileError('That file is empty.');
      return;
    }
    if (file.size > RESUME_MAX_BYTES) {
      setFileError(
        `That file is ${formatBytes(file.size)}. The limit is ${formatBytes(RESUME_MAX_BYTES)}.`,
      );
      return;
    }
    setUploading(true);
    setFileName(file.name);
    try {
      onResumeChange(await uploadResume(file, kind));
    } catch (error) {
      setFileError(uploadErrorMessage(error));
      setFileName(null);
    } finally {
      setUploading(false);
    }
  }

  const handleConfirm = React.useCallback(
    async (profile: Parameters<typeof confirmProfile>[1]) => {
      if (!resumeId) return;
      onResumeChange(await confirmProfile(resumeId, profile));
    },
    [resumeId, onResumeChange],
  );

  const processing = Boolean(
    resume && !settled && poll.phase !== 'timeout' && poll.phase !== 'error',
  );
  const showEditor = Boolean(resume && (settled || poll.phase === 'timeout'));

  return (
    <div className="space-y-6">
      <section aria-labelledby="resume-upload-heading" className="space-y-3">
        <h3 id="resume-upload-heading" className="font-medium">
          Upload your resume
        </h3>
        <p className="text-sm text-muted-foreground">
          PDF or plain text, up to {formatBytes(RESUME_MAX_BYTES)}. Contact details are removed
          before anything is stored. You can review everything we extract.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <input
            ref={inputRef}
            id="resume-file"
            type="file"
            accept=".pdf,.txt,application/pdf,text/plain"
            className="sr-only"
            onChange={handleFile}
            disabled={uploading}
            aria-describedby={fileError ? 'resume-file-error' : undefined}
          />
          <Button
            variant="outline"
            onClick={() => inputRef.current?.click()}
            loading={uploading}
            aria-controls="resume-file"
          >
            {!uploading && <Upload aria-hidden="true" />}
            {uploading ? 'Uploading…' : resume ? 'Upload a different file' : 'Choose file'}
          </Button>
          {(fileName || resume) && !fileError && (
            <span className="inline-flex items-center gap-1.5 text-sm text-muted-foreground">
              <FileText className="size-4" aria-hidden="true" />
              {fileName ?? 'Resume uploaded'}
            </span>
          )}
        </div>
        {fileError && (
          <p id="resume-file-error" role="alert" className="text-sm font-medium text-destructive">
            {fileError}
          </p>
        )}
      </section>

      {processing && (
        <Alert variant="info" title="Reading your resume…">
          This usually takes a few seconds. You can wait, or continue without waiting.
        </Alert>
      )}
      {poll.phase === 'timeout' && (
        <Alert variant="warning" title="Still processing">
          <p>
            Extraction is taking longer than expected. You can check again, enter your skills
            yourself below, or continue without them.
          </p>
          <Button variant="outline" size="sm" className="mt-2" onClick={poll.restart}>
            Check again
          </Button>
        </Alert>
      )}
      {poll.phase === 'error' && poll.error && (
        <Alert variant="error" title="We could not check your resume's status">
          <p>{poll.error.message}</p>
          <Button variant="outline" size="sm" className="mt-2" onClick={poll.restart}>
            Try again
          </Button>
        </Alert>
      )}
      {resume?.status === 'failed' && (
        <Alert variant="warning" title="We could not extract a profile automatically">
          You can add your skills yourself below, or continue without them.
          {resume.error && <span className="block text-xs">Reason code: {resume.error}</span>}
        </Alert>
      )}
      {resume?.truncated && (
        <Alert variant="info" title="Only part of your resume was read">
          Your resume was longer than the processing limit, so only the first part was used.
        </Alert>
      )}

      {showEditor && resume && (
        <section aria-labelledby="profile-heading" className="space-y-3 border-t pt-6">
          <div className="space-y-1">
            <h3 id="profile-heading" className="font-medium">
              Review your profile
            </h3>
            <p className="text-sm text-muted-foreground">
              Check what we found and correct anything that is wrong. Only a profile you confirm is
              used, and only as a starting point — your answers decide your results.
            </p>
          </div>
          <ProfileEditor
            key={resume.resume_id}
            initial={initialProfile(resume.profile, resume.suggestion)}
            topics={topics}
            confirmed={resume.profile_confirmed}
            onConfirm={handleConfirm}
          />
        </section>
      )}

      <div className="flex flex-col-reverse gap-3 border-t pt-6 sm:flex-row sm:justify-between">
        <Button variant="ghost" onClick={onSkip} disabled={uploading}>
          Skip — continue without a resume
        </Button>
        <div className="flex flex-col items-stretch gap-2 sm:items-end">
          <Button onClick={onContinue} disabled={!resume || uploading}>
            Continue
          </Button>
          {resume && !resume.profile_confirmed && (
            <p className="text-xs text-muted-foreground">
              Your profile is not confirmed yet, so it will not be used.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
