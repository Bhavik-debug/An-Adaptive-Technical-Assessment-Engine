import { apiRequest } from './client';
import type { CandidateProfile, JobDescription, Resume } from './types';

/** Backend defaults (backend/app/config.py). The server re-checks; these only give early feedback. */
export const RESUME_MAX_BYTES = 5 * 1024 * 1024;
export const JD_MIN_CHARS = 40;
export const JD_MAX_CHARS = 20_000;

export type ResumeKind = 'pdf' | 'text';

/** The upload's Content-Type. The server sniffs the bytes and does not trust this. */
export function resumeKind(file: File): ResumeKind | null {
  const name = file.name.toLowerCase();
  if (file.type === 'application/pdf' || name.endsWith('.pdf')) return 'pdf';
  if (file.type === 'text/plain' || name.endsWith('.txt')) return 'text';
  return null;
}

/** POST /api/resumes -> 202 ResumeOut. The file is the raw body (not multipart). */
export function uploadResume(file: File, kind: ResumeKind): Promise<Resume> {
  return apiRequest<Resume>('/api/resumes', {
    method: 'POST',
    body: file,
    headers: { 'Content-Type': kind === 'pdf' ? 'application/pdf' : 'text/plain' },
  });
}

/** GET /api/resumes/{id} */
export function getResume(resumeId: string, signal?: AbortSignal): Promise<Resume> {
  return apiRequest<Resume>(`/api/resumes/${encodeURIComponent(resumeId)}`, { signal });
}

/** PATCH /api/resumes/{id}/profile -> replaces the confirmed profile as a whole. */
export function confirmProfile(resumeId: string, profile: CandidateProfile): Promise<Resume> {
  return apiRequest<Resume>(`/api/resumes/${encodeURIComponent(resumeId)}/profile`, {
    method: 'PATCH',
    json: profile,
  });
}

/** POST /api/job-descriptions -> 202 JobDescriptionOut (extraction runs in the background). */
export function submitJobDescription(text: string): Promise<JobDescription> {
  return apiRequest<JobDescription>('/api/job-descriptions', { method: 'POST', json: { text } });
}

/** GET /api/job-descriptions/{id} */
export function getJobDescription(jdId: string, signal?: AbortSignal): Promise<JobDescription> {
  return apiRequest<JobDescription>(`/api/job-descriptions/${encodeURIComponent(jdId)}`, {
    signal,
  });
}
