/** Pure helpers for the interview screen's question and answer text. */

import { MAX_ANSWER_CHARS } from '@/api/sessions';

/** A submission that has not answered within this is treated as "outcome unknown". */
export const SUBMIT_TIMEOUT_MS = 30_000;

/**
 * Characters as the backend counts them (Python `len`: code points), not
 * UTF-16 units - an answer with emoji must not be refused early.
 */
export function answerLength(text: string): number {
  return Array.from(text).length;
}

export function validateAnswer(text: string): string | null {
  if (!text.trim()) return 'Write an answer before submitting.';
  if (answerLength(text) > MAX_ANSWER_CHARS)
    return `Keep your answer under ${MAX_ANSWER_CHARS.toLocaleString()} characters.`;
  return null;
}

export interface Segment {
  code: boolean;
  text: string;
}

export function splitFences(text: string): Segment[] {
  const segments: Segment[] = [];
  const fence = /```[^\n`]*\n([\s\S]*?)```/g;
  let last = 0;
  let match: RegExpExecArray | null;
  while ((match = fence.exec(text)) !== null) {
    if (match.index > last) segments.push({ code: false, text: text.slice(last, match.index) });
    segments.push({ code: true, text: (match[1] ?? '').replace(/\n$/, '') });
    last = match.index + match[0].length;
  }
  if (last < text.length) segments.push({ code: false, text: text.slice(last) });
  return segments.filter((s) => s.code || s.text.trim() !== '');
}
