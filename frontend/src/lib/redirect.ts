/** Only internal paths are honoured as post-login destinations (no open redirect). */
export function safeRedirectPath(value: unknown, fallback = '/dashboard'): string {
  if (typeof value !== 'string') return fallback;
  if (!value.startsWith('/') || value.startsWith('//') || value.includes('\\')) return fallback;
  if (value.startsWith('/login') || value.startsWith('/register')) return fallback;
  return value;
}
