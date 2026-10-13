import { MAX_PASSWORD_BYTES, MIN_PASSWORD_LENGTH } from '@/api/auth';

// Deliberately loose: the server's EmailStr is the authority. This only
// catches obvious typos before a round trip.
const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export function validateEmail(value: string): string | null {
  const email = value.trim();
  if (!email) return 'Enter your email address.';
  if (!EMAIL_PATTERN.test(email)) return 'Enter a valid email address, like name@example.com.';
  return null;
}

export function validateNewPassword(value: string): string | null {
  if (!value) return 'Choose a password.';
  if (value.length < MIN_PASSWORD_LENGTH) return `Use at least ${MIN_PASSWORD_LENGTH} characters.`;
  if (new TextEncoder().encode(value).length > MAX_PASSWORD_BYTES)
    return 'That password is too long.';
  return null;
}

export function validateLoginPassword(value: string): string | null {
  if (!value) return 'Enter your password.';
  if (new TextEncoder().encode(value).length > MAX_PASSWORD_BYTES)
    return 'That password is too long.';
  return null;
}

export type FieldErrors<K extends string> = Partial<Record<K, string>>;

export function hasErrors(errors: Record<string, string | null | undefined>): boolean {
  return Object.values(errors).some(Boolean);
}
