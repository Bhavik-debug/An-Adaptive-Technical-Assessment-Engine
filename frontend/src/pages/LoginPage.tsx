import * as React from 'react';
import { Link, useLocation, useNavigate } from 'react-router';

import { ApiError } from '@/api/client';
import { useAuth } from '@/auth/AuthProvider';
import type { LoginRedirectState } from '@/auth/guards';
import { safeRedirectPath } from '@/lib/redirect';
import { AuthLayout } from '@/components/AuthLayout';
import { FormField } from '@/components/FormField';
import { GoogleSignIn } from '@/components/GoogleSignIn';
import { Alert } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { validateEmail, validateLoginPassword, type FieldErrors } from '@/lib/validation';

type Field = 'email' | 'password';

export function LoginPage() {
  const { signIn, sessionExpired, restoreError, retryRestore } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as LoginRedirectState | null)?.from;

  const [email, setEmail] = React.useState('');
  const [password, setPassword] = React.useState('');
  const [errors, setErrors] = React.useState<FieldErrors<Field>>({});
  const [formError, setFormError] = React.useState<string | null>(null);
  const [submitting, setSubmitting] = React.useState(false);
  // Google's exchange is in flight: the email form waits for it.
  const [googleBusy, setGoogleBusy] = React.useState(false);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitting || googleBusy) return;
    const next: FieldErrors<Field> = {};
    const emailError = validateEmail(email);
    const passwordError = validateLoginPassword(password);
    if (emailError) next.email = emailError;
    if (passwordError) next.password = passwordError;
    setErrors(next);
    setFormError(null);
    if (emailError || passwordError) {
      document.getElementById(emailError ? 'email' : 'password')?.focus();
      return;
    }

    setSubmitting(true);
    try {
      await signIn({ email: email.trim(), password });
      navigate(safeRedirectPath(from), { replace: true });
    } catch (error) {
      setPassword('');
      if (error instanceof ApiError) {
        if (error.status === 401) setFormError('The email or password is incorrect.');
        else if (error.status === 422 && Object.keys(error.fieldErrors).length) {
          setErrors({
            email: error.fieldErrors.email,
            password: error.fieldErrors.password,
          });
        } else setFormError(error.message);
      } else {
        setFormError('Something went wrong. Please try again.');
      }
      setSubmitting(false);
    }
  }

  return (
    <AuthLayout
      title="Sign in"
      description="Welcome back. Sign in to continue your interview practice."
      footer={
        <>
          New here?{' '}
          <Link
            to="/register"
            className="font-medium text-primary underline-offset-4 hover:underline"
          >
            Create an account
          </Link>
        </>
      }
    >
      <GoogleSignIn
        mode="signin"
        disabled={submitting}
        onBusyChange={setGoogleBusy}
        onSuccess={() => navigate(safeRedirectPath(from), { replace: true })}
      />
      <form noValidate onSubmit={handleSubmit} className="mt-4 space-y-4">
        {sessionExpired && !formError && (
          <Alert variant="warning" title="Your session has ended">
            Please sign in again to continue.
          </Alert>
        )}
        {restoreError && !formError && (
          <Alert variant="warning" title="Could not restore your session">
            <p>{restoreError}</p>
            <Button variant="link" className="h-auto p-0" onClick={retryRestore}>
              Try again
            </Button>
          </Alert>
        )}
        {formError && (
          <Alert variant="error" title="Sign-in failed">
            {formError}
          </Alert>
        )}
        <FormField id="email" label="Email" error={errors.email}>
          {(a11y) => (
            <Input
              {...a11y}
              type="email"
              autoComplete="email"
              inputMode="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={submitting || googleBusy}
            />
          )}
        </FormField>
        <FormField id="password" label="Password" error={errors.password}>
          {(a11y) => (
            <Input
              {...a11y}
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={submitting || googleBusy}
            />
          )}
        </FormField>
        <Button type="submit" className="w-full" loading={submitting} disabled={googleBusy}>
          {submitting ? 'Signing in…' : 'Sign in'}
        </Button>
        <p className="text-xs text-muted-foreground">
          Password reset is not available yet. If you are locked out after repeated attempts, wait a
          few minutes and try again.
        </p>
      </form>
    </AuthLayout>
  );
}
