import * as React from 'react';
import { Link, useNavigate } from 'react-router';

import { MIN_PASSWORD_LENGTH } from '@/api/auth';
import { ApiError } from '@/api/client';
import { useAuth } from '@/auth/AuthProvider';
import { AuthLayout } from '@/components/AuthLayout';
import { FormField } from '@/components/FormField';
import { GoogleSignIn } from '@/components/GoogleSignIn';
import { Alert } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { validateEmail, validateNewPassword, type FieldErrors } from '@/lib/validation';

type Field = 'email' | 'password' | 'confirm';

export function RegisterPage() {
  const { signUp } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = React.useState('');
  const [password, setPassword] = React.useState('');
  const [confirm, setConfirm] = React.useState('');
  const [errors, setErrors] = React.useState<FieldErrors<Field>>({});
  const [formError, setFormError] = React.useState<string | null>(null);
  const [submitting, setSubmitting] = React.useState(false);
  // Google's exchange is in flight: the email form waits for it.
  const [googleBusy, setGoogleBusy] = React.useState(false);

  function validate(): FieldErrors<Field> {
    const next: FieldErrors<Field> = {};
    const emailError = validateEmail(email);
    const passwordError = validateNewPassword(password);
    if (emailError) next.email = emailError;
    if (passwordError) next.password = passwordError;
    if (!confirm) next.confirm = 'Re-enter your password.';
    else if (confirm !== password) next.confirm = 'The passwords do not match.';
    return next;
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitting || googleBusy) return;
    const next = validate();
    setErrors(next);
    setFormError(null);
    const firstInvalid = (['email', 'password', 'confirm'] as const).find((f) => next[f]);
    if (firstInvalid) {
      document.getElementById(firstInvalid)?.focus();
      return;
    }

    setSubmitting(true);
    try {
      await signUp({ email: email.trim(), password });
      navigate('/dashboard', { replace: true });
    } catch (error) {
      if (error instanceof ApiError) {
        if (error.status === 409) {
          setErrors({ email: 'An account with this email already exists.' });
          document.getElementById('email')?.focus();
        } else if (error.status === 422 && Object.keys(error.fieldErrors).length) {
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
      title="Create your account"
      description="Practise technical interviews that adapt to your answers."
      footer={
        <>
          Already have an account?{' '}
          <Link to="/login" className="font-medium text-primary underline-offset-4 hover:underline">
            Sign in
          </Link>
        </>
      }
    >
      <GoogleSignIn
        mode="signup"
        disabled={submitting}
        onBusyChange={setGoogleBusy}
        onSuccess={() => navigate('/dashboard', { replace: true })}
      />
      <form noValidate onSubmit={handleSubmit} className="mt-4 space-y-4">
        {formError && (
          <Alert variant="error" title="We could not create your account">
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
        <FormField
          id="password"
          label="Password"
          hint={`At least ${MIN_PASSWORD_LENGTH} characters. A short phrase is easy to remember.`}
          error={errors.password}
        >
          {(a11y) => (
            <Input
              {...a11y}
              type="password"
              autoComplete="new-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={submitting || googleBusy}
            />
          )}
        </FormField>
        <FormField id="confirm" label="Confirm password" error={errors.confirm}>
          {(a11y) => (
            <Input
              {...a11y}
              type="password"
              autoComplete="new-password"
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              disabled={submitting || googleBusy}
            />
          )}
        </FormField>
        <Button type="submit" className="w-full" loading={submitting} disabled={googleBusy}>
          {submitting ? 'Creating account…' : 'Create account'}
        </Button>
      </form>
    </AuthLayout>
  );
}
