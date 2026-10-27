import { Link } from 'react-router';

import { buttonVariants } from '@/components/ui/button';

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
