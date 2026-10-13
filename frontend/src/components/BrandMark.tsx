import { Link } from 'react-router';

export function BrandMark({ to = '/' }: { to?: string }) {
  return (
    <Link
      to={to}
      className="inline-flex items-center gap-2 rounded-md font-semibold text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <span
        aria-hidden="true"
        className="flex size-7 items-center justify-center rounded-md bg-primary text-primary-foreground"
      >
        <svg viewBox="0 0 32 32" className="size-4" fill="none">
          <path
            d="M9 22l7-12 7 12"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </span>
      <span>Adaptive Interviewer</span>
    </Link>
  );
}
