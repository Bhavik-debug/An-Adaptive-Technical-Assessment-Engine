import { LogOut } from 'lucide-react';
import * as React from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router';

import { useAuth } from '@/auth/AuthProvider';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

import { BrandMark } from './BrandMark';

const navItems = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/interviews/new', label: 'New interview' },
];

/** The signed-in shell: header with navigation and account, then the page. */
export function AppLayout() {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();
  const [signingOut, setSigningOut] = React.useState(false);

  async function handleSignOut() {
    setSigningOut(true);
    await signOut();
    navigate('/login', { replace: true });
  }

  return (
    <div className="flex min-h-svh flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-card focus:px-3 focus:py-2 focus:shadow"
      >
        Skip to content
      </a>
      <header className="border-b bg-card">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
          <BrandMark to="/dashboard" />
          <nav aria-label="Main" className="order-3 w-full sm:order-none sm:w-auto">
            <ul className="flex gap-1">
              {navItems.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    end
                    className={({ isActive }) =>
                      cn(
                        'inline-flex h-9 items-center rounded-md px-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                        isActive && 'bg-accent text-accent-foreground',
                      )
                    }
                  >
                    {item.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          <div className="ml-auto flex items-center gap-3">
            {user && (
              <span
                className="hidden max-w-56 truncate text-sm text-muted-foreground md:inline"
                title={user.email}
              >
                {user.email}
              </span>
            )}
            <Button variant="outline" size="sm" onClick={handleSignOut} loading={signingOut}>
              {!signingOut && <LogOut aria-hidden="true" />}
              Sign out
            </Button>
          </div>
        </div>
      </header>
      <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6 sm:py-8">
        <Outlet />
      </main>
    </div>
  );
}
