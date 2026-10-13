import { Navigate, Route, Routes } from 'react-router';

import { AuthProvider } from '@/auth/AuthProvider';
import { PublicOnly, RequireAuth } from '@/auth/guards';
import { AppLayout } from '@/components/AppLayout';
import { DashboardPage } from '@/pages/dashboard/DashboardPage';
import { LoginPage } from '@/pages/LoginPage';
import {
  InterviewPlaceholderPage,
  NotFoundPage,
  ReportPlaceholderPage,
} from '@/pages/placeholders';
import { RegisterPage } from '@/pages/RegisterPage';
import { SetupWizardPage } from '@/pages/setup/SetupWizardPage';

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route element={<PublicOnly />}>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
      </Route>
      <Route element={<RequireAuth />}>
        <Route element={<AppLayout />}>
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/interviews/new" element={<SetupWizardPage />} />
          <Route path="/interviews/:sessionId" element={<InterviewPlaceholderPage />} />
          <Route path="/interviews/:sessionId/report" element={<ReportPlaceholderPage />} />
        </Route>
      </Route>
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}

export function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  );
}
