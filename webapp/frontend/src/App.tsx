import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DatasetProvider } from './context/DatasetContext';
import { ToastProvider } from './components/Toast';
import Layout from './components/Layout';
import OverviewPage from './pages/OverviewPage';
import AuthorsPage from './pages/AuthorsPage';
import RecordsPage from './pages/RecordsPage';
import ErasurePage from './pages/ErasurePage';
import UndoPage from './pages/UndoPage';
import AuditPage from './pages/AuditPage';
import ResetPage from './pages/ResetPage';

const qc = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, retry: 1 },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={qc}>
      <DatasetProvider>
        <ToastProvider>
          <BrowserRouter>
            <Layout>
              <Routes>
                <Route path="/" element={<OverviewPage />} />
                <Route path="/authors" element={<AuthorsPage />} />
                <Route path="/records" element={<RecordsPage />} />
                <Route path="/erasure" element={<ErasurePage />} />
                <Route path="/undo" element={<UndoPage />} />
                <Route path="/audit" element={<AuditPage />} />
                <Route path="/reset" element={<ResetPage />} />
              </Routes>
            </Layout>
          </BrowserRouter>
        </ToastProvider>
      </DatasetProvider>
    </QueryClientProvider>
  );
}
