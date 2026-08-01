import type { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';
import { useDataset } from '../context/DatasetContext';
import {
  BarChart3,
  Users,
  FileText,
  Scale,
  RotateCcw,
  ClipboardList,
  RefreshCw,
} from 'lucide-react';

const NAV_ITEMS = [
  { to: '/', label: 'Dataset Overview', icon: BarChart3 },
  { to: '/authors', label: 'Author Browser', icon: Users },
  { to: '/records', label: 'Record Browser', icon: FileText },
  { to: '/erasure', label: 'Right-to-Erasure', icon: Scale },
  { to: '/undo', label: 'Undo Revocation', icon: RotateCcw },
  { to: '/audit', label: 'Audit Log', icon: ClipboardList },
  { to: '/reset', label: 'Reset Demo', icon: RefreshCw },
];

export default function Layout({ children }: { children: ReactNode }) {
  const { dataset, setDataset, datasets } = useDataset();

  return (
    <div className="flex min-h-screen bg-slate-50">
      <aside className="w-64 shrink-0 bg-white border-r border-slate-200 flex flex-col sticky top-0 h-screen overflow-y-auto">
        <div className="p-5 border-b border-slate-200">
          <h1 className="text-lg font-bold text-slate-800">OriginBlame</h1>
          <p className="text-xs text-slate-500 mt-0.5">
            ML Unlearning Compliance
          </p>
        </div>

        <div className="px-4 py-3 border-b border-slate-200">
          <label className="block text-xs font-medium text-slate-500 mb-1">
            Dataset
          </label>
          <select
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
            className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {datasets.map((ds) => (
              <option key={ds} value={ds}>
                {ds}
              </option>
            ))}
          </select>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-2.5 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-blue-50 text-blue-700'
                    : 'text-slate-600 hover:bg-slate-100 hover:text-slate-800'
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="p-4 border-t border-slate-200">
          <p className="text-xs text-slate-400 leading-relaxed">
            Record- and token-level data provenance for AI training datasets.
          </p>
        </div>
      </aside>

      <main className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto p-6">{children}</div>
      </main>
    </div>
  );
}
