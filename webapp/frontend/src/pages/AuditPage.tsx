import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { useDataset } from '../context/DatasetContext';
import { useAuditLog } from '../api/hooks';
import type { AuditEntry } from '../api/types';

const OP_ICONS: Record<string, string> = {
  init: '🚀',
  revoke_author: '🔴',
  revoke: '🔴',
  revoke_section: '🔴',
  restore_author: '🟢',
  restore_section: '🟢',
  clean: '🧹',
};

export default function AuditPage() {
  const { dataset } = useDataset();
  const { data, isLoading } = useAuditLog(dataset);
  const [filterOp, setFilterOp] = useState('');
  const [expanded, setExpanded] = useState<Set<number>>(new Set([0, 1, 2]));

  if (isLoading || !data) {
    return <div className="text-slate-500 py-10 text-center">Loading...</div>;
  }

  const ops = [...new Set(data.entries.map((e) => e.op))].sort();
  const filtered = filterOp
    ? data.entries.filter((e) => e.op === filterOp)
    : data.entries;

  const toggle = (idx: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  };

  return (
    <div className="space-y-5">
      <h2 className="text-2xl font-bold text-slate-800">Audit Log</h2>

      <div className="flex items-center gap-3">
        <select
          value={filterOp}
          onChange={(e) => setFilterOp(e.target.value)}
          className="rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">All operations</option>
          {ops.map((op) => (
            <option key={op} value={op}>
              {op}
            </option>
          ))}
        </select>
        <span className="text-xs text-slate-500">
          {filtered.length} entries (most recent first)
        </span>
      </div>

      {filtered.length === 0 ? (
        <div className="bg-slate-50 border border-slate-200 rounded-lg px-4 py-8 text-center text-slate-500 text-sm">
          No audit log entries found.
        </div>
      ) : (
        <div className="space-y-2">
          {filtered.slice(0, 100).map((entry, i) => (
            <AuditEntryCard
              key={i}
              entry={entry}
              expanded={expanded.has(i)}
              onToggle={() => toggle(i)}
            />
          ))}
          {filtered.length > 100 && (
            <p className="text-xs text-slate-500 text-center">
              Showing 100 of {filtered.length} entries
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function AuditEntryCard({
  entry,
  expanded,
  onToggle,
}: {
  entry: AuditEntry;
  expanded: boolean;
  onToggle: () => void;
}) {
  const icon = OP_ICONS[entry.op] || '📌';
  const detailStr =
    typeof entry.detail === 'string'
      ? entry.detail
      : JSON.stringify(entry.detail);

  return (
    <div className="bg-white border border-slate-200 rounded-lg">
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-2 px-4 py-3 text-left hover:bg-slate-50 transition-colors"
      >
        {expanded ? (
          <ChevronDown size={14} className="text-slate-400 shrink-0" />
        ) : (
          <ChevronRight size={14} className="text-slate-400 shrink-0" />
        )}
        <span className="shrink-0">{icon}</span>
        <span className="text-xs text-slate-500 font-mono shrink-0">
          [{entry.ts}]
        </span>
        <span className="text-sm font-medium text-slate-800 shrink-0">
          {entry.op}
        </span>
        <span className="text-sm text-slate-500 truncate ml-1">
          {detailStr.slice(0, 60)}
        </span>
      </button>
      {expanded && (
        <div className="px-4 pb-3 pt-1 border-t border-slate-100">
          {typeof entry.detail === 'object' && entry.detail !== null ? (
            <pre className="text-xs text-slate-700 bg-slate-50 rounded-md p-3 overflow-x-auto">
              {JSON.stringify(entry.detail, null, 2)}
            </pre>
          ) : detailStr ? (
            <p className="text-sm text-slate-600">{detailStr}</p>
          ) : null}
          {entry.cmd && (
            <p className="text-xs text-slate-400 mt-2 font-mono">
              Command: {entry.cmd}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
