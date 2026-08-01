import { useState, useCallback, useRef, useLayoutEffect } from 'react';
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';

interface Column<T> {
  key: string;
  header: string;
  render?: (row: T) => React.ReactNode;
  className?: string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  total: number;
  page: number;
  limit: number;
  onPageChange: (p: number) => void;
  onLimitChange?: (l: number) => void;
  onRowClick?: (row: T, idx: number) => void;
  selectedIdx?: number;
  emptyMessage?: string;
}

const PAGE_SIZES = [10, 20, 50, 100];

export default function DataTable<T>({
  columns,
  data,
  total,
  page,
  limit,
  onPageChange,
  onLimitChange,
  onRowClick,
  selectedIdx,
  emptyMessage = 'No data',
}: DataTableProps<T>) {
  const totalPages = Math.max(1, Math.ceil(total / limit));
  const start = total > 0 ? (page - 1) * limit + 1 : 0;
  const end = Math.min(page * limit, total);
  const [jumpPage, setJumpPage] = useState('');
  const tableRef = useRef<HTMLDivElement>(null);
  const prevPage = useRef(page);

  useLayoutEffect(() => {
    if (page !== prevPage.current && tableRef.current) {
      tableRef.current.scrollIntoView({ block: 'start' });
    }
    prevPage.current = page;
  }, [page]);

  const handlePageChange = useCallback((p: number) => {
    onPageChange(p);
  }, [onPageChange]);

  const handleJump = () => {
    const p = parseInt(jumpPage, 10);
    if (p >= 1 && p <= totalPages) {
      handlePageChange(p);
      setJumpPage('');
    }
  };

  return (
    <div ref={tableRef}>
      <div className="flex items-center justify-between mb-2 text-xs text-slate-500">
        <span>{total > 0 ? `${start}-${end} of ${total.toLocaleString()}` : emptyMessage}</span>
        {onLimitChange && total > 0 && (
          <select
            value={limit}
            onChange={(e) => onLimitChange(Number(e.target.value))}
            className="border border-slate-200 rounded px-1.5 py-0.5 text-xs bg-white"
          >
            {PAGE_SIZES.map((s) => (
              <option key={s} value={s}>{s}/page</option>
            ))}
          </select>
        )}
      </div>
      <div className="overflow-x-auto rounded-lg border border-slate-200">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-200">
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`px-3 py-2 text-left font-medium text-slate-600 whitespace-nowrap ${col.className || ''}`}
                >
                  {col.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr
                key={i}
                onClick={() => onRowClick?.(row, i)}
                className={`border-b border-slate-100 transition-colors ${
                  onRowClick ? 'cursor-pointer hover:bg-blue-50' : ''
                } ${
                  selectedIdx === (page - 1) * limit + i
                    ? 'bg-blue-50 ring-1 ring-blue-200'
                    : ''
                }`}
              >
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={`px-3 py-2 text-slate-700 ${col.className || ''}`}
                  >
                    {col.render
                      ? col.render(row)
                      : String((row as Record<string, unknown>)[col.key] ?? '')}
                  </td>
                ))}
              </tr>
            ))}
            {data.length === 0 && (
              <tr>
                <td
                  colSpan={columns.length}
                  className="px-3 py-8 text-center text-slate-400"
                >
                  {emptyMessage}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-2 text-xs">
          <div className="flex items-center gap-1">
            <button
              disabled={page <= 1}
              onClick={() => handlePageChange(1)}
              className="p-1 rounded border border-slate-200 disabled:opacity-30 hover:bg-slate-100"
              title="First"
            >
              <ChevronsLeft size={12} />
            </button>
            <button
              disabled={page <= 1}
              onClick={() => handlePageChange(page - 1)}
              className="p-1 rounded border border-slate-200 disabled:opacity-30 hover:bg-slate-100"
            >
              <ChevronLeft size={12} />
            </button>
            <span className="px-1 text-slate-500">
              {page}/{totalPages}
            </span>
            <button
              disabled={page >= totalPages}
              onClick={() => handlePageChange(page + 1)}
              className="p-1 rounded border border-slate-200 disabled:opacity-30 hover:bg-slate-100"
            >
              <ChevronRight size={12} />
            </button>
            <button
              disabled={page >= totalPages}
              onClick={() => handlePageChange(totalPages)}
              className="p-1 rounded border border-slate-200 disabled:opacity-30 hover:bg-slate-100"
              title="Last"
            >
              <ChevronsRight size={12} />
            </button>
          </div>
          {totalPages > 10 && (
            <form
              onSubmit={(e) => { e.preventDefault(); handleJump(); }}
              className="flex items-center gap-1"
            >
              <input
                type="number"
                min={1}
                max={totalPages}
                value={jumpPage}
                onChange={(e) => setJumpPage(e.target.value)}
                placeholder={`1-${totalPages}`}
                className="w-12 border border-slate-200 rounded px-1 py-0.5 text-xs text-center"
              />
              <button
                type="submit"
                className="px-1.5 py-0.5 rounded bg-slate-100 border border-slate-200 hover:bg-slate-200 text-xs"
              >
                Go
              </button>
            </form>
          )}
        </div>
      )}
    </div>
  );
}
