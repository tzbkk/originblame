import { useState, useEffect, useRef } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useDataset } from '../context/DatasetContext';
import { useAuthors, useAuthorDetail, useRevokeAuthor, useRestoreAuthor, useOverview } from '../api/hooks';
import { useToast } from '../components/Toast';
import DataTable from '../components/DataTable';
import type { AuthorItem } from '../api/types';

export default function AuthorsPage() {
  const { dataset } = useDataset();
  const navigate = useNavigate();
  const { toast } = useToast();
  const revokeAuthor = useRevokeAuthor(dataset);
  const restoreAuthor = useRestoreAuthor(dataset);
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [limit, setLimit] = useState(20);
  const [searchParams, setSearchParams] = useSearchParams();
  const urlId = searchParams.get('id') || '';
  const [selectedId, setSelectedIdRaw] = useState(urlId);
  const [activeTab, setActiveTab] = useState<'author' | 'contributor'>('author');
  const [authorPage, setAuthorPage] = useState(1);
  const [contribPage, setContribPage] = useState(1);
  const detailRef = useRef<HTMLDivElement>(null);

  const setSelectedId = (id: string) => {
    setSelectedIdRaw(id);
    setSearchParams(id ? { id } : {});
    setAuthorPage(1);
    setContribPage(1);
  };

  useEffect(() => {
    if (urlId && urlId !== selectedId) {
      setSelectedIdRaw(urlId);
    }
  }, [urlId]);

  const { data: authorsData } = useAuthors(dataset, {
    search: search || undefined,
    page,
    limit,
  });
  const { data: detail } = useAuthorDetail(dataset, selectedId, authorPage, contribPage, limit);
  const { data: overview } = useOverview(dataset);

  useEffect(() => {
    if (urlId && overview?.author_ranking && !search) {
      const idx = overview.author_ranking.indexOf(urlId);
      if (idx >= 0) {
        setPage(Math.floor(idx / limit) + 1);
      }
    }
  }, [urlId, overview]);

  // Scroll to detail panel after data loads on pagination
  useEffect(() => {
    if (detail && detailRef.current) {
      detailRef.current.scrollIntoView({ block: 'start' });
    }
  }, [detail?.author_records, detail?.contributor_records]);

  const handleRevoke = () => {
    if (!detail) return;
    const name = detail.author.name;
    const records = detail.metrics.records_as_author + detail.metrics.records_as_contributor;
    const ok = window.confirm(
      `Revoke author "${name}"?\n\nThis will affect ${records} records and ${detail.metrics.sections_as_author} sections.\n\nYou can undo this from the Undo page.`
    );
    if (!ok) return;
    revokeAuthor.mutate(detail.author.email, {
      onSuccess: (res) => toast(res.message, 'success'),
      onError: (err) => toast(`Revoke failed: ${err.message}`, 'error'),
    });
  };

  const handleRestore = () => {
    if (!detail) return;
    const ok = window.confirm(`Restore author "${detail.author.name}"?`);
    if (!ok) return;
    restoreAuthor.mutate(detail.author.id, {
      onSuccess: (res) => toast(res.message, 'success'),
      onError: (err) => toast(`Restore failed: ${err.message}`, 'error'),
    });
  };

  const authorColumns = [
    { key: 'name', header: 'Name', className: 'min-w-[180px]' },
    { key: 'sections', header: 'Sections', render: (r: AuthorItem) => r.sections.toLocaleString() },
    {
      key: 'revoked',
      header: 'Revoked',
      render: (r: AuthorItem) =>
        r.revoked ? <span className="text-red-500">Revoked</span> : <span className="text-slate-400">&mdash;</span>,
    },
  ];

  return (
    <div className="space-y-5">
      <h2 className="text-2xl font-bold text-slate-800">Author Browser</h2>

      <div className="flex gap-3 items-center">
        <input
          type="text"
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(1); }}
          placeholder="Search author by name..."
          className="flex-1 max-w-md rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        {authorsData && (
          <span className="text-xs text-slate-500">
            {authorsData.total.toLocaleString()} authors
          </span>
        )}
      </div>

      <DataTable
        columns={authorColumns}
        data={authorsData?.items || []}
        total={authorsData?.total || 0}
        page={page}
        limit={limit}
        onPageChange={setPage}
        onLimitChange={(l) => { setLimit(l); setPage(1); }}
        onRowClick={(row) => setSelectedId(row.id)}
        selectedIdx={
          selectedId
            ? authorsData?.items.findIndex((a) => a.id === selectedId) ?? -1
            : -1
        }
      />

      {detail && (
        <div ref={detailRef} className="border border-slate-200 rounded-lg bg-white p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-slate-50 rounded-md px-4 py-2">
                <p className="text-xs text-slate-500">Author</p>
                <p className="font-semibold text-slate-800">{detail.author.name}</p>
                <p className="text-xs text-slate-400">{detail.author.email}</p>
              </div>
              {detail.author.revoked ? (
                <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-700 border border-red-200">
                  Revoked
                </span>
              ) : (
                <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-green-100 text-green-700 border border-green-200">
                  Active
                </span>
              )}
            </div>
            <div className="flex gap-2">
              {!detail.author.revoked && (
                <button
                  onClick={handleRevoke}
                  disabled={revokeAuthor.isPending}
                  className="px-3 py-1.5 rounded-md text-sm font-medium bg-red-600 text-white hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {revokeAuthor.isPending ? 'Revoking...' : 'Revoke Author'}
                </button>
              )}
              {detail.author.revoked && (
                <button
                  onClick={handleRestore}
                  disabled={restoreAuthor.isPending}
                  className="px-3 py-1.5 rounded-md text-sm font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {restoreAuthor.isPending ? 'Restoring...' : 'Restore Author'}
                </button>
              )}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="bg-slate-50 rounded-md p-3">
              <p className="text-xs text-slate-500">Sections (author)</p>
              <p className="font-semibold text-slate-800">
                {detail.metrics.sections_as_author.toLocaleString()}
              </p>
            </div>
            <div className="bg-slate-50 rounded-md p-3">
              <p className="text-xs text-slate-500">Records (author)</p>
              <p className="font-semibold text-slate-800">
                {detail.metrics.records_as_author.toLocaleString()}
              </p>
            </div>
            <div className="bg-slate-50 rounded-md p-3">
              <p className="text-xs text-slate-500">Records (contributor)</p>
              <p className="font-semibold text-slate-800">
                {detail.metrics.records_as_contributor.toLocaleString()}
              </p>
            </div>
          </div>

          <div className="flex border-b border-slate-200">
            <button
              onClick={() => setActiveTab('author')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'author'
                  ? 'border-blue-600 text-blue-700'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              Author ({detail.author_total.toLocaleString()})
            </button>
            <button
              onClick={() => setActiveTab('contributor')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'contributor'
                  ? 'border-blue-600 text-blue-700'
                  : 'border-transparent text-slate-500 hover:text-slate-700'
              }`}
            >
              Contributor ({detail.contributor_total.toLocaleString()})
            </button>
          </div>

          {activeTab === 'author' ? (
            detail.author_records.length > 0 ? (
              <>
                <RecordTable
                  records={detail.author_records}
                  onRecordClick={(r) => navigate(`/records?hash=${encodeURIComponent(r.line_hash || '')}#detail`)}
                />
                <PaginationControls
                  page={detail.author_page}
                  total={detail.author_total}
                  limit={detail.author_limit}
                  onPageChange={setAuthorPage}
                />
              </>
            ) : (
              <p className="text-slate-500 text-sm py-4">No records attributed as author.</p>
            )
          ) : detail.contributor_records.length > 0 ? (
            <>
              <ContributorTable
                records={detail.contributor_records}
                onRecordClick={(r) => navigate(`/records?hash=${encodeURIComponent(r.line_hash || '')}#detail`)}
                />
                <PaginationControls
                  page={detail.contributor_page}
                total={detail.contributor_total}
                limit={detail.contributor_limit}
                onPageChange={setContribPage}
              />
            </>
          ) : (
            <p className="text-slate-500 text-sm py-4">No records in contributor scope.</p>
          )}
        </div>
      )}
    </div>
  );
}

function RecordTable({ records, onRecordClick }: { records: Array<{ title: string; heading: string; sec_path: string; year: string; preview: string; line_hash?: string }>; onRecordClick?: (r: { title: string; heading: string; sec_path: string; year: string; preview: string; line_hash?: string }) => void }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-slate-50 border-b border-slate-200">
            <th className="px-3 py-2 text-left font-medium text-slate-600">Title</th>
            <th className="px-3 py-2 text-left font-medium text-slate-600">Heading</th>
            <th className="px-3 py-2 text-left font-medium text-slate-600">Year</th>
            <th className="px-3 py-2 text-left font-medium text-slate-600">Preview</th>
          </tr>
        </thead>
        <tbody>
          {records.map((r, i) => (
            <tr key={i} className={`border-b border-slate-100 ${onRecordClick ? 'cursor-pointer hover:bg-blue-50' : ''}`} onClick={() => onRecordClick?.(r)}>
              <td className="px-3 py-2 text-blue-600 hover:underline">{r.title}</td>
              <td className="px-3 py-2 text-slate-700">{r.heading}</td>
              <td className="px-3 py-2 text-slate-700">{r.year}</td>
              <td className="px-3 py-2 text-slate-500 max-w-[300px] truncate">{r.preview}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ContributorTable({ records, onRecordClick }: { records: Array<{ title: string; heading: string; year: string; preview: string; is_author: boolean; line_hash?: string }>; onRecordClick?: (r: { title: string; heading: string; year: string; preview: string; line_hash?: string }) => void }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-slate-50 border-b border-slate-200">
            <th className="px-3 py-2 text-left font-medium text-slate-600">Title</th>
            <th className="px-3 py-2 text-left font-medium text-slate-600">Heading</th>
            <th className="px-3 py-2 text-left font-medium text-slate-600">Year</th>
            <th className="px-3 py-2 text-left font-medium text-slate-600">Role</th>
            <th className="px-3 py-2 text-left font-medium text-slate-600">Preview</th>
          </tr>
        </thead>
        <tbody>
          {records.map((r, i) => (
            <tr key={i} className={`border-b border-slate-100 ${onRecordClick ? 'cursor-pointer hover:bg-blue-50' : ''}`} onClick={() => onRecordClick?.(r)}>
              <td className="px-3 py-2 text-blue-600 hover:underline">{r.title}</td>
              <td className="px-3 py-2 text-slate-700">{r.heading}</td>
              <td className="px-3 py-2 text-slate-700">{r.year}</td>
              <td className="px-3 py-2">
                {r.is_author ? (
                  <span className="text-blue-600">author</span>
                ) : (
                  <span className="text-slate-500">contributor only</span>
                )}
              </td>
              <td className="px-3 py-2 text-slate-500 max-w-[250px] truncate">{r.preview}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PaginationControls({ page, total, limit, onPageChange }: { page: number; total: number; limit: number; onPageChange: (p: number) => void }) {
  const totalPages = Math.ceil(total / limit);
  if (totalPages <= 1) return null;
  return (
    <div className="flex items-center justify-between px-2 py-2 border-t border-slate-200">
      <span className="text-xs text-slate-500">
        {(page - 1) * limit + 1}–{Math.min(page * limit, total)} of {total.toLocaleString()}
      </span>
      <div className="flex gap-1">
        <button
          onClick={() => onPageChange(page - 1)}
          disabled={page <= 1}
          className="px-2 py-1 text-xs rounded border border-slate-300 disabled:opacity-40 hover:bg-slate-50"
        >
          Prev
        </button>
        <span className="px-2 py-1 text-xs text-slate-600">{page} / {totalPages}</span>
        <button
          onClick={() => onPageChange(page + 1)}
          disabled={page >= totalPages}
          className="px-2 py-1 text-xs rounded border border-slate-300 disabled:opacity-40 hover:bg-slate-50"
        >
          Next
        </button>
      </div>
    </div>
  );
}
