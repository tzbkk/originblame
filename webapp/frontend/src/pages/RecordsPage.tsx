import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useDataset } from '../context/DatasetContext';
import { useRecords, useRecordDetail, useRevokeRecord, useRestoreSection } from '../api/hooks';
import { useToast } from '../components/Toast';
import DataTable from '../components/DataTable';
import type { RecordItem, RecordDetail } from '../api/types';

export default function RecordsPage() {
  const { dataset } = useDataset();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const urlHash = searchParams.get('hash') || '';
  const [search, setSearch] = useState('');
  const [authorFilter, setAuthorFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [yearFilter, setYearFilter] = useState('');
  const [page, setPage] = useState(1);
  const [limit, setLimit] = useState(20);
  const [selectedHash, setSelectedHash] = useState(urlHash);

  useEffect(() => {
    if (urlHash) {
      setSelectedHash(urlHash);
    }
  }, [urlHash]);

  const { data: recordsData } = useRecords(dataset, {
    search: search || undefined,
    author: authorFilter || undefined,
    status: statusFilter || undefined,
    year: yearFilter || undefined,
    page,
    limit,
  });
  const { data: detail } = useRecordDetail(dataset, selectedHash, "");

  const columns = [
    { key: 'title', header: 'Title', className: 'min-w-[150px]' },
    { key: 'heading', header: 'Heading', className: 'min-w-[100px]' },
    {
      key: 'preview',
      header: 'Preview',
      className: 'max-w-[300px]',
      render: (r: RecordItem) => (
        <span className="text-slate-500 truncate block max-w-[300px]">{r.preview}</span>
      ),
    },
    { key: 'authors', header: 'Authors', className: 'min-w-[150px]' },
    { key: 'year', header: 'Year' },
    {
      key: 'status',
      header: 'Status',
      render: (r: RecordItem) =>
        r.status === 'revoked' ? (
          <span className="text-red-600 font-medium">Revoked</span>
        ) : (
          <span className="text-green-600 font-medium">Active</span>
        ),
    },
  ];

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-2xl font-bold text-slate-800">Record Browser</h2>
        <p className="text-sm text-slate-500 mt-1">
          Browse, search, and inspect training records with provenance metadata.
        </p>
      </div>

      <div className="grid grid-cols-4 gap-3">
        <input
          type="text"
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(1); }}
          placeholder="Search title or text..."
          className="rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <input
          type="text"
          value={authorFilter}
          onChange={(e) => { setAuthorFilter(e.target.value); setPage(1); }}
          placeholder="Filter by author..."
          className="rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <select
          value={statusFilter}
          onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
          className="rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">All Status</option>
          <option value="active">Active</option>
          <option value="revoked">Revoked</option>
        </select>
        <input
          type="text"
          value={yearFilter}
          onChange={(e) => { setYearFilter(e.target.value); setPage(1); }}
          placeholder="Filter by year..."
          className="rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <DataTable
        columns={columns}
        data={recordsData?.items || []}
        total={recordsData?.total || 0}
        page={page}
        limit={limit}
        onPageChange={setPage}
        onLimitChange={(l) => { setLimit(l); setPage(1); }}
        onRowClick={(row) => { setSelectedHash(row.line_hash || ''); navigate(`/records?hash=${encodeURIComponent(row.line_hash || '')}#detail`, { replace: true }); }}
      />

      {detail && (
        <div id="detail">
          <RecordDetailPanel
          detail={detail}
          dataset={dataset}
          hash={selectedHash}
          onNavigateAuthor={(id) => navigate(`/authors?id=${id}`)}
        />
        </div>
      )}
    </div>
  );
}

function RecordDetailPanel({
  detail,
  dataset,
  hash: _hash,
  onNavigateAuthor,
}: {
  detail: RecordDetail;
  dataset: string;
  hash: string;
  onNavigateAuthor: (id: string) => void;
}) {
  const { toast } = useToast();
  const revokeRecord = useRevokeRecord(dataset);
  const restoreSection = useRestoreSection(dataset);

  const handleRevoke = () => {
    const path = detail.sections[0]?.path || '';
    const ok = window.confirm(
      `Revoke this record?\n\n"${detail.title}"\n\nThis will revoke the owning section. You can undo this from the Undo page.`
    );
    if (!ok) return;
    revokeRecord.mutate(path, {
      onSuccess: (res) => toast(res.message, 'success'),
      onError: (err) => toast(`Revoke failed: ${err.message}`, 'error'),
    });
  };

  const handleRestore = () => {
    const hash = detail.sections[0]?.section_hash;
    if (!hash) return;
    const ok = window.confirm(`Restore this record?\n\n"${detail.title}"`);
    if (!ok) return;
    restoreSection.mutate(hash, {
      onSuccess: (res) => toast(res.message, 'success'),
      onError: (err) => toast(`Restore failed: ${err.message}`, 'error'),
    });
  };

  return (
    <div className="border border-slate-200 rounded-lg bg-white p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h3 className="text-lg font-semibold text-slate-800">Record Detail</h3>
          {detail.revoked ? (
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
          {!detail.revoked && (
            <button
              onClick={handleRevoke}
              disabled={revokeRecord.isPending}
              className="px-3 py-1.5 rounded-md text-sm font-medium bg-red-600 text-white hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {revokeRecord.isPending ? 'Revoking...' : 'Revoke Record'}
            </button>
          )}
          {detail.revoked && detail.sections.length > 0 && (
            <button
              onClick={handleRestore}
              disabled={restoreSection.isPending}
              className="px-3 py-1.5 rounded-md text-sm font-medium bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {restoreSection.isPending ? 'Restoring...' : 'Restore Record'}
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 text-sm">
        <div>
          <span className="text-slate-500">Title:</span>{' '}
          <span className="text-slate-800">{detail.title}</span>
        </div>
        <div>
          <span className="text-slate-500">Heading:</span>{' '}
          <span className="text-slate-800">{detail.heading}</span>
        </div>
        <div>
          <span className="text-slate-500">Year:</span>{' '}
          <span className="text-slate-800">{detail.year}</span>
        </div>
      </div>
      <div className="text-sm">
        <span className="text-slate-500">License:</span>{' '}
        <span className="text-slate-800">{detail.license}</span>
      </div>
      <div className="text-sm">
        <span className="text-slate-500">Doc Hash:</span>{' '}
        <span className="text-slate-800 font-mono text-xs select-all">{detail.line_hash || '—'}</span>
      </div>
      <div className="text-sm">
        <span className="text-slate-500">Authors ({detail.authors.length}):</span>{' '}
        <span className="text-slate-800">
          {detail.authors.map((a, i) => (
            <span key={a.id || i}>
              {i > 0 && ', '}
              <button
                onClick={() => onNavigateAuthor(a.id)}
                className="text-blue-600 hover:text-blue-800 hover:underline font-medium"
              >
                {a.name || a.id}
              </button>
              {a.email && <span className="text-slate-400 text-xs ml-1">&lt;{a.email}&gt;</span>}
            </span>
          ))}
        </span>
      </div>
      {detail.sections.length > 0 && detail.sections.map((s, i) => (
        <div className="text-sm space-y-1 border-t border-slate-100 pt-3" key={i}>
          <div className="flex items-center gap-2">
            <span className="text-slate-500">Section:</span>{' '}
            <span className="text-slate-800 font-mono text-xs select-all">{s.section_hash.slice(0, 16)}...</span>
            {s.revoked ? (
              <span className="px-1.5 py-0.5 rounded text-xs bg-red-50 text-red-600 border border-red-100">revoked</span>
            ) : (
              <span className="px-1.5 py-0.5 rounded text-xs bg-green-50 text-green-600 border border-green-100">active</span>
            )}
          </div>
          <div>
            <span className="text-slate-500">Path:</span>{' '}
            <span className="text-slate-700">{s.path}</span>
          </div>
          <div>
            <span className="text-slate-500">Section License:</span>{' '}
            <span className="text-slate-700">{s.license}</span>
            {s.year && (
              <>
                <span className="text-slate-400 mx-2">|</span>
                <span className="text-slate-500">Year:</span>{' '}
                <span className="text-slate-700">{s.year}</span>
              </>
            )}
          </div>
          {s.contributors.length > 0 && (
            <div>
              <span className="text-slate-500">Contributors ({s.contributors.length}):</span>{' '}
              <span className="text-slate-700">{s.contributors.join(', ')}</span>
            </div>
          )}
        </div>
      ))}
      <div>
        <p className="text-sm font-medium text-slate-600 mb-1">Full Text</p>
        <textarea
          readOnly
          value={detail.text}
          className="w-full h-96 rounded-md border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700 font-mono resize-y focus:outline-none"
        />
      </div>
    </div>
  );
}
