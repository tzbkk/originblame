import { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useDataset } from '../context/DatasetContext';
import {
  useAuthors,
  useErasureImpact,
  useRevokeAuthor,
  useRevokeSection,
  useRevokeRecord,
} from '../api/hooks';
import { useToast } from '../components/Toast';
import MetricCard from '../components/MetricCard';

type RevokeType = 'author' | 'section' | 'record';

export default function ErasurePage() {
  const { dataset } = useDataset();
  const { toast } = useToast();
  const [revokeType, setRevokeType] = useState<RevokeType>('author');
  const [target, setTarget] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  const { data: searchResults } = useAuthors(dataset, {
    search: searchQuery || undefined,
    limit: 20,
  });
  const { data: impact, isLoading: impactLoading } = useErasureImpact(
    dataset,
    revokeType,
    target,
  );

  const revokeAuthor = useRevokeAuthor(dataset);
  const revokeSection = useRevokeSection(dataset);
  const revokeRecord = useRevokeRecord(dataset);

  const handleRevoke = () => {
    if (!impact || impact.is_already_revoked) return;
    const onSuccess = (msg: string) => {
      toast(msg);
      setTarget('');
    };
    const onError = (err: Error) => toast(err.message, 'error');

    if (revokeType === 'author') {
      revokeAuthor.mutate(target, {
        onSuccess: () => onSuccess(`Author revoked`),
        onError,
      });
    } else if (revokeType === 'section') {
      revokeSection.mutate(target, {
        onSuccess: () => onSuccess('Section revoked'),
        onError,
      });
    } else {
      revokeRecord.mutate(target, {
        onSuccess: () => onSuccess('Record revoked'),
        onError,
      });
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-800">Right-to-Erasure</h2>
        <p className="text-sm text-slate-500 mt-1">
          Revoke author, section, or record-level provenance to fulfill GDPR Art. 17.
        </p>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-slate-800 mb-3">Step 1: Select Target</h3>
        <div className="flex gap-4 mb-4">
          {(['author', 'section', 'record'] as RevokeType[]).map((t) => (
            <label key={t} className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="revokeType"
                checked={revokeType === t}
                onChange={() => { setRevokeType(t); setTarget(''); }}
                className="accent-blue-600"
              />
              <span className="text-sm font-medium text-slate-700 capitalize">{t}</span>
            </label>
          ))}
        </div>

        {revokeType === 'author' && (
          <div className="space-y-3 max-w-md">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search author by name..."
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {searchQuery && searchResults?.items && searchResults.items.length > 0 && (
              <div className="border border-slate-200 rounded-lg overflow-hidden">
                {searchResults.items.slice(0, 10).map((a) => (
                  <div
                    key={a.id}
                    onClick={() => { setTarget(a.email); setSearchQuery(''); }}
                    className={`px-3 py-2 text-sm cursor-pointer hover:bg-blue-50 border-b border-slate-100 last:border-0 ${
                      target === a.email ? 'bg-blue-50 ring-1 ring-blue-200' : ''
                    }`}
                  >
                    <span className="font-medium text-slate-700">{a.name}</span>
                    <span className="text-slate-400 ml-2 text-xs">{a.sections} sections</span>
                    {a.revoked && <span className="text-red-500 ml-2 text-xs">(revoked)</span>}
                  </div>
                ))}
              </div>
            )}
            {target && <p className="text-sm text-green-700">Selected: {target}</p>}
          </div>
        )}

        {revokeType === 'section' && (
          <div className="space-y-3 max-w-md">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search section by path or title..."
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {searchQuery && searchResults?.items && searchResults.items.length > 0 && (
              <div className="border border-slate-200 rounded-lg overflow-hidden">
                {searchResults.items.slice(0, 10).map((a) => (
                  <div key={a.id} onClick={() => { setTarget(a.id); setSearchQuery(''); }}
                    className="px-3 py-2 text-sm cursor-pointer hover:bg-blue-50 border-b border-slate-100 last:border-0">
                    <span className="font-medium text-slate-700">{a.name}</span>
                    <span className="text-slate-400 ml-2 text-xs">{a.email}</span>
                  </div>
                ))}
              </div>
            )}
            {target && <p className="text-sm text-green-700">Selected: {target.slice(0,16)}...</p>}
          </div>
        )}

        {revokeType === 'record' && (
          <div className="space-y-3 max-w-md">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search record by title..."
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {searchQuery && searchResults?.items && searchResults.items.length > 0 && (
              <div className="border border-slate-200 rounded-lg overflow-hidden">
                {searchResults.items.slice(0, 10).map((a) => (
                  <div key={a.id} onClick={() => { setTarget(a.name); setSearchQuery(''); }}
                    className="px-3 py-2 text-sm cursor-pointer hover:bg-blue-50 border-b border-slate-100 last:border-0">
                    <span className="font-medium text-slate-700">{a.name}</span>
                  </div>
                ))}
              </div>
            )}
            {target && <p className="text-sm text-green-700">Selected: {target}</p>}
          </div>
        )}
      </div>

      {target && impactLoading && (
        <div className="text-slate-500 text-sm">Loading impact assessment...</div>
      )}

      {target && impact && (
        <>
          <div>
            <h3 className="text-lg font-semibold text-slate-800 mb-3">Step 2: Impact Assessment</h3>
            {revokeType === 'author' ? (
              <div className="grid grid-cols-2 gap-4">
                <div className="border border-slate-200 rounded-lg p-4">
                  <p className="font-medium text-slate-700 mb-2">As Author (blame)</p>
                  <div className="grid grid-cols-2 gap-3">
                    <MetricCard label="Sections" value={impact.affected_sections} />
                    <MetricCard label="Records" value={impact.affected_records} />
                  </div>
                </div>
                <div className="border border-slate-200 rounded-lg p-4">
                  <p className="font-medium text-slate-700 mb-2">As Contributor (page-level)</p>
                  <div className="grid grid-cols-2 gap-3">
                    <MetricCard label="Sections" value={impact.affected_contrib_sections} />
                    <MetricCard label="Records" value={impact.affected_contrib_records} />
                  </div>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-3 gap-4">
                <MetricCard label="Affected Sections" value={impact.affected_sections} />
                <MetricCard label="Affected Records" value={impact.affected_records} />
                <MetricCard label="Total Sections" value={impact.total_sections} />
              </div>
            )}
            <p className="text-xs text-slate-500 mt-2">{impact.revoke_desc}</p>
          </div>

          {impact.comparison && (
            <div>
              <h3 className="text-lg font-semibold text-slate-800 mb-3">Step 3: Compare Deletion Methods</h3>
              <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-5">
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart
                    data={[
                      {
                        name: 'Sections',
                        'Dataset-level': impact.comparison.file_level,
                        'Contributor scope': impact.comparison.contributor_level,
                        OriginBlame: impact.comparison.record_level,
                      },
                    ]}
                  >
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip formatter={(value) => Number(value).toLocaleString()} />
                    <Legend />
                    <Bar dataKey="Dataset-level" fill="#e74c3c" radius={[2, 2, 0, 0]} />
                    <Bar dataKey="Contributor scope" fill="#f39c12" radius={[2, 2, 0, 0]} />
                    <Bar dataKey="OriginBlame" fill="#27ae60" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <p className="text-sm text-slate-600 mt-2">
                  Dataset-level would remove{' '}
                  <strong>{impact.comparison.file_level.toLocaleString()}</strong> sections,
                  contributor-level{' '}
                  <strong>{impact.comparison.contributor_level.toLocaleString()}</strong>,
                  OriginBlame pinpoints{' '}
                  <strong>{impact.comparison.record_level.toLocaleString()}</strong> &mdash;{' '}
                  <strong>{impact.comparison.factor}x</strong> over-deletion avoided.
                </p>
              </div>
            </div>
          )}

          <div>
            <h3 className="text-lg font-semibold text-slate-800 mb-3">Step 4: Execute Revocation</h3>
            {impact.is_already_revoked ? (
              <div className="bg-green-50 border border-green-200 text-green-700 rounded-md px-4 py-3 text-sm">
                <strong>{impact.target_name}</strong> has been revoked.
              </div>
            ) : (
              <div className="space-y-3">
                <div className="bg-amber-50 border border-amber-200 text-amber-700 rounded-md px-4 py-2 text-sm">
                  {impact.revoke_desc}
                </div>
                <button
                  onClick={handleRevoke}
                  disabled={revokeAuthor.isPending || revokeSection.isPending || revokeRecord.isPending}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-5 py-2.5 rounded-md text-sm transition-colors disabled:opacity-50"
                >
                  Revoke {revokeType === 'author' ? `Author: ${target}` : revokeType === 'section' ? 'Section' : 'Record'}
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
