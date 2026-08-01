import { useDataset } from '../context/DatasetContext';
import { useRevoked, useRestoreAuthor, useRestoreSection } from '../api/hooks';
import { useToast } from '../components/Toast';
import MetricCard from '../components/MetricCard';

export default function UndoPage() {
  const { dataset } = useDataset();
  const { toast } = useToast();
  const { data, isLoading, refetch } = useRevoked(dataset);
  const restoreAuthor = useRestoreAuthor(dataset);
  const restoreSection = useRestoreSection(dataset);

  if (isLoading || !data) {
    return <div className="text-slate-500 py-10 text-center">Loading...</div>;
  }

  const handleRestoreAuthor = (id: string, name: string) => {
    restoreAuthor.mutate(id, {
      onSuccess: () => {
        toast(`Restored author: ${name}`);
        refetch();
      },
      onError: (err) => toast(err.message, 'error'),
    });
  };

  const handleRestoreAllAuthors = () => {
    Promise.all(
      data.revoked_authors.map((a) =>
        restoreAuthor.mutateAsync(a.id),
      ),
    )
      .then(() => {
        toast(`Restored ${data.revoked_authors.length} authors`);
        refetch();
      })
      .catch((err) => toast(err.message, 'error'));
  };

  const handleRestoreSection = (hash: string) => {
    restoreSection.mutate(hash, {
      onSuccess: () => {
        toast('Section restored');
        refetch();
      },
      onError: (err) => toast(err.message, 'error'),
    });
  };

  const handleRestoreAllSections = () => {
    Promise.all(
      data.revoked_sections.map((s) =>
        restoreSection.mutateAsync(s.section_hash),
      ),
    )
      .then(() => {
        toast(`Restored ${data.revoked_sections.length} sections`);
        refetch();
      })
      .catch((err) => toast(err.message, 'error'));
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-800">Undo Revocation</h2>
        <p className="text-sm text-slate-500 mt-1">
          Restore revoked authors, sections, and records.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <MetricCard label="Revoked Authors" value={data.revoked_authors.length} />
        <MetricCard label="Revoked Sections" value={data.revoked_sections.length} />
        <MetricCard label="Affected (cascade)" value={data.cascade_count} />
      </div>

      {data.revoked_authors.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-slate-800">
              Revoked Authors
            </h3>
            {data.revoked_authors.length > 1 && (
              <button
                onClick={handleRestoreAllAuthors}
                className="bg-green-500 hover:bg-green-600 text-white font-medium px-4 py-2 rounded-md text-sm transition-colors"
              >
                Restore All Authors
              </button>
            )}
          </div>
          <div className="space-y-2">
            {data.revoked_authors.map((a) => (
              <div
                key={a.id}
                className="flex items-center justify-between bg-white border border-slate-200 rounded-lg px-4 py-3"
              >
                <div>
                  <p className="font-medium text-slate-800">{a.name}</p>
                  <p className="text-xs text-slate-500">
                    {a.email} &middot; {a.affected_sections} sections affected
                  </p>
                </div>
                <button
                  onClick={() => handleRestoreAuthor(a.id, a.name)}
                  className="bg-green-500 hover:bg-green-600 text-white font-medium px-4 py-1.5 rounded-md text-sm transition-colors"
                >
                  Restore
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.revoked_sections.length > 0 && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-slate-800">
              Revoked Sections (direct)
            </h3>
            {data.revoked_sections.length > 1 && (
              <button
                onClick={handleRestoreAllSections}
                className="bg-green-500 hover:bg-green-600 text-white font-medium px-4 py-2 rounded-md text-sm transition-colors"
              >
                Restore All Sections
              </button>
            )}
          </div>
          <div className="space-y-2">
            {data.revoked_sections.map((s) => (
              <div
                key={s.section_hash}
                className="flex items-center justify-between bg-white border border-slate-200 rounded-lg px-4 py-3"
              >
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-slate-800 truncate">
                    {s.title} #{s.heading}
                  </p>
                  <p className="text-xs text-slate-500">
                    Authors: {s.authors} &middot; {s.record_count} record(s)
                  </p>
                </div>
                <button
                  onClick={() => handleRestoreSection(s.section_hash)}
                  className="bg-green-500 hover:bg-green-600 text-white font-medium px-4 py-1.5 rounded-md text-sm transition-colors ml-3"
                >
                  Restore
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.revoked_authors.length === 0 &&
        data.revoked_sections.length === 0 &&
        data.cascade_count === 0 && (
          <div className="bg-slate-50 border border-slate-200 rounded-lg px-4 py-8 text-center text-slate-500 text-sm">
            No revoked entries found.
          </div>
        )}
    </div>
  );
}
