import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useDataset } from '../context/DatasetContext';
import { useOverview, useAuthors } from '../api/hooks';
import MetricCard from '../components/MetricCard';
import DataTable from '../components/DataTable';
import type { AuthorItem } from '../api/types';

export default function OverviewPage() {
  const { dataset } = useDataset();
  const navigate = useNavigate();
  const { data, isLoading } = useOverview(dataset);
  const [page, setPage] = useState(1);
  const [limit, setLimit] = useState(20);
  const { data: authorsData } = useAuthors(dataset, { page, limit });

  if (isLoading || !data) {
    return <div className="text-slate-500 py-10 text-center">Loading...</div>;
  }

  const chartData = data.top_authors.slice(0, 200);

  const columns = [
    { key: 'name', header: 'Name', className: 'min-w-[200px]' },
    { key: 'email', header: 'Email', className: 'min-w-[200px]' },
    {
      key: 'sections',
      header: 'Sections',
      render: (r: AuthorItem) => r.sections.toLocaleString(),
    },
    { key: 'contribution_pct', header: 'Contribution %' },
    {
      key: 'revoked',
      header: 'Revoked',
      render: (r: AuthorItem) =>
        r.revoked ? (
          <span className="text-red-500 font-medium">Yes</span>
        ) : (
          <span className="text-slate-400">&mdash;</span>
        ),
    },
  ];

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-slate-800">Dataset Overview</h2>

      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Records" value={data.records - data.revoked.records} delta={data.revoked.records} />
        <MetricCard label="Sections" value={data.sections - data.revoked.sections} delta={data.revoked.sections} />
        <MetricCard label="Authors" value={data.authors - data.revoked.authors} delta={data.revoked.authors} />
        <MetricCard label="Contributors" value={data.contributors - data.revoked.authors} delta={data.revoked.authors} />
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-5">
        <h3 className="text-sm font-semibold text-slate-700 mb-3">
          Author Contribution &mdash; Top {chartData.length} Authors
        </h3>
        <ResponsiveContainer width="100%" height={420}>
          <BarChart data={chartData}>
            <XAxis dataKey="name" tick={false} />
            <YAxis />
            <Tooltip
              formatter={(value) => [Number(value).toLocaleString(), 'Sections']}
            />
            <Bar dataKey="sections" fill="#2563eb" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h3 className="text-lg font-semibold text-slate-800 mb-3">Author Table</h3>
        <DataTable
          columns={columns}
          data={authorsData?.items || []}
          total={authorsData?.total || 0}
          page={page}
          limit={limit}
          onPageChange={setPage}
          onLimitChange={(l) => { setLimit(l); setPage(1); }}
          onRowClick={(row) => navigate(`/authors?id=${row.id}`)}
        />
      </div>
    </div>
  );
}
