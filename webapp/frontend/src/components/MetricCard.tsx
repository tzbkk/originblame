interface MetricCardProps {
  label: string;
  value: string | number;
  delta?: number;
}

export default function MetricCard({ label, value, delta }: MetricCardProps) {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-5">
      <p className="text-sm font-medium text-slate-500">{label}</p>
      <p className="text-2xl font-bold text-slate-800 mt-1">
        {typeof value === 'number' ? value.toLocaleString() : value}
      </p>
      {delta != null && delta !== 0 && (
        <p className="text-sm font-medium text-red-500 mt-1">
          {delta.toLocaleString()} revoked
        </p>
      )}
    </div>
  );
}
