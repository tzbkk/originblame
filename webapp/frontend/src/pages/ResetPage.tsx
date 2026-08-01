import { useNavigate } from 'react-router-dom';
import { useDataset } from '../context/DatasetContext';
import { useResetDemo } from '../api/hooks';
import { useToast } from '../components/Toast';

export default function ResetPage() {
  const { dataset } = useDataset();
  const { toast } = useToast();
  const navigate = useNavigate();
  const reset = useResetDemo(dataset);

  const handleReset = () => {
    reset.mutate(undefined, {
      onSuccess: () => {
        toast('Demo state reset. All revocations undone.');
        navigate('/');
      },
      onError: (err) => toast(err.message, 'error'),
    });
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-slate-800">Reset Demo</h2>
        <p className="text-sm text-slate-500 mt-1">
          Restore .ob/ to pre-revocation state from backup (wipes all changes
          since first revoke).
        </p>
      </div>

      <button
        onClick={handleReset}
        disabled={reset.isPending}
        className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-6 py-3 rounded-md text-sm transition-colors disabled:opacity-50"
      >
        {reset.isPending ? 'Resetting...' : 'Reset Demo State'}
      </button>
    </div>
  );
}
