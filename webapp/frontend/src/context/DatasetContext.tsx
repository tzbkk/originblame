import {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useDatasets } from '../api/hooks';

interface DatasetCtx {
  dataset: string;
  setDataset: (ds: string) => void;
  datasets: string[];
}

const Ctx = createContext<DatasetCtx>({
  dataset: '',
  setDataset: () => {},
  datasets: [],
});

export function DatasetProvider({ children }: { children: ReactNode }) {
  const [dataset, setDatasetRaw] = useState('');
  const qc = useQueryClient();
  const { data } = useDatasets();

  const resolved = dataset || data?.default || '';
  if (!dataset && data?.default) setDatasetRaw(data.default);

  const setDataset = useCallback(
    (ds: string) => {
      setDatasetRaw(ds);
      qc.invalidateQueries();
    },
    [qc],
  );

  return (
    <Ctx.Provider
      value={{ dataset: resolved, setDataset, datasets: data?.datasets || [] }}
    >
      {children}
    </Ctx.Provider>
  );
}

export const useDataset = () => useContext(Ctx);
