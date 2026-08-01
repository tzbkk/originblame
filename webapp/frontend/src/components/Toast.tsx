import { useState, useCallback, createContext, useContext, type ReactNode } from 'react';

interface ToastCtx {
  toast: (msg: string, type?: 'success' | 'error' | 'info') => void;
}

const ToastContext = createContext<ToastCtx>({ toast: () => {} });
export const useToast = () => useContext(ToastContext);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<
    Array<{ id: number; msg: string; type: string }>
  >([]);

  const toast = useCallback(
    (msg: string, type: 'success' | 'error' | 'info' = 'success') => {
      const id = Date.now();
      setToasts((t) => [...t, { id, msg, type }]);
      setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3500);
    },
    [],
  );

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div className="fixed top-4 right-4 z-50 flex flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`rounded-lg px-4 py-3 shadow-lg text-sm font-medium text-white transition-all ${
              t.type === 'error'
                ? 'bg-red-500'
                : t.type === 'info'
                  ? 'bg-blue-600'
                  : 'bg-green-500'
            }`}
          >
            {t.msg}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
