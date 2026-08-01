# OriginBlame Webapp — Frontend

React + TypeScript + Vite application for the OriginBlame provenance demo.

## Development

```bash
cd webapp/frontend
npm install
npm run dev
```

## Pages

- **Dataset Overview** — summary statistics, top-50 author contribution chart
- **Authors** — searchable author table with paginated record preview
- **Records** — filterable record browser with provenance detail panel
- **Right-to-Erasure** — multi-level revocation (author / section / record) with confirmation dialogs
- **Undo** — restore revoked authors and sections
- **Reset** — reset demo state to initial conditions

## Build

```bash
npm run build   # outputs to dist/
```

## API Proxy

Dev server proxies `/api` to `http://localhost:8000` (see `vite.config.ts`).
