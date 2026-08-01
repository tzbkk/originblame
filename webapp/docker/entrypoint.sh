#!/bin/bash
set -euo pipefail

nginx -g "daemon off;" &
NGINX_PID=$!

cd /app/backend
exec uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
