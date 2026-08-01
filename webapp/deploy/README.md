# Deploy the OriginBlame Webapp

One-command deployment to a Linux server with Nginx + systemd.

## Prerequisites

- Ubuntu/Debian server with sudo access
- A directory containing `.ob/` provenance data (from running the benchmark pipeline)
- (Optional) A domain name pointing to your server

## Quick Deploy

```bash
cd webapp/deploy
sudo bash deploy.sh /path/to/benchmarks/results/pipeline_v2/huggingface-zhwiki-*-ob
```

This will:
1. Install Nginx, Python 3, Node.js, npm
2. Copy the webapp to `/opt/originblame-demo/webapp`
3. Build the React frontend (`npm run build`)
4. Create a Python venv and install FastAPI backend dependencies
5. Set `OB_DIR` to your data directory
6. Configure Nginx as a reverse proxy (static frontend + `/api/` → backend)
7. Start the FastAPI backend as a systemd service

## After Deploy

- Open `http://YOUR_SERVER_IP` in a browser
- Check service status: `systemctl status webapp`
- View logs: `journalctl -u webapp -f`

## HTTPS (Optional)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

Before running certbot, update `server_name _` in `nginx.conf` to your domain.

## Architecture

```
Browser → Nginx (port 80)
  ├── /          → static React frontend (dist/)
  └── /api/*     → FastAPI backend (uvicorn, port 8000)
```

Backend reads `.ob/` provenance data from the directory specified in `/etc/default/webapp`.

## Updating

To update the webapp after code changes:
```bash
sudo systemctl stop webapp
cd webapp/deploy
sudo bash deploy.sh /path/to/data/dir
```
