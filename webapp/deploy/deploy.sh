#!/bin/bash
# One-command deployment: sudo bash deploy.sh /path/to/data/dir
# Installs the OriginBlame webapp (FastAPI backend + React frontend) behind Nginx.
set -euo pipefail

INSTALL_DIR=/opt/originblame-demo
WEBAPP_DIR="${INSTALL_DIR}/webapp"
NGINX_CONF=/etc/nginx/sites-available/originblame
NGINX_LINK=/etc/nginx/sites-enabled/originblame
SERVICE_NAME=webapp
SERVICE_FILE=/etc/systemd/system/${SERVICE_NAME}.service

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: sudo bash deploy.sh /path/to/data/dir"
    echo ""
    echo "  /path/to/data/dir  Directory containing .ob/ (e.g. benchmarks/results/.../zhwiki-*-ob)"
    exit 1
fi

DATA_DIR="$(cd "$1" && pwd)"

if [ ! -d "${DATA_DIR}/.ob" ]; then
    echo "Error: ${DATA_DIR}/.ob does not exist."
    exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "Error: This script must be run as root (sudo)."
    exit 1
fi

echo "=== OriginBlame Webapp Deployment ==="
echo "Data directory : ${DATA_DIR}"
echo "Install target : ${INSTALL_DIR}"
echo ""

# 1. Install system dependencies
echo "[1/7] Installing dependencies..."
apt-get update -qq
apt-get install -y -qq nginx python3 python3-pip python3-venv nodejs npm > /dev/null
echo "       Done."

# 2. Copy webapp to install directory
echo "[2/7] Installing webapp to ${INSTALL_DIR}..."
mkdir -p "${INSTALL_DIR}"
cp -r "${SCRIPT_DIR}/.." "${WEBAPP_DIR}"

# Symlink the ob Python package if available
OB_PYTHON_SRC="${SCRIPT_DIR}/../../rust-originblame/python/src"
if [ -d "${OB_PYTHON_SRC}" ]; then
    echo "${WEBAPP_DIR}/../../rust-originblame/python/src" > "${WEBAPP_DIR}/backend/.pth"
fi
echo "       Done."

# 3. Build frontend
echo "[3/7] Building React frontend..."
cd "${WEBAPP_DIR}/frontend"
npm ci --quiet 2>/dev/null || npm install --quiet 2>/dev/null
npm run build
echo "       Done."

# 4. Setup Python backend
echo "[4/7] Setting up Python backend..."
python3 -m venv "${WEBAPP_DIR}/backend/.venv"
"${WEBAPP_DIR}/backend/.venv/bin/pip" install --quiet -r "${WEBAPP_DIR}/backend/requirements.txt"
echo "       Done."

# 5. Configure OB_DIR
echo "[5/7] Configuring OB_DIR=${DATA_DIR}..."
mkdir -p /etc/default
cat > /etc/default/webapp <<EOF
OB_DIR=${DATA_DIR}
EOF
echo "       Done."

# 6. Install Nginx config + systemd service
echo "[6/7] Configuring Nginx + systemd..."
cp "${SCRIPT_DIR}/nginx.conf" "${NGINX_CONF}"
ln -sf "${NGINX_CONF}" "${NGINX_LINK}"

if [ -f /etc/nginx/sites-enabled/default ]; then
    rm -f /etc/nginx/sites-enabled/default
fi
nginx -t
systemctl enable nginx
systemctl restart nginx

cp "${SCRIPT_DIR}/webapp.service" "${SERVICE_FILE}"
sed -i '/^\[Service\]/a EnvironmentFile=/etc/default/webapp' "${SERVICE_FILE}"
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"
echo "       Done."

# 7. Verify
echo "[7/7] Verifying..."
sleep 2
if systemctl is-active --quiet "${SERVICE_NAME}"; then
    STATUS="running"
else
    STATUS="FAILED — check: journalctl -u ${SERVICE_NAME}"
fi

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "  Webapp  : http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo 'YOUR_SERVER_IP')"
echo "  Service : systemctl status ${SERVICE_NAME}"
echo "  Logs    : journalctl -u ${SERVICE_NAME} -f"
echo "  Status  : ${STATUS}"
echo ""
echo "To add HTTPS:"
echo "  sudo apt install certbot python3-certbot-nginx"
echo "  sudo certbot --nginx -d your-domain.com"
