#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f data/cert.pem ] && [ -f data/key.pem ] || scripts/make_cert.sh
LAN_IP="$(ipconfig getifaddr en0)"
echo "Phone:     https://${LAN_IP}:8443/phone"
echo "Dashboard: https://localhost:8443/"
exec uv run uvicorn app.server:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile data/key.pem --ssl-certfile data/cert.pem
