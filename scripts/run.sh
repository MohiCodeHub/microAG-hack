#!/usr/bin/env bash
# Usage: scripts/run.sh [--fresh]   --fresh first wipes stored frame hashes (scripts/reset_data.sh)
# Set LAN_IP=... to pick the address the phone should use.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" = "--fresh" ]; then
  scripts/reset_data.sh
fi
LAN_IP="$(scripts/lan_ip.sh)"

# True when the cert exists, is valid for another hour, and was issued for $LAN_IP.
cert_ok() {
  [ -f data/cert.pem ] && [ -f data/key.pem ] || return 1
  openssl x509 -in data/cert.pem -noout -checkend 3600 >/dev/null || return 1
  local text
  text="$(openssl x509 -in data/cert.pem -noout -text)"
  grep -Eq "IP Address:${LAN_IP//./\\.}(,|\$)" <<<"$text"
}
cert_ok || scripts/make_cert.sh "$LAN_IP"  # e.g. first run, or a new Wi-Fi gave a new IP

echo "Phone:     https://${LAN_IP}:8443/phone"
echo "Dashboard: https://localhost:8443/"
exec uv run uvicorn app.server:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile data/key.pem --ssl-certfile data/cert.pem
