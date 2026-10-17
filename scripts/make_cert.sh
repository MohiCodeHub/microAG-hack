#!/usr/bin/env bash
# Self-signed cert valid for localhost and this machine's LAN IP (phone needs HTTPS for camera/mic).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
LAN_IP="$(ipconfig getifaddr en0)"
openssl req -x509 -newkey rsa:2048 -nodes -days 7 \
  -keyout data/key.pem -out data/cert.pem -subj "/CN=shift-agent" \
  -addext "subjectAltName=IP:${LAN_IP},DNS:localhost"
echo "cert for ${LAN_IP} written to data/cert.pem"
