#!/usr/bin/env bash
# Prints this machine's LAN IP: $LAN_IP if set, else the first macOS interface (en0, en1, ...)
# with an address. Exits non-zero with a hint when none is found.
set -euo pipefail
if [ -n "${LAN_IP:-}" ]; then
  echo "$LAN_IP"
  exit 0
fi
for i in 0 1 2 3 4 5 6 7 8 9; do
  ip="$(ipconfig getifaddr "en$i" 2>/dev/null || true)"
  if [ -n "$ip" ]; then
    echo "$ip"
    exit 0
  fi
done
echo "error: no LAN IP found on en0-en9. Join the Wi-Fi the phone uses, or set it by hand:" >&2
echo "  LAN_IP=192.168.1.23 scripts/run.sh" >&2
exit 1
