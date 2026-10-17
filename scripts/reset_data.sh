#!/usr/bin/env bash
# Forget the frame hashes of earlier sessions (rehearsals), so the duplicate-footage check
# starts clean. Keeps the HTTPS certificate.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f data/hashes.json
echo "cleared data/hashes.json"
