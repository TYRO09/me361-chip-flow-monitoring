#!/usr/bin/env bash
# Self-signed certificate for the laptop's LAN IP (phones only allow camera/mic on HTTPS).
# The phone shows a one-time "not private" warning: tap Advanced -> continue.
set -euo pipefail
cd "$(dirname "$0")"
IP=$(python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("10.255.255.255", 1)); print(s.getsockname()[0])
except OSError:
    print("127.0.0.1")
PY
)
mkdir -p certs
openssl req -x509 -newkey rsa:2048 -nodes -days 30 -keyout certs/key.pem -out certs/cert.pem \
  -subj "/CN=chip-monitor" -addext "subjectAltName=IP:${IP},IP:127.0.0.1,DNS:localhost" 2>/dev/null
echo "certificate written to app/certs for ${IP} (valid 30 days)"
