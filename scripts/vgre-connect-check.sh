#!/usr/bin/env bash
# vgre-connect-check — verify WAN/LAN connectivity to a VGRE master before starting a worker.
#
# Usage:
#   vgre-connect-check <HOST:PORT>
#   vgre-connect-check 102.22.143.9:7777
#
# Exit 0 when TCP port is open; exit 1 with actionable diagnostics otherwise.

set -euo pipefail

ADDR="${1:-${VGRE_CLUSTER_MASTER_ADDRESS:-}}"
if [[ -z "$ADDR" ]]; then
    echo "Usage: vgre-connect-check <HOST:PORT>" >&2
    echo "   or: export VGRE_CLUSTER_MASTER_ADDRESS=HOST:PORT && vgre-connect-check" >&2
    exit 1
fi

HOST="${ADDR%%:*}"
MPORT="${ADDR##*:}"
[[ "$MPORT" == "$HOST" ]] && MPORT="${VGRE_PORT:-7777}"

TOKEN_FILE="${VGRE_TCP_AUTH_TOKEN_FILE:-$HOME/.vgre/token}"

echo "=== VGRE Master Connectivity Check ==="
echo "Target: $HOST:$MPORT"
echo ""

# ── 1. TCP port ───────────────────────────────────────────────────────────────
_tcp_open=0
if command -v nc >/dev/null 2>&1; then
    if nc -z -G 8 -w 8 "$HOST" "$MPORT" 2>/dev/null; then
        _tcp_open=1
    fi
elif command -v bash >/dev/null 2>&1; then
    if (echo >/dev/tcp/"$HOST"/"$MPORT") 2>/dev/null; then
        _tcp_open=1
    fi
fi

if [[ $_tcp_open -eq 1 ]]; then
    echo "[OK] TCP $HOST:$MPORT is reachable."
else
    echo "[FAIL] TCP $HOST:$MPORT is NOT reachable (connection timed out or refused)."
    echo ""
    echo "The worker cannot join until the master accepts inbound TCP on this port."
    echo ""
    echo "On the LINUX MASTER, run these checks:"
    echo "  vgre-start --master                    # or: vgre-worker --is-master --port $MPORT"
    echo "  ss -tlnp | grep $MPORT                 # must show 0.0.0.0:$MPORT or *:$MPORT LISTEN"
    echo "  curl -4 -s ifconfig.me                 # confirm public IP matches $HOST"
    echo "  vgre-token fingerprint                 # must match worker fingerprint"
    echo ""
    echo "Firewall / NAT (on master + router):"
    echo "  sudo ufw allow $MPORT/tcp"
    echo "  sudo ufw allow 7778/udp               # LAN discovery (optional)"
    echo "  Port-forward TCP $MPORT → master LAN IP on your router"
    echo ""
    echo "Tip: use Tailscale/ZeroTier — both nodes get routable IPs, no port-forward."
    exit 1
fi

# ── 2. Token (informational) ──────────────────────────────────────────────────
if [[ -f "$TOKEN_FILE" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
        _FP=$(tr -d '[:space:]' < "$TOKEN_FILE" | sha256sum | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
        _FP=$(tr -d '[:space:]' < "$TOKEN_FILE" | shasum -a 256 | awk '{print $1}')
    else
        _FP=""
    fi
    [[ -n "$_FP" ]] && echo "[OK] Local token fingerprint: ${_FP:0:16}... (verify same on master)"
else
    echo "[WARN] No token at $TOKEN_FILE"
fi

echo ""
echo "Connectivity looks good. Start the worker:"
echo "  vgre-start --worker --master-address $HOST:$MPORT"
exit 0
