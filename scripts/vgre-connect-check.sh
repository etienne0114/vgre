#!/usr/bin/env bash
# vgre-connect-check — verify WAN/LAN connectivity to a VGRE master before starting a worker.
#
# Usage:
#   vgre-connect-check <HOST:PORT>
#   vgre-connect-check <HOST:PORT>
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
    _tcp_err="timeout"
    if command -v nc >/dev/null 2>&1; then
        _nc_out=$(nc -z -G 5 -w 5 "$HOST" "$MPORT" 2>&1) || true
        if echo "$_nc_out" | grep -qi "refused"; then
            _tcp_err="refused"
        fi
    fi

    echo "[FAIL] TCP $HOST:$MPORT is NOT reachable."
    if [[ "$_tcp_err" == "refused" ]]; then
        echo "       Connection refused — host is reachable but nothing accepts port $MPORT."
        echo "       On master: vgre-start --master  &&  ss -tlnp | grep $MPORT"
    else
        echo "       Connection timed out — usually router/NAT (port-forward) or ISP CGNAT."
        echo "       Master may work on LAN (192.168.x.x) but public IP is not forwarded."
    fi
    echo ""
    echo "The worker cannot join until inbound TCP $MPORT reaches the master."
    echo ""
    echo "On the LINUX MASTER, run these checks:"
    echo "  ss -tlnp | grep $MPORT                 # must show 0.0.0.0:$MPORT LISTEN"
    echo "  hostname -I | awk '{print \$1}'          # LAN IP for router port-forward"
    echo "  curl -4 -s ifconfig.me                 # confirm public IP matches $HOST"
    echo "  vgre-discover --register               # optional cross-LAN lookup"
    echo "  vgre-token fingerprint                 # must match worker"
    echo ""
    echo "Router (required unless using Tailscale):"
    echo "  Port-forward TCP $MPORT → <master LAN IP>"
    echo "  On master: hostname -I | awk '{print \$1}'"
    echo ""
    echo "Full Linux checklist:"
    echo "  vgre-print-linux-setup --public-ip $HOST"
    echo ""
    echo "Tip: Tailscale on both nodes — no port-forward.  brew install --cask tailscale"
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
