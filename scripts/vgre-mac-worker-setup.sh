#!/usr/bin/env bash
# One-shot worker setup: CLI, env, discovery probe, connectivity check.
# Works on macOS and Linux worker nodes.
#
# Usage:
#   vgre-mac-worker-setup
#   vgre-mac-worker-setup <HOST:PORT>

set -euo pipefail

_vgre_entry="${BASH_SOURCE[0]}"
while [[ -L "$_vgre_entry" ]]; do
    _link=$(readlink "$_vgre_entry")
    case "$_link" in /*) _vgre_entry="$_link" ;; *) _vgre_entry="$(cd "$(dirname "$_vgre_entry")" && pwd)/$_link" ;; esac
done
SCRIPT_DIR="$(cd "$(dirname "$_vgre_entry")" && pwd)"
VGRE_DIR="${HOME}/.vgre"
ENV_FILE="${VGRE_DIR}/env"
MASTER_ADDR="${1:-}"
PORT="${VGRE_PORT:-7777}"

_is_private_ip() {
    local ip="$1"
    [[ "$ip" == "127.0.0.1" || "$ip" == "localhost" || "$ip" == "::1" ]] && return 0
    [[ "$ip" =~ ^10\. ]] && return 0
    [[ "$ip" =~ ^192\.168\. ]] && return 0
    [[ "$ip" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\. ]] && return 0
    return 1
}

_upsert_env() {
    local key="$1" val="$2"
    mkdir -p "$VGRE_DIR"
    touch "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    local tmp
    tmp=$(mktemp)
    if grep -q "^export ${key}=" "$ENV_FILE" 2>/dev/null; then
        grep -v "^export ${key}=" "$ENV_FILE" > "$tmp" || : > "$tmp"
    else
        cp "$ENV_FILE" "$tmp" 2>/dev/null || : > "$tmp"
    fi
    printf 'export %s="%s"\n' "$key" "$val" >> "$tmp"
    mv -f "$tmp" "$ENV_FILE"
}

echo "=== VGRE Worker Setup ==="
echo ""

# 1. CLI symlinks
# shellcheck source=vgre-cli-install.sh
. "$SCRIPT_DIR/vgre-cli-install.sh"
vgre_install_cli_symlinks "$SCRIPT_DIR"
vgre_ensure_cli_path
echo "[OK] CLI tools installed in $(vgre_cli_bin_dir)"

# 2. Load existing env before probing discovery
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

# 3. Cross-LAN kvdb lookup (works after master runs vgre-discover --register)
echo "[...] Probing cross-LAN discovery (kvdb.io)..."
if _found=$(vgre-discover --find-auto 2>/dev/null | grep "^VGRE_CLUSTER_MASTER_ADDRESS=" || true); then
    if [[ -n "$_found" ]]; then
        # shellcheck disable=SC2163
        export "$_found"
        echo "[OK] Master registered in kvdb: $VGRE_CLUSTER_MASTER_ADDRESS"
    fi
else
    echo "[INFO] No kvdb registration yet (master must run: vgre-discover --register)"
fi

# 4. Resolve master address: CLI arg > env > kvdb result
MASTER_ADDR="${MASTER_ADDR:-${VGRE_CLUSTER_MASTER_ADDRESS:-}}"
if [[ -z "$MASTER_ADDR" ]]; then
    echo ""
    echo "[ERROR] No master address configured."
    echo "  Pass:  vgre-mac-worker-setup <HOST:PORT>"
    echo "  Or set VGRE_CLUSTER_MASTER_ADDRESS in ~/.vgre/env"
    echo "  Or on master run: vgre-discover --register  (same token on both nodes)"
    exit 1
fi

# 5. WAN-friendly env
_upsert_env VGRE_PORT "$PORT"
_upsert_env VGRE_CLUSTER_CONNECT_TIMEOUT_SEC "${VGRE_CLUSTER_CONNECT_TIMEOUT_SEC:-30}"
_upsert_env VGRE_INSTALL_DIR "${VGRE_INSTALL_DIR:-$HOME/.local/share/VGRE}"
_upsert_env VGRE_TCP_AUTH_TOKEN_FILE "${VGRE_TCP_AUTH_TOKEN_FILE:-$VGRE_DIR/token}"
if _is_private_ip "${MASTER_ADDR%%:*}"; then
    _upsert_env VGRE_CLUSTER_MASTER_ADDRESS "$MASTER_ADDR"
else
    echo "[INFO] WAN address not saved to ~/.vgre/env (default worker mode is LAN)."
    echo "       Use: vgre-start --worker --wan --master-address $MASTER_ADDR"
fi
echo "[OK] Updated $ENV_FILE"

# shellcheck disable=SC1090
source "$ENV_FILE"

# 6. Worker binary
WORKER="${VGRE_INSTALL_DIR}/vgre-worker"
if [[ ! -x "$WORKER" ]]; then
    WORKER="$(command -v vgre-worker 2>/dev/null || true)"
fi
if [[ -z "$WORKER" || ! -x "$WORKER" ]]; then
    echo "[WARN] vgre-worker not found — run: bash scripts/vgre_sync.sh"
else
    echo "[OK] Worker binary: $WORKER"
fi

# 7. Token check
if [[ ! -f "$VGRE_DIR/token" ]]; then
    echo "[WARN] No token at $VGRE_DIR/token — run: vgre-token generate"
else
    echo "[OK] Token file present"
fi

# 8. Connectivity
echo ""
if command -v vgre-connect-check >/dev/null 2>&1; then
    vgre-connect-check "$MASTER_ADDR" || {
        echo ""
        bash "$SCRIPT_DIR/vgre-print-linux-setup.sh" --public-ip "${MASTER_ADDR%%:*}" || true
        exit 1
    }
else
    echo "[WARN] vgre-connect-check not on PATH"
fi

echo ""
echo "Start worker:"
echo "  vgre-start --worker --master-address $MASTER_ADDR"
