#!/usr/bin/env bash
# VGRE Start — launch a master or worker node with one command.
#
# Prerequisites: run install_local.sh once (generates ~/.vgre/env automatically).
#
# Usage:
#   vgre-start --master                          Start master (runs the dashboard)
#   vgre-start --worker                          Start worker (auto-discovers master)
#   vgre-start --worker --master-ip 10.0.0.5    Start worker, connect to specific master
#   vgre-start --worker --port 7778             Start worker on a non-default port
#   vgre-start --test                           Quick local test: master + worker on same machine

set -e

# ── Auto-source environment file (set by install_local.sh) ───────────────────
# This makes all VGRE env vars available without the user having to run
# "source ~/.vgre/env" or "export ..." manually.
VGRE_ENV_FILE="${VGRE_ENV_FILE:-$HOME/.vgre/env}"
if [[ -f "$VGRE_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$VGRE_ENV_FILE"
fi

# Add the installed lib directory to the dynamic linker search path so the
# worker binary can find libvgre*.so / libvgre*.dylib without manual setup.
_VGRE_LIB_DIR="${VGRE_INSTALL_DIR:-$HOME/.local/share/VGRE}/lib"
if [[ -d "$_VGRE_LIB_DIR" ]]; then
    export LD_LIBRARY_PATH="$_VGRE_LIB_DIR:${LD_LIBRARY_PATH:-}"       # Linux
    [[ "$(uname -s)" == "Darwin" ]] && \
        export DYLD_LIBRARY_PATH="$_VGRE_LIB_DIR:${DYLD_LIBRARY_PATH:-}" # macOS
fi

INSTALL_DIR="${VGRE_INSTALL_DIR:-$HOME/.local/share/VGRE}"
TOKEN_FILE="${VGRE_TCP_AUTH_TOKEN_FILE:-$HOME/.vgre/token}"
PORT="${VGRE_PORT:-7777}"  # Must match kDefaultClusterPort in tcp_cluster_defaults.h
MODE=""
MASTER_IP=""
MASTER_ADDRESS=""
EXTRA_ARGS=()

# ── Parse Arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --master)           MODE="master" ;;
        --worker)           MODE="worker" ;;
        --test)             MODE="test" ;;
        --master-ip)        MASTER_IP="$2"; shift ;;
        --master-address)   MASTER_ADDRESS="$2"; shift ;;
        --port)             PORT="$2"; shift ;;
        --threads)          EXTRA_ARGS+=("--threads" "$2"); shift ;;
        --help|-h)
            cat <<'EOF'
vgre-start -- VGRE Cluster Launcher

Usage:
  vgre-start --master                          Start master + dashboard
  vgre-start --worker                          Start worker (LAN auto-discover)
  vgre-start --worker --master-ip <IP>         LAN: connect to specific master IP
  vgre-start --worker --master-address <H:P>   WAN: hostname/IPv4/IPv6 + port
  vgre-start --test                            Local self-test (master+worker)

Options:
  --port <N>       TCP port (default 7777)
  --threads <N>    Worker thread count (default: auto)
EOF
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1  (run vgre-start --help)"
            exit 1
            ;;
    esac
    shift
done

if [[ -z "$MODE" ]]; then
    echo "Usage: vgre-start --master | --worker [--master-ip IP] [--port PORT]"
    echo "       vgre-start --test   (local self-test, master + worker on same machine)"
    exit 1
fi

# ── Load Auth Token ───────────────────────────────────────────────────────────
if [[ -f "$TOKEN_FILE" ]]; then
    export VGRE_TCP_AUTH_TOKEN_FILE="$TOKEN_FILE"
elif [[ -n "$VGRE_TCP_AUTH_TOKEN" ]]; then
    : # already in env — fine
else
    echo ""
    echo "❌ No auth token found."
    echo "   Run:  bash scripts/setup-cluster.sh"
    echo "   Or:   export VGRE_TCP_AUTH_TOKEN=your-token"
    echo ""
    exit 1
fi

# Show SHA256 fingerprint of the active token so users can verify master/worker match
_ACTIVE_TOKEN=""
if [[ -f "$TOKEN_FILE" ]]; then
    _ACTIVE_TOKEN=$(cat "$TOKEN_FILE" 2>/dev/null || true)
elif [[ -n "$VGRE_TCP_AUTH_TOKEN" ]]; then
    _ACTIVE_TOKEN="$VGRE_TCP_AUTH_TOKEN"
fi
if [[ -n "$_ACTIVE_TOKEN" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
        _FP=$(printf '%s' "$_ACTIVE_TOKEN" | sha256sum | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
        _FP=$(printf '%s' "$_ACTIVE_TOKEN" | shasum -a 256 | awk '{print $1}')
    else
        _FP="(sha256sum unavailable)"
    fi
    echo "🔑 Token fingerprint (SHA256): ${_FP:0:16}..."
    echo "   (master and worker MUST show the same fingerprint)"
    echo ""
fi

# ── Library Path ──────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"              # Linux
[[ "$(uname -s)" == "Darwin" ]] && \
    export DYLD_LIBRARY_PATH="$INSTALL_DIR/lib:${DYLD_LIBRARY_PATH:-}"      # macOS

# Resolve the worker binary flexibly: installed locations (both layouts that
# past installers used), PATH, then the repo build tree for development.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_BIN=""
for cand in \
    "$INSTALL_DIR/vgre-worker" \
    "$INSTALL_DIR/bin/vgre-worker" \
    "$(command -v vgre-worker 2>/dev/null || true)" \
    "$_SCRIPT_DIR/../build/src/advanced/vgre-worker"; do
    [[ -n "$cand" && -x "$cand" ]] && WORKER_BIN="$cand" && break
done
if [[ -z "$WORKER_BIN" ]]; then
    echo "❌ vgre-worker not found (looked in $INSTALL_DIR, PATH, and the repo build tree)."
    echo "   Run install_local.sh (or scripts/vgre_sync.sh) to build and install first."
    exit 1
fi

# Resolve the dashboard launcher (Linux bundle or macOS .app).
_resolve_dashboard() {
    local cand
    for cand in \
        "$INSTALL_DIR/vgre-launch.sh" \
        "$INSTALL_DIR/vgre_dashboard" \
        "$INSTALL_DIR/vgre_dashboard.app/Contents/MacOS/vgre_dashboard"; do
        [[ -x "$cand" ]] && echo "$cand" && return 0
    done
    return 1
}

# Portable one-shot ping: -W is seconds on Linux but milliseconds on macOS,
# where -t (overall timeout, seconds) is the equivalent.
_ping_ok() {
    if [[ "$(uname -s)" == "Darwin" ]]; then
        ping -c 1 -t 2 "$1" >/dev/null 2>&1
    else
        ping -c 1 -W 2 "$1" >/dev/null 2>&1
    fi
}

# ── Start ─────────────────────────────────────────────────────────────────────
case "$MODE" in

    master)
        # Auto-detect real public IP so master broadcasts with it in UDP pings.
        # Workers on different LANs will receive the correct public address.
        _DISCOVER="$(command -v vgre-discover 2>/dev/null || true)"
        [[ -z "$_DISCOVER" ]] && [[ -x "$INSTALL_DIR/vgre-discover" ]] && _DISCOVER="$INSTALL_DIR/vgre-discover"
        [[ -z "$_DISCOVER" ]] && [[ -x "$(dirname "$0")/vgre-discover.sh" ]] && _DISCOVER="$(dirname "$0")/vgre-discover.sh"
        if [[ -n "$_DISCOVER" ]]; then
            echo "[...] Detecting public IP for WAN broadcast..."
            # Run in a sub-shell so its 'export' doesn't affect this script's env;
            # instead capture the printed assignment and eval it.
            _ADV=$(bash "$_DISCOVER" --set-master 2>/dev/null | grep "^VGRE_CLUSTER_ADVERTISED_ADDRESS=" || true)
            [[ -n "$_ADV" ]] && export "$_ADV" && echo "[OK] $_ADV"
        fi

        echo "Starting VGRE Master Node..."
        echo "  Port:  $PORT"
        echo "  Token: $TOKEN_FILE"
        echo ""
        export VGRE_PORT="$PORT"
        DASHBOARD_BIN="$(_resolve_dashboard)" || {
            echo "❌ Dashboard not found in $INSTALL_DIR. Run install_local.sh first."
            exit 1
        }
        exec "$DASHBOARD_BIN"
        ;;

    worker)
        if [[ -n "$MASTER_ADDRESS" ]]; then
            # WAN / explicit hostname:port — handled by getaddrinfo in C++ engine
            export VGRE_CLUSTER_MASTER_ADDRESS="$MASTER_ADDRESS"
            echo "Starting VGRE Worker → master at $MASTER_ADDRESS (WAN)"
        elif [[ -n "$MASTER_IP" ]]; then
            # Legacy LAN shorthand: only an IP was given, append port
            export VGRE_CLUSTER_NODES="$MASTER_IP:$PORT"
            export VGRE_CLUSTER_MASTER_ADDRESS="$MASTER_IP:$PORT"
            # Verify the master IP is reachable before starting the worker.
            if command -v ping >/dev/null 2>&1; then
                if ! _ping_ok "$MASTER_IP"; then
                    echo "[WARN] Master IP $MASTER_IP is not reachable (ping timed out)."
                    echo "       Check the IP address and firewall rules."
                fi
            fi
            echo "Starting VGRE Worker → master at $MASTER_IP:$PORT"
        else
            echo "Starting VGRE Worker (auto-discovering master on local subnet)..."
        fi
        echo "  Port:  $PORT"
        echo "  Token: $TOKEN_FILE"
        echo ""
        exec "$WORKER_BIN" --port "$PORT" "${EXTRA_ARGS[@]}"
        ;;

    test)
        echo "Starting local self-test (master + worker on same machine)..."
        echo "  Token: $TOKEN_FILE"
        echo "  Press Ctrl+C to stop."
        echo ""
        # Start worker in background
        "$WORKER_BIN" --port "$PORT" "${EXTRA_ARGS[@]}" &
        WORKER_PID=$!
        trap 'echo "Stopping test..."; kill $WORKER_PID 2>/dev/null; exit 0' INT TERM
        echo "Worker started (PID $WORKER_PID). Starting master dashboard..."
        sleep 1
        export VGRE_PORT="$PORT"
        DASHBOARD_BIN="$(_resolve_dashboard)" || {
            echo "❌ Dashboard not found in $INSTALL_DIR. Run install_local.sh first."
            kill $WORKER_PID 2>/dev/null || true
            exit 1
        }
        "$DASHBOARD_BIN" &
        MASTER_PID=$!
        trap 'echo "Stopping test..."; kill $WORKER_PID $MASTER_PID 2>/dev/null; exit 0' INT TERM
        wait $MASTER_PID
        kill $WORKER_PID 2>/dev/null || true
        ;;
esac
