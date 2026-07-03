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

# Resolve scripts/ even when invoked via ~/.local/bin/vgre-start symlink.
_vgre_entry="${BASH_SOURCE[0]}"
while [[ -L "$_vgre_entry" ]]; do
    _vgre_link=$(readlink "$_vgre_entry")
    case "$_vgre_link" in
        /*) _vgre_entry="$_vgre_link" ;;
        *) _vgre_entry="$(cd "$(dirname "$_vgre_entry")" && pwd)/$_vgre_link" ;;
    esac
done
SCRIPT_DIR="$(cd "$(dirname "$_vgre_entry")" && pwd)"

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
SKIP_CONNECT_CHECK=0
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
        --skip-connect-check) SKIP_CONNECT_CHECK=1 ;;
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
  --skip-connect-check  Skip TCP preflight (worker WAN mode)
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
WORKER_BIN=""
for cand in \
    "$INSTALL_DIR/vgre-worker" \
    "$INSTALL_DIR/bin/vgre-worker" \
    "$(command -v vgre-worker 2>/dev/null || true)" \
    "$SCRIPT_DIR/../build/src/advanced/vgre-worker"; do
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

_resolve_discover_script() {
    local cand
    for cand in \
        "$(command -v vgre-discover 2>/dev/null || true)" \
        "$INSTALL_DIR/vgre-discover" \
        "$SCRIPT_DIR/vgre-discover.sh"; do
        [[ -n "$cand" && -x "$cand" ]] && { echo "$cand"; return 0; }
    done
    return 1
}

# Cross-LAN: workers with the same token look up the master's public IP via kvdb.io.
_try_cross_lan_master_discovery() {
    [[ -n "$MASTER_ADDRESS" || -n "$MASTER_IP" ]] && return 0
    [[ -n "${VGRE_CLUSTER_MASTER_ADDRESS:-}" ]] && return 0

    local _discover
    _discover="$(_resolve_discover_script)" || return 0

    echo "[...] Cross-LAN discovery: looking up master (token-keyed kvdb.io)..."
    local _found
    _found=$(bash "$_discover" --find-auto 2>/dev/null | grep "^VGRE_CLUSTER_MASTER_ADDRESS=" || true)
    if [[ -n "$_found" ]]; then
        # shellcheck disable=SC2163
        export "$_found"
        MASTER_ADDRESS="$VGRE_CLUSTER_MASTER_ADDRESS"
        echo "[OK] Master found via cross-LAN discovery: $MASTER_ADDRESS"
        return 0
    fi

    echo "[INFO] Cross-LAN: no registered master yet."
    echo "       On the Linux master run:  vgre-start --master"
    echo "       (registers public IP automatically; same token required)"
    echo "       Falling back to LAN UDP broadcast (same subnet only)..."
    echo ""
    return 0
}

# Fail fast when the master TCP port is unreachable (WAN / explicit address modes).
_check_master_tcp() {
    local addr="$1"
    [[ -z "$addr" ]] && return 0
    local host="${addr%%:*}"
    local mport="${addr##*:}"
    [[ "$mport" == "$host" ]] && mport="$PORT"

    local check_script=""
    for cand in \
        "$(command -v vgre-connect-check 2>/dev/null || true)" \
        "$SCRIPT_DIR/vgre-connect-check.sh"; do
        [[ -n "$cand" && -x "$cand" ]] && { check_script="$cand"; break; }
    done

    if [[ -n "$check_script" ]]; then
        VGRE_CLUSTER_MASTER_ADDRESS="$host:$mport" bash "$check_script" "$host:$mport"
        return $?
    fi

    if command -v nc >/dev/null 2>&1; then
        if nc -z -G 8 -w 8 "$host" "$mport" 2>/dev/null; then
            echo "[OK] Master TCP $host:$mport is reachable."
            return 0
        fi
    fi

    echo ""
    echo "❌ Cannot reach master at $host:$mport (TCP timeout/refused)."
    echo "   Run: vgre-connect-check $host:$mport"
    echo "   On Linux master: vgre-start --master  &&  ss -tlnp | grep $mport"
    echo "   Open firewall: sudo ufw allow $mport/tcp"
    echo "   Port-forward TCP $mport on your router to the master machine."
    return 1
}

# ── Start ─────────────────────────────────────────────────────────────────────
case "$MODE" in

    master)
        # Auto-detect real public IP so master broadcasts with it in UDP pings.
        # Workers on different LANs will receive the correct public address.
        _DISCOVER="$(_resolve_discover_script || true)"
        if [[ -n "$_DISCOVER" ]]; then
            echo "[...] Detecting public IP for WAN broadcast..."
            # Run in a sub-shell so its 'export' doesn't affect this script's env;
            # instead capture the printed assignment and eval it.
            _ADV=$(bash "$_DISCOVER" --set-master 2>/dev/null | grep "^VGRE_CLUSTER_ADVERTISED_ADDRESS=" || true)
            [[ -n "$_ADV" ]] && export "$_ADV" && echo "[OK] $_ADV"
            echo "[...] Registering master for cross-LAN workers (token-keyed kvdb.io)..."
            if bash "$_DISCOVER" --register 2>/dev/null; then
                echo "[OK] Master registered for cross-LAN discovery (same token on workers)."
            else
                echo "[WARN] Cross-LAN registration skipped (check internet / kvdb.io)."
                echo "       Workers on other networks need: vgre-start --worker --master-address <IP>:$PORT"
            fi
            echo ""
        fi

        echo "Starting VGRE Master Node..."
        echo "  Port:  $PORT"
        echo "  Token: $TOKEN_FILE"
        echo ""
        export VGRE_PORT="$PORT"
        DASHBOARD_BIN="$(_resolve_dashboard)" || true
        if [[ -z "$DASHBOARD_BIN" ]] && [[ -x "$WORKER_BIN" ]]; then
            echo "[WARN] Dashboard not found — starting headless master (vgre-worker --is-master)."
            export VGRE_PORT="$PORT"
            exec "$WORKER_BIN" --is-master --port "$PORT" "${EXTRA_ARGS[@]}"
        fi
        DASHBOARD_BIN="${DASHBOARD_BIN:?Dashboard not found in $INSTALL_DIR. Run install_local.sh or vgre_sync.sh first.}"
        exec "$DASHBOARD_BIN"
        ;;

    worker)
        _try_cross_lan_master_discovery
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
        elif [[ -n "${VGRE_CLUSTER_MASTER_ADDRESS:-}" ]]; then
            echo "Starting VGRE Worker → master at $VGRE_CLUSTER_MASTER_ADDRESS (configured)"
        else
            echo "Starting VGRE Worker (auto-discovering master on local subnet)..."
        fi
        echo "  Port:  $PORT"
        echo "  Token: $TOKEN_FILE"
        echo ""

        # WAN links: longer connect timeout unless user already set one.
        [[ -z "${VGRE_CLUSTER_CONNECT_TIMEOUT_SEC:-}" ]] && \
            export VGRE_CLUSTER_CONNECT_TIMEOUT_SEC=30

        _TARGET=""
        [[ -n "$MASTER_ADDRESS" ]] && _TARGET="$MASTER_ADDRESS"
        [[ -z "$_TARGET" && -n "$MASTER_IP" ]] && _TARGET="${MASTER_IP}:${PORT}"
        [[ -z "$_TARGET" && -n "${VGRE_CLUSTER_MASTER_ADDRESS:-}" ]] && _TARGET="$VGRE_CLUSTER_MASTER_ADDRESS"

        if [[ $SKIP_CONNECT_CHECK -eq 0 && -n "$_TARGET" ]]; then
            _check_master_tcp "$_TARGET" || {
                _PRINT_SETUP="$SCRIPT_DIR/vgre-print-linux-setup.sh"
                if [[ -x "$_PRINT_SETUP" ]]; then
                    echo ""
                    bash "$_PRINT_SETUP" --public-ip "${_TARGET%%:*}" 2>/dev/null || true
                fi
                exit 1
            }
        fi

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
