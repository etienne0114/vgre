#!/bin/bash
# VGRE Start — launch a master or worker node with one command.
#
# Prerequisites: run scripts/setup-cluster.sh once on each machine first.
#
# Usage:
#   vgre-start --master                          Start master (runs the dashboard)
#   vgre-start --worker                          Start worker (auto-discovers master)
#   vgre-start --worker --master-ip 10.0.0.5    Start worker, connect to specific master
#   vgre-start --worker --port 7778             Start worker on a non-default port
#   vgre-start --test                           Quick local test: master + worker on same machine

set -e

INSTALL_DIR="${VGRE_INSTALL_DIR:-$HOME/.local/share/VGRE}"
TOKEN_FILE="${VGRE_TCP_AUTH_TOKEN_FILE:-$HOME/.vgre/token}"
PORT="${VGRE_PORT:-7777}"
MODE=""
MASTER_IP=""
EXTRA_ARGS=()

# ── Parse Arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --master)       MODE="master" ;;
        --worker)       MODE="worker" ;;
        --test)         MODE="test" ;;
        --master-ip)    MASTER_IP="$2"; shift ;;
        --port)         PORT="$2"; shift ;;
        --threads)      EXTRA_ARGS+=("--threads" "$2"); shift ;;
        --help|-h)
            grep '^# ' "$0" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1  (run vgre-start --help)"
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

# ── Library Path ──────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="$INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"

WORKER_BIN="$INSTALL_DIR/vgre-worker"
if [[ ! -x "$WORKER_BIN" ]]; then
    # Fall back to PATH or local build for development
    WORKER_BIN="$(command -v vgre-worker 2>/dev/null || true)"
fi
if [[ -z "$WORKER_BIN" || ! -x "$WORKER_BIN" ]]; then
    echo "❌ vgre-worker not found. Run scripts/vgre_sync.sh to build and install first."
    exit 1
fi

# ── Start ─────────────────────────────────────────────────────────────────────
case "$MODE" in

    master)
        echo "Starting VGRE Master Node..."
        echo "  Port:  $PORT"
        echo "  Token: $TOKEN_FILE"
        echo ""
        # Master is embedded in the dashboard; export cluster port for it to pick up.
        export VGRE_PORT="$PORT"
        DASHBOARD_BIN="$INSTALL_DIR/vgre-launch.sh"
        [[ -x "$DASHBOARD_BIN" ]] || DASHBOARD_BIN="$INSTALL_DIR/vgre_dashboard"
        exec "$DASHBOARD_BIN"
        ;;

    worker)
        if [[ -n "$MASTER_IP" ]]; then
            export VGRE_CLUSTER_NODES="$MASTER_IP:$PORT"
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
        DASHBOARD_BIN="$INSTALL_DIR/vgre-launch.sh"
        [[ -x "$DASHBOARD_BIN" ]] || DASHBOARD_BIN="$INSTALL_DIR/vgre_dashboard"
        "$DASHBOARD_BIN" &
        MASTER_PID=$!
        trap 'echo "Stopping test..."; kill $WORKER_PID $MASTER_PID 2>/dev/null; exit 0' INT TERM
        wait $MASTER_PID
        kill $WORKER_PID 2>/dev/null || true
        ;;
esac
