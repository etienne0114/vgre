#!/usr/bin/env bash
# Print copy-paste setup steps for the Linux master (run from any worker or master).
#
# Usage:
#   vgre-print-linux-setup
#   vgre-print-linux-setup --public-ip <IP> [--lan-ip <IP>] [--port N]
#
# Values are inferred when omitted:
#   --public-ip  VGRE_CLUSTER_MASTER_ADDRESS, vgre-discover, or placeholder
#   --lan-ip     hostname -I on Linux, or placeholder

set -euo pipefail

VGRE_DIR="${HOME}/.vgre"
ENV_FILE="${VGRE_DIR}/env"
LAN_IP=""
PUBLIC_IP=""
PORT="${VGRE_PORT:-7777}"
DISCOVERY_PORT="${VGRE_CLUSTER_UDP_ANNOUNCE_PORT:-7778}"

# Load persisted VGRE env (master address, port, etc.)
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lan-ip)     LAN_IP="$2"; shift 2 ;;
        --public-ip)  PUBLIC_IP="$2"; shift 2 ;;
        --port)       PORT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: vgre-print-linux-setup [--lan-ip IP] [--public-ip IP] [--port N]"
            exit 0
            ;;
        *) echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
    esac
done

_resolve_public_ip() {
    if [[ -n "$PUBLIC_IP" ]]; then
        echo "$PUBLIC_IP"
        return
    fi
    if [[ -n "${VGRE_CLUSTER_MASTER_ADDRESS:-}" ]]; then
        echo "${VGRE_CLUSTER_MASTER_ADDRESS%%:*}"
        return
    fi
    if command -v vgre-discover >/dev/null 2>&1; then
        local ip
        ip=$(vgre-discover 2>/dev/null | awk '/^Public IP/ {print $3; exit}' || true)
        if [[ -n "$ip" ]]; then
            echo "$ip"
            return
        fi
    fi
    echo "<YOUR_PUBLIC_IP>"
}

_resolve_lan_ip() {
    if [[ -n "$LAN_IP" ]]; then
        echo "$LAN_IP"
        return
    fi
    if [[ "$(uname -s)" == "Linux" ]] && command -v hostname >/dev/null 2>&1; then
        local ip
        ip=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
        if [[ -n "$ip" ]]; then
            echo "$ip"
            return
        fi
    fi
    echo "<MASTER_LAN_IP>   # on master: hostname -I | awk '{print \$1}'"
}

PUBLIC_IP=$(_resolve_public_ip)
LAN_IP=$(_resolve_lan_ip)
HOST_LABEL="$(hostname 2>/dev/null || echo linux-master)"

cat <<EOF
══════════════════════════════════════════════════════════════════
 LINUX MASTER — manual setup (copy/paste on ${HOST_LABEL})
══════════════════════════════════════════════════════════════════

Your master must listen locally. Workers on other networks need inbound
TCP ${PORT} to reach this machine through the router.

── 1. Confirm master is running ──────────────────────────────────

ss -tlnp | grep ${PORT}
# Expect: LISTEN 0.0.0.0:${PORT}  (vgre_dashboard or vgre-worker)

── 2. Open host firewall ─────────────────────────────────────────

sudo ufw allow ${PORT}/tcp
sudo ufw allow ${DISCOVERY_PORT}/udp
sudo ufw status | grep -E '${PORT}|${DISCOVERY_PORT}'

── 3. Register public IP for cross-LAN discovery (optional) ─────

vgre-discover --register
# Workers with the same token can then: vgre-discover --find-auto

── 4. Router port-forward (REQUIRED for WAN without VPN) ─────────

Log into your router admin page and add:

  Service name : VGRE
  Protocol     : TCP
  External port: ${PORT}
  Internal IP  : ${LAN_IP}
  Internal port: ${PORT}

Verify public IP matches what workers use:
  curl -4 -s ifconfig.me
  # Should print: ${PUBLIC_IP}

── 5. Test from OUTSIDE your home Wi‑Fi (phone on cellular) ─────

  nc -zv ${PUBLIC_IP} ${PORT}

── 6. If port-forward fails (CGNAT / mobile ISP) ───────────────

Install Tailscale on Linux AND the worker — no router config needed:

  # Linux master:
  curl -fsSL https://tailscale.com/install.sh | sh
  sudo tailscale up

  # macOS worker:
  brew install --cask tailscale
  # Open Tailscale app, sign in with the same account

  tailscale ip -4    # use this IP instead of ${PUBLIC_IP}
  vgre-connect-check <TAILSCALE_IP>:${PORT}
  vgre-start --worker --master-address <TAILSCALE_IP>:${PORT}

── 7. Token must match on both nodes ───────────────────────────

vgre-token fingerprint

══════════════════════════════════════════════════════════════════
 WORKER — after Linux/router is fixed
══════════════════════════════════════════════════════════════════

  vgre-connect-check ${PUBLIC_IP}:${PORT}
  vgre-start --worker --master-address ${PUBLIC_IP}:${PORT}

EOF
