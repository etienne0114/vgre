#!/usr/bin/env bash
# vgre-discover — Detect real public IP and enable cross-LAN token-keyed discovery
#
# Usage:
#   vgre-discover                  Show public IP + worker connection command
#   vgre-discover --set-master     Set VGRE_CLUSTER_ADVERTISED_ADDRESS (source this)
#   vgre-discover --register       Push master's IP to token-keyed KV store
#   vgre-discover --find           Find master via token (worker side, prints connect cmd)
#   vgre-discover --unregister     Remove this master from the KV store
#
# Cross-LAN auto-discovery (--register / --find) uses a free, no-account KV store
# (kvdb.io).  Each cluster gets its own bucket derived from its auth token so only
# nodes with the same token can find each other.
#
# Quick start:
#   # On master:
#   vgre-discover --register       (prints BUCKET_ID — share with workers once)
#   vgre-start --master
#
#   # On each worker (different LAN, has the same token file):
#   vgre-discover --find <BUCKET_ID>
#   vgre-start --worker --master-address <result>

VGRE_DIR="${HOME}/.vgre"
TOKEN_FILE="${VGRE_DIR}/token"
PUBLIC_IP_FILE="${VGRE_DIR}/public_ip"
BUCKET_ID_FILE="${VGRE_DIR}/discovery_bucket"
PORT="${VGRE_PORT:-7777}"
KV_BASE="https://kvdb.io"

mkdir -p "$VGRE_DIR"
chmod 700 "$VGRE_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────

_sha256() {
    if command -v sha256sum &>/dev/null; then
        echo -n "$1" | sha256sum | awk '{print $1}'
    elif command -v shasum &>/dev/null; then
        echo -n "$1" | shasum -a 256 | awk '{print $1}'
    else
        echo ""
    fi
}

_token_fingerprint() {
    [[ -f "$TOKEN_FILE" ]] || { echo ""; return; }
    local tok
    tok=$(cat "$TOKEN_FILE" | tr -d '[:space:]')
    [[ -n "$tok" ]] && _sha256 "$tok" || echo ""
}

_detect_public_ip() {
    local apis=(
        "https://api.ipify.org"
        "https://ipecho.net/plain"
        "https://icanhazip.com"
        "https://checkip.amazonaws.com"
        "https://api4.my-ip.io/ip"
    )
    for api in "${apis[@]}"; do
        local ip
        ip=$(curl -sf --max-time 5 --ipv4 "$api" 2>/dev/null | tr -d '[:space:]') || continue
        if [[ "$ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "$ip"
            return 0
        fi
    done
    return 1
}

# Get or create the per-cluster kvdb.io bucket.
# The bucket ID is stored in ~/.vgre/discovery_bucket.
# An explicit VGRE_DISCOVERY_BUCKET_ID env var always wins.
_bucket_id() {
    if [[ -n "${VGRE_DISCOVERY_BUCKET_ID:-}" ]]; then
        echo "$VGRE_DISCOVERY_BUCKET_ID"
        return 0
    fi
    if [[ -f "$BUCKET_ID_FILE" ]]; then
        cat "$BUCKET_ID_FILE"
        return 0
    fi

    echo "[INFO] No discovery bucket configured — creating one at kvdb.io (free)..." >&2
    local host
    host=$(hostname -s 2>/dev/null || echo "vgre")
    local resp
    resp=$(curl -sf --max-time 10 -X POST \
        -d "email=vgre-${host}@cluster.local" \
        "${KV_BASE}/" 2>/dev/null) || true

    # Response format: {"bucket_id":"ABC123",...}
    local bid
    bid=$(echo "$resp" | grep -o '"bucket_id":"[^"]*"' | cut -d'"' -f4)

    if [[ -z "$bid" ]]; then
        echo "" >&2
        echo "[ERROR] Could not create discovery bucket." >&2
        echo "        1. Visit https://kvdb.io and create a free bucket." >&2
        echo "        2. Set: export VGRE_DISCOVERY_BUCKET_ID=<your-bucket-id>" >&2
        echo "        3. Re-run: vgre-discover --register" >&2
        return 1
    fi

    echo "$bid" > "$BUCKET_ID_FILE"
    chmod 600 "$BUCKET_ID_FILE"
    echo "$bid"
}

# The discovery key within the bucket — first 32 chars of token fingerprint.
# Workers with the same token generate the same key automatically.
_discovery_key() {
    local fp
    fp=$(_token_fingerprint)
    if [[ -z "$fp" ]]; then
        echo "[ERROR] No auth token found at $TOKEN_FILE" >&2
        echo "        Run: vgre-token generate" >&2
        return 1
    fi
    echo "vgre-${fp:0:32}"
}

# ── Commands ──────────────────────────────────────────────────────────────────

cmd_show() {
    echo ""
    echo "=== VGRE Public IP Discovery ==="
    echo ""

    local ip
    echo "[...] Detecting public IP..." >&2
    if ! ip=$(_detect_public_ip); then
        echo "[ERROR] Could not detect public IP. Check internet connectivity." >&2
        exit 1
    fi

    # Save for later use by vgre-start
    echo "$ip" > "$PUBLIC_IP_FILE"
    chmod 600 "$PUBLIC_IP_FILE"

    local fp
    fp=$(_token_fingerprint)

    echo "Public IP  : $ip"
    echo "TCP port   : $PORT  (firewall must allow inbound TCP $PORT)"
    if [[ -n "$fp" ]]; then
        echo "Token FP   : ${fp:0:16}..."
    fi
    echo ""
    echo "Share with workers on other networks:"
    echo ""
    echo "  vgre-start --worker --master-address ${ip}:${PORT}"
    echo ""
    if [[ -n "$fp" ]]; then
        echo "For auto-discovery (workers use same token):"
        echo "  vgre-discover --register           (run on master once)"
        echo "  vgre-discover --find <BUCKET_ID>   (run on each worker)"
    fi
    echo ""
    echo "NAT / firewall note:"
    echo "  Your router must port-forward TCP $PORT to this machine."
    echo "  Or use a VPN (Tailscale / ZeroTier) — no port-forward needed."
    echo ""
}

cmd_set_master() {
    local ip
    if ! ip=$(_detect_public_ip); then
        echo "[WARN] Could not detect public IP — VGRE_CLUSTER_ADVERTISED_ADDRESS not set" >&2
        return 0
    fi
    echo "$ip" > "$PUBLIC_IP_FILE"
    chmod 600 "$PUBLIC_IP_FILE"
    export VGRE_CLUSTER_ADVERTISED_ADDRESS="${ip}:${PORT}"
    echo "[OK] VGRE_CLUSTER_ADVERTISED_ADDRESS=${ip}:${PORT}"
    echo "     Master will broadcast this address so workers across NAT connect correctly."
}

cmd_register() {
    local ip
    echo "[...] Detecting public IP..."
    if ! ip=$(_detect_public_ip); then
        echo "[ERROR] Could not detect public IP." >&2
        exit 1
    fi

    local key
    key=$(_discovery_key) || exit 1

    local bucket
    bucket=$(_bucket_id) || exit 1

    local fp
    fp=$(_token_fingerprint)
    local payload="${ip}:${PORT}|fp=${fp:0:16}|ts=$(date -u +%s)"

    echo "[...] Registering ${ip}:${PORT} at kvdb.io bucket ${bucket}..."
    local http_code
    http_code=$(curl -sf --max-time 10 -o /dev/null -w "%{http_code}" \
        -X POST "${KV_BASE}/${bucket}/${key}" \
        -d "$payload" 2>/dev/null)

    if [[ "$http_code" != "200" && "$http_code" != "201" ]]; then
        echo "[ERROR] Registration failed (HTTP ${http_code})." >&2
        echo "        Bucket: $bucket | Key: $key" >&2
        echo "        Try: export VGRE_DISCOVERY_BUCKET_ID=<your-bucket-id> and retry." >&2
        exit 1
    fi

    echo ""
    echo "[OK] Master registered successfully."
    echo ""
    echo "Bucket ID : $bucket"
    echo "Master IP : ${ip}:${PORT}"
    echo "Token FP  : ${fp:0:16}..."
    echo ""
    echo "Share the BUCKET_ID with workers:"
    echo ""
    echo "  vgre-discover --find $bucket"
    echo ""
    echo "Workers with the same token will connect to ${ip}:${PORT} automatically."
    echo "Re-run --register if your public IP changes."
    echo ""
}

cmd_find() {
    local bucket="${1:-}"
    if [[ -z "$bucket" ]]; then
        # Try locally stored bucket ID
        if [[ -f "$BUCKET_ID_FILE" ]]; then
            bucket=$(cat "$BUCKET_ID_FILE")
        elif [[ -n "${VGRE_DISCOVERY_BUCKET_ID:-}" ]]; then
            bucket="$VGRE_DISCOVERY_BUCKET_ID"
        else
            echo "[ERROR] Usage: vgre-discover --find <BUCKET_ID>" >&2
            echo "        Get the BUCKET_ID from the master: vgre-discover --register" >&2
            exit 1
        fi
    fi

    local key
    key=$(_discovery_key) || exit 1

    echo "[...] Looking up master in bucket ${bucket}..."
    local payload
    payload=$(curl -sf --max-time 10 "${KV_BASE}/${bucket}/${key}" 2>/dev/null)

    if [[ -z "$payload" ]]; then
        echo "[ERROR] No master registered in bucket ${bucket}." >&2
        echo "        Ask the master to run: vgre-discover --register" >&2
        exit 1
    fi

    # payload format: "IP:PORT|fp=FINGERPRINT|ts=TIMESTAMP"
    local addr
    addr=$(echo "$payload" | cut -d'|' -f1)
    local stored_fp
    stored_fp=$(echo "$payload" | grep -o 'fp=[^|]*' | cut -d= -f2)
    local local_fp
    local_fp=$(_token_fingerprint)

    # Verify token fingerprint matches (first 16 chars)
    if [[ -n "$stored_fp" && -n "$local_fp" && "${local_fp:0:16}" != "$stored_fp" ]]; then
        echo "[ERROR] Token fingerprint mismatch!" >&2
        echo "        Registered: $stored_fp" >&2
        echo "        Local:      ${local_fp:0:16}" >&2
        echo "        Make sure you and the master have the same token." >&2
        exit 1
    fi

    echo ""
    echo "[OK] Master found: $addr"
    echo ""
    echo "Connect with:"
    echo "  vgre-start --worker --master-address $addr"
    echo ""

    # Save master address for use by vgre-start
    local ip
    ip=$(echo "$addr" | cut -d: -f1)
    echo "${ip}" > "$PUBLIC_IP_FILE"
    export VGRE_CLUSTER_MASTER_ADDRESS="$addr"
    echo "VGRE_CLUSTER_MASTER_ADDRESS=$addr"
}

cmd_unregister() {
    local key
    key=$(_discovery_key) || exit 1
    local bucket
    bucket=$(_bucket_id) || exit 1

    curl -sf --max-time 10 -X DELETE "${KV_BASE}/${bucket}/${key}" >/dev/null 2>&1 || true
    echo "[OK] Discovery registration removed."
}

# ── Entry point ───────────────────────────────────────────────────────────────
case "${1:-}" in
    ""|--show)      cmd_show ;;
    --set-master)   cmd_set_master ;;
    --register)     cmd_register ;;
    --find)         shift; cmd_find "${1:-}" ;;
    --unregister)   cmd_unregister ;;
    --help|-h)
        grep '^#' "$0" | head -30 | sed 's/^# \?//'
        ;;
    *)
        echo "[ERROR] Unknown command: $1  (run vgre-discover --help)"
        exit 1
        ;;
esac
