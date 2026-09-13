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

# Deterministic bucket from auth token — same token on master and worker ⇒ same bucket.
_token_bucket_id() {
    local fp
    fp=$(_token_fingerprint)
    [[ -n "$fp" ]] || return 1
    echo "vgre-${fp:0:16}"
}

# Bucket for lookups: env > file > token-derived (no auto-create).
_resolve_bucket_for_lookup() {
    if [[ -n "${VGRE_DISCOVERY_BUCKET_ID:-}" ]]; then
        echo "$VGRE_DISCOVERY_BUCKET_ID"
        return 0
    fi
    if [[ -f "$BUCKET_ID_FILE" ]]; then
        tr -d '[:space:]' < "$BUCKET_ID_FILE"
        return 0
    fi
    _token_bucket_id
}

_is_private_ip() {
    local ip="$1"
    [[ "$ip" == "127.0.0.1" || "$ip" == "localhost" || "$ip" == "::1" ]] && return 0
    [[ "$ip" =~ ^10\. ]] && return 0
    [[ "$ip" =~ ^192\.168\. ]] && return 0
    [[ "$ip" =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\. ]] && return 0
    return 1
}

_persist_master_address() {
    local addr="$1"
    local env_file="${VGRE_DIR}/env"
    mkdir -p "$VGRE_DIR"
    [[ -f "$env_file" ]] || touch "$env_file"
    chmod 600 "$env_file"
    local tmp
    tmp=$(mktemp)
    if grep -q "VGRE_CLUSTER_MASTER_ADDRESS" "$env_file" 2>/dev/null; then
        grep -v "VGRE_CLUSTER_MASTER_ADDRESS" "$env_file" > "$tmp"
    else
        cp "$env_file" "$tmp" 2>/dev/null || : > "$tmp"
    fi
    printf '\nexport VGRE_CLUSTER_MASTER_ADDRESS="%s"\n' "$addr" >> "$tmp"
    mv -f "$tmp" "$env_file"
}

# Get or create the per-cluster kvdb.io bucket.
# Priority: VGRE_DISCOVERY_BUCKET_ID > ~/.vgre/discovery_bucket > token bucket > auto-create.
_bucket_id() {
    local bid=""

    if [[ -n "${VGRE_DISCOVERY_BUCKET_ID:-}" ]]; then
        bid="$VGRE_DISCOVERY_BUCKET_ID"
    elif [[ -f "$BUCKET_ID_FILE" ]]; then
        bid=$(tr -d '[:space:]' < "$BUCKET_ID_FILE")
    else
        bid=$(_token_bucket_id 2>/dev/null) || true
    fi

    # Persist to file so subsequent sessions don't need the env var.
    if [[ -n "$bid" ]]; then
        if [[ ! -f "$BUCKET_ID_FILE" ]] || [[ "$(tr -d '[:space:]' < "$BUCKET_ID_FILE")" != "$bid" ]]; then
            echo "$bid" > "$BUCKET_ID_FILE"
            chmod 600 "$BUCKET_ID_FILE"
        fi
        echo "$bid"
        return 0
    fi

    # Legacy fallback: auto-create a random kvdb.io bucket.
    # is writable only after confirming that email.
    echo "[INFO] No discovery bucket configured — creating one at kvdb.io (free)..." >&2
    echo "[INFO] You will receive a verification email. Visit https://kvdb.io/login to activate." >&2
    local host
    host=$(hostname -s 2>/dev/null || echo "vgre")
    local resp
    resp=$(curl -s --max-time 10 -X POST \
        -d "email=vgre-${host}@cluster.local" \
        "${KV_BASE}/" 2>/dev/null) || true

    bid=$(echo "$resp" | grep -o '"bucket_id":"[^"]*"' | cut -d'"' -f4)

    if [[ -z "$bid" ]]; then
        echo "" >&2
        echo "[ERROR] Could not auto-create discovery bucket." >&2
        echo "        Solution A (easiest): create a free bucket manually:" >&2
        echo "          1. Visit https://kvdb.io → Get Started → enter your email" >&2
        echo "          2. Confirm the verification email you receive" >&2
        echo "          3. Copy the bucket ID from your browser URL" >&2
        echo "          4. Run: export VGRE_DISCOVERY_BUCKET_ID=<your-bucket-id>" >&2
        echo "          5. Re-run: vgre-discover --register" >&2
        echo "        Solution B: share the IP manually — vgre-discover (no flags)" >&2
        return 1
    fi

    echo "$bid" > "$BUCKET_ID_FILE"
    chmod 600 "$BUCKET_ID_FILE"
    echo "[INFO] Bucket $bid created. Check your email to verify before writing." >&2
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

    # Capture both response body AND HTTP status code.
    # Using a temp file avoids subshell issues with multi-line output from -w.
    local tmp_body
    tmp_body=$(mktemp)
    local http_code
    http_code=$(curl -s --max-time 10 -o "$tmp_body" -w "%{http_code}" \
        -X POST "${KV_BASE}/${bucket}/${key}" \
        -d "$payload" 2>/dev/null)
    local resp_body
    resp_body=$(cat "$tmp_body" 2>/dev/null)
    rm -f "$tmp_body"

    if [[ "$http_code" != "200" && "$http_code" != "201" ]]; then
        echo "[ERROR] Registration failed (HTTP ${http_code})." >&2
        if [[ -n "$resp_body" ]]; then
            echo "        Server says: ${resp_body}" >&2
        fi
        if echo "$resp_body" | grep -qi "not verified\|activate\|confirm"; then
            echo "" >&2
            echo "        Your kvdb.io account needs email verification:" >&2
            echo "          1. Visit https://kvdb.io/login" >&2
            echo "          2. Verify your email address" >&2
            echo "          3. Re-run: vgre-discover --register" >&2
        elif [[ "$http_code" == "403" ]]; then
            echo "" >&2
            echo "        Bucket ID: $bucket" >&2
            echo "        If this bucket was just created, verify the email first:" >&2
            echo "          visit https://kvdb.io/login" >&2
            echo "        Or create a new bucket: visit https://kvdb.io" >&2
        else
            echo "        Bucket: $bucket | Key: $key" >&2
        fi
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
    local quiet="${2:-0}"
    if [[ -z "$bucket" ]]; then
        bucket=$(_resolve_bucket_for_lookup 2>/dev/null) || bucket=""
        if [[ -z "$bucket" ]]; then
            [[ "$quiet" == "1" ]] && return 1
            echo "[ERROR] Usage: vgre-discover --find [BUCKET_ID]" >&2
            echo "        Get the BUCKET_ID from the master: vgre-discover --register" >&2
            exit 1
        fi
    fi

    local key
    key=$(_discovery_key) || { [[ "$quiet" == "1" ]] && return 1; exit 1; }

    # Save bucket ID to file so future sessions don't need the env var.
    if [[ ! -f "$BUCKET_ID_FILE" ]] || [[ "$(tr -d '[:space:]' < "$BUCKET_ID_FILE")" != "$bucket" ]]; then
        echo "$bucket" > "$BUCKET_ID_FILE"
        chmod 600 "$BUCKET_ID_FILE"
    fi

    [[ "$quiet" != "1" ]] && echo "[...] Looking up master in bucket ${bucket}..."
    local tmp_body
    tmp_body=$(mktemp)
    local http_code
    http_code=$(curl -s --max-time 10 -o "$tmp_body" -w "%{http_code}" \
        "${KV_BASE}/${bucket}/${key}" 2>/dev/null)
    local payload
    payload=$(cat "$tmp_body" 2>/dev/null)
    rm -f "$tmp_body"

    if [[ "$http_code" == "404" || -z "$payload" || "$payload" == "Not Found" ]]; then
        [[ "$quiet" == "1" ]] && return 1
        echo "[ERROR] No master registered in bucket ${bucket}." >&2
        echo "        Ask the master to run: vgre-discover --register" >&2
        echo "        (or: vgre-start --master — registers automatically)" >&2
        exit 1
    fi

    if [[ "$http_code" != "200" ]]; then
        [[ "$quiet" == "1" ]] && return 1
        echo "[ERROR] Lookup failed (HTTP ${http_code})." >&2
        if [[ -n "$payload" ]]; then
            echo "        Server says: ${payload}" >&2
        fi
        if echo "$payload" | grep -qi "not verified\|activate"; then
            echo "        Verify the bucket owner's email at https://kvdb.io/login" >&2
        fi
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
        [[ "$quiet" == "1" ]] && return 1
        echo "[ERROR] Token fingerprint mismatch!" >&2
        echo "        Registered: $stored_fp" >&2
        echo "        Local:      ${local_fp:0:16}" >&2
        echo "        Make sure you and the master have the same token." >&2
        exit 1
    fi

    [[ "$quiet" != "1" ]] && echo ""
    [[ "$quiet" != "1" ]] && echo "[OK] Master found: $addr"
    [[ "$quiet" != "1" ]] && echo ""
    [[ "$quiet" != "1" ]] && echo "Connect with:"
    [[ "$quiet" != "1" ]] && echo "  vgre-start --worker --master-address $addr"
    [[ "$quiet" != "1" ]] && echo ""

    # Save master address for vgre-start and the worker process.
    local ip
    ip=$(echo "$addr" | cut -d: -f1)
    echo "${ip}" > "$PUBLIC_IP_FILE"
    # Persist only LAN addresses; public IPs break same-network workers via NAT hairpin.
    if _is_private_ip "$ip"; then
        _persist_master_address "$addr"
    fi
    export VGRE_CLUSTER_MASTER_ADDRESS="$addr"
    echo "VGRE_CLUSTER_MASTER_ADDRESS=$addr"
}

cmd_find_auto() {
    cmd_find "" 1
}

cmd_unregister() {
    local key
    key=$(_discovery_key) || exit 1
    local bucket
    bucket=$(_bucket_id) || exit 1

    local http_code
    http_code=$(curl -s --max-time 10 -o /dev/null -w "%{http_code}" \
        -X DELETE "${KV_BASE}/${bucket}/${key}" 2>/dev/null)
    if [[ "$http_code" == "200" || "$http_code" == "204" ]]; then
        echo "[OK] Discovery registration removed."
    else
        echo "[WARN] DELETE returned HTTP ${http_code} — entry may not have existed."
    fi
}

# ── Entry point ───────────────────────────────────────────────────────────────
case "${1:-}" in
    ""|--show)      cmd_show ;;
    --set-master)   cmd_set_master ;;
    --register)     cmd_register ;;
    --find)         shift; cmd_find "${1:-}" 0 ;;
    --find-auto)    cmd_find_auto ;;
    --unregister)   cmd_unregister ;;
    --help|-h)
        cat <<'HELP'
vgre-discover -- Detect real public IP and enable cross-LAN token-keyed cluster discovery

Usage:
  vgre-discover                  Show public IP + worker connection command
  vgre-discover --set-master     Set VGRE_CLUSTER_ADVERTISED_ADDRESS for this session
                                 (called automatically by vgre-start --master)
  vgre-discover --register       Register master's IP in a token-keyed KV store
                                 Workers with the same token can find the master
                                 without knowing its IP. Prints BUCKET_ID to share.
  vgre-discover --find [BUCKET]  Find master's IP using the local token + BUCKET_ID
                                 Verifies token fingerprint before connecting.
  vgre-discover --unregister     Remove registration from KV store

Cross-LAN discovery flow (no manual IP sharing needed):
  # On master (run once, or after IP changes):
  vgre-discover --register
  # Output: Bucket ID: aBcDeFgH  <- share with workers

  # On each worker (same token, any network):
  vgre-discover --find aBcDeFgH
  vgre-start --worker --master-address <printed address>

Environment:
  VGRE_DISCOVERY_BUCKET_ID   Override the kvdb.io bucket ID
  VGRE_PORT                  TCP port (default 7777)
  VGRE_TCP_AUTH_TOKEN_FILE   Path to token file (default ~/.vgre/token)
HELP
        ;;

    *)
        echo "[ERROR] Unknown command: $1  (run vgre-discover --help)"
        exit 1
        ;;
esac
