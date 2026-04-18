#!/bin/bash
# VGRE Cluster Setup — run once on every machine that joins the cluster.
#
# What this does:
#   1. Creates or reuses an auth token shared by all nodes (master + workers).
#   2. Saves the token to ~/.vgre/token (chmod 600).
#   3. Adds VGRE_TCP_AUTH_TOKEN_FILE to your shell profile so it loads automatically.
#   4. Prints the exact commands to start master and worker.
#
# Usage:
#   bash scripts/setup-cluster.sh
#
# After this you only need:
#   vgre-start --master           (on the machine running your CUDA app)
#   vgre-start --worker           (on each compute node, same subnet)
#   vgre-start --worker --master-ip 192.168.1.50   (different subnet)

set -e

VGRE_DIR="$HOME/.vgre"
TOKEN_FILE="$VGRE_DIR/token"
INSTALL_DIR="$HOME/.local/share/VGRE"
BIN_DIR="$HOME/.local/bin"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║       VGRE Cluster Setup                 ║"
echo "╚══════════════════════════════════════════╝"
echo ""

mkdir -p "$VGRE_DIR"
chmod 700 "$VGRE_DIR"

# ── Step 1: Auth Token ────────────────────────────────────────────────────────
if [[ -f "$TOKEN_FILE" ]]; then
    PREVIEW=$(cat "$TOKEN_FILE" | head -c 8)
    echo "Existing token found: ${PREVIEW}... (truncated)"
    printf "Keep existing token? [Y/n]: "
    read -r KEEP
    if [[ "$KEEP" == "n" || "$KEEP" == "N" ]]; then
        rm -f "$TOKEN_FILE"
    fi
fi

if [[ ! -f "$TOKEN_FILE" ]]; then
    echo ""
    echo "Auth Token Options:"
    echo "  1) Generate a random secure token  (recommended)"
    echo "  2) Enter your own token            (paste from another node)"
    printf "Choice [1]: "
    read -r CHOICE
    CHOICE="${CHOICE:-1}"

    if [[ "$CHOICE" == "2" ]]; then
        printf "Enter token: "
        read -r -s TOKEN
        echo ""
        if [[ ${#TOKEN} -lt 8 ]]; then
            echo "❌ Token must be at least 8 characters. Aborting."
            exit 1
        fi
    else
        # Generate a random 32-byte hex token
        if command -v openssl >/dev/null 2>&1; then
            TOKEN=$(openssl rand -hex 32)
        else
            TOKEN=$(head -c 32 /dev/urandom | od -A n -t x1 | tr -d ' \n' | head -c 64)
        fi
        echo "Generated token: $TOKEN"
    fi

    printf '%s' "$TOKEN" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
    echo "✅ Token saved to $TOKEN_FILE"
fi

# ── Step 2: Shell Profile ─────────────────────────────────────────────────────
PROFILE_LINE="export VGRE_TCP_AUTH_TOKEN_FILE=\"$TOKEN_FILE\""
PROFILE_ADDED=0
for PROFILE in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
    if [[ -f "$PROFILE" ]] && ! grep -q "VGRE_TCP_AUTH_TOKEN_FILE" "$PROFILE"; then
        { echo ""; echo "# VGRE cluster auth token"; echo "$PROFILE_LINE"; } >> "$PROFILE"
        echo "✅ Added token path to $PROFILE"
        PROFILE_ADDED=1
    fi
done
if [[ $PROFILE_ADDED -eq 0 ]] && ! grep -qr "VGRE_TCP_AUTH_TOKEN_FILE" "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile" 2>/dev/null; then
    { echo ""; echo "# VGRE cluster auth token"; echo "$PROFILE_LINE"; } >> "$HOME/.profile"
    echo "✅ Added token path to ~/.profile"
fi

# Apply to current session
export VGRE_TCP_AUTH_TOKEN_FILE="$TOKEN_FILE"

# ── Step 3: Make vgre-start available ────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
START_SCRIPT="$SCRIPT_DIR/vgre-start.sh"

if [[ -f "$START_SCRIPT" ]]; then
    chmod +x "$START_SCRIPT"
    # Symlink into bin if install dir exists; also works from repo directly
    mkdir -p "$BIN_DIR"
    ln -sf "$START_SCRIPT" "$BIN_DIR/vgre-start" 2>/dev/null || true
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Setup complete! Here is how to use your cluster:       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                          ║"
echo "║  MASTER node (the machine running your CUDA app):        ║"
echo "║    vgre-start --master                                   ║"
echo "║                                                          ║"
echo "║  WORKER node (same subnet — auto-discovered):            ║"
echo "║    vgre-start --worker                                   ║"
echo "║                                                          ║"
echo "║  WORKER node (different subnet):                         ║"
echo "║    vgre-start --worker --master-ip <MASTER_IP>           ║"
echo "║                                                          ║"
echo "║  Copy the token file to each worker at the same path:    ║"
echo "║    scp $TOKEN_FILE  user@worker-machine:$TOKEN_FILE"
echo "║                                                          ║"
echo "║  NOTE: Restart your terminal for the profile change      ║"
echo "║        to take effect, or run:                           ║"
echo "║    source ~/.bashrc   (or ~/.zshrc / ~/.profile)         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
