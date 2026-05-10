#!/usr/bin/env sh
# vgre-token — VGRE Auth Token Manager (Linux / macOS)
#
# Run from ANYWHERE after install:
#   vgre-token generate              Generate a new secure 64-hex token
#   vgre-token show                  Print the stored token value
#   vgre-token fingerprint           Print the SHA-256 fingerprint
#   vgre-token set <TOKEN>           Store a specific token (from another node)
#   vgre-token verify                Check env-var matches the stored file
#   vgre-token copy                  Print the scp command to share with a worker
#   vgre-token revoke                Delete the stored token (prompts for confirmation)
#
# The token is stored in ~/.vgre/token (chmod 600).
# The env-var VGRE_TCP_AUTH_TOKEN_FILE is written to ~/.vgre/env so it is
# loaded automatically by every new terminal (sourced by install_local.sh).
#
# Platform: Linux, macOS (any POSIX sh; no bash required)

set -eu

# ── Paths ─────────────────────────────────────────────────────────────────────
VGRE_DIR="${VGRE_DIR:-$HOME/.vgre}"
TOKEN_FILE="$VGRE_DIR/token"
ENV_FILE="$VGRE_DIR/env"

# ── Colors (only when attached to a terminal) ──────────────────────────────────
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; RESET='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; DIM=''; RESET=''
fi

ok()   { printf "${GREEN}✓${RESET} %s\n" "$*"; }
info() { printf "${CYAN}▸${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}⚠${RESET} %s\n" "$*"; }
die()  { printf "${RED}✗${RESET} %s\n" "$*" >&2; exit 1; }

# ── Helpers ───────────────────────────────────────────────────────────────────
_ensure_dir() {
    mkdir -p "$VGRE_DIR"
    chmod 700 "$VGRE_DIR"
}

_sha256() {
    # Works on Linux (sha256sum) and macOS (shasum -a 256)
    if command -v sha256sum >/dev/null 2>&1; then
        printf '%s' "$1" | sha256sum | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        printf '%s' "$1" | shasum -a 256 | awk '{print $1}'
    else
        echo "(sha256 unavailable)"
    fi
}

_read_token() {
    [ -f "$TOKEN_FILE" ] || die "No token found at $TOKEN_FILE. Run: vgre-token generate"
    # Strip trailing newline/CR
    tr -d '\r\n' < "$TOKEN_FILE"
}

_write_token() {
    _ensure_dir
    # Write without trailing newline so the hash matches on every platform
    printf '%s' "$1" > "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
}

_update_env_file() {
    _ensure_dir
    # Create env file if absent
    [ -f "$ENV_FILE" ] || touch "$ENV_FILE"
    chmod 600 "$ENV_FILE"

    # Update or insert VGRE_TCP_AUTH_TOKEN_FILE
    if grep -q "VGRE_TCP_AUTH_TOKEN_FILE" "$ENV_FILE" 2>/dev/null; then
        # Replace existing line (portable sed -i)
        TMP=$(mktemp)
        grep -v "VGRE_TCP_AUTH_TOKEN_FILE" "$ENV_FILE" > "$TMP"
        printf '\nexport VGRE_TCP_AUTH_TOKEN_FILE="%s"\n' "$TOKEN_FILE" >> "$TMP"
        mv -f "$TMP" "$ENV_FILE"
    else
        printf '\n# VGRE auth token — written by vgre-token\nexport VGRE_TCP_AUTH_TOKEN_FILE="%s"\n' \
            "$TOKEN_FILE" >> "$ENV_FILE"
    fi
}

_add_to_shell_profile() {
    SOURCE_LINE="# VGRE environment"
    LOAD_LINE="[ -f \"$ENV_FILE\" ] && . \"$ENV_FILE\""
    for PROFILE in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.bash_profile" "$HOME/.profile"; do
        if [ -f "$PROFILE" ]; then
            if ! grep -q "vgre/env" "$PROFILE" 2>/dev/null; then
                printf '\n%s\n%s\n' "$SOURCE_LINE" "$LOAD_LINE" >> "$PROFILE"
                ok "Added env sourcing to $PROFILE"
            fi
        fi
    done
}

_gen_token() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 32
    else
        od -An -tx1 /dev/urandom | tr -d ' \n' | head -c 64
    fi
}

# ── Print fingerprint block ───────────────────────────────────────────────────
_print_fingerprint_block() {
    TOK="$1"
    FP=$(_sha256 "$TOK")
    printf "\n"
    printf "${BOLD}Token fingerprint (SHA-256):${RESET}\n"
    printf "  ${CYAN}%s${RESET}\n" "$FP"
    printf "\n"
    printf "${DIM}Run  vgre-token fingerprint  on every node."
    printf " Both master and worker MUST show identical output.${RESET}\n\n"
}

# ── Commands ──────────────────────────────────────────────────────────────────
cmd_generate() {
    if [ -f "$TOKEN_FILE" ]; then
        EXISTING=$(_read_token)
        EXISTING_FP=$(_sha256 "$EXISTING")
        warn "A token already exists (SHA-256: ${EXISTING_FP%"${EXISTING_FP#????????????????}"}...)"
        printf "Generating a new token will disconnect all workers. Continue? [y/N]: "
        read -r ans
        case "$ans" in y|Y) ;; *) info "Aborted. Existing token kept."; exit 0;; esac
    fi

    info "Generating 256-bit random token..."
    TOKEN=$(_gen_token)
    _write_token "$TOKEN"
    _update_env_file
    _add_to_shell_profile
    ok "Token saved to $TOKEN_FILE"
    _print_fingerprint_block "$TOKEN"
    ok "Environment file updated: $ENV_FILE"
    warn "Copy $TOKEN_FILE to every worker:  scp $TOKEN_FILE user@WORKER_IP:$TOKEN_FILE"
    printf "\n${DIM}New terminals will load the token automatically."
    printf " Current shell: run  . %s ${RESET}\n\n" "$ENV_FILE"
}

cmd_show() {
    TOK=$(_read_token)
    printf "\n${BOLD}Stored token:${RESET}\n"
    printf "  %s\n\n" "$TOK"
    printf "${DIM}Location: %s${RESET}\n\n" "$TOKEN_FILE"
}

cmd_fingerprint() {
    TOK=$(_read_token)
    _print_fingerprint_block "$TOK"
}

cmd_set() {
    [ -n "${1:-}" ] || die "Usage: vgre-token set <TOKEN>"
    TOKEN="$1"
    [ ${#TOKEN} -ge 8 ] || die "Token must be at least 8 characters."
    _write_token "$TOKEN"
    _update_env_file
    _add_to_shell_profile
    ok "Token saved to $TOKEN_FILE"
    _print_fingerprint_block "$TOKEN"
    ok "Environment file updated: $ENV_FILE"
}

cmd_verify() {
    TOK=$(_read_token)
    FILE_FP=$(_sha256 "$TOK")

    printf "\n${BOLD}File token fingerprint:${RESET}\n"
    printf "  ${CYAN}%s${RESET}  (%s)\n\n" "$FILE_FP" "$TOKEN_FILE"

    # Check environment variable
    ENV_TOK="${VGRE_TCP_AUTH_TOKEN:-}"
    FILE_TOK_PATH="${VGRE_TCP_AUTH_TOKEN_FILE:-}"

    if [ -n "$FILE_TOK_PATH" ] && [ -f "$FILE_TOK_PATH" ]; then
        ENV_TOK_VALUE=$(tr -d '\r\n' < "$FILE_TOK_PATH")
        ENV_FP=$(_sha256 "$ENV_TOK_VALUE")
        if [ "$ENV_FP" = "$FILE_FP" ]; then
            ok "VGRE_TCP_AUTH_TOKEN_FILE matches stored token."
        else
            warn "VGRE_TCP_AUTH_TOKEN_FILE points to a DIFFERENT token!"
            printf "  Env file fingerprint: %s\n" "$ENV_FP"
        fi
    elif [ -n "$ENV_TOK" ]; then
        ENV_FP=$(_sha256 "$ENV_TOK")
        if [ "$ENV_FP" = "$FILE_FP" ]; then
            ok "VGRE_TCP_AUTH_TOKEN env var matches stored token."
        else
            warn "VGRE_TCP_AUTH_TOKEN env var does NOT match stored token!"
            printf "  Env var fingerprint:  %s\n" "$ENV_FP"
        fi
    else
        warn "No VGRE_TCP_AUTH_TOKEN or VGRE_TCP_AUTH_TOKEN_FILE in environment."
        info "Run: . $ENV_FILE   to load it."
    fi
    printf "\n"
}

cmd_copy() {
    [ -f "$TOKEN_FILE" ] || die "No token found. Run: vgre-token generate"
    printf "\n${BOLD}Copy token to a worker machine:${RESET}\n\n"
    printf "  ${CYAN}scp %s user@WORKER_IP:%s${RESET}\n\n" "$TOKEN_FILE" "$TOKEN_FILE"
    printf "${DIM}Replace  user@WORKER_IP  with the actual worker address."
    printf "\nRun  vgre-token fingerprint  on both machines to confirm they match.${RESET}\n\n"
}

cmd_revoke() {
    [ -f "$TOKEN_FILE" ] || die "No token found at $TOKEN_FILE"
    TOK=$(_read_token)
    FP=$(_sha256 "$TOK")
    warn "About to DELETE the stored token."
    printf "  Fingerprint: %s\n" "$FP"
    printf "  Location:    %s\n\n" "$TOKEN_FILE"
    printf "This will disconnect ALL workers using this token. Continue? [y/N]: "
    read -r ans
    case "$ans" in
        y|Y)
            rm -f "$TOKEN_FILE"
            # Remove the env-var line from ENV_FILE
            if [ -f "$ENV_FILE" ]; then
                TMP=$(mktemp)
                grep -v "VGRE_TCP_AUTH_TOKEN_FILE" "$ENV_FILE" > "$TMP" || true
                mv -f "$TMP" "$ENV_FILE"
            fi
            ok "Token revoked."
            ;;
        *) info "Aborted. Token kept." ;;
    esac
}

cmd_help() {
    cat <<'HELP'

vgre-token — VGRE Auth Token Manager

Usage:
  vgre-token generate              Generate a new secure 64-hex token
  vgre-token show                  Print the stored token value
  vgre-token fingerprint           Print the SHA-256 fingerprint
  vgre-token set <TOKEN>           Store a specific token (paste from another node)
  vgre-token verify                Check env-var and file token match
  vgre-token copy                  Print the scp command to share the token
  vgre-token revoke                Delete the stored token

Token file:  ~/.vgre/token  (chmod 600)
Env file:    ~/.vgre/env    (auto-sourced by new terminals after install)

Quick start:
  # On master — generate once:
  vgre-token generate

  # On every worker — paste the token from the master:
  vgre-token set <paste-token-here>
  # OR copy the file directly:
  scp master:~/.vgre/token ~/.vgre/token

  # Verify both match:
  vgre-token fingerprint   # must be identical on master and worker

HELP
}

# ── Entry point ───────────────────────────────────────────────────────────────
CMD="${1:-help}"
shift || true

case "$CMD" in
    generate)    cmd_generate ;;
    show)        cmd_show ;;
    fingerprint) cmd_fingerprint ;;
    set)         cmd_set "${1:-}" ;;
    verify)      cmd_verify ;;
    copy)        cmd_copy ;;
    revoke)      cmd_revoke ;;
    help|--help|-h) cmd_help ;;
    *)
        printf "${RED}Unknown command: %s${RESET}\n\n" "$CMD" >&2
        cmd_help
        exit 1
        ;;
esac
