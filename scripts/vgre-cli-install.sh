#!/usr/bin/env sh
# vgre-cli-install.sh — install VGRE CLI symlinks and ensure ~/.local/bin is on PATH.
# POSIX sh; sourced by vgre_sync.sh, install_local.sh, and vgre-token.sh.

# Resolve the real scripts/ path when a caller is invoked via ~/.local/bin symlinks.
vgre_resolve_script_dir() {
    _target="$1"
    while [ -L "$_target" ]; do
        _link=$(readlink "$_target")
        case "$_link" in
            /*) _target="$_link" ;;
            *) _target="$(CDPATH= cd -- "$(dirname "$_target")" && pwd)/$_link" ;;
        esac
    done
    CDPATH= cd -- "$(dirname "$_target")" && pwd
}

vgre_cli_bin_dir() {
    printf '%s\n' "${VGRE_BIN_DIR:-$HOME/.local/bin}"
}

vgre_ensure_cli_path() {
    BIN_DIR=$(vgre_cli_bin_dir)
    VGRE_DIR="${VGRE_DIR:-$HOME/.vgre}"
    ENV_FILE="$VGRE_DIR/env"

    mkdir -p "$BIN_DIR"

    # Persist PATH in ~/.vgre/env (sourced by shell profiles after install).
    mkdir -p "$VGRE_DIR"
    if [ ! -f "$ENV_FILE" ]; then
        : > "$ENV_FILE"
        chmod 600 "$ENV_FILE"
    fi
    if ! grep -q '\.local/bin' "$ENV_FILE" 2>/dev/null; then
        printf '\n# VGRE CLI tools\nexport PATH="%s:${PATH}"\n' "$BIN_DIR" >> "$ENV_FILE"
    fi

    # macOS zsh reads .zprofile for login shells; Linux/macOS also use .bashrc/.zshrc.
    PATH_LINE="export PATH=\"$BIN_DIR:\$PATH\""
    for PROFILE in \
        "$HOME/.zshrc" \
        "$HOME/.zprofile" \
        "$HOME/.bashrc" \
        "$HOME/.bash_profile" \
        "$HOME/.profile"; do
        if [ -f "$PROFILE" ] && ! grep -q '\.local/bin' "$PROFILE" 2>/dev/null; then
            printf '\n# VGRE CLI tools\n%s\n' "$PATH_LINE" >> "$PROFILE"
        fi
    done

    # Make the current shell/session work immediately when sourced.
    case ":$PATH:" in
        *":$BIN_DIR:"*) ;;
        *) export PATH="$BIN_DIR:$PATH" ;;
    esac
}

vgre_install_cli_symlinks() {
    SCRIPT_DIR="$1"
    BIN_DIR=$(vgre_cli_bin_dir)
    mkdir -p "$BIN_DIR"
    for _script in vgre-token.sh vgre-start.sh vgre-discover.sh; do
        if [ -f "$SCRIPT_DIR/$_script" ]; then
            chmod +x "$SCRIPT_DIR/$_script" 2>/dev/null || true
            ln -sf "$SCRIPT_DIR/$_script" "$BIN_DIR/${_script%.sh}"
        fi
    done
}
