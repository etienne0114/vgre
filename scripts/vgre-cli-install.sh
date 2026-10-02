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

# Single source of truth for CLI wrapper scripts (basename without .sh for commands).
vgre_cli_script_names() {
    printf '%s\n' \
        vgre \
        vgre-token \
        vgre-start \
        vgre-discover \
        vgre-connect-check \
        vgre-print-linux-setup \
        vgre-mac-worker-setup
}

vgre_cli_script_files() {
    _name="$1"
    printf '%s.sh\n' "$_name"
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

    # zsh always sources ~/.zshenv (login, non-login, and IDE terminals).
    ZSHENV="$HOME/.zshenv"
    if [ -f "$ZSHENV" ]; then
        if ! grep -q '\.local/bin' "$ZSHENV" 2>/dev/null; then
            printf '\n# VGRE CLI tools\nexport PATH="%s:$PATH"\n' "$BIN_DIR" >> "$ZSHENV"
        fi
    else
        printf '# zsh environment — created by VGRE installer\n# VGRE CLI tools\nexport PATH="%s:$PATH"\n' "$BIN_DIR" > "$ZSHENV"
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
    vgre_cli_script_names | while read -r _name; do
        _script=$(vgre_cli_script_files "$_name")
        if [ -f "$SCRIPT_DIR/$_script" ]; then
            chmod +x "$SCRIPT_DIR/$_script" 2>/dev/null || true
            ln -sf "$SCRIPT_DIR/$_script" "$BIN_DIR/$_name"
        fi
    done

    if command -v brew >/dev/null 2>&1; then
        _brew_bin="$(brew --prefix 2>/dev/null)/bin"
        if [ -n "$_brew_bin" ] && [ -d "$_brew_bin" ] && [ -w "$_brew_bin" ]; then
            vgre_cli_script_names | while read -r _name; do
                _script=$(vgre_cli_script_files "$_name")
                if [ -f "$SCRIPT_DIR/$_script" ]; then
                    ln -sf "$SCRIPT_DIR/$_script" "$_brew_bin/$_name"
                fi
            done
        fi
    fi
}

# Install the vgre Python package (model CLI: version / info / pull / chat / generate / train / tokenize)
# into a dedicated venv at ~/.vgre/venv. Debian/Ubuntu mark the system Python
# "externally managed" (PEP 668), so a venv is the only clean, sudo-free path.
# Best-effort: prints guidance and returns non-zero rather than aborting.
#   $1 = path to the Python package (…/bindings/python)
#   $2 = directory containing the freshly-built libvgre shared libraries
# Return: 0 ok · 2 venv module missing (caller may install python3-venv + retry)
#         · 1 other failure
vgre_copy_python_native_libraries() {
    _native_dir="$1"
    _pyenv="$2"
    [ -d "$_native_dir" ] || return 1

    _package_lib_dir=$("$_pyenv/bin/python" -c \
        'import os, sysconfig; print(os.path.join(sysconfig.get_paths()["purelib"], "vgre", "lib"))') || return 1
    _package_dir=$(dirname "$_package_lib_dir")
    [ -d "$_package_dir" ] || return 1
    mkdir -p "$_package_lib_dir" || return 1

    _copied_main=0
    for _native_file in "$_native_dir"/libvgre*; do
        [ -f "$_native_file" ] || [ -L "$_native_file" ] || continue
        cp -P "$_native_file" "$_package_lib_dir/" || return 1
        case "$(basename "$_native_file")" in
            libvgre.so|libvgre.dylib) _copied_main=1 ;;
        esac
    done
    [ "$_copied_main" -eq 1 ]
}

vgre_setup_python_cli() {
    _pkg_dir="$1"
    _native_dir="${2:-}"
    _pyenv="${VGRE_VENV:-$HOME/.vgre/venv}"
    [ -d "$_pkg_dir" ] || return 1

    _pybin=""
    for _p in python3 python; do
        if command -v "$_p" >/dev/null 2>&1; then _pybin="$_p"; break; fi
    done
    if [ -z "$_pybin" ]; then
        printf '  [WARN] python3 not found — model subcommands (vgre pull/chat/generate/train/tokenize) unavailable\n'
        return 1
    fi

    if [ ! -x "$_pyenv/bin/python" ]; then
        if ! "$_pybin" -m venv "$_pyenv" >/dev/null 2>&1; then
            printf '  [WARN] python venv module missing — install it (e.g. sudo apt-get install python3-venv), then re-run.\n'
            return 2
        fi
    fi

    "$_pyenv/bin/python" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
    if ! "$_pyenv/bin/python" -m pip install --quiet "$_pkg_dir" >/dev/null 2>&1; then
        printf '  [WARN] pip install failed (offline?). Retry:  %s/bin/pip install "%s"\n' "$_pyenv" "$_pkg_dir"
        return 1
    fi

    if [ -n "$_native_dir" ] && ! vgre_copy_python_native_libraries "$_native_dir" "$_pyenv"; then
        printf '  [WARN] could not sync native libraries from %s into the Python package\n' "$_native_dir"
        return 1
    fi

    if "$_pyenv/bin/python" -c \
        'from vgre import LanguageModel; model = LanguageModel(vocab=256, n_layer=1, d_model=8, n_head=1, d_ff=16, max_seq=8); model.close()' \
        >/dev/null 2>&1; then
        printf '  [OK] vgre model CLI ready — try:  vgre chat  or  vgre pull smollm2\n'
        return 0
    fi
    printf '  [WARN] model CLI native API check failed; re-run the source sync after rebuilding libvgre\n'
    return 1
}

# When run directly: bash scripts/vgre-cli-install.sh
if [ "$(basename "$0")" = "vgre-cli-install.sh" ]; then
    _VGRE_CLI_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
    vgre_install_cli_symlinks "$_VGRE_CLI_DIR"
    vgre_ensure_cli_path
    printf 'VGRE CLI installed:\n'
    vgre_cli_script_names | while read -r _name; do
        printf '  %s/%s\n' "$(vgre_cli_bin_dir)" "$_name"
    done
    if command -v brew >/dev/null 2>&1 && [ -w "$(brew --prefix 2>/dev/null)/bin" ]; then
        printf '  (also linked in %s/bin)\n' "$(brew --prefix)"
    fi
    printf '\nOpen a new terminal or run:  source ~/.vgre/env\n'
fi
