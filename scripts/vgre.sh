#!/usr/bin/env sh
# vgre — the unified VGRE command.
#
# Installed into ~/.local/bin by install_local.sh / vgre_sync.sh, so after a
# source install `vgre <command>` works from any directory. It dispatches:
#   * cluster/runtime tools  -> the sibling vgre-<cmd> scripts / binaries
#   * model & tokenizer tools -> the Python package CLI (python3 -m vgre)
#
# Windows ships the same commands as vgre-*.bat / vgre-*.ps1 (see scripts/).
set -eu

VGRE_VERSION="0.1.4"

# Resolve this script's real directory even when invoked via a ~/.local/bin
# symlink, so we can find the sibling vgre-<cmd>.sh wrappers.
_self="$0"
while [ -h "$_self" ]; do
    _dir=$(cd "$(dirname "$_self")" && pwd)
    _link=$(readlink "$_self")
    case "$_link" in
        /*) _self="$_link" ;;
        *)  _self="$_dir/$_link" ;;
    esac
done
SCRIPT_DIR=$(cd "$(dirname "$_self")" && pwd)

# The managed virtualenv that vgre_sync.sh / install_local.sh populate with the
# vgre Python package (numpy-based). Overridable with VGRE_VENV.
_VGRE_VENV_PY="${VGRE_VENV:-$HOME/.vgre/venv}/bin/python"

# Source installs keep the current native library outside the Python package.
# Prefer that library over the copy bundled in site-packages, which can lag
# behind after a source sync and miss newly added C API symbols.
_select_native_library() {
    if [ -n "${VGRE_LIB_PATH:-}" ] && [ -f "$VGRE_LIB_PATH" ]; then
        return
    fi

    case "$(uname -s 2>/dev/null || echo Linux)" in
        Darwin) _native_name="libvgre.dylib" ;;
        *) _native_name="libvgre.so" ;;
    esac

    _install_dir="${VGRE_INSTALL_DIR:-$HOME/.local/share/VGRE}"
    for _candidate in \
        "$_install_dir/lib/$_native_name" \
        "$SCRIPT_DIR/../build/$_native_name" \
        "$SCRIPT_DIR/../build/lib/$_native_name"; do
        if [ -f "$_candidate" ]; then
            export VGRE_LIB_PATH="$_candidate"
            return
        fi
    done
}

_select_native_library

# Echo the first interpreter that can actually `import vgre`, or return 1.
# Priority: $VGRE_PYTHON → the managed venv → system python3/python.
_python_with_vgre() {
    if [ -n "${VGRE_PYTHON:-}" ] && "$VGRE_PYTHON" -c 'import vgre' >/dev/null 2>&1; then
        echo "$VGRE_PYTHON"; return 0
    fi
    if [ -x "$_VGRE_VENV_PY" ] && "$_VGRE_VENV_PY" -c 'import vgre' >/dev/null 2>&1; then
        echo "$_VGRE_VENV_PY"; return 0
    fi
    for _p in python3 python; do
        if command -v "$_p" >/dev/null 2>&1 && "$_p" -c 'import vgre' >/dev/null 2>&1; then
            echo "$_p"; return 0
        fi
    done
    return 1
}

# Print correct, copy-pasteable setup steps (works on PEP 668 / Debian).
_emit_setup_help() {
    _repo=$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd || echo "<vgre-source>")
    cat >&2 <<EOF
vgre: the Python model package isn't set up, so '$1' can't run yet.

The quickest fix is to re-run the sync, which installs it into a venv for you:
    ./scripts/vgre_sync.sh

Or set it up by hand (Debian/Ubuntu's system Python is "externally managed",
so use a venv — do NOT 'pip install' into the system Python):
    python3 -m venv ~/.vgre/venv
    ~/.vgre/venv/bin/pip install vgre          # from PyPI
    # …or from this source checkout:
    ~/.vgre/venv/bin/pip install $_repo/bindings/python

Then re-run:  vgre $1 ...
(If 'python3 -m venv' fails: sudo apt-get install python3-venv)
EOF
}

usage() {
    cat <<'EOF'
vgre — run CUDA and a local language model on the CPU (no GPU required).

Usage: vgre <command> [options]

Model & runtime (Python package — set up by ./scripts/vgre_sync.sh):
  info                 show version, native backend, library path, platform
  chat                 interactive chat with a model
  pull                 download and configure a model from a repository
  generate             generate text (trains a tiny demo model if no --model)
  train                train a small language model on a text corpus
  tokenize             byte / BPE tokenization helpers

Cluster & node (source install — install_local.sh):
  start                start a master or worker node
  worker               run the worker binary directly
  token                manage the shared cluster auth token
  discover             find / publish the master's public IP
  connect-check        verify connectivity to a master before starting a worker
  dashboard            launch the Flutter monitoring dashboard

Other:
  version, --version   print the VGRE version and native backend status
  help,    --help      show this help

Run `vgre <command> --help` for a command's own options.
EOF
}

version() {
    printf 'vgre %s\n' "$VGRE_VERSION"
    if _py=$(_python_with_vgre); then
        "$_py" -m vgre version 2>/dev/null | grep -iE 'native' || true
    else
        printf 'native backend: model package not set up — run ./scripts/vgre_sync.sh\n'
    fi
}

# Delegate a model/tokenizer subcommand to the Python CLI.
run_python_cli() {
    _sub="$1"; shift
    # 1. A ready interpreter (venv / $VGRE_PYTHON / system) that has the package.
    if _py=$(_python_with_vgre); then
        exec "$_py" -m vgre "$_sub" "$@"
    fi
    # 2. Source-tree fallback: run straight from bindings/python when this
    #    dispatcher lives in a checkout and the interpreter already has numpy.
    _src="$SCRIPT_DIR/../bindings/python"
    if [ -d "$_src" ]; then
        for _p in python3 python; do
            if command -v "$_p" >/dev/null 2>&1 && \
               PYTHONPATH="$_src${PYTHONPATH:+:$PYTHONPATH}" "$_p" -c 'import vgre' >/dev/null 2>&1; then
                exec env PYTHONPATH="$_src${PYTHONPATH:+:$PYTHONPATH}" "$_p" -m vgre "$_sub" "$@"
            fi
        done
    fi
    # 3. Nothing works — print correct setup steps.
    _emit_setup_help "$_sub"
    exit 127
}

# Delegate a cluster subcommand to a sibling vgre-<name> script/binary.
run_sibling() {
    _name="$1"; shift
    for _cand in "$SCRIPT_DIR/$_name.sh" "$SCRIPT_DIR/$_name" \
                 "${VGRE_INSTALL_DIR:-$HOME/.local/share/VGRE}/$_name" \
                 "$HOME/.local/bin/$_name"; do
        if [ -x "$_cand" ]; then exec "$_cand" "$@"; fi
    done
    echo "vgre: '$_name' is not available in this install." >&2
    echo "      Cluster/dashboard tools come from the source build (install_local.sh)." >&2
    exit 127
}

cmd="${1:-help}"
[ "$#" -gt 0 ] && shift || true

case "$cmd" in
    -h|--help|help)        usage ;;
    -V|--version|version)  version ;;
    info|chat|pull|generate|train|tokenize)   run_python_cli "$cmd" "$@" ;;
    start)      run_sibling vgre-start "$@" ;;
    worker)     run_sibling vgre-worker "$@" ;;
    token)      run_sibling vgre-token "$@" ;;
    discover)   run_sibling vgre-discover "$@" ;;
    connect-check) run_sibling vgre-connect-check "$@" ;;
    dashboard)  run_sibling vgre-dashboard "$@" ;;
    *)
        echo "vgre: unknown command '$cmd'" >&2
        echo >&2
        usage >&2
        exit 2 ;;
esac
