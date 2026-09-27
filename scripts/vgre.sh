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

VGRE_VERSION="0.1.0"

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

_python() {
    if command -v python3 >/dev/null 2>&1; then echo python3;
    elif command -v python >/dev/null 2>&1; then echo python;
    else echo ""; fi
}

usage() {
    cat <<'EOF'
vgre — run CUDA and a local language model on the CPU (no GPU required).

Usage: vgre <command> [options]

Model & runtime (Python package — pip install vgre, or the wheel):
  info                 show version, native backend, library path, platform
  generate             generate text (trains a tiny demo model if no --model)
  train                train a small language model on a text corpus
  tokenize             byte / BPE tokenization helpers

Cluster & node (source install — install_local.sh):
  start                start a master or worker node        (vgre-start)
  worker               run the worker binary directly       (vgre-worker)
  token                manage the shared cluster auth token (vgre-token)
  discover             find / publish the master's public IP (vgre-discover)
  dashboard            launch the Flutter monitoring dashboard (vgre-dashboard)

Other:
  version, --version   print the VGRE version and native backend status
  help,    --help      show this help

Run `vgre <command> --help` for a command's own options.
EOF
}

version() {
    printf 'vgre %s\n' "$VGRE_VERSION"
    _py=$(_python)
    if [ -n "$_py" ] && "$_py" -c "import vgre" >/dev/null 2>&1; then
        "$_py" -m vgre version 2>/dev/null | grep -iE 'native' || true
    else
        printf 'native backend: python package not installed (pip install vgre)\n'
    fi
}

# Delegate a model/tokenizer subcommand to the Python CLI.
run_python_cli() {
    _py=$(_python)
    if [ -z "$_py" ]; then
        echo "vgre: python3 is required for '$1'." >&2; exit 127
    fi
    if ! "$_py" -c "import vgre" >/dev/null 2>&1; then
        echo "vgre: the Python package is not installed. Install it with:" >&2
        echo "        pip install vgre        # or a release wheel" >&2
        exit 127
    fi
    exec "$_py" -m vgre "$@"
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
    info|generate|train|tokenize)   run_python_cli "$cmd" "$@" ;;
    start)      run_sibling vgre-start "$@" ;;
    worker)     run_sibling vgre-worker "$@" ;;
    token)      run_sibling vgre-token "$@" ;;
    discover)   run_sibling vgre-discover "$@" ;;
    dashboard)  run_sibling vgre-dashboard "$@" ;;
    *)
        echo "vgre: unknown command '$cmd'" >&2
        echo >&2
        usage >&2
        exit 2 ;;
esac
