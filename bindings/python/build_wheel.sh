#!/usr/bin/env bash
# Build a self-contained VGRE wheel: compile the native libraries with CMake,
# bundle them into the Python package, then build the wheel.
#
#   bindings/python/build_wheel.sh [BUILD_DIR]
#
# Produces bindings/python/dist/vgre-*.whl, which `pip install`s with no external
# BLAS/ML dependency — `import vgre` then trains and runs the in-tree LM offline.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BUILD_DIR="${1:-$ROOT/build}"

# Portable CPU count (Linux: nproc, macOS: sysctl).
if command -v nproc >/dev/null 2>&1; then JOBS="$(nproc)";
elif command -v sysctl >/dev/null 2>&1; then JOBS="$(sysctl -n hw.ncpu)";
else JOBS=4; fi

# Native shared-library names differ per OS (Linux .so, macOS .dylib, Windows .dll).
# Bundle EVERY matching native library — including the real versioned file, not just
# the unversioned/soname symlinks. (The Linux soname chain is libvgre.so → .so.0 →
# .so.0.1.0 and macOS is libvgre.dylib → .0.dylib → .0.1.0.dylib; copying only the
# first two leaves a dangling symlink and the loader can't dlopen it — which silently
# made NATIVE_AVAILABLE False in the macOS wheel.) A glob picks up all of them.
case "$(uname -s)" in
    Darwin)               GLOB="libvgre*.dylib"; PRIMARY="libvgre.dylib" ;;
    MINGW*|MSYS*|CYGWIN*) GLOB="vgre*.dll";      PRIMARY="vgre.dll" ;;
    *)                    GLOB="libvgre*.so*";   PRIMARY="libvgre.so" ;;
esac

echo "==> Ensuring native libraries are built in $BUILD_DIR"
if [[ ! -e "$BUILD_DIR/$PRIMARY" ]]; then
    cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" --target vgre vgre_cudart -j"$JOBS"
fi

echo "==> Bundling shared libraries into vgre/lib/"
mkdir -p "$HERE/vgre/lib"
shopt -s nullglob
for lib in "$BUILD_DIR"/$GLOB; do
    cp -P "$lib" "$HERE/vgre/lib/" 2>/dev/null || cp "$lib" "$HERE/vgre/lib/"
done
shopt -u nullglob
echo "==> Bundled:"; ls -l "$HERE/vgre/lib/" | sed 's/^/    /'

echo "==> Building wheel"
( cd "$HERE" && python -m build --wheel )

# The wheel bundles a platform-specific native library (.so/.dylib/.dll), so it is
# NOT platform-agnostic — setuptools tags a pure-Python package `py3-none-any`, which
# would make three OS wheels collide in one release and let pip install a Linux wheel
# on macOS. Retag it with this host's platform (e.g. py3-none-linux_x86_64) so each
# OS ships a distinct, correctly-resolved wheel. Falls back to the `any` wheel if the
# `wheel` CLI is unavailable.
PLAT="$(python - <<'PY'
import sysconfig
print(sysconfig.get_platform().replace('-', '_').replace('.', '_'))
PY
)"
if [[ -n "$PLAT" ]] && python -m wheel version >/dev/null 2>&1; then
    for w in "$HERE"/dist/*-any.whl; do
        [[ -e "$w" ]] || continue
        echo "==> Retagging $(basename "$w") → platform $PLAT"
        python -m wheel tags --platform-tag "$PLAT" --remove "$w" >/dev/null
    done
fi

echo "==> Done. Wheel(s):"
ls -1 "$HERE"/dist/*.whl
