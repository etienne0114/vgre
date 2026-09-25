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
case "$(uname -s)" in
    Darwin) LIBS="libvgre.dylib libvgre.0.dylib libvgre_cudart.dylib libvgre_cudart.0.dylib"; PRIMARY="libvgre.dylib" ;;
    MINGW*|MSYS*|CYGWIN*) LIBS="vgre.dll vgre_cudart.dll"; PRIMARY="vgre.dll" ;;
    *)      LIBS="libvgre.so libvgre.so.0 libvgre.so.0.1.0 libvgre_cudart.so libvgre_cudart.so.0"; PRIMARY="libvgre.so" ;;
esac

echo "==> Ensuring native libraries are built in $BUILD_DIR"
if [[ ! -e "$BUILD_DIR/$PRIMARY" ]]; then
    cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" --target vgre vgre_cudart -j"$JOBS"
fi

echo "==> Bundling shared libraries into vgre/lib/"
mkdir -p "$HERE/vgre/lib"
for lib in $LIBS; do
    if [[ -e "$BUILD_DIR/$lib" ]]; then cp -P "$BUILD_DIR/$lib" "$HERE/vgre/lib/" || cp "$BUILD_DIR/$lib" "$HERE/vgre/lib/"; fi
done

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
