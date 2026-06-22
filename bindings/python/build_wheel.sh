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

echo "==> Done. Wheel(s):"
ls -1 "$HERE"/dist/*.whl
