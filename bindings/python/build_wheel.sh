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

echo "==> Ensuring native libraries are built in $BUILD_DIR"
if [[ ! -f "$BUILD_DIR/libvgre.so" ]]; then
    cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD_DIR" --target vgre vgre_cudart -j"$(nproc)"
fi

echo "==> Bundling shared libraries into vgre/lib/"
mkdir -p "$HERE/vgre/lib"
for so in libvgre.so libvgre.so.0 libvgre.so.0.1.0 libvgre_cudart.so libvgre_cudart.so.0; do
    [[ -e "$BUILD_DIR/$so" ]] && cp -aP "$BUILD_DIR/$so" "$HERE/vgre/lib/" || true
done

echo "==> Building wheel"
( cd "$HERE" && python -m build --wheel )

echo "==> Done. Wheel(s):"
ls -1 "$HERE"/dist/*.whl
