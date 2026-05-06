#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build_test}"

CSV_MODE=0
if [[ "${1:-}" == "--csv" ]]; then
  CSV_MODE=1
fi

cd "$PROJECT_ROOT"

echo "[1/3] Configure: cmake -S . -B \"$BUILD_DIR\""
cmake -S . -B "$BUILD_DIR"

echo "[2/3] Build benchmark targets"
cmake --build "$BUILD_DIR" --target \
  test_memory_scheduler_benchmarks \
  test_scheduler_benchmark \
  test_vector_engine_benchmark \
  test_memory_manager_benchmark \
  -j 4

echo "[3/3] Run benchmarks"
if [[ $CSV_MODE -eq 1 ]]; then
  "$BUILD_DIR/tests/test_memory_scheduler_benchmarks"
  "$BUILD_DIR/tests/test_scheduler_benchmark" --csv
  "$BUILD_DIR/tests/test_vector_engine_benchmark" --csv
  "$BUILD_DIR/tests/test_memory_manager_benchmark" --csv
else
  "$BUILD_DIR/tests/test_memory_scheduler_benchmarks"
  "$BUILD_DIR/tests/test_scheduler_benchmark"
  "$BUILD_DIR/tests/test_vector_engine_benchmark"
  "$BUILD_DIR/tests/test_memory_manager_benchmark"
fi
