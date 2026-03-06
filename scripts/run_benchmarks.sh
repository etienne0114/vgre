#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$DIR")"

cd "$PROJECT_ROOT"

# Use the virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="$PROJECT_ROOT/bindings/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$PROJECT_ROOT/build:$LD_LIBRARY_PATH"

echo "Executing REAL VGRE Performance Benchmarks..."

python3 tests/python/benchmark.py

echo ""
echo "Benchmarking verification complete."
