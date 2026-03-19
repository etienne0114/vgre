# VGRE — Virtual GPU Runtime Engine

**A high-fidelity GPU virtualization layer** that allows AI frameworks to run GPU workloads on CPUs by translating GPU instructions into highly optimized parallel CPU execution.

## Overview

VGRE intercepts CUDA/OpenCL operations and translates them into CPU-parallel instructions using:
- **OpenMP** multithreading
- **AVX2/AVX-512/FMA** SIMD vectorization (host-native auto-detection)
- **LLVM ORC JIT** kernel translation with aggressive O3 optimization
- **Adaptive execution** with runtime profiling and hardware calibration

This enables PyTorch, TensorFlow, and other GPU-dependent frameworks to run on machines **without a physical GPU**.

## Quick Start

### Build

```bash
cd virtual-gpu-runtime
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Run Tests

```bash
cd build
ctest --output-on-failure
```

### Run Vector Addition Example

```bash
./build/examples/vector_addition
```

### Python

```python
import numpy as np
import math
from vgre import Runtime, VirtualDevice
from vgre.kernel import vector_add_kernel, Dim3

dev = VirtualDevice()
print(f"VGRE Device: {dev.get_properties().name}")

rt = Runtime()
rt.init(enable_profiling=False)

N = 10_000_000
block_size = 256
grid_size = math.ceil(N / 256)
padded_N = grid_size * block_size

a = np.random.randn(padded_N).astype(np.float32)
b = np.random.randn(padded_N).astype(np.float32)
c = np.zeros(padded_N, dtype=np.float32)

kernel = vector_add_kernel()
rt.launch(kernel, Dim3(grid_size), Dim3(block_size), [a, b, c], parallel=True)
rt.synchronize()

print(f"Result: {c[:5]} ...")
rt.shutdown()
```

### Benchmarks

To reproduce the engine throughput measurements:
```bash
./scripts/run_benchmarks.sh
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Application (PyTorch / TensorFlow / Custom)          │
├──────────────────────────────────────────────────────┤
│  API Layer                                            │
│  ┌────────────────┐  ┌────────────────┐              │
│  │ CUDA Interceptor│  │ OpenCL Adapter │              │
│  └────────┬───────┘  └────────┬───────┘              │
├───────────┴──────────────────┴───────────────────────┤
│  Core Engine                                          │
│  ┌────────────┐  ┌───────────────┐  ┌───────────┐   │
│  │Runtime     │→ │Kernel         │→ │LLVM        │   │
│  │Engine      │  │Parser         │  │Translator  │   │
│  └──────┬─────┘  └───────────────┘  └───────────┘   │
│         │                                             │
│  ┌──────┴─────┐  ┌───────────────┐  ┌───────────┐   │
│  │Scheduler   │→ │CPU Parallel   │→ │Vector     │   │
│  │(thread pool)│  │Executor (OMP) │  │Engine(SIMD)│   │
│  └────────────┘  └───────────────┘  └───────────┘   │
├──────────────────────────────────────────────────────┤
│  Infrastructure                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │Memory    │ │Hybrid    │ │Adaptive  │ │Runtime   ││
│  │Manager   │ │Compute   │ │Execution │ │Profiler  ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘│
└──────────────────────────────────────────────────────┘
```

## Project Structure

```
virtual-gpu-runtime/
├── include/vgre/          # Public headers
│   ├── common/            #   types, error_codes, logger
│   ├── core/              #   device, memory, scheduler, runtime
│   ├── compiler/          #   kernel parser, LLVM translation
│   ├── api/               #   CUDA interceptor, OpenCL adapter
│   ├── runtime/           #   parallel executor, vector engine
│   └── advanced/          #   hybrid compute, adaptive, compression, profiler
├── src/                   # Source implementations
├── bindings/python/       # Python bindings
├── tests/                 # Unit + integration tests
├── examples/              # Runnable examples
└── docs/                  # Documentation
```

## Requirements

- **CMake** ≥ 3.16
- **GCC** ≥ 9 or **Clang** ≥ 10 (C++17)
- **OpenMP**
- **LLVM-18** (Required for Dynamic JIT Execution with ORC API)
- **Python 3** + **NumPy** (for Python bindings)

## Documentation
For complete technical details on the architecture, thread-pool scheduling, and JIT pointer mapping, please visit:
- [Developer Guide](docs/developer_guide.md)
- [API Reference](docs/api_reference.md)

## License

MIT License — See [LICENSE](LICENSE).
