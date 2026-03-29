# VGRE — Virtual GPU Runtime Engine

**A CUDA emulation runtime** that allows CUDA applications to run on CPU without a physical GPU.

> **PROJECT STATUS**: BETA — Production hardening in progress (~75% of planned features implemented)

## What is VGRE?

VGRE intercepts CUDA and OpenCL API calls and executes kernels on CPU using:
- LLVM-18 ORC JIT (Clang AST parsing → LLVM IR → native code, O3-optimised)
- OpenMP + SIMD (AVX2/AVX-512) parallel execution
- Unified Virtual Memory (UVM) with OS page-fault signal handler
- Hardware-backed token storage (Linux keyring, macOS Keychain, Windows CredMan, TPM 2.0)
- TCP cluster networking with authenticated AES-256-CTR encrypted channels
- LZ4 memory compression for distributed transfers

**Use Cases**:
- ✅ Learning CUDA without a GPU
- ✅ Development, CI/CD testing
- ✅ Moderate CUDA applications (vector ops, matrix math, graph workloads)
- ✅ Distributed CPU-cluster compute via VSBP protocol
- ⚠️  Complex, performance-critical GPU workloads (10–100× slower than real GPU)
- ⚠️  Applications that depend on GPU-specific memory bandwidth characteristics

## Current Status

**Overall Completion**: ~75% of planned features
**Production Readiness**: ~75% for CPU-based CUDA emulation workloads

### What Works ✅
- Full CUDA Runtime API (~100 functions: memory, streams, events, device, textures)
- OpenCL 1.2 compatibility layer
- JIT kernel compilation with persistent disk + memory cache (0ms on cache hit)
- 1D/2D/3D texture and surface sampling with multiple filter/addressing modes
- UVM managed memory (`cudaMallocManaged`) with OS-level page fault handling
- CUDA Graphs (graph capture, instantiation, replay, dynamic update)
- Cooperative kernel launch with grid-wide barrier synchronisation
- Stream-ordered memory pools (`cudaMallocAsync` / `cudaFreeAsync`)
- CUDA Driver API (cuInit, cuCtxCreate, cuMemAlloc, cuModuleLoad, cuLaunchKernel)
- P2P peer device access and transfers
- Kernel fusion (consecutive compatible kernels fused into single JIT compilation)
- TCP cluster networking: multi-node partitioned kernel dispatch, telemetry aggregation
- Authenticated encrypted cluster channels (HMAC-SHA256 + AES-256-CTR)
- Hardware-backed auth token storage (keyring/Keychain/CredMan/TPM 2.0)
- Adaptive execution engine: auto-tunes thread count for each kernel
- Chrome trace export (`toChromeTraceJSON`) and C API telemetry
- Python bindings (`vgre_c_api` via ctypes), NumPy-compatible

### Known Limitations ⚠️
- 10–100× slower than real GPU (expected; CPU execution)
- AES-256-CTR cipher: software implementation (no AES-NI acceleration yet)
- Temperature sensing: fully implemented on Linux; heuristic on Windows/macOS
- Vector SIMD width: auto-tuned at runtime by `AdaptiveExecutionEngine`
- Fuzzing suite and CI/CD pipeline: Phase 5 (planned)
- No OpenCL 2.0+ features (SVM, pipes, subgroups)

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
│  │Runtime     │→ │Clang Parser   │→ │LLVM        │   │
│  │Engine      │  │(AST + Cache)  │  │Translator  │   │
│  └──────┬─────┘  └───────────────┘  └───────────┘   │
│         │                                             │
│  ┌──────┴─────┐  ┌───────────────┐  ┌───────────┐   │
│  │Scheduler   │→ │CPU Parallel   │→ │Vector     │   │
│  │(thread pool)│  │Executor (OMP) │  │Engine(SIMD)│   │
│  └────────────┘  └───────────────┘  └───────────┘   │
├──────────────────────────────────────────────────────┤
│  Infrastructure                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │Memory    │ │TCP       │ │Adaptive  │ │Runtime   ││
│  │Manager   │ │Cluster   │ │Execution │ │Profiler  ││
│  │(UVM+SHM) │ │(VSBP)    │ │Engine    │ │(Chrome)  ││
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘│
└──────────────────────────────────────────────────────┘
```

## Project Structure

```
virtual-gpu-runtime/
├── include/vgre/          # Public headers
│   ├── common/            #   types, error_codes, logger, input_validation
│   ├── core/              #   device, memory (UVM), scheduler, runtime, graphs
│   ├── compiler/          #   Clang kernel parser, LLVM translation engine, cache
│   ├── api/               #   CUDA interceptor, OpenCL adapter
│   ├── runtime/           #   parallel executor, vector engine
│   └── advanced/          #   TCP cluster, adaptive engine, profiler, compression, IPC
├── src/                   # Source implementations
├── bindings/python/       # Python bindings (ctypes over C API)
├── tests/                 # 55+ unit + integration tests
├── examples/              # Runnable examples
└── docs/                  # Documentation
```

## Requirements

- **CMake** ≥ 3.16
- **GCC** ≥ 9 or **Clang** ≥ 10 (C++17)
- **OpenMP**
- **LLVM-18** (Required for JIT with ORC API)
- **Python 3** + **NumPy** (for Python bindings)
- **keyutils** (Linux, optional — hardware token storage)
- **tss2-esys** (Linux, optional — TPM 2.0 token storage)

## Documentation
For complete technical details on the architecture, JIT pipeline, cluster protocol, and security model:
- [Developer Guide](docs/developer_guide.md)
- [API Reference](docs/api_reference.md)
- [Architecture](docs/architecture.md)
- [Feature Matrix](docs/feature_matrix.md)

## License

MIT License — See [LICENSE](LICENSE).
