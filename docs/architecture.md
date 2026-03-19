# VGRE Architecture

## System Overview

VGRE (Virtual GPU Runtime Engine) is a modular GPU emulation system composed of six major subsystems that work together to translate GPU workloads into optimized CPU execution.

## Component Diagram

```
                    ┌─────────────────────┐
                    │   Application       │
                    │ (PyTorch/TF/Custom) │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              │        API Layer              │
              │  ┌──────────┐ ┌──────────┐    │
              │  │  CUDA    │ │  OpenCL  │    │
              │  │Interceptor│ │ Adapter  │    │
              │  └─────┬────┘ └────┬─────┘    │
              └────────┴──────────┴───────────┘
                              │
              ┌───────────────┴───────────────┐
              │       Runtime Engine          │
              │  (Orchestrates all subsystems) │
              └─┬───────┬──────────┬──────────┘
                │       │          │
    ┌───────────┴┐ ┌────┴────┐ ┌──┴──────────┐
    │  Compiler  │ │  Core   │ │  Runtime    │
    │  ┌───────┐ │ │┌───────┐│ │┌───────────┐│
    │  │Kernel │ │ ││Memory ││ ││CPU Parallel││
    │  │Parser │ │ ││Manager││ ││Executor   ││
    │  └───┬───┘ │ │└───────┘│ │└───────────┘│
    │  ┌───┴───┐ │ │┌───────┐│ │┌───────────┐│
    │  │ LLVM  │ │ ││Sched- ││ ││Vector     ││
    │  │Engine │ │ ││uler   ││ ││Engine     ││
    │  └───────┘ │ │└───────┘│ │└───────────┘│
    └────────────┘ └─────────┘ └─────────────┘
                              │
              ┌───────────────┴───────────────┐
              │       Advanced Features        │
              │ ┌───────┐ ┌───────┐ ┌───────┐ │
              │ │Hybrid │ │Adaptive│ │Memory │ │
              │ │Compute│ │Exec.  │ │Compress│ │
              │ └───────┘ └───────┘ └───────┘ │
              │ ┌───────────────────────────┐  │
              │ │   Runtime Profiler        │  │
              │ └───────────────────────────┘  │
              └────────────────────────────────┘
```

## Subsystem Details

### 1. API Layer

**Purpose:** Present CUDA/OpenCL-compatible APIs to applications.

| Component | File | Role |
|-----------|------|------|
| `CUDAInterceptor` | `cuda_interceptor.cpp` | CUDA Runtime API shim (`cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel`, etc.) |
| `OpenCLAdapter` | `opencl_adapter.cpp` | OpenCL API compatibility (`clCreateBuffer`, `clEnqueueNDRangeKernel`, etc.) |

Both route all operations through the `RuntimeEngine`.

### 2. Core

**Purpose:** Virtual GPU device, memory management, scheduling.

| Component | Role |
|-----------|------|
| `VirtualGPUDevice` | Virtualizes GPU hardware — device properties, context, streams |
| `MemoryManager` | Manages VRAM-equivalent memory via aligned host allocations |
| `Scheduler` | Work-stealing thread pool distributing blocks across cores |
| `RuntimeEngine` | Top-level orchestrator wiring all subsystems together |

### 3. Compiler

**Purpose:** Parse GPU kernels and translate to CPU-executable code.

| Component | Role |
|-----------|------|
| `KernelParser` | Tokenizes and parses CUDA-like kernel source |
| `LLVMTranslationEngine` | Multi-stage LLVM ORC JIT pipeline with host-native vectorization (AVX2/AVX-512/FMA) and aggressive O3 optimization |

### 4. Runtime

**Purpose:** Execute compiled kernels on CPU with parallelism.

| Component | Role |
|-----------|------|
| `CPUParallelExecutor` | Maps CUDA 3D grid to OpenMP parallel for loops |
| `VectorEngine` | SIMD-accelerated math (add, mul, FMA, dot, sum) with AVX2/SSE4 |

### 5. Advanced Features

| Component | Role |
|-----------|------|
| `HybridComputeManager` | Detects CPU/iGPU/remote nodes; routes workloads |
| `AdaptiveExecutionEngine` | Profiles kernels; auto-tunes thread count and SIMD width |
| `MemoryCompression` | LZ4-style fast compression for memory transfers |
| `RuntimeProfiler` | Per-kernel timing, throughput, GFLOPS; JSON export |

## Data Flow: Kernel Launch

```
1. App calls cudaLaunchKernel("vector_add", source, grid, block, args)
2. CUDAInterceptor → RuntimeEngine.launchKernel(...)
3. RuntimeEngine → KernelParser.parse(source) → KernelIR
4. RuntimeEngine → LLVMTranslationEngine.translate(ir) → CompiledKernelFn
5. RuntimeEngine → CPUParallelExecutor.execute(fn, grid, block, args)
6. CPUParallelExecutor → OpenMP parallel for over blocks
7. Each block: CompiledKernelFn runs with AVX2 vectorization
8. Results written directly to VRAM-equivalent host memory
```

## Performance Optimization Strategy

1. **Pattern matching** — common kernels (vector ops, reductions, matmul) get highly optimized implementations
2. **SIMD vectorization** — AVX2 processes 8 floats per instruction
3. **OpenMP parallelism** — blocks distributed across all CPU cores
4. **Kernel caching** — compiled kernels cached by name to avoid re-translation
5. **Adaptive tuning** — runtime profiler feeds optimal parameters back
6. **Memory alignment** — 64-byte aligned allocations for cache-line efficiency

## Feature Coverage

For a current snapshot of which features are fully implemented vs simulated or partial, see `docs/feature_matrix.md`.
