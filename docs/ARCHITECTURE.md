# VGRE Architecture & Design

**Version**: 1.0.0  
**Last Updated**: 2026-05-06

---

## System Overview

VGRE (Virtual GPU Runtime Engine) is a CUDA emulation runtime that intercepts GPU API calls and executes them on CPU hardware using LLVM JIT compilation and OpenMP parallelization.

```
┌─────────────────────────────────────────────────────────────────┐
│ Application (PyTorch, TensorFlow, Custom CUDA)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ API Interception Layer (LD_PRELOAD / DLL Injection)             │
│ - CUDA Runtime API (~100 functions)                             │
│ - OpenCL 1.2 Adapter                                            │
│ - cuBLAS, cuDNN, NCCL Shims                                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Runtime Engine (src/core/runtime_engine.cpp)                    │
│ - Kernel registration & caching                                 │
│ - Stream & event management                                     │
│ - Memory allocation & UVM page fault handling                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Kernel       │  │ Memory       │  │ Scheduler    │
│ Compiler     │  │ Manager      │  │              │
│ (LLVM JIT)   │  │ (UVM)        │  │ (Streams)    │
└──────────────┘  └──────────────┘  └──────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ CPU Parallel Executor (OpenMP + SIMD)                           │
│ - Block-level parallelization                                   │
│ - Thread-level execution                                        │
│ - SIMD vectorization (AVX2/AVX-512)                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Host CPU (Multi-core, Multi-socket)                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. API Layer (`src/api/`)

**Purpose**: Intercept CUDA/OpenCL calls and route them to the runtime engine.

**Key Files**:
- `cuda_interceptor.h` - CUDA API interception
- `opencl_adapter.h` - OpenCL 1.2 compatibility
- `vgre_c_api.h` - C API for Python bindings
- `cublas_shim.cpp` - cuBLAS implementation
- `cudnn_shim.cpp` - cuDNN implementation
- `nccl_shim.cpp` - NCCL collective operations

**Interception Methods**:
- **Linux/macOS**: `LD_PRELOAD` environment variable loads `libvgre_cudart.so`
- **Windows**: DLL replacement or PATH manipulation
- **Python**: ctypes bindings to C API

### 2. Runtime Engine (`src/core/runtime_engine.cpp`)

**Purpose**: Central coordinator for kernel execution, memory management, and stream scheduling.

**Key Responsibilities**:
- Kernel registration and caching
- Stream creation and management
- Event synchronization
- Device information reporting
- Graph capture and replay

**Key Data Structures**:
```cpp
struct RuntimeEngine {
    std::map<KernelId, CompiledKernel> kernel_cache_;
    std::map<StreamId, Stream> streams_;
    std::map<EventId, Event> events_;
    MemoryManager memory_manager_;
    Scheduler scheduler_;
    AdaptiveExecutionEngine adaptive_engine_;
};
```

### 3. Kernel Compiler (`src/compiler/`)

**Purpose**: Translate GPU kernels to CPU-executable code via LLVM JIT.

**Pipeline**:
```
CUDA Kernel Source
    ↓
Clang Parser (Extract AST)
    ↓
Wrapper Generator (Bridge GPU→CPU concepts)
    ↓
Clang Compiler (-O3 -march=native)
    ↓
LLVM ORC JIT (Generate machine code)
    ↓
Executable Function Pointer
```

**Key Files**:
- `clang_kernel_parser.cpp` - Parse CUDA kernel source
- `llvm_translation_engine.cpp` - LLVM IR generation and JIT
- `kernel_cache.cpp` - Persistent disk + memory cache
- `ptx_translator.cpp` - Inline PTX assembly translation

**Caching Strategy**:
- **Memory Cache**: LRU hash table (0ms lookup on hit)
- **Disk Cache**: `~/.vgre/cache/` directory (5–10ms on hit)
- **Cache Key**: Includes compilation flags, kernel source hash, and target CPU features

### 4. Memory Manager (`src/core/memory_manager.cpp`)

**Purpose**: Manage GPU-like memory allocation and Unified Virtual Memory (UVM).

**Memory Types**:
- **Regular Memory** (`cudaMalloc`): Simple host allocation
- **Managed Memory** (`cudaMallocManaged`): UVM with page-fault handling
- **Async Pools** (`cudaMallocAsync`): Stream-ordered allocation

**UVM Implementation**:
```
1. App calls cudaMallocManaged(ptr, size)
2. VGRE allocates memory and marks as NO_ACCESS
3. App accesses memory → Page Fault (SIGSEGV on Linux, VEH on Windows)
4. Signal handler intercepts → marks page as READ+WRITE
5. App continues normally (transparent to application)
6. VGRE tracks dirty pages for cluster synchronization
```

**Key Features**:
- NUMA-aware allocation (≥2MB bound to NUMA node 0)
- Bandwidth calibration (cached process-wide)
- Dirty page tracking for distributed execution
- Memory pool pre-allocation for `__syncthreads` kernels

### 5. Scheduler (`src/core/scheduler.cpp`)

**Purpose**: Manage asynchronous task execution across streams.

**Stream Model**:
- Each CUDA stream has its own task queue
- Tasks are executed serially within a stream
- Multiple streams execute in parallel

**Task Types**:
- Kernel launch
- Memory copy
- Event record
- Synchronization barrier

**Scheduling Algorithm**:
```cpp
for each stream {
    while (task_queue not empty) {
        task = dequeue();
        if (dependencies satisfied) {
            execute(task);
        } else {
            re_enqueue(task);
        }
    }
}
```

### 6. CPU Parallel Executor (`src/runtime/cpu_parallel_executor.cpp`)

**Purpose**: Execute kernels on CPU cores using OpenMP parallelization.

**GPU→CPU Mapping**:
| GPU Concept | CPU Reality |
|---|---|
| Grid (thousands of blocks) | OpenMP parallel loop |
| Block (group of threads) | SIMD vectorization |
| Thread (single worker) | OS thread |
| Shared Memory | Pre-allocated buffer |

**Execution Model**:
```cpp
#pragma omp parallel for schedule(guided)
for (int block_idx = 0; block_idx < grid_size; block_idx++) {
    // Each OpenMP thread executes one block
    execute_block(block_idx);
}
```

**Synchronization**:
- `__syncthreads()`: Sense-reversing barrier in BlockWorkerPool
- `this_grid().sync()`: GridBarrierState with atomic counter
- `cudaStreamSynchronize()`: Drain per-stream task queue

### 7. Adaptive Execution Engine (`src/advanced/adaptive_execution_engine.cpp`)

**Purpose**: Auto-tune thread count and performance predictions.

**Calibration Process**:
1. Run micro-benchmarks on startup
2. Measure peak GFLOPS and memory bandwidth
3. Store results in process-wide cache
4. Use for telemetry and performance predictions

**Performance Metrics**:
- Kernel execution time (wall-clock)
- Memory bandwidth (GB/s)
- GFLOPS (floating-point operations per second)
- Coalescing efficiency

### 8. TCP Cluster Manager (`src/advanced/tcp_cluster.cpp`)

**Purpose**: Distribute kernel execution across multiple machines.

**Architecture**:
```
Master Node                    Network (TCP+AES)              Worker Nodes
─────────────                  ──────────────────             ────────────
cudaMalloc(ptr, N)
launchRemoteKernel()
  ├── streamArgumentsToWorker()
  │     └── sendPointerArg() ──────────────────────────────→ Receive data
  └── LAUNCH_KERNEL() ──────────────────────────────────────→ Execute kernel
                                                              ├── cuMemAlloc
                                                              ├── cuMemcpyHtoD
                                                              ├── cuLaunchKernel
                                                              └── cuMemcpyDtoH
                      ←─────────────────────────────────────── Return results
```

**Security**:
- HMAC-SHA256 authentication handshake
- AES-256-CTR encryption for all traffic
- 256-bit replay bitmap prevents reuse
- Hardware-backed token storage (keyring/Keychain/CredMan/TPM)

**Discovery**:
- UDP broadcast on local subnet (every 2 seconds)
- Manual node list via `VGRE_CLUSTER_NODES` env var
- Mesh topology support via `VGRE_MESH_PEERS`

---

## Execution Flow

### Kernel Launch Sequence

```
1. Application calls cudaLaunchKernel(kernel_id, grid, block, args)
   ↓
2. API Layer intercepts call
   ├── Validate arguments
   ├── Deep-copy argument pointers (prevent GC invalidation)
   └── Submit to RuntimeEngine
   ↓
3. RuntimeEngine::launchKernel()
   ├── Look up kernel in cache
   ├── If not cached:
   │   ├── Parse kernel source (Clang)
   │   ├── Generate LLVM IR
   │   ├── JIT compile (LLVM ORC)
   │   └── Cache result
   ├── Create task with kernel, grid, block, args
   └── Submit to Scheduler
   ↓
4. Scheduler::enqueueTask()
   ├── Find stream
   ├── Add task to stream queue
   └── Wake up worker thread
   ↓
5. Worker Thread
   ├── Dequeue task from stream
   ├── Check dependencies
   ├── If ready: submit to CPUParallelExecutor
   └── If not ready: re-enqueue
   ↓
6. CPUParallelExecutor::execute()
   ├── Create OpenMP parallel region
   ├── For each block:
   │   ├── Allocate shared memory buffer
   │   ├── Call compiled kernel function
   │   ├── Synchronize threads (__syncthreads)
   │   └── Free shared memory
   └── Return to scheduler
   ↓
7. Scheduler marks task complete
   ├── Signal any waiting events
   └── Dequeue next task from stream
```

### Memory Access Sequence (UVM)

```
1. Application accesses managed memory
   ↓
2. CPU page fault (SIGSEGV on Linux, VEH on Windows)
   ↓
3. Signal handler intercepts
   ├── Identify faulting address
   ├── Find corresponding MasterRegion
   ├── Call mprotect() to make page writable
   ├── Record access in dirty bitmap
   └── Return to application
   ↓
4. Application continues (transparent)
   ↓
5. Migration thread (every 500ms)
   ├── Scan dirty pages
   ├── Batch migrate to NUMA node 0
   └── Clear dirty bitmap
```

---

## Data Structures

### Kernel Cache Entry
```cpp
struct CachedKernel {
    KernelId id;
    std::string source_code;
    std::string compilation_flags;
    CompiledKernelFn function_ptr;
    size_t estimated_gflops;
    std::chrono::system_clock::time_point compiled_at;
};
```

### Stream
```cpp
struct Stream {
    StreamId id;
    std::queue<Task> task_queue;
    std::mutex queue_lock;
    std::condition_variable cv;
    int priority;  // -2 to 2
    bool is_blocking;
};
```

### Task
```cpp
struct Task {
    TaskType type;  // KERNEL_LAUNCH, MEMCPY, EVENT_RECORD, etc.
    std::vector<EventId> dependencies;
    std::function<void()> execute;
    std::chrono::system_clock::time_point submitted_at;
};
```

### MasterRegion (UVM)
```cpp
struct MasterRegion {
    void* host_ptr;
    size_t size;
    std::vector<bool> dirty_bitmap;  // Per-page dirty tracking
    std::atomic<bool> migrated;
    int numa_node;
};
```

---

## Performance Optimizations

### 1. Kernel Fusion
**Idea**: Merge consecutive compatible kernels into a single JIT compilation.

**Conditions**:
- Same stream
- No synchronization between kernels
- Compatible memory access patterns

**Benefit**: Reduces compilation overhead by ~30% for consecutive kernels.

### 2. NUMA-Aware Allocation
**Idea**: Bind large allocations (≥2MB) to NUMA node 0.

**Implementation**:
```cpp
if (size >= 2 * 1024 * 1024) {
    mbind(ptr, size, MPOL_PREFERRED, &node_mask, 1, 0);
}
```

**Benefit**: 2.5–3× speedup for memory-bound kernels on multi-socket systems.

### 3. Bandwidth Calibration Caching
**Idea**: Cache bandwidth measurements process-wide to avoid repeated benchmarks.

**Benefit**: 5–10× faster test suite initialization.

### 4. SharedMemory Pooling
**Idea**: Pre-allocate shared memory buffers outside the block loop.

**Benefit**: Eliminates per-block malloc/free overhead for `__syncthreads` kernels.

### 5. OpenMP Schedule Guided
**Idea**: Use `schedule(guided)` instead of `schedule(dynamic)` for work distribution.

**Benefit**: Reduces atomic overhead on work-distribution queue.

### 6. AES-NI Hardware Acceleration
**Idea**: Use `_mm_aesenc_si128` intrinsics for 4-block parallel AES-256-CTR.

**Benefit**: 8–12× faster encryption than software fallback.

---

## Cross-Platform Implementation

### Linux
- **UVM**: SIGSEGV signal handler
- **NUMA**: `mbind()` for memory placement
- **Token Storage**: Linux Keyring (primary), libsecret (secondary), TPM 2.0 (tertiary)
- **Networking**: Standard POSIX sockets
- **Temperature**: `/sys/class/thermal/` sysfs interface

### Windows
- **UVM**: Vectored Exception Handler (VEH) for `EXCEPTION_ACCESS_VIOLATION`
- **NUMA**: N/A (not exposed to user-mode)
- **Token Storage**: Windows Credential Manager (primary), TPM 2.0 (secondary)
- **Networking**: WinSock2 API
- **Temperature**: `CallNtPowerInformation()` (heuristic)

### macOS
- **UVM**: SIGSEGV signal handler
- **NUMA**: N/A (not exposed)
- **Token Storage**: Keychain (primary), TPM 2.0 (secondary)
- **Networking**: POSIX sockets with `SO_NOSIGPIPE` for broken pipe handling
- **Temperature**: IOKit framework (heuristic)

---

## Security Architecture

### Authentication
- **Handshake**: HMAC-SHA256(token, challenge) → session key
- **Session Key**: PBKDF2(token, salt, 600,000 iterations) → 256-bit key
- **Verification**: Constant-time comparison via `crypto::secure_compare()`

### Encryption
- **Algorithm**: AES-256-CTR (NIST-approved)
- **Hardware Acceleration**: AES-NI intrinsics when available
- **Key Rotation**: Every 10,000 packets
- **Replay Protection**: 256-bit sequence bitmap

### Token Storage
- **Linux**: Kernel keyring (most secure), GNOME Keyring, TPM 2.0, encrypted file fallback
- **Windows**: Credential Manager, TPM 2.0, encrypted file fallback
- **macOS**: Keychain, TPM 2.0, encrypted file fallback

---

## Testing Architecture

**Test Framework**: CTest (CMake native)

**Test Categories**:
- **Unit Tests** (20+): Individual component testing
- **Integration Tests** (30+): Multi-component workflows
- **Advanced Tests** (15+): Cluster, security, performance

**Test Execution**:
```bash
cd build
ctest --output-on-failure
```

**Pre-Test Cleanup**:
- Removes stale OpenMP KMP registration files from `/dev/shm`
- Prevents deadlock during library registration

---

## Configuration & Tuning

### Environment Variables

**Logging**:
```bash
VGRE_LOG_LEVEL=DEBUG|INFO|WARN|ERROR
```

**Performance**:
```bash
VGRE_ENABLE_NUMA=1                    # Enable NUMA awareness (Linux)
VGRE_WORKER_THREADS=16                # Override thread count
VGRE_SIMD_LEVEL=AVX2|AVX512|native    # Force SIMD level
VGRE_UVM_MIGRATION_MS=500             # UVM migration interval
```

**Cluster**:
```bash
VGRE_TCP_AUTH_TOKEN_FILE=~/.vgre/token
VGRE_CLUSTER_NODES=192.168.1.50:7777,10.0.0.100:7777
VGRE_CLUSTER_STRICT_AUTH=1            # Reject mismatched tokens
VGRE_MESH_PEERS=ip:port,...           # Mesh topology
```

---

## Future Enhancements

- [ ] INT8 quantization-aware training
- [ ] Flash Attention integration
- [ ] Fused transformer kernels
- [ ] OpenTelemetry/Prometheus metrics export
- [ ] Kubernetes operator for cluster orchestration
- [ ] WebSocket transport for WAN clusters
- [ ] Zero-copy shared memory for local clusters

---

**Version**: 1.0.0  
**Last Updated**: 2026-05-06
