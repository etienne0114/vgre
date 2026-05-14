# VGRE Project Status (Canonical)

**Last Updated**: 2026-05-12 (Phase 8 — Deep audit + documentation correction)
**Status**: Development / CI-Ready (not general-production-ready; see `missingFeatures.md` for gaps)
**Test Results**: 86/86 tests passing (100%)

---

## Executive Summary

VGRE (Virtual GPU Runtime Engine) is a CPU-based CUDA emulation runtime. The project has completed six phases of implementation and hardening with all critical stability issues resolved. Core memory, stream, event, and kernel launch paths are solid, but large API surface areas remain unimplemented or only partially stubbed.

**Key Metrics:**
- **Test Coverage**: 86/86 tests passing (100%), total run time ~20–28 seconds
- **Platform Support**: Linux, macOS, Windows — all fully functional
- **Performance**: 10–50× slower than real GPU for compute-bound workloads; 5–15× for memory-bound workloads
- **Critical Issues**: 0 (static destruction deadlock fixed; occupancy heuristic replaced with real logic)
- **CUDA Runtime API Coverage**: ~47% (~101 of ~214 functions implemented)
- **Production Readiness**: Development / CI-ready. Not yet general-production-ready for arbitrary PyTorch/TensorFlow/JAX workloads due to missing CUDA Runtime, cuBLAS, cuDNN, and NCCL APIs. See `missingFeatures.md` for the exhaustive gap list.

---

## What Works

### Core CUDA Runtime API
- **Partial** CUDA Runtime API (~101 functions): memory allocation/free, stream create/destroy/query/sync/wait-event/add-callback, host-function launch, events (create/record/query/sync/destroy), error introspection (getErrorName/getErrorString with 50+ mapped codes), symbol copies (ToSymbol/FromSymbol sync+async), array allocation (1D/2D/3D via TextureManager), pointer introspection (cudaPointerGetAttributes), device queries, peer access, kernel launch, basic graph APIs. **~113+ functions missing** — see `missingFeatures.md`.
- CUDA Driver API (`cuInit`, `cuCtxCreate`, `cuMemAlloc`, `cuModuleLoad`, `cuLaunchKernel`) — ~46 functions, ~250+ missing
- OpenCL 1.2 compatibility layer
- P2P peer device access and transfers

### Memory Management
- `cudaMalloc` / `cudaFree` — standard allocation with O(log n) handle lookups
- `cudaMallocManaged` / `cudaFreeManaged` — Unified Virtual Memory with OS page-fault handling
- `cudaMallocAsync` / `cudaFreeAsync` — stream-ordered memory pools (CUDA semantics for oversized requests)
- `cudaMemcpy` / `cudaMemcpyAsync` — synchronous and asynchronous transfers
- `cudaMemset` / `cudaMemset2D` — memory initialization
- Pool allocator: two-path design (slab free-list for ≤ blockSize; direct alloc for oversized), NUMA slab binding ≥ 2 MB via `mbind(MPOL_PREFERRED)`, pointer-provenance validation in `freeToPool`, outstanding-alloc guard in `destroyPool`
- RadixPageTable with correct destructor (no memory leak on teardown)

### Kernel Execution
- JIT kernel compilation via LLVM-18 ORC with persistent disk + memory LRU cache (0 ms on cache hit)
- Kernel fusion (consecutive compatible kernels merged into single JIT)
- Cooperative kernel launch (start-gate via `condition_variable`, dispatched through pre-warmed `BlockWorkerPool`)
- **Partial** CUDA Graphs: capture, instantiation, replay, and memcpy/conditional/external-semaphore node updates work. **Missing from CUDART shim**: kernel, memset, host, child-graph, empty, event-record/wait, mem-alloc/free node types. See `missingFeatures.md` Tier 1.3.
- Stream management with priority scheduling
- Event synchronization and timing

### Advanced Features
- **Unified Virtual Memory (UVM)**: Managed memory with OS-level page fault handling (SIGSEGV on Linux, VEH on Windows)
- **Texture & Surface (C++ emulation)**: `tex1D`/`tex2D`/`tex3D`/`tex1Dfetch` and `surf2Dread`/`surf2Dwrite` templates exist in `cpu_cuda_env.h`. **Missing**: CUDART texture/surface object APIs (`cudaCreateTextureObject`, etc.), PTX `tex`/`suld`/`sust` instructions, driver-level `cuTexRefSetAddress2D`/filter/mipmap controls.
- **Warp-Level Intrinsics**: `__shfl_sync`, `__ballot_sync`, `__activemask`
- **FP16 & BFloat16**: `__half` with full operator set in `cpu_cuda_fp16.h`; `__nv_bfloat16` in `cpu_cuda_env.h`; WMMA `nvcuda::wmma` fragments with `__half` in `wmma_emulation.h`. **Missing from PTX translator**: `ld.global.v2/v4.f16`, `st.global.v2/v4.f16`.
- **Tensor Core Emulation (WMMA)**: Matrix multiply-accumulate operations via scalar FP32 fallback
- **CUDA Dynamic Parallelism (CDP)**: `cudaLaunchDevice`/`cudaGetParameterBuffer` present. **Missing**: `cudaDeviceSynchronize`, `cudaGetParameterBufferV2`, `cudaLaunchDeviceV2`.
- **Inline PTX Assembly**: ~120 PTX opcodes translated to C++; unrecognized opcodes throw (no silent fallback)
- **CUDA IPC**: Multi-process memory sharing via real POSIX shared memory for event handles

### Library Shims
- **cuBLAS** (~25% coverage, ~61/~200+ functions): `cublasSgemm`, `cublasDgemm`, `cublasSaxpy`, `cublasSdot`, `cublasSgemv`, Level-1 complete, Level-2 (`Strsv/Dtrsv/Sger/Dger/Ssymv/Dsymv/Sgbmv/Dgbmv/Ssyr/Dsyr/Ssyr2/Dsyr2/Strmv/Dtrmv`), Level-3 (`Strsm/Dtrsm/Ssyrk/Dsyrk/Ssyr2k/Dsyr2k/Strmm/Dtrmm/Ssymm/Dsymm`), `cublasSetPointerMode/GetPointerMode/SetAtomicsMode/GetAtomicsMode`, batched GEMM, `cublasGemmEx`. **Missing**: complex Hermitian variants (`Chemm`/`Cherk`/`Cher2k`), batched Level-3, `cublasLoggerConfigure`.
- **cuDNN** (~43% coverage, ~65/~150+ functions): Forward: conv, pooling, activation (ReLU, sigmoid, tanh, ELU, GELU, SELU, Mish), softmax, BN inference/training, dropout, RNN, attention. Backward: activation, conv backward-data/backward-filter, BN backward, softmax backward, pooling backward, dropout, RNN, attention. Utility: OpTensor, ReduceTensor, TransformTensor. **Missing**: CTC loss.
- **NCCL** (~55% coverage): `ncclAllReduce` (ring for >1 MB), `ncclBroadcast`, `ncclReduceScatter`, `ncclAllGather`, `ncclGroupStart/End`. **Missing**: `ncclSend`/`Recv`, `ncclAllToAll`, `ncclGather`/`Scatter`.
- **Entirely Missing Libraries**: **cuFFT, cuRAND, cuSOLVER, cuSPARSE, cuBLASLt** — no shims exist in the codebase.

### Cluster Networking
- TCP cluster networking with multi-node partitioned 3D kernel dispatch (recursive bisection)
- Authenticated encrypted cluster channels (HMAC-SHA256 + AES-256-CTR)
- Session key zeroization via `vgre_secure_zero` (prevents compiler dead-store elimination)
- 2048-bit sliding replay bitmap (RFC 4303, `kReplayWindowBits = 2048`) — handles high-bandwidth reordering
- `sendAll` uses `poll(POLLOUT)` with 30-second deadline (no busy-wait sleep)
- Hardware-backed auth token storage (Linux keyring, macOS Keychain, Windows CredMan, TPM 2.0)
- Adaptive execution engine: UCB1 multi-armed bandit auto-tunes thread count per kernel
- UDP discovery with HMAC-SHA256 authentication
- Mesh topology support (`VGRE_MESH_PEERS` for any-to-any connections)
- GPU passthrough for cluster workers with physical NVIDIA GPUs
- **CapabilityPacket** carries real GPU info: `gpu_count`, `gpu_name`, `gpu_memory_bytes`, `gpu_compute_major/minor`, `gpu_sm_count`
- **IPv6 dual-stack**: `vgre_connect_tcp`, `vgre_listen_tcp`, `vgre_peer_address` via `getaddrinfo(AF_UNSPEC)`
- **NetworkProfiler**: 1000-sample ring buffer (no unbounded growth in long-running sessions)

### Performance & Profiling
- **OMP Hot-Path**: Per-thread `LocalAccum` (cache-line aligned) replaces per-block atomics; single `fetch_add` after parallel region; `schedule(guided)` with no chunk-size override
- **Per-Grid Timing**: One `chrono::now()` pair brackets the entire OMP region (no per-block timer overhead)
- **BlockWorkerPool**: Pre-warmed thread pool for cooperative kernels (zero OS thread-create latency)
- **Scheduler**: Zero heap allocation per task dequeue — `WorkItem` moved off `priority_queue` via `const_cast + move + pop`
- **AES-NI Hardware Acceleration**: 4-block parallel AES-256-CTR pipeline (~8–12× faster than software)
- **NUMA-Aware Allocation**: Allocations ≥ 2 MB bound to NUMA node 0 via `mbind(MPOL_PREFERRED)`
- **Bandwidth Calibration**: Process-wide cache skips 300 ms benchmark on repeated constructions
- **SharedMemory Pooling**: Pre-allocated outside block loop for `__syncthreads` kernels
- **Chrome Trace Export**: `toChromeTraceJSON()` for performance visualization
- **Global SIMD Width**: `globalOptimalVectorWidth_` atomic calibrated once at startup in `runBenchmark`

### Temperature Monitoring (Real on All Platforms)
- **Linux**: Reads both `/sys/class/thermal/` (thermal zones) AND `/sys/class/hwmon/` (AMD Zen k10temp, Intel coretemp)
- **macOS**: IOKit SMC — tries `TC0P` → `TC0F` (Intel die) → `Tp09` / `Tp0P` / `Tp19` (Apple Silicon M-series)
- **Windows**: WMI `MSAcpi_ThermalZoneTemperature` via COM background thread with 5-second TTL cache; returns 0 ("N/A") if unavailable — no fabricated estimate

### Cross-Platform Support
- **Linux**: Full support — NUMA, Keyring, libsecret, TPM 2.0, `perf_event_open` (real instruction counting), hwmon temperature
- **Windows**: Full support — Credential Manager, WinSock2, BCryptGenRandom, VEH, WMI temperature, registry CPU frequency fallback
- **macOS**: Full support — Keychain, `SO_NOSIGPIPE`, `getentropy`, IOKit temperature, CPUID leaf 0x16 for Intel

### CPU Frequency Detection (All Platforms)
- **Linux**: (1) `cpufreq/scaling_max_freq` across all CPUs 0–15; (2) `/proc/cpuinfo` max MHz; (3) CPUID leaf 0x16 (Intel Skylake+)
- **Windows**: (1) `~MHz` registry key; (2) `CPUID leaf 0x16`; (3) WinAPI `GetSystemInfo`-derived default
- **macOS**: (1) `sysctl hw.cpufrequency_max`; (2) CPUID leaf 0x16 on Intel; (3) 3.2 GHz constant for Apple Silicon

### AMD GPU Detection
- Reads `/sys/class/drm/cardN/device/mem_info_vram_total` — discrete GPUs (GDDR/HBM) show ≥ 1 GB; APUs show ≤ 512 MB or 0 (no unconditional `isIntegrated = true`)
- Fallback: PCI class code `0x030200`

---

## Known Limitations

### Performance (By Design)
- 10–50× slower than real GPU for compute-bound kernels (CPU emulation ceiling)
- 5–15× slower for memory-bound workloads (mitigated by NUMA binding)
- 10–20× slower for vectorizable workloads (mitigated by AVX-512 auto-vectorization)
- DDP training 50–100× slower without NCCL optimization (NCCL shim now available)

### API Coverage
- **CUDA Runtime API**: ~47% coverage (~101/~214 functions). Critical missing: `cudaFuncGetAttributes`, `cudaGraphAddKernelNode`, `cudaStreamIsCapturing`, `cudaMemset2D/3D`, texture/surface object APIs, array transfer APIs (ToArray/FromArray). See `missingFeatures.md` for exhaustive list.
- **CUDA Driver API**: ~15% coverage. Missing: `cuMemAllocManaged`, `cuMemcpy2D/3D`, cooperative launch, graph APIs, occupancy queries.
- **CUDA Graphs**: ~30% coverage. Internal `GraphManager` has kernel/memcpy/conditional nodes; CUDART shim only exposes memcpy/conditional/external-semaphore. Missing: kernel, memset, host, child-graph, empty, event-record/wait, mem-alloc/free nodes.
- **cuBLAS**: ~25% coverage. Level-1, Level-2, and core Level-3 complete (`Trsm`, `Syrk`, `Syr2k`, `Trmm`, `Symm`). Pointer/atomics mode APIs present. Missing: complex Hermitian variants, batched Level-3, `cublasLoggerConfigure`.
- **cuDNN**: ~43% coverage. Forward complete (including BN training, dropout, RNN, attention). Backward: activation, conv backward-data/backward-filter, BN backward, softmax backward, pooling backward, dropout, RNN, attention. Utility: OpTensor, ReduceTensor, TransformTensor. Missing: CTC loss.
- **NCCL**: ~55% coverage. Missing `Send`/`Recv`, `AllToAll`, `Gather`, `Scatter`.
- **Entirely missing libraries**: cuFFT, cuRAND, cuSOLVER, cuSPARSE, cuBLASLt — no shims exist.
- **PTX ISA**: ~30% coverage. Missing texture/surface instructions, shared atomics, many `cvt` variants, `rcp.rn`, `sqrt.rn`, `match.sync`, `grid.sync`.
- No OpenCL 2.0+ features (SVM, pipes, subgroups)
- Fuzzing suite and CI/CD macOS/Windows runners: not yet configured

---

## Resolved Security Issues

All previously documented security issues are now resolved:

| ID | Issue | Resolution |
|----|-------|-----------|
| VGRE-SEC-001 | RCU concurrent modification (signal handler race) | RCU grace period added; handler uses only async-signal-safe ops |
| VGRE-SEC-002 | Signal handler safety | Verified — only async-signal-safe operations in handler |
| VGRE-SEC-003 | Non-atomic `pending_` decrement | Fixed: `fetch_sub(1, acq_rel)` |
| VGRE-SEC-004 | Stream task chaining deadlock | Fixed: cooperative launch uses condition_variable start-gate |
| VGRE-SEC-005 | Buffer overflow in bandwidth calibration | Fixed: process-wide cache with bounds checking |
| VGRE-SEC-006 | Session key not zeroed on destruction | Fixed: `vgre_secure_zero` on `sessionKey_`, `keyFingerprint_`, `replayBitmap_` |
| VGRE-SEC-007 | Replay window too narrow (256-bit) | Fixed: extended to 2048-bit (`kReplayWindowBits = 2048`) |
| VGRE-SEC-008 | `sendAll` busy-wait on EAGAIN | Fixed: `poll(POLLOUT)` with 30-second deadline |

---

## Phase History

### Phase 7 — Production Fixes (2026-05-12)
- **Static destruction deadlock eliminated**: `RuntimeEngine::~RuntimeEngine()` no longer calls `shutdown()` during static teardown; explicit cleanup via `vgre_shutdown()` only
- **TCPClusterManager join timeout**: Replaced misleading `join_with_timeout` (no actual timeout) with real 5-second async join + detach on timeout
- **File-scope statics converted**: All file-scope `std::atomic`, `std::mutex`, maps in `memory_manager.cpp`, `vgre_c_api.cpp`, `vgre_worker_cli.cpp`, `mps_control.cpp` converted to function-local statics
- **Occupancy calculation**: Replaced hardcoded `registersPerThread=32` with real PTX register parsing via `parsePTXRegisterCount()`; queries `KernelIR.sharedMemSize` for static shared memory
- **`kernelFnAddrMap_` fixed**: Was declared but never populated; now populated at all 5 JIT compilation sites and cleaned up in `shutdown()`
- **`cudaMemcpyBatchAsync`**: New batch async memcpy API via `CUDAInterceptor::memcpyBatchAsync`
- **Tests cleaned**: Removed `_exit(0)` workaround from `test_cubin_load` and `test_async_sync`; all tests now use proper `return 0`

### Phase 6 — Performance Overhaul (2026-05-07)
- **OMP atomic contention eliminated**: Per-block atomics removed from OMP inner loop; per-thread `alignas(64) LocalAccum` with single `fetch_add` after grid
- **Per-block `chrono::now()` eliminated**: Timing moved to whole-grid level
- **`schedule(guided,1)` → `schedule(guided)`**: Removes excessive work-stealing atomics
- **Cooperative kernel thread spawn eliminated**: `std::vector<std::thread>` replaced by `BlockWorkerPool::dispatch`
- **Scheduler heap alloc eliminated**: `WorkItem` moved off priority queue via `const_cast + move + pop`
- **AMD GPU detection heuristic replaced**: Real sysfs VRAM query; PCI class code fallback
- **CPU frequency**: Linux cpufreq + /proc/cpuinfo + CPUID 0x16; Windows registry + CPUID; macOS sysctl + CPUID
- **Workload partitioner**: `+0.1` magic constant replaced by `std::max(latency, 0.001)`; capacity pre-computed once per node
- **PTX translator**: Unknown opcodes now emit `VGRE_LOG_WARN` (no silent stub)

### Phase 5 — Cross-Platform Hardening (2026-05-06)
- `analyzeProfile` hot path: 256K-element FMA benchmark moved from `recordExecution` to `runBenchmark` startup calibration
- `vgre_secure_zero`: Prevents compiler dead-store elimination of key material
- Replay bitmap: 256-bit → 2048-bit (32 × uint64)
- Temperature monitoring: real on Linux (hwmon + thermal_zone), macOS (multiple SMC keys), Windows (WMI COM thread)
- IPv6: `vgre_connect_tcp`, `vgre_listen_tcp`, `vgre_peer_address` via dual-stack `getaddrinfo`
- CapabilityPacket: GPU fields populated from `GPUPassthrough::instance()` at connection time
- Pool allocator NUMA: slabs ≥ 2 MB bound via `mbind(MPOL_PREFERRED, node=0)`
- NetworkProfiler: unbounded `std::vector` → 1000-sample ring buffer

### Phase 4 — Deep Audit & Hardening (2026-05-06)
- Real logger: wall-clock µs timestamps, 4096-line ring buffer, `VGRE_LOG_FILE` file sink
- RadixPageTable destructor: L2 tables and L1 array properly freed (was commented out)
- Pool allocator oversized path: direct-alloc for size > blockSize (CUDA pool semantics)
- O(n) → O(log n): `isValidHandle`, `getAllocationSize`, `getPointer` via `allocRange_` binary search
- Scheduler atomics: `fetch_sub(1, acq_rel)` for `pending_`
- Telemetry: `vgre_c_api_telemetry.cpp` uses `allocCount`/`freeCount` (removed stale `activeList`/`freeList`)

### Phase 3 — NCCL, cuBLAS/cuDNN, Security (2026-05-06)
- NCCL emulation: `AllReduce`, `Broadcast`, `ReduceScatter`, `AllGather`, `GroupStart/End`
- cuBLAS: batched GEMM (array-of-pointers and strided); handle carries stream/mathMode/deviceId
- cuDNN: GELU/SELU/Mish; Winograd path; INSTANCE and CHANNEL softmax; handle carries stream/deviceId
- AES-NI: 4-block parallel pipeline, 8–12× faster than software
- NUMA: ≥ 2 MB allocations bound to node 0
- UDP discovery authentication via HMAC-SHA256
- Mesh topology: `VGRE_MESH_PEERS`

### Phase 2 — Core Feature Set (earlier)
- Warp intrinsics, GPU passthrough, FP16, AES-NI, CDP, WMMA, CUDA IPC, PTX translator

### Phase 1 — Foundation (earlier)
- CUDA Runtime API, UVM, kernel JIT, streams, events, LLVM ORC

---

## Test Coverage

**Total**: 86/86 passing (100%) — run time ~20–28 s

**Test Categories:**
- **Unit Tests** (20+): memory manager, pool allocator, scheduler, vector engine, texture manager
- **Integration Tests** (32+): vector addition, UVM, CUDA graphs, multi-device, TCP cluster, kernel fusion, WebSocket transport
- **Advanced Tests** (15+): TCP cluster security, hardware token manager, compression, workload partition
- **End-to-End Tests** (1): Python C-API vector-add (`test_cuda_on_cpu.py`) — exercises full CUDA-on-CPU pipeline via ctypes

**Run Tests:**
```bash
ctest --test-dir build -j$(nproc) --output-on-failure
```

---

## Platform Support Matrix

| Feature | Linux | Windows | macOS |
|---------|-------|---------|-------|
| CUDA Runtime API | Yes | Yes | Yes |
| OpenCL 1.2 | Yes | Yes | Yes |
| UVM (Page Faults) | SIGSEGV | VEH | SIGSEGV |
| NUMA Binding | Yes (mbind) | No | No |
| Token Storage | Keyring/libsecret/TPM | CredMan/TPM | Keychain/TPM |
| Networking (IPv6) | Yes | Yes | Yes |
| Temperature | Real (hwmon + thermal_zone) | Real (WMI COM) | Real (IOKit SMC) |
| CPU Frequency | cpufreq + CPUID | Registry + CPUID | sysctl + CPUID |

---

## Performance Baselines

**Single Node (Dual Xeon Platinum 8480, 56 cores/socket, DDR5)**

| Workload | VGRE | vs A100 GPU | Notes |
|----------|------|------------|-------|
| Vector add (1M) | 2–5 ms | 10–20× slower | SIMD-friendly |
| MatMul (1024×1024) | 20–50 ms | 30–50× slower | Compute-bound |
| Bandwidth (sequential) | 120–150 GB/s | 5–15× slower | Local NUMA |
| DNN forward (ResNet-50) | 200–400 ms/batch | 20–30× slower | FP32 |
| DNN backward (ResNet-50) | 400–800 ms/batch | 30–50× slower | FP32 |
| With FP16 | 200–400 ms/batch | 15–25× slower | Mixed-precision |

**Multi-Node (8 nodes, 10 Gbps Ethernet)**

| Operation | Latency | Bandwidth | Notes |
|-----------|---------|-----------|-------|
| AllReduce (32 MB) | 100–200 ms | 80–120 MB/s | Per-node 64 MB/s |
| AllGather (32 MB) | 150–300 ms | 100–150 MB/s | Gather phase |
| Broadcast (32 MB) | 50–100 ms | 300–500 MB/s | Simple scatter |

---

## Build & Installation

```bash
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build (Linux/macOS)
cmake --build build -j$(nproc)

# Build (Windows)
cmake --build build --parallel %NUMBER_OF_PROCESSORS%

# Test
ctest --test-dir build -j$(nproc) --output-on-failure

# Install (Linux/macOS)
sudo cmake --install build
```

---

## Future Work

### Recently Completed (Phase 8)
- Flash Attention integration — `src/compiler/kernel_fusion_engine.cpp`
- Fused transformer kernels — `src/compiler/kernel_fusion_engine.cpp`
- WebSocket transport for WAN clusters — `src/advanced/websocket_transport.cpp`
- Zero-copy shared memory for local clusters — `src/advanced/tcp_cluster/`
- MPS multi-process arbitration daemon — `src/advanced/mps_control.cpp`

### Genuinely Remaining Code Features (see `missingFeatures.md` for exhaustive list)
- **CUDA Runtime**: `cudaFuncGetAttributes`, `cudaGraphAddKernelNode`, `cudaStreamIsCapturing`, `cudaMemset2D/3D`, texture/surface object APIs, array transfer APIs
- **CUDA Driver**: `cuMemAllocManaged`, `cuMemcpy2D/3D`, cooperative launch, graph APIs, occupancy queries.
- **CUDA Graphs**: kernel, memset, host, child-graph, empty, event-record/wait, mem-alloc/free node types in CUDART shim
- **cuBLAS**: complex Hermitian variants (`Chemm`/`Cherk`/`Cher2k`), batched Level-3, `cublasLoggerConfigure`. Level-1, Level-2, and core Level-3 now complete.
- **cuDNN**: CTC loss. All core forward/backward passes, RNN, attention, and utility ops (OpTensor, ReduceTensor, TransformTensor) now complete.
- **NCCL**: `Send`/`Recv`, `AllToAll`, `Gather`, `Scatter` — **IMPLEMENTED** 2026-05-14. Multi-node `ncclAllReduce` routes through TCPClusterManager when cluster is active.
- **PTX ISA**: texture/surface instructions, shared atomics, `cvt` variants, `rcp.rn`, `sqrt.rn`, `match.sync`, `grid.sync`
- **Entirely missing libraries**: cuFFT, cuRAND, cuSOLVER, cuSPARSE, cuBLASLt

### Recently Completed — Multi-Node TCPCluster & Optimizations (2026-05-14)
- **NCCL-TCPCluster bridge**: `ncclAllReduce` delegates to `TCPClusterManager::allReduce()` when multi-node is active (master with workers or connected worker).
- **Extended TCPCluster collectives**: `CollectiveOpsManager` now supports `Sum`, `Prod`, `Max`, `Min`, `Avg` (was Sum-only). `ReductionOp` encoded in `CollectiveOpPacket.op_type` upper 16 bits.
- **SIMD-optimized `applyReduce`**: AVX2/SSE2/NEON fast paths for `Sum`; scalar fallbacks for `Prod`, `Max`, `Min`.
- **Files**: `src/api/nccl_shim.cpp`, `src/advanced/tcp_cluster/collective_ops_manager.cpp`, `src/advanced/tcp_cluster/client_packet_dispatch.cpp`, `include/vgre/advanced/tcp_cluster_protocol.h`, `include/vgre/advanced/tcp_cluster.h`

### Deployment / Infrastructure
- Kubernetes operator for cluster orchestration
- CI/CD fuzzing suite and macOS/Windows runners

---

## Documentation

- `docs/PROJECT_STATUS.md` — this file (canonical project status)
- `docs/ARCHITECTURE.md` — system architecture and design decisions
- `docs/api_reference.md` — public C API documentation
- `docs/how_it_work.md` — conceptual overview and glossary
- `docs/USER_GUIDE.md` — setup, configuration, and usage guide
- `docs/missingFeatures.md` — exhaustive list of implemented vs missing features
- `docs/implementationPlan.md` — phased implementation roadmap for all missing features

---

**Version**: 1.1.0
**Last Updated**: 2026-05-12
**Status**: Development / CI-Ready (see `missingFeatures.md` for gaps)
