# VGRE — Virtual GPU Runtime Engine

**A CUDA emulation runtime** that allows CUDA applications to run on CPU without a physical GPU.

> **PROJECT STATUS**: Development / CI-Ready — ✅ 108–110/110 tests passing, all critical stability issues fixed. Major API surface areas are implemented; remaining gaps are primarily complex BLAS, optimized FFT, and sparse solvers. See `docs/missingFeatures.md` for the exhaustive gap list.

## What is VGRE?

VGRE intercepts CUDA and OpenCL API calls and executes kernels on CPU using:
- LLVM-18 ORC JIT (Clang AST parsing → LLVM IR → native code, `-O3 -march=native -fno-math-errno`)
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
- ⚠️  Complex, performance-critical GPU workloads (typically 10–50× slower than real GPU; AVX-512 + OpenMP narrow the gap for memory-bound kernels)
- ℹ️  Bandwidth-bound workloads: `MemoryBandwidthStats` (via `getMemoryBandwidthStats()`) measures your effective bandwidth and the estimated GPU speedup factor so you can quantify the gap for your specific workload

## Current Status

**Core Stability**: ✅ 108–110/110 tests passing, zero critical issues  
**CUDA Runtime API Coverage**: ~95% (~101+ of ~110 commonly-used functions)  
**CUDA Driver API Coverage**: ~95% (~56+ of ~60 commonly-used functions)  
**cuBLAS Coverage**: ~85% (~61+ of ~72 real functions; complex C/Z missing)  
**cuDNN Coverage**: ~90% (~65+ of ~72 major functions)  
**NCCL Coverage**: ~95% (all major collectives + p2p)  
**PTX ISA Coverage**: ~95% (~110+ of ~115 commonly-used instructions)  
**Critical Issues**: 0  
**Cross-Platform**: Linux, Windows, macOS all functional

### Platform Support ✅
- ✅ **Linux**: Full support (all core features + NUMA + Linux Keyring)
- ✅ **Windows**: Full support (all core features + Credential Manager + shared memory)
- ✅ **macOS**: Full support (all core features + Keychain)

See [Cross-Platform Status](docs/CROSS_PLATFORM_STATUS.md) for detailed platform analysis.

### What Works ✅
- **CUDA Runtime API** (~101+ functions): memory alloc/free (`cudaMalloc`, `cudaFree`, `cudaMallocManaged`, `cudaMallocAsync`, `cudaMallocFromPoolAsync`, `cudaMallocPitch`, `cudaMallocArray`, `cudaMalloc3DArray`, `cudaMalloc3D`), stream create/destroy/query/sync/wait-event/add-callback/launch-host-func, events (create/record/query/sync/destroy/elapsed-time), error introspection (`cudaGetErrorName`/`GetErrorString`), symbol copies (`cudaMemcpyToSymbol`/`FromSymbol` sync+async), array allocation (1D/2D/3D via `TextureManager`), pointer introspection (`cudaPointerGetAttributes`), device queries, peer access, kernel launch (`cudaLaunchKernel`, `cudaLaunchCooperativeKernel`, `cudaLaunchKernelExC`), graph APIs (capture, instantiate, launch, clone, destroy, exec update, all 11 node types including kernel, memset, host, child, empty, event-record/wait, mem-alloc/free), texture/surface objects (`cudaCreateTextureObject`, `cudaDestroyTextureObject`, `cudaCreateSurfaceObject`, `cudaDestroySurfaceObject`, legacy `cudaBindTexture`/`cudaBindTextureToArray`/`cudaBindTexture2D`/`cudaBindSurfaceToArray`), external memory/semaphore (`cudaImportExternalMemory`, `cudaDestroyExternalMemory`), stream capture introspection (`cudaStreamIsCapturing`, `cudaStreamGetCaptureInfo_v2`), device/function attributes (`cudaFuncGetAttributes`, `cudaDeviceGetLimit`/`SetLimit`, `cudaDeviceGetCacheConfig`/`SetCacheConfig`), memset 2D/3D/Async variants.
- CUDA Driver API (~56+ functions): context, device, memory, module, stream, texture objects, external memory/semaphore, cooperative launch, occupancy, graphs, IPC.
- OpenCL 1.2 compatibility layer
- JIT kernel compilation with persistent disk + memory cache (0ms on cache hit)
- Texture / Surface C++ emulation (`tex1D`/`tex2D`/`tex3D` templates in `cpu_cuda_env.h`) + CUDART object APIs + driver texref APIs + PTX texture/surface instructions
- UVM managed memory (`cudaMallocManaged`) with OS-level page fault handling
- CUDA Graphs: capture, instantiation, replay, exec mutation, introspection, dependencies, user objects, all 11 node types, `cudaGraphExecUpdate_v2`
- Stream-ordered memory pools (`cudaMallocAsync` / `cudaFreeAsync`)
- Graph node updates (all types)
- Windows shared memory (`CreateFileMapping`/`MapViewOfFile`)
- Cooperative kernel launch
- CDP (CUDA Dynamic Parallelism): `cudaLaunchDevice`, `cudaGetParameterBuffer`, `cudaDeviceSynchronize`, `V2` variants
- P2P peer device access and transfers
- Kernel fusion (consecutive compatible kernels fused into single JIT compilation)
- TCP cluster networking: multi-node partitioned kernel dispatch, telemetry aggregation
- Authenticated encrypted cluster channels (HMAC-SHA256 + AES-256-CTR)
- Hardware-backed auth token storage (keyring/Keychain/CredMan/TPM 2.0)
- Adaptive execution engine: auto-tunes thread count for each kernel
- Chrome trace export (`toChromeTraceJSON`) and C API telemetry
- Python bindings (`vgre_c_api` via ctypes), NumPy-compatible
- Cooperative groups (`thread_block`, `thread_block_tile`, `grid_group`, `multi_grid_group`) + CUB fallback headers
- cuBLAS: Level-1/2/3 real (S/D) routines, Hermitian (Cherk/Zherk/Cher2k/Zher2k/Chemm/Zhemm), batched GEMM, `GemmEx`, logger
- cuDNN: conv, pool, activation, softmax, BN, dropout, RNN, attention, CTC loss, LRN, divisive norm, tensor ops (OpTensor, ReduceTensor, TransformTensor, AddTensor), Backend API wired to legacy
- NCCL: AllReduce, Broadcast, Reduce, AllGather, ReduceScatter, Send, Recv, AllToAll, Gather, Scatter
- cuFFT: reference DFT/IDFT for 1D/2D/3D (C2C, R2C, C2R, Z2Z, D2Z, Z2D)
- cuRAND: XORWOW, MRG32k3a, MTGP32, MT19937 generators
- cuSOLVER: Cholesky (`potrf`), QR (`geqrf`), SVD (`gesvd`), eigenvalues (`syevd`)
- cuSPARSE: CSR SpMV, SpMM, COO, `axpyi`
- cuBLASLt: basic matmul with ReLU/GELU/Bias epilogues

### Recent Improvements (2026-05-15) 🎉
- ✅ **OpenMP Parallelization** — All major O(n²) and O(n³) CPU reference compute paths across cuBLAS, cuBLASLt, cuDNN, cuFFT, cuSPARSE, and core now use conditional `#pragma omp parallel for` with workload-dependent thresholds. Shared `include/vgre/common/openmp_helper.h` header. Build 122/122 targets, 108–110/110 tests passing.

### Previous Improvements (2026-05-13 – 2026-05-14) 🎉
- ✅ **CUDA Runtime Gaps Closed** — `cudaStreamWaitEvent`, `cudaEventQuery`, `cudaStreamAddCallback`, `cudaLaunchHostFunc`, `cudaGetErrorName`/`GetErrorString`, `cudaMemcpyToSymbol`/`FromSymbol` (sync+async), `cudaMallocArray`/`cudaMalloc3DArray`/`cudaMalloc3D`, `cudaPointerGetAttributes`, `cudaMemset2D`/`3D`/`2DAsync`/`3DAsync`, all graph node types (kernel, memset, host, child, empty, event-record/wait, mem-alloc/free), graph introspection & exec mutation, stream capture introspection, texture/surface object APIs, device/function attributes, dependencies & user objects, external memory/semaphore.
- ✅ **cuBLAS Backfill** — Level-2 (trsv, ger, symv, gbmv, syr, syr2, trmv, tbsv, tpsv, spmv, sbmv, spr, spr2, tbmv, tpmv), Level-3 (trsm, syrk, syr2k, trmm, symm, batched variants), Hermitian (Cherk, Zherk, Cher2k, Zher2k, Chemm, Zhemm), `GemmEx`, logger.
- ✅ **cuDNN Backward + Training** — Backward data/filter/bias for conv, BN training, activation/softmax/pooling backward, dropout, RNN, multi-head attention, CTC loss, LRN, divisive normalization, tensor ops, INT8x4/x32 packed layouts.
- ✅ **Missing Libraries** — cuFFT (reference DFT), cuRAND (mt19937_64), cuSOLVER (LAPACK delegation), cuSPARSE (CSR SpMV/SpMM), cuBLASLt (matmul + epilogues) all now functional.
- ✅ **CUDA Driver Expansion** — `cuMemAllocManaged`, `cuMemHostAlloc`/`Register`/`Unregister`, `cuMemAllocPitch`, all 2D/3D memcpy variants, all async memcpy variants, peer copy, all memset variants, stream callback/query/flags/priority/id/ctx/capture, context management, device queries, texture reference APIs, module fat-binary + linker, cooperative launch, occupancy heuristics, graph APIs, external memory/semaphore.
- ✅ **PTX Expansion** — Texture (`tex`/`tld4`/`txq`/`suld`/`sust`), shared atomics, all `cvt.*` rounding modes, FP16 vector loads/stores, wide integer MAD/MUL/DIV/REM, warp primitives (`match.sync`, `elect.sync`), `grid.sync`, `griddepcontrol`, TMA 3D/4D/5D, `cp.reduce.async`, `tcgen05.mma` (Blackwell), `prmt.b32`, `sad.u32`, MMA INT4/binary.
- ✅ **NCCL P2P** — `ncclSend`/`Recv`, `ncclAllToAll`, `ncclGather`/`Scatter`.
- ✅ **Cooperative Groups + CUB** — Full cooperative groups API + CUB fallback headers.
- ✅ **Deployment** — K8s Device Plugin (Go gRPC) + SLURM GRES plugin (C).
- ✅ **CDP** — `cudaDeviceSynchronize`, `cudaGetParameterBufferV2`, `cudaLaunchDeviceV2`.

### Previous Improvements (2026-05-06) 🎉
- ✅ **Eliminated Heuristic Fallbacks** — Kernel parser now requires Clang for accurate instruction analysis
- ✅ **Real Hardware Queries** — All system metrics use actual hardware interfaces
- ✅ **Stability Fixes** — Static destruction deadlock eliminated; occupancy uses real PTX register parsing
- ✅ **All Tests Passing** — All 65 tests passing with real implementations

### Previous Improvements (2026-04-30) 🎉
- ✅ **AES-NI hardware acceleration** — 4-block parallel AES-256-CTR pipeline
- ✅ **JIT kernel compilation upgraded** — `-O3 -march=native -fno-math-errno -fno-trapping-math`
- ✅ **GPU memory bandwidth model** — `recordMemoryBandwidth()` + `getMemoryBandwidthStats()`
- ✅ **JIT cache flag versioning**
- ✅ **NUMA-aware allocation**
- ✅ **Bandwidth calibration cached**
- ✅ **SharedMemory pooled in serial path**
- ✅ **OpenMP schedule `guided`**
- ✅ **UVM migration interval configurable**
- ✅ **CTest LD_LIBRARY_PATH fix**
- ✅ **UDP discovery authentication**
- ✅ **Mesh topology**
- ✅ **Code consolidation**

### Known Limitations ⚠️
- 10–50× slower than real GPU for compute-bound kernels (CPU execution; AVX-512 auto-vectorisation + OpenMP + NUMA binding reduce the gap)
- **Complex BLAS missing**: No `cublasCgemm`, `cublasZgemm`, `cublasCgemv`, etc.
- **cuFFT slow for large transforms**: Reference O(n²) DFT only; no FFTW3/MKL delegation.
- **cuSOLVER limited**: Only Cholesky, QR, SVD, eigenvalues. No LU or least-squares.
- **cuSPARSE limited**: Only CSR SpMV/SpMM. No format conversions or sparse solvers.
- **cuBLASLt no heuristics**: Always falls back to reference GEMM.
- **NCCL no RDMA**: Multi-node uses TCP sockets.

See `docs/missingFeatures.md` for the complete exhaustive list.

---

## Quick Start

```bash
# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
ctest --output-on-failure -j$(nproc)

# Basic usage
./examples/vector_addition
```

## Documentation

- [`docs/implementationPlan.md`](docs/implementationPlan.md) — Implementation roadmap and phase tracker
- [`docs/missingFeatures.md`](docs/missingFeatures.md) — Exhaustive list of missing/incomplete features
- [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) — Canonical project status with coverage metrics
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System architecture
- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) — User guide
- [`docs/api_reference.md`](docs/api_reference.md) — API reference

## License

MIT License — see `LICENSE` file for details.
