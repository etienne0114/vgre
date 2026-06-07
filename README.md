# VGRE — Virtual GPU Runtime Engine

**A CUDA emulation runtime** that allows CUDA applications to run on CPU without a physical GPU.

> **PROJECT STATUS**: CI-Ready — **192/192 tests passing**. Full CUDA, BLAS, DNN, FFT/RNG/solver/sparse, NCCL, profiling, and distributed-cluster paths are implemented. Zero compiler warnings; warnings-as-errors enabled. Auto-detected SIMD, RDMA, TPM2, and libsecret features active at build time. See `docs/missingFeatures.md` for the remaining hardware-level architectural boundaries.

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

**Core Stability**: ✅ 192/192 tests passing, zero warnings, zero critical issues  
**CUDA Runtime API Coverage**: ~95% (~101+ of ~110 commonly-used functions)  
**CUDA Driver API Coverage**: ~95% (~56+ of ~60 commonly-used functions)  
**cuBLAS Coverage**: Level-1/2/3 real + complex C/Z + Hermitian core API implemented  
**cuDNN Coverage**: Major legacy ops + backward/training + Backend API attention routing implemented  
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
- cuBLAS: Level-1/2/3 real and complex C/Z routines, Hermitian variants, batched GEMM, `GemmEx`, pointer/atomics modes, logger
- cuDNN: conv, pool, activation, softmax, BN, dropout, RNN, attention, CTC loss, LRN, divisive norm, tensor ops, INT8 packed layouts, Backend API routing including SDPA attention
- NCCL: AllReduce, Broadcast, Reduce, AllGather, ReduceScatter, Send, Recv, AllToAll, Gather, Scatter, TCP multi-node and optional RDMA/RoCE bulk path
- cuFFT: O(n log n) 1D/2D/3D transforms (C2C, R2C, C2R, Z2Z, D2Z, Z2D), FP16 C16C/R16C/C16R, strided batched `PlanMany`, optional FFTW3
- cuRAND: XORWOW, MRG32k3a, MTGP32, MT19937, Sobol generators with per-handle locking
- cuSOLVER: Dense LAPACK-backed potrf/geqrf/gesvd/syevd/getrf/getrs/ormqr/gelsd and sparse cusolverSp CSR-to-dense LAPACK paths
- cuSPARSE: CSR/COO/CSC SpMV/SpMM, SparseToDense, DenseToSparse, SpSV, SpGEMM, ILU0, IC0
- cuBLASLt: matmul with heuristic cache, scale pointers, amaxD, and full epilogue set

### Recent Improvements (2026-06-07) 🎉
- ✅ **Zero compiler warnings** — All `-Wall -Wextra` warnings resolved: strict-aliasing `reinterpret_cast` replaced with `std::memcpy`, `warn_unused_result` captures, `static thread_local` in headers converted to `extern thread_local` (real semantic bug: each TU had its own independent `g_current_ctx` copy), `volatile` for optimization-unreachable test code, `snprintf` replacing truncating `strncpy`. Warnings-as-errors (`-Werror`) enabled globally.
- ✅ **Auto-detected features** — CMake now auto-probes libibverbs, libtss2-esys, libsecret-1, and `/proc/cpuinfo`/`sysctl` for native SIMD. All four features default ON when their libraries are installed; build never fails due to missing optional deps.
- ✅ **K8s deployment hardening** — New `--is-master` flag in `vgre-worker` for headless master mode in containers. Device plugin rewritten: proper `context.Context` in all gRPC methods, `grpc.NewClient` + `insecure.NewCredentials`, `VGRE_DEVICE_PLUGIN_PATH` env var overrides kubelet socket path, `<-stream.Context().Done()` replaces infinite sleep in `ListAndWatch`. Dockerfiles switched to distroless base image; C++ runtime libs added.
- ✅ **Cross-platform FFI error messages** — `vgre_ffi.dart` now emits OS-specific library load hints (Linux: `LD_LIBRARY_PATH`, macOS: `DYLD_LIBRARY_PATH`, Windows: `%LOCALAPPDATA%\VGRE\lib`) instead of Windows-only error text.
- ✅ **`g_current_ctx` cross-TU bug fixed** — `static thread_local` in a shared header gave every translation unit its own independent per-thread context pointer; `cuCtxSetCurrent` in one TU was invisible to `cuCtxGetCurrent` in another. Fixed with `extern thread_local` + single canonical definition.

### Recent Improvements (2026-05-30) 🎉
- ✅ **Heuristic Elimination (Track 9/10)** — Dynamic bandwidth calibration (Z-score, DDR/HBM detection), CPUID-based FLOPS/IPC detection, powers-of-2 thread search, and problem-size-based algorithm selection replace all prior hardcoded magic numbers. **192/192 tests passing**.
- ✅ **Advanced Math Hardening (Track 8)** — Zero-copy sparse matrix views (CSR↔CSC, CSR→BSR), cache-oblivious matmul/transpose/conv2D, mixed-precision (FP16/BF16/FP8/INT8), block-sparse SIMD (AVX-512/AVX2), and Intel AMX tensor core emulation, all fully integrated.
- ✅ **TLB & SPSC Hardening (Track 7)** — 3-level TLB cache (L1 thread-local 256×8-way AVX2, L2 thread-local 1024×16-way, shared sharded 16×256×4-way), SPSC ring per-stream fast path, and NUMA-aware thread affinity registry.
- ✅ **OpenMP Parallelization** — All major O(n²) and O(n³) CPU compute paths use conditional `#pragma omp parallel for`. Shared `include/vgre/common/openmp_helper.h` header.

### Previous Improvements (2026-05-13 – 2026-05-14) 🎉
- ✅ **CUDA Runtime Gaps Closed** — `cudaStreamWaitEvent`, `cudaEventQuery`, `cudaStreamAddCallback`, `cudaLaunchHostFunc`, `cudaGetErrorName`/`GetErrorString`, `cudaMemcpyToSymbol`/`FromSymbol` (sync+async), `cudaMallocArray`/`cudaMalloc3DArray`/`cudaMalloc3D`, `cudaPointerGetAttributes`, `cudaMemset2D`/`3D`/`2DAsync`/`3DAsync`, all graph node types (kernel, memset, host, child, empty, event-record/wait, mem-alloc/free), graph introspection & exec mutation, stream capture introspection, texture/surface object APIs, device/function attributes, dependencies & user objects, external memory/semaphore.
- ✅ **cuBLAS Backfill** — Level-2 (trsv, ger, symv, gbmv, syr, syr2, trmv, tbsv, tpsv, spmv, sbmv, spr, spr2, tbmv, tpmv), Level-3 (trsm, syrk, syr2k, trmm, symm, batched variants), Hermitian (Cherk, Zherk, Cher2k, Zher2k, Chemm, Zhemm), `GemmEx`, logger.
- ✅ **cuDNN Backward + Training** — Backward data/filter/bias for conv, BN training, activation/softmax/pooling backward, dropout, RNN, multi-head attention, CTC loss, LRN, divisive normalization, tensor ops, INT8x4/x32 packed layouts.
- ✅ **Library Shim Completion** — cuFFT (O(n log n) FFT + FP16), cuRAND, cuSOLVER, cuSPARSE, and cuBLASLt all now have functional implementations.
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
- **Physical GPU counters unavailable**: profiling uses VGRE timeline/Chrome/OTLP and LLVM-IR instruction classification, not NVIDIA PMU/CUPTI counters.
- **Sparse direct factorization with fill-in**: requires an external sparse solver backend; current sparse solver paths are CSR-to-dense or zero-fill incomplete factorizations.
- **Optional transports**: RDMA and gRPC require explicit build flags and system libraries; TCP remains the default.
- **Device-side cuRAND**: intentionally unsupported in the CPU runtime model.

See `docs/missingFeatures.md` for the complete exhaustive list.

---

## Quick Start

```bash
# Build — optional features (RDMA, TPM2, libsecret, SIMD) auto-detected
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Run tests
ctest --output-on-failure -j$(nproc)

# Basic usage
./examples/vector_addition
```

## Documentation

- [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) — Canonical source of truth for all verified capabilities, test status, and coverage metrics
- [`docs/missingFeatures.md`](docs/missingFeatures.md) — Definitive registry of permanent hardware-level architectural limitations (boundary conditions)
- [`docs/implementationPlan.md`](docs/implementationPlan.md) — Forward-looking roadmap tracking advanced future expansions (SASS, RDMA, etc.)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System architecture and execution pipeline
- [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) — User guide and setup instructions
- [`docs/api_reference.md`](docs/api_reference.md) — API reference for C/Python bindings

## License

MIT License — see `LICENSE` file for details.
