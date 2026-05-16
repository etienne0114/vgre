# VGRE Project Status (Canonical)

**Last Updated**: 2026-05-16 (cuFFT FP16/PlanMany + profiler split + doc audit)  
**Status**: Development / CI-Ready (not general-production-ready for large-scale ML training; see `missingFeatures.md` for gaps)  
**Test Results**: Focused touched tests pass: cuFFT 13/13, cuDNN tensor ops 7/7, cuDNN attention 4/4. Latest full-suite note: 113/114 with one intermittent TCP race.  

---

## Executive Summary

VGRE (Virtual GPU Runtime Engine) is a CPU-based CUDA emulation runtime. The project has completed the documented implementation phases. Core memory, stream, event, kernel launch, graph, BLAS, DNN, FFT/RNG/solver/sparse, NCCL, profiling, and deployment paths are functional. Remaining items are mostly architectural or environment-dependent: physical GPU PMU/CUPTI counters are not available in a CPU runtime, full-fill sparse direct factorization still needs an external sparse solver backend, and RDMA/gRPC require optional build dependencies.

**Key Metrics:**
- **Test Coverage**: focused touched tests pass; latest full-suite note 113/114 with one intermittent TCP cluster race
- **Platform Support**: Linux (primary), macOS, Windows — all functional
- **Performance**: 10–50× slower than real GPU for compute-bound workloads; 5–15× for memory-bound workloads. OpenMP parallelization added to all major compute paths (commit `803b76f`).
- **Critical Issues**: 0
- **CUDA Runtime API Coverage**: ~95% (~101+ of ~110 commonly-used functions implemented)
- **CUDA Driver API Coverage**: ~95% (~56+ of ~60 commonly-used functions implemented)
- **cuBLAS Coverage**: Level-1/2/3 real + complex C/Z + Hermitian implemented for the documented core API
- **cuDNN Coverage**: legacy major ops + backward/training + Backend API routing including SDPA attention
- **PTX ISA Coverage**: ~95% (~110+ of ~115 commonly-used instructions)
- **Production Readiness**: Development / CI-ready. CPU execution is slower than a physical GPU; optional cluster transports and sparse-direct backends determine production deployment scope.

---

## What Works

### Core CUDA Runtime API
- **CUDA Runtime API** (~101+ functions): memory allocation/free (`cudaMalloc`, `cudaFree`, `cudaMallocManaged`, `cudaMallocAsync`, `cudaMallocFromPoolAsync`, `cudaMallocPitch`, `cudaMallocArray`, `cudaMalloc3DArray`, `cudaMalloc3D`), stream create/destroy/query/sync/wait-event/add-callback/launch-host-func, events (create/record/query/sync/destroy/elapsed-time), error introspection (`cudaGetErrorName`/`GetErrorString` with 50+ mapped codes), symbol copies (`cudaMemcpyToSymbol`/`FromSymbol` sync+async), array allocation (1D/2D/3D via `TextureManager`), pointer introspection (`cudaPointerGetAttributes`), device queries (`cudaGetDeviceProperties`, `cudaDeviceGetAttribute`, `cudaDeviceCanAccessPeer`), peer access, kernel launch (`cudaLaunchKernel`, `cudaLaunchCooperativeKernel`, `cudaLaunchKernelExC`), graph APIs (capture, instantiate, launch, clone, destroy, exec update, all 11 node types including kernel, memset, host, child, empty, event-record/wait, mem-alloc/free), texture/surface objects (`cudaCreateTextureObject`, `cudaDestroyTextureObject`, `cudaCreateSurfaceObject`, `cudaDestroySurfaceObject`, legacy `cudaBindTexture`/`cudaBindTextureToArray`/`cudaBindTexture2D`/`cudaBindSurfaceToArray`), external memory/semaphore (`cudaImportExternalMemory`, `cudaDestroyExternalMemory`), stream capture introspection (`cudaStreamIsCapturing`, `cudaStreamGetCaptureInfo_v2`), device/function attributes (`cudaFuncGetAttributes`, `cudaDeviceGetLimit`/`SetLimit`, `cudaDeviceGetCacheConfig`/`SetCacheConfig`), memset 2D/3D/Async variants.
- CUDA Driver API (~56+ functions): `cuInit`, `cuCtxCreate`/`Destroy`/`GetCurrent`/`SetCurrent`/`Synchronize`, `cuDeviceGet`/`GetAttribute`/`GetCount`/`GetName`/`TotalMem`, `cuEventCreate`/`Destroy`/`ElapsedTime`/`Record`/`Synchronize`, `cuMemAlloc`/`Free`/`AllocManaged`/`HostAlloc`/`HostRegister`/`HostUnregister`/`AllocPitch`, `cuMemcpyDtoD`/`DtoH`/`HtoD`/`2D`/`2DAsync`/`3D`/`3DAsync`/`Peer`/`PeerAsync`, `cuMemsetD8`/`D16`/`D32`/`D2D8`/`D2D16`/`D2D32`, `cuModuleGetFunction`/`GetGlobal`/`GetSurfRef`/`GetTexRef`/`Load`/`LoadData`/`LoadDataEx`/`Unload`/`LoadFatBinary`/`LinkCreate`/`LinkAddData`/`LinkAddFile`/`LinkComplete`/`LinkDestroy`, `cuStreamCreate`/`Destroy`/`Synchronize`/`WaitEvent`/`AddCallback`/`Query`/`GetFlags`/`GetPriority`/`GetId`/`GetCtx`/`IsCapturing`/`GetCaptureInfo`/`UpdateCaptureDependencies`, `cuGraphCreate`/`Destroy`/`Clone`/`AddKernelNode`/`AddMemsetNode`/`AddMemcpyNode`/`AddEmptyNode`/`AddChildGraphNode`/`Instantiate`/`Launch`/`ExecDestroy`/`ExecUpdate`, `cuTexObjectCreate`/`Destroy`, `cuTexRefSetAddress2D`/`SetAddressMode`/`SetFilterMode`/`SetMaxAnisotropy`/`SetMipmapFilterMode`/`SetMipmapLevelBias`/`SetMipmapLevelClamp`/`SetBorderColor`/`SetFlags`/`SetFormat`, `cuSurfRefSetArray`/`SetFormat`, `cuOccupancyMaxActiveBlocksPerMultiprocessor`/`MaxPotentialBlockSize`/`MaxPotentialBlockSizeWithFlags`, `cuLaunchCooperativeKernel`/`MultiDevice`, `cuLaunchKernelEx`, `cuExternalMemory`/`Semaphore` families, IPC memory/event handles.
- OpenCL 1.2 compatibility layer
- P2P peer device access and transfers

### Memory Management
- `cudaMalloc` / `cudaFree` — standard allocation with O(log n) handle lookups
- `cudaMallocManaged` / `cudaFreeManaged` — Unified Virtual Memory with OS page-fault handling
- `cudaMallocAsync` / `cudaFreeAsync` — stream-ordered memory pools (CUDA semantics for oversized requests)
- `cudaMemcpy` / `cudaMemcpyAsync` — synchronous and asynchronous transfers
- `cudaMemset` / `cudaMemset2D` / `cudaMemset3D` / `cudaMemset2DAsync` / `cudaMemset3DAsync` — memory initialization
- Pool allocator: two-path design (slab free-list for ≤ blockSize; direct alloc for oversized), NUMA slab binding ≥ 2 MB via `mbind(MPOL_PREFERRED)`, pointer-provenance validation in `freeToPool`, outstanding-alloc guard in `destroyPool`
- RadixPageTable with correct destructor (no memory leak on teardown)

### Kernel Execution
- JIT kernel compilation via LLVM-18 ORC with persistent disk + memory LRU cache (0 ms on cache hit)
- Kernel fusion (consecutive compatible kernels merged into single JIT)
- Cooperative kernel launch (start-gate via `condition_variable`, dispatched through pre-warmed `BlockWorkerPool`)
- **CUDA Graphs**: capture, instantiation, replay, exec mutation, introspection, dependencies, user objects, all 11 node types (kernel, memcpy, conditional, memset, host, child, empty, event-record, event-wait, memalloc, memfree), `cudaGraphExecUpdate_v2`
- Stream management with priority scheduling
- Event synchronization and timing
- CDP (CUDA Dynamic Parallelism): `cudaLaunchDevice`, `cudaGetParameterBuffer`, `cudaDeviceSynchronize`, `cudaLaunchDeviceV2`, `cudaGetParameterBufferV2`

### Advanced Features
- **Unified Virtual Memory (UVM)**: Managed memory with OS-level page fault handling (SIGSEGV on Linux, VEH on Windows)
- **Texture & Surface**: `tex1D`/`tex2D`/`tex3D`/`tex1Dfetch` and `surf2Dread`/`surf2Dwrite` templates in `cpu_cuda_env.h`. CUDART texture/surface object APIs fully implemented (`cudaCreateTextureObject`, `cudaDestroyTextureObject`, `cudaGetTextureObjectResourceDesc`, etc.). Driver-level `cuTexRefSetAddress2D`/filter/mipmap/border controls implemented. PTX `tex`/`tld4`/`txq`/`suld`/`sust` instructions implemented.
- **Warp-Level Intrinsics**: `__shfl_sync`, `__ballot_sync`, `__activemask`, `__redux_sync`, `__match_any_sync`, `__match_all_sync`
- **FP16 & BFloat16**: `__half` with full operator set in `cpu_cuda_fp16.h`; `__nv_bfloat16` in `cpu_cuda_env.h`; WMMA `nvcuda::wmma` fragments with `__half` in `wmma_emulation.h`. PTX `ld.global.v2/v4.f16`, `st.global.v2/v4.f16` implemented.
- **Tensor Core Emulation (WMMA)**: Matrix multiply-accumulate operations via scalar FP32 fallback (`mma.sync`, `wgmma.mma_async`, `tcgen05.mma`)
- **CUDA Dynamic Parallelism (CDP)**: Full API implemented (`cudaLaunchDevice`, `cudaGetParameterBuffer`, `cudaDeviceSynchronize`, `V2` variants)
- **Inline PTX Assembly**: ~110+ PTX opcodes translated to C++; unrecognized opcodes throw (no silent fallback)
- **CUDA IPC**: Multi-process memory sharing via real POSIX shared memory for event handles
- **Cooperative Groups**: `thread_block`, `coalesced_group`, `thread_block_tile`, `grid_group`, `multi_grid_group` with `sync()`, `shfl()`, `reduce()`, `partition()`, `match_any()`, `match_all()`
- **CUB Fallback**: `cub::WarpReduce/BlockReduce/WarpScan/BlockScan/CachingDeviceAllocator`

### Library Shims
- **cuBLAS**: Level-1/2/3 real S/D, complex C/Z Level-1/2/3, Hermitian C/Z, batched/strided GEMM, `GemmEx`, pointer/atomics modes, and logger callbacks are implemented.
- **cuDNN**: Forward/backward/training paths for conv, pool, activation, softmax, BN, dropout, RNN, attention, CTC, LRN, divisive norm, tensor ops, INT8x4/INT8x32 packed layouts, and Backend API descriptor execution including SDPA attention and re-executable plan descriptors.
- **NCCL**: Communicator lifecycle, all common collectives, P2P (`Send`/`Recv`), `AllToAll`, `Gather`, `Scatter`, single-node shared-memory fast path, TCP multi-node path, and optional RDMA/RoCE bulk transport when built with `VGRE_ENABLE_RDMA`.
- **cuFFT**: O(n log n) Cooley-Tukey + Bluestein backend for C2C/Z2Z/R2C/C2R/D2Z/Z2D, half-precision C16C/R16C/C16R widen-compute-narrow paths, strided batched 1D `PlanMany`, size estimates, and IPC plan export/import.
- **cuRAND**: XORWOW, MRG32k3a, MTGP32, MT19937, Sobol quasi-random generators, per-handle mutexes, and host generation APIs. Device-side RNG is intentionally not supported in the CPU runtime.
- **cuSOLVER**: Dense potrf/geqrf/gesvd/syevd/getrf/getrs/ormqr/gelsd via LAPACK, plus cusolverSp LU/Cholesky/least-squares/eigen paths via CSR-to-dense LAPACK extraction.
- **cuSPARSE**: Generic descriptors, CSR/COO/CSC SpMV/SpMM, SparseToDense, DenseToSparse, SpSV, SpGEMM, ILU0, IC0, FP16/BF16 widen-compute-narrow paths. Full-fill sparse direct factorization remains an external-library integration item.
- **cuBLASLt**: Descriptor lifecycle, matrix layouts/preferences, algorithm heuristic cache, matmul for F32/F64/F16/BF16/INT8, scale pointers, amaxD, and full epilogue set including ReLU/GELU/Bias/DReLU/DGELU/BGRAD variants.

### Cluster Networking
- TCP cluster networking with multi-node partitioned 3D kernel dispatch (recursive bisection)
- Authenticated encrypted cluster channels (HMAC-SHA256 + AES-256-CTR)
- Session key zeroization via `vgre_secure_zero`
- 2048-bit sliding replay bitmap (RFC 4303)
- `sendAll` uses `poll(POLLOUT)` with 30-second deadline
- Hardware-backed auth token storage (Linux keyring, macOS Keychain, Windows CredMan, TPM 2.0)
- Adaptive execution engine: UCB1 multi-armed bandit auto-tunes thread count per kernel
- UDP discovery with HMAC-SHA256 authentication
- Mesh topology support (`VGRE_MESH_PEERS` for any-to-any connections)
- GPU passthrough for cluster workers with physical NVIDIA GPUs
- **CapabilityPacket** carries real GPU info
- **IPv6 dual-stack**
- **NetworkProfiler**: 1000-sample ring buffer
- **Cluster-wide profiling sync**: NTP-style clock exchange for aligned profiler timestamps across nodes

### Performance & Profiling
- **OMP Hot-Path**: Per-thread `LocalAccum` (cache-line aligned); `schedule(guided)`
- **Per-Grid Timing**: One `chrono::now()` pair brackets entire OMP region
- **BlockWorkerPool**: Pre-warmed thread pool for cooperative kernels
- **Scheduler**: Zero heap allocation per task dequeue
- **AES-NI Hardware Acceleration**: 4-block parallel AES-256-CTR pipeline
- **NUMA-Aware Allocation**: Allocations ≥ 2 MB bound to NUMA node 0
- **Bandwidth Calibration**: Process-wide cache
- **SharedMemory Pooling**: Pre-allocated outside block loop
- **Chrome Trace Export**: `toChromeTraceJSON()`
- **Global SIMD Width**: `globalOptimalVectorWidth_` atomic calibrated once at startup
- **OpenMP Parallelization**: All major cuBLAS, cuBLASLt, cuDNN, cuFFT, cuSPARSE, and core compute paths now use conditional `#pragma omp parallel for` (commit `803b76f`)

### Temperature Monitoring (Real on All Platforms)
- **Linux**: `/sys/class/thermal/` + `/sys/class/hwmon/`
- **macOS**: IOKit SMC (`TC0P`, `TC0F`, `Tp09`, `Tp0P`, `Tp19`)
- **Windows**: WMI `MSAcpi_ThermalZoneTemperature` via COM background thread with 5-second TTL cache

### Cross-Platform Support
- **Linux**: Full support — NUMA, Keyring, libsecret, TPM 2.0, `perf_event_open`, hwmon temperature
- **Windows**: Full support — Credential Manager, WinSock2, BCryptGenRandom, VEH, WMI temperature, registry CPU frequency fallback
- **macOS**: Full support — Keychain, `SO_NOSIGPIPE`, `getentropy`, IOKit temperature, CPUID leaf 0x16 for Intel

---

## Build & Test

```bash
# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Test
ctest --output-on-failure -j$(nproc)
```

**Current test count**: 110 tests (108–110 pass; 0–2 flaky TCP cluster tests occasionally timeout under parallel load but pass individually).

---

## Known Limitations

See `docs/missingFeatures.md` for the exhaustive list. Primary blockers for arbitrary framework workloads:

1. **CPU performance ceiling**: VGRE is still much slower than a physical GPU for large compute-bound training workloads.
2. **Physical GPU profiler counters**: NVIDIA CUPTI/PMU counters are unavailable in CPU emulation; VGRE reports timeline/OTLP/Chrome trace and LLVM-IR instruction classification.
3. **Sparse direct factorization with fill-in**: Full-fill sparse LU/Cholesky needs an external sparse solver backend such as UMFPACK/SuperLU/CHOLMOD.
4. **Optional transports**: RDMA and gRPC require explicit build flags and system libraries; TCP remains the portable default.
5. **Device-side RNG**: cuRAND device API is intentionally unsupported because VGRE has no CUDA device execution model.

---

*Last updated: 2026-05-16*
