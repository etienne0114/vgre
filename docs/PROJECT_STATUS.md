# VGRE Project Status (Canonical)

**Last Updated**: 2026-05-15 (OpenMP optimization + deep doc audit)  
**Status**: Development / CI-Ready (not general-production-ready for large-scale ML training; see `missingFeatures.md` for gaps)  
**Test Results**: 108–110/110 tests passing (≥98%), build 122/122 targets  

---

## Executive Summary

VGRE (Virtual GPU Runtime Engine) is a CPU-based CUDA emulation runtime. The project has completed all nine implementation phases. Core memory, stream, event, kernel launch, graph, BLAS, DNN, and library shim paths are solid. Primary remaining gaps are: complex-precision BLAS, cuFFT optimized backend, cuSOLVER LU/least-squares, cuSPARSE format conversions, and some cuBLASLt heuristic features.

**Key Metrics:**
- **Test Coverage**: 108–110/110 tests passing (≥98%), total run time ~19–25 seconds
- **Platform Support**: Linux (primary), macOS, Windows — all functional
- **Performance**: 10–50× slower than real GPU for compute-bound workloads; 5–15× for memory-bound workloads. OpenMP parallelization added to all major compute paths (commit `803b76f`).
- **Critical Issues**: 0
- **CUDA Runtime API Coverage**: ~95% (~101+ of ~110 commonly-used functions implemented)
- **CUDA Driver API Coverage**: ~95% (~56+ of ~60 commonly-used functions implemented)
- **cuBLAS Coverage**: ~85% (~61+ of ~72 real/integer functions; complex C/Z GEMM/GEMV missing)
- **cuDNN Coverage**: ~90% (~65+ of ~72 major functions; Backend API attention routing missing)
- **PTX ISA Coverage**: ~95% (~110+ of ~115 commonly-used instructions)
- **Production Readiness**: Development / CI-ready. Missing complex BLAS, optimized FFT, and sparse solvers limit arbitrary PyTorch/TensorFlow/JAX workloads.

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
- **cuBLAS** (~61+ functions): `cublasCreate/Destroy/GetVersion`, `cublasSgemm/Dgemm/Hgemm`, `cublasSaxpy/Daxpy`, `cublasSdot/Ddot`, `cublasSnrm2/Dnrm2`, `cublasSscal/Dscal`, `cublasScopy/Dcopy`, `cublasSswap/Dswap`, `cublasSasum/Dasum`, `cublasIsamax/Idamax/Isamin/Idamin`, `cublasSrot/Drot/Srotm/Drotm/Srotg/Drotg/Srotmg/Drotmg`, `cublasSetPointerMode/GetPointerMode/SetAtomicsMode/GetAtomicsMode`, `cublasLoggerConfigure/SetLoggerCallback/GetLoggerCallback`, Level-2 (`Strsv/Dtrsv/Sger/Dger/Ssymv/Dsymv/Sgbmv/Dgbmv/Ssyr/Dsyr/Ssyr2/Dsyr2/Strmv/Dtrmv/Stbsv/Stpsv/Sspmv/Ssbmv/Sspr/Sspr2/Stbmv/Stpmv`), Level-3 (`Strsm/Dtrsm/Ssyrk/Dsyrk/Ssyr2k/Dsyr2k/Strmm/Dtrmm/Ssymm/Dsymm`), batched variants, `cublasGemmEx/GemmBatchedEx/GemmStridedBatchedEx`, Hermitian (`Cherk/Zherk/Cher2k/Zher2k/Chemm/Zhemm`).
  - **Still missing**: Complex C/Z Level-1/2/3 routines (Cgemm, Zgemm, Cgemv, Zgemv, Caxpy, Zaxpy, Cdotc, Zdotc, etc.)
- **cuDNN** (~65+ functions): `cudnnCreate/Destroy`, all descriptors, `cudnnConvolutionForward/BackwardData/BackwardFilter/BackwardBias`, `cudnnActivationForward/Backward`, `cudnnPoolingForward/Backward`, `cudnnSoftmaxForward/Backward`, `cudnnBatchNormalizationForwardInference/ForwardTraining/Backward`, `cudnnDropoutForward/Backward`, `cudnnRNNForwardInference/Training/BackwardData/BackwardWeights`, `cudnnMultiHeadAttnForward/BackwardData/BackwardWeights`, `cudnnCTCLoss`, `cudnnLRNCrossChannelForward/Backward`, `cudnnDivisiveNormalizationForward/Backward`, `cudnnTransformTensor`, `cudnnOpTensor`, `cudnnReduceTensor`, `cudnnAddTensor`, `cudnnBackendCreateDescriptor/DestroyDescriptor/Finalize/Initialize/Execute` (wired to legacy for conv/activation/pool/softmax/reduction/matmul/BN/norm/RNN/concat/signal/gen_stats/bn_bwd_weights).
  - **Still missing**: Backend API attention routing (`CUDNN_BACKEND_OPERATION_ATTENTION_DESCRIPTOR`).
- **NCCL** (~16 functions): `ncclCommInitRank/InitAll/Destroy/Abort/Count/UserRank`, `ncclGetUniqueId/Version/LastError`, `ncclAllReduce/Broadcast/Reduce/AllGather/ReduceScatter`, `ncclGroupStart/GroupEnd`, `ncclSend/Recv`, `ncclAllToAll`, `ncclGather/Scatter`.
  - **Still missing**: Multi-node RDMA transport; topology-aware ring/tree algorithm selection.
- **cuFFT** (~12 functions): `cufftPlan1d/2d/3d/Many`, `cufftDestroy`, `cufftExecC2C/Z2Z/R2C/C2R/D2Z/Z2D`, `cufftSetStream/WorkArea`. Reference CPU DFT/IDFT.
  - **Still missing**: FFTW3/MKL delegation for large transforms.
- **cuRAND** (~18 functions): `curandCreateGenerator/DestroyGenerator/SetPseudoRandomGeneratorSeed/SetGeneratorOffset/SetGeneratorOrdering/GetVersion`, `curandGenerate/GenerateUniform/GenerateNormal/GenerateLogNormal/GeneratePoisson` for XORWOW, MRG32k3a, MTGP32, MT19937.
  - **Still missing**: Per-stream parallel generator partitioning; device-side RNG (intentionally not supported in CPU model).
- **cuSOLVER** (~26 functions): `cusolverDnCreate/Destroy`, `cusolverDnSpotrf/Dpotrf/Sgeqrf/Dgeqrf/Sgesvd/Dgesvd/Ssyevd/Dsyevd`.
  - **Still missing**: `getrf` (LU), `getrs`, `ormqr`, `gelsd`, sparse solvers.
- **cuSPARSE** (~20 functions): `cusparseCreate/Destroy`, `cusparseSpMV/SpMM` (CSR + COO), `cusparseAxpyi`, `cusparseCreateCoo`.
  - **Still missing**: Format conversions (CSR↔CSC↔COO), sparse triangular solve (`SpSV`), sparse factorization.
- **cuBLASLt** (~12 functions): `cublasLtCreate/Destroy`, `cublasLtMatmulDescCreate/Destroy/SetAttribute/GetAttribute`, `cublasLtMatrixLayoutCreate/Destroy/SetAttribute/GetAttribute`, `cublasLtMatmul` with ReLU/GELU/Bias epilogues.
  - **Still missing**: Heuristic selection, algorithm caching, advanced epilogues (DRELU, DGELU, BGRADIENT).

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

1. **Complex BLAS missing**: No `cublasCgemm`, `cublasZgemm`, `cublasCgemv`, etc.
2. **cuFFT slow for large transforms**: Reference O(n²) DFT; no FFTW3/MKL delegation.
3. **cuSOLVER limited**: Only Cholesky, QR, SVD, eigenvalues. No LU or least-squares.
4. **cuSPARSE limited**: Only CSR SpMV/SpMM. No format conversions or sparse solvers.
5. **cuBLASLt no heuristics**: Always falls back to reference GEMM.
6. **NCCL no RDMA**: Multi-node uses TCP sockets, not RDMA.

---

*Last updated: 2026-05-15*
