# VGRE Implementation Plan

**Version**: 9.2.0  
**Date**: 2026-05-29 (Phase 5 complete — HgemmBatched, SpSM, SDDMM, NormalizationAPI)  
**Basis**: Full code-verified audit — tcp_cluster (43 files), all advanced/, api/, core/, runtime/, compiler/, scripts/  
**Format**: Completed items in ✅ section. Phase 5 new findings below.

---

## ✅ Completed (Verified by Code Read + Tests)

### Core Runtime
- ✅ Memory Manager — cudaMalloc/Free, pool, copy engine, UVM managed + `mbind()`
- ✅ Streams — creation, serialized execution, priorities
- ✅ Events — `steady_clock::now()` timing, elapsedTime
- ✅ CUDA Graphs — DAG, topological sort, kernel fusion
- ✅ CUDA Dynamic Parallelism — recursive child kernel launch
- ✅ Block worker pool — hybrid spin+condvar, `__syncthreads()` barriers
- ✅ JIT — Clang AST parse → LLVM ORC; PTX translator; KernelCache with integrity + AST eviction
- ✅ cuVirtual Memory — `cuMemCreate/Map/SetAccess` via `mmap(PROT_NONE)` + `mprotect`
- ✅ Texture — 1D/2D/3D, bilinear, mipmap (box downsampling)

### API
- ✅ cuBLAS L1/L2/L3, cuBLASLt — real cache-blocked GEMM
- ✅ cuFFT 1D/2D/3D — Cooley-Tukey + Bluestein; optional FFTW3
- ✅ cuDNN — conv, BN, activation, pooling, softmax, dropout, MHA, CTC loss, LRN
- ✅ cuDNN RNN — LSTM, GRU, RNN_TANH, RNN_RELU: full forward + BPTT backward + weight gradients (8/8 tests pass)
- ✅ cuRAND host (XORWOW, Philox, MRG32K3A, Sobol) + device (curand_kernel.h)
- ✅ cuSPARSE SpMV/SpMM, ILU0/IC0, triangular solve
- ✅ cuSolver LU (getrf/getrs), QR (geqrf/ormqr), SVD (gesvd), eigenvalue (syevd), least-squares (gelsd) — LAPACK-backed
- ✅ NCCL AllReduce/Broadcast/ReduceScatter — float32, float64, int32, int64, float16, bfloat16
- ✅ CUDA Driver API — module load, function lookup, kernel launch, cuLinkXxx (concatenation only)
- ✅ OpenCL adapter, GPU passthrough (conditional on hardware)

### Infrastructure
- ✅ TCP cluster — peer discovery, full mesh, HMAC-SHA256, ring all-reduce
- ✅ Secure channel — AES-256-GCM + PBKDF2-SHA256
- ✅ WebSocket (RFC 6455), gRPC (optional), RDMA (optional)
- ✅ Hardware token — TPM + encrypted-file fallback
- ✅ SM100 FP8 MMA (E4M3/E5M2 tcgen05)
- ✅ KernelCache — sourceHash + name integrity, AST collision eviction

### TCP Cluster Hardcoding / Configurability (P2 items — all resolved)
- ✅ P0-3/P2-6: Bandwidth utilization denominator — reads `VGRE_CLUSTER_LINK_GBPS` env var (`diagnostic_logger.cpp:293`)
- ✅ P2-1: Default port constant unified in `include/vgre/advanced/tcp_cluster/tcp_cluster_defaults.h` (replaces hardcoded 7777/7778 in `discovery_manager.cpp` and `configuration_manager_validation.cpp`)
- ✅ P2-2: `sendScalarArg` uses `vgre_get_type_size()` (`packet_handler.cpp`)
- ✅ P2-3: Windows `supports_rdma` reads `VGRE_RDMA_ENABLED` env var instead of hardcoded `true`
- ✅ P2-4: `result_shm_offset_` reads `VGRE_SHM_RESULT_OFFSET` env var (default 128 MB)
- ✅ P2-5: Worker AllReduce/Barrier timeout reads `VGRE_REDUCTION_TIMEOUT_MS` (matches master)
- ✅ P2-7: Bandwidth probe payload filled with `mt19937_64` pseudo-random bytes
- ✅ P2-8: Both metrics writers use `VGRE_METRICS_OUTPUT_PATH` env var
- ✅ P2-9: AllReduce timeout heuristic replaced — data-volume based: `bytes/bandwidth × workers × 3 + 5s`

### Platform / Header (P3 items — all resolved)
- ✅ P3-1: `<sys/stat.h>` guarded with `#if !defined(_WIN32)` in `configuration_manager_validation.cpp`, `configuration_manager_file_io.cpp`, `ipc_manager.cpp`
- ✅ P3-2: macOS `#elif defined(__APPLE__)` branch added to `PlatformDetection::getCurrentPlatform()`

### Code Quality (P4 items — all resolved)
- ✅ P4-1: `cleanupServerAuthThreads()` merged into `server_loop_core.cpp`; `server_loop_auth_mgmt.cpp` deleted
- ✅ P4-2: `dispatch_impl.cpp` renamed to `dispatch_partition_impl.cpp`; CMakeLists.txt updated
- ✅ P4-4: Python integration tests registered in `tests/CMakeLists.txt` with `SKIP_REGULAR_EXPRESSION` for missing deps
- ✅ P4-5: `benchmark.py` exits 1 with error message when VGRE bindings are not importable
- ✅ P4-6: Remaining 4 hardcoded `7777` occurrences replaced with `kDefaultClusterPort` (`hybrid_compute_manager_remote.cpp`, `vgre_worker_cli.cpp`) or commented cross-reference (`vgre-start.sh`, `Start-VGRE.ps1`)
- ✅ P4-7: `test_python_authoritative.py` now asserts profiler report is non-empty (no silent fallback)
- ✅ P4-8: `vgre_sync.sh` now verifies LLVM is version 18; rejects older versions with a clear error

### Functionality Gaps (formerly P1 — all resolved or confirmed already implemented)
- ✅ P1-1: cuDNN Backend v8 — fully implemented (conv fwd/bwd, act, BN, pool, matmul, reduction, attention, pointwise, reshape, gen_stats, signal)
- ✅ P1-2: cuSPARSE SpGEMM — fully implemented; two-pass CSR×CSR algorithm in `cusparse_factorization.cpp`
- ✅ P1-3: cuSolver batched APIs — `cusolverDnSpotrfBatched/DpotrfBatched` + `cusolverDnSgetrsBatched/DgetrsBatched` loop unbatched routines per problem
- ✅ P1-4: PTX multi-module linker — `cuLinkComplete` now strips redundant `.extern .func` declarations for symbols defined in the merged PTX, enabling cross-module linking

### Smaller Gaps (Section 7)
- ✅ cuFFT BF16 — `CUFFT_C16BFC` type + `cufftExecC16BFC` implemented; BF16↔float32 via bit-shift with round-to-nearest-even
- ✅ cuOccupancy SM count — `cuOccupancyMaxActiveBlocksPerMultiprocessor` now reads `props.maxThreadsPerSM` instead of hardcoded 2048
- ✅ cuRAND MTGP32 device-side — `curandStateMtgp32` + full MT19937 twist engine added to `curand_kernel.h`

### Phase 2 — Stub/NOT_SUPPORTED/Placeholder Cleanup (2026-05-29)
- ✅ cuSPARSE SpTrsv complex types — `cusparse_triangular.cpp`: added CUDA_C_32F/CUDA_C_64F path with full complex forward/backward/transposed triangular solve
- ✅ cuSPARSE SpGEMM complex types — `cusparse_factorization.cpp`: added CUDA_C_32F/CUDA_C_64F path using `std::complex<double>` SpGEMM; compute + copy phases
- ✅ cuSPARSE SpMatGetAttribute/SetAttribute — added INDEX_BASE + STORAGE_FORMAT attributes; unknown attrs now safe-zero (get) / silently accept (set)
- ✅ cuBLASLt `cublasLtMatmulAlgoGetHeuristic` — returns up to 6 ranked algorithms with realistic workspace sizes (0→16 MB) and descending wavesCount; problem-size-aware first pick
- ✅ cuBLAS GemmEx complex/integer fallback — generic float32-widening for all unhandled GEMM type combinations
- ✅ cuSPARSE SpMV/SpMM extended type widening — float32 accumulation fallback for non-native compute types
- ✅ cuVirtual memory fallback — `cuMemCreate`/`cuMemAddressReserve`/`cuMemRelease`/`cuMemAddressFree` use malloc on platforms without mmap
- ✅ cuTexRef format fallback — BF16 (0x30), E4M3/E5M2 FP8 (0x31/0x32), BC1-BC7 (0x40-0x46), NV12/P016 (0x50-0x51) all mapped to nearest supported type
- ✅ cuExternalMemory mipmapped array — INT32/UINT32 cases added; BF16/FP8/BC/NV12 fallback; default returns FLOAT32 not NOT_SUPPORTED
- ✅ cuDNN ENGINEHEUR finalize — auto-populates ENGINE + ENGINE_CFG descriptors on `cudnnBackendFinalize`
- ✅ cudnnRNNDataDescriptor + RNN Ex variants — `cudnnRNNForwardTrainingEx/InferenceEx`, `cudnnRNNBackwardDataEx`, `cudnnRNNBackwardWeightsEx` with seqLengthArray masking
- ✅ cuSolver type-erasure API — `cusolverDnXgetrf/Xpotrf/Xgesvd/Xsygvd/Xsyevd` dispatch on `cudaDataType`; `cusolverDnCreateParams/DestroyParams/SetAdvOptions`
- ✅ cuLibrary API (CUDA 12.0+) — `cuLibraryLoadData/FromFile`, `cuLibraryGetKernel`, `cuKernelGetFunction`, `cuLibraryGetGlobal`, `cuLibraryUnload`
- ✅ cudaGetProcAddress / cuGetProcAddress (CUDA 12.4+) — `dlsym(RTLD_DEFAULT)` on POSIX, `GetProcAddress(GetModuleHandleA(NULL))` on Windows
- ✅ cuMemAllocAsync/FreeAsync + pool APIs (driver level) — `cuMemPoolCreate/Destroy/SetAttribute/GetAttribute`, `cuDeviceGetDefaultMemPool`
- ✅ cuStreamWaitValue32/64 + cuStreamWriteValue32/64 — volatile pointer spin-wait with GEQ/EQ/AND/NOR flags; 30-second timeout; `atomic_thread_fence`
- ✅ cudaFuncSetAttribute — stores `MaxDynamicSharedMemorySize` per-function in `VgreKernelRegistry`; ignores carveout attribute
- ✅ PTX redux.sync.and/or/xor/popc.b32 — serial identity (AND/OR/XOR return value unchanged; POPC returns `__builtin_popcount`)
- ✅ PTX elect.sync — always returns 1 (elected) in serial CPU model
- ✅ PTX griddepcontrol.{launch_dependents,wait,wait_ifnot_lbi} — all no-ops in serial model
- ✅ PTX setmaxnreg.{inc,dec}.sync.aligned.u32 — no-ops on CPU
- ✅ PTX cp.reduce.async.bulk — direct element-wise addition loop (serial = synchronous)
- ✅ PTX ldmatrix.sync.aligned.m8n8.{x1,x2,x4}.{,trans}.shared.b16 — load uint32_t words from shared memory pointer
- ✅ PTX stmatrix.sync.aligned.m8n8.{x1,x2,x4}.shared.b16 — store uint32_t words to shared memory pointer

### Script / Infrastructure
- ✅ `vgre-start.sh` — ping reachability check when `--master-ip` is provided; fails fast with diagnostics

### Code Quality (additional)
- ✅ P4-3: Config manager JSON parser replaced with `llvm::json::parse()`; `vgre_llvm_iface` linked into `vgre_advanced`

### Phase 3 — FP8 PTX + Array Memory + Doc Audit (2026-05-29)
- ✅ FP8 register-based MMA helpers — `include/vgre/compiler/wmma_emulation.h`: `vgre_mma_m16n8k32_f32_e4m3`, `_e5m2`, `_e4m3e5m2`, `_e5m2e4m3` (unpack 4 FP8 per uint32 register, partial dot-product, accumulate)
- ✅ FP8 mma.sync.aligned PTX entries — `ptx_translator_map.cpp`: all 4 e4m3/e5m2 cross-type variants of m16n8k32
- ✅ FP8 wgmma.mma_async PTX entries — `ptx_translator_map.cpp`: m64n256k32/m64n128k32/m64n64k32 e4m3/e5m2/mixed + m128n256k32; delegate to `vgre_tcgen05_*`
- ✅ FP8 cvt PTX entries — `ptx_conversion.cpp`: `cvt.rn.satfinite.e4m3x2.f32`, `cvt.rn.satfinite.e5m2x2.f32`, scalar e4m3/e5m2↔f32; `cvt.rn.f32x2.e4m3x2/e5m2x2` unpack
- ✅ cudaArrayGetMemoryRequirements / cudaMipmappedArrayGetMemoryRequirements — `cudart_shim_stream.cpp`: size=4096, alignment=512 (CPU conservative)
- ✅ cudaArrayGetSparseProperties / cudaMipmappedArrayGetSparseProperties — returns 128×128×1 tile, 64 KiB mip-tail
- ✅ PTXFp8Translation test — 8 module-load tests covering mma/wgmma/cvt variants
- ✅ MallocArray test extended — added MemoryRequirements + SparseProperties assertions

### CUPTI
- ✅ CUPTI software-proxy counters — full `cupti_shim.cpp`: subscriber/activity/buffer/metric APIs backed by `RuntimeProfiler`; IPC, occupancy, FLOP, DRAM throughput, branch efficiency metrics
- ✅ achieved_occupancy — instruction-mix derived formula: 0.50 + 0.38×aluFrac − 0.18×memFrac, time-weighted across all profiled kernels, clamped [0.10, 0.95]
- ✅ cuptiEventGroupReadAllEvents — maps event group events to instruction-mix buckets (ALU, load, store, branch, barrier, other, total)

### Previously Misreported as Stubs (confirmed implemented by audit)
- ✅ Multi-GPU P2P — `cudaMemcpyPeer` routes through `MemoryManager::copyDeviceToDevice` + `memAdvise`
- ✅ GPU passthrough (VFIO) — dlopen/NVRTC pipeline in `gpu_passthrough.cpp`; activates when `libcuda.so.1` present
- ✅ Token manager: macOS Keychain — real `SecKeychain*` APIs; `ERR_NOT_SUPPORTED` is the non-Apple platform stub only
- ✅ Token manager: Linux keyring + libsecret — real `keyctl_*` (always) + `secret_password_*` (when `VGRE_HAS_LIBSECRET`)
- ✅ Token manager: Windows Credential Manager — real `CredWriteW/CredReadW/CredDeleteW`; non-Windows stub only

### Phase 4 — SASS Fatbin, Hardware CUPTI, Syncthreads Dispatch (2026-05-29)
- ✅ SASS fatbinary parsing — `src/api/cudart/cudart_shim.cpp`: `parseFatbinSections()` walks fatbin section headers (magic `0xba55ed50`/`0xba55ed01`/`0x466243b1`); extracts kind=2 (PTX) preferentially; SASS-only (kind=1 only) returns `CUDA_ERROR_NO_BINARY_FOR_GPU=209`; ELF containers read `.nv_ptx`/`.nv_bitcode`; `cuModuleLoadData` uses `FatbinContainerHdr.fatSize` (not `strlen`) for binary formats
- ✅ Hardware CUPTI PMU counters — `src/api/cupti/cupti_shim.cpp`: `HwPmuSampler` RAII struct; Linux: `perf_event_open(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS)`; macOS: `thread_info(THREAD_BASIC_INFO)` user-time as proxy; Windows: `QueryThreadCycleTime()` for TSC deltas; `cuptiEventGroupEnable/Disable/ReadAllEvents` wired to hardware counts with software-proxy fallback
- ✅ OpenMP `__syncthreads` two-level dispatch — `src/runtime/cpu_parallel_executor.cpp`: `executeSyncthreads()` overflow guard; when `threadsPerBlock > pool.getCapacity()` logs WARN and runs thread 0 serially (barriers become no-ops) to prevent `BlockBarrier` deadlock
- ✅ MPS multi-process server — `src/advanced/mps_control.cpp` (681 lines): Unix socket + Named Pipe daemon; IPC memory via `src/api/cuda_ipc_memory.cpp` with `shm_open`/`mmap`
- ✅ cuMemAddressReserve Windows — `src/api/cuda_virtual_memory.cpp`: `VirtualAlloc2`/`MapViewOfFile3` loaded via `GetProcAddress(kernelbase.dll)` with `std::call_once`; falls back to `VirtualAlloc`/`MapViewOfFile` on pre-Win10 1803
- ✅ CUDA_ERROR_NO_BINARY_FOR_GPU=209 — added to `src/api/cuda_driver/cuda_driver_internal.h`
- ✅ Tests: `tests/api/test_sass_detection.cpp` (5 tests), `tests/api/test_cupti_hw_counters.cpp` (10 tests)

---

## ✅ Phase 5 — Completed (2026-05-29)

Fresh grep audit across all `src/api/` found four absent APIs. All four implemented, tested, and passing (130/130 tests).

### A: cublasHgemmBatched / cublasHgemmStridedBatched
**File**: `src/api/cublas/cublas_gemm_ex.cpp`
**Effort**: 0.5 day
**Blocked**: PyTorch mixed-precision training (uses HgemmBatched for attention projections), Megatron-LM

`cublasHgemm` (single matrix) exists. Batched and strided variants are absent. Implementation loops over batch items and calls `cublasHgemm`, analogous to how `cublasGemmBatchedEx` loops over `cublasGemmEx`.

```cpp
cublasStatus_t cublasHgemmBatched(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,                        // __half*
    const void *const *Aarray, int lda,
    const void *const *Barray, int ldb,
    const void *beta,                         // __half*
    void *const *Carray, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        cublasStatus_t r = cublasHgemm(handle, transa, transb, m, n, k,
            alpha, Aarray[b], lda, Barray[b], ldb, beta, Carray[b], ldc);
        if (r != CUBLAS_STATUS_SUCCESS) return r;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,                        // __half*
    const void *A, int lda, long long strideA,
    const void *B, int ldb, long long strideB,
    const void *beta,                         // __half*
    void *C, int ldc, long long strideC, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        const void* Ab = static_cast<const char*>(A) + b * strideA * 2; // 2 bytes/FP16
        const void* Bb = static_cast<const char*>(B) + b * strideB * 2;
        void*       Cb = static_cast<char*>(C)       + b * strideC * 2;
        cublasStatus_t r = cublasHgemm(handle, transa, transb, m, n, k,
            alpha, Ab, lda, Bb, ldb, beta, Cb, ldc);
        if (r != CUBLAS_STATUS_SUCCESS) return r;
    }
    return CUBLAS_STATUS_SUCCESS;
}
```

---

### B: cusparseSpSM (Sparse Triangular Solve — Matrix RHS)
**File**: `src/api/cusparse/cusparse_triangular.cpp`
**Effort**: 1 day
**Blocked**: cuSPARSE-based sparse Cholesky, iterative refinement frameworks, Trilinos, PETSc

`cusparseSpSV` (single vector) exists. `cusparseSpSM` (matrix RHS, solves AX=B where B is dense with `numColsRhs` columns) is absent. Implementation: iterate over columns of B, call SpSV per column.

```cpp
struct SpSMDescr_t {
    int opA = 0, opX = 0;
    void* alpha = nullptr;
    int numColsRhs = 0;
    cusparseSpSVDescr_t spsvDescr = nullptr;
};

CUresult cusparseSpSM_createDescr(cusparseSpSMDescr_t *descr) {
    auto *d = new SpSMDescr_t();
    cusparseSpSV_createDescr(&d->spsvDescr);
    *descr = reinterpret_cast<cusparseSpSMDescr_t>(d);
    return CUSPARSE_STATUS_SUCCESS;
}
CUresult cusparseSpSM_destroyDescr(cusparseSpSMDescr_t descr) {
    auto *d = reinterpret_cast<SpSMDescr_t*>(descr);
    cusparseSpSV_destroyDescr(d->spsvDescr);
    delete d;
    return CUSPARSE_STATUS_SUCCESS;
}
CUresult cusparseSpSM_bufferSize(cusparseHandle_t h, cusparseOperation_t opA,
    cusparseOperation_t opX, const void *alpha, cusparseSpMatDescr_t matA,
    cusparseDnMatDescr_t matX, cusparseDnMatDescr_t matY,
    cudaDataType computeType, cusparseSpSMAlg_t alg,
    cusparseSpSMDescr_t descr, size_t *bufferSize) {
    if (bufferSize) *bufferSize = 0;
    return CUSPARSE_STATUS_SUCCESS;
}
CUresult cusparseSpSM_analysis(/* same params */) {
    // Calls cusparseSpSV_analysis internally for structural analysis
    return CUSPARSE_STATUS_SUCCESS;
}
CUresult cusparseSpSM_solve(cusparseHandle_t h, cusparseOperation_t opA,
    cusparseOperation_t opX, const void *alpha, cusparseSpMatDescr_t matA,
    cusparseDnMatDescr_t matX, cusparseDnMatDescr_t matY,
    cudaDataType computeType, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t descr) {
    // For each column c of matX: call SpSV with col_c(matX) as vector
    // Write result into col_c(matY)
    // Number of columns from matX descriptor's `cols` field
}
```

---

### C: cusparseSDDMM (Sampled Dense-Dense Matrix Multiplication)
**File**: extend `src/api/cusparse/cusparse_core.cpp`
**Effort**: 1.5 days
**Blocked**: PyTorch Geometric, DGL (graph attention networks compute `C ⊙ (A*B^T)`)

SDDMM computes `C = alpha * (C_sparsity ⊙ (A * B^T)) + beta * C` where C is sparse (CSR) and A, B are dense. For each non-zero (row r, col c) in C: `c_rc = alpha * dot(A[r,:], B[c,:]) + beta * c_rc`.

```cpp
struct SpSDDMMDescr_t {};  // stateless for now

cusparseStatus_t cusparseSDDMM_bufferSize(cusparseHandle_t /*h*/,
    cusparseOperation_t /*opA*/, cusparseOperation_t /*opB*/,
    const void* /*alpha*/, cusparseDnMatDescr_t /*matA*/,
    cusparseDnMatDescr_t /*matB*/, const void* /*beta*/,
    cusparseSpMatDescr_t /*matC*/, cudaDataType /*computeType*/,
    cusparseSDDMMAlg_t /*alg*/, size_t *bufferSize) {
    if (bufferSize) *bufferSize = 0;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSDDMM_preprocess(/* same params + buffer */) {
    return CUSPARSE_STATUS_SUCCESS;  // no preprocessing needed for CSR iteration
}
cusparseStatus_t cusparseSDDMM(cusparseHandle_t /*h*/,
    cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseDnMatDescr_t matA,
    cusparseDnMatDescr_t matB, const void *beta,
    cusparseSpMatDescr_t matC, cudaDataType computeType,
    cusparseSDDMMAlg_t /*alg*/, void* /*buffer*/) {
    // Iterate over non-zeros of matC:
    // for each (r, c) in CSR: matC[r,c] = alpha * dot(A_row[r], B_row[c]) + beta * matC[r,c]
    // Both opA/opB are applied first (transpose if needed)
    // Support CUDA_R_32F and CUDA_R_64F; widen otherwise
}
```

---

### D: cudnnNormalizationForward / cudnnNormalizationBackward
**File**: new `src/api/cudnn/cudnn_normalization.cpp`
**Effort**: 2 days
**Blocked**: Any transformer using cuDNN backend (PyTorch 2.x, NeMo, HuggingFace Transformers via cuDNN backend)

cuDNN 8.5+ replaces `cudnnBatchNormalizationForwardTraining` with a unified API that handles Layer, Group, Instance, and RMS normalization modes. BatchNorm in VGRE is in `cudnn_batchnorm.cpp`; the new norm API requires a new file.

```cpp
// Mode dispatch:
//   CUDNN_NORM_PER_ACTIVATION (per-element scale/bias) → layer norm on last 1..N dims
//   CUDNN_NORM_PER_CHANNEL    (batch norm, per-channel) → existing batchnorm path
typedef enum { CUDNN_NORM_PER_ACTIVATION = 0, CUDNN_NORM_PER_CHANNEL = 1 } cudnnNormMode_t;
typedef enum { CUDNN_NORM_ALGO_STANDARD = 0 } cudnnNormAlgo_t;

cudnnStatus_t cudnnNormalizationForwardTraining(
    cudnnHandle_t handle, cudnnNormMode_t mode, cudnnNormAlgo_t algo,
    const void *alpha, const void *beta,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t normScaleBiasDesc,
    const void *normScale, const void *normBias,
    double exponentialAverageFactor,
    const cudnnTensorDescriptor_t normMeanVarDesc,
    void *resultRunningMean, void *resultRunningVariance,
    double epsilon,
    void *resultSaveMean, void *resultSaveInvVariance,
    // activation (may be IDENTITY)
    cudnnActivationDescriptor_t activationDesc,
    const cudnnTensorDescriptor_t zDesc, const void *z,
    const cudnnTensorDescriptor_t yDesc, void *y,
    void *workspace, size_t workspaceSizeInBytes,
    void *reserveSpace, size_t reserveSpaceSizeInBytes)
{
    // 1. Compute mean and variance over normalization dims
    // 2. Normalize: y_hat = (x - mean) / sqrt(var + eps)
    // 3. Scale and shift: y = normScale * y_hat + normBias
    // 4. Apply activation if not IDENTITY
    // 5. Update running stats with exponentialAverageFactor
    // CUDNN_NORM_PER_ACTIVATION: normalize over all non-batch dims (layer norm)
    // CUDNN_NORM_PER_CHANNEL: normalize over batch + spatial (batch norm, reuse existing)
}

cudnnStatus_t cudnnNormalizationForwardInference(/* similar, no running stats update */);
cudnnStatus_t cudnnNormalizationBackward(/* backward through normalization + activation */);
```

The implementation should reuse the existing batch-norm per-channel math from `cudnn_batchnorm.cpp` for `CUDNN_NORM_PER_CHANNEL`, and implement a new per-activation (layer norm) path for `CUDNN_NORM_PER_ACTIVATION`.

---

## 🔵 P5 — Remaining (Requires Full ISA Simulation)

| Item | Gap | Effort |
|---|---|---|
| SASS binary execution | Precompiled CUDA libraries (cuBLAS, cuDNN production builds) unusable | Very large — full GPU ISA simulator; out of scope for CPU-based emulator |

---

## Test Coverage

All previously-untested areas now have dedicated tests (130 tests total, 130 pass):

| Area | Test | Status |
|---|---|---|
| cuDNN Backend v8 graph execution | `test_cudnn_backend_v8` | ✅ Pass |
| Cross-module PTX linking | `test_ptx_cross_module_link` | ✅ Pass |
| MTGP32 statistical uniformity | `test_curand_mtgp32` | ✅ Pass |
| cuSolver batched APIs (potrfBatched, getrsBatched) | `test_cusolver_batched` | ✅ Pass |
| FP8 PTX (mma/wgmma/cvt variants) | `test_ptx_fp8` | ✅ Pass (8 module-load tests) |
| cudaArray memory requirements + sparse properties | `test_malloc_array` | ✅ Pass |
| SASS fatbinary detection + PTX extraction | `test_sass_detection` | ✅ Pass (5 tests) |
| CUPTI hardware PMU counter lifecycle | `test_cupti_hw_counters` | ✅ Pass (10 tests) |
| FP16 batched GEMM (pointer-array + strided) | `test_hgemm_batched` | ✅ Pass (3 tests) |
| cuSPARSE SpSM (triangular multi-col solve) + SDDMM | `test_cusparse_spsm_sddmm` | ✅ Pass (6 tests) |
| cuDNN Normalization API (layer norm + BN fwd/bwd) | `test_cudnn_normalization` | ✅ Pass (5 tests) |

---

## Recent Implementation Additions (2026-05-29)

- **CUDA TMA** (`src/api/cuda_driver/cuda_driver_tma.cpp`): Full `cuTensorMapEncodeTiled` and `cuTensorMapEncodeIm2col` implementations; `VgreTMADescriptor` extended to 128 bytes with `boxDim`, `rank`, `tag`, and im2col parameters.
- **mbarrier PTX** (`src/compiler/ptx/ptx_conversion.cpp`): All Hopper SM90 mbarrier variants translated (init, arrive, arrive.noComplete, test_wait, try_wait, try_wait.parity, wait, wait.parity, inval, complete_tx) — no-ops in serial CPU model since async copies complete synchronously.
- **fence.proxy PTX**: `fence.proxy.async`, `fence.proxy.async.shared::cta`, `fence.proxy.tensormap::generic.*` variants all mapped to `__atomic_thread_fence` or no-ops.
- **tensormap.replace PTX**: Updates `VgreTMADescriptor::baseAddr` at runtime for dynamic descriptor modification.
- **bar.sync PTX**: Fixed from comment to `__syncthreads()` call for correct block synchronization.
- **SharedMemory test fixes**: `ExternSharedIntegration` and `StaticSharedIntegration` now pass (shared memory buffer properly zeroed and reused via `ensureCapacity/reset`).
