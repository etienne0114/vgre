# VGRE Implementation Plan

**Version**: 2.0.0  
**Date**: 2026-05-15  
**Status**: Living document — update before any feature PR  
**Prerequisite**: Read `missingFeatures.md` for the exhaustive gap list.

---

## 1. Organization Principles

### 1.1 File Size Limits

| Tier | Max Lines | Rationale |
|---|---|---|
| Shim files | 800 lines | Each API category gets its own file; monolithic shims become unmaintainable |
| Core logic | 600 lines | Split by concern (e.g., `graph_manager_nodes.cpp`, `graph_manager_exec.cpp`) |
| Translator | 500 lines per backend | Ampere/Hopper/Blackwell PTX backends are separate files |
| Headers | 400 lines | Forward declarations in `*_fwd.h`; full defs in `*.h` |

### 1.2 Directory Conventions

```
src/api/cudart/          # CUDART shim split by concern (15 files)
src/api/cuda_driver/     # Driver API shim split by concern (11 files)
src/api/cublas/          # cuBLAS backend split by level (7 files)
src/api/cudnn/           # cuDNN backend split by layer type (12 files)
src/api/nccl/            # NCCL backend split by concern (4 files)
src/api/cufft/           # cuFFT shim (1 file — reference DFT/IDFT)
src/api/curand/          # cuRAND shim (1 file)
src/api/cusolver/        # cuSOLVER shim (1 file — LAPACK delegation)
src/api/cusparse/        # cuSPARSE shim (4 .cpp files + cusparse_state.h internal header)
src/api/cublasLt/        # cuBLASLt shim (1 file)
src/compiler/ptx/        # PTX translator split by architecture / op family (4 files)
src/core/graph/          # Graph manager split by concern (7 files)
src/core/texture/        # Texture/surface manager split by concern (5 files)
src/core/memory/         # Memory manager split by concern (6 files)
src/runtime/             # Runtime engine split (8 files incl. vector_engine float/double)
src/advanced/            # Advanced subsystems split (20 files incl. adaptive/hybrid splits)
src/advanced/profiling/  # NOT YET SPLIT — all profiling currently in runtime_profiler.cpp
src/deployment/k8s_device_plugin/  # Go gRPC K8s plugin (4 files)
src/deployment/slurm_gres/         # C SLURM GRES plugin (2 files)
```

### 1.3 Naming Convention

| Pattern | Example | Purpose |
|---|---|---|
| `*_shim.cpp` | `cudart_shim_stream.cpp` | API interception / forwarding |
| `*_backend.cpp` | `cudnn_convolution_backend.cpp` | Actual compute implementation |
| `*_nodes.cpp` | `graph_manager_extended_nodes.cpp` | Graph node-type implementations |
| `*_ops.cpp` | `cudnn_tensor_ops.cpp` | Tensor elementwise / reduction ops |
| `*_ampere.cpp` | `ptx_mma_ampere.cpp` | Architecture-specific PTX backend |
| `*_linux.cpp` | `token_manager_linux.cpp` | Platform-specific implementation |

---

## 2. Phase Roadmap

| Phase | Focus | Status | Risk |
|---|---|---|---|
| **P1** | Critical CUDA Runtime gaps (stream sync, symbols, graphs kernel node) | **DONE** | Low |
| **P2** | cuBLAS Level-2/Level-3 backfill | **DONE** | Low |
| **P3** | cuDNN backward passes + training | **DONE** | Low |
| **P4** | Missing libraries (stub → functional) | **DONE** | Low |
| **P5** | CUDA Driver API expansion | **DONE** | Low |
| **P6** | PTX ISA expansion + texture/surface PTX | **DONE** | Medium |
| **P7** | NCCL p2p + advanced collectives | **DONE** (single-node; multi-node TCP functional) | Low |
| **P8** | Cooperative groups + device-side libraries | **DONE** | Low |
| **P9** | Deployment (K8s Device Plugin, SLURM GRES) | **DONE** | Low |
| **P10** | CDP, profiling, advanced formats, graph updates | **PARTIAL** — CDP + graph update v2 done. Profiling consolidated in `runtime_profiler.cpp` | Low |

---

## 3. Detailed Implementation Status

---

### 3.1 Phase 1 — Critical CUDA Runtime Gaps

**Status**: **DONE** (2026-05-13 – 2026-05-15)

**Implemented**: `cudaStreamWaitEvent`, `cudaEventQuery`, `cudaStreamAddCallback`, `cudaLaunchHostFunc`, `cudaGetErrorName`, `cudaGetErrorString`, `cudaMemcpyToSymbol`, `cudaMemcpyToSymbolAsync`, `cudaMemcpyFromSymbol`, `cudaMemcpyFromSymbolAsync`, `cudaMallocArray`, `cudaMalloc3DArray`, `cudaMalloc3D`, `cudaMemcpy3DAsync`, `cudaHostGetDevicePointer`, `cudaHostGetFlags`, `cudaArrayGetInfo`, `cudaArrayDestroy`, `cudaPointerGetAttributes`, `cudaMemset2D`, `cudaMemset3D`, `cudaMemset2DAsync`, `cudaMemset3DAsync`, `cudaGraphAddKernelNode`, `cudaGraphAddMemsetNode`, `cudaGraphAddHostNode`, `cudaGraphAddChildGraphNode`, `cudaGraphAddEmptyNode`, `cudaGraphAddEventRecordNode`, `cudaGraphAddEventWaitNode`, `cudaGraphAddMemAllocNode`, `cudaGraphAddMemFreeNode`, `cudaGraphKernelNodeSetParams`, `cudaGraphKernelNodeGetParams`, `cudaGraphMemcpyNodeGetParams`, `cudaGraphMemsetNodeGetParams`, `cudaGraphMemsetNodeSetParams`, `cudaGraphHostNodeGetParams`, `cudaGraphHostNodeSetParams`, `cudaGraphNodeGetType`, `cudaGraphGetNodes`, `cudaGraphGetRootNodes`, `cudaGraphGetEdges`, `cudaGraphNodeGetDependencies`, `cudaGraphNodeGetDependentNodes`, `cudaGraphExecKernelNodeSetParams`, `cudaGraphExecMemcpyNodeSetParams`, `cudaGraphExecMemsetNodeSetParams`, `cudaGraphExecHostNodeSetParams`, `cudaGraphExecChildGraphNodeSetParams`, `cudaGraphExecEventRecordNodeSetEvent`, `cudaGraphExecEventWaitNodeSetEvent`, `cudaGraphInstantiateWithFlags`, `cudaGraphInstantiateWithParams`, `cudaGraphExecGetFlags`, `cudaGraphUpload`, `cudaGraphNodeSetEnabled`, `cudaGraphNodeGetEnabled`, `cudaGraphExecNodeSetParams`, `cudaGraphKernelNodeCopyAttributes`, `cudaStreamIsCapturing`, `cudaStreamGetCaptureInfo`, `cudaStreamGetCaptureInfo_v2`, `cudaThreadExchangeStreamCaptureMode`, `cudaStreamUpdateCaptureDependencies`, `cudaStreamCopyAttributes`, `cudaCreateTextureObject`, `cudaDestroyTextureObject`, `cudaGetTextureObjectResourceDesc`, `cudaGetTextureObjectTextureDesc`, `cudaGetTextureObjectResourceViewDesc`, `cudaCreateSurfaceObject`, `cudaDestroySurfaceObject`, `cudaGetSurfaceObjectResourceDesc`, `cudaGetTextureReference`, `cudaGetSurfaceReference`, `cudaBindTexture`, `cudaUnbindTexture`, `cudaBindTextureToArray`, `cudaBindTexture2D`, `cudaBindSurfaceToArray`, `cudaFuncGetAttributes`, `cudaFuncSetCacheConfig`, `cudaFuncSetSharedMemConfig`, `cudaDeviceGetLimit`, `cudaDeviceSetLimit`, `cudaDeviceGetCacheConfig`, `cudaDeviceSetCacheConfig`, `cudaDeviceGetSharedMemConfig`, `cudaDeviceSetSharedMemConfig`, `cudaChooseDevice`, `cudaThreadExit`, `cudaThreadSynchronize`, `cudaThreadSetLimit`, `cudaThreadGetLimit`, `cudaDeviceGetP2PAttribute`, `cudaDeviceFlushGPUDirectRDMAWrites`, `cudaDeviceGetGraphMemAttribute`, `cudaDeviceSetGraphMemAttribute`, `cudaLaunchKernelExC`, `cudaLaunchConfig`, `cudaGraphAddDependencies`, `cudaGraphRemoveDependencies`, `cudaGraphRetainUserObject`, `cudaGraphReleaseUserObject`, `cudaUserObjectCreate`, `cudaUserObjectRetain`, `cudaUserObjectRelease`, `cudaGraphNodeFindInClone`, `cudaGraphDebugDotPrint`, `cudaImportExternalMemory`, `cudaDestroyExternalMemory`, `cudaExternalMemoryGetMappedBuffer`, `cudaExternalMemoryGetMappedMipmappedArray`, `cudaExternalSemaphoreGetSignalNodeParams`, `cudaExternalSemaphoreGetWaitNodeParams`.

**Files**:
- `src/api/cudart/cudart_shim_stream.cpp` — stream sync, event query, callbacks, host func
- `src/api/cudart/cudart_shim.cpp` — error strings, symbols, arrays, pointer attributes
- `src/api/cudart/cudart_shim_memset_nd.cpp` — 2D/3D memset variants
- `src/api/cudart/cudart_shim_graph_nodes.cpp` — CUDART shim wrappers for all node types
- `src/core/graph/graph_manager_introspection.cpp` — get nodes, edges, dependencies, params
- `src/core/graph/graph_manager_exec_update.cpp` — exec mutation, instantiate with flags/params
- `src/core/graph/graph_manager_dependencies.cpp` — dependency add/remove
- `src/api/cudart/cudart_shim_graph_user_objects.cpp` — user object CUDART shim wrappers
- `src/api/cudart/cudart_shim_capture.cpp` — stream capture introspection
- `src/api/cudart/cudart_shim_texture_objects.cpp` — CUDART texture/surface object APIs
- `src/api/cudart/cudart_shim_device_attrs.cpp` — device/function attributes
- `src/api/cudart/cudart_shim_external_memory.cpp` — external memory/semaphore

**Note**: `cudaGraphExecUpdate_v2` is implemented in `src/api/cuda_interceptor_graphs.cpp` (`CUDAInterceptor::graphExecUpdateV2`), not a separate `graph_manager_exec_update_v2.cpp` file.

---

### 3.2 Phase 2 — cuBLAS Level-2 / Level-3 Backfill

**Status**: **DONE** (2026-05-13 – 2026-05-15)

**Implemented Level-1**: `cublasScopy`, `cublasDcopy`, `cublasSswap`, `cublasDswap`, `cublasSrot`, `cublasDrot`, `cublasSrotm`, `cublasSrotmg`, `cublasSrotg`, `cublasSasum`, `cublasDasum`, `cublasIsamax`, `cublasIdamax`, `cublasIsamin`, `cublasIdamin`, `cublasDnrm2`, `cublasDscal`, `cublasGetPointerMode`, `cublasSetPointerMode`, `cublasGetAtomicsMode`, `cublasSetAtomicsMode`

**Implemented Level-2**: `cublasStrsv`, `cublasDtrsv`, `cublasSger`, `cublasDger`, `cublasSsymv`, `cublasDsymv`, `cublasSgbmv`, `cublasDgbmv`, `cublasSsyr`, `cublasDsyr`, `cublasSsyr2`, `cublasDsyr2`, `cublasStrmv`, `cublasDtrmv`, `cublasStbsv`, `cublasStpsv`, `cublasSspmv`, `cublasSsbmv`, `cublasSspr`, `cublasSspr2`, `cublasStbmv`, `cublasStpmv`

**Implemented Level-3**: `cublasSgemm`/`cublasDgemm`, `cublasStrsm`/`cublasDtrsm`, `cublasSsyrk`/`cublasDsyrk`, `cublasSsyr2k`/`cublasDsyr2k`, `cublasStrmm`/`cublasDtrmm`, `cublasSsymm`/`cublasDsymm`, `cublasHgemm`, `cublasGemmEx`, `cublasGemmBatchedEx`, `cublasGemmStridedBatchedEx`, batched SGEMM/DGEMM, strided batched variants

**Implemented Hermitian**: `cublasCherk`, `cublasZherk`, `cublasCher2k`, `cublasZher2k`, `cublasChemm`, `cublasZhemm`

**Files**:
- `src/api/cublas/cublas_level1.cpp` — all Level-1 routines
- `src/api/cublas/cublas_level2.cpp` — Level-2 standard routines (SGEMV, TRSV, GER, SYMV, GBMV, SYR2, TRMV, SYR2K)
- `src/api/cublas/cublas_level2_packed.cpp` — Level-2 packed/banded routines (TBSV, TPSV, SPMV, SBMV, SPR, SPR2, TBMV, TPMV)
- `src/api/cublas/cublas_level3.cpp` — Level-3 GEMM + TRSM + SYRK + TRMM + SYMM + batched GEMM + HGEMM + GemmEx + logging
- `src/api/cublas/cublas_level3_blas3.cpp` — Level-3 batched BLAS3 (batched TRSM, SYRK, SYR2K, TRMM, SYMM)
- `src/api/cublas/cublas_hermitian.cpp` — complex Hermitian routines
- `src/api/cublas/cublas_core.cpp` — pointer mode, atomics mode, logger, SYR

**Implemented Complex (C/Z)**: `cublasCgemm`/`cublasZgemm`, `cublasCgemv`/`cublasZgemv`, `cublasCaxpy`/`cublasZaxpy`, `cublasCdotc`/`cublasZdotc`, `cublasCdotu`/`cublasZdotu`, `cublasCscal`/`cublasZscal`/`cublasCsscal`/`cublasZdscal`, `cublasCcopy`/`cublasZcopy`, `cublasCswap`/`cublasZswap`, `cublasScnrm2`/`cublasDznrm2`, `cublasIcamax`/`cublasIzamax`, `cublasScasum`/`cublasDzasum`, `cublasCrot`/`cublasZrot`, `cublasCtrsv`/`cublasZtrsv`, `cublasCgeru`/`cublasZgeru`, `cublasCgerc`/`cublasZgerc`, `cublasCsyrk`/`cublasZsyrk`, `cublasCsyr2k`/`cublasZsyr2k`, `cublasCtrsm`/`cublasZtrsm`, `cublasCsymm`/`cublasZsymm`, `cublasCtrmm`/`cublasZtrmm`. All in `src/api/cublas/cublas_complex.cpp`. 21 tests pass.

---

### 3.3 Phase 3 — cuDNN Backward Passes + Training

**Status**: **DONE** (2026-05-13 – 2026-05-15)

**Implemented**: `cudnnConvolutionBackwardData`, `cudnnConvolutionBackwardFilter`, `cudnnConvolutionBackwardBias`, `cudnnBatchNormalizationForwardTraining`, `cudnnBatchNormalizationBackward`, `cudnnActivationBackward`, `cudnnSoftmaxBackward`, `cudnnPoolingBackward`, `cudnnDropoutForward`/`Backward`, `cudnnRNNForwardInference`/`Training`/`BackwardData`/`BackwardWeights`, `cudnnMultiHeadAttnForward`/`BackwardData`/`BackwardWeights`, `cudnnTransformTensor`, `cudnnOpTensor`, `cudnnReduceTensor`, `cudnnCTCLoss`, `cudnnLRNCrossChannelForward`/`Backward`, `cudnnDivisiveNormalizationForward`/`Backward`, `cudnnAddTensor`

**Files**:
- `src/api/cudnn/cudnn_convolution.cpp`
- `src/api/cudnn/cudnn_batchnorm.cpp`
- `src/api/cudnn/cudnn_activation.cpp`
- `src/api/cudnn/cudnn_softmax.cpp`
- `src/api/cudnn/cudnn_pooling.cpp`
- `src/api/cudnn/cudnn_dropout.cpp`
- `src/api/cudnn/cudnn_rnn.cpp`
- `src/api/cudnn/cudnn_attention.cpp`
- `src/api/cudnn/cudnn_tensor_ops.cpp`
- `src/api/cudnn/cudnn_lrn.cpp`
- `src/api/cudnn/cudnn_divisive_norm.cpp`
- `src/api/cudnn/cudnn_ctc_loss.cpp`

**INT8x4 / INT8x32 packed layouts**: Supported in `cudnnConvolutionForward`, `cudnnPoolingForward/Backward`, `cudnnActivationForward/Backward` via dequantize→FP32 compute→requantize path. Implemented inside existing files; **no separate `cudnn_int8_packed.cpp` file**.

---

### 3.4 Phase 4 — Missing Libraries (Stub → Functional)

**Status**: **DONE** (2026-05-14 – 2026-05-15)

#### 3.4.1 cuFFT

**File**: `src/api/cufft/cufft_core.cpp` (579 lines, single file)

**Implemented**: `cufftPlan1d/2d/3d/Many`, `cufftDestroy`, `cufftExecC2C/Z2Z/R2C/C2R/D2Z/Z2D`, `cufftSetStream/WorkArea`, `cufftEstimate*`, `cufftGetSize*`, `cufftMakePlanMany`, IPC plan export/import

**Implementation**: O(n log n) Cooley-Tukey radix-2 FFT for power-of-2 sizes + Bluestein's algorithm for arbitrary sizes. OpenMP-parallelized butterfly stages and batch/row/column loops. Optional FFTW3 delegation (`VGRE_HAS_FFTW3`, auto-detected by CMake `pkg_check_modules`). Plans stored in global handle map for reuse.

**Test**: `tests/api/test_cufft.cpp` — 11 tests (pow2 roundtrip, non-pow2, single-frequency, Z2Z double, R2C+C2R, D2Z+Z2D, batched, 2D, prime-size, Parseval, plan management). All pass.

---

#### 3.4.2 cuRAND

**File**: `src/api/curand/curand_core.cpp` (single file, not split)

**Implemented**: `curandCreateGenerator/DestroyGenerator/SetPseudoRandomGeneratorSeed/SetGeneratorOffset/SetGeneratorOrdering/GetVersion`, `curandGenerate/GenerateUniform/GenerateNormal/GenerateLogNormal/GeneratePoisson` for XORWOW, MRG32k3a, MTGP32, MT19937

**Implementation**: Host API uses `std::mt19937_64` seeded from `/dev/urandom` / `BCryptGenRandom` / `getentropy()`. Device API returns `CURAND_STATUS_NOT_SUPPORTED`.

**Thread safety**: Per-handle `std::mutex genMutex` inside `GeneratorState`; registry uses `unique_ptr<GeneratorState>` for stable addresses. All `curandGenerate*` calls lock `genMutex` before accessing the engine — concurrent calls on different handles are fully parallel. Quasi-random Sobol direction-vector tables are guarded by a separate `g_sobolMutex`. Device-side `curand_*` returns `CURAND_STATUS_NOT_SUPPORTED` by design.

---

#### 3.4.3 cuSOLVER

**File**: `src/api/cusolver/cusolver_core.cpp` (single file, not split)

**Implemented**: `cusolverDnCreate/Destroy`, `cusolverDnSpotrf/Dpotrf/Sgeqrf/Dgeqrf/Sgesvd/Dgesvd/Ssyevd/Dsyevd` (Cholesky, QR, SVD, eigenvalues), `cusolverDnSgetrf/Dgetrf/Sgetrs/Dgetrs` (LU factorization + triangular solve), `cusolverDnSormqr/Dormqr` (apply Q from QR), `cusolverDnSgelsd/Dgelsd` (least-squares driver)

**Implementation**: Delegates to system LAPACK with proper workspace queries.

**Dense API complete**: potrf, geqrf, gesvd, syevd, getrf, getrs, ormqr, gelsd — all via LAPACK delegation with proper `lwork=-1` workspace queries.
**Still missing**: `cusolverSp*` sparse solver API (requires UMFPACK/CHOLMOD). See `missingFeatures.md` §3.6.

---

#### 3.4.4 cuSPARSE

**Files**: `src/api/cusparse/cusparse_core.cpp` + `cusparse_format.cpp` + `cusparse_triangular.cpp` (split at 2026-05-16 when core reached 1017 lines)

**Shared state**: `src/api/cusparse/cusparse_state.h` (internal, not public)

**Implemented**:
- Core: `cusparseCreate/Destroy`, `cusparseSpMV/SpMM` (CSR + COO + CSC, S/D/C/Z/FP16/BF16/INT8), `cusparseAxpyi`, buffer-size queries
- Format: `cusparseCreateCsc`, `cusparseSparseToDense`, `cusparseDenseToSparse` (bufferSize+analysis+compress), `cusparseSpMatGetAttribute/SetAttribute`, `cusparseSpMatGetSize`, `cusparseCsrSetPointers`
- Triangular: `cusparseSpSV` (full forward/backward/transposed substitution for lower+upper triangular CSR)

**Also implemented** (2026-05-16): `cusparseSpGEMM` (3-phase, `cusparse_factorization.cpp`), `cusparseScsrilu02`/`Dcsrilu02` (ILU0), `cusparseScsric02`/`Dcsric02` (IC0). Bug-fixed 2026-05-16: SpGEMM first-pass used corrupt dual-purpose sentinel; replaced with dedicated `inUse[]` bool array.
**Still missing**: Full-fill sparse factorization (UMFPACK-style). `cusolverSp*` separately tracked.

---

#### 3.4.5 cuBLASLt

**File**: `src/api/cublaslt/cublaslt_core.cpp` (single file, 837 lines)

**Implemented**: `cublasLtCreate/Destroy`, `cublasLtMatmulDescCreate/Destroy/SetAttribute/GetAttribute`, `cublasLtMatrixLayoutCreate/Destroy/SetAttribute/GetAttribute`, `cublasLtMatmulPreferenceCreate/Destroy/SetAttribute/GetAttribute`, `cublasLtMatmulAlgoGetHeuristic` (singular + plural), `cublasLtMatmul` with all epilogues: ReLU, GELU, Bias, ReLU+Bias, GELU+Bias, DRELU, DGELU, DRELU_BGRAD, DGELU_BGRAD, BGRADA/BGRADB. Scale pointer attributes (A/B/C/D scale, amaxD, epilogueAuxPtr, biasDataType). All data types: F32, F64, F16, BF16, INT8→INT32.

**Algorithm caching** (2026-05-16): 128-entry LRU cache keyed on (M, N, K, dtypeA, dtypeB, dtypeC, epilogue, transA, transB). `cublasLtMatmulAlgoGetHeuristic` populates cache and returns in O(1) for repeated problem shapes. Cache eviction is LRU via `std::list` + `std::unordered_map`. No remaining algorithmic gaps.

---

### 3.5 Phase 5 — CUDA Driver API Expansion

**Status**: **DONE** (2026-05-14)

**Implemented**: `cuMemAllocManaged`, `cuMemHostAlloc`, `cuMemHostGetDevicePointer`, `cuMemHostRegister`, `cuMemHostUnregister`, `cuMemAllocPitch`, `cuMemcpy2D`, `cuMemcpy2DAsync`, `cuMemcpy3D`, `cuMemcpy3DAsync`, `cuMemcpyDtoDAsync`, `cuMemcpyDtoHAsync`, `cuMemcpyHtoDAsync`, `cuMemcpyAsync`, `cuMemcpyPeer`, `cuMemcpyPeerAsync`, `cuMemsetD8`, `cuMemsetD16`, `cuMemsetD32`, `cuMemsetD2D8`, `cuMemsetD2D16`, `cuMemsetD2D32`, `cuStreamAddCallback`, `cuStreamQuery`, `cuStreamGetFlags`, `cuStreamGetPriority`, `cuStreamGetId`, `cuStreamGetCtx`, `cuStreamIsCapturing`, `cuStreamGetCaptureInfo`, `cuStreamUpdateCaptureDependencies`, `cuEventQuery`, `cuCtxGetDevice`, `cuCtxGetFlags`, `cuCtxGetLimit`, `cuCtxSetLimit`, `cuCtxGetCacheConfig`, `cuCtxSetCacheConfig`, `cuCtxGetSharedMemConfig`, `cuCtxSetSharedMemConfig`, `cuCtxGetStreamPriorityRange`, `cuCtxGetId`, `cuCtxGetApiVersion`, `cuCtxPopCurrent`, `cuCtxPushCurrent`, `cuCtxAttach`, `cuCtxDetach`, `cuDeviceGetUuid`, `cuDeviceGetTexture1DLinearMaxWidth`, `cuDeviceGetP2PAttribute`, `cuDeviceGetGraphMemAttribute`, `cuDeviceSetGraphMemAttribute`, `cuDeviceFlushGPUDirectRDMAWrites`, `cuGraphCreate`, `cuGraphClone`, `cuGraphDestroy`, `cuGraphInstantiate`, `cuGraphLaunch`, `cuGraphExecDestroy`, `cuGraphAddMemcpyNode`, `cuGraphAddMemsetNode`, `cuGraphAddKernelNode`, `cuStreamBeginCapture`, `cuStreamBeginCaptureToGraph`, `cuStreamEndCapture`, `cuImportExternalMemory`, `cuDestroyExternalMemory`, `cuExternalMemoryGetMappedBuffer`, `cuExternalMemoryGetMappedMipmappedArray`, `cuImportExternalSemaphore`, `cuDestroyExternalSemaphore`, `cuSignalExternalSemaphoresAsync`, `cuWaitExternalSemaphoresAsync`, `cuProfilerStart`, `cuProfilerStop`, `cuGetErrorName`, `cuGetErrorString`, `cuTexRefSetAddress2D`, `cuTexRefSetAddressMode`, `cuTexRefSetFilterMode`, `cuTexRefSetMaxAnisotropy`, `cuTexRefSetMipmapFilterMode`, `cuTexRefSetMipmapLevelBias`, `cuTexRefSetMipmapLevelClamp`, `cuTexRefSetBorderColor`, `cuSurfRefSetFormat`, `cuModuleLoadFatBinary`, `cuModuleLink*`, `cuModuleGetLoadingMode`, `cuLaunchCooperativeKernel`, `cuLaunchCooperativeKernelMultiDevice`, `cuLaunchKernelEx`, `cuOccupancyMaxActiveBlocksPerMultiprocessor`, `cuOccupancyMaxPotentialBlockSize`, `cuOccupancyMaxPotentialBlockSizeWithFlags`

**Files**:
- `src/api/cuda_driver/cuda_driver_memory.cpp`
- `src/api/cuda_driver/cuda_driver_memcpy.cpp`
- `src/api/cuda_driver/cuda_driver_memset.cpp`
- `src/api/cuda_driver/cuda_driver_stream_event.cpp`
- `src/api/cuda_driver/cuda_driver_device_context.cpp`
- `src/api/cuda_driver/cuda_driver_module.cpp`
- `src/api/cuda_driver/cuda_driver_graph.cpp`
- `src/api/cuda_driver/cuda_driver_external.cpp`
- `src/api/cuda_driver/cuda_driver_texture.cpp`
- `src/api/cuda_driver/cuda_driver_errors.cpp`
- `src/api/cuda_driver/cuda_driver_occupancy.cpp`

**Still missing**: `cuDeviceGetNvSciSyncAttributes` (platform-specific, not applicable on Linux)

---

### 3.6 Phase 6 — PTX ISA Expansion

**Status**: **DONE** (2026-05-14 – 2026-05-15)

**Implemented PTX**:
- Texture: `tex.1d/2d/3d.f32` scalar + v2/v4, `tld4.2d.v4.f32.f32`, `txq.width/height/depth/channels`
- Surface: `suld.2d.f32/v2/v4`, `sust.2d.f32/v2/v4`
- Shared atomics: `atom.shared.add/cas/exch/max/min/and/or/xor/inc/dec` (all variants)
- Conversions: all `cvt.*` rounding modes (`rn/rz/rm/rp`) for `f32/f64 ↔ s32/u32/s64/u64/f16`, saturating variants
- FP variants: `sqrt.rn/rz/rm/rp.f32`, `rcp.rn.f32`, `div.rn.f32/f64`
- FP16 vectors: `ld.global.v2/v4.f16`, `st.global.v2/v4.f16`
- Wide integer: `mad.hi.s32/u32/s64/u64`, `mul.hi.s32/u32/s64/u64`, `div/rem.s64/u64`
- Warp: `vote.sync`, `shfl.sync.idx/down/up/bfly`, `activemask`, `redux.sync`, `match.sync.eq/lt`, `elect.sync`
- Control: `bar.sync/arrive`, `membar.gl/sys/cta`, `ret`, `grid.sync`, `griddepcontrol.launch_dependents/wait`
- MMA: `mma.sync.aligned` (Ampere shapes, INT4, binary `b1`), `wgmma.mma_async` (Hopper shapes)
- TMA: `cp.async.bulk.tensor.1d/2d/3d/4d/5d`, `cp.reduce.async.add/min/max.f32/f64`
- Blackwell: `tcgen05.mma.cta_group::1` (m64n256k16, m64n128k16, m64n256k8, m128n256k16)
- Misc: `lop3.b32`, `popc`, `clz`, `bitcast`, `prmt.b32` (all modes), `sad.u32`, `dsad`
- Cooperative group primitives: `match.sync.eq`, `elect.sync`

**Files**:
- `src/compiler/ptx/ptx_texture_ops.cpp`
- `src/compiler/ptx/ptx_shared_atomics.cpp`
- `src/compiler/ptx/ptx_conversion.cpp`
- `src/compiler/ptx/ptx_translator_map.cpp`
- `include/vgre/compiler/wmma_emulation.h`

**Key design decisions**:
- `tex` v2/v4 vector returns replicate scalar sample to all channels (single-channel float TextureManager)
- `txq` returns conservative defaults (1024×1024×1, 1 channel)
- `tcgen05.mma` delegates to existing `wgmma` GEMM math (no true SM100-specific behavior)

---

### 3.7 Phase 7 — NCCL Point-to-Point & Advanced Collectives

**Status**: **DONE** (single-node; multi-node TCP functional)

**Implemented**: `ncclSend`, `ncclRecv`, `ncclAllToAll`, `ncclGather`, `ncclScatter`

**File**: `src/api/nccl_shim.cpp`

**Design**: Single-node uses shared-memory `p2p_slots` with two-phase barrier. Multi-node routes through `TCPClusterManager`.

---

### 3.8 Phase 8 — Cooperative Groups & Device-Side Libraries

**Status**: **DONE**

**Implemented**:
- `thread_block`, `coalesced_group`, `thread_block_tile<4/8/16/32>`, `grid_group`, `multi_grid_group`
- Member functions: `sync()`, `size()`, `thread_rank()`, `group_index()`, `thread_index()`, `shfl()`, `shfl_up()`, `shfl_down()`, `shfl_xor()`
- Algorithms: `reduce()`, `reduce_sum()`, `reduce_min()`, `reduce_max()`, `partition()`, `match_any()`, `match_all()`
- CUB Fallback: `cub::WarpReduce` (Sum, Min, Max), `cub::BlockReduce`, `cub::WarpScan`, `cub::BlockScan`, `cub::CachingDeviceAllocator`

**Files**:
- `include/vgre/compiler/cuda_device_libs/cooperative_groups.h`
- `include/vgre/compiler/cuda_device_libs/cub_fallback.h`

---

### 3.9 Phase 9 — Deployment

**Status**: **DONE**

**K8s Device Plugin**: `src/deployment/k8s_device_plugin/main.go` (Go gRPC) + `Dockerfile` + `daemonset.yaml` + `README.md`

**SLURM GRES Plugin**: `src/deployment/slurm_gres/slurm_gres_vgpu.cpp` + `slurm_gres_vgpu.h`

---

### 3.10 Phase 10 — CDP, Profiling, Advanced Formats, Graph Updates

**Status**: **PARTIAL**

#### 3.10.1 CUDA Dynamic Parallelism (CDP)

**Status**: **DONE**

**Implemented**: `cudaDeviceSynchronize`, `cudaGetParameterBufferV2`, `cudaLaunchDeviceV2`

**File**: `src/runtime/cdp_executor.cpp`

**Note**: `cudart_shim_cdp.cpp` was **never created**; CDP is fully in `cdp_executor.cpp`.

---

#### 3.10.2 Profiling / Observability

**Status**: **PARTIAL — consolidated in `runtime_profiler.cpp`**

**Implemented**: Kernel timeline with nanosecond timestamps, Chrome trace export (`toChromeTraceJSON()`), `InstructionSample` struct with heuristic instruction mix estimation, `RuntimeProfiler::recordEvent` for NVTX-style ranges.

**File**: `src/advanced/runtime_profiler.cpp`

**Note**: The claimed separate files (`cupti_equivalent.cpp`, `kernel_timeline.cpp`, `instruction_sampler.cpp`) were **never created**. All profiling functionality lives in `runtime_profiler.cpp`.

**Caveat**: `InstructionSample` is purely heuristic (no hardware PMU counters). See `missingFeatures.md` §3.4.

---

#### 3.10.3 cuDNN INT8x4 / INT8x32 Packed Tensor Layouts

**Status**: **DONE** (inside existing files)

**Implemented**: `CUDNN_DATA_INT8x32` supported in `cudnnConvolutionForward`, `cudnnPoolingForward/Backward`, `cudnnActivationForward/Backward` via dequantize→FP32 compute→requantize path.

**Note**: `src/api/cudnn/cudnn_int8_packed.cpp` was **never created**. INT8 packed logic is in `cudnn_convolution.cpp`, `cudnn_pooling.cpp`, `cudnn_activation.cpp`.

---

#### 3.10.4 Graph Exec Update v2

**Status**: **DONE** (inside interceptor, not a separate file)

**Implemented**: `cudaGraphExecUpdate_v2` with per-node `cudaGraphExecUpdateResultInfo` error reporting.

**File**: `src/api/cuda_interceptor_graphs.cpp`

**Note**: `src/core/graph/graph_manager_exec_update_v2.cpp` was **never created**. Functionality is in `CUDAInterceptor::graphExecUpdateV2()`.

---

## 4. File Manifest Summary

### New Directories (created)

| Directory | Purpose | Files |
|---|---|---|
| `src/api/cudart/` | Split monolithic CUDART shim | 15 `.cpp` files |
| `src/api/cuda_driver/` | Split monolithic driver shim | 9 `.cpp` files |
| `src/api/cublas/` | cuBLAS by BLAS level | 7 `.cpp` files |
| `src/api/cudnn/` | cuDNN by layer type | 12 `.cpp` files |
| `src/api/cufft/` | cuFFT shim | 1 `.cpp` file |
| `src/api/curand/` | cuRAND shim | 1 `.cpp` file |
| `src/api/cusolver/` | cuSOLVER LAPACK delegation | 1 `.cpp` file |
| `src/api/cusparse/` | cuSPARSE shim (core, format, triangular, factorization) | 4 `.cpp` files + `cusparse_state.h` |
| `src/api/cublasLt/` | cuBLASLt shim | 1 `.cpp` file |
| `src/compiler/ptx/` | PTX translator by family | 4 `.cpp` files |
| `src/core/graph/` | Graph manager by concern | 7 `.cpp` files |
| `src/deployment/k8s_device_plugin/` | K8s Device Plugin | 4 files (Go) |
| `src/deployment/slurm_gres/` | SLURM GRES plugin | 2 files (C) |

### Directories / Files That Were Planned But Never Created

| Planned | Reason | Actual Location |
|---|---|---|
| `src/core/graph/graph_manager_exec_update_v2.cpp` | Functionality merged into interceptor | `src/api/cuda_interceptor_graphs.cpp` |
| `src/api/cudnn/cudnn_int8_packed.cpp` | Logic merged into existing convolution/pooling/activation files | `src/api/cudnn/cudnn_convolution.cpp`, etc. |
| `src/api/cudart/cudart_shim_cdp.cpp` | CDP is runtime concern, not CUDART shim | `src/runtime/cdp_executor.cpp` |
| `src/advanced/profiling/` directory | Profiling kept in single file | `src/advanced/runtime_profiler.cpp` |
| `src/compiler/cuda_device_libs/` | Already exists as `include/vgre/compiler/cuda_device_libs/` | `include/vgre/compiler/cuda_device_libs/cooperative_groups.h`, `cub_fallback.h` |

---

## 5. CMake & Build Integration

### 5.1 Per-Subdirectory CMakeLists.txt

Each new subdirectory gets its own `CMakeLists.txt`:

```cmake
# src/api/cufft/CMakeLists.txt (if ever split)
file(GLOB CUFFT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_library(vgre_cufft OBJECT ${CUFFT_SOURCES})
target_link_libraries(vgre_cufft PRIVATE vgre_core)

# Optional: FFTW3 delegation
find_package(FFTW3 COMPONENTS double single)
if(FFTW3_FOUND)
    target_compile_definitions(vgre_cufft PRIVATE VGRE_HAS_FFTW3)
    target_link_libraries(vgre_cufft PRIVATE FFTW3::FFTW3)
endif()
```

### 5.2 Conditional Compilation

```cmake
option(VGRE_ENABLE_CUFFT "Build cuFFT shim" ON)
option(VGRE_ENABLE_CURAND "Build cuRAND shim" ON)
option(VGRE_ENABLE_CUSOLVER "Build cuSOLVER shim" ON)
option(VGRE_ENABLE_CUSPARSE "Build cuSPARSE shim" ON)
option(VGRE_ENABLE_CUBLASLT "Build cuBLASLt shim" ON)
```

### 5.3 Cross-Platform Build Requirements

| Feature | Linux | Windows | macOS |
|---|---|---|---|
| **cuRAND entropy** | `getrandom()` / `/dev/urandom` | `BCryptGenRandom()` | `getentropy()` |
| **cuFFT** | Reference DFT (no FFTW3 linked yet) | Reference DFT | Reference DFT |
| **cuSOLVER** | System `liblapack.so.3` | Intel MKL or OpenBLAS | Accelerate.framework / OpenBLAS |
| **Shared memory (IPC)** | POSIX `shm_open` | `CreateFileMapping` | POSIX `shm_open` |
| **External semaphores** | POSIX named semaphores | Windows named semaphores | POSIX named semaphores |
| **K8s Device Plugin** | Unix domain sockets | Named pipes | Unix domain sockets |
| **SLURM GRES** | `slurm_spank.h` | N/A | N/A |

### 5.4 Mesh Topology Integration

The existing mesh topology system must be extended for new NCCL p2p and advanced collectives:

| Collective | Mesh Impact | File |
|---|---|---|
| `ncclSend`/`ncclRecv` | Direct rank→rank TCP if not already connected | `src/api/nccl_shim.cpp` |
| `ncclAllToAll` | Full-duplex mesh already supports this | `src/api/nccl_shim.cpp` |
| `ncclGather`/`ncclScatter` | Root connects to all others | `src/api/nccl_shim.cpp` |

**Design rule**: Any new distributed feature must call `TCPClusterManager::ensureConnected(rank)` before sending data.

### 5.5 Functioning / Runtime Engine Integration

| Subsystem | Integration Point | Requirement |
|---|---|---|
| **Adaptive Execution Engine** | New compute-heavy shims call `recordExecution()` | Thread count auto-tuned per kernel |
| **Runtime Profiler** | New APIs emit NVTX ranges | OTLP export already works |
| **Memory Manager** | New allocations go through `MemoryManager` | UVM tracking, NUMA binding |
| **Scheduler** | Graph node replay uses `StreamScheduler` | New node types must enqueue correctly |
| **Resource Ledger** | Track GPU-memory-equivalent allocations | Prevent CPU host OOM |

---

## 6. Testing Strategy

### 6.1 Test File Organization

```
tests/api/cudart/          # CUDART shim tests
tests/api/cuda_driver/     # Driver API tests
tests/api/cublas/           # cuBLAS tests
tests/api/cudnn/            # cuDNN tests
tests/api/cufft/            # cuFFT tests
tests/compiler/ptx/         # PTX instruction tests
tests/core/graph/           # Graph node-type tests
tests/core/texture/         # Texture/surface object tests
```

### 6.2 Minimum Test Coverage per Phase

| Phase | New Tests | Target Coverage |
|---|---|---|
| P1 | 15–20 | Every new CUDART shim function |
| P2 | 10–15 | Level-2/Level-3 reference vs OpenBLAS |
| P3 | 15–20 | Backward pass gradient check |
| P4 | 20–30 | Round-trip correctness for each new library |
| P5 | 10–15 | Driver API parity with Runtime API |
| P6 | 10–15 | PTX instruction → C++ output verification |
| P7 | 5–10 | P2P correctness via mock cluster |
| P8 | 5–10 | Cooperative groups compile + run |
| P9 | 3–5 | Plugin integration tests |

---

## 7. Documentation Checklist

**Before any PR merges that implements a gap, update these documents in this order:**

1. **`docs/missingFeatures.md`** — Remove the implemented item from "Missing" or mark as resolved.
2. **`docs/PROJECT_STATUS.md`** — Update coverage percentages and "What Works" section.
3. **`docs/PRODUCTION_READINESS_REPORT.md`** — If the gap was critical, update the Executive Summary.
4. **`docs/implementationPlan.md`** — Mark the item as `DONE` with date and commit hash.
5. **`README.md`** (top-level) — If the feature is user-facing, update "What Works."

---

## 8. Progress Tracker

### Phase 1 — Critical CUDA Runtime Gaps

| # | Feature | Status | Date |
|---|---|---|---|
| 1.1 | `cudaStreamWaitEvent` | **DONE** | 2026-05-13 |
| 1.2 | `cudaEventQuery` | **DONE** | 2026-05-13 |
| 1.3 | `cudaStreamAddCallback` | **DONE** | 2026-05-13 |
| 1.4 | `cudaLaunchHostFunc` | **DONE** | 2026-05-13 |
| 1.5 | `cudaGetErrorName` / `cudaGetErrorString` | **DONE** | 2026-05-13 |
| 1.6 | `cudaMemcpyToSymbol` / `cudaMemcpyFromSymbol` | **DONE** | 2026-05-13 |
| 1.7 | `cudaMallocArray` / `cudaMalloc3DArray` | **DONE** | 2026-05-13 |
| 1.8 | `cudaPointerGetAttributes` | **DONE** | 2026-05-13 |
| 1.9 | `cudaMemset2D/3D/2DAsync/3DAsync` | **DONE** | 2026-05-13 |
| 1.10 | `cudaGraphAddKernelNode` | **DONE** | 2026-05-13 |
| 1.11 | `cudaGraphAddMemsetNode` | **DONE** | 2026-05-13 |
| 1.12 | `cudaGraphAddHostNode` | **DONE** | 2026-05-13 |
| 1.13 | `cudaGraphAddChildGraphNode` | **DONE** | 2026-05-13 |
| 1.14 | `cudaGraphAddEmptyNode` | **DONE** | 2026-05-13 |
| 1.15 | `cudaGraphAddEventRecordNode` / `EventWaitNode` | **DONE** | 2026-05-13 |
| 1.16 | `cudaGraphAddMemAllocNode` / `MemFreeNode` | **DONE** | 2026-05-13 |
| 1.17 | Graph introspection APIs (`GetNodes`, `GetEdges`, `NodeGetType`, etc.) | **DONE** | 2026-05-13 |
| 1.18 | Graph exec mutation APIs (`ExecKernelNodeSetParams`, `NodeSetEnabled`, etc.) | **DONE** | 2026-05-13 |
| 1.19 | Stream capture introspection (`IsCapturing`, `GetCaptureInfo_v2`, etc.) | **DONE** | 2026-05-13 |
| 1.20 | CUDART texture/surface object APIs | **DONE** | 2026-05-13 |
| 1.21 | CUDA Runtime device/function attributes | **DONE** | 2026-05-13 |
| 1.22 | CUDA Graph dependencies & user objects | **DONE** | 2026-05-13 |
| 1.23 | CUDART external memory/semaphore APIs | **DONE** | 2026-05-13 |
| 1.24 | `cudaGraphExecUpdate_v2` | **DONE** | 2026-05-15 |

### Phase 2 — cuBLAS

| # | Feature | Status | Date |
|---|---|---|---|
| 2.1 | Level-1 BLAS completion (copy, swap, rot, rotm, rotg, asum, amax/amin, Dnrm2, Dscal) | **DONE** | 2026-05-13 |
| 2.2 | Level-2 BLAS (trsv, ger, symv, gbmv, syr, syr2, trmv, tbsv, tpsv, spmv, sbmv, spr, spr2, tbmv, tpmv) | **DONE** | 2026-05-13/15 |
| 2.3 | Level-3 BLAS (trsm, syrk, syr2k, trmm, symm, batched variants) | **DONE** | 2026-05-13/15 |
| 2.4 | `cublasGemmEx` / `GemmBatchedEx` / `GemmStridedBatchedEx` | **DONE** | 2026-05-13 |
| 2.5 | Hermitian routines (Cherk, Zherk, Cher2k, Zher2k, Chemm, Zhemm) | **DONE** | 2026-05-14/15 |
| 2.6 | Logger configure / callback | **DONE** | 2026-05-14 |
| 2.7 | **Complex C/Z Level-1/2/3 GEMM/GEMV** | **DONE** | 2026-05-15 |

### Phase 3 — cuDNN Backward Passes + Training

| # | Feature | Status | Date |
|---|---|---|---|
| 3.1 | Convolution backward (data, filter, bias) | **DONE** | 2026-05-13/15 |
| 3.2 | Batch normalization (training, backward) | **DONE** | 2026-05-13 |
| 3.3 | Activation / softmax / pooling backward | **DONE** | 2026-05-14 |
| 3.4 | Dropout forward/backward | **DONE** | 2026-05-14 |
| 3.5 | RNN forward/backward | **DONE** | 2026-05-14 |
| 3.6 | Multi-head attention forward/backward | **DONE** | 2026-05-14 |
| 3.7 | CTC loss | **DONE** | 2026-05-14 |
| 3.8 | Divisive normalization | **DONE** | 2026-05-14 |
| 3.9 | LRN | **DONE** | 2026-05-14 |
| 3.10 | Tensor ops (OpTensor, ReduceTensor, TransformTensor, AddTensor) | **DONE** | 2026-05-14/15 |
| 3.11 | INT8x4 / INT8x32 packed layouts | **DONE** | 2026-05-15 |
| 3.12 | Backend API wired to legacy + attention routing | **DONE** | 2026-05-16 |

### Phase 4 — Missing Libraries

| # | Feature | Status | Date |
|---|---|---|---|
| 4.1 | cuFFT reference DFT/IDFT | **DONE** | 2026-05-14 |
| 4.2 | cuRAND pseudo-random generators | **DONE** | 2026-05-14 |
| 4.3 | cuSOLVER dense (potrf, geqrf, gesvd, syevd) | **DONE** | 2026-05-14 |
| 4.4 | cuSPARSE CSR SpMV/SpMM | **DONE** | 2026-05-14 |
| 4.5 | cuBLASLt basic matmul + ReLU/GELU/Bias | **DONE** | 2026-05-14 |
| 4.6 | cuFFT FFTW3 delegation | **DONE** | 2026-05-15 |
| 4.7 | cuSOLVER LU/least-squares (getrf, getrs, ormqr, gelsd) | **DONE** | 2026-05-15 |
| 4.8 | cuSPARSE format conversions (CSC, SparseToDense, DenseToSparse) + SpSV | **DONE** | 2026-05-16 |
| 4.9 | cuSPARSE SpGEMM (3-phase sparse×sparse) + ILU0 + IC0 | **DONE** | 2026-05-16 |
| 4.10 | cuBLASLt LRU algorithm cache (128-entry, O(1) heuristic lookup) | **DONE** | 2026-05-16 |
| 4.11 | cuDNN Backend attention routing (SDPA via descriptor graph) | **DONE** | 2026-05-16 |
| 4.12 | cuRAND per-handle mutex (thread-safe concurrent generation) | **DONE** | 2026-05-16 |
| 4.13 | Texture SRGB gamma decode (`TextureDescriptor::srgbDecode` flag) | **DONE** | 2026-05-16 |
| 4.14 | cuSolver dense API complete: getrf + getrs + ormqr + gelsd via LAPACK | **DONE** | 2026-05-16 |
| 4.15 | cuSOLVER sparse API (cusolverSp) | **MISSING** — requires UMFPACK/CHOLMOD | — |

### Phase 5 — CUDA Driver API

| # | Feature | Status | Date |
|---|---|---|---|
| 5.1 | Memory & copies (managed, host, pitch, 2D/3D, peer, async) | **DONE** | 2026-05-14 |
| 5.2 | Streams & events (callback, query, flags, priority, id, ctx, capture) | **DONE** | 2026-05-14 |
| 5.3 | Context & device (limits, cache, smem, uuid, P2P, graph mem) | **DONE** | 2026-05-14 |
| 5.4 | Graphs, external resources, profiler, error strings | **DONE** | 2026-05-14 |
| 5.5 | Texture / surface reference (SetAddress2D, filter, mipmap, border) | **DONE** | 2026-05-14 |
| 5.6 | Module loading (fat binary, linker, loading mode) | **DONE** | 2026-05-15 |
| 5.7 | Cooperative launch + occupancy | **DONE** | 2026-05-14 |

### Phase 6 — PTX ISA

| # | Feature | Status | Date |
|---|---|---|---|
| 6.1 | Texture / surface instructions | **DONE** | 2026-05-14 |
| 6.2 | Shared-memory atomics | **DONE** | 2026-05-14 |
| 6.3 | Conversion & precision variants | **DONE** | 2026-05-14 |
| 6.4 | `rcp.rn`, `sqrt.rn`, `div.rn` | **DONE** | 2026-05-14 |
| 6.5 | FP16 vector loads/stores | **DONE** | 2026-05-14 |
| 6.6 | Wide integer MAD/MUL/DIV/REM | **DONE** | 2026-05-15 |
| 6.7 | `match.sync`, `elect.sync` | **DONE** | 2026-05-14 |
| 6.8 | `grid.sync`, `griddepcontrol` | **DONE** | 2026-05-14 |
| 6.9 | `cp.async.bulk.tensor.3d/4d/5d` | **DONE** | 2026-05-14 |
| 6.10 | `tcgen05.mma` (Blackwell) | **DONE** | 2026-05-15 |
| 6.11 | `prmt.b32`, `sad.u32`, `dsad` | **DONE** | 2026-05-15 |
| 6.12 | MMA INT4 / binary `b1` | **DONE** | 2026-05-15 |

### Phase 7 — NCCL

| # | Feature | Status | Date |
|---|---|---|---|
| 7.1 | `ncclSend` / `ncclRecv` | **DONE** | 2026-05-14 |
| 7.2 | `ncclAllToAll` | **DONE** | 2026-05-14 |
| 7.3 | `ncclGather` / `ncclScatter` | **DONE** | 2026-05-14 |
| 7.4 | Multi-node RDMA transport | **MISSING** | — |

### Phase 8 — Cooperative Groups & Device Libraries

| # | Feature | Status | Date |
|---|---|---|---|
| 8.1 | Cooperative groups C++ API | **DONE** | 2026-05-14 |
| 8.2 | CUB fallback headers | **DONE** | 2026-05-14 |

### Phase 9 — Deployment

| # | Feature | Status | Date |
|---|---|---|---|
| 9.1 | K8s Device Plugin (Go gRPC) | **DONE** | 2026-05-14 |
| 9.2 | SLURM GRES plugin (C) | **DONE** | 2026-05-14 |

### Phase 10 — CDP, Profiling, Advanced Formats

| # | Feature | Status | Date |
|---|---|---|---|
| 10.1 | CDP (`cudaDeviceSynchronize`, `GetParameterBufferV2`, `LaunchDeviceV2`) | **DONE** | 2026-05-14 |
| 10.2 | Profiling (kernel timeline, Chrome trace, `InstructionSample`) | **PARTIAL** | 2026-05-14 |
| 10.5 | cuBLASLt DRELU/DGELU/BGRAD epilogues + scale ptrs | **DONE** | 2026-05-16 |
| 10.6 | cuDNN Backend attention routing | **DONE** | 2026-05-16 |
| 10.7 | cuRAND thread safety (per-handle mutex) | **DONE** | 2026-05-16 |
| 10.8 | Texture SRGB gamma decode | **DONE** | 2026-05-16 |
| 10.9 | cuSPARSE CSC + SparseToDense + DenseToSparse + SpSV | **DONE** | 2026-05-16 |
| 10.3 | cuDNN INT8x4 / INT8x32 packed layouts | **DONE** | 2026-05-15 |
| 10.4 | `cudaGraphExecUpdate_v2` | **DONE** | 2026-05-15 |

---

## 9. OpenMP Performance Optimization

**Status**: **DONE** (commit `803b76f`, 2026-05-15)

All major O(n²) and O(n³) CPU reference compute paths across cuBLAS, cuBLASLt, cuDNN, cuFFT, cuSPARSE, and core now use conditional OpenMP parallelization. Shared `include/vgre/common/openmp_helper.h` consolidates all OpenMP includes. See commit message for full details.

---

## 10. File-Splitting Refactor (2026-05-15)

**Status**: **DONE** — all splits verified, 110/110 tests pass, zero compilation errors.

Large monolithic source files (>800 lines) were split into smaller, logically grouped files. All methods verified present via automated function-signature comparison against git originals.

### 10.1 Split Summary

| Original File | Lines | Split Into | New File Count |
|---|---|---|---|
| `src/api/cudart/cudart_shim.cpp` | 1429 | `cudart_shim.cpp` (994) + `cudart_memory_pool.cpp` (240) + `cudart_cooperative.cpp` (201) + `cudart_mipmap.cpp` (90) | 4 |
| `src/api/cublas/cublas_level2.cpp` | 1018 | `cublas_level2.cpp` (455) + `cublas_level2_packed.cpp` (570) | 2 |
| `src/api/cublas/cublas_level3.cpp` | 1173 | `cublas_level3.cpp` (1025) + `cublas_level3_blas3.cpp` (154) | 2 |
| `src/api/cuda_interceptor.cpp` | 1147 | `cuda_interceptor.cpp` (869) + `cuda_interceptor_device.cpp` (309) | 2 |
| `src/runtime/vector_engine.cpp` | 1163 | `vector_engine.cpp` (460) + `vector_engine_float.cpp` (371) + `vector_engine_double.cpp` (412) | 3 |
| `src/core/graph/graph_manager.cpp` | 975 | `graph_manager.cpp` (228) + `graph_manager_nodes.cpp` (519) + `graph_manager_serde.cpp` (265) | 3 |
| `src/advanced/adaptive_execution_engine.cpp` | 1082 | `adaptive_execution_engine.cpp` (489) + `adaptive_execution_engine_record.cpp` (251) + `adaptive_execution_engine_tune.cpp` (535) | 3 |
| `src/advanced/hybrid_compute_manager.cpp` | 930 | `hybrid_compute_manager.cpp` (342) + `hybrid_compute_manager_remote.cpp` (288) + `hybrid_compute_manager_workload.cpp` (383) | 3 |

### 10.2 Cross-File Dependencies Introduced

| Dependency | Solution |
|---|---|
| `CUDAModuleRegistry` (cudart_shim.cpp) used by `cudart_cooperative.cpp` | Cross-file helpers: `vgre_lookup_kernel_name()`, `vgre_lookup_kernel_source()` |
| `kVgreCudaVersion` (anonymous namespace) used by `cuda_interceptor_device.cpp` | Duplicated `constexpr` in anonymous namespace |
| `cudaMemPool_t` typedef used by `cudart_memory_pool.cpp` | Duplicated `using` declaration |
| `pickExplorationThreadCount` (anonymous namespace) used by `adaptive_execution_engine_tune.cpp` | Duplicated anonymous namespace block |
| Batched BLAS3 functions call non-batched L3 functions | `extern "C"` forward declarations in `cublas_level3_blas3.cpp` |

### 10.3 CMakeLists.txt Files Updated

- `src/core/CMakeLists.txt` — added `graph_manager_nodes.cpp`, `graph_manager_serde.cpp`
- `src/advanced/CMakeLists.txt` — added 4 split files (adaptive + hybrid)
- `src/runtime/CMakeLists.txt` — added `vector_engine_float.cpp`, `vector_engine_double.cpp`
- `src/api/CMakeLists.txt` — added `cuda_interceptor_device.cpp`, `cublas_level2_packed.cpp`, `cublas_level3_blas3.cpp`, `cudart_memory_pool.cpp`, `cudart_cooperative.cpp`, `cudart_mipmap.cpp`
- `CMakeLists.txt` (top-level) — added `cuda_interceptor_device.cpp`, `cublas_level2_packed.cpp`, `cublas_level3_blas3.cpp` to `vgre_cudart` target

---

*Last updated: 2026-05-16 (second pass). 113/113 tests passing. Sparse factorization (ILU0/IC0/SpGEMM), cuBLASLt LRU cache, RESOURCE_LOCK test serialization added.*
