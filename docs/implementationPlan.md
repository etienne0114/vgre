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
src/api/cufft/           # cuFFT shim split into execution + planning/state
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
src/advanced/profiling/  # Not used; profiling split in src/advanced/runtime_profiler*.cpp
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
| **P10** | CDP, profiling, advanced formats, graph updates | **DONE** — CDP, graph update v2, profiler timeline, OTLP/Chrome export, LLVM-IR instruction classification | Low |

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

**Files**: `src/api/cufft/cufft_core.cpp` (execution kernels), `src/api/cufft/cufft_plan.cpp` (plan lifecycle, size queries, IPC), `src/api/cufft/cufft_internal.h` (internal plan state)

**Implemented**: `cufftPlan1d/2d/3d/Many`, `cufftDestroy`, `cufftExecC2C/Z2Z/R2C/C2R/D2Z/Z2D`, half-precision `cufftExecC16C/R16C/C16R`, `cufftSetStream/WorkArea`, `cufftEstimate*`, `cufftGetSize*`, `cufftMakePlanMany` with strided batched 1D layouts, IPC plan export/import

**Implementation**: O(n log n) Cooley-Tukey radix-2 FFT for power-of-2 sizes + Bluestein's algorithm for arbitrary sizes. OpenMP-parallelized butterfly stages and batch/row/column loops. Optional FFTW3 delegation (`VGRE_HAS_FFTW3`, auto-detected by CMake `pkg_check_modules`). Plans stored in global handle map for reuse.

**Test**: `tests/api/test_cufft.cpp` — 13 tests (pow2 roundtrip, non-pow2, single-frequency, Z2Z double, R2C+C2R, D2Z+Z2D, batched, 2D, prime-size, Parseval, plan management, FP16 C16C roundtrip, strided `PlanMany`). All pass.

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
**Sparse API complete** (cusolverSp, 2026-05-16): csrlsvlu, csrlsvchol, csrlsqvqr, csreigvsi — via CSR→dense extraction + LAPACK. See `missingFeatures.md` §3.6 for full table.

---

#### 3.4.4 cuSPARSE

**Files**: `src/api/cusparse/cusparse_core.cpp` + `cusparse_format.cpp` + `cusparse_triangular.cpp` + `cusparse_factorization.cpp` (split at 2026-05-16 when core reached 1017 lines)

**Shared state**: `src/api/cusparse/cusparse_state.h` (internal, not public)

**Implemented**:
- Core: `cusparseCreate/Destroy`, `cusparseSpMV/SpMM` (CSR + COO + CSC, S/D/C/Z/FP16/BF16/INT8), `cusparseAxpyi`, buffer-size queries
- Format: `cusparseCreateCsc`, `cusparseSparseToDense`, `cusparseDenseToSparse` (bufferSize+analysis+compress), `cusparseSpMatGetAttribute/SetAttribute`, `cusparseSpMatGetSize`, `cusparseCsrSetPointers`
- Triangular: `cusparseSpSV` (full forward/backward/transposed substitution for lower+upper triangular CSR)

**Also implemented** (2026-05-16): `cusparseSpGEMM` (3-phase, `cusparse_factorization.cpp`), `cusparseScsrilu02`/`Dcsrilu02` (ILU0), `cusparseScsric02`/`Dcsric02` (IC0). Bug-fixed 2026-05-16: SpGEMM first-pass used corrupt dual-purpose sentinel; replaced with dedicated `inUse[]` bool array.
**Still missing**: Full-fill sparse factorization (UMFPACK-style with fill-in). Requires external library (UMFPACK/SuperLU); architectural limitation.

---

#### 3.4.5 cuBLASLt

**Files**: `src/api/cublaslt/cublaslt_core.cpp` (478 lines) + `cublaslt_matmul.cpp` (285 lines) + `cublaslt_state.h` (internal, 110 lines). Split 2026-05-16 when core reached 914 lines.

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

**Status**: **DONE**

#### 3.10.1 CUDA Dynamic Parallelism (CDP)

**Status**: **DONE**

**Implemented**: `cudaDeviceSynchronize`, `cudaGetParameterBufferV2`, `cudaLaunchDeviceV2`

**File**: `src/runtime/cdp_executor.cpp`

**Note**: `cudart_shim_cdp.cpp` was **never created**; CDP is fully in `cdp_executor.cpp`.

---

#### 3.10.2 Profiling / Observability

**Status**: **DONE — split by concern**

**Implemented**: Kernel timeline with nanosecond timestamps, Chrome trace export (`toChromeTraceJSON()`), OTLP JSON/HTTP export, `InstructionSample` struct with deterministic LLVM-IR instruction classification when IR is available, conservative unclassified totals when IR is unavailable, `RuntimeProfiler::recordEvent` for NVTX-style ranges.

**Files**:
- `src/advanced/runtime_profiler.cpp` — events, aggregates, JSON/Chrome/OTLP export
- `src/advanced/runtime_profiler_instruction.cpp` — instruction sample recording and LLVM-IR classification

**Caveat**: VGRE is a CPU runtime, so it does not expose physical NVIDIA CUPTI/PMU counters. Instruction mix is derived from generated LLVM IR, not guessed from source text.

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
| `src/api/cublasLt/` | cuBLASLt shim (core + matmul) | 2 `.cpp` files + `cublaslt_state.h` |
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
| `src/advanced/profiling/` directory | Kept under `src/advanced/` to match existing CMake layout | `runtime_profiler.cpp`, `runtime_profiler_instruction.cpp` |
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
| **cuFFT** | Built-in O(n log n), optional FFTW3 | Built-in O(n log n) | Built-in O(n log n) |
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
| 4.1 | cuFFT O(n log n) FFT/IDFT backend | **DONE** | 2026-05-14/16 |
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
| 4.15 | cuSOLVER sparse API (cusolverSp) | **DONE** (2026-05-16) — CSR→dense + LAPACK | 2026-05-16 |

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
| 7.4 | Multi-node RDMA transport | **DONE** (optional `VGRE_ENABLE_RDMA`, libibverbs/RoCE, TCP fallback) | 2026-05-16 |

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
| 10.2 | Profiling (kernel timeline, Chrome trace, OTLP, LLVM-IR `InstructionSample`) | **DONE** | 2026-05-16 |
| 10.15 | cuFFT FP16 execution + strided `PlanMany` + plan/source split | **DONE** | 2026-05-16 |
| 10.5 | cuBLASLt DRELU/DGELU/BGRAD epilogues + scale ptrs | **DONE** | 2026-05-16 |
| 10.10 | cuBLASLt file split (914→core 362+matmul 337 lines) + cublaslt_state.h | **DONE** | 2026-05-16 |
| 10.11 | cuSPARSE SpSV FP16/BF16 widen-compute-narrow path | **DONE** | 2026-05-16 |
| 10.12 | cuSPARSE SpGEMM FP16/BF16 widen-compute-narrow path | **DONE** | 2026-05-16 |
| 10.13 | FLOPCounting RESOURCE_LOCK vgre_llvm_jit (prevent parallel LLVM JIT contention) | **DONE** | 2026-05-16 |
| 10.14 | test_phase3_cluster shutdown race fix (missing cluster.shutdown() on early return) | **DONE** | 2026-05-16 |
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
- `src/api/CMakeLists.txt` — added `cuda_interceptor_device.cpp`, `cublas_level2_packed.cpp`, `cublas_level3_blas3.cpp`, `cudart_memory_pool.cpp`, `cudart_cooperative.cpp`, `cudart_mipmap.cpp`, `cudart_shim_array_memcpy.cpp`
- `CMakeLists.txt` (top-level) — added `cuda_interceptor_device.cpp`, `cublas_level2_packed.cpp`, `cublas_level3_blas3.cpp` to `vgre_cudart` target

---

## 11. Sixth Pass Fixes (2026-05-16)

**Status**: **DONE** — 116/116 tests pass.

### 11.1 TCPClusterPreservation test — VS Code port conflict

`tests/advanced/test_tcp_cluster_preservation.cpp` used hardcoded ports 17777–17783.
Port 17779 was bound by VS Code (pid found via `ss -tlnp`), causing `bind() = EADDRINUSE`.
**Fix**: Replaced all 8 hardcoded ports with `findFreePort()` (bind-to-0 OS allocation).

### 11.2 cudnn_backend_api.cpp — duplicate definitions + broken extern linkage

`cudnn_backend_api.cpp` re-declared `cudnnBackendDescriptorType_t`, `cudnnBackendAttributeName_t`, `BackendNode` inline, duplicating `cudnn_backend_internal.h`. Globals `g_backendNodes`/`g_nextBackendId` were in an anonymous namespace (internal linkage) but declared `extern` in the header.
**Fix**: Changed include to `cudnn_backend_internal.h`, removed ~150 lines of duplicates, moved globals to file scope, removed `static` from the 4 functions the header declares.

### 11.3 cudart_shim_array_memcpy.cpp — missing from build + 2 compile errors

`src/api/cudart/cudart_shim_array_memcpy.cpp` existed but was NOT in any CMakeLists.txt. Also had:
- `cudaArray_const_t` undefined
- `memcpyToArray`/`memcpyFromArray` called with 5 args (missing `kind`)
**Fix**: Added to `vgre_cudart_shims` in `src/api/CMakeLists.txt`; added typedef and the missing `kind` cast.

---

---

## 12. Seventh Pass — Missing API Declarations + New cuRAND Test Suite (2026-05-16)

**Status**: **DONE** — 117/117 tests pass.

### 12.1 cuRAND — Missing Header Declarations (4 functions)

`include/vgre/api/curand_shim.h` was missing declarations for 4 functions that existed in the .cpp:
- `curandGeneratePoisson` — Poisson-distributed uint32 output
- `curandGenerateSeeds` — re-seed generator from system entropy
- `curandGetVersion` — returns 1100000 (cuRAND 11.0)
- `curandGetGeneratorIpcHandle` / `curandCreateGeneratorFromIpcHandle` — IPC export/import

Also fixed: `curandGetDirectionVectors32/64` changed from `*` (caller-allocates 20000 entries)
to `**` (returns pointer to pre-computed internal static table), matching NVIDIA cuRAND API.
`curandGeneratorIpcHandle_t` is now a public typedef in the header (was private `vgre_curand_ipc_handle_t`).

### 12.2 cuFFT — Stream Association + Version Query

`CufftPlan` struct lacked a `stream` field: `cufftSetStream` was a validate-only stub.
- Added `void *stream` to `CufftPlan` in `cufft_internal.h`
- `cufftSetStream`: now stores stream pointer in plan map
- `cufftGetStream`: new function returning stored stream
- `cufftGetVersion`: new function returning 11000 (cuFFT 11.0)
- `cufft_shim.h`: both new functions declared

### 12.3 cuBLASLt — Missing Declarations (5 functions)

`cublaslt_shim.h` was missing declarations for 5 functions already implemented:
- `cublasLtGetVersion`, `cublasLtGetCudartVersion`
- `cublasLtGetStatusName`, `cublasLtGetStatusString`
- `cublasLtMatmulAlgoGetHeuristics` (plural) — now delegates to singular form (LRU cache shared)

### 12.4 CUDA Runtime — `cudaArrayDestroy`

`cudaArrayDestroy` (CUDA 1.x legacy alias for `cudaFreeArray`) was absent from all source files.
Added to `cudart_shim_stream.cpp` as a one-line wrapper around `cudaFreeArray`.

### 12.5 New Test: tests/api/test_curand.cpp (15 tests)

First dedicated cuRAND test suite added:
version query, lifecycle, seeding reproducibility, uniform/normal/log-normal/uint/Poisson generation,
GenerateSeeds, Sobol32/64 quasi-random, direction vectors, IPC handle export+import,
normal double, offset reproducibility. All 15 pass.

Also extended:
- `test_cufft.cpp`: 3 new tests (cufftGetVersion, cufftSetStream/GetStream round-trip, exec-with-stream)
- `test_cublaslt.cpp`: 3 new tests (version queries, status strings, plural heuristics)

---

---

## 13. Cross-Platform Hardening (2026-05-17)

**Status**: **DONE** — all three platforms (Linux / macOS / Windows) build and run cleanly.

### 13.1 Windows — vgre_sync.bat dynamic VS detection

| Problem | Fix |
|---|---|
| Hardcoded `Visual Studio\2022` paths everywhere | Replaced with `vswhere.exe` query (`-latest -requires Microsoft.VisualStudio.Workload.NativeDesktop -property installationPath`) |
| `(x86)` in `%ProgramFiles(x86)%` breaks `for … do (` blocks | Captured `_PF86=%ProgramFiles(x86)%` and `_PF64=%ProgramFiles%` at top level; used `!_PF86!\` inside all nested blocks |
| `- was unexpected at this time` batch parser crash | Fixed by the `_PF86`/`_PF64` extraction above; no parentheses inside nested block delimiters |
| Generator mismatch (`Ninja` vs `Visual Studio 17 2022`) | Added `CMakeCache.txt` parser: reads `CMAKE_GENERATOR:INTERNAL`, deletes cache if generator changes |
| `-- /m:1` not valid for Ninja/NMake | Replaced with `--parallel %NUMBER_OF_PROCESSORS%` (CMake ≥ 3.12, generator-agnostic) |
| vgre-token not recognized before build | CLI tools now installed BEFORE the cmake/build step; current session PATH also updated |
| `-ExecutionPolicy` missing on nested PowerShell calls | All `powershell -File` invocations now pass `-ExecutionPolicy Bypass` |

**Generator priority**: Ninja (from `%LLVM_DIR%\bin` or `PATH`) → VS MSBuild → NMake.

### 13.2 macOS — vgre_sync.sh improvements

| Problem | Fix |
|---|---|
| `libomp.dylib` not found on Homebrew Apple Silicon | Added `/opt/homebrew/lib/libomp.dylib` and `/opt/homebrew/opt/libomp/lib/libomp.dylib` to OMP search paths |
| Flutter download URL wrong on macOS | macOS now fetches `.zip`; Linux fetches `.tar.xz` |
| `ldd` not available on macOS for ASAN check | Uses `otool -L` on macOS; `ldd` on Linux |
| GCC symlink creation on macOS fails | `ln -sf gcc-* gcc` block guarded to Linux only |
| vgre-worker wrapper missing `DYLD_LIBRARY_PATH` | macOS deployment block creates wrapper script with both `LD_LIBRARY_PATH` and `DYLD_LIBRARY_PATH` |

### 13.3 CMakeLists.txt fixes

| Problem | Fix |
|---|---|
| `string(REGEX REPLACE "\\P" …)` crash with backslash paths | Added `string(REPLACE "\\" "/" _dia_path "${_dia_path}")` before REGEX on Windows DIA SDK path |
| `Could NOT find LAPACK` on Windows (default ON) | `VGRE_USE_LAPACK` now defaults OFF on `WIN32`; ON on Linux/macOS |
| DIA SDK Strategy 2 missed newer VS versions | Added VS 2025/18/17/16/15 and Preview edition to enumeration |
| `INT_MAX` undeclared in tcp_cluster_manager.cpp | Added `#include <climits>` |

### 13.4 vgre-start.sh

`vgre-start.sh` now sets both `LD_LIBRARY_PATH` (Linux) and `DYLD_LIBRARY_PATH` (macOS) before starting the master process.

---

## 14. Cluster / GPU / Mesh Improvements (2026-05-17)

**Status**: **DONE** — master/worker cluster is fully functioning across all platforms; supports both CPU and GPU backends; full-mesh topology auto-enabled.

### 14.1 Worker CLI — --threads and --no-gpu flags

File: `src/advanced/vgre_worker_cli.cpp`

| Flag | Behaviour |
|---|---|
| `--threads <n>` | Calls `vgre_set_config("VGRE_WORKER_THREADS", n)` to pin OpenMP thread count |
| `--no-gpu` | Skips both GPUPassthrough and IGPUOpenCLExecutor probes; worker runs CPU-only |

On startup the worker probes GPUPassthrough (dGPU) and then IGPUOpenCLExecutor (iGPU/OpenCL), logs the detected backend, and sets `has_igpu` in `CapabilityPacket` accordingly.

### 14.2 GPU dispatch cascade in dispatch_manager_remote.cpp

File: `src/advanced/tcp_cluster/dispatch_manager_remote.cpp`

Dispatch order for every remote kernel:
1. `GPUPassthrough::launchOnGPU()` — uses real NVIDIA dGPU if present
2. `IGPUOpenCLExecutor::execute()` — uses integrated GPU / OpenCL if available
3. `RuntimeEngine::launchKernel()` — CPU JIT fallback (always present)

### 14.3 GPU-aware worker selection

File: `src/advanced/tcp_cluster/tcp_cluster_manager.cpp`

New method `getGpuCapableWorker()`:
- Scans worker capability map for `has_igpu == true`
- Among GPU-capable workers, picks the one with the lowest `in_flight_kernels` count
- Falls back to `getFirstActiveWorker()` if no GPU workers available

`hybrid_compute_manager_workload.cpp` calls `getGpuCapableWorker()` before falling back to `getFirstActiveWorker()` for `REMOTE_NODE` dispatch.

### 14.4 Auto-mesh topology

File: `src/advanced/tcp_cluster/server_packet_dispatch.cpp`

When the second `CAPABILITY` packet is received (i.e., at least 2 workers are connected), `mesh_topology_enabled_` is set to `true` automatically. Previously this required the environment variable `VGRE_ENABLE_MESH_TOPOLOGY=1`.

### 14.5 test_phase3_cluster — RUN_SERIAL

`tests/CMakeLists.txt` now sets `RUN_SERIAL TRUE` on `test_phase3_cluster`. Under `ctest -j$(nproc)` the cluster test was CPU-starved and timing out; running it serially eliminates the contention.

---

## 15. RDMA Transport Improvements (2026-05-17)

**Status**: **DONE** — RDMA is real and fully functioning on Linux InfiniBand/RoCE; all previously identified bugs fixed.

### 15.1 Issues found and fixed

| Bug | Fix | File |
|---|---|---|
| Per-call `ibv_reg_mr` (extremely expensive — one MR per send) | Registers full source MR once per `rdmaWriteToRemote` call for the whole buffer | `src/advanced/rdma_transport.cpp` |
| Spin-wait `pollCompletion` burns 100% CPU | First 1000 spins use `__builtin_ia32_pause()` / ARM `yield`; after 1000 spins uses `std::this_thread::yield()` | `src/advanced/rdma_transport.cpp` |
| No chunking — `rdmaWriteToRemote` limited to bounce buffer size | Chunked loop: each iteration writes one `bounceCapacity()`-sized chunk via RDMA then sends `DATA_HEADER_RDMA` | `src/advanced/tcp_cluster/memory_sync_manager.cpp` |
| Receiver ignored `dst_offset` — always wrote to allocation base | `DATA_HEADER_RDMA` handler now uses `rdmaPkt.chunk_size` and `rdmaPkt.dst_offset` for correct multi-chunk reassembly | `src/advanced/tcp_cluster/client_packet_dispatch.cpp` |
| ARM cache coherency — no fence after bounce copy | Added `std::atomic_thread_fence(std::memory_order_acquire)` after memcpy from bounce buffer | `src/advanced/tcp_cluster/client_packet_dispatch.cpp` |

### 15.2 DataHeaderRDMAPacket protocol change

`include/vgre/advanced/tcp_cluster_protocol.h`:

```
Before: uint64_t size;          // total transfer size
After:  uint32_t chunk_size;    // bytes in this bounce-buffer chunk
        uint32_t dst_offset;    // byte offset into target allocation
```

Total struct size remains 16 bytes. `dst_offset == 0` means first (or only) chunk; receiver allocates on first chunk only.

### 15.3 Platforms

| Platform | RDMA Support | Notes |
|---|---|---|
| Linux (InfiniBand / RoCE) | Full | `libibverbs`, QP state machine, MR registration, bounce buffer |
| Linux (no RDMA hardware) | TCP fallback | `VGRE_ENABLE_RDMA=OFF` (default) uses `DATA_HEADER` + `DATA_BODY` |
| Windows | TCP only | `libibverbs` not available; compile guard `#ifdef VGRE_ENABLE_RDMA` |
| macOS | TCP only | Same as Windows |

---

## 16. vgre-token & CLI Tools (2026-05-17)

**Status**: **DONE** — `vgre-token` can be used from any terminal immediately after `vgre_sync.bat` or `vgre_sync.sh` runs, before the build completes.

### 16.1 vgre-token install subcommand (scripts/vgre-token.ps1)

New `install` subcommand:
- Copies `vgre-token.ps1` + `vgre-token.bat` to `%LOCALAPPDATA%\VGRE\scripts\`
- Adds the directory to the **User**-scope PATH (persistent across reboots)
- Also adds to `$env:PATH` in the **current session** so the command works immediately without a restart

### 16.2 Install-VGRETools.ps1 (scripts/Install-VGRETools.ps1)

New standalone installer that requires no prior build:
- Copies `vgre-token.ps1`, `Setup-VGRECluster.ps1`, `Start-VGRE.ps1`, `vgre_env.ps1`, `Install-VGRETools.ps1`
- Creates `.bat` launcher stubs for `vgre-start` and `Setup-VGRECluster`
- Updates both User PATH and current-session PATH

### 16.3 Before-build installation in vgre_sync.bat

CLI tools are installed at the start of `vgre_sync.bat` before the cmake configure and build steps. If the build fails for any reason the user still has `vgre-token` available on the PATH.

---

---

## 17. Eleventh Pass — PTX Texture Metadata, cuSPARSE UMFPACK, NCCL Tree (2026-05-17)

**Status**: **DONE** — 117/117 tests pass.

### 17.1 PTX txq — real texture dimension queries

Previously `txq.width/height/depth/channels` returned hardcoded constants (1024/1024/1/1).

| Change | Files |
|---|---|
| `TextureManager::TextureInfo` struct + `getTextureInfo(id, out)` method | `include/vgre/core/texture_manager.h` |
| `getTextureInfo` implementation (channels = elementSize/sizeof(float) for FLOAT32) | `src/core/texture/texture_manager_sampling.cpp` |
| `vgre_txq_{width,height,depth,channels}(uint64_t tex)` extern C | `src/compiler/texture_builtins.cpp`, `include/vgre/compiler/cpu_cuda_env.h` |
| PTX translator: `txq.*` handlers call real functions | `src/compiler/ptx/ptx_texture_ops.cpp` |

### 17.2 PTX tex v2/v4 — per-channel fetch for packed vector textures

Previously all channels of `tex.*.v2/v4` were copies of channel 0 (scalar replicated).

| Change | Files |
|---|---|
| `readElementChannel(tex, linearIndex, channel)` — reads nth float within packed element | `src/core/texture/texture_manager_sampling.cpp` |
| `sampleTexelChan(tex, x, y, z, channel)` — coordinate-to-channel aware sample | `src/core/texture/texture_manager_sampling.cpp` |
| `tex1DChan / tex2DChan / tex3DChan` public methods with bilinear interpolation per-channel | `include/vgre/core/texture_manager.h`, `src/core/texture/texture_manager_sampling.cpp` |
| `vgre_tex{1D,2D,3D}_chan_f32(tex, …, channel)` extern C | `src/compiler/texture_builtins.cpp`, `include/vgre/compiler/cpu_cuda_env.h` |
| PTX translator: v2 handlers emit 2 per-channel calls; v4 handlers emit 4 per-channel calls | `src/compiler/ptx/ptx_texture_ops.cpp` |
| `tld4.2d.v4` updated to fetch four distinct 2×2 footprint texels | `src/compiler/ptx/ptx_texture_ops.cpp` |

### 17.3 cuSPARSE optional UMFPACK (full fill-in sparse LU)

| Change | Files |
|---|---|
| `option(VGRE_USE_UMFPACK …)` + pkg-config/manual UMFPACK detection | `CMakeLists.txt` (§8b) |
| `VGRE_HAS_UMFPACK` compile definition + library linking via `vgre_apply_optional_deps()` | `CMakeLists.txt` |
| `umfpack_lu_inplace<T>()` — symbolic+numeric factorization via UMFPACK, scatter L/U back into CSR sparsity | `src/api/cusparse/cusparse_factorization.cpp` |
| `cusparseScsrilu02` / `cusparseDcsrilu02` route through UMFPACK when available; ILU(0) fallback otherwise | `src/api/cusparse/cusparse_factorization.cpp` |

**Usage**: `cmake -DVGRE_USE_UMFPACK=ON ..` (requires `libumfpack-dev` / SuiteSparse).

### 17.4 NCCL binary tree AllReduce for medium tensors

Three-tier algorithm selection in `ncclAllReduce`:

| Tensor size | Algorithm | Notes |
|---|---|---|
| ≤ 64 KB | Flat barrier (root reduces all) | O(1) sync rounds, lowest latency |
| 64 KB – 1 MB | Binary tree reduce (`tree_allreduce`) | log₂(N) rounds; each level distributes reduction across half the ranks |
| > 1 MB | Ring allreduce (`ring_allreduce`) | Bandwidth-optimal chunk-pipelined |

Binary tree reduce implementation: bottom-up deposit-and-reduce per log₂(N) rounds using existing shared-state barriers; broadcast phase fans result back down. File: `src/api/nccl/nccl_collectives.cpp`.

---

*Last updated: 2026-05-17 (eleventh pass). 117/117 tests pass. PTX txq real metadata, tex v2/v4 per-channel, cuSPARSE UMFPACK optional, NCCL three-tier AllReduce.*
