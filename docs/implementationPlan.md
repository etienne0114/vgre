# VGRE Implementation Plan

**Version**: 1.0.0  
**Date**: 2026-05-12  
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

### 1.2 Directory Conventions (following `tcp_cluster/`, `memory/`, `token/`)

```
src/api/cudart/          # CUDART shim split by concern
src/api/cuda_driver/     # Driver API shim split by concern
src/api/cublas/          # cuBLAS backend split by level
src/api/cudnn/           # cuDNN backend split by layer type
src/api/nccl/            # NCCL backend split by collective type
src/api/cufft/           # NEW: cuFFT shim (entirely missing library)
src/api/curand/          # NEW: cuRAND shim
src/api/cusolver/        # NEW: cuSOLVER shim
src/api/cusparse/        # NEW: cuSPARSE shim
src/api/cublasLt/        # NEW: cuBLASLt shim
src/compiler/ptx/        # PTX translator split by architecture
src/core/graph/          # Graph manager split by node type
src/core/texture/        # Texture/surface manager split by dimension
src/compiler/cuda_device_libs/  # FP16, BF16, cooperative_groups headers
```

### 1.3 Naming Convention

| Pattern | Example | Purpose |
|---|---|---|
| `*_shim.cpp` | `cudart_shim_stream.cpp` | API interception / forwarding |
| `*_backend.cpp` | `cudnn_convolution_backend.cpp` | Actual compute implementation |
| `*_nodes.cpp` | `graph_manager_kernel_nodes.cpp` | Graph node-type implementations |
| `*_ops.cpp` | `nccl_p2p_ops.cpp` | NCCL point-to-point operations |
| `*_ampere.cpp` | `ptx_mma_ampere.cpp` | Architecture-specific PTX backend |
| `*_linux.cpp` | `token_manager_linux.cpp` | Platform-specific implementation |

---

## 2. Phase Roadmap

| Phase | Focus | Est. Duration | Risk |
|---|---|---|---|
| **P1** | Critical CUDA Runtime gaps (stream sync, symbols, graphs kernel node) | 2–3 weeks | Low — patterns already exist |
| **P2** | cuBLAS Level-2/Level-3 backfill | 2–3 weeks | Low — OpenBLAS delegation pattern |
| **P3** | cuDNN backward passes + training | 3–4 weeks | Medium — needs reference kernels |
| **P4** | Missing libraries (stub → functional) | 4–6 weeks | Medium — large surface area |
| **P5** | CUDA Driver API expansion | 3–4 weeks | Low — driver shim pattern exists |
| **P6** | PTX ISA expansion + texture/surface PTX | 3–4 weeks | Medium — parser complexity |
| **P7** | NCCL p2p + advanced collectives | 2 weeks | Low — networking layer solid |
| **P8** | Cooperative groups + device-side libraries | 2 weeks | Low — `cpu_cuda_env.h` pattern |
| **P9** | Deployment (K8s Device Plugin, SLURM GRES) | 2 weeks | Low — not runtime-critical |

---

## 3. Detailed Implementation Plans

---

### 3.1 Phase 1 — Critical CUDA Runtime Gaps

**Goal**: Unblock PyTorch/TensorFlow basic inference graphs.

#### 3.1.1 Stream Synchronization & Queries

**Status**: **DONE** — all functions implemented in `src/api/cudart/cudart_shim_stream.cpp` and `src/api/cudart/cudart_shim.cpp`.

**Implemented**: `cudaStreamWaitEvent`, `cudaEventQuery`, `cudaStreamAddCallback`, `cudaLaunchHostFunc`, `cudaGetErrorName`, `cudaGetErrorString`

**Files**:
- `src/api/cudart/cudart_shim_stream.cpp` — stream sync, event query, callbacks, host func
- `src/api/cudart/cudart_shim.cpp` — error strings table

---

#### 3.1.2 Memory & Symbols

**Status**: **DONE** — all functions implemented in `src/api/cudart/cudart_shim.cpp` and `src/api/cudart/cudart_shim_memset_nd.cpp`.

**Implemented**: `cudaMemcpyToSymbol`, `cudaMemcpyToSymbolAsync`, `cudaMemcpyFromSymbol`, `cudaMemcpyFromSymbolAsync`, `cudaMallocArray`, `cudaMalloc3DArray`, `cudaMalloc3D`, `cudaMemcpy3DAsync`, `cudaHostGetDevicePointer`, `cudaHostGetFlags`, `cudaArrayGetInfo`, `cudaArrayDestroy`, `cudaPointerGetAttributes`, `cudaMemset2D`, `cudaMemset3D`, `cudaMemset2DAsync`, `cudaMemset3DAsync`

**Files**:
- `src/api/cudart/cudart_shim.cpp` — symbols, arrays, pointer attributes
- `src/api/cudart/cudart_shim_memset_nd.cpp` — 2D/3D memset variants

**Modify**:
- `src/core/memory/memory_manager.cpp` — add `symbolLookup_` map for `__device__` constant symbols.

---

#### 3.1.3 CUDA Graphs — Kernel & Essential Node Types

**Status**: **DONE** — all node types implemented in `src/core/graph/graph_manager_extended_nodes.cpp` and `src/api/cudart/cudart_shim_graph_nodes.cpp`.

**Implemented**: `cudaGraphAddKernelNode`, `cudaGraphAddMemsetNode`, `cudaGraphAddHostNode`, `cudaGraphAddChildGraphNode`, `cudaGraphAddEmptyNode`, `cudaGraphAddEventRecordNode`, `cudaGraphAddEventWaitNode`, `cudaGraphAddMemAllocNode`, `cudaGraphAddMemFreeNode`

**Files**:
- `src/core/graph/graph_manager_extended_nodes.cpp` — kernel, memset, host, child, event, memalloc nodes
- `src/api/cudart/cudart_shim_graph_nodes.cpp` — CUDART shim wrappers for all node types

**Modify**:
- `include/vgre/core/graph_manager.h` — declare new node-type add/update methods.

**Key design decisions**:
- Kernel node: `GraphManager::addKernelNode` already exists internally; expose it via CUDART shim.
- Memset node: Reuse `MemoryManager::memsetD8/D16/D32` paths with graph node wrapper.
- Host node: Store `std::function<void()>` in node; execute during graph replay via `StreamScheduler`.
- Mem-alloc/free node: Use stream-ordered pool allocator (`cudaMallocFromPoolAsync` path) tied to graph lifetime.

---

#### 3.1.4 Graph Node Introspection & Exec Mutation

**Status**: **DONE** — introspection and exec mutation implemented in `src/core/graph/graph_manager_introspection.cpp` and `src/core/graph/graph_manager_exec_update.cpp`.

**Implemented**: `cudaGraphKernelNodeSetParams`, `cudaGraphKernelNodeGetParams`, `cudaGraphMemcpyNodeGetParams`, `cudaGraphMemsetNodeGetParams`, `cudaGraphMemsetNodeSetParams`, `cudaGraphHostNodeGetParams`, `cudaGraphHostNodeSetParams`, `cudaGraphNodeGetType`, `cudaGraphGetNodes`, `cudaGraphGetRootNodes`, `cudaGraphGetEdges`, `cudaGraphNodeGetDependencies`, `cudaGraphNodeGetDependentNodes`, `cudaGraphExecKernelNodeSetParams`, `cudaGraphExecMemcpyNodeSetParams`, `cudaGraphExecMemsetNodeSetParams`, `cudaGraphExecHostNodeSetParams`, `cudaGraphExecChildGraphNodeSetParams`, `cudaGraphExecEventRecordNodeSetEvent`, `cudaGraphExecEventWaitNodeSetEvent`, `cudaGraphInstantiateWithFlags`, `cudaGraphInstantiateWithParams`, `cudaGraphExecGetFlags`, `cudaGraphUpload`, `cudaGraphNodeSetEnabled`, `cudaGraphNodeGetEnabled`, `cudaGraphExecNodeSetParams`, `cudaGraphKernelNodeCopyAttributes`

**Files**:
- `src/core/graph/graph_manager_introspection.cpp` — get nodes, edges, dependencies, params
- `src/core/graph/graph_manager_exec_update.cpp` — exec mutation, instantiate with flags/params
- `src/api/cudart/cudart_shim_graph_exec.cpp` — CUDART shim wrappers for exec APIs

**Still missing**:
- `cudaGraphExecUpdate_v2` with per-node error reporting → Phase 10.4

---

#### 3.1.5 Stream Capture Introspection

**Status**: **DONE** — implemented in `src/api/cudart/cudart_shim_capture.cpp`.

**Implemented**: `cudaStreamIsCapturing`, `cudaStreamGetCaptureInfo`, `cudaStreamGetCaptureInfo_v2`, `cudaThreadExchangeStreamCaptureMode`, `cudaStreamUpdateCaptureDependencies`, `cudaStreamCopyAttributes`

**File**: `src/api/cudart/cudart_shim_capture.cpp`

---

#### 3.1.6 Texture & Surface — CUDART Object APIs

**Status**: **DONE** — implemented in `src/api/cudart/cudart_shim_texture_objects.cpp`.

**Implemented**: `cudaCreateTextureObject`, `cudaDestroyTextureObject`, `cudaGetTextureObjectResourceDesc`, `cudaGetTextureObjectTextureDesc`, `cudaGetTextureObjectResourceViewDesc`, `cudaCreateSurfaceObject`, `cudaDestroySurfaceObject`, `cudaGetSurfaceObjectResourceDesc`, `cudaGetTextureReference`, `cudaGetSurfaceReference`, `cudaBindTexture`, `cudaUnbindTexture`, `cudaBindTextureToArray`, `cudaBindTexture2D`, `cudaBindSurfaceToArray`

**Files**:
- `src/api/cudart/cudart_shim_texture_objects.cpp` — CUDART texture/surface object APIs
- `src/core/texture_manager.cpp` — existing texture manager (still monolithic, future split candidate)

---

#### 3.1.7 CUDA Runtime — Device & Function Attributes

**Status**: **DONE** — implemented in `src/api/cudart/cudart_shim_device_attrs.cpp`.

**Implemented**: `cudaFuncGetAttributes`, `cudaFuncSetCacheConfig`, `cudaFuncSetSharedMemConfig`, `cudaDeviceGetLimit`, `cudaDeviceSetLimit`, `cudaDeviceGetCacheConfig`, `cudaDeviceSetCacheConfig`, `cudaDeviceGetSharedMemConfig`, `cudaDeviceSetSharedMemConfig`, `cudaChooseDevice`, `cudaThreadExit`, `cudaThreadSynchronize`, `cudaThreadSetLimit`, `cudaThreadGetLimit`, `cudaDeviceGetP2PAttribute`, `cudaDeviceFlushGPUDirectRDMAWrites`, `cudaDeviceGetGraphMemAttribute`, `cudaDeviceSetGraphMemAttribute`, `cudaLaunchKernelExC`, `cudaLaunchConfig`

**File**: `src/api/cudart/cudart_shim_device_attrs.cpp`

**Key design decisions**:
- `cudaFuncGetAttributes`: Query `KernelIR` for register count, shared memory, and block-size limits; populate `cudaFuncAttributes`.
- `cudaDeviceGetLimit/SetLimit`: Track `stackSize`, `printfFifoSize`, `mallocHeapSize` in `DeviceProperties`.
- `cudaThread*`: Legacy 1.x APIs — implement as thin wrappers around existing 4.x device APIs.
- `cudaLaunchKernelExC`: Extended launch with `cudaLaunchConfig_t` — forward to existing launch path with extra attributes.

---

#### 3.1.8 CUDA Graphs — Dependencies & User Objects

**Status**: **DONE** — implemented in `src/core/graph/graph_manager_dependencies.cpp` and `src/api/cudart/cudart_shim_graph_user_objects.cpp`.

**Implemented**: `cudaGraphAddDependencies`, `cudaGraphRemoveDependencies`, `cudaGraphRetainUserObject`, `cudaGraphReleaseUserObject`, `cudaUserObjectCreate`, `cudaUserObjectRetain`, `cudaUserObjectRelease`, `cudaGraphNodeFindInClone`, `cudaGraphDebugDotPrint`

**Files**:
- `src/core/graph/graph_manager_dependencies.cpp` — dependency add/remove
- `src/api/cudart/cudart_shim_graph_user_objects.cpp` — user object CUDART shim wrappers

**Key design decisions**:
- Dependencies: `GraphManager` already tracks adjacency lists; `AddDependencies` inserts edges, `RemoveDependencies` deletes them with cycle check.
- User objects: Reference-counted `std::shared_ptr<void>` with custom deleter, keyed by graph node ID.

---

#### 3.1.9 External Memory & Semaphore — CUDART

**Status**: **DONE** — implemented in `src/api/cudart/cudart_shim_external_memory.cpp`.

**Implemented**: `cudaImportExternalMemory`, `cudaDestroyExternalMemory`, `cudaExternalMemoryGetMappedBuffer`, `cudaExternalMemoryGetMappedMipmappedArray`, `cudaExternalSemaphoreGetSignalNodeParams`, `cudaExternalSemaphoreGetWaitNodeParams`

**File**: `src/api/cudart/cudart_shim_external_memory.cpp`

**Key design decisions**:
- These are the CUDART counterparts to driver `cuExternalMemory*` / `cuExternalSemaphore*` APIs. Reuse the existing external-semaphore infrastructure in `src/api/cuda_external_semaphore.cpp`.
- `cudaImportExternalMemory` maps to `cuImportExternalMemory` + handle wrapping.

---

### 3.2 Phase 2 — cuBLAS Level-2 / Level-3 Backfill

**Goal**: Enable PyTorch `torch.linalg.solve`, `torch.mm` with non-GEMM paths.

**Status**: **DONE** — all Level-1/2/3 BLAS, pointer mode, atomics mode implemented in `src/api/cublas/`.

#### 3.2.1 Level-1 BLAS Completion

**Implemented**: `cublasScopy`, `cublasDcopy`, `cublasSswap`, `cublasDswap`, `cublasSrot`, `cublasDrot`, `cublasSrotm`, `cublasSrotmg`, `cublasSrotg`, `cublasSasum`, `cublasDasum`, `cublasIsamax`, `cublasIdamax`, `cublasDnrm2`, `cublasDscal`, `cublasGetPointerMode`, `cublasSetPointerMode`, `cublasGetAtomicsMode`, `cublasSetAtomicsMode`

**Files**:
- `src/api/cublas/cublas_level1.cpp` — all Level-1 routines
- `src/api/cublas/cublas_core.cpp` — pointer mode, atomics mode APIs

---

#### 3.2.2 Level-2 BLAS

**Implemented**: `cublasStrsv`, `cublasDtrsv`, `cublasSger`, `cublasDger`, `cublasSsymv`, `cublasDsymv`, `cublasSgbmv`, `cublasDgbmv`, `cublasSsyr`, `cublasDsyr`, `cublasSsyr2`, `cublasDsyr2`, plus `cublasStrmv`/`cublasDtrmv`

**File**: `src/api/cublas/cublas_level2.cpp`

**Key design decision**: Reference CBLAS when available, otherwise clean scalar reference loops.

---

#### 3.2.3 Level-3 BLAS

**Implemented**: `cublasSgemm`/`cublasDgemm`, `cublasStrsm`/`cublasDtrsm`, `cublasSsyrk`/`cublasDsyrk`, `cublasSsyr2k`/`cublasDsyr2k`, `cublasStrmm`/`cublasDtrmm`, `cublasSsymm`/`cublasDsymm`, `cublasHgemm`, `cublasGemmEx`, `cublasGemmBatchedEx`, `cublasGemmStridedBatchedEx`, batched SGEMM/DGEMM, strided batched variants

**File**: `src/api/cublas/cublas_level3.cpp`

**Key design decision**: Reference CBLAS/OpenBLAS when available; custom reference kernels for TRSM, SYRK, SYR2K, TRMM, SYMM. HGEMM routes through FP32 with half↔float conversion. GEMMEx handles INT8→FP32 dequantization.

---

### 3.3 Phase 3 — cuDNN Backward Passes + Training

**Goal**: Enable PyTorch training (backward pass, BN training, dropout).

#### 3.3.1 Convolution Backward

**Status**: **DONE** — implemented in `src/api/cudnn/cudnn_convolution.cpp`.

**Implemented**: `cudnnConvolutionBackwardData`, `cudnnConvolutionBackwardFilter`, `cudnnConvolutionBackwardBias`

**File**: `src/api/cudnn/cudnn_convolution.cpp`

**Key design decision**: Backward data = transposed convolution (im2col + GEMM with flipped filters). Backward filter = im2col + GEMM with input activations and output gradients.

---

#### 3.3.2 Batch Normalization Training

**Status**: **DONE** — implemented in `src/api/cudnn/cudnn_batchnorm.cpp`.

**Implemented**: `cudnnBatchNormalizationForwardTraining`, `cudnnBatchNormalizationBackward`

**File**: `src/api/cudnn/cudnn_batchnorm.cpp`

---

#### 3.3.3 Other Backward Passes

**Status**: **DONE** — implemented in their respective feature files.

**Implemented**: `cudnnActivationBackward` (`cudnn_activation.cpp`), `cudnnSoftmaxBackward` (`cudnn_softmax.cpp`), `cudnnPoolingBackward` (`cudnn_pooling.cpp`)

---

#### 3.3.4 Advanced Layers

**Status**: Mostly **DONE** — dropout, RNN, attention, tensor ops, CTC loss implemented. LRN still missing.

**Implemented**: `cudnnDropoutForward/Backward` (`cudnn_dropout.cpp`), `cudnnRNNForward/Backward` (`cudnn_rnn.cpp`), `cudnnMultiHeadAttnForward/Backward` (`cudnn_attention.cpp`), `cudnnTransformTensor`, `cudnnOpTensor`, `cudnnReduceTensor` (`cudnn_tensor_ops.cpp`), `cudnnCTCLoss` (`cudnn_ctc_loss.cpp`)

**Still missing**: `cudnnDivisiveNormalizationForward/Backward`, `cudnnLRNCrossChannelForward/Backward`

**Files**:
- `src/api/cudnn/cudnn_dropout.cpp`
- `src/api/cudnn/cudnn_rnn.cpp`
- `src/api/cudnn/cudnn_attention.cpp`
- `src/api/cudnn/cudnn_tensor_ops.cpp`

---

#### 3.3.5 cuDNN Backend API (v8+)

**Status**: **DONE** (minimal stub) — descriptor lifecycle (`Create/Destroy/SetAttribute/GetAttribute/Finalize`), `Initialize`, `Populate` implemented. `Execute` returns `CUDNN_STATUS_NOT_SUPPORTED` until full descriptor-graph-to-legacy-path wiring is completed.

**New files**:
```
src/api/cudnn/cudnn_backend_api.cpp               (400–500 lines)
```

**Key design decisions**:
- The cuDNN v8+ Backend API replaces the legacy frontend (descriptors + `cudnn*Forward/Backward`). It uses a descriptor-graph pattern where nodes represent operations, tensors, and engines.
- Implementation strategy: Parse the descriptor graph at `cudnnBackendFinalize` time; translate to the existing legacy cuDNN shim paths (conv, pool, activation, etc.). This avoids reimplementing all math from scratch.
- If a Backend descriptor graph maps to an unimplemented operation (e.g., RNN attention), return `CUDNN_STATUS_NOT_SUPPORTED`.

---

### 3.4 Phase 4 — Missing Libraries (Stub → Functional)

**Goal**: Provide graceful degradation for dependent applications.

#### 3.4.1 cuFFT

**New files**:
```
include/vgre/api/cufft_shim.h                       (150–200 lines)
src/api/cufft/cufft_shim.cpp                      (400–500 lines)
src/api/cufft/cufft_plan.cpp                      (150–200 lines)
src/api/cufft/cufft_transform_1d.cpp              (200–300 lines)
src/api/cufft/cufft_transform_2d.cpp              (200–300 lines)
src/api/cufft/cufft_transform_3d.cpp              (200–300 lines)
```

**Implementation strategy**: Delegate 1D/2D/3D FFT to FFTW3 (if available) or a built-in reference Cooley-Tukey radix-2 implementation. Return `CUFFT_STATUS_NOT_SUPPORTED` for unsupported types (e.g., half-precision FFT).

---

#### 3.4.2 cuRAND

**New files**:
```
include/vgre/api/curand_shim.h                      (150–200 lines)
src/api/curand/curand_shim.cpp                    (300–400 lines)
src/api/curand/curand_host_api.cpp                (200–250 lines)  # host-side generation
src/api/curand/curand_device_api.cpp              (100–150 lines)  # device-side stubs
```

**Implementation strategy**: Host API uses `std::mt19937_64` or `PCG64` seeded from `/dev/urandom` / `BCryptGenRandom`. Device API returns `CURAND_STATUS_NOT_SUPPORTED` (no device-side RNG in CPU emulation).

---

#### 3.4.3 cuSOLVER

**New files**:
```
include/vgre/api/cusolver_shim.h                    (200–250 lines)
src/api/cusolver/cusolver_shim.cpp                (400–500 lines)
src/api/cusolver/cusolver_dense.cpp               (300–400 lines)  # QR, SVD, Cholesky, eigen
src/api/cusolver/cusolver_sparse.cpp              (200–300 lines)  # sparse factorization stubs
```

**Implementation strategy**: Delegate dense routines to LAPACK (via OpenBLAS/LAPACKE). Return `CUSOLVER_STATUS_NOT_SUPPORTED` for sparse paths until cuSPARSE is ready.

---

#### 3.4.4 cuSPARSE

**New files**:
```
include/vgre/api/cusparse_shim.h                  (200–250 lines)
src/api/cusparse/cusparse_shim.cpp                (400–500 lines)
src/api/cusparse/cusparse_spmv.cpp                (150–200 lines)  # SpMV
src/api/cusparse/cusparse_spmm.cpp                (150–200 lines)  # SpMM
src/api/cusparse/cusparse_conversions.cpp         (150–200 lines)  # CSR/CSC/COO format conv
```

**Implementation strategy**: CSR SpMV uses OpenMP-parallelized row loops. Dense→sparse conversions are memory copies with index computation.

---

#### 3.4.5 cuBLASLt

**New files**:
```
include/vgre/api/cublasLt_shim.h                  (150–200 lines)
src/api/cublasLt/cublasLt_shim.cpp                (250–350 lines)
src/api/cublasLt/cublasLt_matmul.cpp              (200–300 lines)
```

**Implementation strategy**: cuBLASLt is a lightweight GEMM API with fused epilogues (bias, ReLU, GELU). Delegate the GEMM to OpenBLAS; apply epilogues as a post-processing kernel.

---

### 3.5 Phase 5 — CUDA Driver API Expansion

**Goal**: Enable lower-level framework interop (JAX, TensorFlow XLA).

#### 3.5.1 Memory & Copies

**Status**: **DONE**

**Implemented**: `cuMemAllocManaged`, `cuMemHostAlloc`, `cuMemHostGetDevicePointer`, `cuMemHostRegister`, `cuMemHostUnregister`, `cuMemAllocPitch`, `cuMemcpy2D`, `cuMemcpy2DAsync`, `cuMemcpy3D`, `cuMemcpy3DAsync`, `cuMemcpyDtoDAsync`, `cuMemcpyDtoHAsync`, `cuMemcpyHtoDAsync`, `cuMemcpyAsync`, `cuMemcpyPeer`, `cuMemcpyPeerAsync`, `cuMemsetD8`, `cuMemsetD16`, `cuMemsetD32`, `cuMemsetD2D8`, `cuMemsetD2D16`, `cuMemsetD2D32`

**Files**:
- `src/api/cuda_driver/cuda_driver_memory.cpp` — `cuMemAllocManaged`, `cuMemHostAlloc`, `cuMemHostGetDevicePointer`, `cuMemHostRegister`, `cuMemHostUnregister`, `cuMemAllocPitch`
- `src/api/cuda_driver/cuda_driver_memcpy.cpp` — all 1D async, peer, 2D, 3D memcpy variants with full `CUDA_MEMCPY2D`/`CUDA_MEMCPY3D` parameter validation
- `src/api/cuda_driver/cuda_driver_memset.cpp` — `cuMemsetD8/D16/D32`, `cuMemsetD2D8/D2D16/D2D32` with real element-wise loops

---

#### 3.5.2 Streams & Events

**Status**: **DONE**

**Implemented**: `cuStreamAddCallback`, `cuStreamQuery`, `cuStreamGetFlags`, `cuStreamGetPriority`, `cuStreamGetId`, `cuStreamGetCtx`, `cuStreamIsCapturing`, `cuStreamGetCaptureInfo`, `cuStreamUpdateCaptureDependencies`, `cuEventQuery`

**File**: `src/api/cuda_driver/cuda_driver_stream_event.cpp`

**Notes**:
- `cuStreamGetId` returns the stream handle itself as the canonical ID.
- `cuStreamIsCapturing` / `cuStreamGetCaptureInfo` delegate to `RuntimeEngine` stream capture introspection.
- `cuStreamUpdateCaptureDependencies` maps to `RuntimeEngine::streamUpdateCaptureDependencies` with replace-mode support.

---

#### 3.5.3 Context & Device

**Status**: **DONE** (partial — NvSciSync not applicable on Linux)

**Implemented**: `cuCtxGetDevice`, `cuCtxGetFlags`, `cuCtxGetLimit`, `cuCtxSetLimit`, `cuCtxGetCacheConfig`, `cuCtxSetCacheConfig`, `cuCtxGetSharedMemConfig`, `cuCtxSetSharedMemConfig`, `cuCtxGetStreamPriorityRange`, `cuCtxGetId`, `cuCtxGetApiVersion`, `cuCtxPopCurrent`, `cuCtxPushCurrent`, `cuCtxAttach`, `cuCtxDetach`, `cuDeviceGetUuid`, `cuDeviceGetTexture1DLinearMaxWidth`, `cuDeviceGetP2PAttribute`, `cuDeviceGetGraphMemAttribute`, `cuDeviceSetGraphMemAttribute`, `cuDeviceFlushGPUDirectRDMAWrites`

**File**: `src/api/cuda_driver/cuda_driver_device_context.cpp`

**Note**: `cuCtxGetLimit`/`cuCtxSetLimit` use real per-device static storage (`g_ctxLimits`) with mutex protection. `cuDeviceGetUuid` generates deterministic UUIDs from device ordinal.

**Still missing**: `cuDeviceGetNvSciSyncAttributes` (platform-specific, not applicable on Linux)

---

#### 3.5.4 Graphs, External Resources, Profiler, Error Strings

**Status**: **DONE**

**Implemented**:
- **Graphs**: `cuGraphCreate`, `cuGraphClone`, `cuGraphDestroy`, `cuGraphInstantiate`, `cuGraphLaunch`, `cuGraphExecDestroy`, `cuGraphAddMemcpyNode`, `cuGraphAddMemsetNode` (placeholder), `cuGraphAddKernelNode` (placeholder), `cuStreamBeginCapture`, `cuStreamBeginCaptureToGraph`, `cuStreamEndCapture`
- **External**: `cuImportExternalMemory`, `cuDestroyExternalMemory`, `cuExternalMemoryGetMappedBuffer`, `cuExternalMemoryGetMappedMipmappedArray`, `cuImportExternalSemaphore`, `cuDestroyExternalSemaphore`, `cuSignalExternalSemaphoresAsync`, `cuWaitExternalSemaphoresAsync`
- **Profiler/Errors**: `cuProfilerStart`, `cuProfilerStop`, `cuGetErrorName`, `cuGetErrorString`

**Files**:
- `src/api/cuda_driver/cuda_driver_graph.cpp` — graph lifecycle, instantiation, execution, stream capture
- `src/api/cuda_driver/cuda_driver_external.cpp` — external memory (host alloc mapping) and semaphore (eventfd on Linux)
- `src/api/cuda_driver/cuda_driver_errors.cpp` — error strings and profiler control

**Notes**:
- Graph memset/kernel node addition return placeholder node IDs since RuntimeEngine does not yet expose dedicated APIs; this prevents caller crashes while maintaining API compatibility.
- External memory imports allocate equivalent host-side blocks (no actual DMA-capable VRAM in CPU model).
- External semaphores use Linux eventfd (opaque FD and timeline FD) with poll-based wait.

---

#### 3.5.5 Texture / Surface Reference Gaps (Driver)

**Status**: **DONE**

**Implemented**: `cuTexRefSetAddress2D`, `cuTexRefSetAddressMode`, `cuTexRefSetFilterMode`, `cuTexRefSetMaxAnisotropy`, `cuTexRefSetMipmapFilterMode`, `cuTexRefSetMipmapLevelBias`, `cuTexRefSetMipmapLevelClamp`, `cuTexRefSetBorderColor`, `cuSurfRefSetFormat`

**File**: `src/api/cuda_driver/cuda_driver_texture.cpp`

**Key design decisions**:
- `cuTexRefSetAddress2D`: Pitched 2D texture binding with `TextureManager::createTexture`.
- Filter modes mapped to `TextureFilterMode::POINT`/`LINEAR`.
- Mipmap fields stored in `CUtexref_st` since `TextureDescriptor` has limited mipmap support.
- `cuSurfRefSetFormat`: No-op since VGRE surface references reuse texture format mapping.

---

### 3.6 Phase 6 — PTX ISA Expansion

**Goal**: Unblock more kernels (texture sampling, shared atomics, precise FP variants).

#### 3.6.1 Texture / Surface Instructions

**Status**: **DONE**

**Implemented PTX**:
- `tex.1d.f32.s32/f32`, `tex.1d.v2.f32.s32/f32`, `tex.1d.v4.f32.s32/f32`
- `tex.2d.f32.f32`, `tex.2d.v2.f32.f32`, `tex.2d.v4.f32.f32`
- `tex.3d.f32.f32`, `tex.3d.v2.f32.f32`, `tex.3d.v4.f32.f32`
- `tld4.2d.v4.f32.f32`
- `txq.width.u32`, `txq.height.u32`, `txq.depth.u32`, `txq.channels.u32`
- `suld.2d.f32`, `suld.2d.v2.f32`, `suld.2d.v4.f32`
- `sust.2d.f32`, `sust.2d.v2.f32`, `sust.2d.v4.f32`

**File**: `src/compiler/ptx/ptx_texture_ops.cpp`

**Key design decisions**:
- Texture instructions map to existing `vgre_tex1D_f32` / `vgre_tex2D_f32` / `vgre_tex3D_f32` builtins.
- VGRE TextureManager currently supports single-channel float textures; v2/v4 vector returns replicate the scalar sample to all channels.
- Surface load/store map to `vgre_surf2Dread_f32` / `vgre_surf2Dwrite_f32`.
- Texture queries (`txq`) return conservative defaults (1024×1024×1, 1 channel) since VGRE does not expose texture metadata queries.
- `splitOperands` was extended to handle `{}` braces for potential vector operand grouping.

---

#### 3.6.2 Shared-Memory Atomics

**Status**: **DONE**

**Implemented PTX**:
- `atom.shared.add.s32/u32/u64/f32/f64`
- `atom.shared.cas.b32/b64`
- `atom.shared.exch.b32/b64`
- `atom.shared.max.s32/u32`, `atom.shared.min.s32/u32`
- `atom.shared.and/or/xor.b32`
- `atom.shared.inc/dec.u32`

**File**: `src/compiler/ptx/ptx_shared_atomics.cpp`

**Design**: Shared atomics map to the same GCC/Clang `__atomic_*` builtins as global atomics. In VGRE's CPU model, `__shared__` memory is local host memory, so sequential-consistency atomics work correctly.

---

#### 3.6.3 Conversion & Precision Variants

**Status**: **DONE**

**Implemented PTX**:
- All missing `cvt.*` rounding modes: `rn/rz/rm/rp` for `f32/f64 → s32/u32`, `f64 → f32`, `f32 → f64`, `f32 → f16`, `f16 → f32`, `s64/u64 ↔ f32/f64`, and saturating variants.
- `sqrt.rn/rz/rm/rp.f32` (maps to `__builtin_sqrtf`)
- FP16 vector loads/stores: `ld.global.v2.f16`, `ld.global.v4.f16`, `st.global.v2.f16`, `st.global.v4.f16`
- Cooperative group primitives: `match.sync.eq.b32/b64/lt.b32/b64`, `elect.sync`
- Grid synchronization: `grid.sync` (maps to `vgre_jit_syncgrid`), `griddepcontrol.launch_dependents`, `griddepcontrol.wait`

**File**: `src/compiler/ptx/ptx_conversion.cpp`

**Notes**:
- `rcp.rn.f32`, `div.rn.f32/f64` were already present in `ptx_translator_map.cpp`.
- `match.sync` / `elect.sync` return identity values in serial CPU model (full mask / thread 0 elected).
- `grid.sync` delegates to the existing `vgre_jit_syncgrid()` cooperative-grid barrier.

---

#### 3.6.4 Hopper / Blackwell Extensions

**Status**: **DONE**

**Implemented**:
- `cp.async.bulk.tensor.3d/4d/5d.global.shared::cta.bulk_group` → `vgre_tma_load_3d/4d/5d` helpers
- `cp.reduce.async.add.f32/f64`, `cp.reduce.async.min.f32`, `cp.reduce.async.max.f32` → `vgre_cp_reduce_async_*` atomic helpers
- `tcgen05.mma.cta_group::1.m64n256k16.f32.bf16.bf16`, `m64n128k16`, `m64n256k16.f32.f16.f16`, `m64n128k16.f16`, `m64n256k8.tf32`, `m128n256k16.bf16` → `vgre_tcgen05_*` helpers (delegate to existing wgmma GEMM math)
- `match.sync`, `elect.sync`, `grid.sync`, `griddepcontrol` — see 3.6.3 above.

**Already present**: `cp.async.bulk.tensor.1d/2d` and `wgmma.mma_async` variants are in `ptx_translator_map.cpp`.

**Files modified**:
- `src/compiler/ptx/ptx_conversion.cpp` — added PTX mappings for TMA 3D/4D/5D, cp.reduce.async, tcgen05
- `include/vgre/compiler/wmma_emulation.h` — added `vgre_tma_load_3d/4d/5d`, `vgre_cp_reduce_async_*`, `vgre_tcgen05_*` helpers

---

### 3.7 Phase 7 — NCCL Point-to-Point & Advanced Collectives

**Status**: **DONE** (single-node; multi-node TCPCluster integration is future work)

**Implemented**: `ncclSend`, `ncclRecv`, `ncclAllToAll`, `ncclGather`, `ncclScatter`

**File**: `src/api/nccl_shim.cpp` (added to existing shim)

**Key design decisions**:
- Single-node shared-memory: all operations use the existing `NcclGroupState` two-phase barrier with `p2p_slots` for point-to-point buffer exchange.
- `ncclSend` writes into the destination rank's slot; `ncclRecv` reads from its own slot after barrier.
- `ncclAllToAll` deposits `count`-sized chunks for every peer, then copies the aggregated slot into `recvbuff`.
- `ncclGather` uses the existing `sendbufs` array; root concatenates all chunks into `recvbuff`.
- `ncclScatter` uses `root_sendbuf`; each rank copies its slice after barrier.
- Multi-node TCPCluster routing is noted as future work (Phase 7 extended) — requires `TCPClusterManager::sendToRank` / `recvFromRank` APIs.

---

### 3.8 Phase 8 — Cooperative Groups & Device-Side Libraries

**Status**: **DONE**

**Implemented**:
- **Cooperative Groups**: `thread_block`, `coalesced_group`, `thread_block_tile<4/8/16/32>`, `grid_group`, `multi_grid_group`
- **Member functions**: `sync()`, `size()`, `thread_rank()`, `group_index()`, `thread_index()`, `shfl()`, `shfl_up()`, `shfl_down()`, `shfl_xor()`
- **Algorithms**: `reduce()`, `reduce_sum()`, `reduce_min()`, `reduce_max()`, `partition()`, `match_any()`, `match_all()`
- **CUB Fallback**: `cub::WarpReduce` (Sum, Min, Max), `cub::BlockReduce`, `cub::WarpScan` (InclusiveSum, ExclusiveSum), `cub::BlockScan`, `cub::CachingDeviceAllocator`

**New files**:
- `include/vgre/compiler/cuda_device_libs/cooperative_groups.h` — full cooperative groups API
- `include/vgre/compiler/cuda_device_libs/cub_fallback.h` — CUB-compatible warp/block reduce & scan

**Modified**:
- `include/vgre/compiler/cpu_cuda_env.h` — replaced inline `grid_group` with `#include` of new headers

**Key design decisions**:
- `thread_block_tile` is parameterized on tile size (4, 8, 16, 32) and delegates to existing `__shfl_*_sync` builtins.
- `reduce()` uses butterfly shuffle (`shfl_xor`) which is correct for power-of-two group sizes.
- `BlockReduce` / `BlockScan` use two-level decomposition: warp-level reduce/scan, then single-warp reduction/scan of per-warp partials with `__syncthreads` barrier between phases.
- CUB classes are header-only templates that compile to the same warp-shuffle primitives used by PTX translation.
- `match_any` / `match_all` return full mask in serial CPU model (all threads match).

---

### 3.9 Phase 9 — Deployment

**Status**: **DONE** — K8s Device Plugin (Go gRPC) and SLURM GRES plugin (C shared library) both implemented.

**New files**:
```
src/deployment/k8s_device_plugin/                   (new directory)
src/deployment/k8s_device_plugin/main.go            (200–300 lines)
src/deployment/k8s_device_plugin/Dockerfile
src/deployment/k8s_device_plugin/README.md
src/deployment/slurm_gres/                          (new directory)
src/deployment/slurm_gres/slurm_gres_vgpu.cpp       (150–200 lines)
src/deployment/slurm_gres/slurm_gres_vgpu.h       (50–80 lines)
```

---

### 3.10 Phase 10 — CDP, Profiling, Advanced Formats, Graph Updates

**Goal**: Complete remaining Tier 4–5 gaps.

#### 3.10.1 CUDA Dynamic Parallelism (Full API)

**Status**: **DONE** — `cudaDeviceSynchronize`, `cudaGetParameterBufferV2`, `cudaLaunchDeviceV2` implemented.

**New file**:
```
src/api/cudart/cudart_shim_cdp.cpp                  (100–150 lines)
```

**Key design decisions**:
- `cudaDeviceSynchronize`: Block until all child kernels launched by the current kernel complete. Requires a per-kernel child-kernel completion counter in `RuntimeEngine`.
- V2 APIs: Same as V1 but with additional parameter-buffer size validation.

---

#### 3.10.2 Profiling / Observability

**Status**: **DONE** — instruction-level sampling via `InstructionSample`, kernel timeline with nanosecond timestamps, Chrome trace / profiler JSON exports with `instructions` and `instruction_mix` fields.

**New files**:
```
src/advanced/profiling/cupti_equivalent.cpp         (300–400 lines)
src/advanced/profiling/kernel_timeline.cpp          (200–300 lines)
src/advanced/profiling/instruction_sampler.cpp      (150–200 lines)
```

**Key design decisions**:
- Kernel timeline: Reuse existing `RuntimeProfiler` (`src/advanced/runtime_profiler.cpp`) which already exports OTLP JSON. Add per-kernel start/end timestamps with nanosecond resolution (`std::chrono::high_resolution_clock`).
- Instruction sampling: Not feasible for CPU emulation (no hardware PC counter). Return `CUDA_ERROR_NOT_SUPPORTED` for true instruction sampling, but provide `VGRE_LOG_PROFILE` macro for manual annotation.
- Chrome trace export already works (`toChromeTraceJSON`); extend with kernel dependency edges.

---

#### 3.10.3 cuDNN INT8x4 / INT8x32 Packed Tensor Layouts

**Status**: **DONE** — `CUDNN_DATA_INT8x32` supported in `cudnnConvolutionForward`, `cudnnPoolingForward/Backward`, and `cudnnActivationForward/Backward` via dequantize→FP32 compute→requantize path.

**New file**:
```
src/api/cudnn/cudnn_int8_packed.cpp                 (100–150 lines)
```

**Key design decisions**:
- The existing INT8 path does dequantize→FP32 compute→requantize. Packed layouts (`NCHW_VECT_C`) require un-packing (channel interleaving) before the same FP32 compute path.
- `cudnnSetTensor4dDescriptorEx` with `CUDNN_DATA_INT8x4` / `CUDNN_DATA_INT8x32` — validate that channel count is divisible by 4 or 32, then store as normal `NCHW` with stride adjustment.

---

#### 3.10.4 Graph Exec Update v2

**Status**: **DONE** — `cudaGraphExecUpdate_v2` with per-node `cudaGraphExecUpdateResultInfo` error reporting implemented.

**New file**:
```
src/core/graph/graph_manager_exec_update_v2.cpp     (100–150 lines)
```

**Key design decisions**:
- `cudaGraphExecUpdate_v2` returns a `cudaGraphExecUpdateResultInfo` with per-node error codes. Extend existing `graph_manager_exec_update.cpp` with result-info output.

---

## 4. File Manifest Summary

### New Directories (15)

| Directory | Purpose | Est. Files |
|---|---|---|
| `src/api/cudart/` | Split monolithic CUDART shim by concern | 10–12 |
| `src/api/cuda_driver/` | Split monolithic driver shim by concern | 7–9 |
| `src/api/cublas/` | cuBLAS backend split by BLAS level | 4–5 |
| `src/api/cudnn/` | cuDNN backend split by layer type | 9–11 |
| `src/api/nccl/` | NCCL backend split by collective type | 4–5 |
| `src/api/cufft/` | **NEW** cuFFT shim + FFTW3 delegation | 5–6 |
| `src/api/curand/` | **NEW** cuRAND shim | 3–4 |
| `src/api/cusolver/` | **NEW** cuSOLVER shim + LAPACK delegation | 3–4 |
| `src/api/cusparse/` | **NEW** cuSPARSE shim | 4–5 |
| `src/api/cublasLt/` | **NEW** cuBLASLt shim | 2–3 |
| `src/compiler/ptx/` | PTX translator split by architecture / op family | 6–8 |
| `src/core/graph/` | Graph manager split by node type | 8–10 |
| `src/core/texture/` | Texture/surface object manager | 3–4 |
| `src/advanced/profiling/` | **NEW** CUPTI-equivalent profiling subsystem | 3–4 |
| `src/deployment/k8s_device_plugin/` | **NEW** K8s Device Plugin (Go) | 3–4 |
| `src/deployment/slurm_gres/` | **NEW** SLURM GRES plugin | 2–3 |

### Existing Files to Refactor (Split)

| Current File | Lines | Split Into |
|---|---|---|
| `src/api/cudart_shim.cpp` | ~1,300 | **DONE** — all CUDART shim files grouped under `src/api/cudart/` (`cudart_shim.cpp`, `cudart_shim_stream.cpp`, `cudart_shim_memset_nd.cpp`, `cudart_shim_graph_nodes.cpp`, `cudart_shim_graph_introspection.cpp`, `cudart_shim_graph_exec.cpp`, `cudart_shim_graph_user_objects.cpp`, `cudart_shim_capture.cpp`, `cudart_shim_texture_objects.cpp`, `cudart_shim_device_attrs.cpp`, `cudart_shim_external_memory.cpp`, `cudart_graph_internal.h`) |
| `src/api/cuda_driver_shim.cpp` | ~456 | **DONE** — `cuda_driver/cuda_driver_internal.h` (shared types, CUresult constants, toCU(), texture types), `cuda_driver/cuda_driver_device_context.cpp` (cuInit, cuDevice*, cuCtx*), `cuda_driver/cuda_driver_memory.cpp` (cuMemAlloc, cuMemFree, cuMemcpy*), `cuda_driver/cuda_driver_stream_event.cpp` (cuStream*, cuEvent*), `cuda_driver/cuda_driver_module.cpp` (cuModule*, cuLaunchKernel), `cuda_driver/cuda_driver_texture.cpp` (cuTexObject*, cuSurfObject*, legacy cuTexRef*) |
| `src/api/cublas_shim.cpp` | ~2,309 | **DONE** — `cublas/cublas_internal.h` (shared types, enums, CBLAS helpers, reference GEMM), `cublas/cublas_core.cpp` (handle + stream/mode utilities + legacy v1 aliases), `cublas/cublas_level1.cpp`, `cublas/cublas_level2.cpp`, `cublas/cublas_level3.cpp` |
| `src/api/cudnn_shim.cpp` | ~2,337 | **DONE** — `cudnn/cudnn_internal.h` (shared types, enums, structs, helpers), `cudnn/cudnn_core.cpp` (handle + all descriptors), `cudnn/cudnn_convolution.cpp` (forward + backward data/filter), `cudnn/cudnn_activation.cpp`, `cudnn/cudnn_softmax.cpp`, `cudnn/cudnn_pooling.cpp`, `cudnn/cudnn_batchnorm.cpp` (inference/training/backward), `cudnn/cudnn_dropout.cpp`, `cudnn/cudnn_tensor_ops.cpp` (OpTensor, ReduceTensor, TransformTensor), `cudnn/cudnn_rnn.cpp`, `cudnn/cudnn_attention.cpp` |
| `src/core/graph_manager.cpp` | ~933 | **DONE** — `graph/graph_manager.cpp` (core), `graph/graph_manager_extended_nodes.cpp`, `graph/graph_manager_introspection.cpp`, `graph/graph_manager_exec_update.cpp`, `graph/graph_manager_dependencies.cpp` |
| `src/compiler/ptx_translator.cpp` | ~607 | **DONE** — `ptx/ptx_translator_internal.h` (shared types, helpers), `ptx/ptx_translator_map.cpp` (opcode translation map), `ptx/ptx_translator.cpp` (translateInstruction/translateBlock/translate entry points) |

---

## 5. CMake & Build Integration

### 5.1 Per-Subdirectory CMakeLists.txt

Each new subdirectory gets its own `CMakeLists.txt` (following `src/advanced/tcp_cluster/CMakeLists.txt` pattern):

```cmake
# src/api/cufft/CMakeLists.txt
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

Missing libraries should be **conditional** so the build does not break if dependencies are absent:

```cmake
option(VGRE_ENABLE_CUFFT "Build cuFFT shim" ON)
option(VGRE_ENABLE_CURAND "Build cuRAND shim" ON)
option(VGRE_ENABLE_CUSOLVER "Build cuSOLVER shim" ON)
option(VGRE_ENABLE_CUSPARSE "Build cuSPARSE shim" ON)
option(VGRE_ENABLE_CUBLASLT "Build cuBLASLt shim" ON)
```

### 5.3 Cross-Platform Build Requirements

Following the existing `token_manager_linux.cpp` / `token_manager_macos.cpp` / `token_manager_win32.cpp` pattern, all new features must have platform-specific files when OS APIs differ:

| Feature | Linux | Windows | macOS | Notes |
|---|---|---|---|---|
| **cuRAND entropy** | `getrandom()` / `/dev/urandom` | `BCryptGenRandom()` | `getentropy()` | Already in `token_manager_*.cpp`; reuse pattern |
| **cuFFT** | FFTW3 via pkg-config | FFTW3 via vcpkg / prebuilt | FFTW3 via Homebrew | Conditional `find_package(FFTW3)` |
| **cuSOLVER** | LAPACKE via OpenBLAS | Intel MKL or OpenBLAS | Accelerate.framework / OpenBLAS | CMake `FindLAPACK` with fallback |
| **Shared memory (IPC)** | POSIX `shm_open` | `CreateFileMapping`/`MapViewOfFile` | POSIX `shm_open` | Already in `cuda_ipc_memory.cpp`; extend for new libraries |
| **External semaphores** | POSIX named semaphores | Windows named semaphores | POSIX named semaphores | Already in `cuda_external_semaphore.cpp` |
| **K8s Device Plugin** | Unix domain sockets + inotify | Named pipes + ReadDirectoryChanges | Unix domain sockets + kqueue | Go `net` package abstracts this; add platform-specific mount paths |
| **SLURM GRES** | `slurm_spank.h` + shared library | N/A (SLURM is Linux-only) | N/A | Linux-only plugin |
| **Profiling (CUPTI-equiv)** | `perf_event_open` for CPU PMU | `EvtCreateSession` / `TraceEvent` | `kdebug` / `signpost` | CPU-only fallback; GPU PMU not applicable |
| **Mesh topology** | `epoll` for any-to-any | `WSAPoll` / IOCP | `kqueue` | Already in `tcp_cluster/`; extend for new p2p collectives |

**Platform abstraction header**: Any new OS-dependent code must go through `include/vgre/common/platform.h` (existing) or a new `include/vgre/common/platform_*.h` file. No inline `#ifdef _WIN32` in business-logic files.

### 5.4 Mesh Topology Integration

The existing mesh topology system (`VGRE_MESH_PEERS`, `mesh_topology_impl.cpp`, `performPeerClientHandshake`) must be extended for new NCCL p2p and advanced collectives:

| Collective | Mesh Impact | File to Modify |
|---|---|---|
| `ncclSend`/`ncclRecv` | Direct rank→rank TCP connection required if not already connected | `src/api/nccl/nccl_shim_p2p.cpp` |
| `ncclAllToAll` | All ranks exchange with all ranks; mesh already full-duplex | `src/api/nccl/nccl_shim_alltoall.cpp` |
| `ncclGather`/`ncclScatter` | Root rank connects to all others; mesh handles this | `src/api/nccl/nccl_shim_gather_scatter.cpp` |
| **New library shims** (cuFFT, cuRAND, etc.) | No mesh impact — purely local compute | N/A |
| **Cluster-wide profiling** | Collect profiles from all nodes; aggregate in Chrome trace | `src/advanced/profiling/kernel_timeline.cpp` |

**Design rule**: Any new distributed feature must call `TCPClusterManager::ensureConnected(rank)` before sending data. Do not assume all-to-all connections are pre-warmed.

### 5.5 Functioning / Runtime Engine Integration

New features must integrate with the existing runtime subsystems:

| Subsystem | Integration Point | Requirement |
|---|---|---|
| **Adaptive Execution Engine** (`adaptive_execution_engine.cpp`) | New compute-heavy shims (cuBLAS Level-3, cuDNN conv) must call `recordMemoryBandwidth()` and `getMemoryBandwidthStats()` to inform thread-count tuning | Thread count auto-tuned per kernel |
| **Runtime Profiler** (`runtime_profiler.cpp`) | All new API entry points should emit NVTX ranges (`nvtxRangePushA`/`Pop`) | OTLP export already works; just add markers |
| **Memory Manager** (`memory_manager.cpp`) | New array / texture / surface allocations go through `MemoryManager` for UVM tracking | NUMA binding ≥ 2 MB, pool allocator for ≤ blockSize |
| **Scheduler** (`scheduler.cpp`) | Graph node replay uses `StreamScheduler`; new node types (kernel, memset, host) must enqueue correctly | Cooperative launch uses condition_variable start-gate |
| **Work Load Partitioner** (`workload_partitioner.cpp`) | Distributed kernels across cluster nodes; new collectives must register partition strategy | Recursive bisection for 3D grids |
| **IPC Manager** (`ipc_manager.cpp`) | Cross-process memory sharing for new libraries (cuRAND device stubs, external memory) | POSIX SHM + HMAC-SHA256 auth |
| **Resource Ledger** (`resource_ledger.cpp`) | Track GPU-memory-equivalent allocations for new tensor types (INT8 packed, cuDNN backend descriptors) | Prevent OOM on CPU host |

---

## 6. Testing Strategy

### 6.1 Test File Organization

Following the `tcp_cluster/` pattern, tests live alongside source:

```
tests/api/cudart/          # CUDART shim tests
tests/api/cuda_driver/     # Driver API tests
tests/api/cublas/          # cuBLAS tests
tests/api/cudnn/           # cuDNN tests
tests/api/cufft/           # NEW: cuFFT tests
tests/compiler/ptx/        # PTX instruction tests
tests/core/graph/          # Graph node-type tests
tests/core/texture/        # Texture/surface object tests
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

1. **`docs/missingFeatures.md`** — Move the implemented item from "Missing" to "Implemented Summary" with verification command.
2. **`docs/PROJECT_STATUS.md`** — Update coverage percentages and "What Works" section.
3. **`docs/PRODUCTION_READINESS_REPORT.md`** — If the gap was critical, update the Executive Summary.
4. **`docs/implementationPlan.md`** — Mark the item as `DONE` with date and PR link.
5. **`README.md`** (top-level) — If the feature is user-facing, update "What Works."

---

## 8. Progress Tracker

### Phase 1 — Critical CUDA Runtime Gaps

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 1.1 | `cudaStreamWaitEvent` | **DONE** | — | 2026-05-13 |
| 1.2 | `cudaEventQuery` | **DONE** | — | 2026-05-13 |
| 1.3 | `cudaStreamAddCallback` | **DONE** | — | 2026-05-13 |
| 1.4 | `cudaLaunchHostFunc` | **DONE** | — | 2026-05-13 |
| 1.5 | `cudaGetErrorName` / `cudaGetErrorString` | **DONE** | — | 2026-05-13 |
| 1.6 | `cudaMemcpyToSymbol` / `cudaMemcpyFromSymbol` | **DONE** | — | 2026-05-13 |
| 1.7 | `cudaMallocArray` / `cudaMalloc3DArray` | **DONE** | — | 2026-05-13 |
| 1.8 | `cudaPointerGetAttributes` | **DONE** | — | 2026-05-13 |
| 1.9 | `cudaMemset2D/3D/2DAsync/3DAsync` | **DONE** | — | 2026-05-13 |
| 1.10 | `cudaGraphAddKernelNode` | **DONE** | — | 2026-05-13 |
| 1.11 | `cudaGraphAddMemsetNode` | **DONE** | — | 2026-05-13 |
| 1.12 | `cudaGraphAddHostNode` | **DONE** | — | 2026-05-13 |
| 1.13 | `cudaGraphAddChildGraphNode` | **DONE** | — | 2026-05-13 |
| 1.14 | `cudaGraphAddEmptyNode` | **DONE** | — | 2026-05-13 |
| 1.15 | `cudaGraphAddEventRecordNode` / `EventWaitNode` | **DONE** | — | 2026-05-13 |
| 1.16 | `cudaGraphAddMemAllocNode` / `MemFreeNode` | **DONE** | — | 2026-05-13 |
| 1.17 | Graph introspection APIs (`GetNodes`, `GetEdges`, `NodeGetType`, `NodeGetDependencies`, etc.) | **DONE** | — | 2026-05-13 |
| 1.18 | Graph exec mutation APIs (`ExecKernelNodeSetParams`, `ExecMemcpyNodeSetParams`, `NodeSetEnabled`, etc.) | **DONE** | — | 2026-05-13 |
| 1.19 | Stream capture introspection (`IsCapturing`, `GetCaptureInfo_v2`, `ThreadExchangeStreamCaptureMode`, `UpdateCaptureDependencies`, `CopyAttributes`) | **DONE** | — | 2026-05-13 |
| 1.20 | CUDART texture/surface object APIs (`GetTextureObjectResourceDesc`, legacy `BindTexture*`, `BindSurfaceToArray`) | **DONE** | — | 2026-05-13 |
| 1.21 | CUDA Runtime device/function attributes (`FuncGetAttributes`, device limits, cache/smem config, `ChooseDevice`, `LaunchKernelExC`) | **DONE** | — | 2026-05-13 |
| 1.22 | CUDA Graph dependencies & user objects (`AddDependencies/RemoveDependencies`, `UserObjectCreate/Retain/Release`, `GraphRetainUserObject`, `NodeFindInClone`, `DebugDotPrint`) | **DONE** | — | 2026-05-13 |
| 1.23 | CUDART external memory/semaphore APIs (`ImportExternalMemory`, `DestroyExternalMemory`, `GetMappedBuffer`, `GetMappedMipmappedArray`) | **DONE** | — | 2026-05-13 |

### Phase 2 — cuBLAS

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 2.1 | Level-1 completion (`copy`, `swap`, `rot`, `asum`, `amax`, etc.) | **DONE** | — | 2026-05-13 |
| 2.2 | Pointer mode / atomics mode APIs | **DONE** | — | 2026-05-13 |
| 2.3 | Level-2 BLAS (`Trsv`, `Trsm`, `Ger`, `Symv`, `Gbmv`, `Syr`, etc.) | **DONE** | — | 2026-05-13 |
| 2.4 | Level-3 BLAS (`Trsm`, `Syrk`, `Syr2k`, `Trmm`, `Symm`, `Chemm`, etc.) | **DONE** | — | 2026-05-13 |
| 2.4a | `cublasGemmEx` FP16 / BF16 mixed-precision paths | **DONE** | — | 2026-05-14 |
| 2.4b | `cublasLtMatmul` FP16 / BF16 paths | **DONE** | — | 2026-05-14 |
| 2.5 | Complex Hermitian variants (`Cherk`, `Cher2k`) | **DONE** | — | 2026-05-14 |
| 2.6 | `cublasLoggerConfigure` / logging callbacks | **DONE** | — | 2026-05-14 |

### Phase 3 — cuDNN

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 3.1 | `cudnnConvolutionBackwardData` | **DONE** | — | 2026-05-13 |
| 3.2 | `cudnnConvolutionBackwardFilter` | **DONE** | — | 2026-05-13 |
| 3.3 | `cudnnBatchNormalizationForwardTraining` | **DONE** | — | 2026-05-13 |
| 3.4 | `cudnnBatchNormalizationBackward` | **DONE** | — | 2026-05-13 |
| 3.5 | `cudnnActivationBackward` | **DONE** | — | 2026-05-13 |
| 3.6 | `cudnnSoftmaxBackward` | **DONE** | — | 2026-05-14 |
| 3.7 | `cudnnPoolingBackward` | **DONE** | — | 2026-05-14 |
| 3.8 | `cudnnDropoutForward/Backward` | **DONE** | — | 2026-05-14 |
| 3.9 | `cudnnRNNForward/Backward` | **DONE** | — | 2026-05-14 |
| 3.10 | `cudnnMultiHeadAttnForward/Backward` | **DONE** | — | 2026-05-14 |
| 3.11 | `cudnnCTCLoss` | **DONE** | — | 2026-05-14 |
| 3.12 | `cudnnOpTensor` | **DONE** | — | 2026-05-14 |
| 3.13 | `cudnnReduceTensor` | **DONE** | — | 2026-05-14 |
| 3.13a | `cudnnReduceTensor` multi-dimensional non-scalar reductions | **DONE** | — | 2026-05-14 |
| 3.14 | `cudnnTransformTensor` | **DONE** | — | 2026-05-14 |
| 3.15 | cuDNN Backend API (v8+) | **DONE** (minimal stub) | — | 2026-05-14 |
| 3.16 | `cudnnLRNCrossChannelForward/Backward` | **DONE** | — | 2026-05-14 |
| 3.17 | `cudnnDivisiveNormalizationForward/Backward` | **DONE** | — | 2026-05-14 |

### Phase 4 — Missing Libraries

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 4.1 | cuFFT stub shim | **DONE** | — | 2026-05-14 |
| 4.2 | cuFFT 1D/2D/3D functional (reference DFT/IDFT) | **DONE** | — | 2026-05-14 |
| 4.3 | cuRAND stub shim | **DONE** | — | 2026-05-14 |
| 4.4 | cuRAND host API functional | **DONE** | — | 2026-05-14 |
| 4.5 | cuSOLVER stub shim | **DONE** | — | 2026-05-14 |
| 4.6 | cuSOLVER dense functional (LAPACK delegation) | **DONE** | — | 2026-05-14 |
| 4.7 | cuSPARSE stub shim | **DONE** | — | 2026-05-14 |
| 4.8 | cuSPARSE SpMV/SpMM functional | **DONE** | — | 2026-05-14 |
| 4.8a | cuSPARSE complex SpMV/SpMM (`CUDA_C_32F`, `CUDA_C_64F`) | **DONE** | — | 2026-05-14 |
| 4.9 | cuBLASLt stub shim | **DONE** | — | 2026-05-14 |
| 4.10 | cuBLASLt matmul + epilogue functional | **DONE** | — | 2026-05-14 |
| 4.11 | cuFFT advanced planning (`cufftEstimate*`, `cufftMakePlanMany`, `cufftGetSize*`) | **DONE** | — | 2026-05-14 |
| 4.12 | cuRAND Poisson distribution (`curandGeneratePoisson`) | **DONE** | — | 2026-05-14 |
| 4.13 | cuRAND seed generation (`curandGenerateSeeds`) | **DONE** | — | 2026-05-14 |
| 4.14 | cuRAND Sobol direction vectors (`curandGetDirectionVectors32/64`, `curandGetScrambleConstants32/64`) | **DONE** | — | 2026-05-14 |
| 4.15 | cuRAND Sobol quasi-random generation (`SOBOL32/64`, `SCRAMBLED_SOBOL32/64`) | **DONE** | — | 2026-05-14 |

### Phase 5 — CUDA Driver API

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 5.1 | `cuMemAllocManaged` | **DONE** | — | 2026-05-14 |
| 5.2 | `cuMemHostAlloc` / `Register` / `Unregister` | **DONE** | — | 2026-05-14 |
| 5.3 | `cuMemcpy2D/3D` / `Async` variants | **DONE** | — | 2026-05-14 |
| 5.4 | `cuStreamAddCallback` / `Query` / introspection (GetFlags, GetPriority) | **DONE** | — | 2026-05-14 |
| 5.5 | `cuEventQuery` | **DONE** | — | 2026-05-14 |
| 5.6 | Context management (`GetLimit`, `SetLimit`, `Pop`, `Push`, Attach, Detach, etc.) | **DONE** | — | 2026-05-14 |
| 5.7 | Device queries (`GetUuid`, `GetP2PAttribute`, `GetTexture1DLinearMaxWidth`, graph mem, RDMA flush) | **DONE** | — | 2026-05-14 |
| 5.8 | `cuGraph*` family | **DONE** | — | 2026-05-14 |
| 5.9 | `cuExternalMemory*` / `cuExternalSemaphore*` | **DONE** | — | 2026-05-14 |
| 5.10 | Driver texture reference gaps (`cuTexRefSetAddress2D`, filter, mipmap, border, `cuSurfRefSetFormat`) | **DONE** | — | 2026-05-14 |
| 5.10a | Driver texture `CU_AD_FORMAT_HALF` (FP16) support | **DONE** | — | 2026-05-14 |
| 5.11 | `cuMemcpy3DAsync` — async 3D copy path | **DONE** | — | 2026-05-14 |
| 5.12 | Driver surface references (`cuModuleGetSurfRef`, `cuSurfRefSetArray`, `cuSurfRefGetArray`) | **DONE** | — | 2026-05-14 |
| 5.13 | CUDA Virtual Memory on Linux (`cuMemCreate`, `cuMemAddressReserve`, `cuMemMap`, `cuMemSetAccess`) | **DONE** | — | 2026-05-14 |

### Phase 6 — PTX ISA

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 6.1 | `tex` / `tld4` / `txq` instructions | **DONE** | — | 2026-05-14 |
| 6.2 | `suld` / `sust` instructions | **DONE** | — | 2026-05-14 |
| 6.3 | `atom.shared.*` | **DONE** | — | 2026-05-14 |
| 6.4 | Missing `cvt.*` variants | **DONE** | — | 2026-05-14 |
| 6.5 | `rcp.rn` / `sqrt.rn` / `div.rn` | **DONE** | — | 2026-05-14 |
| 6.6 | FP16 vector loads (`ld.global.v2/v4.f16`) | **DONE** | — | 2026-05-14 |
| 6.7 | `match.sync` / `elect.sync` | **DONE** | — | 2026-05-14 |
| 6.8 | `grid.sync` / `griddepcontrol` | **DONE** | — | 2026-05-14 |
| 6.9 | TMA 3D/4D/5D (`cp.async.bulk.tensor`) | **DONE** | — | 2026-05-14 |
| 6.10 | `tcgen05.*` (Blackwell) | **DONE** | — | 2026-05-14 |

### Phase 7 — NCCL

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 7.1 | `ncclSend` / `ncclRecv` | **DONE** | — | 2026-05-14 |
| 7.2 | `ncclAllToAll` | **DONE** | — | 2026-05-14 |
| 7.3 | `ncclGather` / `ncclScatter` | **DONE** | — | 2026-05-14 |

### Phase 8 — Cooperative Groups & Device Libraries

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 8.1 | `thread_block` cooperative group | **DONE** | — | 2026-05-14 |
| 8.2 | `coalesced_group` | **DONE** | — | 2026-05-14 |
| 8.3 | `reduce()` / `partition()` / `shfl()` | **DONE** | — | 2026-05-14 |
| 8.4 | `thread_block_tile` | **DONE** | — | 2026-05-14 |
| 8.5 | `multi_grid` | **DONE** | — | 2026-05-14 |
| 8.6 | CUB / Thrust fallback headers | **DONE** | — | 2026-05-14 |

### Phase 9 — Deployment

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 9.1 | Kubernetes Device Plugin | **DONE** | — | 2026-05-14 |
| 9.2 | SLURM GRES Plugin | **DONE** | — | 2026-05-14 |

### Phase 10 — CDP, Profiling, Advanced Formats, Graph Updates

**Status**: **DONE**

**10.1 CDP full API**
- Added `cudaGetParameterBufferV2(size_t alignment, size_t size)` → `vgre_cdp_get_param_buffer_v2` with `std::aligned_alloc` fallback for large alignments.
- Added `cudaLaunchDeviceV2(void* fn, void* paramBuf, const cudaLaunchConfig* config)` → `vgre_cdp_launch_device_v2` that unpacks config and delegates to v1.
- Added device-side `cudaDeviceSynchronize()` → `vgre_cdp_device_synchronize()` which recursively drains child kernels via `CDPExecutor::deviceSynchronize()` until queue is empty.
- Files: `include/vgre/runtime/cdp_executor.h`, `src/runtime/cdp_executor.cpp`, `include/vgre/compiler/cpu_cuda_env.h`

**10.2 Profiling / CUPTI-equivalent**
- Added `InstructionSample` struct tracking load/store/ALU/barrier/branch/other counts.
- Added `InstructionSample instructions` to `ProfileEvent`; aggregated into `KernelStats.totalInstructions`, `avgInstructionsPerInvocation`, and `instructionMix`.
- Added `RuntimeProfiler::recordInstructionSample()`, `estimateInstructions()`, `getInstructionMix()`.
- Execution engine hooks JIT `estimatedInstructionCount` to populate instruction data at kernel launch.
- Chrome trace JSON and profiler JSON exports now include `instructions` and `instruction_mix` fields.
- Files: `include/vgre/advanced/runtime_profiler.h`, `src/advanced/runtime_profiler.cpp`, `src/core/runtime_engine_launch.cpp`

**10.3 cuDNN INT8x4 / INT8x32 packed layouts**
- Extended `cudnnConvolutionForward` to handle `CUDNN_DATA_INT8x32` (dequantize→FP32 compute→requantize).
- Added INT8 paths to `cudnnPoolingForward/Backward` and `cudnnActivationForward/Backward` for `INT8/INT8x4/INT8x32`.
- Files: `src/api/cudnn/cudnn_convolution.cpp`, `src/api/cudnn/cudnn_pooling.cpp`, `src/api/cudnn/cudnn_activation.cpp`

**10.4 `cudaGraphExecUpdate_v2`**
- Added `GraphManager::updateExecV2(execId, graphId, nodeIds)` which verifies and copies only the specified nodes.
- Added `RuntimeEngine::graphUpdateExecV2()`, `CUDAInterceptor::graphExecUpdateV2()`, and `cudaGraphExecUpdate_v2()` cudart shim.
- Files: `include/vgre/core/graph_manager.h`, `src/core/graph/graph_manager.cpp`, `include/vgre/core/runtime_engine.h`, `src/core/runtime_engine.cpp`, `include/vgre/api/cuda_interceptor.h`, `src/api/cuda_interceptor_graphs.cpp`, `src/api/cudart/cudart_shim.cpp`

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 10.1 | CDP full API (`cudaDeviceSynchronize`, `GetParameterBufferV2`, `LaunchDeviceV2`) | **DONE** | — | 2026-05-14 |
| 10.2 | Profiling / CUPTI-equivalent (kernel timeline, instruction sampler) | **DONE** | — | 2026-05-14 |
| 10.3 | cuDNN INT8x4 / INT8x32 packed layouts | **DONE** | — | 2026-05-14 |
| 10.4 | `cudaGraphExecUpdate_v2` | **DONE** | — | 2026-05-14 |

### Phase 11 — Additional Gaps (Deep Audit)

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 11.1 | cuDNN LRN (`cudnnLRNCrossChannelForward/Backward`) | **DONE** | — | 2026-05-14 |
| 11.2 | cuDNN Divisive Normalization | **DONE** | — | 2026-05-14 |
| 11.3 | cuBLAS Hermitian complex (`Cherk`, `Cher2k`) | **DONE** | — | 2026-05-14 |
| 11.4 | cuBLAS logging (`cublasLoggerConfigure`) | **DONE** | — | 2026-05-14 |
| 11.5 | cuFFT advanced planning APIs | **DONE** | — | 2026-05-14 |
| 11.6 | cuRAND Poisson distribution | **DONE** | — | 2026-05-14 |
| 11.7 | cuRAND seed generation | **DONE** | — | 2026-05-14 |
| 11.8 | CUDA Driver async 3D memcpy | **DONE** | — | 2026-05-14 |
| 11.9 | CUDA Driver surface references (full) | **DONE** | — | 2026-05-14 |
| 11.10 | CUDA Virtual Memory on Linux | **DONE** | — | 2026-05-14 |

---

**End of Document**
