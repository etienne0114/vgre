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

**Missing**: `cudaStreamWaitEvent`, `cudaEventQuery`, `cudaStreamAddCallback`, `cudaLaunchHostFunc`, `cudaGetErrorName`, `cudaGetErrorString`

**New files**:
```
src/api/cudart/cudart_shim_stream_sync.cpp      (300–400 lines)
src/api/cudart/cudart_shim_event_query.cpp     (150–200 lines)
src/api/cudart/cudart_shim_callbacks.cpp       (200–250 lines)
src/api/cudart/cudart_shim_error_strings.cpp   (100–150 lines)  # static table
```

**Rationale**: `cudart_shim.cpp` is already 1,300+ lines. Splitting by concern prevents bloat.

**Key design decisions**:
- `cudaStreamWaitEvent`: Reuse existing `CUDAInterceptor::streamWaitEvent` internal path; add CUDART shim wrapper.
- `cudaEventQuery`: Non-blocking status check on `EventManager`; do not reuse `cudaEventSynchronize` path.
- `cudaStreamAddCallback`: Legacy — enqueue a `std::function<void()>` into the stream task queue.
- `cudaLaunchHostFunc`: Modern replacement — same mechanism as callbacks but with `cudaHostFn_t` signature.

---

#### 3.1.2 Memory & Symbols

**Missing**: `cudaMemcpyToSymbol`, `cudaMemcpyToSymbolAsync`, `cudaMemcpyFromSymbol`, `cudaMemcpyFromSymbolAsync`, `cudaMallocArray`, `cudaMalloc3DArray`, `cudaMalloc3D`, `cudaMemcpy3DAsync`, `cudaHostGetDevicePointer`, `cudaHostGetFlags`, `cudaArrayGetInfo`, `cudaArrayDestroy`, `cudaPointerGetAttributes`, `cudaMemset2D`, `cudaMemset3D`, `cudaMemset2DAsync`, `cudaMemset3DAsync`

**New files**:
```
src/api/cudart/cudart_shim_symbols.cpp         (250–300 lines)
src/api/cudart/cudart_shim_arrays.cpp          (300–400 lines)
src/api/cudart/cudart_shim_memset_nd.cpp       (200–250 lines)
src/api/cudart/cudart_shim_pointer_attrs.cpp   (100–150 lines)
```

**Modify**:
- `src/core/memory/memory_manager.cpp` — add `symbolLookup_` map for `__device__` constant symbols.

**Key design decisions**:
- Symbols: Store `__constant__` variables in a `std::unordered_map<std::string, DeviceAllocation>` inside `MemoryManager`. `cudaMemcpyToSymbol` does a keyed lookup + memcpy.
- Arrays: Wrap `cudaArray_t` as a `struct ArrayDesc { void* data; cudaChannelFormatDesc desc; size_t width, height, depth; }`.
- `cudaPointerGetAttributes`: Query `MemoryManager::getPointerType()` — already has host/device/managed classification.

---

#### 3.1.3 CUDA Graphs — Kernel & Essential Node Types

**Missing**: `cudaGraphAddKernelNode`, `cudaGraphAddMemsetNode`, `cudaGraphAddHostNode`, `cudaGraphAddChildGraphNode`, `cudaGraphAddEmptyNode`, `cudaGraphAddEventRecordNode`, `cudaGraphAddEventWaitNode`, `cudaGraphAddMemAllocNode`, `cudaGraphAddMemFreeNode`

**New files**:
```
src/core/graph/graph_manager_kernel_nodes.cpp      (300–400 lines)
src/core/graph/graph_manager_memset_nodes.cpp      (150–200 lines)
src/core/graph/graph_manager_host_nodes.cpp        (100–150 lines)
src/core/graph/graph_manager_child_nodes.cpp       (100–150 lines)
src/core/graph/graph_manager_event_nodes.cpp       (100–150 lines)
src/core/graph/graph_manager_memalloc_nodes.cpp    (150–200 lines)
```

**Modify**:
- `src/api/cudart/cudart_shim_graphs.cpp` — add shim wrappers for all new node types.
- `include/vgre/core/graph_manager.h` — declare new node-type add/update methods.

**Rationale**: `graph_manager.cpp` is 35,000 bytes (~1,000 lines). Splitting by node type keeps each file under 400 lines.

**Key design decisions**:
- Kernel node: `GraphManager::addKernelNode` already exists internally; expose it via CUDART shim.
- Memset node: Reuse `MemoryManager::memsetD8/D16/D32` paths with graph node wrapper.
- Host node: Store `std::function<void()>` in node; execute during graph replay via `StreamScheduler`.
- Mem-alloc/free node: Use stream-ordered pool allocator (`cudaMallocFromPoolAsync` path) tied to graph lifetime.

---

#### 3.1.4 Graph Node Introspection & Exec Mutation

**Missing**: `cudaGraphKernelNodeSetParams`, `cudaGraphKernelNodeGetParams`, `cudaGraphMemcpyNodeGetParams`, `cudaGraphMemsetNodeGetParams`, `cudaGraphMemsetNodeSetParams`, `cudaGraphHostNodeGetParams`, `cudaGraphHostNodeSetParams`, `cudaGraphNodeGetType`, `cudaGraphGetNodes`, `cudaGraphGetRootNodes`, `cudaGraphGetEdges`, `cudaGraphNodeGetDependencies`, `cudaGraphNodeGetDependentNodes`, `cudaGraphExecKernelNodeSetParams`, `cudaGraphExecMemcpyNodeSetParams`, `cudaGraphExecMemsetNodeSetParams`, `cudaGraphExecHostNodeSetParams`, `cudaGraphExecChildGraphNodeSetParams`, `cudaGraphExecEventRecordNodeSetEvent`, `cudaGraphExecEventWaitNodeSetEvent`, `cudaGraphInstantiateWithFlags`, `cudaGraphInstantiateWithParams`, `cudaGraphExecGetFlags`, `cudaGraphUpload`, `cudaGraphNodeSetEnabled`, `cudaGraphNodeGetEnabled`, `cudaGraphExecNodeSetParams`, `cudaGraphKernelNodeCopyAttributes`

**New files**:
```
src/core/graph/graph_manager_introspection.cpp      (300–400 lines)
src/core/graph/graph_manager_exec_update.cpp        (250–350 lines)
src/core/graph/graph_manager_upload.cpp           (100–150 lines)
```

---

#### 3.1.5 Stream Capture Introspection

**Missing**: `cudaStreamIsCapturing`, `cudaStreamGetCaptureInfo`, `cudaStreamGetCaptureInfo_v2`, `cudaThreadExchangeStreamCaptureMode`, `cudaStreamUpdateCaptureDependencies`, `cudaStreamCopyAttributes`

**New file**:
```
src/api/cudart/cudart_shim_capture.cpp              (200–250 lines)
```

---

#### 3.1.6 Texture & Surface — CUDART Object APIs

**Missing**: `cudaCreateTextureObject`, `cudaDestroyTextureObject`, `cudaGetTextureObjectResourceDesc`, `cudaGetTextureObjectTextureDesc`, `cudaGetTextureObjectResourceViewDesc`, `cudaCreateSurfaceObject`, `cudaDestroySurfaceObject`, `cudaGetSurfaceObjectResourceDesc`, `cudaGetTextureReference`, `cudaGetSurfaceReference`, `cudaBindTexture`, `cudaUnbindTexture`, `cudaBindTextureToArray`, `cudaBindTexture2D`, `cudaBindSurfaceToArray`

**New files**:
```
src/core/texture/texture_object_manager.cpp       (250–350 lines)
src/core/texture/surface_object_manager.cpp         (150–200 lines)
src/api/cudart/cudart_shim_texture_objects.cpp      (200–250 lines)
src/api/cudart/cudart_shim_texture_legacy.cpp       (150–200 lines)  # bind/unbind
```

**Rationale**: `texture_manager.cpp` is already 39,000 bytes. Object APIs should live in their own files, not the existing legacy texture reference path.

---

#### 3.1.7 CUDA Runtime — Device & Function Attributes

**Missing**: `cudaFuncGetAttributes`, `cudaFuncSetCacheConfig`, `cudaFuncSetSharedMemConfig`, `cudaDeviceGetLimit`, `cudaDeviceSetLimit`, `cudaDeviceGetCacheConfig`, `cudaDeviceSetCacheConfig`, `cudaDeviceGetSharedMemConfig`, `cudaDeviceSetSharedMemConfig`, `cudaChooseDevice`, `cudaThreadExit`, `cudaThreadSynchronize`, `cudaThreadSetLimit`, `cudaThreadGetLimit`, `cudaDeviceGetP2PAttribute`, `cudaDeviceFlushGPUDirectRDMAWrites`, `cudaDeviceGetGraphMemAttribute`, `cudaDeviceSetGraphMemAttribute`, `cudaLaunchKernelExC`, `cudaLaunchConfig`

**New file**:
```
src/api/cudart/cudart_shim_device_attrs.cpp         (200–300 lines)
```

**Key design decisions**:
- `cudaFuncGetAttributes`: Query `KernelIR` for register count, shared memory, and block-size limits; populate `cudaFuncAttributes`.
- `cudaDeviceGetLimit/SetLimit`: Track `stackSize`, `printfFifoSize`, `mallocHeapSize` in `DeviceProperties`.
- `cudaThread*`: Legacy 1.x APIs — implement as thin wrappers around existing 4.x device APIs.
- `cudaLaunchKernelExC`: Extended launch with `cudaLaunchConfig_t` — forward to existing launch path with extra attributes.

---

#### 3.1.8 CUDA Graphs — Dependencies & User Objects

**Missing**: `cudaGraphAddDependencies`, `cudaGraphRemoveDependencies`, `cudaGraphRetainUserObject`, `cudaGraphReleaseUserObject`, `cudaUserObjectCreate`, `cudaUserObjectRetain`, `cudaUserObjectRelease`, `cudaGraphNodeFindInClone`, `cudaGraphDebugDotPrint`

**New files**:
```
src/core/graph/graph_manager_dependencies.cpp       (100–150 lines)
src/core/graph/graph_manager_user_objects.cpp     (100–150 lines)
```

**Key design decisions**:
- Dependencies: `GraphManager` already tracks adjacency lists; `AddDependencies` inserts edges, `RemoveDependencies` deletes them with cycle check.
- User objects: Reference-counted `std::shared_ptr<void>` with custom deleter, keyed by graph node ID.

---

#### 3.1.9 External Memory & Semaphore — CUDART

**Missing**: `cudaImportExternalMemory`, `cudaDestroyExternalMemory`, `cudaExternalMemoryGetMappedBuffer`, `cudaExternalMemoryGetMappedMipmappedArray`, `cudaExternalSemaphoreGetSignalNodeParams`, `cudaExternalSemaphoreGetWaitNodeParams`

**New file**:
```
src/api/cudart/cudart_shim_external_memory.cpp      (150–200 lines)
```

**Key design decisions**:
- These are the CUDART counterparts to driver `cuExternalMemory*` / `cuExternalSemaphore*` APIs. Reuse the existing external-semaphore infrastructure in `src/api/cuda_external_semaphore.cpp`.
- `cudaImportExternalMemory` maps to `cuImportExternalMemory` + handle wrapping.

---

### 3.2 Phase 2 — cuBLAS Level-2 / Level-3 Backfill

**Goal**: Enable PyTorch `torch.linalg.solve`, `torch.mm` with non-GEMM paths.

#### 3.2.1 Level-1 BLAS Completion

**Missing**: `cublasScopy`, `cublasDcopy`, `cublasSswap`, `cublasDswap`, `cublasSrot`, `cublasDrot`, `cublasSrotm`, `cublasSrotmg`, `cublasSrotg`, `cublasSasum`, `cublasDasum`, `cublasIsamax`, `cublasIdamax`, `cublasDnrm2`, `cublasDscal`, `cublasGetPointerMode`, `cublasSetPointerMode`, `cublasGetAtomicsMode`, `cublasSetAtomicsMode`, `cublasLoggerConfigure`

**New files**:
```
src/api/cublas/cublas_shim_level1.cpp               (300–400 lines)
src/api/cublas/cublas_shim_pointer_mode.cpp         (50–80 lines)
src/api/cublas/cublas_shim_atomics.cpp              (50–80 lines)
```

---

#### 3.2.2 Level-2 BLAS

**Missing**: `cublasStrsv`, `cublasDtrsv`, `cublasStrsm`, `cublasDtrsm`, `cublasSger`, `cublasDger`, `cublasSsymv`, `cublasDsymv`, `cublasSgbmv`, `cublasDgbmv`, `cublasSsyr`, `cublasDsyr`, `cublasSsyr2`, `cublasDsyr2`, plus packed/banded/triangular variants

**New files**:
```
src/api/cublas/cublas_shim_level2.cpp               (400–500 lines)
src/api/cublas/cublas_level2_backend.cpp            (300–400 lines)  # OpenBLAS delegation
```

**Key design decision**: Delegate to OpenBLAS for all Level-2/Level-3 routines. VGRE's cuBLAS shim already uses OpenBLAS for GEMM; extend the same pattern.

---

#### 3.2.3 Level-3 BLAS

**Missing**: `cublasStrsm`, `cublasDtrsm`, `cublasSsyrk`, `cublasDsyrk`, `cublasSsyr2k`, `cublasDsyr2k`, `cublasStrmm`, `cublasDtrmm`, `cublasSsymm`, `cublasDsymm`, `cublasChemm`, `cublasCherk`, `cublasCher2k`, plus batched variants

**New files**:
```
src/api/cublas/cublas_shim_level3.cpp               (400–500 lines)
src/api/cublas/cublas_level3_backend.cpp            (300–400 lines)  # OpenBLAS delegation
```

---

### 3.3 Phase 3 — cuDNN Backward Passes + Training

**Goal**: Enable PyTorch training (backward pass, BN training, dropout).

#### 3.3.1 Convolution Backward

**Missing**: `cudnnConvolutionBackwardData`, `cudnnConvolutionBackwardFilter`, `cudnnConvolutionBackwardBias`

**New file**:
```
src/api/cudnn/cudnn_convolution_backward.cpp        (400–500 lines)
```

**Key design decision**: Backward data = transposed convolution (im2col + GEMM with flipped filters). Backward filter = im2col + GEMM with input activations and output gradients.

---

#### 3.3.2 Batch Normalization Training

**Missing**: `cudnnBatchNormalizationForwardTraining`, `cudnnBatchNormalizationBackward`

**New file**:
```
src/api/cudnn/cudnn_batchnorm_training.cpp          (200–300 lines)
```

---

#### 3.3.3 Other Backward Passes

**Missing**: `cudnnActivationBackward`, `cudnnSoftmaxBackward`, `cudnnPoolingBackward`

**New file**:
```
src/api/cudnn/cudnn_backward_generic.cpp            (250–350 lines)
```

---

#### 3.3.4 Advanced Layers

**Missing**: `cudnnDropoutForward/Backward`, `cudnnRNNForward/Backward`, `cudnnMultiHeadAttnForward/Backward`, `cudnnCTCLoss`, `cudnnDivisiveNormalizationForward/Backward`, `cudnnLRNCrossChannelForward/Backward`, `cudnnTransformTensor`, `cudnnAddTensor`, `cudnnOpTensor`, `cudnnReduceTensor`

**New files**:
```
src/api/cudnn/cudnn_dropout.cpp                     (150–200 lines)
src/api/cudnn/cudnn_rnn.cpp                         (300–400 lines)  # LSTM/GRU cell math
src/api/cudnn/cudnn_attention.cpp                   (200–300 lines)  # multi-head attention
src/api/cudnn/cudnn_ctc_loss.cpp                    (100–150 lines)
src/api/cudnn/cudnn_op_tensor.cpp                   (150–200 lines)
src/api/cudnn/cudnn_reduce_tensor.cpp             (150–200 lines)
```

---

#### 3.3.5 cuDNN Backend API (v8+)

**Missing**: `cudnnBackendCreateDescriptor`, `cudnnBackendDestroyDescriptor`, `cudnnBackendSetAttribute`, `cudnnBackendGetAttribute`, `cudnnBackendInitialize`, `cudnnBackendExecute`, `cudnnBackendPopulate`, `cudnnBackendFinalize`, plus engine/heuristics/plan descriptors

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

**Missing**: `cuMemAllocManaged`, `cuMemHostAlloc`, `cuMemHostGetDevicePointer`, `cuMemHostRegister`, `cuMemHostUnregister`, `cuMemAllocPitch`, `cuMemcpy2D`, `cuMemcpy2DAsync`, `cuMemcpy3D`, `cuMemcpy3DAsync`, `cuMemcpyDtoDAsync`, `cuMemcpyDtoHAsync`, `cuMemcpyHtoDAsync`, `cuMemcpyAsync`, `cuMemcpyPeer`, `cuMemcpyPeerAsync`, `cuMemsetD8`, `cuMemsetD16`, `cuMemsetD32`, `cuMemsetD2D8`, `cuMemsetD2D16`, `cuMemsetD2D32`

**New files**:
```
src/api/cuda_driver/cuda_driver_shim_memory.cpp       (300–400 lines)
src/api/cuda_driver/cuda_driver_shim_memcpy.cpp       (250–350 lines)
src/api/cuda_driver/cuda_driver_shim_memset.cpp       (150–200 lines)
```

---

#### 3.5.2 Streams & Events

**Missing**: `cuStreamAddCallback`, `cuStreamQuery`, `cuStreamGetFlags`, `cuStreamGetPriority`, `cuStreamGetId`, `cuStreamGetCtx`, `cuStreamGetCaptureInfo`, `cuStreamIsCapturing`, `cuStreamUpdateCaptureDependencies`, `cuEventQuery`

**New file**:
```
src/api/cuda_driver/cuda_driver_shim_stream.cpp       (200–300 lines)
```

---

#### 3.5.3 Context & Device

**Missing**: `cuCtxGetDevice`, `cuCtxGetFlags`, `cuCtxGetLimit`, `cuCtxSetLimit`, `cuCtxGetCacheConfig`, `cuCtxSetCacheConfig`, `cuCtxGetSharedMemConfig`, `cuCtxSetSharedMemConfig`, `cuCtxGetStreamPriorityRange`, `cuCtxGetId`, `cuCtxGetApiVersion`, `cuCtxPopCurrent`, `cuCtxPushCurrent`, `cuCtxAttach`, `cuCtxDetach`, `cuDeviceGetUuid`, `cuDeviceGetTexture1DLinearMaxWidth`, `cuDeviceGetP2PAttribute`, `cuDeviceGetNvSciSyncAttributes`, `cuDeviceGetGraphMemAttribute`, `cuDeviceSetGraphMemAttribute`, `cuDeviceFlushGPUDirectRDMAWrites`

**New file**:
```
src/api/cuda_driver/cuda_driver_shim_context.cpp        (250–350 lines)
```

---

#### 3.5.4 Graphs, External Resources, Profiler

**Missing**: `cuGraph*`, `cuExternalMemory*`, `cuExternalSemaphore*`, `cuProfilerStart`, `cuProfilerStop`, `cuGetErrorName`, `cuGetErrorString`

**New files**:
```
src/api/cuda_driver/cuda_driver_shim_graph.cpp        (300–400 lines)
src/api/cuda_driver/cuda_driver_shim_external.cpp     (150–200 lines)
src/api/cuda_driver/cuda_driver_shim_profiler.cpp     (50–80 lines)
```

---

#### 3.5.5 Texture / Surface Reference Gaps (Driver)

**Missing**: `cuTexRefSetAddress2D`, `cuTexRefSetAddressMode`, `cuTexRefSetFilterMode`, `cuTexRefSetMaxAnisotropy`, `cuTexRefSetMipmapFilterMode`, `cuTexRefSetMipmapLevelBias`, `cuTexRefSetMipmapLevelClamp`, `cuTexRefSetBorderColor`, `cuSurfRefSetFormat` (missing formats: half, signed/unsigned normalized, NV12)

**New file**:
```
src/api/cuda_driver/cuda_driver_shim_texref.cpp       (150–200 lines)
```

**Key design decisions**:
- `cuTexRefSetAddress2D`: Pitched 2D texture binding; compute pitch and bind to `TextureManager`.
- Filter modes: `CU_TR_FILTER_MODE_POINT` (nearest) and `CU_TR_FILTER_MODE_LINEAR` — map to existing bilinear/trilinear math in `TextureManager`.
- `cuSurfRefSetFormat`: Extend the existing format switch in `cuda_driver_shim.cpp` to include `CU_AD_FORMAT_HALF`, `CU_AD_FORMAT_SIGNED_INT8`, `CU_AD_FORMAT_UNSIGNED_INT8` with normalized variants.

---

### 3.6 Phase 6 — PTX ISA Expansion

**Goal**: Unblock more kernels (texture sampling, shared atomics, precise FP variants).

#### 3.6.1 Texture / Surface Instructions

**Missing PTX**: `tex`, `tld4`, `txq`, `suld`, `sust`

**New files**:
```
src/compiler/ptx/ptx_texture_ops.cpp                (300–400 lines)
src/compiler/ptx/ptx_surface_ops.cpp                (200–250 lines)
```

**Key design decision**: These map to the existing `vgre_tex1D_f32` / `vgre_surf2Dwrite_f32` C++ builtins in `cpu_cuda_env.h`. The PTX translator should emit calls to those functions rather than inline math.

---

#### 3.6.2 Shared-Memory Atomics

**Missing PTX**: `atom.shared.add`, `atom.shared.cas`, `atom.shared.exch`, `atom.shared.max`, `atom.shared.min`

**New file**:
```
src/compiler/ptx/ptx_shared_atomics.cpp             (150–200 lines)
```

---

#### 3.6.3 Conversion & Precision Variants

**Missing PTX**: Dozens of `cvt.*` variants, `rcp.rn.f32`, `sqrt.rn.f32`, `div.rn.f32/f64`, FP16 vector loads (`ld.global.v2.f16`, `ld.global.v4.f16`)

**New files**:
```
src/compiler/ptx/ptx_conversion.cpp                 (200–300 lines)
src/compiler/ptx/ptx_fp16_vector.cpp              (100–150 lines)
```

---

#### 3.6.4 Hopper / Blackwell Extensions

**Missing PTX**: `cp.async.bulk.tensor.3d/4d/5d`, `tcgen05.*` (Blackwell SM100), `cp.reduce.async`, `wgmma.mma_async` additional shapes, `match.sync`, `elect.sync`, `grid.sync`, `griddepcontrol`

**New files**:
```
src/compiler/ptx/ptx_hopper_tma_extended.cpp        (150–200 lines)
src/compiler/ptx/ptx_blackwell_tcgen05.cpp          (200–300 lines)
src/compiler/ptx/ptx_warp_match.cpp                 (100–150 lines)
src/compiler/ptx/ptx_grid_sync.cpp                (100–150 lines)
```

---

### 3.7 Phase 7 — NCCL Point-to-Point & Advanced Collectives

**Missing**: `ncclSend`, `ncclRecv`, `ncclAllToAll`, `ncclGather`, `ncclScatter`

**New files**:
```
src/api/nccl/nccl_shim_p2p.cpp                      (200–300 lines)
src/api/nccl/nccl_shim_alltoall.cpp               (150–200 lines)
src/api/nccl/nccl_shim_gather_scatter.cpp         (150–200 lines)
```

**Key design decision**: Point-to-point reuses the existing `TCPClusterManager` transport. `ncclSend`/`ncclRecv` are thin wrappers over `sendAll`/`recvAll` with rank addressing.

---

### 3.8 Phase 8 — Cooperative Groups & Device-Side Libraries

**Missing**: `thread_block`, `coalesced_group`, `reduce()`, `partition()`, `shfl()`, `thread_block_tile`, `multi_grid`

**New files**:
```
include/vgre/compiler/cuda_device_libs/cooperative_groups.h    (300–400 lines)
include/vgre/compiler/cuda_device_libs/cub_fallback.h            (200–300 lines)
```

**Modify**:
- `include/vgre/compiler/cpu_cuda_env.h` — add `thread_block` and `thread_block_tile` classes.

---

### 3.9 Phase 9 — Deployment

**Missing**: K8s Device Plugin, SLURM GRES

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

**Missing**: `cudaDeviceSynchronize`, `cudaGetParameterBufferV2`, `cudaLaunchDeviceV2`

**New file**:
```
src/api/cudart/cudart_shim_cdp.cpp                  (100–150 lines)
```

**Key design decisions**:
- `cudaDeviceSynchronize`: Block until all child kernels launched by the current kernel complete. Requires a per-kernel child-kernel completion counter in `RuntimeEngine`.
- V2 APIs: Same as V1 but with additional parameter-buffer size validation.

---

#### 3.10.2 Profiling / Observability

**Missing**: Full CUPTI-equivalent profiling — instruction-level sampling, concurrent kernel timeline, PC sampling, `cudaProfilerInitialize` / `cudaProfilerStart` / `cudaProfilerStop` driver-level equivalents

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

**Missing**: `CUDNN_TENSOR_NCHW_VECT_C` descriptor support, INT8x4 and INT8x32 packed-channel layouts for convolution

**New file**:
```
src/api/cudnn/cudnn_int8_packed.cpp                 (100–150 lines)
```

**Key design decisions**:
- The existing INT8 path does dequantize→FP32 compute→requantize. Packed layouts (`NCHW_VECT_C`) require un-packing (channel interleaving) before the same FP32 compute path.
- `cudnnSetTensor4dDescriptorEx` with `CUDNN_DATA_INT8x4` / `CUDNN_DATA_INT8x32` — validate that channel count is divisible by 4 or 32, then store as normal `NCHW` with stride adjustment.

---

#### 3.10.4 Graph Exec Update v2

**Missing**: `cudaGraphExecUpdate_v2`, newer graph update variants with error node reporting

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
| `src/api/cudart_shim.cpp` | ~1,300 | `cudart_shim_stream_sync.cpp`, `cudart_shim_symbols.cpp`, `cudart_shim_arrays.cpp`, `cudart_shim_texture_objects.cpp`, `cudart_shim_graphs.cpp`, `cudart_shim_capture.cpp`, `cudart_shim_device_attrs.cpp`, `cudart_shim_external_memory.cpp` |
| `src/api/cuda_driver_shim.cpp` | ~450 | `cuda_driver_shim_memory.cpp`, `cuda_driver_shim_memcpy.cpp`, `cuda_driver_shim_stream.cpp`, `cuda_driver_shim_context.cpp`, `cuda_driver_shim_texref.cpp` |
| `src/api/cublas_shim.cpp` | ~620 | `cublas_shim_level1.cpp`, `cublas_shim_level2.cpp`, `cublas_shim_level3.cpp`, `cublas_level2_backend.cpp`, `cublas_level3_backend.cpp` |
| `src/api/cudnn_shim.cpp` | ~520 | `cudnn_convolution_backward.cpp`, `cudnn_batchnorm_training.cpp`, `cudnn_backward_generic.cpp`, `cudnn_dropout.cpp`, `cudnn_rnn.cpp`, `cudnn_attention.cpp`, `cudnn_backend_api.cpp` |
| `src/core/graph_manager.cpp` | ~1,000 | `graph_manager_kernel_nodes.cpp`, `graph_manager_memset_nodes.cpp`, `graph_manager_host_nodes.cpp`, `graph_manager_introspection.cpp`, `graph_manager_exec_update.cpp`, `graph_manager_dependencies.cpp`, `graph_manager_user_objects.cpp` |
| `src/compiler/ptx_translator.cpp` | ~510 | `ptx_texture_ops.cpp`, `ptx_surface_ops.cpp`, `ptx_shared_atomics.cpp`, `ptx_conversion.cpp`, `ptx_hopper_tma_extended.cpp` |

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
| 1.3 | `cudaStreamAddCallback` | TODO | — | — |
| 1.4 | `cudaLaunchHostFunc` | TODO | — | — |
| 1.5 | `cudaGetErrorName` / `cudaGetErrorString` | TODO | — | — |
| 1.6 | `cudaMemcpyToSymbol` / `cudaMemcpyFromSymbol` | TODO | — | — |
| 1.7 | `cudaMallocArray` / `cudaMalloc3DArray` | TODO | — | — |
| 1.8 | `cudaPointerGetAttributes` | TODO | — | — |
| 1.9 | `cudaMemset2D/3D` | TODO | — | — |
| 1.10 | `cudaGraphAddKernelNode` | TODO | — | — |
| 1.11 | `cudaGraphAddMemsetNode` | TODO | — | — |
| 1.12 | `cudaGraphAddHostNode` | TODO | — | — |
| 1.13 | `cudaGraphAddChildGraphNode` | TODO | — | — |
| 1.14 | `cudaGraphAddEmptyNode` | TODO | — | — |
| 1.15 | `cudaGraphAddEventRecordNode` / `EventWaitNode` | TODO | — | — |
| 1.16 | `cudaGraphAddMemAllocNode` / `MemFreeNode` | TODO | — | — |
| 1.17 | Graph introspection APIs (`GetNodes`, `GetEdges`, etc.) | TODO | — | — |
| 1.18 | Graph exec mutation APIs | TODO | — | — |
| 1.19 | Stream capture introspection | TODO | — | — |
| 1.20 | CUDART texture/surface object APIs | TODO | — | — |
| 1.21 | CUDA Runtime device/function attributes | TODO | — | — |
| 1.22 | CUDA Graph dependencies & user objects | TODO | — | — |
| 1.23 | CUDART external memory/semaphore APIs | TODO | — | — |

### Phase 2 — cuBLAS

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 2.1 | Level-1 completion (`copy`, `swap`, `rot`, `asum`, `amax`, etc.) | TODO | — | — |
| 2.2 | Pointer mode / atomics mode APIs | TODO | — | — |
| 2.3 | Level-2 BLAS (`Trsv`, `Trsm`, `Ger`, `Symv`, `Gbmv`, `Syr`, etc.) | TODO | — | — |
| 2.4 | Level-3 BLAS (`Trsm`, `Syrk`, `Syr2k`, `Trmm`, `Symm`, `Chemm`, etc.) | TODO | — | — |

### Phase 3 — cuDNN

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 3.1 | `cudnnConvolutionBackwardData` | TODO | — | — |
| 3.2 | `cudnnConvolutionBackwardFilter` | TODO | — | — |
| 3.3 | `cudnnBatchNormalizationForwardTraining` | TODO | — | — |
| 3.4 | `cudnnBatchNormalizationBackward` | TODO | — | — |
| 3.5 | `cudnnActivationBackward` | TODO | — | — |
| 3.6 | `cudnnSoftmaxBackward` | TODO | — | — |
| 3.7 | `cudnnPoolingBackward` | TODO | — | — |
| 3.8 | `cudnnDropoutForward/Backward` | TODO | — | — |
| 3.9 | `cudnnRNNForward/Backward` | TODO | — | — |
| 3.10 | `cudnnMultiHeadAttnForward/Backward` | TODO | — | — |
| 3.11 | `cudnnCTCLoss` | TODO | — | — |
| 3.12 | `cudnnOpTensor` | TODO | — | — |
| 3.13 | `cudnnReduceTensor` | TODO | — | — |
| 3.14 | `cudnnTransformTensor` | TODO | — | — |
| 3.15 | cuDNN Backend API (v8+) | TODO | — | — |

### Phase 4 — Missing Libraries

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 4.1 | cuFFT stub shim | TODO | — | — |
| 4.2 | cuFFT 1D/2D/3D functional (FFTW3 delegation) | TODO | — | — |
| 4.3 | cuRAND stub shim | TODO | — | — |
| 4.4 | cuRAND host API functional | TODO | — | — |
| 4.5 | cuSOLVER stub shim | TODO | — | — |
| 4.6 | cuSOLVER dense functional (LAPACK delegation) | TODO | — | — |
| 4.7 | cuSPARSE stub shim | TODO | — | — |
| 4.8 | cuSPARSE SpMV/SpMM functional | TODO | — | — |
| 4.9 | cuBLASLt stub shim | TODO | — | — |
| 4.10 | cuBLASLt matmul + epilogue functional | TODO | — | — |

### Phase 5 — CUDA Driver API

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 5.1 | `cuMemAllocManaged` | TODO | — | — |
| 5.2 | `cuMemHostAlloc` / `Register` / `Unregister` | TODO | — | — |
| 5.3 | `cuMemcpy2D/3D` / `Async` variants | TODO | — | — |
| 5.4 | `cuStreamAddCallback` / `Query` / introspection | TODO | — | — |
| 5.5 | `cuEventQuery` | TODO | — | — |
| 5.6 | Context management (`GetLimit`, `SetLimit`, `Pop`, `Push`, etc.) | TODO | — | — |
| 5.7 | Device queries (`GetUuid`, `GetP2PAttribute`, etc.) | TODO | — | — |
| 5.8 | `cuGraph*` family | TODO | — | — |
| 5.9 | `cuExternalMemory*` / `cuExternalSemaphore*` | TODO | — | — |
| 5.10 | Driver texture reference gaps (`cuTexRefSetAddress2D`, filter, mipmap, border) | TODO | — | — |

### Phase 6 — PTX ISA

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 6.1 | `tex` / `tld4` / `txq` instructions | TODO | — | — |
| 6.2 | `suld` / `sust` instructions | TODO | — | — |
| 6.3 | `atom.shared.*` | TODO | — | — |
| 6.4 | Missing `cvt.*` variants | TODO | — | — |
| 6.5 | `rcp.rn` / `sqrt.rn` / `div.rn` | TODO | — | — |
| 6.6 | FP16 vector loads (`ld.global.v2/v4.f16`) | TODO | — | — |
| 6.7 | `match.sync` / `elect.sync` | TODO | — | — |
| 6.8 | `grid.sync` / `griddepcontrol` | TODO | — | — |
| 6.9 | TMA 3D/4D/5D (`cp.async.bulk.tensor`) | TODO | — | — |
| 6.10 | `tcgen05.*` (Blackwell) | TODO | — | — |

### Phase 7 — NCCL

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 7.1 | `ncclSend` / `ncclRecv` | TODO | — | — |
| 7.2 | `ncclAllToAll` | TODO | — | — |
| 7.3 | `ncclGather` / `ncclScatter` | TODO | — | — |

### Phase 8 — Cooperative Groups & Device Libraries

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 8.1 | `thread_block` cooperative group | TODO | — | — |
| 8.2 | `coalesced_group` | TODO | — | — |
| 8.3 | `reduce()` / `partition()` / `shfl()` | TODO | — | — |
| 8.4 | `thread_block_tile` | TODO | — | — |
| 8.5 | `multi_grid` | TODO | — | — |
| 8.6 | CUB / Thrust fallback headers | TODO | — | — |

### Phase 9 — Deployment

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 9.1 | Kubernetes Device Plugin | TODO | — | — |
| 9.2 | SLURM GRES Plugin | TODO | — | — |

### Phase 10 — CDP, Profiling, Advanced Formats, Graph Updates

| # | Feature | Status | PR | Date |
|---|---|---|---|---|
| 10.1 | CDP full API (`cudaDeviceSynchronize`, `GetParameterBufferV2`, `LaunchDeviceV2`) | TODO | — | — |
| 10.2 | Profiling / CUPTI-equivalent (kernel timeline, instruction sampler) | TODO | — | — |
| 10.3 | cuDNN INT8x4 / INT8x32 packed layouts | TODO | — | — |
| 10.4 | `cudaGraphExecUpdate_v2` | TODO | — | — |

---

**End of Document**
