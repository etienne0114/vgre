# VGRE — Missing Features & Implementation Roadmap

**Research Date**: 2026-05-12 (deep audit)  
**Audit Methodology**: Full `src/` and `include/` grep + manual file inspection; every "missing" claim verified by absence in source code. All previously-claimed "missing" items were re-verified against the actual source.  
**Goal**: Production-ready deployment for PyTorch, TensorFlow, JAX, and distributed ML workloads.

---

## Implemented Summary (verified)

| Category | Count | Key Implemented APIs |
|---|---|---|
| CUDA Runtime API | ~97 | `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemcpy2D/Async`, `cudaMemcpyAsync`, `cudaMemcpyPeer/Async`, `cudaMallocAsync`, `cudaFreeAsync`, `cudaMallocFromPoolAsync`, `cudaMemPoolCreate/Destroy/TrimTo/SetAttribute/GetAttribute/SetAccess/GetAccess/ExportToShareableHandle/ImportFromShareableHandle/ExportPointer/ImportPointer`, `cudaMemAdvise`, `cudaMemPrefetchAsync`, `cudaMemRangeGetAttribute/Attributes`, `cudaHostAlloc`, `cudaHostRegister`, `cudaHostUnregister`, `cudaStreamCreate/CreateWithFlags/CreateWithPriority`, `cudaStreamDestroy/Query/Synchronize/WaitEvent/AddCallback`, `cudaStreamBeginCapture/BeginCaptureToGraph/EndCapture`, `cudaStreamGetId/GetDevice/GetFlags/GetPriority/GetAttribute/SetAttribute`, `cudaEventCreate/CreateWithFlags/Destroy/Record/Query/Synchronize/ElapsedTime`, `cudaDeviceGetAttribute/GetByPCIBusId/GetPCIBusId/GetStreamPriorityRange/CanAccessPeer`, `cudaDeviceEnablePeerAccess/DisablePeerAccess`, `cudaDeviceReset/Synchronize`, `cudaSetDevice/GetDevice/GetDeviceCount`, `cudaGetDeviceFlags/SetDeviceFlags/GetDeviceProperties`, `cudaGetLastError/PeekAtLastError`, `cudaDriverGetVersion/RuntimeGetVersion`, `cudaProfilerStart/ProfilerStop`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor/WithFlags`, `cudaLaunchKernel`, `cudaLaunchCooperativeKernel/MultiDevice`, `cudaMallocPitch/MipmappedArray`, `cudaFreeHost/FreeMipmappedArray`, `cudaGenerateMipmaps`, `cudaGetMipmappedArrayLevel`, `cudaGetSymbolAddress`, `cudaMemset/MemsetAsync`, `cudaMemGetInfo`, `cudaMemcpyBatchAsync`, `cudaMemcpy3DBatchAsync`, `cudaGraphCreate/Destroy/Clone`, `cudaGraphInstantiate/Launch`, `cudaGraphExecDestroy/Update`, `cudaGraphAddMemcpyNode/MemcpyNode1D/ConditionalNode/ExternalSemaphoreSignalNode/ExternalSemaphoreWaitNode`, `cudaGraphExecExternalSemaphoreSignalNodeSetParams/ExternalSemaphoreWaitNodeSetParams`, `cudaSignalExternalSemaphoresAsync/WaitExternalSemaphoresAsync`, `cudaImportExternalSemaphore/DestroyExternalSemaphore` |
| CUDA Driver API | ~46 | `cuCtxCreate/Destroy/GetCurrent/SetCurrent/Synchronize`, `cuDeviceGet/GetAttribute/GetCount/GetName/TotalMem`, `cuEventCreate/Destroy/ElapsedTime/Record/Synchronize`, `cuInit`, `cuLaunchKernel`, `cuMemAlloc/Free`, `cuMemcpyDtoD/DtoH/HtoD`, `cuModuleGetFunction/GetGlobal/GetSurfRef/GetTexRef/Load/LoadData/LoadDataEx/Unload`, `cuStreamCreate/Destroy/Synchronize/WaitEvent`, `cuSurfObjectCreate/Destroy`, `cuSurfRefGetArray/SetArray`, `cuTexObjectCreate/Destroy`, `cuTexRefCreate/GetAddress/SetAddress/SetFlags/SetFormat`, `cuMemCreate/Release/AddressReserve/Map/Unmap/SetAccess`, `cuMulticastCreate/AddDevice/BindMem/GetGranularity/Unbind`, plus IPC memory/event handles |
| cuBLAS | ~27 | `cublasCreate/Destroy` (v2), `cublasGet/SetStream/SetMathMode/GetMathMode/GetVersion`, `cublasSgemm/Dgemm/Hgemm/Sgemv` (v2), `cublasSaxpy/Daxpy/Sdot/Ddot/Snrm2/Sscal` (v2), `cublasDgemmBatched/StridedBatched`, `cublasSgemmBatched/StridedBatched`, `cublasGemmEx/GemmBatchedEx/GemmStridedBatchedEx` |
| cuDNN | ~36 | `cudnnCreate/Destroy`, descriptors (tensor, filter, conv, pooling, activation), `cudnnSetTensor4dDescriptor/GetTensor4dDescriptor`, `cudnnSetFilter4dDescriptor`, `cudnnSetConvolution2dDescriptor/GetConvolution2dForwardOutputDim/FindConvolutionForwardAlgorithm/GetConvolutionForwardAlgorithm_v7/GetConvolutionForwardWorkspaceSize`, `cudnnSetPooling2dDescriptor/GetPooling2dDescriptor/GetPooling2dForwardOutputDim`, `cudnnSetActivationDescriptor`, `cudnnActivationForward/Backward`, `cudnnConvolutionForward`, `cudnnPoolingForward`, `cudnnSoftmaxForward`, `cudnnBatchNormalizationForwardInference`, `cudnnGetVersion/GetStream/SetStream` |
| NCCL | ~16 | `ncclCommInitRank/InitAll/Destroy/Abort/Count/UserRank`, `ncclGetUniqueId/Version/LastError`, `ncclAllReduce/Broadcast/Reduce/AllGather/ReduceScatter`, `ncclGroupStart/GroupEnd` |
| NVTX | ~26 | `nvtxMarkA/MarkEx`, `nvtxRangeStartA/StartEx/End`, `nvtxRangePushA/PushEx/Pop`, domains (`CreateA/CreateW/Create/Destroy/MarkEx/RangeStartEx/RangeEnd/RangePushEx/RangePop/RegisterStringA/RegisterStringW`), resource naming (`NameCategoryA/NameCudaDeviceA/NameCudaDeviceEx/NameCudaEventA/NameCudaEventEx/NameCudaStreamA/NameCudaStreamEx/NameCudaThreadEx/NameOsThreadA`) |
| PTX ISA | ~120 | Integer (`add/sub/mul/mad/div/rem/neg` s32/u32/s64/u64, `mul.hi`, `add.cc/addc.cc/sub.cc/subc.cc`, `mad.hi.cc`), FP (`add/sub/mul/div/fma/sqrt/rsqrt/rcp/neg/abs/min/max` f32/f64, `add.rn/mul.rn`), bitwise (`and/or/xor/not/shl/shr`), memory (`ld/st.global/shared/local` f32/u32/s32/f64/u64/s64/v2.f32/v4.f32/nc/cs), atomics (`atom.global.add/cas/exch/max/min` f32/s32/u32/u64, `red.global.add`), warp (`vote.sync`, `shfl.sync.idx/down/up/bfly`, `activemask`, `redux.sync`, `bar.red`), control (`bar.sync/arrive`, `membar.gl/sys/cta`, `ret`), MMA (`mma.sync.aligned` Ampere shapes, `wgmma.mma_async` Hopper shapes, `cp.async.bulk.tensor` 1D/2D, `wgmma.fence/commit_group/wait_group`), `lop3.b32`, `popc/clz`, `bitcast`, special-register `mov`, FP intrinsics (`ex2/lg2/sin/cos`), `setp/ selp` |
| Cooperative Groups | ~1 type | `grid_group` with `sync()`, `size()`, `thread_rank()`; `this_grid()` in `cpu_cuda_env.h` |
| Texture / Surface (C++) | ~6 | `tex1D/tex2D/tex3D/tex1Dfetch`, `surf2Dread/surf2Dwrite` templates in `cpu_cuda_env.h` |
| BFloat16 (C++) | ~5 | `__nv_bfloat16`, `__float2bfloat16`, `__bfloat162float`, `__hadd_bf`, `__hmul_bf` in `cpu_cuda_env.h` |
| Dynamic Parallelism (device) | ~2 | `cudaLaunchDevice`, `cudaGetParameterBuffer` in `cpu_cuda_env.h` |

---

## Audit Summary

The previous `missingFeatures.md` (dated 2026-05-12) **dangerously overstated implementation completeness**. A line-by-line source-code audit reveals that while the *core* memory, stream, and launch paths are solid, large surface areas of the CUDA Runtime API, CUDA Graphs, cuBLAS, cuDNN, NCCL, and PTX ISA are either **completely missing** or **only partially stubbed**.

| Category | Estimated Coverage | Reality Check |
|---|---|---|
| CUDA Runtime API | ~45% | 97 functions implemented; ~117+ missing including `cudaMemcpyToSymbol`, `cudaFuncGetAttributes`, texture/surface objects, array APIs, graph kernel nodes, stream capture introspection |
| CUDA Driver API | ~15% | 46 functions implemented; 250+ missing including `cuEventQuery`, `cuStreamAddCallback`, `cuMemAllocManaged`, `cuMemHostRegister`, 2D/3D copies, grid-level sync, context limits, cooperative launch, graph APIs |
| CUDA Graphs | ~30% | Internal `GraphManager` has KERNEL, MEMCPY, CONDITIONAL nodes. CUDART shim only exposes MEMCPY, CONDITIONAL, EXTERNAL-SEMAPHORE nodes. KERNEL, MEMSET, HOST, CHILD, EMPTY, EVENT, MEM-ALLOC/FREE nodes missing from shim. |
| PTX ISA Coverage | ~30% | Core arithmetic + warp shuffle + basic atomics + Ampere/Hopper MMA are present. Missing: texture/surface loads, 3D+ TMA, `tcgen05`, shared atomics, FP16 vector loads, `match.sync`, `grid.sync`, `prmt`, `rcp.rn`, `sqrt.rn`, most `cvt` variants. |
| cuBLAS | ~13% | 27 functions implemented; 200+ missing. Only `Gemm`, `Gemv`, `Axpy`, `Dot`, `Nrm2`, `Scal`. Missing: all Level-2 except `Gemv`, all Level-3 except `Gemm`, triangular solve (`Trsm`/`Trsv`), symmetric (`Syrk`/`Syr2k`), batched `Trsm`, pointer modes. |
| cuDNN | ~24% | 36 functions implemented; 120+ missing. Forward-only: conv, activation, pooling, softmax, BN inference. Missing: all backward passes, BN training, dropout, RNN/LSTM/GRU, attention, CTC loss, `OpTensor`, `ReduceTensor`, `TransformTensor`. |
| NCCL | ~55% | AllReduce, Broadcast, Reduce, AllGather, ReduceScatter present. Missing: point-to-point `Send`/`Recv`, `AllToAll`, `Gather`, `Scatter`. |
| Profiling / Observability | ~65% | NVTX (~26/40), basic OTLP export present. Missing: CUPTI-equivalent kernel timestamps, instruction-level sampling, concurrent kernel profiling. |
| Texture / Surface | ~20% | Texture objects (`cuTexObjectCreate/Destroy`) + legacy `cuTexRef` partially present. Surface references return `CUDA_ERROR_NOT_SUPPORTED`. Missing `cuTexRefSetAddress2D`, filter modes, mipmap controls, array binding, CUDART texture-object APIs. |
| Entire Libraries | 0% | **cuFFT, cuRAND, cuSOLVER, cuSPARSE, cuBLASLt** — no shims exist anywhere in the codebase. |

---

## Tier 1 — Critical Missing (blocks production PyTorch/TensorFlow)

### 1.1 CUDA Runtime API — Stream Synchronization & Queries

| # | Missing API | Impact | Verification |
|---|---|---|---|
| 1.1.1 | `cudaStreamWaitEvent` | ✅ **IMPLEMENTED** 2026-05-13 — CUDART shim now forwards to `CUDAInterceptor::streamWaitEvent`. Test: `tests/api/test_stream_wait_event.cpp`. | `grep -rn "^cudaError_t cudaStreamWaitEvent" src/api/cudart_shim_stream.cpp` → match |
| 1.1.2 | `cudaEventQuery` | ✅ **IMPLEMENTED** 2026-05-13 — uses `Event::isQueryReady()`; returns `cudaSuccess` if recorded, `cudaErrorNotReady` otherwise. Test: `tests/api/test_event_query.cpp`. | `grep -rn "^cudaError_t cudaEventQuery" src/api/cudart_shim_stream.cpp` → match |
| 1.1.3 | `cudaStreamAddCallback` | ✅ **IMPLEMENTED** 2026-05-13 — enqueues a lambda via `Scheduler::submitStreamTask()` that invokes the user callback with `cudaSuccess` after all prior stream work completes. Test: `tests/api/test_stream_add_callback.cpp`. | `grep -rn "^cudaError_t cudaStreamAddCallback" src/api/cudart_shim_stream.cpp` → match |
| 1.1.4 | `cudaLaunchHostFunc` | High — modern replacement for stream callbacks. | Absent |
| 1.1.5 | `cudaGetErrorName` / `cudaGetErrorString` | Medium — error introspection required for robust error handling. | Absent |

### 1.2 CUDA Runtime API — Memory & Symbols

| # | Missing API | Impact | Verification |
|---|---|---|---|
| 1.2.1 | `cudaMemcpyToSymbol` / `cudaMemcpyToSymbolAsync` / `cudaMemcpyFromSymbol` / `cudaMemcpyFromSymbolAsync` | **CRITICAL** — device-symbol constant memory transfers used by many kernels. | Absent |
| 1.2.2 | `cudaMallocArray` / `cudaMalloc3DArray` | High — CUDA arrays required for texture/surface pipelines. | Absent |
| 1.2.3 | `cudaMalloc3D` | Medium — 3D pitched allocations. | Absent |
| 1.2.4 | `cudaMemcpy3DAsync` | Medium — `cudaMemcpy3DBatchAsync` exists; standalone 3D async memcpy missing. | Absent |
| 1.2.5 | `cudaHostGetDevicePointer` / `cudaHostGetFlags` | Medium — pinned-memory introspection. | Absent |
| 1.2.6 | `cudaArrayGetInfo` / `cudaArrayDestroy` | Medium — array lifetime management. | Absent |
| 1.2.7 | `cudaPointerGetAttributes` | Medium — generic pointer introspection (host/device/managed). | Absent |
| 1.2.8 | `cudaMemset2D` / `cudaMemset3D` / `cudaMemset2DAsync` / `cudaMemset3DAsync` | Medium — pitched/3D memset. | Absent |
| 1.2.9 | `cudaMemcpyToArray` / `cudaMemcpyFromArray` / `cudaMemcpy2DToArray` / `cudaMemcpy2DFromArray` | Medium — array transfer APIs. | Absent |

### 1.3 CUDA Graphs — Kernel & Essential Node Types

| # | Missing API / Node Type | Impact | Verification |
|---|---|---|---|
| 1.3.1 | `cudaGraphAddKernelNode` | **CRITICAL** — `GraphManager::addKernelNode` exists internally, but the CUDART shim does **not** expose `cudaGraphAddKernelNode`, so frameworks cannot add kernel nodes to graphs via the runtime API. | No match in `src/api/cudart_shim.cpp` |
| 1.3.2 | `cudaGraphAddMemsetNode` | High — graph-based memory initialization. | Absent |
| 1.3.3 | `cudaGraphAddHostNode` | Medium — CPU callbacks inside graphs. | Absent |
| 1.3.4 | `cudaGraphAddChildGraphNode` | Medium — graph nesting. | Absent |
| 1.3.5 | `cudaGraphAddEmptyNode` | Medium — dependency placeholders. | Absent |
| 1.3.6 | `cudaGraphAddEventRecordNode` / `cudaGraphAddEventWaitNode` | Medium — graph-internal event synchronization. | Absent |
| 1.3.7 | `cudaGraphAddMemAllocNode` / `cudaGraphAddMemFreeNode` | Medium — stream-ordered graph memory. | Absent |

### 1.4 CUDA Graphs — Node Introspection & Exec Mutation

| # | Missing API | Impact | Verification |
|---|---|---|---|
| 1.4.1 | `cudaGraphKernelNodeSetParams` / `cudaGraphKernelNodeGetParams` | High — dynamic kernel parameter updates. | Absent |
| 1.4.2 | `cudaGraphMemcpyNodeGetParams` / `cudaGraphMemsetNodeGetParams` / `cudaGraphMemsetNodeSetParams` | Medium | Absent |
| 1.4.3 | `cudaGraphHostNodeGetParams` / `cudaGraphHostNodeSetParams` | Medium | Absent |
| 1.4.4 | `cudaGraphNodeGetType` | Medium | Absent |
| 1.4.5 | `cudaGraphGetNodes` / `cudaGraphGetRootNodes` / `cudaGraphGetEdges` | Medium | Absent |
| 1.4.6 | `cudaGraphNodeGetDependencies` / `cudaGraphNodeGetDependentNodes` | Medium | Absent |
| 1.4.7 | `cudaGraphExecKernelNodeSetParams` | High — executable graph kernel updates. | Absent |
| 1.4.8 | `cudaGraphExecMemcpyNodeSetParams` / `cudaGraphExecMemsetNodeSetParams` / `cudaGraphExecHostNodeSetParams` / `cudaGraphExecChildGraphNodeSetParams` | Medium | Absent |
| 1.4.9 | `cudaGraphExecEventRecordNodeSetEvent` / `cudaGraphExecEventWaitNodeSetEvent` | Medium | Absent |
| 1.4.10 | `cudaGraphInstantiateWithFlags` / `cudaGraphInstantiateWithParams` | Medium — modern graph instantiation options. | Absent |
| 1.4.11 | `cudaGraphExecGetFlags` | Low | Absent |
| 1.4.12 | `cudaGraphUpload` | Medium — upload exec graph without re-instantiating. | Absent |
| 1.4.13 | `cudaGraphNodeSetEnabled` / `cudaGraphNodeGetEnabled` | Medium — toggle nodes in exec graph. | Absent |
| 1.4.14 | `cudaGraphExecNodeSetParams` | Medium — generic exec node param updates. | Absent |
| 1.4.15 | `cudaGraphKernelNodeCopyAttributes` | Low | Absent |

### 1.5 CUDA Graphs — Dependencies & User Objects

| # | Missing API | Impact | Verification |
|---|---|---|---|
| 1.5.1 | `cudaGraphAddDependencies` / `cudaGraphRemoveDependencies` | Medium — explicit dependency editing. | Absent |
| 1.5.2 | `cudaGraphRetainUserObject` / `cudaGraphReleaseUserObject` | Low — reference-counted objects tied to graph lifetime. | Absent |
| 1.5.3 | `cudaUserObjectCreate` / `cudaUserObjectRetain` / `cudaUserObjectRelease` | Low — standalone user object lifetime. | Absent |
| 1.5.4 | `cudaGraphNodeFindInClone` | Low | Absent |
| 1.5.5 | `cudaGraphDebugDotPrint` | Low — graph visualization for debugging. | Absent |

### 1.6 CUDA Runtime — Stream Capture Introspection

| # | Missing API | Impact | Verification |
|---|---|---|---|
| 1.6.1 | `cudaStreamIsCapturing` | High — query if a stream is in capture mode. | Absent |
| 1.6.2 | `cudaStreamGetCaptureInfo` / `cudaStreamGetCaptureInfo_v2` | High — capture state introspection. | Absent |
| 1.6.3 | `cudaThreadExchangeStreamCaptureMode` | Medium — switch capture mode per-thread. | Absent |
| 1.6.4 | `cudaStreamUpdateCaptureDependencies` | Medium — modify capture dependency frontier. | Absent |
| 1.6.5 | `cudaStreamCopyAttributes` | Low — copy stream attributes. | Absent |

---

## Tier 2 — High Priority Missing (needed for full framework compatibility)

### 2.1 CUDA Driver API

| # | Missing API | Notes |
|---|---|---|
| 2.1.1 | `cuMemAllocManaged` | Unified Memory allocation |
| 2.1.2 | `cuMemHostAlloc` / `cuMemHostGetDevicePointer` / `cuMemHostRegister` / `cuMemHostUnregister` | Pinned / registered host memory |
| 2.1.3 | `cuMemAllocPitch` | Pitched device allocations |
| 2.1.4 | `cuMemcpy2D` / `cuMemcpy2DAsync` / `cuMemcpy3D` / `cuMemcpy3DAsync` | Structured memory copies |
| 2.1.5 | `cuMemcpyDtoDAsync` / `cuMemcpyDtoHAsync` / `cuMemcpyHtoDAsync` / `cuMemcpyAsync` | Async structured copies |
| 2.1.6 | `cuMemcpyPeer` / `cuMemcpyPeerAsync` | Peer memory copies |
| 2.1.7 | `cuMemsetD8` / `cuMemsetD16` / `cuMemsetD32` / `cuMemsetD2D8` / `cuMemsetD2D16` / `cuMemsetD2D32` | Device memset variants |
| 2.1.8 | `cuLaunchCooperativeKernel` / `cuLaunchCooperativeKernelMultiDevice` | Cooperative multi-thread-block launches |
| 2.1.9 | `cuLaunchKernelEx` | Extended kernel launch (params struct) |
| 2.1.10 | `cuStreamAddCallback` | Driver-level stream callbacks |
| 2.1.11 | `cuStreamQuery` / `cuStreamGetFlags` / `cuStreamGetPriority` / `cuStreamGetId` / `cuStreamGetCtx` | Stream introspection |
| 2.1.12 | `cuStreamGetCaptureInfo` / `cuStreamIsCapturing` / `cuStreamUpdateCaptureDependencies` | Stream capture introspection |
| 2.1.13 | `cuEventQuery` | Non-blocking event status |
| 2.1.14 | `cuModuleLoadFatBinary` / `cuModuleLink*` / `cuModuleGetLoadingMode` | Module linking / fatbinary |
| 2.1.15 | `cuTexRefSetAddress2D` / `cuTexRefSetArray` / `cuTexRefSetAddressMode` / `cuTexRefSetFilterMode` / `cuTexRefSetMaxAnisotropy` / `cuTexRefSetMipmap*` / `cuTexRefSetBorderColor` | Texture sampling controls |
| 2.1.16 | `cuSurfRefSetFormat` | Surface reference format |
| 2.1.17 | `cuCtxGetDevice` / `cuCtxGetFlags` / `cuCtxGetLimit` / `cuCtxSetLimit` / `cuCtxGetCacheConfig` / `cuCtxSetCacheConfig` / `cuCtxGetSharedMemConfig` / `cuCtxSetSharedMemConfig` / `cuCtxGetStreamPriorityRange` / `cuCtxGetId` / `cuCtxGetApiVersion` / `cuCtxPopCurrent` / `cuCtxPushCurrent` / `cuCtxAttach` / `cuCtxDetach` | Context management |
| 2.1.18 | `cuDeviceGetUuid` / `cuDeviceGetTexture1DLinearMaxWidth` / `cuDeviceGetP2PAttribute` / `cuDeviceGetNvSciSyncAttributes` / `cuDeviceGetGraphMemAttribute` / `cuDeviceSetGraphMemAttribute` / `cuDeviceFlushGPUDirectRDMAWrites` | Device queries |
| 2.1.19 | `cuOccupancyMaxActiveBlocksPerMultiprocessor` / `cuOccupancyMaxPotentialBlockSize` / `cuOccupancyMaxPotentialBlockSizeWithFlags` | Occupancy |
| 2.1.20 | `cuGraph*` family | All driver graph APIs missing |
| 2.1.21 | `cuExternalMemory*` / `cuExternalSemaphore*` families | External resource interop |
| 2.1.22 | `cuProfilerStart` / `cuProfilerStop` | Driver profiler |
| 2.1.23 | `cuGetErrorName` / `cuGetErrorString` | Error introspection |

### 2.2 CUDA Runtime — Device & Function Attributes

| # | Missing API | Notes |
|---|---|---|
| 2.2.1 | `cudaFuncGetAttributes` | Kernel occupancy / register introspection |
| 2.2.2 | `cudaFuncSetCacheConfig` / `cudaFuncSetSharedMemConfig` | L1 / shared memory split config |
| 2.2.3 | `cudaDeviceGetLimit` / `cudaDeviceSetLimit` | Stack size, malloc heap size, etc. |
| 2.2.4 | `cudaDeviceGetCacheConfig` / `cudaDeviceSetCacheConfig` | L1/shared config |
| 2.2.5 | `cudaDeviceGetSharedMemConfig` / `cudaDeviceSetSharedMemConfig` | Shared memory bank size |
| 2.2.6 | `cudaChooseDevice` | Device selection heuristic |
| 2.2.7 | `cudaThreadExit` / `cudaThreadSynchronize` / `cudaThreadSetLimit` / `cudaThreadGetLimit` | Legacy thread APIs |
| 2.2.8 | `cudaDeviceGetP2PAttribute` | Peer access attribute queries |
| 2.2.9 | `cudaDeviceFlushGPUDirectRDMAWrites` | RDMA flush |
| 2.2.10 | `cudaDeviceGetGraphMemAttribute` / `cudaDeviceSetGraphMemAttribute` | Graph memory pool attributes |
| 2.2.11 | `cudaLaunchKernelExC` / `cudaLaunchConfig` | Extended / configurable kernel launch |

### 2.3 Texture & Surface — CUDART Object APIs

| # | Missing API | Notes |
|---|---|---|
| 2.3.1 | `cudaCreateTextureObject` / `cudaDestroyTextureObject` | Runtime texture object creation |
| 2.3.2 | `cudaGetTextureObjectResourceDesc` / `cudaGetTextureObjectTextureDesc` / `cudaGetTextureObjectResourceViewDesc` | Texture object introspection |
| 2.3.3 | `cudaCreateSurfaceObject` / `cudaDestroySurfaceObject` | Runtime surface object creation |
| 2.3.4 | `cudaGetSurfaceObjectResourceDesc` | Surface object introspection |
| 2.3.5 | `cudaGetTextureReference` / `cudaGetSurfaceReference` | Reference lookup |
| 2.3.6 | `cudaBindTexture` / `cudaUnbindTexture` / `cudaBindTextureToArray` / `cudaBindTexture2D` | Legacy linear/2D texture binding |
| 2.3.7 | `cudaBindSurfaceToArray` | Surface binding |

### 2.4 External Memory & Semaphore — CUDART

| # | Missing API | Notes |
|---|---|---|
| 2.4.1 | `cudaImportExternalMemory` / `cudaDestroyExternalMemory` | External memory import |
| 2.4.2 | `cudaExternalMemoryGetMappedBuffer` / `cudaExternalMemoryGetMappedMipmappedArray` | External memory mapping |
| 2.4.3 | `cudaExternalSemaphoreGetSignalNodeParams` / `cudaExternalSemaphoreGetWaitNodeParams` | External semaphore node params |

---

## Tier 3 — Medium Priority Missing (needed for full BLAS/DNN coverage)

### 3.1 cuBLAS — Level 1 BLAS

| # | Missing Function | Notes |
|---|---|---|
| 3.1.1 | `cublasScopy` / `cublasDcopy` | Vector copy |
| 3.1.2 | `cublasSswap` / `cublasDswap` | Vector swap |
| 3.1.3 | `cublasSrot` / `cublasDrot` / `cublasSrotm` / `cublasSrotmg` / `cublasSrotg` | Givens rotations |
| 3.1.4 | `cublasSasum` / `cublasDasum` | Absolute sum |
| 3.1.5 | `cublasIsamax` / `cublasIdamax` | Index of max absolute value |
| 3.1.6 | `cublasDnrm2` / `cublasDscal` | Already have `Snrm2`/`Sscal`; `D` variants missing |
| 3.1.7 | `cublasGetPointerMode` / `cublasSetPointerMode` | Host/device pointer mode |
| 3.1.8 | `cublasGetAtomicsMode` / `cublasSetAtomicsMode` | Atomics control |
| 3.1.9 | `cublasLoggerConfigure` | Logging callback setup |

### 3.2 cuBLAS — Level 2 BLAS (all missing except `Gemv`)

| # | Missing Function | Notes |
|---|---|---|
| 3.2.1 | `cublasStrsv` / `cublasDtrsv` | Triangular solve |
| 3.2.2 | `cublasStrsm` / `cublasDtrsm` | Triangular solve matrix |
| 3.2.3 | `cublasSger` / `cublasDger` | Rank-1 update |
| 3.2.4 | `cublasSsymv` / `cublasDsymv` | Symmetric matrix-vector multiply |
| 3.2.5 | `cublasSgbmv` / `cublasDgbmv` | Banded matrix-vector multiply |
| 3.2.6 | `cublasSsyr` / `cublasDsyr` / `cublasSsyr2` / `cublasDsyr2` | Symmetric rank update |
| 3.2.7 | `cublasStbsv` / `cublasStpsv` / `cublasSspmv` / `cublasSsbmv` / `cublasSspr` / `cublasSspr2` / `cublasStbmv` / `cublasStpmv` / `cublasStrmv` | Packed / banded / triangular variants |

### 3.3 cuBLAS — Level 3 BLAS (all missing except `Gemm`)

| # | Missing Function | Notes |
|---|---|---|
| 3.3.1 | `cublasStrsm` / `cublasDtrsm` | Triangular solve (Level 3) |
| 3.3.2 | `cublasSsyrk` / `cublasDsyrk` / `cublasSsyr2k` / `cublasDsyr2k` | Symmetric rank-k update |
| 3.3.3 | `cublasStrmm` / `cublasDtrmm` | Triangular matrix multiply |
| 3.3.4 | `cublasSsymm` / `cublasDsymm` | Symmetric matrix multiply |
| 3.3.5 | `cublasChemm` / `cublasCherk` / `cublasCher2k` | Complex Hermitian variants |
| 3.3.6 | `cublas*trsmBatched` / `cublas*syrkBatched` | Batched Level-3 |

### 3.4 cuDNN — Backward Passes

| # | Missing Function | Notes |
|---|---|---|
| 3.4.1 | `cudnnConvolutionBackwardData` | Data gradient |
| 3.4.2 | `cudnnConvolutionBackwardFilter` | Filter gradient |
| 3.4.3 | `cudnnConvolutionBackwardBias` | Bias gradient |
| 3.4.4 | `cudnnBatchNormalizationForwardTraining` | Training-mode BN |
| 3.4.5 | `cudnnBatchNormalizationBackward` | BN gradients |
| 3.4.6 | `cudnnActivationBackward` | Already implemented! (keep) |
| 3.4.7 | `cudnnSoftmaxBackward` | Softmax gradient |
| 3.4.8 | `cudnnPoolingBackward` | Pooling gradient |

### 3.5 cuDNN — Advanced Layers

| # | Missing Function | Notes |
|---|---|---|
| 3.5.1 | `cudnnDropoutForward` / `cudnnDropoutBackward` | Dropout regularization |
| 3.5.2 | `cudnnRNNForward` / `cudnnRNNBackward` | LSTM / GRU / vanilla RNN |
| 3.5.3 | `cudnnMultiHeadAttnForward` / `cudnnMultiHeadAttnBackward` | Transformer attention |
| 3.5.4 | `cudnnCTCLoss` | CTC loss for OCR/speech |
| 3.5.5 | `cudnnDivisiveNormalizationForward` / `Backward` | LRN variants |
| 3.5.6 | `cudnnLRNCrossChannelForward` / `Backward` | Local response normalization |
| 3.5.7 | `cudnnTransformTensor` | Data type / layout conversion |
| 3.5.8 | `cudnnAddTensor` | Broadcasted elementwise add |
| 3.5.9 | `cudnnOpTensor` | Generic elementwise ops |
| 3.5.10 | `cudnnReduceTensor` | Reductions (sum, max, etc.) |

### 3.6 NCCL — Point-to-Point & Advanced Collectives

| # | Missing Function | Notes |
|---|---|---|
| 3.6.1 | `ncclSend` / `ncclRecv` | Point-to-point required for pipeline parallelism |
| 3.6.2 | `ncclAllToAll` | All-to-all exchange (MoE, expert parallelism) |
| 3.6.3 | `ncclGather` / `ncclScatter` | Classic collectives |

### 3.7 Entire Libraries — No Shim Exists

| # | Missing Library | Notes |
|---|---|---|
| 3.7.1 | **cuFFT** | Fast Fourier Transform — no files, no stubs, no references anywhere in `src/` or `include/` |
| 3.7.2 | **cuRAND** | Random number generation — completely absent |
| 3.7.3 | **cuSOLVER** | Dense/sparse linear solvers (QR, SVD, Cholesky, eigenvalues) — completely absent |
| 3.7.4 | **cuSPARSE** | Sparse matrix operations (SpMV, SpMM, CSR/CSC/COO) — completely absent |
| 3.7.5 | **cuBLASLt** | Lightweight GEMM with fused epilogues — completely absent |
| 3.7.6 | **cuDNN Backend API** | Only legacy frontend API partially implemented; `cudnnBackend*` descriptors, engines, and execution plans missing |

---

## Tier 4 — Medium-Low Priority Missing (PTX ISA, Texture, Surface, Advanced)

### 4.1 PTX ISA — Missing Instructions

| # | Missing Instruction Family | Notes |
|---|---|---|
| 4.1.1 | `tex` / `tld4` / `txq` / `suld` / `sust` | Texture / surface load/store/query |
| 4.1.2 | `cp.async.bulk.tensor.3d` / `4d` / `5d` | Higher-dimensional TMA |
| 4.1.3 | `tcgen05.*` | Blackwell (SM100) 5th-gen tensor core instructions |
| 4.1.4 | `atom.shared.*` | Shared-memory atomics (only `atom.global.*` present) |
| 4.1.5 | `match.sync` / `match.any` | Warp match operations (Hopper+) |
| 4.1.6 | `grid.sync` / `griddepcontrol` | Grid-level synchronization |
| 4.1.7 | `elect.sync` | Warp election (Hopper+) |
| 4.1.8 | `prmt` | Byte permutation |
| 4.1.9 | `sad` / `dsad` | Sum of absolute differences |
| 4.1.10 | `ld.global.v4.f16` / `st.global.v4.f16` / `ld.global.v2.f16` | FP16 vector memory |
| 4.1.11 | `cvt.*` variants | Dozens of missing conversion precisions (e.g., `cvt.rn.f16.f64`, `cvt.rn.bf16.f32`, `cvt.rz.u8.f32`, etc.) |
| 4.1.12 | `rcp.rn.f32` / `sqrt.rn.f32` | Only `rcp.approx.f32` and `sqrt.approx.f32` present; RN variants missing |
| 4.1.13 | `div.rn.f32` / `div.rn.f64` | Only `div.approx.f32` present |
| 4.1.14 | `mad.hi.s32` / `mad.lo.s64` / `mad.hi.s64` | Missing MAD wide-integer variants |
| 4.1.15 | `mma.sync` additional shapes/precisions | Many Ampere/Hopper `mma.sync` shape/type combinations not covered |
| 4.1.16 | `cp.reduce.async` | Async copy-with-reduction (Hopper+) |
| 4.1.17 | `wgmma.mma_async` additional shapes | Only m64n{256,128,64}k16 and m64n256k8 covered |

### 4.2 Texture / Surface Reference Gaps

| # | Missing Feature | Notes |
|---|---|---|
| 4.2.1 | `cuModuleGetSurfRef` / `cuSurfRefSetArray` / `cuSurfRefGetArray` | Returns `CUDA_ERROR_NOT_SUPPORTED` |
| 4.2.2 | `cuTexRefSetFormat` missing formats | Only UINT8/16/32, INT8/16/32, FLOAT supported. Missing: half (`CU_AD_FORMAT_HALF`), signed/unsigned normalized, NV12 |
| 4.2.3 | `cuTexRefSetAddressMode` / `cuTexRefSetBorderColor` | Texture addressing / border |
| 4.2.4 | `cuTexRefSetFlags` only handles normalized coords; missing `CU_TRSF_READ_AS_INTEGER`, `CU_TRSF_SRGB` | Partial implementation |

### 4.3 Cooperative Groups / Device-Side Libraries

| # | Missing Feature | Notes |
|---|---|---|
| 4.3.1 | `cooperative_groups::*` C++ API | `grid_group` with `sync()`, `size()`, `thread_rank()` and `this_grid()` are present in `cpu_cuda_env.h`. Missing: `thread_block`, `coalesced_group`, `reduce()`, `partition()`, `shfl()`, `thread_block_tile`, `multi_grid`. |
| 4.3.2 | CUB / Thrust runtime hooks | No CUB or Thrust interoperability layer. Device-wide primitives (scan, sort, reduce) would need fallback paths. |
| 4.3.3 | `cuda_bf16.h` device math | `__nv_bfloat16`, `__float2bfloat16`, `__bfloat162float`, `__hadd_bf`, `__hmul_bf` are present in `cpu_cuda_env.h`. |
| 4.3.4 | `cuda_fp16.h` device math | `__half`, `__float2half`, `__half2float`, `__hadd`, `__hmul`, `__hfma`, `__hexp`, `__hsqrt`, `__half2`, operator overloads, and WMMA `nvcuda::wmma` fragment ops with `__half` are all present in `cpu_cuda_fp16.h` and `wmma_emulation.h`. **PTX-level FP16 vector loads (`ld.global.v2.f16`, `ld.global.v4.f16`) are missing** from the translator. |
| 4.3.5 | Texture / surface builtins (C++ layer) | `tex1D/tex2D/tex3D/tex1Dfetch` and `surf2Dread/surf2Dwrite` templates exist in `cpu_cuda_env.h` (C++ emulation). **PTX-level** `tex`/`suld`/`sust` instructions are still missing from the translator. |
| 4.3.6 | Dynamic Parallelism (device-side) | `cudaLaunchDevice` and `cudaGetParameterBuffer` are present in `cpu_cuda_env.h`. Full CDP API (`cudaDeviceSynchronize`, `cudaGetParameterBufferV2`, `cudaLaunchDeviceV2`) missing. |

---

## Tier 5 — Low Priority / Deployment

| # | Feature | Notes |
|---|---|---|
| 5.1 | K8s Device Plugin / SLURM GRES | Deployment integration, not a runtime API gap |
| 5.2 | cuDNN INT8x4 / INT8x32 packed tensor layouts | Basic INT8 path exists; packed channel layouts need extra descriptor work |
| 5.3 | Full CUPTI-equivalent profiling | Instruction-level sampling, concurrent kernel timeline, PC sampling |
| 5.4 | CUDA Dynamic Parallelism (CDP) full parameter passing | `kernelFn` parameter ignored by design; child kernel enqueue works but not full CDP API surface |
| 5.5 | `cudaGraphExecUpdate_v2` / newer graph update variants | Only basic `cudaGraphExecUpdate` present |

---

## Cross-Platform, Mesh Topology, and Functioning Gaps

### Cross-Platform Considerations

| # | Gap | Linux | Windows | macOS | Impact |
|---|---|---|---|---|---|
| CP.1 | **cuRAND entropy source** | `getrandom()` implemented | `BCryptGenRandom()` implemented | `getentropy()` implemented | ✅ All platforms covered |
| CP.2 | **cuFFT dependency** | FFTW3 via pkg-config | vcpkg / manual build | Homebrew / manual build | Missing library — no shim yet |
| CP.3 | **cuSOLVER LAPACK** | OpenBLAS + LAPACKE | Intel MKL or OpenBLAS | Accelerate.framework or OpenBLAS | Missing library — no shim yet |
| CP.4 | **Shared-memory IPC** | POSIX `shm_open` ✅ | `CreateFileMapping` ✅ | POSIX `shm_open` ✅ | Already works for existing IPC |
| CP.5 | **External semaphores** | POSIX named semaphores ✅ | Windows named semaphores ✅ | POSIX named semaphores ✅ | Already works |
| CP.6 | **K8s Device Plugin** | Unix domain sockets + inotify | Named pipes + `ReadDirectoryChangesW` | Unix domain sockets + kqueue | Tier 5 — deployment only |
| CP.7 | **SLURM GRES** | `slurm_spank.h` shared lib | N/A (Linux-only) | N/A | Tier 5 — Linux only |
| CP.8 | **Profiling (CUPTI-equiv)** | `perf_event_open` for CPU PMU | `EvtCreateSession` / `TraceEvent` | `kdebug` / `signpost` | Tier 5 — CPU-only fallback |
| CP.9 | **Mesh I/O polling** | `epoll` ✅ | `WSAPoll` / IOCP ✅ | `kqueue` ✅ | Already in `tcp_cluster/` |
| CP.10 | **Temperature sensing** | sysfs (`/sys/class/thermal`) ✅ | Registry heuristic ⚠️ | IOKit heuristic ⚠️ | Partial — Windows/macOS use heuristics |
| CP.11 | **NUMA binding** | `mbind(MPOL_PREFERRED)` ✅ | `SetThreadAffinityMask` + `VirtualAllocExNuma` | No NUMA API | Partial — macOS lacks explicit NUMA control |
| CP.12 | **Platform entropy** | `getrandom()` ✅ | `BCryptGenRandom()` ✅ | `getentropy()` ✅ | Already works |

**Cross-platform rule for new features**: Any feature that touches OS APIs (entropy, shared memory, semaphores, profiling, NUMA) must follow the `*_linux.cpp` / `*_macos.cpp` / `*_win32.cpp` split pattern. No inline `#ifdef` in business-logic files.

---

### Mesh Topology Considerations

| # | Gap | Current State | Impact |
|---|---|---|---|
| MT.1 | **Mesh peer discovery** | `VGRE_MESH_PEERS=ip:port,...` env var + UDP discovery with HMAC-SHA256 ✅ | Works for existing AllReduce/Broadcast |
| MT.2 | **Dynamic peer join/leave** | No hot-add/hot-remove of mesh peers | High for elastic clusters (K8s autoscaling) |
| MT.3 | **ncclSend/Recv mesh routing** | Not implemented — needs direct rank→rank TCP path per mesh edge | Blocks pipeline parallelism |
| MT.4 | **ncclAllToAll mesh bandwidth** | Would use full mesh N×N TCP; no topology-aware tree/ring optimization | Medium — CPU bandwidth is less constrained than GPU |
| MT.5 | **Mesh security for new p2p** | Reuse existing AES-256-CTR + 2048-bit replay bitmap ✅ | Security already solved |
| MT.6 | **Cluster-wide profiling sync** | No cross-node timeline synchronization (clocks may drift) | Medium — NTP or logical clocks needed |
| MT.7 | **Mesh partition for new collectives** | `workload_partitioner.cpp` only handles kernel grids; collectives need own partition strategy | Low — can reuse recursive bisection |

**Mesh topology rule for new distributed features**: Any new NCCL collective must call `TCPClusterManager::ensureConnected(rank)` before data transfer. Do not assume the full mesh is pre-connected.

---

### Functioning / Runtime Engine Integration Gaps

| # | Gap | Subsystem | Current State | Impact |
|---|---|---|---|---|
| F.1 | **Adaptive thread tuning for new BLAS/DNN** | `adaptive_execution_engine.cpp` | Works for existing GEMM/conv; not integrated with new Level-2/3 BLAS or cuDNN backward passes | Medium — new compute paths may use suboptimal thread counts |
| F.2 | **NVTX markers for new APIs** | `runtime_profiler.cpp` | Existing APIs instrumented; new shims must add `nvtxRangePushA/Pop` | Low — missing markers only hurt debugging |
| F.3 | **Memory bandwidth tracking for new allocations** | `memory_manager.cpp` | `recordMemoryBandwidth()` works for `cudaMalloc`-family; new array/texture allocations not tracked | Low — stats incomplete |
| F.4 | **UVM page-fault for new memory types** | `memory_manager_managed.cpp` | SIGSEGV/VEH handlers cover device/managed memory; CUDA arrays and external memory not in UVM path | Medium — arrays may fault uncaught |
| F.5 | **Graph scheduler for new node types** | `scheduler.cpp` | Only memcpy/conditional nodes scheduled; kernel/memset/host/child nodes need scheduler integration | **Critical** — graph replay will deadlock |
| F.6 | **Resource ledger for new descriptors** | `resource_ledger.cpp` | Tracks basic allocations; cuDNN backend descriptors, INT8 packed tensors, graph user objects not tracked | Low — OOM risk on descriptor-heavy workloads |
| F.7 | **IPC sharing for new libraries** | `ipc_manager.cpp` | Works for CUDA IPC handles; cuRAND state, cuFFT plans, external memory handles not shareable | Medium — multi-process model breaks |
| F.8 | **Kernel fusion for new ops** | `kernel_fusion_engine.cpp` | Fuses consecutive compatible kernels; new BLAS/DNN ops may break fusion chains | Low — fusion is opportunistic |
| F.9 | **Chrome trace for new graph nodes** | `runtime_profiler.cpp` | Exports kernel timeline; new graph node types need trace events | Low — debugging only |
| F.10 | **Workload partition for new collectives** | `workload_partitioner.cpp` | Recursive bisection for 3D grids; p2p collectives need no partitioning | N/A |

**Functioning rule for new features**: Every new feature that allocates memory must go through `MemoryManager`. Every new compute path must integrate with `AdaptiveExecutionEngine` for thread tuning. Every new distributed operation must integrate with `TCPClusterManager` for mesh connectivity.

---

## Recommendations

1. **Immediate (blocks PyTorch/TF graphs)**: Implement `cudaGraphAddKernelNode`, `cudaMemcpyToSymbol`, `cudaFuncGetAttributes`.
2. **High priority**: Add missing cuBLAS Level-2/Level-3 functions (`Trsm`, `Trsv`, `Syrk`, `Ger`) and cuDNN backward passes.
3. **Graph completeness**: Backfill memset, host, child-graph, empty, event-record/wait, and mem-alloc/free node types in the CUDART shim.
4. **PTX expansion**: Add `tex`/`suld` instructions, shared-memory atomics, and missing `cvt`/`rcp.rn`/`sqrt.rn` variants to unblock more kernels.
5. **Missing libraries**: Create stub shims for cuFFT, cuRAND, cuSOLVER, cuSPARSE that return `CUFFT_STATUS_NOT_IMPLEMENTED` etc. so dependent apps fail gracefully instead of missing-symbol crashes.
6. **Cross-platform discipline**: New OS-dependent code must use the `*_linux.cpp` / `*_macos.cpp` / `*_win32.cpp` split pattern. Update `docs/CROSS_PLATFORM_STATUS.md` when adding platform-specific behavior.
7. **Mesh topology discipline**: New distributed features must call `TCPClusterManager::ensureConnected(rank)` before data transfer. Update `docs/ARCHITECTURE.md` cluster networking section when adding new collectives.
8. **Functioning discipline**: Every new memory allocation goes through `MemoryManager`. Every new compute path integrates with `AdaptiveExecutionEngine`. Every new graph node type integrates with `StreamScheduler`.
9. **Documentation discipline**: This file must be updated **before** any PR merges that claim to close a gap. Verify every claim with `grep -rn "^cudaError_t <function>" src/api/`.

