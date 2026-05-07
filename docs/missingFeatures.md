# VGRE — Missing Features & Implementation Roadmap

**Research Date**: 2026-05-07  
**Based on**: NVIDIA CUDA Runtime API docs, PTX ISA 9.2, PyTorch/TensorFlow CUDA requirements,  
GPGPU-Sim gap analysis, OpenTelemetry GPU semconv, cuBLAS/cuDNN production API coverage.  
**Current Test Count**: 69/69 passing  
**Goal**: Close every gap so VGRE can run real PyTorch, TensorFlow, and distributed ML workloads  
without hitting missing-API errors.

---

## Coverage Summary

| Category | Implemented | Partial | Missing |
|---|---|---|---|
| CUDA Runtime API | 85% | 10% | 5% |
| Memory Management | 80% | 10% | 10% |
| CUDA Graphs | 70% | 15% | 15% |
| PTX ISA Coverage | 65% | — | 35% |
| Profiling / Observability | 40% | 30% | 30% |
| cuBLAS / cuDNN | 75% | 15% | 10% |
| Cluster / Networking | 60% | 10% | 30% |
| Inference / Quantization | 20% | 10% | 70% |

---

## Phase 7 — Tier 1: Critical (blocks real PyTorch/TF workloads today)

### 7.1 NVTX Profiling Markers — FULLY MISSING

**Why critical**: Every real CUDA workload instruments itself with NVTX ranges.
PyTorch instruments its entire training loop (forward, backward, optimizer step) with
`nvtxRangePushEx`/`nvtxRangePop`. Without a shim, applications that call NVTX functions
crash at link time or silently skip profiling, and no Nsight Systems timeline is possible.

**APIs to implement** (`src/api/nvtx_shim.cpp` + `include/vgre/api/nvtx.h`):
```
nvtxMarkEx(eventAttrib)
nvtxRangeStartEx(eventAttrib)  → returns nvtxRangeId_t
nvtxRangeEnd(id)
nvtxRangePushEx(eventAttrib)   → returns nesting depth
nvtxRangePop()
nvtxDomainCreate(name)         → returns nvtxDomainHandle_t
nvtxDomainDestroy(domain)
nvtxNameCudaStreamEx(domain, stream, name)
nvtxNameCudaDeviceEx(domain, device, name)
nvtxNameCudaThreadEx(domain, tid, name)
nvtxDomainMarkEx(domain, eventAttrib)
nvtxDomainRangeStartEx(domain, eventAttrib)
nvtxDomainRangeEnd(domain, id)
nvtxDomainRangePushEx(domain, eventAttrib)
nvtxDomainRangePop(domain)
```

**Integration**: Forward all ranges to `RuntimeProfiler::beginKernel` / `endKernel` for
correlation with OTLP traces. Store range name + color in the profiler event ring buffer.

**References**:
- https://nvidia.github.io/NVTX/
- https://docs.nvidia.com/cuda/profiler-users-guide/index.html

---

### 7.2 `cudaMallocFromPoolAsync` and Full Memory Pool API — PARTIALLY MISSING

**Why critical**: PyTorch 2.x ships its own CUDA caching allocator that calls
`cudaMallocFromPoolAsync` directly. Pool creation/destruction exists but allocation
from a named pool does not, causing a hard crash on any PyTorch memory operation.

**What exists**: `cudaMemPoolCreate`, `cudaMemPoolDestroy` in `src/api/cudart_shim.cpp:553`

**APIs to add** (`src/api/cudart_shim.cpp`):
```
cudaMallocFromPoolAsync(devPtr, size, pool, stream)
cudaMemPoolSetAttribute(pool, attr, value)   — release threshold, reuse policy
cudaMemPoolGetAttribute(pool, attr, value)
cudaMemPoolTrimTo(pool, minBytesToKeep)      — release unused memory to OS
cudaMemPoolSetAccess(pool, descList, count)  — inter-device visibility
cudaMemPoolGetAccess(flags, pool, location)
cudaMemPoolExportToShareableHandle(handle, pool, handleType, flags)
cudaMemPoolImportFromShareableHandle(pool, handle, handleType, flags)
cudaMemPoolExportPointer(exportData, ptr)
cudaMemPoolImportPointer(ptr, pool, exportData)
```

**Implementation**: Route `cudaMallocFromPoolAsync` through
`MemoryManager::allocateFromPool(pool_handle, size)` on the stream task queue.
Pool attributes stored in `MemoryPool::attrs` map. `TrimTo` calls `destroyPool`
on slab blocks that are not in use and rebuilds the free list.

**References**:
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html

---

### 7.3 Stream Attribute APIs — FULLY MISSING

**Why critical**: Frameworks call `cudaStreamGetId` to tag telemetry and
`cudaStreamGetAttribute`/`cudaStreamSetAttribute` to set access policy windows
(L2 cache hit-rate hints). Without them, any framework that queries stream metadata
receives `cudaErrorInvalidValue` and aborts.

**APIs to add** (`src/api/cudart_shim_stream.cpp`):
```
cudaStreamGetId(stream, streamId*)           — unique uint64 per stream
cudaStreamGetAttribute(stream, attr, val*)   — cudaStreamAttrID enum
cudaStreamSetAttribute(stream, attr, val*)
cudaStreamGetFlags(stream, flags*)           — cudaStreamNonBlocking etc.
cudaStreamGetPriority(stream, priority*)     — -2..2 priority level
cudaStreamGetDevice(stream, device*)         — device owning the stream
```

**Implementation**: Add `uint64_t id` (monotonic counter), `int flags`, `int priority`,
`StreamAttrMap attrs` fields to the `Stream` struct in `scheduler.h`.
Assign ID atomically at `cudaStreamCreate` time.

**References**:
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html

---

### 7.4 `cudaMemRangeGetAttribute` / `cudaMemRangeGetAttributes` — FULLY MISSING

**Why critical**: PyTorch's UVM path queries prefetch state before deciding whether
to issue `cudaMemPrefetchAsync`. Without this, it prefetches redundantly or skips
beneficial prefetch, causing page-fault storms on large tensor accesses.

**APIs to add** (`src/api/cuda_interceptor_memory.cpp`):
```
cudaMemRangeGetAttribute(data, dataSize, attribute, devPtr, count)
cudaMemRangeGetAttributes(data[], dataSizes[], attributes[], numAttributes, devPtr, count)
```

**Attributes to support**:
```
cudaMemRangeAttributeReadMostly          → ManagedRegion::advise & READ_MOSTLY bit
cudaMemRangeAttributePreferredLocation   → ManagedRegion::preferredLocation
cudaMemRangeAttributeAccessedBy          → ManagedRegion::accessedByDevices bitmask
cudaMemRangeAttributeLastPrefetchLocation→ ManagedRegion::lastPrefetchDev
```

**Implementation**: Look up the `ManagedRegion` by `devPtr` address via the existing
`allocRange_` binary-search map in `MemoryManager`, then read the tracked fields.

**References**:
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html

---

### 7.5 PTX Extended Integer and Logic Instructions — FULLY MISSING

**Why critical**: Cryptographic kernels, hash functions, and any kernel using 64-bit
pointer arithmetic on 32-bit hardware emits `add.cc`, `addc`, `sub.cc`, `subc` for
multi-word addition with carry. `lop3` (3-operand logic) is emitted by CUDA's
bit-manipulation builtins (`__funnelshift_l`, `__brev`). Without these, the PTX
translator emits unknown-opcode warnings and produces incorrect code.

**Instructions to add** (`src/compiler/ptx_translator.cpp` `getMap()`):

```cpp
// Carry-flag arithmetic (simplified: CC register modeled as thread-local int _cc)
{"add.cc.u32",   [](auto& o){ return "_cc=(unsigned)"+o[0]+"<(unsigned)"+o[1]+
                                      "; "+o[0]+"="+o[1]+"+"+o[2]+";"; }},
{"add.cc.s32",   [](auto& o){ return "_cc=(unsigned)"+o[0]+"<(unsigned)"+o[1]+
                                      "; "+o[0]+"="+o[1]+"+"+o[2]+";"; }},
{"addc.u32",     [](auto& o){ return o[0]+"="+o[1]+"+"+o[2]+"+_cc;"; }},
{"addc.cc.u32",  [](auto& o){ return "{ unsigned _t="+o[1]+"+"+o[2]+"; "
                                      "unsigned _tc=_t+_cc; _cc=(_tc<_t); "
                                      o[0]+"=_tc; }"; }},
{"sub.cc.u32",   [](auto& o){ return "_cc="+o[1]+"<"+o[2]+
                                      "; "+o[0]+"="+o[1]+"-"+o[2]+";"; }},
{"subc.u32",     [](auto& o){ return o[0]+"="+o[1]+"-"+o[2]+"-_cc;"; }},
{"subc.cc.u32",  [](auto& o){ return "{ unsigned _b="+o[2]+"+_cc; "
                                      "_cc="+o[1]+"<_b; "+o[0]+"="+o[1]+"-_b; }"; }},
{"mad.hi.cc.u32",[](auto& o){ return "{ unsigned long long _p="
                                      "(unsigned long long)"+o[1]+"*"+o[2]+"; "
                                      "unsigned _hi=(unsigned)(_p>>32); "
                                      "_cc=_hi+"+o[3]+"<_hi; "+o[0]+"=_hi+"+o[3]+"; }"; }},
// 3-operand logic: result = LUT[{a_bit,b_bit,c_bit}]
// LUT is an 8-bit immediate; bit i = result when inputs = binary(i)
{"lop3.b32",     [](auto& o){ return o[0]+"=vgre_lop3_b32("
                                      +o[1]+","+o[2]+","+o[3]+","
                                      +"(unsigned char)("+o[4]+"));"; }},
// Warp-level reduction (used by cooperative-groups reduce)
{"redux.sync.add.s32",[](auto& o){ return o[0]+"="+o[1]+";"; }}, // serial: already reduced
{"redux.sync.add.u32",[](auto& o){ return o[0]+"="+o[1]+";"; }},
```

Add the `vgre_lop3_b32` helper near the top of the translator:
```cpp
static inline unsigned vgre_lop3_b32(unsigned a, unsigned b, unsigned c, unsigned char lut) {
    unsigned r = 0;
    for (int i = 31; i >= 0; --i) {
        int idx = ((a>>i)&1)<<2 | ((b>>i)&1)<<1 | ((c>>i)&1);
        r |= ((lut>>idx)&1) << i;
    }
    return r;
}
```

**References**:
- https://docs.nvidia.com/cuda/parallel-thread-execution/ (PTX ISA 9.2, §9.7.1)

---

### 7.6 `cudaStreamBeginCaptureToGraph` — FULLY MISSING

**Why critical**: PyTorch's CUDA Graphs integration (used by `torch.compile` and the
`CUDAGraph` class) calls this to replay a previously captured graph by appending new
nodes into an existing graph object, rather than creating a new one each time.
Without it, `torch.compile` with `mode="reduce-overhead"` raises `cudaErrorNotSupported`.

**API to add** (`src/api/cudart_shim.cpp` + `src/core/runtime_engine.cpp`):
```cpp
cudaError_t cudaStreamBeginCaptureToGraph(
    cudaStream_t stream,
    cudaGraph_t graph,
    const cudaGraphNode_t *dependencies,
    const cudaGraphEdgeData *dependencyData,
    size_t numDependencies,
    cudaStreamCaptureMode mode);
```

**Implementation**: Identical to `cudaStreamBeginCapture` but instead of allocating
a new `GraphState`, it sets the stream's capture target to the provided pre-existing
`graph` handle. Dependencies passed in become the initial frontier nodes.

**References**:
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
- https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/

---

## Phase 8 — Tier 2: High Priority (production deployment)

### 8.1 Virtual Memory Management API (`cuMemCreate` / `cuMemMap` family) — FULLY MISSING

**Why critical**: vLLM, vAttention, and any LLM serving framework that implements
paged attention uses `cuMemCreate`/`cuMemMap` to grow the KV-cache in-place without
`cudaMemcpy`. Without it, these frameworks cannot initialize.

**New file**: `src/api/cuda_virtual_memory.cpp`  
**New header**: `include/vgre/api/cuda_virtual_memory.h`

**APIs to implement**:
```
cuMemCreate(handle*, size, prop*, flags)       → mmap(PROT_NONE) + PhysicalAlloc registry
cuMemRelease(handle)                           → munmap physical backing
cuMemAddressReserve(ptr*, size, align, hint, flags) → mmap(PROT_NONE) VA reservation
cuMemAddressFree(ptr, size)                    → munmap VA range
cuMemMap(ptr, size, offset, handle, flags)     → mprotect(PROT_READ|PROT_WRITE)
cuMemUnmap(ptr, size)                          → mprotect(PROT_NONE)
cuMemSetAccess(ptr, size, desc*, count)        → per-device access permission map
cuMemGetAccess(flags*, location*, ptr)
cuMemGetAllocationGranularity(gran*, prop*, option) → sysconf(_SC_PAGESIZE) or 2MB
cuMemGetAllocationPropertiesFromHandle(prop*, handle)
cuMemExportToShareableHandle(handle*, mem, type, flags) → POSIX fd or name
cuMemImportFromShareableHandle(handle*, shareableHandle, type)
```

**Design**: Maintain a `VirtualMemoryRegistry` (map from VA → physical handle + offset).
`cuMemMap` calls `mprotect(PROT_READ|PROT_WRITE)` on the reserved range and records the
mapping. `cuMemUnmap` revokes access. `cuMemSetAccess` is a no-op on single-device but
updates a permissions table for multi-device emulation.

**References**:
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html
- https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html

---

### 8.2 External Semaphores — FULLY MISSING

**Why critical**: Required for interoperability with graphics APIs (Vulkan, OpenGL,
D3D12) and for timeline semaphore-based synchronization in multi-process pipelines.
Any application that uses `cudaGraphAddExternalSemaphoreSignalNode` currently fails.

**New file**: `src/api/cuda_external_semaphore.cpp`

**APIs to implement**:
```
cudaImportExternalSemaphore(extSem*, desc)
cudaDestroyExternalSemaphore(extSem)
cudaSignalExternalSemaphoresAsync(extSems, params, count, stream)
cudaWaitExternalSemaphoresAsync(extSems, params, count, stream)
```

**Handle types to support (Linux)**:
```
CUDA_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD         → eventfd(0, EFD_NONBLOCK)
CUDA_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD → monotonic counter eventfd
```

**Implementation**: Wrap Linux `eventfd` for the opaque FD type. Signal = `eventfd_write(1)`.
Wait = enqueue a task on the stream that calls `epoll_wait(fd, 1ms)` in a loop.
Timeline variant: write/read the 64-bit counter directly for ordered synchronization.

**References**:
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html

---

### 8.3 INT8 / FP8 Quantization Operators — DATA TYPES EXIST, OPERATORS MISSING

**Why critical**: INT8 inference is the standard for production deployment. TensorRT,
PyTorch's `torch.ao.quantization`, and ONNX Runtime all call cuDNN INT8 convolution
and cuBLAS INT8 GEMM. Without these, quantized model inference is impossible.

**What exists**: INT8 format enums in texture/NCCL. No computation paths.

**cuDNN additions** (`src/api/cudnn_shim.cpp`):
```
cudnnConvolutionForward (CUDNN_DATA_INT8):
  → dequantize INT8 inputs using scale factor
  → run existing FP32 conv path
  → requantize FP32 output to INT8 with output scale
  → clamp to [-128, 127]
cudnnConvolutionForward (CUDNN_DATA_INT8x4):
  → same but 4-element SIMD pack (reinterpret as 4× INT8)
```

**cuBLAS additions** (`src/api/cublas_shim.cpp`):
```
cublasGemmEx(CUDA_R_8I, CUDA_R_8I, CUDA_R_32I):
  → scale INT8 inputs to float, run refSgemm, round result to int32
cublasGemmBatchedEx(... INT8 types ...):
  → loop over batch with above
```

**New quantization helper** (`include/vgre/runtime/quantization.h`):
```cpp
struct QuantParams { float scale; int8_t zero_point; };
int8_t  vgre_quantize_f32_to_i8(float x, QuantParams p);
float   vgre_dequantize_i8_to_f32(int8_t x, QuantParams p);
uint8_t vgre_quantize_f32_to_u8(float x, QuantParams p);  // FP8 via bit-casting
```

**References**:
- https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html
- https://docs.nvidia.com/cuda/cublas/#cublasgemmbatchedex

---

### 8.4 CUDA Graph SWITCH Conditional Node — PARTIALLY MISSING

**What exists**: IF (flags=0) and WHILE (flags=1) in `src/core/graph_manager.cpp:211`.

**What is missing**: SWITCH (flags=2), used in mixture-of-experts (MoE) and dynamic
dispatch graphs where a device-side integer selects among N subgraphs.

**Addition in `src/core/graph_manager.cpp`**:
```cpp
case 2: { // SWITCH — integer condition selects one of N child subgraphs
    int branch = static_cast<int>(node.conditionFn());
    int nChildren = static_cast<int>(node.children.size());
    // SWITCH has N+1 children: child[0] is always executed (default),
    // children[1..N] correspond to branch values 0..N-1.
    int target = (branch >= 0 && branch < nChildren - 1) ? branch + 1 : 0;
    executeSubgraph(node.children[target]);
    break;
}
```

**References**:
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html (cudaGraphConditionalNodeType)

---

### 8.5 Real Occupancy Calculation — SIMPLIFIED TO HEURISTIC

**What exists**: `cudaOccupancyMaxActiveBlocksPerMultiprocessor` at
`src/api/cudart_shim.cpp:749` uses `hardware_threads / blockSize`. Ignores register
pressure and shared memory, so it returns wrong values for any register-bound kernel.

**Correct formula** (Ampere SM):
```
warpsPerBlock     = ceil(blockSize / 32)
maxWarpsPerSM     = 48   (A100) / 32 (Turing)
maxBlocksPerSM    = 32   (A100)
maxRegsPerSM      = 65536
maxSharedMemPerSM = 102400 bytes

blocksLimitedByWarps   = floor(maxWarpsPerSM / warpsPerBlock)
blocksLimitedByRegs    = floor(maxRegsPerSM / (warpsPerBlock * 32 * registersPerThread))
blocksLimitedBySMem    = floor(maxSharedMemPerSM / max(dynamicSMem + staticSMem, 1))
blocksLimitedByHW      = maxBlocksPerSM

activeBlocks = min(all four limits, blocksLimitedByHW)
```

**In `src/api/cudart_shim.cpp`**: Read `registersPerThread` from `launchBoundsMap_`
(already populated by `__cudaRegisterFunction`). Read `staticSMem` from `KernelIR`.
Emit `VGRE_LOG_DEBUG` with the limiting factor for profiling guidance.

**References**:
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html

---

## Phase 9 — Tier 3: Medium Priority (production completeness)

### 9.1 OpenTelemetry `hw.gpu.*` Semantic Conventions — PARTIAL

**What exists**: OTLP/JSON trace export via HTTP at `src/advanced/runtime_profiler.cpp:368`.

**What is missing**: The correct metric names from the OTel Hardware Semantic Convention.
Currently exports custom span names. Observability stacks (Prometheus, Grafana) cannot
scrape VGRE metrics because they don't match the expected attribute names.

**Metrics to add** (`src/advanced/runtime_profiler.cpp` `toOTLPJSON()`):
```
hw.gpu.memory.limit      → DeviceProperties::totalGlobalMem
hw.gpu.memory.usage      → MemoryManager::getCurrentUsage()
hw.gpu.memory.utilization→ usage / limit as [0.0, 1.0]
hw.gpu.utilization       → AdaptiveExecutionEngine::getAverageLoad() as [0.0, 1.0]
hw.gpu.errors            → always 0 (no ECC on CPU emulation)
hw.status                → "ok" when running, "degraded" on thermal throttle
```

**Resource attributes**:
```
hw.id          = "vgre-0"
hw.type        = "gpu"
hw.model.name  = "VGRE Virtual GPU"
driver.version = "1.1.0"
```

**References**:
- https://opentelemetry.io/docs/specs/semconv/hardware/gpu/

---

### 9.2 Ampere PTX `mma.sync` Instructions — MISSING

**Why needed**: FlashAttention, CUTLASS, and Triton all emit the lower-level Ampere
`mma.sync.aligned` family directly rather than the Volta `wmma` namespace.
Without these, PTX from any modern CUDA library emits unknown-opcode warnings.

**Instructions to add** (`src/compiler/ptx_translator.cpp`):
```cpp
// Ampere m16n8k16 (FP16 inputs, FP32 accumulators)
{"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", [](auto& o){
    // o[0..3] = D accumulators (4 FP32), o[4..7] = A (4 FP16 packed),
    // o[8..9] = B (2 FP16 packed), o[10..13] = C accumulators
    return "vgre_mma_m16n8k16_f32_f16_f16_f32(" +
           o[0]+","+o[1]+","+o[2]+","+o[3]+"," +
           o[4]+","+o[5]+","+o[6]+","+o[7]+"," +
           o[8]+","+o[9]+"," +
           o[10]+","+o[11]+","+o[12]+","+o[13]+");";
}},
// m16n8k8 (TF32 inputs)
{"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32", [](auto& o){
    return "vgre_mma_m16n8k8_tf32(" +
           o[0]+","+o[1]+","+o[2]+","+o[3]+"," +
           o[4]+","+o[5]+","+o[6]+","+o[7]+");";
}},
// m8n8k4 (FP64)
{"mma.sync.aligned.m8n8k4.row.col.f64", [](auto& o){
    return "vgre_mma_m8n8k4_f64(" + o[0]+","+o[1]+","+o[2]+","+o[3]+");";
}},
```

Implement helpers in `include/vgre/compiler/wmma_emulation.h` reusing the existing
AVX-512 `mma_avx512` path from the `detail::` namespace.

**References**:
- https://docs.nvidia.com/cuda/parallel-thread-execution/ §9.7.14 (mma instruction)

---

### 9.3 `cudaMemcpyBatchAsync` / `cudaMemcpy3DBatchAsync` — MISSING

**Why needed**: CUDA 12.6+ batch memcpy APIs used by memory-bandwidth-limited
training pipelines to pipeline multiple transfers in a single API call.

**Implementation** (`src/api/cuda_interceptor_memory.cpp`):
Iterate over the batch array and call `cudaMemcpyAsync` for each element on the
same stream. Functionally correct (single-stream serialized); performance matches
real hardware for CPU-backed memory which is already serialized.

```cpp
cudaError_t cudaMemcpyBatchAsync(cudaMemcpyNodeParams *params, size_t count,
                                  size_t *failIdx, cudaStream_t stream);
cudaError_t cudaMemcpy3DBatchAsync(cudaMemcpy3DParms *params, size_t count,
                                    size_t *failIdx, cudaStream_t stream);
```

**References**:
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html

---

### 9.4 NCCL Topology-Aware Algorithm Selection — NAIVE ONLY

**What exists**: Two-phase barrier all-reduce (`src/api/nccl_shim.cpp`). Works
correctly but uses O(N) serial reduction on root then broadcast. For large tensors
(>1 MB) this is 2-3× slower than ring algorithm.

**What to add** (`src/api/nccl_shim.cpp`):

```
Ring AllReduce (large tensors > 1 MB):
  chunk = tensor / nranks
  pipeline: each rank passes chunk[i] → rank[i+1], accumulate, repeat nranks-1 times
  then broadcast: each rank passes accumulated chunk[i] → rank[i+1]
  bandwidth-optimal: O(2(N-1)/N × data) bytes transferred

Recursive Halving/Doubling (small tensors < 100 KB):
  log₂(N) rounds: each rank exchanges with partner at distance 2^k
  latency-optimal: O(log₂N) rounds

Selection logic:
  bytes = count * ncclDataTypeSize(datatype)
  if (bytes > 1*1024*1024) → ring_allreduce()
  else                     → recursive_halving_allreduce()
```

**References**:
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
- https://arxiv.org/html/2507.04786v1 (Demystifying NCCL)

---

### 9.5 `cudaGraphAddExternalSemaphoreSignalNode` / `cudaGraphAddExternalSemaphoreWaitNode` — MISSING

**Why needed**: CUDA Graphs can capture semaphore signal/wait operations for
zero-copy integration with graphics pipelines (Vulkan timeline semaphores).

**APIs to add** (`src/api/cudart_shim.cpp`):
```
cudaGraphAddExternalSemaphoreSignalNode(pGraphNode, graph, deps, numDeps, nodeParams)
cudaGraphAddExternalSemaphoreWaitNode(pGraphNode, graph, deps, numDeps, nodeParams)
cudaGraphExecExternalSemaphoreSignalNodeSetParams(graphExec, node, nodeParams)
cudaGraphExecExternalSemaphoreWaitNodeSetParams(graphExec, node, nodeParams)
```

**Implementation**: Wrap external semaphore signal/wait calls from §8.2 as graph nodes
using the existing `GraphNode::type = HOST_NODE` mechanism with a lambda callback.

---

### 9.6 `cudaSyncthreadsCount` / `cudaSyncthreadsAnd` / `cudaSyncthreadsOr` — MISSING

**Why needed**: These are predicate-aware barrier intrinsics emitted by Thrust and
CUB for warp-reduction patterns. Currently the PTX translator only handles `bar.sync`.

**PTX instructions to add** (`src/compiler/ptx_translator.cpp`):
```cpp
{"bar.sync",     [](auto&){ return "__syncthreads();"; }},
{"bar.red.popc.u32", [](auto& o){
    // count of threads where predicate is true
    return o[0] + " = __syncthreads_count(" + o[2] + ");";
}},
{"bar.red.and.pred", [](auto& o){
    return o[0] + " = __syncthreads_and(" + o[2] + ");";
}},
{"bar.red.or.pred",  [](auto& o){
    return o[0] + " = __syncthreads_or(" + o[2] + ");";
}},
```

---

## Phase 10 — Tier 4: Advanced / Future

### 10.1 CUDA MPS Multi-Process Sharing

**Why needed**: In production inference serving, multiple worker processes
(different Python interpreter instances) must share a single virtual GPU to amortize
memory overhead. Without MPS, each process gets its own full VGRE context.

**Architecture**:
- Central `vgre-mps-server` process owns all GPU memory and kernel execution
- Client processes connect via Unix domain socket (or named pipe on Windows)
- Protocol: serialized `KernelLaunchRequest` / `MemAllocRequest` / `MemSyncRequest`
- Server deserializes, executes, returns handle + result
- Client processes get opaque memory handles (not real pointers)

**Files to create**:
```
src/mps/mps_server.cpp     — daemon process with Unix socket listener
src/mps/mps_client.cpp     — LD_PRELOAD shim that redirects CUDA API to socket
include/vgre/mps/mps.h     — shared protocol structs
```

**References**:
- https://docs.nvidia.com/deploy/mps/index.html

---

### 10.2 gRPC / Protobuf Cluster Transport

**Why needed**: Distributed ML frameworks (Ray Serve, Horovod, DeepSpeed) use gRPC
for worker-to-worker and coordinator-to-worker communication. The current VSBP TCP
protocol is VGRE-specific and incompatible with these frameworks.

**Design**: Keep VSBP as the fast-path for kernel launch / memory sync. Add gRPC as
an optional transport for control-plane operations (node registration, health checks,
metric export). Define `.proto` service:

```proto
service VGRECluster {
  rpc RegisterWorker(WorkerInfo) returns (WorkerAck);
  rpc LaunchKernel(KernelRequest) returns (KernelResult);
  rpc SyncMemory(MemorySyncRequest) returns (MemorySyncResult);
  rpc GetTelemetry(Empty) returns (stream TelemetryEvent);
}
```

**CMake**: `option(VGRE_ENABLE_GRPC "Enable gRPC cluster transport" OFF)` with
`find_package(gRPC REQUIRED)` and protoc code generation.

**References**:
- https://grpc.io/docs/what-is-grpc/introduction/

---

### 10.3 Hopper PTX Instructions (`wgmma`, TMA, `cp.async.bulk`) — MISSING

**Why needed**: CUTLASS 3.x, FlashAttention-3, and any kernel compiled for
`sm_90` uses Hopper-specific instructions. Without emulation, these kernels silently
produce wrong results or emit hundreds of unknown-opcode warnings.

**Instructions to add** (`src/compiler/ptx_translator.cpp`):

```cpp
// Warp Group MMA (wgmma) — 128-thread collective
{"wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16", [](auto& o){
    // Emulate as 4× m16n64k16 wmma calls tiled over the larger shape
    return "vgre_wgmma_m64n256k16_bf16_f32("+o[0]+"...);";
}},
// Tensor Memory Accelerator (TMA) bulk async copy
{"cp.async.bulk.tensor.1d.global.shared::cta.bulk_group", [](auto& o){
    return "memcpy((void*)("+o[0]+"), (const void*)("+o[1]+"), "+o[2]+");";
}},
// Fence for TMA proxy operations
{"fence.proxy.tensormap::generic.acquire.gpu", [](auto&){
    return "__atomic_thread_fence(__ATOMIC_ACQUIRE);";
}},
{"fence.proxy.async", [](auto&){
    return "__atomic_thread_fence(__ATOMIC_SEQ_CST);";
}},
// warpgroup-level barrier (replaces bar.sync for 128-thread groups)
{"wgmma.wait_group.sync.aligned", [](auto& o){
    return "/* wgmma.wait_group - all wgmma complete */";
}},
```

**References**:
- https://docs.nvidia.com/cuda/parallel-thread-execution/ §9.7.15–16 (wgmma, cp.async.bulk)

---

### 10.4 Kubernetes Device Plugin / SLURM GRES Resource

**Why needed**: Enable cluster job scheduling of VGRE virtual GPUs without
physical GPU hardware. Allows HPC and cloud-native workloads to request
`nvidia.com/gpu: 1` resources that VGRE fulfills.

**Design**:
- Kubernetes: implement the [Device Plugin API](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
  (`ListAndWatch`, `Allocate` RPCs via gRPC)
- SLURM: register VGRE as a GRES (`gres/gpu`) via `gres.conf` and a custom
  `autodetect=nvml` shim that reports virtual GPU count

**Files**:
```
src/k8s/device_plugin.cpp      — gRPC server implementing v1beta1.DevicePlugin
src/slurm/gres_vgre.c          — SLURM GRES plugin shared library
```

**References**:
- https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/
- https://slurm.schedmd.com/gres.html

---

### 10.5 `cuMemMulticast` API (Multi-GPU Broadcast) — MISSING

**Why needed**: NVLink-scale multi-GPU training uses `cuMulticastCreate` to
atomically bind the same physical memory to multiple GPUs for zero-copy broadcast.
Emulation: bind the same VA to multiple VGRE device contexts sharing a process.

**APIs to stub** (`src/api/cuda_virtual_memory.cpp`):
```
cuMulticastCreate(mcHandle*, prop*)             → return shared_ptr<PhysicalAlloc>
cuMulticastAddDevice(mcHandle, device)          → register device in multicast set
cuMulticastBindMem(mcHandle, offset, mem, size) → map same physical backing to all devices
cuMulticastGetGranularity(gran*, prop*, option) → same as cuMemGetAllocationGranularity
```

**References**:
- https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html

---

## Complete Feature Status Table

| # | Feature | Status | Priority | Phase |
|---|---------|--------|----------|-------|
| 1 | NVTX profiling markers | MISSING | Critical | 7.1 |
| 2 | `cudaMallocFromPoolAsync` + pool attrs | PARTIAL | Critical | 7.2 |
| 3 | Stream attribute APIs (GetId, GetAttr, SetAttr) | MISSING | Critical | 7.3 |
| 4 | `cudaMemRangeGetAttribute` | MISSING | Critical | 7.4 |
| 5 | PTX `add.cc`, `addc`, `sub.cc`, `lop3` | MISSING | Critical | 7.5 |
| 6 | `cudaStreamBeginCaptureToGraph` | MISSING | Critical | 7.6 |
| 7 | Virtual memory API (`cuMemCreate`/`cuMemMap`) | MISSING | High | 8.1 |
| 8 | External semaphores | MISSING | High | 8.2 |
| 9 | INT8 / FP8 quantization operators | PARTIAL | High | 8.3 |
| 10 | Graph SWITCH conditional node | PARTIAL | High | 8.4 |
| 11 | Real occupancy calculation | PARTIAL | High | 8.5 |
| 12 | Graph external semaphore nodes | MISSING | High | 9.5 |
| 13 | OpenTelemetry `hw.gpu.*` conventions | PARTIAL | Medium | 9.1 |
| 14 | Ampere PTX `mma.sync.aligned` | MISSING | Medium | 9.2 |
| 15 | `cudaMemcpyBatchAsync` | MISSING | Medium | 9.3 |
| 16 | NCCL ring / tree algorithm selection | PARTIAL | Medium | 9.4 |
| 17 | PTX `bar.red.popc/and/or` | MISSING | Medium | 9.6 |
| 18 | CUDA MPS multi-process sharing | MISSING | Advanced | 10.1 |
| 19 | gRPC / Protobuf transport | MISSING | Advanced | 10.2 |
| 20 | Hopper PTX (`wgmma`, TMA) | MISSING | Advanced | 10.3 |
| 21 | Kubernetes / SLURM device plugin | MISSING | Advanced | 10.4 |
| 22 | `cuMemMulticast` (multi-GPU broadcast) | MISSING | Advanced | 10.5 |

---

## Effort Estimates

| Phase | Features | Estimated Days |
|---|---|---|
| Phase 7 (Tier 1 — unblocks PyTorch/TF today) | 7.1 – 7.6 | ~10 days |
| Phase 8 (Tier 2 — production deployment) | 8.1 – 8.5, 9.5 | ~13 days |
| Phase 9 (Tier 3 — completeness) | 9.1 – 9.4, 9.6 | ~10 days |
| Phase 10 (Tier 4 — advanced) | 10.1 – 10.5 | ~25 days |
| **Total** | **22 features** | **~58 days** |

---

## How to Verify Completeness

After implementing each phase, run:

```bash
# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVGRE_ENABLE_RDMA=OFF
cmake --build build -j$(nproc)

# Unit + integration tests (must stay 69/69)
ctest --test-dir build -j$(nproc) --output-on-failure

# PyTorch CUDA compatibility check (Phase 7 gate)
python3 -c "import torch; torch.zeros(1).cuda(); print('CUDA OK')"

# Occupancy API check (Phase 8 gate)
python3 -c "import torch; print(torch.cuda.get_device_capability())"

# CUDA Graphs smoke test (Phase 7.6 gate)
python3 -c "
import torch
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = torch.zeros(1024).cuda() + 1
g.replay()
print('Graph OK:', y[0].item())
"

# Nsight Systems profiling (Phase 7.1 gate — requires Nsight installed)
nsys profile --trace=cuda,nvtx ./build/examples/vector_addition
```

---

## References

| Topic | URL |
|---|---|
| CUDA Runtime API | https://docs.nvidia.com/cuda/cuda-runtime-api/ |
| PTX ISA 9.2 | https://docs.nvidia.com/cuda/parallel-thread-execution/ |
| CUDA Programming Guide | https://docs.nvidia.com/cuda/cuda-programming-guide/ |
| CUDA Graphs | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html |
| CUDA Virtual Memory | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html |
| CUDA Memory Pools | https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html |
| External Resource Interop | https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html |
| NVTX Library | https://nvidia.github.io/NVTX/ |
| CUDA MPS | https://docs.nvidia.com/deploy/mps/index.html |
| NCCL Collectives | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html |
| cuBLAS API | https://docs.nvidia.com/cuda/cublas/ |
| cuDNN API | https://developer.nvidia.com/cudnn |
| TensorRT Quantization | https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html |
| OTel GPU Semconv | https://opentelemetry.io/docs/specs/semconv/hardware/gpu/ |
| PyTorch CUDA Graphs | https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/ |
| GPUDirect RDMA | https://docs.nvidia.com/cuda/gpudirect-rdma/ |
| MLIR GPU Dialect | https://mlir.llvm.org/docs/Dialects/GPU/ |
| Kubernetes Device Plugin | https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/ |

---

**Version**: 1.0  
**Last Updated**: 2026-05-07  
**Next Review**: After Phase 7 completion
