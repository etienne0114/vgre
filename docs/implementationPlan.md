# VGRE Implementation Plan

**Last Updated**: 2026-06-07

This document tracks every remaining implementation task — real gaps, security hardening, and new capabilities. All completed work has been removed. Only unfinished items are listed.

---

## Completion Tracker

| Track | Title | Status |
|---|---|---|
| A | gRPC stub diagnostics | 🔴 Not started |
| B | cuLink FATBIN/CUBIN input handling | 🔴 Not started |
| C | CDP kernelFn name-based fallback | 🔴 Not started |
| D | NVSCI sync object proxy | 🔴 Not started |
| E | cuDNN backend API — full coverage | 🟡 Partial |
| F | Security: token KDF salt + env leak | 🔴 Not started |
| G | Windows WSAPoll liveness probe | 🔴 Not started |
| H | Multiple virtual GPU devices | 🔴 Not started |
| I | Flash Attention GQA | 🔴 Not started |
| J | NVTX → Perfetto trace export | 🔴 Not started |
| K | Dynamic CUDA Graph node fusion | 🔴 Not started |
| L | ELLPACK / Blocked-ELL sparse formats | ✅ Done |
| M | Tensor parallel intra-process AllReduce | ✅ Done |
| N | PTX → LLVM bitcode direct path (Triton) | 🔴 Not started |
| O | Kernel auto-vectorization hints | 🔴 Not started |
| P | MPS per-client resource quotas | 🔴 Not started |
| Q | Nsight-compatible `.nsys-rep` export | 🔴 Not started |
| R | Signed JIT cache (HMAC validation) | 🔴 Not started |
| S | Cluster work stealing | 🔴 Not started |
| T | Cache key includes CPU arch (CPUID) | 🔴 Not started |
| U | cuSPARSE SpGEMM reuse | 🔴 Not started |
| V | iGPU OpenCL transpiler — SIMD extensions | 🟡 Partial |
| W | Device-side cuRAND via JIT injection | 🔴 Not started |

---

## Track A — gRPC Stub Diagnostics

**File**: `src/advanced/grpc_transport.cpp`

When `VGRE_ENABLE_GRPC` is OFF at build time, gRPC methods silently return error codes. Add a build-time marker so the stub implementation returns `VGREResult::ERR_NOT_BUILT` with a log message:

```
[VGRE-ERROR] gRPC transport: built without gRPC support.
             Rebuild with -DVGRE_ENABLE_GRPC=ON and install libgrpc++-dev.
```

Also: if `VGRE_GRPC_PORT` is set in the environment at runtime but gRPC is not built, emit a startup warning rather than silently ignoring the variable.

---

## Track B — cuLink FATBIN and CUBIN Input

**File**: `src/api/cuda_driver/cuda_driver_module.cpp`

`cuLinkAddData` currently ignores `CU_JIT_INPUT_FATBINARY` and `CU_JIT_INPUT_OBJECT` blobs.

**Changes required**:
1. Detect input type in `cuLinkAddData`.
2. For `CU_JIT_INPUT_FATBINARY`: parse the ELF-like fatbinary container, locate the embedded `.ptx` section (PTX is always present in NVCC-generated fatbins for compute capability ≥ sm_30), and extract it as a PTX buffer.
3. For `CU_JIT_INPUT_OBJECT`: scan for a `.nv_fatbin` ELF section containing the fatbinary, then apply step 2.
4. For all other unrecognized types: return `CUDA_ERROR_INVALID_VALUE` with a descriptive log message (do not silently succeed with an empty result).

---

## Track C — CDP kernelFn Name-Based Fallback

**File**: `src/runtime/cdp_executor.cpp`, `src/core/runtime_engine.cpp`

`vgre_cdp_launch_device` resolves child kernels by address via `lookupKernelIdByFn`. If the address is not in `kernelFnAddrMap_` (e.g., the kernel was registered by name only), the child launch is silently dropped.

**Changes required**:
1. In `vgre_cdp_launch_device`: when `kid == 0`, log a warning instead of silently returning.
2. In `RuntimeEngine`: expose `lookupKernelIdByName(const char*)` that searches `kernelRegistry_` by name.
3. In `vgre_cdp_launch_device`: on `kid == 0`, attempt name-based lookup using a symbol table extracted from the parent kernel's PTX module (the kernel name is recoverable from the PTX `.entry` directive that produced the function pointer).

---

## Track D — NVSCI Sync Object Proxy

**File**: `src/api/cuda_external_semaphore.cpp`

`CUDA_EXTERNAL_SEM_NVSCISYNCOBJ` is currently "not supported" and does nothing.

**Implementation**:
- Linux: back each NvSci sync object with a `eventfd(0, EFD_SEMAPHORE)`. `cudaSignalExternalSemaphoresAsync` writes `1` to the eventfd; `cudaWaitExternalSemaphoresAsync` does a blocking `read` from it, dispatched on a background thread so the CUDA stream does not stall.
- Windows: back with `CreateEvent(NULL, FALSE, FALSE, NULL)`. Signal → `SetEvent`; wait → `WaitForSingleObject(handle, INFINITE)` on a background thread.
- macOS: same as Linux via `kqueue` + `kevent` on a pipe fd.

This covers the most common use case: Jetson-style pipeline sync between DLA and GPU stages, emulated entirely in software.

---

## Track E — cuDNN Backend API Full Coverage

**File**: `src/api/cudnn/cudnn_backend_api.cpp`

**Audit task**: Find every `return CUDNN_STATUS_NOT_SUPPORTED` in the file. For each one:

1. Identify the operation descriptor type and variant.
2. If the operation is a pointwise activation not yet handled (e.g., GELU exact, soft-plus, ELU, CELU, hard-swish), add a case to the pointwise dispatch table.
3. If the operation is a reduction not yet handled (variance, norm, amax), add a case to the reduction dispatch.
4. If the operation requires a truly unimplementable GPU-specific feature (e.g., in-place SM-level fencing), document it clearly and return `CUDNN_STATUS_NOT_SUPPORTED` with a log message naming the specific missing feature.

**Acceptance**: `grep -c "CUDNN_STATUS_NOT_SUPPORTED" cudnn_backend_api.cpp` returns 0 (or only cases with explicit documented reasons).

---

## Track F — Security: Token KDF Salt and Environment Credential Leak

### F.1 — Dynamic KDF Salt

**File**: `src/advanced/token/token_manager_fallback.cpp:296`

The legacy code path uses the literal `"vgre_fallback_kdf"` (17 bytes) as the PBKDF2 password when machine-specific identity detection fails. Replace with:

```cpp
uint8_t random_salt[32];
#if defined(__linux__)
getrandom(random_salt, sizeof(random_salt), 0);
#else
// /dev/urandom or BCryptGenRandom
#endif
// Store random_salt alongside the encrypted token file (in a .salt sidecar file).
// On decrypt, read the sidecar; if absent, use legacy path and re-derive.
```

Existing tokens using the legacy path must be transparently migrated on first successful unlock.

### F.2 — Environment Credential Zeroization

**File**: `src/advanced/token/token_manager_fallback.cpp`, `src/advanced/tcp_cluster/configuration_manager_file_io.cpp`

After reading `VGRE_TCP_AUTH_TOKEN` from the environment:

```cpp
const char* env_token = std::getenv("VGRE_TCP_AUTH_TOKEN");
if (env_token) {
    std::string token(env_token);
    // Zeroize env var in-place before returning
    volatile char* p = const_cast<volatile char*>(env_token);
    while (*p) *p++ = '\0';
    ::unsetenv("VGRE_TCP_AUTH_TOKEN");
    return token;
}
```

Document in the USER_GUIDE that `VGRE_TCP_AUTH_TOKEN_FILE` is the secure path; `VGRE_TCP_AUTH_TOKEN` is available only for containerized ephemeral deployments.

---

## Track G — Windows WSAPoll Liveness Probe

**File**: `src/advanced/tcp_cluster/client_loop.cpp`

WSAPoll does not reliably report `POLLHUP`/`POLLERR` on reset connections. After any `WSAPoll` timeout with no events on the control socket, issue a zero-byte TCP probe:

```cpp
#if defined(_WIN32)
char probe = 0;
int r = send(sock, &probe, 0, 0);
if (r == SOCKET_ERROR) {
    int e = WSAGetLastError();
    if (e == WSAECONNRESET || e == WSAECONNABORTED || e == WSAENETRESET) {
        // Connection is dead — trigger reconnect
        handleDisconnect();
    }
}
#endif
```

---

## Track H — Multiple Virtual GPU Devices

**Files**: `src/core/runtime_engine.cpp`, `src/api/cudart/cudart_shim_device_attrs.cpp`, `src/api/cuda_interceptor.cpp`

VGRE exposes one virtual device. PyTorch DDP, Horovod, and any framework that calls `cudaGetDeviceCount()` expecting ≥ 2 devices cannot run multi-GPU training without modification.

**Design**:
1. `RuntimeEngine` holds a vector of `VirtualDevice` structs, each with its own `BlockWorkerPool`, `MemoryManager`, and `Scheduler`. Count controlled by `VGRE_VIRTUAL_DEVICE_COUNT` (default 1, max 8).
2. `cudaGetDeviceCount` returns `VGRE_VIRTUAL_DEVICE_COUNT`.
3. `cudaSetDevice(i)` / `cudaGetDevice()` bind to the thread-local `VirtualDevice[i]`.
4. Intra-node P2P: `cudaMemcpyPeer(dst, dstDev, src, srcDev, n)` → `memcpy` (same process memory space, so no serialization needed). Report `cudaDeviceCanAccessPeer == 1` for all virtual device pairs.
5. Partition host cores equally: with 16 physical cores and `VGRE_VIRTUAL_DEVICE_COUNT=4`, each virtual device gets 4-core affinity.

---

## Track I — Flash Attention GQA (Grouped Query Attention)

**File**: `src/compiler/kernel_fusion_engine.cpp`

The current `detectFlashAttention` does not recognize GQA patterns (used by LLaMA-2, Mistral, Gemma, Falcon, Phi-3).

**Changes required**:
1. Extend pattern detection to identify `kv_head_repeat` stride: `K` and `V` matrices have stride `num_kv_heads` vs Q's `num_heads`, with `num_heads % num_kv_heads == 0`.
2. In `genFlashAttentionSource`: emit an outer loop over query head groups that iterates `num_heads / num_kv_heads` Q heads per K/V head, reusing the K/V tile in cache for each Q group.
3. Add a unit test: `test_flash_attention_gqa` with `num_heads=32, num_kv_heads=8` (LLaMA-2 7B config).

---

## Track J — NVTX → Perfetto Trace Export

**File**: `src/api/vgre_c_api_telemetry.cpp` (new function), `include/vgre/api/vgre_c_api.h`

**New API**:
```c
int vgre_export_perfetto_trace(const char* output_path);
```

**Implementation**:
1. Serialize the VGRE telemetry timeline into Perfetto's `TracePacket` protobuf format.
2. For each kernel execution: emit a `TrackEvent` slice with kernel name, duration, and grid/block dims as debug annotations.
3. For each memory operation: emit a counter event on the `MemoryBandwidth` track.
4. Write to `output_path` as a binary `.perfetto-trace` file.
5. Update `docs/USER_GUIDE.md` with a section on opening the trace in `ui.perfetto.dev`.

---

## Track K — Dynamic CUDA Graph Node Fusion

**File**: `src/api/cudart/cudart_shim_graph_exec.cpp`, `src/core/graph/graph_executor.cpp`

After `VGRE_GRAPH_FUSION_THRESHOLD` replays (configurable, default 3), analyze the replay log for consecutive compatible nodes:
- Two `MEMCPY_H2D` nodes that feed one kernel with no inter-kernel dependency → fuse into a single multi-buffer copy node.
- Two kernels with compatible grid/block dims, no shared output dependency, same stream → fuse into a single JIT kernel that sequentially executes both bodies.

The optimized graph is stored as a second `CompiledGraph` alongside the original. Subsequent replays use the optimized version. `cudaGraphExecUpdate_v2` invalidates the optimized graph if the topology changes.

---

## Track L — ELLPACK and Blocked-ELL Sparse Formats

**File**: `src/api/cusparse/cusparse_core.cpp`

`cusparseCreateEll` must stop returning `CUSPARSE_STATUS_NOT_SUPPORTED`.

**Implementation**:
1. `SpMV_ELLPACK`: store column indices and values as `(N × max_nnz_per_row)` dense matrices. For each row, iterate `max_nnz_per_row` slots; skip padding slots (`col_idx == -1`). Use AVX2/AVX-512 vector loads for value rows (aligned to 32/64 bytes).
2. `SpMM_ELLPACK`: outer loop over output columns; inner loop is `SpMV_ELLPACK` per column. Parallelize outer loop with `#pragma omp parallel for`.
3. Wire into the `cusparseSpMV` / `cusparseSpMM` dispatch switch on `cusparseFormat_t == CUSPARSE_FORMAT_ELL`.

---

## Track M — Tensor Parallel Intra-Process AllReduce

**Dependencies**: Track H (multiple virtual devices) must be complete first.

**File**: `src/api/nccl/nccl_communicator.cpp`

When all communicator ranks are virtual devices in the same process, `ncclAllReduce` can bypass TCP entirely:

1. Detect same-process topology: if all `ncclUniqueId` holders share the same `getpid()`, use the shared-memory AllReduce path.
2. Implement a ring AllReduce over shared process memory: each rank writes its slice to a shared buffer, reads and accumulates from the next rank's slot. Use `std::atomic_thread_fence(std::memory_order_acq_rel)` between steps.
3. Configure via `VGRE_TP_DEGREE=N`. Document in USER_GUIDE as the "tensor-parallel fast path".

---

## Track N — PTX → LLVM Bitcode Direct Path

**File**: `src/compiler/llvm_translation_engine.cpp`

Triton and MLIR produce LLVM bitcode (`.bc`) directly, not PTX text. The current `cuModuleLoadData` path attempts PTX parsing on bitcode and fails or produces wrong output.

**Changes required**:
1. Detect LLVM bitcode magic bytes (`0x42 0x43` — "BC") at the start of the data blob.
2. Call `llvm::parseBitcodeFile(MemoryBufferRef, Context)` instead of the PTX text scanner.
3. Strip `nvvm.annotations` metadata (these are NVPTX-specific; the ORC JIT compiles for the host CPU, not NVPTX).
4. Submit the parsed `llvm::Module` directly to the ORC JIT `IRLayer` — bypassing the Clang compilation step.
5. Cache the resulting native code by hash of the bitcode blob.

---

## Track O — Kernel Auto-Vectorization Hints

**File**: `src/compiler/llvm_translation_engine.cpp`

After kernel IR analysis (register count, shared memory, detected access pattern), inject LLVM metadata before JIT:

| Access pattern | Annotation injected |
|---|---|
| Element-wise (no dependencies) | `llvm.loop.vectorize.width = VGRE_SIMD_WIDTH` |
| Reduction | `llvm.loop.vectorize.width = VGRE_SIMD_WIDTH`, `interleave.count = 4` |
| Stencil (stride-1 read/write) | `llvm.loop.vectorize.width = VGRE_SIMD_WIDTH`, `llvm.loop.unroll.count = 2` |
| Scatter/gather (indirect indexing) | No hint (auto-vectorizer would pessimize) |

Use the existing pattern classifier in `adaptive_execution_engine.cpp` to detect the access pattern. This integrates with the existing `-O3 -march=native` compilation and requires no new dependencies.

---

## Track P — MPS Per-Client Resource Quotas

**File**: `src/advanced/mps_control.cpp`

When `VGRE_MPS_PIPE` is active, any connected client can exhaust all system memory or CPU. Add a quota layer:

1. Define `VgreMPSPolicy`:
   ```cpp
   struct VgreMPSPolicy {
       size_t maxMemoryBytes   = SIZE_MAX;  // per-client allocation limit
       float  maxThreadFraction = 1.0f;     // fraction of worker pool
       size_t maxQueueDepth    = 256;       // pending kernel limit
   };
   ```
2. Load from JSON via `VGRE_MPS_POLICY_FILE` (same format as `VGRE_CONFIG_FILE`).
3. Enforce in `handleMalloc` (return `cudaErrorMemoryAllocation` on quota exceeded), in `handleLaunch` (queue and return `cudaSuccess` but throttle execution to `maxThreadFraction`), and in `handleEnqueue` (return `cudaErrorLaunchTimeout` on depth exceeded).

---

## Track Q — Nsight-Compatible Trace Export

**File**: new `src/api/nsight_exporter.cpp`

Implement export to Nsight Systems' SQLite-based `.nsys-rep` format (schema is publicly documented).

**Minimum viable schema**:
- `StringIds` table: kernel names, API names
- `KernelLaunches` table: kernel name id, start_ns, end_ns, gridX/Y/Z, blockX/Y/Z
- `MemcpyOps` table: direction, size_bytes, start_ns, end_ns
- `NvtxEvents` table: domain, range_name, start_ns, end_ns

**New API**:
```c
int vgre_export_nsight_trace(const char* output_path);
```

---

## Track R — Signed JIT Cache

**File**: `src/compiler/kernel_cache.cpp`

The disk cache verifies compiled binaries by source SHA-256 only. An attacker with `~/.vgre/cache/` write access can substitute malicious `.so` files.

**Fix**:
1. Derive a cache signing key: `HMAC-SHA256(cluster_auth_token, "vgre_jit_cache_v1")`.
2. On cache write: compute `HMAC-SHA256(signing_key, compiled_binary_bytes)` and store alongside the `.so` as a `.hmac` sidecar.
3. On cache read: recompute HMAC and compare with constant-time `crypto::secure_compare`. On mismatch: log `[VGRE-SECURITY] JIT cache tamper detected`, delete the entry, and recompile.
4. If no cluster auth token is available (single-node use): use machine-identity-derived key from `token_manager_fallback.cpp` (post-Track-F fix).

---

## Track S — Cluster Work Stealing

**File**: `src/advanced/tcp_cluster.cpp`

The 3D recursive bisection partitioner assigns fixed slices at kernel launch time. Slow workers (thermal throttle, NUMA miss) create stragglers.

**Protocol**:
1. Workers report slice completion time to master via a new `WORK_COMPLETE` packet (includes actual vs estimated duration).
2. Master tracks completion times with EWMA (α=0.25, already implemented for accuracy factor).
3. If a worker finishes more than `VGRE_STEAL_THRESHOLD_MS` (default 50 ms) before the slowest outstanding worker is projected to finish, master sends the idle worker a `STEAL_WORK` packet with a sub-slice of the slowest worker's remaining range.
4. The slow worker continues its original slice; the idle worker steals the tail. Both report completion independently. Master deduplicates using range tags.

---

## Track T — Cache Key Includes CPU Architecture

**File**: `src/compiler/kernel_cache.cpp`

JIT binaries cached on an AVX-512 machine will cause `SIGILL` on an AVX2 host. The cache key must include the detected SIMD feature set.

**Change**:
```cpp
std::string cacheKey(const std::string& source, const std::string& flags) {
    std::string cpuFeats = VGRESIMDDetector::getFeatureString(); // "avx2|aes|f16c"
    return sha256(source + "|" + flags + "|" + cpuFeats);
}
```

`VGRESIMDDetector::getFeatureString()` reads CPUID and returns a sorted, canonical feature string. This changes all existing cache keys — existing entries are invalidated on next access (cache miss → recompile → new entry).

---

## Track U — cuSPARSE SpGEMM Reuse

**File**: `src/api/cusparse/cusparse_factorization.cpp`

`cusparseSpGEMM` is implemented (one-shot). `cusparseSpGEMM_reuse` — which separates symbolic analysis (sparsity structure) from numeric computation (values) — is missing.

**Design**:
1. `cusparseSpGEMM_reuse_work_estimation`: run the symbolic CSR SpGEMM analysis (`nnz` computation via row-wise intersection); store the result `(rowPtrC, colIndC)` in the `SpGEMMDescr`.
2. `cusparseSpGEMM_reuse_nnz`: return the stored `nnzC`.
3. `cusparseSpGEMM_reuse_copy`: allocate `valC` and run only the numeric phase (value accumulation), reusing the pre-computed `colIndC`.
4. All three operations share the same `SpGEMMDescr` handle — allocation is done once.

---

## Track V — iGPU OpenCL Transpiler Extensions

**File**: `src/runtime/igpu_opencl_executor.cpp`

Current gaps in the CUDA→OpenCL C transpiler:

| Feature | Current handling | Fix needed |
|---|---|---|
| `tex1D`/`tex2D`/`tex3D` | Not translated → CPU fallback | Emit `read_imagef(image, sampler, coord)` |
| `extern __shared__ float s[]` | Not translated → CPU fallback | Detect dynamic shared mem size from launch params; emit `local float s[N]` in the kernel preamble |
| `__shfl_sync` / `__ballot_sync` | Stripped | Emit OpenCL 2.0 sub-group functions (`sub_group_shuffle`, `sub_group_ballot`) where available; fall back to local memory exchange otherwise |
| Device `printf` | Stripped | Emit `printf` (supported in OpenCL 1.2+ for debugging) |
| `asm volatile` | Stripped with a comment | Acceptable — emit a `// asm volatile stripped` comment for clarity |

---

## Track W — Device-Side cuRAND via JIT Injection

**File**: `src/compiler/llvm_translation_engine.cpp`, `src/api/curand/curand_shim.cpp`

Device-side `curand_uniform(state)`, `curand_normal(state)`, `curand_log_normal(state)` are currently unimplemented in the JIT path.

**Implementation**:
1. At PTX scan time, detect `curand_uniform` / `curand_normal` calls (they appear as `call.uni` PTX instructions to `_Z15curand_uniformPj`-style mangled names).
2. Replace each call site with an inlined XORWOW step using the VGRE XORWOW state layout:
   ```
   // XORWOW step: t = x ^ (x >> 2); x = w; w = v; v = y; y = z; z = t ^ z ^ (t << 1) ^ (z << 4)
   ```
3. The `curandState_t*` argument maps to a per-thread offset in the kernel's shared memory or argument space.
4. `curand_normal` emits a Box-Muller pair in registers: saves one sample for the next call via a carry slot in the state struct.
