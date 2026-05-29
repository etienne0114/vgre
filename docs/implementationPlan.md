# VGRE Implementation Plan

**Version**: 8.0.0  
**Date**: 2026-05-19 (updated)  
**Basis**: Full code-verified audit — tcp_cluster (43 files), all advanced/, api/, core/, runtime/, compiler/, scripts/  
**Format**: Priority-ordered. Completed items moved to ✅ section. Remaining items have file + line reference and concrete fix.

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

### Script / Infrastructure
- ✅ `vgre-start.sh` — ping reachability check when `--master-ip` is provided; fails fast with diagnostics

### Code Quality (additional)
- ✅ P4-3: Config manager JSON parser replaced with `llvm::json::parse()`; `vgre_llvm_iface` linked into `vgre_advanced`

### CUPTI
- ✅ CUPTI software-proxy counters — full `cupti_shim.cpp`: subscriber/activity/buffer/metric APIs backed by `RuntimeProfiler`; IPC, occupancy, FLOP, DRAM throughput, branch efficiency metrics

### Previously Misreported as Stubs (confirmed implemented by audit)
- ✅ Multi-GPU P2P — `cudaMemcpyPeer` routes through `MemoryManager::copyDeviceToDevice` + `memAdvise`
- ✅ GPU passthrough (VFIO) — dlopen/NVRTC pipeline in `gpu_passthrough.cpp`; activates when `libcuda.so.1` present
- ✅ Token manager: macOS Keychain — real `SecKeychain*` APIs; `ERR_NOT_SUPPORTED` is the non-Apple platform stub only
- ✅ Token manager: Linux keyring + libsecret — real `keyctl_*` (always) + `secret_password_*` (when `VGRE_HAS_LIBSECRET`)
- ✅ Token manager: Windows Credential Manager — real `CredWriteW/CredReadW/CredDeleteW`; non-Windows stub only

---

## 🔵 P5 — Long-Term / Large Scope

These require significant architectural work. Listed for roadmap awareness.

| Item | Gap | Effort |
|---|---|---|
| SASS binary execution | Precompiled CUDA libraries unusable | Very large — full ISA simulator |
| MPS multi-process | Single process per device | Large — IPC context sharing |
| OpenMP for `__syncthreads` kernels | Very large kernels exhaust OS thread limit | Medium — two-level dispatch |
| CUPTI hardware counters | Hardware PMU access | Medium — VFIO PMU passthrough |
| FP8 PTX full suite (wgmma.*) | Hopper FP8 training | Large — PTX translator + emulation |
| cuSPARSE Generic API | All modern sparse frameworks | Medium — new descriptor-based API |

---

## File Structure Map (Before Implementation)

This section defines exactly which files to create vs. extend for each feature, and tracks current line counts to avoid bloating any single file.

### Existing Files — Extend In-Place

| File | Current Lines | Features to Add | Acceptable? |
|---|---|---|---|
| `src/api/cudart/cudart_shim_device_attrs.cpp` | 414 | `cudaFuncSetAttribute` (+~30 lines) | ✅ extend |
| `src/api/cuda_driver/cuda_driver_memory.cpp` | 81 | `cuMemAllocAsync/FreeAsync` + pool APIs (+~80 lines) | ✅ extend |
| `src/api/cuda_driver/cuda_driver_stream_event.cpp` | 138 | `cuStreamWaitValue32/64`, `cuStreamWriteValue32/64` (+~80 lines) | ✅ extend |
| `src/api/cudnn/cudnn_rnn.cpp` | 616 | `cudnnRNNForwardTrainingEx/InferenceEx` (+~120 lines) | ✅ extend (same domain) |
| `src/compiler/ptx/ptx_translator_map.cpp` | 566 | `ldmatrix/stmatrix` x1/x2/x4/trans + `redux.sync` and/or/xor/popc (+~100 lines) | ✅ extend |
| `src/compiler/ptx/ptx_conversion.cpp` | 256 | `elect.sync`, `griddepcontrol.*`, `setmaxnreg.*`, `cp.reduce.async.bulk.*`, FP8 cvt (+~80 lines) | ✅ extend |
| `src/api/nccl/nccl_p2p.cpp` | 180 | No changes needed — `ncclSend`/`ncclRecv` already implemented ✅ | — |
| `src/api/cusolver/cusolver_core.cpp` | 756 | **Do not extend** — already large; route to new file below | ⛔ split |
| `src/api/cudnn/cudnn_backend_api.cpp` | 781 | **Do not extend** — already large; POINTWISE/ENGINEHEUR go to new file | ⛔ split |

### New Files to Create

| New File | Target Lines | Features | CMakeLists Target |
|---|---|---|---|
| `src/api/cuda_driver/cuda_driver_library.cpp` | ~180 | `cuLibraryLoadData/FromFile`, `cuLibraryGetKernel`, `cuKernelGetFunction`, `cuLibraryGetGlobal`, `cuLibraryUnload` | `vgre_driver` |
| `src/api/cudart/cudart_proc_address.cpp` | ~60 | `cudaGetProcAddress` (runtime level) | `vgre_cudart` |
| `src/api/cuda_driver/cuda_driver_proc_address.cpp` | ~50 | `cuGetProcAddress` (driver level) | `vgre_driver` |
| `src/api/cudnn/cudnn_backend_pointwise.cpp` | ~220 | `CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR`, `CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR` dispatch + execution | `vgre_cudnn` |
| `src/api/cudnn/cudnn_backend_resample.cpp` | ~120 | `CUDNN_BACKEND_OPERATION_RESAMPLE_FWD/BWD_DESCRIPTOR` | `vgre_cudnn` |
| `src/api/cusolver/cusolver_type_erasure.cpp` | ~200 | `cusolverDnXgetrf`, `cusolverDnXpotrf`, `cusolverDnXgesvd`, `cusolverDnXsygvd` | `vgre_cusolver` |
| `src/compiler/ptx/ptx_fp8.cpp` | ~300 | FP8 MMA (`mma.sync.aligned.*.e4m3/e5m2`), wgmma stubs, `cvt.rn.satfinite.e4m3x2/e5m2x2.f32` | `vgre_compiler` |

### New Header Files to Create

| New Header | Purpose |
|---|---|
| `include/vgre/api/cuda_driver_library.h` | `CUlibrary_st`, `CUkernel_st`, `CUlibraryOption` forward declarations and function prototypes |
| `include/vgre/api/cuda_driver_memops.h` | `CU_STREAM_WAIT_VALUE_GEQ/EQ/AND/NOR` flag constants, `cuStreamWaitValue32/64`, `cuStreamWriteValue32/64` prototypes |

### Existing Headers to Extend

| Header | Addition |
|---|---|
| `include/vgre/compiler/wmma_emulation.h` | `vgre_pack_fp8_e4m3`, `vgre_pack_fp8_e5m2`, `vgre_mma_fp8_e4m3_16x8x32`, `vgre_mma_fp8_e5m2_16x8x32` inline functions |

### CMakeLists.txt Changes Required

Each new `.cpp` file must be added to the correct target's source list. The existing pattern in `CMakeLists.txt` uses `target_sources(vgre_<name> PRIVATE ...)`. The new files map to:

```
vgre_driver:   cuda_driver_library.cpp, cuda_driver_proc_address.cpp
vgre_cudart:   cudart_proc_address.cpp
vgre_cudnn:    cudnn_backend_pointwise.cpp, cudnn_backend_resample.cpp
vgre_cusolver: cusolver_type_erasure.cpp
vgre_compiler: ptx_fp8.cpp
```

### Corrected: Previously Listed as Missing but Already Implemented

| Feature | File | Status |
|---|---|---|
| `ncclSend` / `ncclRecv` | `src/api/nccl/nccl_p2p.cpp` | ✅ Real barrier-based shared-memory implementation, 180 lines |
| `ncclAllGather` | `src/api/nccl/nccl_collectives.cpp` | ✅ Implemented at line ~399 |
| `ncclAllToAll`, `ncclGather`, `ncclScatter` | `src/api/nccl/nccl_p2p.cpp` | ✅ Implemented |

---

## 🔴 P1 — Critical: Framework-Blocking Missing Features (2026-05-29 Audit)

These are absent from the codebase and block major frameworks from running. Each entry includes the precise file, implementation approach, and effort.

---

### P1-A: cudaFuncSetAttribute
**File**: `src/api/cudart/cudart_shim_device_attrs.cpp`
**Effort**: 0.5 day
**Blocked**: FlashAttention-2, CUTLASS 3.x, Triton (any kernel using >48 KB shared memory)

```
// Add after cudaFuncSetSharedMemConfig (~line 127):
cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int value) {
    // VGRE kernel shared memory limit is governed by JIT-allocated SharedMemory buffer,
    // which is sized at launch time from cudaLaunchKernel's sharedMem parameter.
    // For cudaFuncAttributeMaxDynamicSharedMemorySize we store the per-function override
    // in VgreKernelRegistry so that cudaLaunchKernel can enforce it.
    if (attr == cudaFuncAttributeMaxDynamicSharedMemorySize) {
        VgreKernelRegistry::instance().setMaxSharedMemBytes(func, static_cast<size_t>(value));
    }
    // cudaFuncAttributePreferredSharedMemoryCarveout: no L1/shared partition on CPU, ignore
    return cudaSuccess;
}
```

`VgreKernelRegistry` (or equivalent) must store `size_t maxSharedMemBytes` per `const void* func` key. If that map doesn't exist, add a `std::unordered_map<const void*, size_t>` guarded by a `std::mutex` in a new `src/runtime/kernel_registry.cpp`.

---

### P1-B: redux.sync Bitwise Warp Reductions (PTX)
**File**: `src/compiler/ptx/ptx_translator_map.cpp`
**Effort**: 0.5 day
**Blocked**: FlashAttention-2, sparse attention kernels

Add to the redux.sync section of the PTX map alongside the existing add/min/max entries:

```cpp
{"redux.sync.and.b32", [](auto& o) {
    // Emulate warp-AND: in serial CPU model all 32 lanes are the same thread,
    // so the result is just the value itself (AND-identity preserved).
    return o[0] + " = " + o[1] + ";  /* redux.sync.and.b32 serial */";
}},
{"redux.sync.or.b32", [](auto& o) {
    return o[0] + " = " + o[1] + ";  /* redux.sync.or.b32 serial */";
}},
{"redux.sync.xor.b32", [](auto& o) {
    return o[0] + " = " + o[1] + ";  /* redux.sync.xor.b32 serial */";
}},
{"redux.sync.popc.b32", [](auto& o) {
    return o[0] + " = __builtin_popcount((unsigned)" + o[1] + ");  /* redux.sync.popc serial */";
}},
```

The same serial-identity reasoning applies as for `redux.sync.add` — in a single-thread-per-warp model, the warp reduction of any value against itself is the value itself (for AND/OR/XOR). For POPC, popcount of a single u32 is the correct result.

---

### P1-C: cuStreamWaitValue32 / cuStreamWriteValue32
**File**: `src/api/cuda_driver/cuda_driver_stream.cpp` (new section, or new file `cuda_driver_stream_memops.cpp`)
**Effort**: 0.5 day
**Blocked**: NCCL advanced sync, GPUDirect RDMA sync patterns

```cpp
CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                              cuuint32_t value, unsigned int flags) {
    // Busy-wait on the 32-bit value at addr with the given comparison.
    // On CPU emulator this is a simple spin — the address is host-accessible.
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(addr));
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (true) {
        uint32_t cur = *ptr;
        bool cond = false;
        if (flags & CU_STREAM_WAIT_VALUE_GEQ) cond = (cur >= value);
        else if (flags & CU_STREAM_WAIT_VALUE_EQ) cond = (cur == value);
        else if (flags & CU_STREAM_WAIT_VALUE_AND) cond = (cur & value) != 0;
        else if (flags & CU_STREAM_WAIT_VALUE_NOR) cond = (cur | value) == 0;
        if (cond) break;
        if (std::chrono::steady_clock::now() > deadline) return CUDA_ERROR_TIMEOUT;
        std::this_thread::yield();
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    return CUDA_SUCCESS;
}

CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                               cuuint32_t value, unsigned int flags) {
    volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(addr));
    std::atomic_thread_fence(std::memory_order_release);
    *ptr = value;
    return CUDA_SUCCESS;
}
// Also implement 64-bit variants: cuStreamWaitValue64, cuStreamWriteValue64
```

Add `CU_STREAM_WAIT_VALUE_GEQ`, `CU_STREAM_WAIT_VALUE_EQ`, `CU_STREAM_WAIT_VALUE_AND`, `CU_STREAM_WAIT_VALUE_NOR` constants to `include/vgre/api/cuda_driver_types.h` if not already defined.

---

### P1-D: cuMemAllocAsync / cuMemFreeAsync (CUDA Driver Level)
**File**: `src/api/cuda_driver/cuda_driver_memory.cpp`
**Effort**: 1 day
**Blocked**: PyTorch 2.1+ caching allocator, CUDA Graph capture with stream-ordered memory

The runtime-level `cudaMallocAsync` exists; the driver-level variants are completely absent. Implementation delegates to the existing memory manager:

```cpp
CUresult cuMemAllocAsync(CUdeviceptr* dptr, size_t bytesize, CUstream hStream) {
    // Stream-ordered allocation: in VGRE all streams execute serially on CPU,
    // so allocation order is already correct. Delegate to MemoryManager.
    void* ptr = nullptr;
    cudaError_t err = MemoryManager::instance().allocate(&ptr, bytesize);
    if (err != cudaSuccess) return CUDA_ERROR_OUT_OF_MEMORY;
    *dptr = reinterpret_cast<CUdeviceptr>(ptr);
    return CUDA_SUCCESS;
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
    MemoryManager::instance().free(reinterpret_cast<void*>(dptr));
    return CUDA_SUCCESS;
}

// Memory pool APIs (minimal — return default pool, ignore pool attributes)
CUresult cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* poolProps) {
    static CUmemoryPool_st defaultPool;
    *pool = &defaultPool;
    return CUDA_SUCCESS;
}
CUresult cuMemPoolDestroy(CUmemoryPool pool) { return CUDA_SUCCESS; }
CUresult cuDeviceGetDefaultMemPool(CUmemoryPool* pool_out, CUdevice dev) {
    static CUmemoryPool_st defaultPool;
    *pool_out = &defaultPool;
    return CUDA_SUCCESS;
}
CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) {
    return CUDA_SUCCESS;
}
CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void* value) {
    if (attr == CU_MEMPOOL_ATTR_RELEASE_THRESHOLD) {
        *reinterpret_cast<uint64_t*>(value) = UINT64_MAX;
    }
    return CUDA_SUCCESS;
}
```

---

### P1-E: cudaGetProcAddress / cuGetProcAddress (CUDA 12.4+)
**File**: new `src/api/cudart/cudart_proc_address.cpp` + new `src/api/cuda_driver/cuda_driver_proc_address.cpp`
**Effort**: 1 day
**Blocked**: Any framework built with CUDA 12.4+ toolkit

This is a versioned symbol table. VGRE must maintain a string→function-pointer map for every exported symbol. Implementation approach:

1. Create a static lookup table `std::unordered_map<std::string, void*> g_vgre_symbol_table` populated at startup by iterating every exported function via a registration macro.
2. Alternatively (simpler), use `dlsym(RTLD_SELF, symbol)` on Linux/macOS and `GetProcAddress(GetModuleHandle(NULL), symbol)` on Windows — this resolves exported symbols without a manual table.

```cpp
cudaError_t cudaGetProcAddress(const char* symbol, void** pfn,
                                int cudaVersion, uint64_t flags,
                                cudaDriverEntryPointQueryResult* driverStatus) {
#if defined(_WIN32)
    *pfn = reinterpret_cast<void*>(GetProcAddress(GetModuleHandle(NULL), symbol));
#else
    *pfn = dlsym(RTLD_DEFAULT, symbol);
#endif
    if (!*pfn) {
        if (driverStatus) *driverStatus = cudaDriverEntryPointSymbolNotFound;
        return cudaErrorSymbolNotFound;
    }
    if (driverStatus) *driverStatus = cudaDriverEntryPointSuccess;
    return cudaSuccess;
}

CUresult cuGetProcAddress(const char* symbol, void** pfn,
                           int cudaVersion, cuuint64_t flags) {
#if defined(_WIN32)
    *pfn = reinterpret_cast<void*>(GetProcAddress(GetModuleHandle(NULL), symbol));
#else
    *pfn = dlsym(RTLD_DEFAULT, symbol);
#endif
    return *pfn ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}
```

Add `#include <dlfcn.h>` (Linux/macOS) guarded with `#if !defined(_WIN32)` and `#include <windows.h>` for Win32 path. Link with `-ldl` on Linux (already present in CMakeLists.txt).

---

### P1-F: cuLibrary API (CUDA 12.0+)
**File**: new `src/api/cuda_driver/cuda_driver_library.cpp`
**Effort**: 2 days
**Blocked**: Any framework compiled with CUDA 12.0+ toolkit (uses `cuLibraryLoadData` instead of `cuModuleLoad`)

```cpp
// CUlibrary is just a wrapper around CUmodule in VGRE
struct CUlibrary_st {
    CUmodule module;
    std::unordered_map<std::string, CUkernel_st*> kernels;
};
struct CUkernel_st {
    CUfunction func;
    std::string name;
};

CUresult cuLibraryLoadData(CUlibrary* library, const void* code,
                            CUjit_option* jitOptions, void** jitOptionsValues,
                            unsigned int numJitOptions,
                            CUlibraryOption* libraryOptions, void** libraryOptionValues,
                            unsigned int numLibraryOptions) {
    auto* lib = new CUlibrary_st();
    CUresult res = cuModuleLoadData(&lib->module, code);
    if (res != CUDA_SUCCESS) { delete lib; return res; }
    *library = lib;
    return CUDA_SUCCESS;
}

CUresult cuLibraryLoadFromFile(CUlibrary* library, const char* fileName,
                                CUjit_option* jitOptions, void** jitOptionsValues,
                                unsigned int numJitOptions,
                                CUlibraryOption* libraryOptions,
                                void** libraryOptionValues,
                                unsigned int numLibraryOptions) {
    auto* lib = new CUlibrary_st();
    CUresult res = cuModuleLoad(&lib->module, fileName);
    if (res != CUDA_SUCCESS) { delete lib; return res; }
    *library = lib;
    return CUDA_SUCCESS;
}

CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name) {
    auto it = library->kernels.find(name);
    if (it != library->kernels.end()) { *pKernel = it->second; return CUDA_SUCCESS; }
    auto* k = new CUkernel_st();
    k->name = name;
    CUresult res = cuModuleGetFunction(&k->func, library->module, name);
    if (res != CUDA_SUCCESS) { delete k; return res; }
    library->kernels[name] = k;
    *pKernel = k;
    return CUDA_SUCCESS;
}

CUresult cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel) {
    *pFunc = kernel->func;
    return CUDA_SUCCESS;
}

CUresult cuLibraryGetGlobal(CUdeviceptr* dptr, size_t* bytes,
                             CUlibrary library, const char* name) {
    return cuModuleGetGlobal(dptr, bytes, library->module, name);
}

CUresult cuLibraryUnload(CUlibrary library) {
    for (auto& [name, k] : library->kernels) delete k;
    cuModuleUnload(library->module);
    delete library;
    return CUDA_SUCCESS;
}
```

Add `CUlibrary`, `CUkernel`, `CUlibraryOption` types to `include/vgre/api/cuda_driver_types.h`. Register all functions in `CMakeLists.txt` under `vgre_driver`.

---

## 🟠 P2 — High Priority: PTX Instruction Gaps

### P2-A: ldmatrix.sync.aligned / stmatrix.sync.aligned
**File**: `src/compiler/ptx/ptx_translator_map.cpp`
**Effort**: 2 days
**Blocked**: CUTLASS 3.x, Triton, all WMMA-based kernels

`ldmatrix` loads 8×8 FP16 tiles from shared memory into registers. In the CPU serial model, registers are C++ variables in the JIT-compiled function. The PTX operands map as: `o[0]` = output register (or register pair/quad), `o[1]` = shared memory address.

The `.x1` variant loads one 8×8 tile (128 bits = one `uint4`). `.x2` loads two tiles. `.x4` loads four tiles:

```cpp
{"ldmatrix.sync.aligned.m8n8.x1.shared.b16", [](auto& o) {
    // Load 16 bytes (128-bit) from shared memory address o[1] into register o[0].
    // On CPU, shared mem is a host pointer; cast and dereference.
    return "{ const uint32_t* _lm = reinterpret_cast<const uint32_t*>(" + o[1] + ");"
           "  " + o[0] + " = _lm[0]; }";
}},
{"ldmatrix.sync.aligned.m8n8.x2.shared.b16", [](auto& o) {
    // o[0],o[1] are two destination registers; o[2] is address
    return "{ const uint32_t* _lm = reinterpret_cast<const uint32_t*>(" + o[2] + ");"
           "  " + o[0] + " = _lm[0]; " + o[1] + " = _lm[1]; }";
}},
{"ldmatrix.sync.aligned.m8n8.x4.shared.b16", [](auto& o) {
    return "{ const uint32_t* _lm = reinterpret_cast<const uint32_t*>(" + o[4] + ");"
           "  " + o[0] + "=_lm[0]; " + o[1] + "=_lm[1]; "
           "  " + o[2] + "=_lm[2]; " + o[3] + "=_lm[3]; }";
}},
// Transposed variants: same data but logically transposed 8x8 tile
// In CPU emulation, transposition is handled by the matmul emulator, not by ldmatrix itself
{"ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16", [](auto& o) {
    return "{ const uint32_t* _lm = reinterpret_cast<const uint32_t*>(" + o[1] + ");"
           "  " + o[0] + " = _lm[0]; }  /* trans: matmul handles layout */";
}},
{"ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16", [](auto& o) {
    return "{ const uint32_t* _lm = reinterpret_cast<const uint32_t*>(" + o[2] + ");"
           "  " + o[0] + " = _lm[0]; " + o[1] + " = _lm[1]; }";
}},
{"ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16", [](auto& o) {
    return "{ const uint32_t* _lm = reinterpret_cast<const uint32_t*>(" + o[4] + ");"
           "  " + o[0] + "=_lm[0]; " + o[1] + "=_lm[1]; "
           "  " + o[2] + "=_lm[2]; " + o[3] + "=_lm[3]; }";
}},
// stmatrix: store register values to shared memory
{"stmatrix.sync.aligned.m8n8.x1.shared.b16", [](auto& o) {
    return "{ uint32_t* _sm = reinterpret_cast<uint32_t*>(" + o[0] + ");"
           "  _sm[0] = " + o[1] + "; }";
}},
{"stmatrix.sync.aligned.m8n8.x2.shared.b16", [](auto& o) {
    return "{ uint32_t* _sm = reinterpret_cast<uint32_t*>(" + o[0] + ");"
           "  _sm[0]=" + o[1] + "; _sm[1]=" + o[2] + "; }";
}},
{"stmatrix.sync.aligned.m8n8.x4.shared.b16", [](auto& o) {
    return "{ uint32_t* _sm = reinterpret_cast<uint32_t*>(" + o[0] + ");"
           "  _sm[0]=" + o[1] + "; _sm[1]=" + o[2] + ";"
           "  _sm[2]=" + o[3] + "; _sm[3]=" + o[4] + "; }";
}},
```

### P2-B: FP8 PTX Instructions (Hopper SM90)
**File**: `src/compiler/ptx/ptx_conversion.cpp`
**Effort**: 4 days
**Blocked**: CUTLASS FP8 kernels, Transformer Engine, TensorRT FP8 inference

FP8 mma/wgmma PTX must be translated to the existing `vgre_mma_fp8_e4m3` / `vgre_mma_fp8_e5m2` functions in `wmma_emulation.h`. The cvt instructions convert float32 to packed FP8 bytes:

```cpp
// FP8 type conversion (float32 → packed e4m3/e5m2)
{"cvt.rn.satfinite.e4m3x2.f32", [](auto& o) {
    return o[0] + " = vgre_pack_fp8_e4m3(" + o[1] + ", " + o[2] + ");";
}},
{"cvt.rn.satfinite.e5m2x2.f32", [](auto& o) {
    return o[0] + " = vgre_pack_fp8_e5m2(" + o[1] + ", " + o[2] + ");";
}},
// FP8 MMA variants — delegate to wmma_emulation vgre_mma_fp8_e4m3/e5m2
{"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", [](auto& o) {
    // D[0..3]=A[0..7] × B[0..3] + C[0..3] in FP8 e4m3 precision
    return "vgre_mma_fp8_e4m3_16x8x32(" +
           o[0]+","+o[1]+","+o[2]+","+o[3]+","     // D (4 f32 outputs)
           +o[4]+","+o[5]+","+o[6]+","+o[7]+","    // A (8 packed fp8 inputs)
           +o[8]+","+o[9]+","                       // B (2 packed fp8 inputs)
           +o[10]+","+o[11]+","+o[12]+","+o[13]+");"; // C (4 f32 accumulator)
}},
{"mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32", [](auto& o) {
    return "vgre_mma_fp8_e5m2_16x8x32(" +
           o[0]+","+o[1]+","+o[2]+","+o[3]+","
           +o[4]+","+o[5]+","+o[6]+","+o[7]+","
           +o[8]+","+o[9]+","
           +o[10]+","+o[11]+","+o[12]+","+o[13]+");";
}},
```

Helper functions `vgre_pack_fp8_e4m3`, `vgre_pack_fp8_e5m2`, `vgre_mma_fp8_e4m3_16x8x32`, `vgre_mma_fp8_e5m2_16x8x32` must be added to `include/vgre/compiler/wmma_emulation.h`. The pack functions combine two float32 inputs into one uint16_t containing two FP8 values. The MMA functions use the existing FP8 quantize/dequantize logic already in `wmma_emulation.h` for the tcgen05 path.

For `wgmma.mma_async.*` variants: these are tile-level warp-group MMA. In the serial CPU model they can be translated to sequences of `vgre_mma_fp8_*` calls with appropriate tile addressing.

### P2-C: elect.sync / griddepcontrol / setmaxnreg PTX
**File**: `src/compiler/ptx/ptx_conversion.cpp`
**Effort**: 1 day

```cpp
// elect.sync: one lane per warp is "elected". In serial CPU model, thread is always elected.
{"elect.sync", [](auto& o) {
    // o[0] = predicate output, o[1] = membermask
    return o[0] + " = 1;  /* elect.sync: always elected in serial model */";
}},
// griddepcontrol: CDP2 grid dependency (serial CPU: dependencies already satisfied)
{"griddepcontrol.launch_dependents", [](auto&) {
    return "/* griddepcontrol.launch_dependents: serial noop */";
}},
{"griddepcontrol.wait",   [](auto&){ return "/* griddepcontrol.wait: serial noop */"; }},
{"griddepcontrol.wait_ifnot_lbi", [](auto&){ return "/* griddepcontrol.wait_ifnot_lbi: serial noop */"; }},
// setmaxnreg: register reconfiguration (no-op on CPU)
{"setmaxnreg.inc.sync.aligned.u32", [](auto&){ return "/* setmaxnreg: no-op on CPU */"; }},
{"setmaxnreg.dec.sync.aligned.u32", [](auto&){ return "/* setmaxnreg: no-op on CPU */"; }},
// cp.reduce.async.bulk (write-combining reduction to global, serial = direct store)
{"cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.f32", [](auto& o) {
    return "{ float* _dst=(float*)(uintptr_t)(" + o[0] + ");"
           "  const float* _src=(const float*)(uintptr_t)(" + o[1] + ");"
           "  size_t _n=(" + o[2] + ")>>2;"
           "  for(size_t _i=0;_i<_n;_i++) _dst[_i]+=_src[_i]; }";
}},
```

---

## 🟡 P3 — Medium Priority: cuDNN Backend v8 Gaps

### P3-A: CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR
**File**: `src/api/cudnn/cudnn_backend_api.cpp`
**Effort**: 3 days
**Blocked**: cuDNN Frontend library v0.7+ (PyTorch 2.x uses this for fused ops)

Add `case CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR:` to the descriptor switch. The execution phase must apply the pointwise op (RELU, GELU, SWISH, SIGMOID, ADD, MUL, etc.) to a tensor descriptor:

```cpp
case CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR: {
    auto xDesc = desc->getAttr<cudnnBackendDescriptor_t>(CUDNN_ATTR_OPERATION_POINTWISE_XDESC);
    auto yDesc = desc->getAttr<cudnnBackendDescriptor_t>(CUDNN_ATTR_OPERATION_POINTWISE_YDESC);
    auto pwDesc = desc->getAttr<cudnnBackendDescriptor_t>(CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR);
    // pwDesc has CUDNN_ATTR_POINTWISE_MODE (e.g. CUDNN_POINTWISE_RELU_FWD)
    // Apply the pointwise op element-wise across the tensor data pointers
    auto mode = pwDesc->getAttr<cudnnPointwiseMode_t>(CUDNN_ATTR_POINTWISE_MODE);
    // Dispatch to existing activation logic in cudnn_activation.cpp
    cudnn_apply_pointwise_op(xDesc, yDesc, mode, desc->alpha1, desc->alpha2);
    break;
}
```

### P3-B: CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR
**File**: `src/api/cudnn/cudnn_backend_api.cpp`
**Effort**: 1 day

Engine heuristic must return at least one `CUDNN_BACKEND_ENGINE_CFG_DESCRIPTOR`. In VGRE there is only one engine (the CPU emulation engine), so the heuristic always returns it:

```cpp
case CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: {
    // Return single engine config pointing to VGRE's only engine
    auto& results = desc->engineHeurResults;
    results.resize(1);
    results[0] = createVgreEngineConfig(desc);
    desc->setAttr(CUDNN_ATTR_ENGINEHEUR_RESULTS, results.data(), 1);
    break;
}
```

### P3-C: cudnnRNNForwardTrainingEx / InferenceEx
**File**: `src/api/cudnn/cudnn_rnn.cpp`
**Effort**: 1.5 days

The `Ex` variants add packed sequence support via `cudnnRNNDataDescriptor_t`. Implementation: read the sequence lengths from the data descriptor and zero-mask padding positions in the forward/backward passes:

```cpp
cudnnStatus_t cudnnRNNForwardTrainingEx(
    cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc,
    const cudnnRNNDataDescriptor_t xDesc, const void* x,
    const cudnnTensorDescriptor_t hxDesc, const void* hx,
    const cudnnTensorDescriptor_t cxDesc, const void* cx,
    const cudnnFilterDescriptor_t wDesc, const void* w,
    const cudnnRNNDataDescriptor_t yDesc, void* y,
    const cudnnTensorDescriptor_t hyDesc, void* hy,
    const cudnnTensorDescriptor_t cyDesc, void* cy,
    const cudnnTensorDescriptor_t kDesc, const void* keys,
    const cudnnRNNDataDescriptor_t cDesc, void* cAttn,
    const cudnnRNNDataDescriptor_t iDesc, void* iAttn,
    const cudnnRNNDataDescriptor_t qDesc, void* qAttn,
    void* workSpace, size_t workSpaceSizeInBytes,
    void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    // Extract seqLengthArray from xDesc, call internal RNN forward with masking
    std::vector<int> seqLens = cudnnRNNDataDescGetSeqLengths(xDesc);
    return vgre_rnn_forward_variable_len(handle, rnnDesc, seqLens.data(),
                                         x, hx, cx, w, y, hy, cy,
                                         workSpace, reserveSpace);
}
```

---

## 🟡 P3-D: cuSolver 64-bit Type-Erasure API
**File**: `src/api/cusolver/cusolver_dense.cpp`
**Effort**: 1 day
**Blocked**: JAX `jax.scipy.linalg`, Julia `CUDA.jl`, modern Python CUDA wrappers

The `X`-prefix functions are type-erasure wrappers that dispatch based on `cudaDataType`:

```cpp
cusolverStatus_t cusolverDnXgetrf(cusolverDnHandle_t handle,
    cusolverDnParams_t params, int64_t m, int64_t n,
    cudaDataType dataTypeA, void* A, int64_t lda,
    int64_t* ipiv, cudaDataType computeType,
    void* bufferOnDevice, size_t workspaceInBytesOnDevice,
    void* bufferOnHost, size_t workspaceInBytesOnHost,
    int* info) {
    if (dataTypeA == CUDA_R_32F)
        return cusolverDnSgetrf(handle, (int)m, (int)n, (float*)A, (int)lda,
                                (float*)bufferOnDevice, (int*)ipiv, info);
    if (dataTypeA == CUDA_R_64F)
        return cusolverDnDgetrf(handle, (int)m, (int)n, (double*)A, (int)lda,
                                (double*)bufferOnDevice, (int*)ipiv, info);
    return CUSOLVER_STATUS_NOT_SUPPORTED;
}
// Implement cusolverDnXpotrf, cusolverDnXgesvd, cusolverDnXsygvd similarly
```

---

## ✅ P3-E: ncclSend / ncclRecv / ncclAllToAll / ncclGather / ncclScatter — Already Implemented
**File**: `src/api/nccl/nccl_p2p.cpp` (180 lines)
**Status**: Fully implemented with real barrier-based shared-memory p2p using `p2p_slots`, generation counter, and condvar wait. No action needed.
Also confirmed: `ncclAllGather` in `src/api/nccl/nccl_collectives.cpp` at line ~399.

---

## 🔵 P5 — Long-Term / Large Scope

These require significant architectural work. Listed for roadmap awareness.

| Item | Gap | Effort |
|---|---|---|
| SASS binary execution | Precompiled CUDA libraries unusable | Very large — full ISA simulator |
| MPS multi-process | Single process per device | Large — IPC context sharing |
| OpenMP `__syncthreads` | Very large kernels exhaust OS thread limit | Medium — two-level dispatch |
| CUPTI hardware counters | Hardware PMU access | Medium — VFIO PMU passthrough |
| FP8 PTX full suite (wgmma.*) | Hopper FP8 training at scale | Large — PTX translator + emulation |
| cuSPARSE Generic API | All modern sparse frameworks | Medium — new descriptor-based API |
| cuMemAddressReserve Windows | cuVirtual memory on Windows | 1 day — VirtualAlloc2/MapViewOfFile3 |

---

## Test Coverage

All previously-untested areas now have dedicated tests (123 tests total, 121 pass, 2 Python skipped):

| Area | Test | Status |
|---|---|---|
| cuDNN Backend v8 graph execution | `test_cudnn_backend_v8` | ✅ Pass |
| Cross-module PTX linking | `test_ptx_cross_module_link` | ✅ Pass |
| MTGP32 statistical uniformity | `test_curand_mtgp32` | ✅ Pass |
| cuSolver batched APIs (potrfBatched, getrsBatched) | `test_cusolver_batched` | ✅ Pass |

### Tests Needed for New P1/P2/P3 Features

| Feature | Recommended Test File | What to Test |
|---|---|---|
| `cudaFuncSetAttribute` | `tests/api/test_func_set_attribute.cpp` | Kernel with 64 KB dynamic smem; verify attribute stored and launch succeeds |
| `redux.sync` bitwise | `tests/integration/test_ptx_redux_bitwise.cpp` | PTX kernel with `redux.sync.and/or/xor`; verify values match CPU reference |
| `cuStreamWaitValue32` | `tests/api/test_stream_memops.cpp` | Write value, then wait with GEQ flag; verify ordering |
| `cuMemAllocAsync` driver | `tests/api/test_cumem_async.cpp` | Pool create, alloc, free, pool destroy |
| `cudaGetProcAddress` | `tests/api/test_proc_address.cpp` | Resolve `cudaMemcpy`, `cudaLaunchKernel`; verify non-null |
| `cuLibraryLoadData` | `tests/api/test_cu_library.cpp` | Load PTX via cuLibraryLoadData; call kernel via cuKernelGetFunction |
| `ldmatrix/stmatrix` PTX | `tests/integration/test_ptx_ldmatrix.cpp` | PTX kernel with ldmatrix.x4; verify values match manual load |
| FP8 cvt + mma PTX | `tests/integration/test_ptx_fp8_mma.cpp` | Pack FP32→FP8, run m16n8k32 MMA, verify output within tolerance |
| `ncclSend/ncclRecv` | `tests/api/test_nccl_p2p.cpp` | 2-rank communicator, send from rank 0, recv on rank 1 |
| `cusolverDnXgetrf` | `tests/api/test_cusolver_xgetrf.cpp` | 4×4 float32 and float64 solve; verify residual < 1e-5 |
| cuDNN POINTWISE op | `tests/api/test_cudnn_pointwise.cpp` | Build graph with RELU pointwise op; verify output matches CPU relu |
| `cudnnRNNForwardTrainingEx` | `tests/api/test_cudnn_rnn_ex.cpp` | Variable-length sequence batch; verify packed output matches fixed-length reference |

---

## Recent Implementation Additions (2026-05-29)

- **CUDA TMA** (`src/api/cuda_driver/cuda_driver_tma.cpp`): Full `cuTensorMapEncodeTiled` and `cuTensorMapEncodeIm2col` implementations; `VgreTMADescriptor` extended to 128 bytes with `boxDim`, `rank`, `tag`, and im2col parameters.
- **mbarrier PTX** (`src/compiler/ptx/ptx_conversion.cpp`): All Hopper SM90 mbarrier variants translated (init, arrive, arrive.noComplete, test_wait, try_wait, try_wait.parity, wait, wait.parity, inval, complete_tx) — no-ops in serial CPU model since async copies complete synchronously.
- **fence.proxy PTX**: `fence.proxy.async`, `fence.proxy.async.shared::cta`, `fence.proxy.tensormap::generic.*` variants all mapped to `__atomic_thread_fence` or no-ops.
- **tensormap.replace PTX**: Updates `VgreTMADescriptor::baseAddr` at runtime for dynamic descriptor modification.
- **bar.sync PTX**: Fixed from comment to `__syncthreads()` call for correct block synchronization.
- **SharedMemory test fixes**: `ExternSharedIntegration` and `StaticSharedIntegration` now pass (shared memory buffer properly zeroed and reused via `ensureCapacity/reset`).
