# VGRE Implementation Plan

**Version**: 5.0.0  
**Date**: 2026-05-19  
**Basis**: Full code-verified audit — tcp_cluster (43 files), all advanced/, api/, core/, runtime/, compiler/, scripts/  
**Format**: Priority-ordered. Each item has a file + line reference and a concrete fix description.

---

## ✅ Completed (Verified by Code Read)

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
- ✅ cuRAND host (XORWOW, Philox, MRG32K3A, Sobol) + device (curand_kernel.h)
- ✅ cuSPARSE SpMV/SpMM, ILU0/IC0, triangular solve
- ✅ cuSolver LU/QR/SVD (LAPACK-backed)
- ✅ NCCL AllReduce/Broadcast/ReduceScatter (float32, float64, int32, int64)
- ✅ CUDA Driver API — module load, function lookup, kernel launch, cuLinkXxx (concatenation only)
- ✅ OpenCL adapter, GPU passthrough (conditional on hardware)

### Infrastructure
- ✅ TCP cluster — peer discovery, full mesh, HMAC-SHA256, ring all-reduce
- ✅ Secure channel — AES-256-GCM + PBKDF2-SHA256
- ✅ WebSocket (RFC 6455), gRPC (optional), RDMA (optional)
- ✅ Hardware token — TPM + encrypted-file fallback
- ✅ SM100 FP8 MMA (E4M3/E5M2 tcgen05)
- ✅ KernelCache — sourceHash + name integrity, AST collision eviction

---

## 🔴 P0 — Silent Wrong Results (Fix Before Any Other Work)

These must be fixed first because they silently return `SUCCESS` with incorrect data.

### P0-1: LSTM and GRU Compute Vanilla RNN Instead
**File**: `src/api/cudnn/cudnn_rnn.cpp` lines 46-91  
`rd->mode` is never checked. All modes run identical tanh RNN.

**Fix**: Add switch on `rd->mode`:
- `CUDNN_LSTM`: implement 4-gate cell (forget f, input i, output o, cell gate g), track cell state `c` across timesteps
- `CUDNN_GRU`: implement 3-gate cell (reset r, update z, new n = tanh(W_n×x + r⊙(U_n×h)))

**Estimated effort**: 2 days. Both forward and backward passes needed.

### P0-2: FP16/BF16 AllReduce Produces Garbage
**File**: `src/advanced/tcp_cluster/collective_ops_manager.cpp`  
`applyReduce<T>` has explicit instantiations for float, double, int32, int64 only. `ncclFloat16`/`ncclBfloat16` dispatch hits an unhandled code path.

**Fix**: Add `applyReduce<uint16_t>` with correct FP16 accumulation (upcast to float, accumulate, downcast). BF16: use existing `wgmma_bf16_to_f32` / `f32_to_fp8e*` conversions.

**Estimated effort**: 1 day.

### P0-3: Bandwidth Utilization Always Shows ≤1% on Fast NICs
**File**: `src/advanced/tcp_cluster/diagnostic_logger.cpp:293`  
`bandwidth_utilization = average_bandwidth_gbps / 1.0` — denominator hardcoded to 1.

**Fix**: Read link speed via `VGRE_CLUSTER_LINK_GBPS` env var. If unset, detect from NIC using `ethtool -s` output on Linux or `GetIfTable2` on Windows. Default to the measured peak if detection fails.

**Estimated effort**: 0.5 days.

---

## 🟠 P1 — Functionality Gaps (Impact ML Workloads)

### P1-1: cuDNN Backend v8 Execution Graph
**File**: `src/api/cudnn/cudnn_backend_api.cpp:744`  
`cudnnBackendExecute` returns NOT_SUPPORTED. PyTorch ≥ 2.0 uses v8 backend for all cuDNN ops.

**Fix**: Map the v8 descriptor graph to existing v7 calls (cudnnConvolutionForward, etc.). The descriptor types already parsed; the execution path just needs a dispatch table.

**Estimated effort**: 4-6 days.

### P1-2: cuSPARSE SpGEMM Without External Library
**File**: `src/api/cusparse/cusparse_factorization.cpp:497,561`  
Returns NOT_SUPPORTED. Graph neural networks (DGL, PyG) use SpGEMM for message passing.

**Fix**: Implement row-merge CSR×CSR multiplication in-house. Algorithm:
1. Symbolic pass: compute per-row nnz of output
2. Numeric pass: accumulate with hash-map or sorted merge

**Estimated effort**: 3-4 days.

### P1-3: cuSolver Batched APIs
Not implemented. PyTorch uses these for transformer attention.

**Fix**: Loop the unbatched routines with stride arithmetic:
```cpp
for (int b = 0; b < batchSize; ++b)
    cusolverDnSgetrf(A + b*lda*n, ldb, ipiv + b*n, info + b);
```

**Estimated effort**: 1 day.

### P1-4: PTX Multi-Module Symbol Linking
**File**: `src/api/cuda_driver/cuda_driver_module.cpp:189-273`  
`cuLinkComplete` concatenates PTX without symbol resolution. Cross-module calls produce JIT errors.

**Fix**: After concatenating PTX buffers:
1. Parse each buffer's `.func` declarations to build a symbol table
2. Rewrite cross-module calls to match the merged symbol names
3. Deduplicate shared `.extern` and `.visible` declarations before passing to LLVM JIT

**Estimated effort**: 3-4 days.

---

## 🟡 P2 — TCP Cluster Hardcoding and Heuristics

### P2-1: Unify Default Port Constant
**Files**: 6 locations listed in missingFeatures.md §1.2

**Fix**: Create `include/vgre/advanced/tcp_cluster_defaults.h`:
```cpp
namespace vgre::advanced {
constexpr int kDefaultClusterPort    = 7777;
constexpr int kDefaultDiscoveryPort  = 7778;
}
```
Replace all 6 hardcoded occurrences with this constant.

**Estimated effort**: 1 hour.

### P2-2: Use `vgre_get_type_size()` in `sendScalarArg`
**File**: `src/advanced/tcp_cluster/memory_sync_manager.cpp:202`

**Fix**: Replace:
```cpp
size_t arg_size = 8;
if (type == ArgType::INT32 || ...) arg_size = 4;
```
With:
```cpp
size_t arg_size = vgre_get_type_size(static_cast<int>(type));
```

**Estimated effort**: 5 minutes.

### P2-3: Fix Windows RDMA Assumption
**File**: `src/advanced/tcp_cluster/shared_utilities_base.cpp:60`

**Fix**: Change `info.supports_rdma = true` on Windows to:
```cpp
info.supports_rdma = (vgre_get_config("VGRE_RDMA_ENABLED") != nullptr);
```

**Estimated effort**: 5 minutes.

### P2-4: Make SHM Result Offset Configurable
**File**: `src/advanced/tcp_cluster/dispatch_manager_core.cpp:14`

**Fix**:
```cpp
result_shm_offset_([]() -> uint64_t {
    const char* e = vgre_get_config("VGRE_SHM_RESULT_OFFSET");
    return e ? std::stoull(e) : 128ULL * 1024 * 1024;
}())
```

**Estimated effort**: 15 minutes.

### P2-5: Worker AllReduce/Barrier Timeout Must Match Master
**File**: `collective_ops_manager.cpp:145,179,210`

**Fix**: Replace `std::chrono::seconds(30)` with the same configurable lookup used on master:
```cpp
int ms = 30000;
const char* e = vgre_get_config("VGRE_REDUCTION_TIMEOUT_MS");
if (e) { int v = std::atoi(e); if (v > 0) ms = v; }
wait_for(lock, std::chrono::milliseconds(ms), predicate);
```

**Estimated effort**: 30 minutes.

### P2-6: Fix Bandwidth Utilization Denominator
**File**: `diagnostic_logger.cpp:293` — see P0-3 above.

### P2-7: Fix Bandwidth Probe Payload (Use Random Data)
**File**: `server_loop_connection_handling.cpp:73`

**Fix**: Fill probe buffer with pseudo-random bytes instead of zeroes:
```cpp
std::mt19937_64 rng(std::random_device{}());
for (size_t i = sizeof(uint64_t); i < probe_buf.size(); ++i)
    probe_buf[i] = static_cast<uint8_t>(rng());
```

**Estimated effort**: 15 minutes.

### P2-8: Unify Metrics Output to Single Configurable Path
**Files**: `diagnostic_logger.cpp:414` (writes `/tmp/vgre_tcp_cluster_metrics.json`) and `tcp_cluster_metrics.cpp:62` (writes `vgre_tcp_cluster_metrics.json` in CWD)

**Fix**: Consolidate to one function in `tcp_cluster_metrics.cpp`. Path controlled by `VGRE_METRICS_OUTPUT_PATH` env var. Remove the redundant write in `diagnostic_logger.cpp`.

**Estimated effort**: 1 hour.

### P2-9: AllReduce Timeout Heuristic Is Unsound
**File**: `collective_ops_manager.cpp:87-93`  
Current: `2 × maxKernelLatencyMs + 5000`. Unrelated to AllReduce data volume.

**Fix**: Base the timeout on actual data transfer estimate:
```cpp
// Estimate: size/bandwidth + 3× network RTT buffer
double transferMs = (total_bytes * 8.0) / (measured_bandwidth_gbps * 1e9) * 1000.0;
reductionTimeoutMs = static_cast<int>(transferMs * num_workers * 3.0) + 5000;
reductionTimeoutMs = std::max(reductionTimeoutMs, 10000);
```

**Estimated effort**: 1 hour.

---

## 🟡 P3 — Platform / Header Issues

### P3-1: `<sys/stat.h>` Used Unguarded in 3 Files
**Files**:
- `src/advanced/tcp_cluster/configuration_manager_validation.cpp:14`
- `src/advanced/tcp_cluster/configuration_manager_file_io.cpp:7`
- `src/advanced/ipc_manager.cpp:8`

These will fail to compile on Windows without `#if !defined(_WIN32)` guards. The uses are typically `stat()` for file existence / mtime checks.

**Fix**: Replace `stat()` calls with `vgre::os::file_exists()` and `vgre::os::file_mtime()` from `os_backend.h`, which already have the cross-platform implementations.

**Estimated effort**: 1-2 hours.

### P3-2: macOS Platform Not Mapped in PlatformDetection
**File**: `src/advanced/tcp_cluster/shared_utilities_base.cpp:57-70`  
`PlatformDetection::getCurrentPlatform()` has `#ifdef _WIN32`, `#elif defined(__linux__)`, but no `#elif defined(__APPLE__)`. macOS falls through to `PlatformType::UNKNOWN` with no SHM/RDMA flags set.

**Fix**: Add `#elif defined(__APPLE__)` branch:
```cpp
info.type = PlatformType::MACOS; info.name = "macOS";
info.supports_shm_local = true; info.supports_rdma = false;
```

**Estimated effort**: 10 minutes.

---

## 🟢 P4 — Code Quality / Structural Issues

### P4-1: Merge `server_loop_auth_mgmt.cpp` Into `server_loop_core.cpp`
**File**: `server_loop_auth_mgmt.cpp` (26 lines, 1 function)  
A 26-line file is not a module. Move `cleanupServerAuthThreads()` into `server_loop_core.cpp` and delete the file.

**Estimated effort**: 15 minutes.

### P4-2: Dispatch Files Naming Confusion
4 files implement the `DispatchManager` class: `dispatch_impl.cpp` (actually partition dispatch), `dispatch_manager_core.cpp`, `dispatch_manager_partitioned.cpp`, `dispatch_manager_remote.cpp`.

`dispatch_impl.cpp` is misnamed — its header comment says "Partition dispatch implementation". Rename to `dispatch_partition_impl.cpp` for clarity.

**Estimated effort**: 5 minutes (rename + update CMakeLists.txt).

### P4-3: Configuration Manager Hand-Rolled JSON Parser
**File**: `configuration_manager_file_io.cpp`  
The file implements its own JSON/YAML/INI parser using string scanning. The rest of the codebase uses `llvm::json::parse()` (available via `<llvm/Support/JSON.h>` already linked). The hand-rolled parser has edge cases (no Unicode escape handling, no nested object support beyond 2 levels).

**Fix**: Replace the JSON parsing section with `llvm::json::parse()`. YAML/INI can remain hand-rolled since LLVM JSON only covers JSON.

**Estimated effort**: 1-2 days (careful migration with tests).

### P4-4: Python Tests Not in CTest
**Files**: `tests/python/test_pytorch.py`, `tests/python/test_tensorflow.py`

**Fix**: Add to `tests/CMakeLists.txt`:
```cmake
find_program(PYTHON3 python3)
if (PYTHON3)
    add_test(NAME PyTorchIntegration
             COMMAND ${PYTHON3} ${CMAKE_SOURCE_DIR}/tests/python/test_pytorch.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(PyTorchIntegration PROPERTIES
                         ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}"
                         TIMEOUT 120)
endif()
```

**Estimated effort**: 30 minutes.

### P4-5: Benchmark Script Silently Passes on Missing Bindings
**File**: `tests/python/benchmark.py` + `scripts/run_benchmarks.sh`

**Fix**: In `benchmark.py`, after the import check:
```python
if not VGRE_AVAILABLE:
    print("ERROR: VGRE bindings not importable", file=sys.stderr)
    sys.exit(1)
```

**Estimated effort**: 5 minutes.

---

## 🔵 P5 — Long-Term / Large Scope

These require significant architectural work. Listed for roadmap awareness.

| Item | Gap | Effort |
|---|---|---|
| LSTM / GRU (P0-1) | Already listed above — must be P0 | 2 days |
| cuDNN Backend v8 | PyTorch 2.x dependency | 4-6 days |
| cuSPARSE SpGEMM | GNN workloads | 3-4 days |
| PTX multi-module linker | Separate compilation | 3-4 days |
| CUDA TMA instructions | Hopper kernels | Medium — PTX translator extension |
| SASS binary execution | Precompiled CUDA libraries | Very large — full ISA simulator |
| Multi-GPU P2P | Frameworks using cudaMemcpyPeer | Large — multi-context model |
| CUPTI hardware counters | Profiling | Medium — proxy software counters |
| MPS multi-process | Single process per device | Large — IPC context sharing |
| cuFFT BF16 | bfloat16 complex FFT | 0.5 days |
| cuDNN RNN backward (BPTT) | RNN training | 2 days |
| OpenMP for `__syncthreads` kernels | Single-threaded fallback | Medium — two-level dispatch |
| OS keystore integration | macOS Keychain, libsecret, DPAPI | 1 day each |

---

## Test Coverage Gaps (Currently Untested)

The 117-test suite passes but does NOT validate:

| Untested Area | Risk If Undetected |
|---|---|
| LSTM/GRU numerical output | Silent wrong gradients in training |
| FP16/BF16 AllReduce | Wrong parameters after distributed step |
| cuDNN Backend v8 | PyTorch 2.x ops fail silently |
| Bandwidth utilization accuracy | Monitoring shows 0% on fast cluster |
| Cross-module PTX linking | Separate-compilation apps JIT-fail |
| Python binding importability | `run_benchmarks.sh` exits 0 on broken install |
| MTGP32 device cuRAND | JIT compile error in kernel |
| cuSolver batched | Transformer attention fails |
