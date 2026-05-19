# VGRE Implementation Plan

**Version**: 6.0.0  
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

### Code Quality (P4 items — partial)
- ✅ P4-1: `cleanupServerAuthThreads()` merged into `server_loop_core.cpp`; `server_loop_auth_mgmt.cpp` deleted
- ✅ P4-4: Python integration tests registered in `tests/CMakeLists.txt` with `SKIP_REGULAR_EXPRESSION` for missing deps
- ✅ P4-5: `benchmark.py` exits 1 with error message when VGRE bindings are not importable

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

##  P4 — Code Quality / Structural Issues (remaining)

### P4-2: Dispatch Files Naming Confusion
4 files implement the `DispatchManager` class: `dispatch_impl.cpp` (actually partition dispatch), `dispatch_manager_core.cpp`, `dispatch_manager_partitioned.cpp`, `dispatch_manager_remote.cpp`.

`dispatch_impl.cpp` is misnamed — its header comment says "Partition dispatch implementation". Rename to `dispatch_partition_impl.cpp` for clarity.

**Estimated effort**: 5 minutes (rename + update CMakeLists.txt).

### P4-3: Configuration Manager Hand-Rolled JSON Parser
**File**: `configuration_manager_file_io.cpp`  
The file implements its own JSON/YAML/INI parser using string scanning. The rest of the codebase uses `llvm::json::parse()` (available via `<llvm/Support/JSON.h>` already linked). The hand-rolled parser has edge cases (no Unicode escape handling, no nested object support beyond 2 levels).

**Fix**: Replace the JSON parsing section with `llvm::json::parse()`. YAML/INI can remain hand-rolled since LLVM JSON only covers JSON.

**Estimated effort**: 1-2 days (careful migration with tests).

### P4-6: Remaining Hardcoded Port Occurrences
**Files** (not yet updated to use `kDefaultClusterPort`):
- `src/advanced/hybrid_compute_manager_remote.cpp:91` → `node.port = 7777;`
- `src/advanced/vgre_worker_cli.cpp:48` → `int port = 7777;`
- `scripts/vgre-start.sh:35` → `PORT="${VGRE_PORT:-7777}"`
- `scripts/Start-VGRE.ps1:123` → `$port = "7777"`

The `tcp_cluster_defaults.h` constant exists but these four locations were not yet updated.

**Estimated effort**: 30 minutes.

### P4-7: `test_python_authoritative.py` Masks Launch Failures
**File**: `tests/python/test_python_authoritative.py:62`  
Comment: `# Launch a dummy kernel if possible, or just check JSON`. When kernel launch fails the test falls back to JSON-only check and still passes, hiding actual launch errors.

**Estimated effort**: 1 hour.

### P4-8: `vgre_sync.sh` Does Not Verify LLVM Version
**File**: `scripts/vgre_sync.sh`  
Installs LLVM if missing but does not verify it is version 18. Installing system `llvm-17` produces a silent miscompile.

**Fix**: Add `clang-18 --version | grep -q 'version 18'` check before proceeding.

**Estimated effort**: 30 minutes.

---

## 🔵 P5 — Long-Term / Large Scope

These require significant architectural work. Listed for roadmap awareness.

| Item | Gap | Effort |
|---|---|---|
| cuDNN Backend v8 | PyTorch 2.x dependency (P1-1) | 4-6 days |
| cuSPARSE SpGEMM | GNN workloads (P1-2) | 3-4 days |
| cuSolver batched APIs | Transformer attention (P1-3) | 1 day |
| PTX multi-module linker | Separate compilation (P1-4) | 3-4 days |
| CUDA TMA instructions | Hopper kernels | Medium — PTX translator extension |
| SASS binary execution | Precompiled CUDA libraries | Very large — full ISA simulator |
| Multi-GPU P2P | Frameworks using cudaMemcpyPeer | Large — multi-context model |
| CUPTI hardware counters | Profiling | Medium — proxy software counters |
| MPS multi-process | Single process per device | Large — IPC context sharing |
| cuFFT BF16 | bfloat16 complex FFT | 0.5 days |
| OpenMP for `__syncthreads` kernels | Single-threaded fallback | Medium — two-level dispatch |
| OS keystore integration | macOS Keychain, libsecret, DPAPI | 1 day each |
| cuOccupancy accurate SM count | Hardcoded 2048 threads/SM (Ampere only) | 0.5 days |

---

## Test Coverage Gaps (Currently Untested)

The 119-test suite passes (2 Python tests skip when deps missing) but does NOT validate:

| Untested Area | Risk If Undetected |
|---|---|
| cuDNN Backend v8 | PyTorch 2.x ops fail silently |
| Cross-module PTX linking | Separate-compilation apps JIT-fail |
| MTGP32 device cuRAND | JIT compile error in kernel |
| cuSolver batched | Transformer attention fails |
| cuOccupancy SM heuristic | Suboptimal launch configs go undetected |
| `vgre_sync.sh` LLVM version | Silent miscompile with wrong LLVM |
