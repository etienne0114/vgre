# VGRE Implementation Plan

**Version**: 7.0.0  
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

---

## � P4 — Code Quality / Structural Issues (remaining)

### P4-3: Configuration Manager Hand-Rolled JSON Parser
**File**: `configuration_manager_file_io.cpp`  
The file implements its own JSON/YAML/INI parser using string scanning. The rest of the codebase uses `llvm::json::parse()` (available via `<llvm/Support/JSON.h>` already linked). The hand-rolled parser has edge cases (no Unicode escape handling, no nested object support beyond 2 levels).

**Fix**: Replace the JSON parsing section with `llvm::json::parse()`. YAML/INI can remain hand-rolled since LLVM JSON only covers JSON.

**Estimated effort**: 1-2 days (careful migration with tests).

---

## 🔵 P5 — Long-Term / Large Scope

These require significant architectural work. Listed for roadmap awareness.

| Item | Gap | Effort |
|---|---|---|
| CUDA TMA instructions | Hopper kernels | Medium — PTX translator extension |
| SASS binary execution | Precompiled CUDA libraries | Very large — full ISA simulator |
| Multi-GPU P2P | Frameworks using cudaMemcpyPeer | Large — multi-context model |
| CUPTI hardware counters | Profiling | Medium — proxy software counters |
| MPS multi-process | Single process per device | Large — IPC context sharing |
| OpenMP for `__syncthreads` kernels | Single-threaded fallback | Medium — two-level dispatch |
| OS keystore integration | macOS Keychain, libsecret, DPAPI | 1 day each |

---

## Test Coverage Gaps (Currently Untested)

The 119-test suite passes (2 Python tests skip when deps missing) but does NOT validate:

| Untested Area | Risk If Undetected |
|---|---|
| cuDNN Backend v8 graph execution | Untested op type crashes silently |
| Cross-module PTX linking | Merged PTX may still have symbol conflicts not caught by unit tests |
| MTGP32 statistical uniformity | MT19937 approximation may differ from real MTGP32 output |
| cuSolver batched correctness | Loop-based batch correctness under concurrent use |
