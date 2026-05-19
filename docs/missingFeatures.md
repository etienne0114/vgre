# VGRE — Honest Feature Status (Code-Verified Audit)

**Audit Date**: 2026-05-19 (v2)  
**Method**: Direct source file reads + grep analysis of every file category.  
**Scope**: tcp_cluster (all 43 files), advanced/, core/, api/, runtime/, compiler/, scripts/  
**Policy**: ✅ Fixed = real computation confirmed in source. ⚠️ Issue = confirmed by reading the actual line(s) cited. Items not listed here have been resolved.

---

## Section 1 — TCP Cluster: Open Issues

### 1.1 Configuration Manager Hand-Rolled JSON Parser
**File**: `src/advanced/tcp_cluster/configuration_manager_file_io.cpp`  
Implements its own JSON/YAML/INI parser by string scanning. The codebase has `llvm::json::parse()` via `<llvm/Support/JSON.h>` already linked. The hand-rolled parser has no Unicode escape handling and only supports nested objects up to 2 levels.

---

## Section 2 — Platform Header Issues

| File | Header / Syscall | Status |
|---|---|---|
| `adaptive_execution_engine*.cpp` (3 files) | `<dirent.h>`, `<sys/ioctl.h>`, `<sys/syscall.h>` (Linux); `<IOKit/IOKitLib.h>` (macOS) | Guarded — **OK** |
| `secure_channel_crypto.cpp` | `<sys/random.h>` | Guarded — **OK** |
| `nccl_communicator.cpp`, `nccl_core.cpp` | `<sys/random.h>` | Guarded — **OK** |
| `vector_engine*.cpp` | `<sys/syscall.h>` SYS_arch_prctl for AMX | Guarded — **OK** |
| `websocket_transport.cpp` | `<sys/select.h>` | Guarded — **OK** |
| `configuration_manager_validation.cpp` | `<sys/stat.h>` | ✅ **Fixed** — wrapped in `#if !defined(_WIN32)` |
| `configuration_manager_file_io.cpp` | `<sys/stat.h>` | ✅ **Fixed** — wrapped in `#if !defined(_WIN32)` |
| `ipc_manager.cpp` | `<sys/stat.h>` | ✅ **Fixed** — wrapped in `#if !defined(_WIN32)` |
| `scheduler_numa.cpp` | `<dirent.h>`, `<sys/sysctl.h>` | ⚠️ `sysctl` path for NUMA on Linux uses raw syscall — partially guarded |

---

## Section 3 — Codebase-Wide: Confirmed Real Implementations

| Component | Verified |
|---|---|
| cuBLAS L1/L2/L3 | ✅ Cache-blocked GEMM, CBLAS delegation |
| cuFFT 1D/2D/3D | ✅ Cooley-Tukey + Bluestein |
| cuDNN Conv/BN/Act/Pool/Softmax/Dropout/MHA/CTC/LRN | ✅ Real CPU loops, OpenMP |
| cuDNN RNN — LSTM, GRU, RNN_TANH, RNN_RELU | ✅ Full forward + BPTT backward + weight gradients (8/8 tests) |
| cuRAND host + device | ✅ XORWOW, Philox, MRG32K3A, Sobol |
| cuSPARSE SpMV/SpMM, ILU0/IC0, triangular solve | ✅ Real CSR |
| cuSolver LU/QR/SVD/eigen/least-sq + batched potrf/getrs | ✅ LAPACK-backed; batched loops over unbatched LAPACK |
| cuSPARSE SpGEMM | ✅ Two-pass CSR×CSR (symbolic + numeric passes in `cusparse_factorization.cpp`) |
| cuDNN Backend v8 | ✅ Full dispatch table (conv fwd/bwd, act, BN, pool, matmul, reduction, attention, pointwise) |
| cuFFT BF16 complex (`CUFFT_C16BFC`) | ✅ BF16↔float32 round-trip + Cooley-Tukey FFT |
| cuRAND MTGP32 device-side | ✅ `curandStateMtgp32` + MT19937 twist engine in `curand_kernel.h` |
| PTX multi-module linker | ✅ `.extern .func` deduplication after concatenation in `cuLinkComplete` |
| NCCL AllReduce/Broadcast/ReduceScatter | ✅ float32, float64, int32, int64, float16, bfloat16 |
| Events (timing) | ✅ steady_clock |
| CUDA Graphs | ✅ Real DAG/topological sort |
| CDP | ✅ Real recursive launch |
| UVM migration | ✅ Real mbind() syscall |
| cuVirtual memory | ✅ Real mmap(PROT_NONE) |
| TCP cluster networking | ✅ Real TCP/UDP/HMAC/AES |
| Secure channel | ✅ AES-256-GCM + PBKDF2 |
| KernelCache | ✅ Integrity checks + AST eviction |
| SM100 FP8 MMA | ✅ E4M3/E5M2 tcgen05 |

---

## Section 4 — Confirmed Wrong Results (Silent)

**These APIs return `SUCCESS` but compute incorrect output.**

| API | What's Wrong | Who Is Affected |
|---|---|---|
| No confirmed wrong-result APIs at this time | — | — |

**Resolved since last audit:**
- `cudnnRNNForwardInference/Training` with `CUDNN_LSTM` — ✅ Fixed: real 4-gate cell with cell state
- `cudnnRNNForwardInference/Training` with `CUDNN_GRU` — ✅ Fixed: real reset/update/new gates
- `cudnnRNNBackwardData` / `cudnnRNNBackwardWeights` — ✅ Fixed: full BPTT implemented
- `ncclAllReduce` with `ncclFloat16`/`ncclBfloat16` — ✅ Fixed: `applyReduceFp16` with proper upcast/accumulate/downcast
- Bandwidth utilization display — ✅ Fixed: reads `VGRE_CLUSTER_LINK_GBPS`
- `cuOccupancyMaxActiveBlocksPerMultiprocessor` — ✅ Fixed: reads `props.maxThreadsPerSM` (fallback 2048)

---

## Section 5 — Confirmed Stubs (Accept Calls, Return Success / NOT_SUPPORTED, Do Nothing Real)

| API / Feature | File | Gap |
|---|---|---|
| Token manager: macOS Keychain | `token_manager_macos.cpp:102` | Always `ERR_NOT_SUPPORTED` |
| Token manager: Linux libsecret | `token_manager_linux.cpp:204,218` | Always `ERR_NOT_SUPPORTED` |
| Token manager: Windows DPAPI | `token_manager_win32.cpp:70` | Always `ERR_NOT_SUPPORTED` |
| VFIO GPU passthrough | `gpu_passthrough.cpp` | Detected, not activated |

---

## Section 6 — Not Implemented (No Code Exists)

| Missing | Impact |
|---|---|
| SASS execution | Pre-compiled CUDA libraries unusable |
| CUDA TMA instructions | Hopper TMA kernels fail to JIT-compile |
| Multi-GPU P2P | `cudaMemcpyPeer` uses slow host staging |
| Hardware performance counters | No CUPTI/profiling capability |
| MPS multi-process | Single process per virtual device |
