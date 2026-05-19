# VGRE — Honest Feature Status (Code-Verified Audit)

**Audit Date**: 2026-05-19 (updated)  
**Method**: Direct source file reads + grep analysis of every file category.  
**Scope**: tcp_cluster (all 43 files), advanced/, core/, api/, runtime/, compiler/, scripts/  
**Policy**: ✅ Fixed = real computation confirmed in source. ⚠️ Issue = confirmed by reading the actual line(s) cited. Items not listed here have been resolved.

---

## Section 1 — TCP Cluster: Open Issues

### 1.1 Default Port Still Hardcoded in 4 Locations
`include/vgre/advanced/tcp_cluster/tcp_cluster_defaults.h` defines `kDefaultClusterPort=7777` and `kDefaultDiscoveryPort=7778`. `discovery_manager.cpp` and `configuration_manager_validation.cpp` were updated. The following **still use raw literals**:

| File | Line | Value |
|---|---|---|
| `src/advanced/hybrid_compute_manager_remote.cpp` | 91 | `node.port = 7777;` |
| `src/advanced/vgre_worker_cli.cpp` | 48 | `int port = 7777;` |
| `scripts/vgre-start.sh` | 35 | `PORT="${VGRE_PORT:-7777}"` |
| `scripts/Start-VGRE.ps1` | 123 | `$port = "7777"` |

### 1.2 Configuration Manager Hand-Rolled JSON Parser
**File**: `src/advanced/tcp_cluster/configuration_manager_file_io.cpp`  
Implements its own JSON/YAML/INI parser by string scanning. The codebase has `llvm::json::parse()` via `<llvm/Support/JSON.h>` already linked. The hand-rolled parser has no Unicode escape handling and only supports nested objects up to 2 levels.

### 1.3 `tcp_cluster_validation.cpp` Unknown-Platform Branch Blocks Cluster Join
**File**: `src/advanced/tcp_cluster/tcp_cluster_validation.cpp` lines 250, 265  
```cpp
VGRE_LOG_ERROR("TCPCluster", "Unknown platform - mesh topology not supported");
```
Any platform not explicitly `_WIN32` or `__linux__` (e.g., RISC-V, FreeBSD) triggers `ERR_NOT_SUPPORTED` and cannot join the cluster. macOS is now detected in `PlatformDetection` but the validation path still lacks an `__APPLE__` branch.

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

## Section 3 — Scripts: Open Issues

### 3.1 `test_python_authoritative.py` Masks Launch Failures
**File**: `tests/python/test_python_authoritative.py:62`
```python
# Launch a dummy kernel if possible, or just check JSON
```
When kernel launch fails the test falls back to JSON-only check and still passes. Launch failures are invisible.

### 3.2 `vgre_sync.sh` Does Not Verify LLVM Version
**File**: `scripts/vgre_sync.sh`  
Installs LLVM if missing but does not verify it is version 18. System `llvm-17` produces a silent miscompile. Requires `clang-18 --version` check.

### 3.3 `setup-cluster.sh` No Reachability Check for Master IP
**File**: `scripts/setup-cluster.sh:16`  
No validation that the master IP is reachable before starting worker. Typos cause silent connection failures.

---

## Section 4 — Codebase-Wide: Confirmed Real Implementations

| Component | Verified |
|---|---|
| cuBLAS L1/L2/L3 | ✅ Cache-blocked GEMM, CBLAS delegation |
| cuFFT 1D/2D/3D | ✅ Cooley-Tukey + Bluestein |
| cuDNN Conv/BN/Act/Pool/Softmax/Dropout/MHA/CTC/LRN | ✅ Real CPU loops, OpenMP |
| cuDNN RNN — LSTM, GRU, RNN_TANH, RNN_RELU | ✅ Full forward + BPTT backward + weight gradients (8/8 tests) |
| cuRAND host + device | ✅ XORWOW, Philox, MRG32K3A, Sobol |
| cuSPARSE SpMV/SpMM, ILU0/IC0, triangular solve | ✅ Real CSR |
| cuSolver LU (getrf/getrs), QR (geqrf/ormqr), SVD (gesvd), eigen (syevd), least-sq (gelsd) | ✅ LAPACK-backed |
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

## Section 5 — Confirmed Wrong Results (Silent)

**These APIs return `SUCCESS` but compute incorrect output.**

| API | What's Wrong | Who Is Affected |
|---|---|---|
| `cuOccupancyMaxActiveBlocksPerMultiprocessor` | Always uses 2048 threads/SM (Ampere-specific hardcode) | Auto-tuned launch configs on non-Ampere GPUs |

**Resolved since last audit:**
- `cudnnRNNForwardInference/Training` with `CUDNN_LSTM` — ✅ Fixed: real 4-gate cell with cell state
- `cudnnRNNForwardInference/Training` with `CUDNN_GRU` — ✅ Fixed: real reset/update/new gates
- `cudnnRNNBackwardData` / `cudnnRNNBackwardWeights` — ✅ Fixed: full BPTT implemented
- `ncclAllReduce` with `ncclFloat16`/`ncclBfloat16` — ✅ Fixed: `applyReduceFp16` with proper upcast/accumulate/downcast
- Bandwidth utilization display — ✅ Fixed: reads `VGRE_CLUSTER_LINK_GBPS`

---

## Section 6 — Confirmed Stubs (Accept Calls, Return Success / NOT_SUPPORTED, Do Nothing Real)

| API / Feature | File | Gap |
|---|---|---|
| PTX multi-module linker (`cuLink*`) | `cuda_driver_module.cpp:189-273` | Concatenates PTX only, no cross-module symbol resolution |
| cuDNN Backend v8 execution | `cudnn_backend_api.cpp:744` | Returns `CUDNN_STATUS_NOT_SUPPORTED` |
| cuSPARSE SpGEMM | `cusparse_factorization.cpp:497,561` | Returns `NOT_SUPPORTED` (requires UMFPACK) |
| cuSolver batched APIs (potrf/getrf batched) | Not implemented | No code exists |
| cuRAND MTGP32 device-side | `curand_kernel.h` | Only XORWOW + Philox present |
| Token manager: macOS Keychain | `token_manager_macos.cpp:102` | Always `ERR_NOT_SUPPORTED` |
| Token manager: Linux libsecret | `token_manager_linux.cpp:204,218` | Always `ERR_NOT_SUPPORTED` |
| Token manager: Windows DPAPI | `token_manager_win32.cpp:70` | Always `ERR_NOT_SUPPORTED` |
| VFIO GPU passthrough | `gpu_passthrough.cpp` | Detected, not activated |

---

## Section 7 — Not Implemented (No Code Exists)

| Missing | Impact |
|---|---|
| cuDNN Backend v8 graph execution | PyTorch ≥ 2.0 ops broken |
| cuSPARSE SpGEMM | Graph neural networks (DGL, PyG) broken |
| cuSolver batched (Potrf, Getrf batched) | Transformer attention batched solves broken |
| SASS execution | Pre-compiled CUDA libraries unusable |
| PTX cross-module symbol linking | Separate compilation workflows broken |
| CUDA TMA instructions | Hopper TMA kernels fail to JIT-compile |
| Multi-GPU P2P | `cudaMemcpyPeer` uses slow host staging |
| Hardware performance counters | No CUPTI/profiling capability |
| MPS multi-process | Single process per virtual device |
| cuFFT CUDA_C_16BF | Bfloat16 complex FFT absent |
