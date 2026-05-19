# VGRE — Honest Feature Status (Code-Verified Audit)

**Audit Date**: 2026-05-19  
**Method**: Direct source file reads + grep analysis of every file category.  
**Scope**: tcp_cluster (all 43 files), advanced/, core/, api/, runtime/, compiler/, scripts/  
**Policy**: Fixed = real computation confirmed. Issue = confirmed by reading the actual line(s) cited.

---

## Section 1 — TCP Cluster: Issues Found

### 1.1 Hardcoded 1 Gbps Bandwidth Reference (Wrong on Fast NICs)
**File**: `src/advanced/tcp_cluster/diagnostic_logger.cpp` line 293
```cpp
network_quality_.bandwidth_utilization =
    std::min(1.0, network_quality_.average_bandwidth_gbps / 1.0); // ← 1.0 = 1 Gbps assumed
```
`bandwidth_utilization` is computed as measured Gbps divided by a hardcoded `1.0`. On a 10/25/100 Gbps NIC, the utilization display is pegged at ≤10/4/1% even under full load. The reference bandwidth must be configurable via `VGRE_CLUSTER_LINK_GBPS` or auto-detected from the NIC.

### 1.2 Default Port Hardcoded in Five Different Places
**Files + lines**:
- `src/advanced/tcp_cluster/discovery_manager.cpp:45` → `return 7778;` (UDP discovery port)
- `src/advanced/tcp_cluster/discovery_manager.cpp:235,242,246` → `int worker_port = 7777`
- `src/advanced/tcp_cluster/configuration_manager_validation.cpp:216` → `int port = 7777; // Default port`
- `src/advanced/hybrid_compute_manager_remote.cpp:91` → `node.port = 7777; // Default cluster port`
- `src/advanced/vgre_worker_cli.cpp:48` → `int port = 7777;`
- `scripts/vgre-start.sh:35` → `PORT="${VGRE_PORT:-7777}"`

No shared constant. If the default port changes, it must be updated in 6 locations. The port used by `DiscoveryManager`, `hybrid_compute_manager_remote`, the CLI, and the start script can silently diverge.

### 1.3 sendScalarArg Ignores `vgre_get_type_size()` Already Available
**File**: `src/advanced/tcp_cluster/memory_sync_manager.cpp` lines 202-204
```cpp
size_t arg_size = 8;
if (type == ArgType::INT32 || type == ArgType::UINT32 || type == ArgType::FLOAT32) {
    arg_size = 4;
}
```
The function only branches on 3 types (4-byte) vs. everything else (8-byte). The codebase already has `vgre_get_type_size(int)` in `include/vgre/common/types.h` lines 46-59 which handles all 7 ArgType values. Using it here would be a one-line fix and would correctly handle any future ArgType additions.

### 1.4 Windows Platform Incorrectly Claims RDMA Support
**File**: `src/advanced/tcp_cluster/shared_utilities_base.cpp` line 60
```cpp
info.type = PlatformType::WINDOWS; info.name = "Windows";
info.supports_shm_local = true;
info.supports_rdma = true;  // ← wrong
```
Windows RDMA requires ND (Network Direct) or RoCE drivers that are absent on most systems. `supports_rdma = true` for Windows causes the metrics export and dispatch path to attempt RDMA on Windows where it will silently fail and fall through to TCP. Should be `false` by default with explicit opt-in via `VGRE_RDMA_ENABLED`.

### 1.5 Hardcoded 128 MB Shared Memory Result Offset
**File**: `src/advanced/tcp_cluster/dispatch_manager_core.cpp` line 14
```cpp
DispatchManager::DispatchManager(TCPClusterManager* parent) 
    : parent_(parent), result_shm_offset_(128ULL * 1024 * 1024) {
```
The shared memory region always starts result writes at a fixed 128 MB offset. No runtime check that the SHM region is actually that large. Large kernel outputs could overflow undetected.

### 1.6 AllReduce Barrier Timeout: 30s Hardcoded on Worker Side, Configurable on Master Side
**Files**: `collective_ops_manager.cpp` lines 145 and 179,210
```cpp
// Worker waiting for master AllReduce result — hardcoded 30s:
bool success = parent_->reduction_cv_.wait_for(lock, std::chrono::seconds(30), ...);

// Worker waiting for barrier — hardcoded 30s:
auto result = parent_->barrier_cv_.wait_for(lock, std::chrono::seconds(30), ...);
```
The master-side reduction timeout is configurable via `VGRE_REDUCTION_TIMEOUT_MS`. The worker-side timeouts are hardcoded at 30 seconds. If the master is slow or the network has high latency, workers time out prematurely. Both should read from `VGRE_REDUCTION_TIMEOUT_MS`.

### 1.7 Reduction Timeout Heuristic: 2× Max Kernel Latency + 5s
**File**: `collective_ops_manager.cpp` lines 87-93
```cpp
reductionTimeoutMs = static_cast<int>(maxWorkerLatencyMs * 2.0) + 5000;
```
The master's auto-computed timeout is 2× the slowest observed single-kernel latency plus 5 seconds. A slow kernel's latency is unrelated to how long an AllReduce on 100 MB of float32 data takes across 8 workers. Timeouts derived this way will either be too short (fast kernels, large AllReduce) or too long (slow kernels, tiny AllReduce).

### 1.8 Collective AllReduce Missing FP16/BF16/INT8 Types
**File**: `collective_ops_manager.cpp`  
`applyReduce<T>` has template instantiations for `float`, `double`, `int32_t`, `int64_t` (line 327-330). No FP16, BF16, or INT8. NCCL AllReduce for `ncclFloat16` or `ncclBfloat16` (used by PyTorch AMP / mixed-precision training) will silently fall through to the scalar path that operates on `uint8_t` — producing garbage results.

### 1.9 Bandwidth Probe Payload Is All Zeros
**File**: `server_loop_connection_handling.cpp` lines 73-75
```cpp
std::vector<uint8_t> probe_buf(sizeof(uint64_t) + kProbePayloadBytes, 0);
// payload initialized to all zeros
```
A zero-filled 1 MB probe may be accelerated by OS/NIC zero-page optimizations or TCP offload, making the measured bandwidth artificially high. Real bandwidth probes should use pseudo-random data to prevent compression artifacts.

### 1.10 Configuration Manager Split Into 6 Files With Redundant Includes
**Files**: `configuration_manager_core.cpp`, `configuration_manager_file_io.cpp`, `configuration_manager_validation.cpp`, `configuration_manager_monitoring.cpp`, `configuration_manager_backup.cpp`, `configuration_manager_documentation.cpp`  
All 6 include `shared_utilities.h` and re-open the same `ClusterConfig` struct repeatedly. The file_io parser (`configuration_manager_file_io.cpp`) implements its own hand-rolled JSON/YAML/INI parser instead of using the system LLVM JSON library already in the codebase.

### 1.11 `server_loop_auth_mgmt.cpp` Has One Function (26 Lines)
**File**: `server_loop_auth_mgmt.cpp` — 26 lines total, one function `cleanupServerAuthThreads`.  
A single 26-line function does not justify a separate compilation unit. The server loop is split into 4 files: core (68 lines), auth_mgmt (26 lines), connection_handling (254 lines), data_handling (86 lines). The auth_mgmt file exists only because a previous split aimed at sub-500-line files was applied mechanically.

### 1.12 Duplicate Metrics Output: Two Files for the Same Data
**Files**:
- `diagnostic_logger.cpp:414` writes to `/tmp/vgre_tcp_cluster_metrics.json`
- `tcp_cluster_metrics.cpp:62` writes to `vgre_tcp_cluster_metrics.json` (CWD)  

Same JSON schema, different paths. Monitoring tools reading one file will miss data from the other. Neither location is configurable via environment variable.

### 1.13 Discovery Manager `getUdpAnnouncePort()` Ignores Configuration
**File**: `discovery_manager.cpp` lines 42-46
```cpp
int DiscoveryManager::getUdpAnnouncePort() {
    const char* env = vgre_get_config("VGRE_MESH_DISCOVERY_PORT");
    if (env) { try { return std::stoi(env); } catch (...) {} }
    return 7778;
}
```
The function reads `VGRE_MESH_DISCOVERY_PORT` but `configuration_manager_core.cpp` stores the same value as `config.mesh_discovery_port` (loaded separately from `VGRE_MESH_DISCOVERY_PORT`). If the discovery manager and configuration manager are initialized in a different order, they may use different ports.

### 1.14 `tcp_cluster_validation.cpp` Fails on macOS (Big-Endian Check)
**File**: `tcp_cluster_validation.cpp` lines 250, 265
```cpp
VGRE_LOG_ERROR("TCPCluster", "Platform endianness not supported for mesh topology");
VGRE_LOG_ERROR("TCPCluster", "Unknown platform - mesh topology not supported");
```
The endianness check returns `ERR_NOT_SUPPORTED` for big-endian platforms. Current ARM Macs use little-endian, so this is fine for production. But the platform-unknown branch also returns failure — any new platform (e.g., RISC-V) would silently refuse to join a cluster.

---

## Section 2 — Remaining Platform Header Issues

Most platform-specific headers across the codebase ARE correctly guarded inside `#if defined(_WIN32)` / `#elif defined(__linux__)` / `#if defined(__APPLE__)` blocks. No unguarded leaks were found. The following files use POSIX-specific syscalls that are **not in `os_backend.h`** and will fail to compile on Windows or require porting:

| File | Header / Syscall | Status |
|---|---|---|
| `adaptive_execution_engine*.cpp` (3 files) | `<dirent.h>`, `<sys/ioctl.h>`, `<sys/syscall.h>` (Linux); `<IOKit/IOKitLib.h>` (macOS) | Guarded by `#if defined(__linux__)` / `#if defined(__APPLE__)` — **OK** |
| `secure_channel_crypto.cpp` | `<sys/random.h>` getrandom/getentropy | Guarded by `#if !defined(_WIN32)` — **OK** |
| `nccl_communicator.cpp`, `nccl_core.cpp` | `<sys/random.h>` | Guarded — **OK** |
| `vector_engine*.cpp` | `<sys/syscall.h>` SYS_arch_prctl for AMX | Guarded — **OK** |
| `websocket_transport.cpp` | `<sys/select.h>` | Guarded — **OK** |
| `configuration_manager_validation.cpp` | `<sys/stat.h>` | ⚠️ **NOT guarded** — will fail on Windows (no `stat()` as-is) |
| `configuration_manager_file_io.cpp` | `<sys/stat.h>` | ⚠️ **NOT guarded** — same issue |
| `ipc_manager.cpp` | `<sys/stat.h>` | ⚠️ **NOT guarded** — same issue |
| `scheduler_numa.cpp` | `<dirent.h>`, `<sys/sysctl.h>` | ⚠️ Partially guarded but `sysctl` path for NUMA on Linux uses raw syscall |

**Action required**: Wrap all `<sys/stat.h>` uses in `#if !defined(_WIN32)` and add `_stat` / `GetFileAttributesEx` equivalents, or route through `os_backend.h::file_exists()` / `file_mtime()` which already provide the cross-platform API.

---

## Section 3 — Scripts: Issues Found

### 3.1 `run_benchmarks.sh` Silently Skips VGRE If Bindings Missing
**File**: `scripts/run_benchmarks.sh` + `tests/python/benchmark.py` lines 6-16
```python
try:
    from vgre import VirtualDevice, Runtime
    VGRE_AVAILABLE = True
except ImportError:
    VGRE_AVAILABLE = False  # silently skips all VGRE benchmarks
```
`run_benchmarks.sh` exits 0 (success) even when VGRE bindings are not importable. A CI pipeline running `./scripts/run_benchmarks.sh` will pass even if the Python bindings are broken. Fix: exit non-zero when `VGRE_AVAILABLE = False`.

### 3.2 `test_pytorch.py` and `test_tensorflow.py` Not Integrated into CTest
**Files**: `tests/python/test_pytorch.py`, `tests/python/test_tensorflow.py`  
These Python tests exist but are not registered in `tests/CMakeLists.txt`. They never run under `ctest`. Coverage of the Python binding layer is absent from the 117-test suite.

### 3.3 `test_python_authoritative.py` Has a Dummy Kernel Comment
**File**: `tests/python/test_python_authoritative.py` line 62
```python
# Launch a dummy kernel if possible, or just check JSON
```
The test falls back to "just check JSON" when the kernel launch fails, masking launch failures as passing tests.

### 3.4 `vgre_sync.sh` Does Not Verify LLVM Version
**File**: `scripts/vgre_sync.sh`  
The build script installs LLVM if not present but does not verify the version. VGRE requires LLVM 18 specifically (`llvm-18`, `clang-18`). Installing any LLVM (e.g., system `llvm-17`) would result in a silent miscompile.

### 3.5 `setup-cluster.sh` Example IP Is a Documentation Stub
**File**: `scripts/setup-cluster.sh` line 16
```bash
#   vgre-start --worker --master-ip 192.168.1.50   (different subnet)
```
No validation that the master IP is reachable before starting worker. A typo causes a silent connection failure with no diagnostic output.

---

## Section 4 — Codebase-Wide: Confirmed Real Implementations

The following were audited and confirmed to have **real computation, no stubs**:

| Component | Verified |
|---|---|
| cuBLAS L1/L2/L3 | ✅ Cache-blocked GEMM, CBLAS delegation |
| cuFFT | ✅ Cooley-Tukey + Bluestein |
| cuDNN Conv/BN/Act/Pool/Softmax/Dropout | ✅ Real CPU loops, OpenMP |
| cuDNN RNN (vanilla tanh only — see Section 5) | ⚠️ Real, but LSTM/GRU wrong |
| cuDNN MHA | ✅ Real QKV + softmax |
| cuDNN CTC Loss | ✅ Real forward-backward CTC |
| cuRAND (host + device) | ✅ XORWOW, Philox, Sobol |
| cuSPARSE SpMV/SpMM | ✅ Real CSR |
| cuSolver LU/QR/SVD | ✅ LAPACK-backed |
| NCCL AllReduce | ✅ 3 algorithms, real reduce |
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
| `cudnnRNNForwardInference` with `CUDNN_LSTM` | Cell state ignored; no forget/input/output gates | Any LSTM network |
| `cudnnRNNForwardInference` with `CUDNN_GRU` | No reset/update gates | Any GRU network |
| `ncclAllReduce` with `ncclFloat16`/`ncclBfloat16` | Falls through to uint8 scalar path (wrong accumulation) | PyTorch AMP training |
| `cuOccupancyMaxActiveBlocksPerMultiprocessor` | Always uses 2048 threads/SM (Ampere-specific) | Auto-tuned launch configs |
| Bandwidth utilization display | Always shows ≤1% on >1 Gbps NICs | Dashboard monitoring |

---

## Section 6 — Confirmed Stubs (Accept Calls, Return Success, Do Nothing Real)

| API / Feature | File | Gap |
|---|---|---|
| PTX multi-module linker (`cuLink*`) | `cuda_driver_module.cpp:189-273` | Concatenates PTX, no symbol resolution |
| cuDNN Backend v8 execution | `cudnn_backend_api.cpp:744` | Returns NOT_SUPPORTED |
| cuSPARSE SpGEMM | `cusparse_factorization.cpp:497,561` | Returns NOT_SUPPORTED without UMFPACK |
| cuSolver batched APIs | Not implemented | No code exists |
| cuRAND MTGP32 device-side | `curand_kernel.h` | Only XORWOW + Philox present |
| Token manager: macOS Keychain | `token_manager_macos.cpp:102` | Always ERR_NOT_SUPPORTED |
| Token manager: Linux libsecret | `token_manager_linux.cpp:204,218` | Always ERR_NOT_SUPPORTED |
| Token manager: Windows DPAPI | `token_manager_win32.cpp:70` | Always ERR_NOT_SUPPORTED |
| cuDNN RNN backward (BPTT) | `cudnn_rnn.cpp` | Aliases forward inference |
| VFIO GPU passthrough | `gpu_passthrough.cpp` | Detected, not activated |

---

## Section 7 — Not Implemented (No Code Exists)

| Missing | Impact |
|---|---|
| LSTM / GRU cell gates | Silent wrong results for all RNN training |
| cuDNN Backend v8 graph execution | PyTorch ≥ 2.0 ops broken |
| cuSPARSE SpGEMM | Graph neural networks (DGL, PyG) broken |
| cuSolver batched (Potrf, Getrf) | Transformer attention batched solves broken |
| SASS execution | Pre-compiled CUDA libraries unusable |
| PTX cross-module linking | Separate compilation workflows broken |
| CUDA TMA instructions | Hopper TMA kernels fail to JIT-compile |
| FP16/BF16 AllReduce | Mixed-precision training AllReduce gives garbage |
| Multi-GPU P2P | `cudaMemcpyPeer` uses slow host staging |
| Hardware performance counters | No CUPTI/profiling capability |
| MPS | Single process per virtual device |
| cuFFT CUDA_C_16BF | Bfloat16 complex FFT absent |
| Configurable link bandwidth for utilization | Monitoring display wrong on fast NICs |
