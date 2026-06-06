# VGRE Missing Features, Stubs, Mocks, Security and Cross-Platform Issues

This document catalogues all current gaps, partial implementations, stubs, mocks, fallbacks, security vulnerabilities, and cross-platform compatibility issues within the VGRE (Virtual GPU Runtime Engine) codebase.

---

## 1. Code-Level Stubs, Mocks, and Placeholders

These items represent empty definitions, dummy values, or test-only mocks that lack standard runtime logic.

### 1.1 gRPC Transport Stubs
* **Location**: `src/advanced/grpc_transport.cpp`, `include/vgre/advanced/grpc_transport.h`
* **Details**: When `-DVGRE_ENABLE_GRPC=ON` is not specified, all gRPC connection and transport methods (e.g. `VGREGRPCClient`) are compiled as empty stubs returning error codes.
* **Code artifacts**: `void* stubPtr_ = nullptr;` opaque wrapper around `VGRECluster::Stub`.

### 1.2 CUDA Linker (cuLink API) Stubs
* **Location**: `src/api/cuda_driver/cuda_driver_module.cpp`
* **Details**: The linker APIs `cuLinkCreate`, `cuLinkAddData`, and `cuLinkComplete` are stubbed. VGRE does not perform actual Device Link-Time Optimization (LTO) across compilation units; it only collects the PTX blobs in-memory.
* **Impact**: Multi-Translation Unit (TU) device programs relying on cross-module `extern __device__` linkage will fail.

### 1.3 CUB Fallback Caching Device Allocator ✅ Fixed
* **Location**: `include/vgre/compiler/cuda_device_libs/cub_fallback.h`
* **Status**: Replaced with a real thread-safe pool allocator using per-size free lists with 64-byte alignment (cache-line). Freed blocks are cached up to 16 entries per size class, avoiding repeated malloc/free for repeated CUB allocation patterns.

### 1.4 Clang Kernel Parser AST Parsing Stubs
* **Location**: `src/compiler/clang_kernel_parser.cpp`
* **Details**: Employs multiple stubs to bypass compiler parsing overhead and memory usage during AST analysis:
  - `kAstAnalysisStub`: Minimal CUDA stub replacing the heavy `cuda_fp16.h` and driver headers.
  - Cooperative groups stubs for AST analysis.
  - `partition_copy`, `inclusive_scan`, and `exclusive_scan` stubs that compile but redirect to sequential copies.

### 1.5 CDP (CUDA Dynamic Parallelism) Device-Side Stubs
* **Location**: `src/runtime/cdp_executor.cpp`
* **Details**: The C interface calls from device-side CDP stubs ignore the `kernelFn` function pointer returned by `__device_stub__xxx`.

### 1.6 LAPACK Complex Operations ✅ Fixed
* **Location**: `src/api/cusolver/lapack_fallback.cpp`
* **Status**: `cgesvd_`/`zgesvd_` replaced with correct complex SVD via augmented real form (Golub-Van Loan §2.6.3): builds real (2M)×(2N) block matrix B=[Re(A),-Im(A);Im(A),Re(A)], computes real SVD, extracts σ_k from paired singular values and reconstructs complex U/VT. `cheevd_`/`zheevd_` similarly uses a real (2N)×(2N) augmented Hermitian matrix. `ssygvd_`/`dsygvd_` (generalized eigenvalue) implemented via Cholesky reduction.

### 1.7 Mock Interfaces for Testing
* **Location**: `include/vgre/advanced/tcp_cluster/internal/interfaces.h`
* **Details**: Defines mocks and fakes for cluster manager and memory synchronization interfaces to simulate environment behaviors in unit tests.

### 1.8 Dummy Workspace Queries
* **Location**: `src/api/cusolver/cusolver_core.cpp`
* **Details**: Employs dummy vectors and matrices (e.g., `dummy_a`, `dummy_b`, `dummy_s`) to mock parameter sizes when calling LAPACK workspace size query functions (`sgelsd_`, `dgelsd_`, `sgesvd_`, `dgesvd_`).

---

## 2. Partial Implementations and Fallback Gaps

These features contain functional implementations but have critical missing components or rely on fallback logic.

### 2.1 SASS Binary Decoder — Tensor Core Opcodes
* **Location**: `src/compiler/sass/sass_decoder.cpp`
* **Details**: HMMA (`.884`, `.1688`) and WGMMA (SM90) opcodes decode to comment-only PTX lines (`// HMMA_884_F32: ...`) instead of emitting valid `wmma.mma.sync` intrinsics.
* **Impact**: Cubins relying exclusively on SASS tensor core instructions will run but fail to compute correct numerical outputs.

### 2.2 GPUDirect RDMA QP Handshake Gap
* **Location**: `src/advanced/rdma_transport.cpp`
* **Details**: The functions `sendQPInfo()` and `recvQPInfo()` return `false` immediately.
* **Impact**: Peers cannot serialize and exchange Queue Pair parameters (LID, QPN, PSN, rkey, remote bounce-buffer address) over `SecureChannel`. Cross-node transfers fallback to AVX-accelerated `streamingMemcpy` and LZ4 over TCP.

### 2.3 cuDNN Graph API Fusions and RNG Backend ✅ Fixed
* **Location**: `src/api/cudnn/cudnn_graph.cpp`, `src/api/cudnn/cudnn_backend_api.cpp`
* **Status**: CONV_NORM fused execution implemented in `cudnn_executeFusedConvBN()` with two paths: inference folds BN into Conv weights (W_f=W·γ/σ, b_f=β-γμ/σ), training uses two-pass Conv→mean/var→normalize. RNG backend implements Philox 4×32 (10 rounds) with uniform, normal (Box-Muller), and Bernoulli distributions.

### 2.4 iGPU OpenCL Transpiler Gaps
* **Location**: `src/runtime/igpu_opencl_executor.cpp`
* **Details**: Translates CUDA source to OpenCL C via regex-based substitutions.
* **Gaps**: Does not support texture operations, dynamic shared memory, cooperative groups, or inline assembly (`asm volatile`). Complex CUDA kernels fallback to CPU execution.

### 2.5 CUPTI PMU Counter Heuristics
* **Location**: `src/api/cupti/cupti_shim.cpp`
* **Details**: Since host CPUs lack GPU hardware performance monitors, metrics like `achieved_occupancy` and `l1_global_load_hit_rate` are calculated using proxy instructions-mix ratio heuristics.

---

## 3. Hardcoded Limits and API Restrictions

### 3.1 Hardcoded Attributes and API Limits — Partially Fixed
* **Location**: `src/api/cudart/cudart_shim_device_attrs.cpp`
* **Status**:
  - `mallocHeapSize`: dynamically derived from host RAM via `sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE) / 16`, clamped to [8MB, 512MB]. ✅ Fixed
  - `ptxVersion`/`binaryVersion`: reads `VGRE_COMPUTE_CAPABILITY` env var (format "major.minor" or "majorminor"), defaults to SM 8.6. ✅ Fixed
  - Stack size (8KB) and Printf FIFO (1MB) remain hardcoded — appropriate defaults for a CPU emulator.

### 3.2 NVSCI Sync Objects Unimplemented
* **Location**: `src/api/cuda_external_semaphore.cpp`
* **Details**: NVSCI sync objects are explicitly marked as "not supported", which prevents standard inter-process coordination schemas.

---

## 4. Security Vulnerabilities and Gaps

These issues represent gaps in encryption, access control, credential management, and input validation that pose security risks.

### 4.1 Plaintext Telemetry & Workload Fallbacks ✅ Fixed
* **Location**: `src/advanced/tcp_cluster/packet_handler.cpp`
* **Status**: Both `sendPacket()` and `sendPacketDirect()` now reject non-handshake packets with `ERR_AUTH_FAILED` when the SecureChannel is provided but not yet initialized. Only `SECURE_HANDSHAKE` and `SECURE_HANDSHAKE_ACK` packet types are permitted in plaintext during the bootstrap phase.

### 4.2 Insecure Cryptographic Salt in Token Fallback
* **Location**: `src/advanced/token/token_manager_fallback.cpp`
* **Vulnerability**: The encrypted token storage (`FALLBACK_ENCRYPTED`) uses a hardcoded identity seed `"vgre_fallback_v2:" + identity` and `"vgre_fallback_kdf"`.
* **Impact**: Compromise of a host machine allows attackers to easily reconstruct the KDF parameters and decrypt stored cluster tokens.

### 4.3 Process Leakage via Environment Credentials
* **Location**: `src/advanced/token/token_manager_fallback.cpp`, `src/advanced/tcp_cluster/configuration_manager_file_io.cpp`
* **Vulnerability**: The cluster relies on environment variables (`VGRE_TCP_AUTH_TOKEN`, `VGRE_TOKEN_FALLBACK_PATH`) for cluster credentials.
* **Impact**: Credentials may leak via process monitoring utilities (`ps`), shell command history logs, or system configuration exports.

### 4.4 Unsanitized PTX Assembly Injection (JIT RCE) ✅ Fixed
* **Location**: `src/compiler/llvm_translation_engine.cpp`
* **Status**: `validateKernelSource()` static function added before `doTranslate()`. Rejects kernels containing inline-asm syscall opcodes (`syscall`, `sysenter`, `int 0x80`) and dangerous OS/network function calls (`system`, `popen`, `execve`, `socket`, `dlopen`, etc.) via regex scan. Returns `ERR_INVALID_VALUE` with a descriptive log message.

### 4.5 Insecure IPC/SHM Permissions ✅ Fixed
* **Location**: `src/core/shm_manager.cpp`
* **Status**: POSIX `shm_open()` permission changed from `0666` to `0600` (owner read/write only). Eliminates local-user read/write access to shared GPU memory segments.

---

## 5. Cross-Platform Compatibility Issues

These issues prevent standard, consistent compilation and execution across Windows, macOS, and Linux.

### 5.1 Windows vs. POSIX Socket Discrepancies
* **Location**: `include/vgre/common/sockets.h`, `src/advanced/tcp_cluster/windows_socket_manager_lifecycle.cpp`
* **Details**: 
  - Socket initialization (`WSAStartup`/`WSACleanup`) and error fetching (`WSAGetLastError()`) are Windows-only.
  - POSIX functions (`poll`, `socket`, `ioctl`) map differently on Windows (e.g. using `WSAPoll`, `closesocket`, `ioctlsocket`).
  - `WSAPoll` behaves differently than POSIX `poll` under connection failures, leading to infinite blocking states.

### 5.2 CPU Thread Affinity Differences
* **Location**: `src/core/scheduler_numa.cpp`, `src/core/scheduler_worker.cpp`
* **Details**:
  - Linux uses `pthread_setaffinity_np` with `cpu_set_t`.
  - macOS lacks POSIX affinity sets and uses soft affinity hints (`THREAD_AFFINITY_POLICY`).
  - Windows uses `SetThreadAffinityMask` which only supports 64 logical processors unless processor groups are managed.

### 5.3 NUMA Memory Allocation Discrepancies
* **Location**: `src/core/memory/memory_manager.cpp`, `src/core/memory/uvm_migration.cpp`
* **Details**:
  - Linux-specific page migration is implemented via `SYS_mbind`.
  - Windows requires `VirtualAllocExNuma` to achieve NUMA-aware allocation, which is currently stubbed with standard allocations.
  - macOS lacks a native NUMA API exposed to user space.

### 5.4 IPC Shared Memory Lifecycle Gaps
* **Location**: `src/core/shm_manager.cpp`
* **Details**:
  - POSIX shared memory uses `shm_open` and `shm_unlink` on Unix.
  - Windows requires Win32 file mappings (`CreateFileMapping`, `OpenFileMapping`, `MapViewOfFile`) which have completely different lifecycles and global/local naming scopes (`Global\vgre_...`).

### 5.5 Compiler Barriers and Inline Assembly
* **Location**: `src/advanced/rdma_transport.cpp`, `src/runtime/block_worker_pool.cpp`
* **Details**:
  - Employs GCC/Clang built-ins like `__builtin_ia32_pause()` and `asm volatile("yield" ::: "memory")`.
  - On MSVC (Windows), these cause compilation errors. MSVC requires compiler intrinsics (`_mm_pause()`, `_ReadWriteBarrier()`). MSVC x64 does not support inline assembly.

### 5.6 Dynamic Library (dlopen vs LoadLibrary) Lifecycle
* **Location**: `src/api/cupti/cupti_shim.cpp`, `src/advanced/gpu_passthrough.cpp`
* **Details**:
  - Uses `dlopen`/`dlsym`/`dlclose` on Linux/macOS.
  - Requires `LoadLibrary`/`GetProcAddress`/`FreeLibrary` on Windows, along with different naming conventions (`.so`/`.dylib` vs `.dll`).

### 5.7 Artificial Delays and Sleep Latency
* **Location**: Multiple scheduler and transport files
* **Details**: Hardcoded `sleep_for`, `sleep_until`, and yield statements introduce platform-dependent scheduling delays and high CPU cycles during polling.
