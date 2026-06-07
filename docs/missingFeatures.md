# VGRE — Open Gaps, Enhancements, and Innovation Targets

**Last Updated**: 2026-06-07

This document is the authoritative, forward-looking ledger of everything VGRE still needs: genuine implementation gaps, API boundaries, security holes, and creative capabilities that would make this a fully production-grade GPU emulation platform.

All items marked ✅ Fixed in previous versions have been removed. Only unresolved and future-looking items remain.

---

## 1. Remaining Implementation Gaps

### 1.1 gRPC Transport — Stub When Not Enabled
- **Location**: `src/advanced/grpc_transport.cpp`, `include/vgre/advanced/grpc_transport.h`
- **Details**: When `-DVGRE_ENABLE_GRPC` is not specified at build time, all gRPC client methods return error codes from empty stubs. There is no runtime fallback that gracefully degrades to TCP. An application that explicitly requests gRPC transport (`VGRE_GRPC_PORT`) will silently fail.
- **Fix needed**: `VGREGRPCClient::connect()` should detect the stub state and return a clear `ERR_NOT_BUILT` diagnostic with a hint to rebuild with `-DVGRE_ENABLE_GRPC=ON`, rather than silently returning success or an opaque error.

### 1.2 cuLink — PTX-Only; FATBIN and CUBIN Blobs Unsupported
- **Location**: `src/api/cuda_driver/cuda_driver_module.cpp`
- **Details**: `cuLinkComplete` merges multi-TU PTX correctly (strips duplicate headers, resolves `extern __device__` functions and global variables, concatenates module bodies). However, `cuLinkAddData` silently ignores `CU_JIT_INPUT_FATBINARY` and `CU_JIT_INPUT_OBJECT` blobs — only `CU_JIT_INPUT_PTX` blobs are processed.
- **Impact**: Programs that pass pre-compiled FATBIN or NVCC object files to `cuLinkAddData` will get an empty output from `cuLinkComplete` instead of a linked module.
- **Fix needed**: Detect non-PTX input types and return `CUDA_ERROR_INVALID_VALUE` with a diagnostic, or implement FATBIN unwrapping to extract the embedded PTX section.

### 1.3 Clang Kernel Parser — AST Analysis Stubs
- **Location**: `src/compiler/clang_kernel_parser.cpp`
- **Details**:
  - `kAstAnalysisStub` replaces actual CUDA system headers during AST analysis. Deep intrinsics or type-dependent template expansions that require the real `cuda_fp16.h` definitions will produce incorrect occupancy estimates.
  - `partition_copy`, `inclusive_scan`, and `exclusive_scan` stubs compile and run but redirect to sequential copies — correct result, wrong performance model (occupancy/register estimates will under-count SIMD work).
  - Cooperative groups stubs mean any kernel that uses `cg::thread_block::sync()` at AST analysis time is analyzed without the real barrier cost.

### 1.4 CDP Device-Side — Function Pointer Registration Race
- **Location**: `src/runtime/cdp_executor.cpp:150`, `src/core/runtime_engine.cpp:530`
- **Details**: `vgre_cdp_launch_device` looks up the kernel ID by the raw function pointer via `lookupKernelIdByFn`. This works only if the kernel was registered by `__device_stub__xxx` before the child launch is attempted. If the host program calls `cuLaunchKernel` by name (not pointer), or if the host registers the kernel lazily after the parent kernel has already started, `kid == 0` and the child launch is silently dropped.
- **Fix needed**: Fall back to kernel-name-based lookup when address lookup returns 0. The parent kernel's PTX module should provide a symbol table that maps stub addresses to kernel names.

### 1.5 NVSCI Sync Objects
- **Location**: `src/api/cuda_external_semaphore.cpp:119–120`
- **Details**: `CUDA_EXTERNAL_SEM_NVSCISYNCOBJ` type is explicitly skipped with "not supported". NvSci is the NVIDIA chip-to-chip interconnect sync primitive used on Jetson platforms and in DLA+GPU pipelines.
- **Fix needed**: Implement via a POSIX eventfd (Linux) or `CreateEvent` (Windows) as a CPU-side proxy. The NvSci semantics (wait-before-signal ordering, fence export) can be emulated with a `std::promise`/`std::future` pair without needing actual NvSci hardware.

### 1.6 cuDNN Backend API — Incomplete Operation Coverage
- **Location**: `src/api/cudnn/cudnn_backend_api.cpp:1959`
- **Details**: Several backend operations fall through to `CUDNN_STATUS_NOT_SUPPORTED`. The current comment in the file acknowledges this. Affected operations include some pointwise op types beyond ReLU/Sigmoid/Tanh and some reduction variants in the v9 graph API.
- **Fix needed**: Audit every `CUDNN_STATUS_NOT_SUPPORTED` return in `cudnn_backend_api.cpp` and implement the missing pointwise and reduction descriptor execution paths.

---

## 2. Permanent Hardware Boundaries

These are CPU architectural constraints — not software bugs. They cannot be "fixed" but must be clearly communicated to users.

### 2.1 CUPTI Hardware Performance Counters
- **Location**: `src/api/cupti/cupti_shim.cpp`
- **Boundary**: GPU-specific PMU counters (`achieved_occupancy`, `l1_global_load_hit_rate`, `sm_active_cycles`) are computed via instruction-mix heuristics because host CPUs have no GPU warp schedulers or L1/L2 GPU cache hardware.
- **Enhancement path**: Integrate with Linux `perf_event_open` to provide real L1/L2 CPU cache hit rates and IPC as proxy metrics. Document the proxy-to-GPU mapping so users can interpret them correctly.

### 2.2 iGPU OpenCL Transpiler — Unsupported CUDA Features
- **Location**: `src/runtime/igpu_opencl_executor.cpp`
- **Boundary**: The regex-based CUDA→OpenCL C transpiler strips inline `asm volatile`, replaces cooperative groups includes with no-ops, and does not translate:
  - `tex1D`/`tex2D`/`tex3D` texture instructions
  - Dynamic shared memory (`extern __shared__ float s[]`)
  - Warp-level intrinsics (`__shfl_sync`, `__ballot_sync`)
  - `printf` from device kernels
  Kernels using these features fall back to CPU execution — correct result, but the iGPU path is bypassed.
- **Enhancement path**: Extend transpiler to emit `read_imagef`/`write_imagef` for texture calls, emit `local float s[N]` for static shared memory, and warn on dynamic shared memory.

### 2.3 Device-Side cuRAND
- **Details**: Device-side cuRAND (`curand_uniform`, `curand_normal` called from within a kernel) is not emulated in the JIT path. Only host-side cuRAND generator APIs are implemented.
- **Boundary**: True device-side cuRAND requires per-thread PRNG state living in registers, which the JIT compiler would need to inject into the translated kernel code. This is an architectural limitation of the current JIT model.
- **Enhancement path**: Inject XORWOW per-thread state via a hidden kernel argument in the JIT wrapper. The PTX translator can recognize `curand_uniform`/`curand_normal` calls and replace them with the CPU-side XORWOW inline expansion.

---

## 3. Security — Remaining Gaps

### 3.1 Token Fallback — Hardcoded KDF String
- **Location**: `src/advanced/token/token_manager_fallback.cpp:296`
- **Vulnerability**: One code path (the legacy key derivation branch) still uses the literal string `"vgre_fallback_kdf"` (17 bytes) as a static PBKDF2 password. Even though the primary path builds a machine-identity string, the legacy branch is reachable on systems where DMI/UUID/CPU detection fails.
- **Fix needed**: Replace the hardcoded literal with `getrandom()`/`/dev/urandom`-derived salt stored alongside the token file. Regenerate existing fallback tokens on first use.

### 3.2 Environment Credential Leakage (`VGRE_TCP_AUTH_TOKEN`)
- **Location**: `src/advanced/token/token_manager_fallback.cpp`, `src/advanced/tcp_cluster/configuration_manager_file_io.cpp`
- **Vulnerability**: When `VGRE_TCP_AUTH_TOKEN` is set as an environment variable, the raw token value is visible in `/proc/PID/environ` and in `ps e` output on Linux. This is a credential exposure risk on shared machines.
- **Fix needed**: After reading the token from the environment, immediately overwrite the environment variable entry with zeros (`memset(getenv("VGRE_TCP_AUTH_TOKEN"), 0, ...)` plus `unsetenv`). Document that the preferred path is `VGRE_TCP_AUTH_TOKEN_FILE`.

### 3.3 WSAPoll Failure Detection (Windows)
- **Location**: `src/advanced/tcp_cluster/client_loop.cpp:427`
- **Details**: WSAPoll does not reliably report `POLLHUP`/`POLLERR` on Windows when a remote peer resets a TCP connection. The comment in the file acknowledges the `WSAEINVAL` behavior. Under rapid worker reconnection cycles, a half-open TCP socket may not be detected as dead for the full keepalive timeout (~2 minutes without explicit keepalive tuning).
- **Fix needed**: On Windows, after a `WSAPoll` timeout with no events, issue a zero-byte `send()` as a TCP liveness probe. A `WSAECONNRESET` or `WSAECONNABORTED` response identifies a dead connection without waiting for keepalive expiry.

---

## 4. Missing Innovation — What This Engine Needs Next

These are capabilities that are absent from the current codebase and are required for VGRE to be a complete, production-grade GPU emulation platform covering modern AI/ML and HPC workloads.

### 4.1 Multiple Virtual GPU Devices
- **Gap**: VGRE exposes exactly one virtual GPU device per process. `cudaGetDeviceCount` always returns 1. Multi-GPU training frameworks (PyTorch DDP, Horovod) require at least 2 devices per node to run without modification.
- **Design**: Partition the host CPU's cores into N virtual device pools (each with its own `BlockWorkerPool`, `MemoryManager`, `Scheduler`). Map `cudaSetDevice(i)` to pool `i`. Intra-node P2P copies between virtual devices use `memcpy` + shared-memory zero-copy.
- **Target**: Support 2–8 virtual devices per process, configurable via `VGRE_VIRTUAL_DEVICE_COUNT=N`.

### 4.2 Flash Attention GQA (Grouped Query Attention)
- **Gap**: The kernel fusion engine detects standard multi-head attention (equal Q/K/V head counts). Grouped Query Attention (GQA) — used by LLaMA-2, Mistral, Gemma, and Falcon — has fewer K/V heads than Q heads (`num_kv_heads < num_heads`). The current detector fails to recognize the GQA pattern and falls through to the unfused path.
- **Design**: Extend `KernelFusionEngine::detectFlashAttention` to detect the `kv_head_repeat` stride pattern. Generate fused GQA source in `genFlashAttentionSource` that tiles K/V heads across Q groups.
- **Impact**: LLaMA-2-7B inference on CPU — currently 3× slower than optimal due to unfused attention.

### 4.3 NVTX → Perfetto Trace Export
- **Gap**: VGRE exports profiling data only as Chrome JSON traces (`toChromeTraceJSON`). Perfetto (used by Android, Google, and NVIDIA Nsight Systems) has become the dominant tracing standard and offers superior flamegraph, counter, and flow-event support.
- **Design**: Implement `toPerfettoProto()` that serializes the VGRE timeline into Perfetto's `TracePacket` protobuf format. Write to a `.perfetto-trace` file. This enables direct loading into `ui.perfetto.dev` and Nsight Systems analysis workflows.

### 4.4 CUDA Dynamic Graph Optimization (Node Fusion Across Replays)
- **Gap**: CUDA Graphs are captured and replayed identically on every `cudaGraphLaunch`. There is no inter-replay learning: if two consecutive memory-copy nodes feed a single kernel node on every replay, they are never fused into a single DMA+compute node.
- **Design**: After N graph replays (configurable via `VGRE_GRAPH_FUSION_THRESHOLD`), analyze the replay trace for patterns of consecutive compatible nodes. Emit a fused `CompiledGraph` that merges them. This mirrors the "graph optimization" step in CUDA's own runtime.
- **Impact**: Models with fixed-topology graphs (BERT inference, ResNet) see 15–25% fewer kernel launches per forward pass.

### 4.5 cuSPARSE ELLPACK and Blocked-ELL Formats
- **Gap**: VGRE supports CSR, COO, CSC, and BSR sparse formats. ELLPACK and Blocked-ELL (used heavily by sparse attention in transformers and graph neural networks) are absent. `cusparseCreateEll` returns `CUSPARSE_STATUS_NOT_SUPPORTED`.
- **Design**: Implement `SpMV_ELLPACK` using AVX2/AVX-512 row-strided vector loads (all rows padded to the same number of non-zeros, enabling cache-line-aligned SIMD access). Wire into `cusparseSpMV` dispatch.

### 4.6 cuDNN Sparse Attention (Transformer Engine v9 Graph)
- **Gap**: The cuDNN v9 backend graph supports `CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR` for dense attention, but sparse attention patterns (local window attention, BigBird, Longformer block-sparse patterns) are unimplemented.
- **Design**: Detect `SDPA_STATS` descriptor variant with `attnMaskType == CUDNN_ATTN_MASK_TYPE_CAUSAL` + explicit mask tensor. Route to a block-sparse attention kernel using SELLPACK storage for the attention mask.

### 4.7 Tensor Parallel Collective Emulation
- **Gap**: Large language models (GPT-4, LLaMA-70B) use tensor parallelism: a single weight matrix is sharded across devices, and an `AllReduce` is used after each matmul. VGRE's NCCL supports `AllReduce` over the TCP cluster, but there is no intra-process tensor-parallel path for multi-virtual-device setups.
- **Design**: Once multiple virtual devices are implemented (4.1), add an intra-process `AllReduce` fast path that uses `memcpy` + atomic reduction instead of TCP. Configure via `VGRE_TP_DEGREE=N`. This enables HuggingFace `device_map="auto"` to work with the VGRE multi-device backend.

### 4.8 PTX → LLVM Bitcode Direct Path (Triton Compatibility)
- **Gap**: PyTorch 2.0+ uses the Triton compiler to generate kernels. Triton emits LLVM bitcode (or PTX via NVVM), which may contain Triton-specific `nvvm.annotations` metadata and non-standard calling conventions. The current PTX translator may misparse or skip these annotations.
- **Design**: Add a `LLVM_IR` input path to `cuModuleLoadData`: detect LLVM bitcode magic (`BC` header), pass directly to LLVM's `ParseBitcodeFile`, and link into the ORC JIT session without going through the PTX text-parsing stage. This eliminates the PTX round-trip and handles any LLVM IR producer (Triton, MLIR, JAX).

### 4.9 Kernel Auto-Vectorization Hints
- **Gap**: VGRE compiles translated kernels with `-O3 -march=native`, but there is no mechanism to inject LLVM loop vectorization pragmas that match the kernel's known access pattern. A kernel that the developer knows is purely element-wise (e.g., cuBLAS `scal`) gets the same generic `-O3` treatment as a complex reduction.
- **Design**: After kernel IR analysis (register count, shared memory, access pattern), annotate the LLVM IR with `llvm.loop.vectorize.width` and `llvm.loop.interleave.count` metadata based on the detected access pattern. Reduction kernels get wider interleave; element-wise kernels get maximum vectorize.width for the detected SIMD level.

### 4.10 MPS Per-Client Resource Quotas
- **Gap**: VGRE MPS accepts arbitrary client connections but imposes no per-client resource limits. A runaway client process can allocate the entire host RAM or saturate all CPU cores, starving other MPS clients.
- **Design**: Implement a `VgreMPSPolicy` struct (configurable via `VGRE_MPS_POLICY_FILE`) with per-client limits: `maxMemoryBytes`, `maxThreadFraction`, `maxQueueDepth`. The MPS daemon enforces these in `handleMalloc` and `handleLaunch` before dispatching to the runtime.

### 4.11 Real-Time Profiler Integration (Nsight-Compatible Traces)
- **Gap**: VGRE produces Chrome JSON and OpenTelemetry spans, but these cannot be loaded into NVIDIA Nsight Systems or Nsight Compute — the primary tools developers use to understand GPU workloads.
- **Design**: Implement a subset of the NVTX3 SDK interface that serializes to Nsight's `.nsys-rep` SQLite format. Even a read-only viewer integration (export + convert) would let developers use familiar tooling on VGRE traces.

### 4.12 Automatic Kernel Recompilation for Host Architecture Changes
- **Gap**: The JIT disk cache (`~/.vgre/cache/`) keys compilations on source hash + compiler flags, but not on the host CPU's microarchitecture. If a user copies a cache from a machine with AVX-512 to one with only AVX2, the cached binaries silently execute AVX-512 instructions on the AVX2 host, causing `SIGILL`.
- **Design**: Include the CPUID feature flags (`avx2`, `avx512f`, `amx_int8`, etc.) as part of the cache key. On cache miss due to arch mismatch, recompile and overwrite. Print a `[VGRE-INFO]` message explaining the recompilation.

### 4.13 Cluster Work Stealing Between Nodes
- **Gap**: The TCP cluster uses 3D recursive bisection to partition kernels across nodes at launch time. If one node is slower (thermal throttling, NUMA miss), the partition is fixed — there is no mechanism to steal remaining work from the slow node's queue.
- **Design**: Implement a simple work-stealing protocol: if a worker finishes its slice more than `VGRE_STEAL_THRESHOLD_MS` before the expected completion time, it queries the master for remaining work. The master splits any in-flight slice and redirects the tail to the idle worker.

### 4.14 Signed JIT Binary Cache
- **Gap**: The disk cache at `~/.vgre/cache/` stores compiled shared object binaries verified by SHA-256 checksum of the source. An attacker with write access to `~/.vgre/cache/` can substitute a malicious `.so` that passes the source checksum (checksum is of the source, not the binary).
- **Fix needed**: Sign cached binaries with an HMAC-SHA256 keyed on the machine's cluster auth token. Verify the signature before `dlopen`-ing any cached binary. Reject and recompile on HMAC failure.

---

## 5. API Coverage Gaps

### 5.1 CUDA Virtual Memory Management
- **Gap**: `cuMemCreate` / `cuMemMap` / `cuMemAddressReserve` / `cuMemSetAccess` — the virtual memory API introduced in CUDA 10.2 — is implemented for the happy path but `cuMemRetainAllocationHandle` (IPC for virtual allocations) is a stub.

### 5.2 cuSPARSE `cusparseSpGEMM_reuse`
- **Gap**: `cusparseSpGEMM` (one-shot) is implemented. The `_reuse` variant — which amortizes symbolic analysis across multiple numerical factorizations of the same sparsity pattern — is not. Used heavily by graph neural network frameworks.

### 5.3 `cudaStreamBeginCapture` with `cudaStreamCaptureModeThreadLocal`
- **Gap**: `cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)` works. The `ThreadLocal` capture mode (only the calling thread's API calls are captured) is not enforced — all threads' API calls are captured regardless of mode.

### 5.4 `cuMemcpyAsync` with `CU_MEM_OPERATION_TYPE_MEMSET`
- **Gap**: Async memset dispatched through the driver API `cuMemcpyAsync` with the operation-type overload is not routed to the scheduler — it falls back to synchronous `memset`.
