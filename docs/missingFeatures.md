# VGRE Missing & Partially Implemented Features

This document provides a rigorous checklist of **current gaps and partial implementations** in VGRE (Virtual GPU Runtime Engine). Every item was verified against the actual source files on 2026-06-06.

Items that appear complete in documentation but have code-level limitations are noted under **Partial Implementation**.

**Last Updated**: 2026-06-06 (Deep Code-Verified Audit)

---

## 1. Hardware-Level Architectural Limitations (Permanent Boundary Conditions)

These represent native GPU physical features that cannot be duplicated on a CPU without physical hardware.

### 1.1 SASS Binary Execution — Tensor Core Opcodes

*   **Description**: HMMA (`.884`, `.1688`) and WGMMA (SM90) SASS opcodes are decoded in `src/compiler/sass/sass_decoder.cpp` but emit **comment-only placeholders** in the synthesized PTX (`// HMMA_884_F32: wmma matmul rd=... rs1=...`), not real `wmma.mma.sync.aligned` intrinsics. Standard SASS opcodes (60+ arithmetic, memory, control-flow, atomic, warp-level) decode correctly.
*   **Impact**: Kernels that rely exclusively on tensor-core instructions for correctness (e.g., some cuBLAS/cuDNN internal cubins) will produce zero/incorrect output when executed through the SASS decoder.
*   **Emulation Behavior**: All other SASS opcodes decode to valid PTX and compile via the LLVM JIT. For cubins with no PTX section at all, `CUDA_ERROR_NO_BINARY_FOR_GPU` is returned to allow the application's fallback paths to engage.

### 1.2 Physical CUPTI PMU Hardware Counters

*   **Description**: Physical hardware units (streaming multiprocessor warp dispatchers, texture cache units, PCIe bus monitors) do not exist on a host CPU.
*   **Emulation Behavior**: CUPTI subscribers receive high-fidelity telemetry by reading and scaling native host CPU PMU performance counters (`perf_event_open` on Linux, `QueryThreadCycleTime` on Windows, `thread_basic_info` on macOS). When a real NVIDIA GPU and driver are present, calls are forwarded to the native `libcupti.so`; the CPU PMU proxy also runs in parallel to produce unified OTLP output. Metrics like `achieved_occupancy` and `l1_global_load_hit_rate` are heuristic estimates derived from the instruction-mix fraction, not hardware measurements.

### 1.3 Physical GPUDirect RDMA & PCIe P2P — QP Handshake Gap

*   **Description**: The RDMA transport (`src/advanced/rdma_transport.cpp`, compiled with `-DVGRE_ENABLE_RDMA=ON`) has real InfiniBand Verbs calls: `ibv_open_device`, `ibv_alloc_pd`, `ibv_reg_mr`, `ibv_create_qp(IBV_QPT_RC)`, and the full QP state machine (RESET→INIT→RTR→RTS with RoCE GRH support). A 256 MB bounce buffer is pre-registered and a zero-copy allocation cache (`preRegisterAllocation`) avoids per-transfer `ibv_reg_mr` overhead.
*   **Partial Gap**: `sendQPInfo()` and `recvQPInfo()` in `rdma_transport.cpp` both `return false` immediately. These functions are responsible for exchanging QP parameters (LID, QPN, PSN, rkey, remote bounce-buffer address) with the remote peer over the `SecureChannel`. Since `RDMAConnection::connect()` calls them, it cannot successfully complete the QP handshake. **Cross-node RDMA Write transfers do not work end-to-end.** Local memory registration and QP object creation work correctly.
*   **Fallback**: Peer-to-peer copies fall back to AVX-accelerated `streamingMemcpy`. Cross-NUMA/cross-node transfers use LZ4 memory compression over TCP. Local inter-process communication uses POSIX/Windows shared memory.

### 1.4 Physical GPU Virtualization (vGPU/VFIO)

*   **Description**: VGRE operates entirely in user-space as an API interception runtime. It does not virtualize the hardware kernel driver layer (`/dev/nvidia*`) at the VFIO/hypervisor level.
*   **Emulation Behavior**: Offloading to remote GPU workers uses dynamic `dlopen` of CUDA/NVRTC, operating as a high-performance proxy. The Kubernetes operator and TCP cluster manager orchestrate multi-node dispatch without any hardware-virtualization layer.

### 1.5 Native Cross-Platform NUMA

*   **Description**: Thread binding and memory allocation optimizations differ between platforms.
*   **Emulation Behavior**: Fully supported on Linux via `/sys/devices/system/node` scanning and `pthread_setaffinity_np`. On macOS, `thread_policy_set(THREAD_AFFINITY_POLICY)` provides soft affinity hints. On Windows, thread affinity masks are set but NUMA-aware `VirtualAllocExNuma` is not used — memory allocation uses the standard heap. UVM page migration via `SYS_mbind` is Linux-only.

---

## 2. Partial Implementations (Code Exists, Specific Gaps Remain)

These items have substantial real implementations but with documented holes.

### 2.1 cuDNN Graph API — CONV_NORM Fusion and RNG Backend

*   **Description**: The cuDNN Graph API (`src/api/cudnn/cudnn_graph.cpp`) implements real fusion detection and execution. `CONV_ACTIVATION`, `GEMM_POINTWISE`, `GEMM_ACTIVATION`, and `NORM_ACTIVATION` fusions all have working execution paths.
*   **Gap 1 — CONV_NORM**: The `CONV_FWD → NORM_FWD` pattern is detected as `FusionKind::CONV_NORM` but `executeFusedPair` returns `false` for this kind (no fused execution path), so it falls back to sequential: two separate backend calls. A true batched Conv+BatchNorm fused kernel is not implemented.
*   **Gap 2 — RNG Backend**: `CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR` returns `CUDNN_STATUS_NOT_SUPPORTED`. Dropout noise generation via the backend graph API is not executed; callers must use `cudnnDropoutForward` or host-side RNG instead.

### 2.2 RDMA QP Info Exchange (see §1.3 above)

Cross-node RDMA is structurally complete except for the QP info serialization over SecureChannel. See §1.3 for full details.

### 2.3 SASS Tensor Core Opcode Decode (see §1.1 above)

HMMA.884, HMMA.1688, and WGMMA decode to comment-only placeholders. See §1.1 for full details.

### 2.4 gRPC Transport

*   **Description**: `src/advanced/grpc_transport.cpp` has a full gRPC client wrapper (`VGREGRPCClient`). However, it is compiled only when the gRPC library is available (`-DVGRE_ENABLE_GRPC=ON`). Without this flag, all gRPC methods are empty stubs that return error codes.
*   **Impact**: Distributed workload dispatch over gRPC requires an explicit build-time dependency on `libgrpc`. The default build uses the TCP cluster transport instead.

### 2.5 OpenCL Adapter — iGPU Path

*   **Description**: `src/runtime/igpu_opencl_executor.cpp` translates CUDA kernel source to OpenCL C via regex substitution and dispatches on integrated GPUs. The transpiler covers basic CUDA indexing builtins (`blockIdx`, `threadIdx`, `gridDim`, `blockDim`) and warp shuffles (via `cl_intel_subgroups` or local-memory fallback).
*   **Limitations**: Only handles single-source CUDA kernels; does not handle texture operations, dynamic shared memory, cooperative groups, or inline PTX (`asm volatile`). Complex third-party CUDA libraries that use device-side binary features are not supported via this path.

### 2.6 CUDA Linker (cuLink API)

*   **Description**: `src/api/cuda_driver/cuda_driver_module.cpp` notes "VGRE does not have a CUDA linker; we provide stubs that collect PTX data." The `cuLinkCreate` / `cuLinkAddData` / `cuLinkComplete` API is partially stubbed — it collects PTX blobs but does not perform actual device-link-time optimization (LTO) across compilation units.
*   **Impact**: Multi-TU device programs that rely on cross-module `extern __device__` linkage may fail or produce incomplete output.

---

## 3. Removed Items (Previously Listed, Now Implemented)

The following items appeared in earlier versions of this document but have been removed because they are now fully or substantially implemented:

| Item | Version Removed | Replacement |
|---|---|---|
| cuDNN Graph API (v9+) fused operations | v12.0.0 | See `implementationPlan.md` Track 4 — `CONV_ACTIVATION`, `GEMM_*`, `NORM_ACTIVATION` are fused; `CONV_NORM` is the one remaining gap (§2.1 above) |
