# VGRE Feature Coverage Matrix

This matrix summarizes what is **measured**, **estimated**, or **virtualized** in the current implementation, plus notable gaps.

## Backend

| Subsystem | Status | Notes |
|---|---|---|
| CUDA Interceptor / C API | Partial | Adds broad device attribute coverage + stream priority/query, mem info, runtime/driver version, device flags, PCI queries, host alloc/register, memcpy2D, mallocPitch, peer access/memcpy. |
| CUDA Driver API | Supported | cuInit/device/ctx/mem/stream/event coverage; supports cuModuleLoadData(Ex) for ELF/cubin and PTX. |
| OpenCL Adapter | Minimal | Synchronous, compatibility facade with robust machine-specific Platform/Device IDs. |
| Kernel Parser | Supported | Metadata extraction via manual tokenizer/regex; uses LLVM JIT for **accurate instruction counting** and performance accountancy. |
| LLVM JIT | Robust | Full pipeline with wrapper generation, caching, and **Thread-Local Storage (TLS)** context. |
| Execution Model | Functional | Hybrid: SIMD vectorization for pure kernels; **Thread-parallel (SIMT)** for barrier/syncthreads kernels. |
| Shared Memory | Partial | Static arrays packed; `extern` mapped to dynamic. Collision-free via TLS, but limited to one dynamic pointer. |
| __syncthreads | Supported | Correctly synchronized via per-block barrier; scalable to 1024 parallel workers (physical thread pool limit). |
| Memory Manager | Managed | VRAM managed via aligned host allocations; UVM residency managed via mprotect/VEH. Thread-safe atomic metrics. Supports configurable latency modeling for topology-aware transfers. |
| Telemetry | Reported | Real-time GFLOPS/BW reporting based on **LLVM-calibrated instruction counts**; EMA (alpha=0.3) smoothing applied. |
| Runtime Profiler | Functional | Per-kernel timing and JSON export. |
| CUDA Graphs | Supported | Serial topological DAG execution with cycle detection and lifecycle validation. |
| Global Symbols | Supported | Resolution of globally exported symbols between Host and JIT via `extern "C"` linkage. |
| Distributed / TCP Cluster | Experimental | Limited arg count and memory coherence; best for demos. |

## Dashboard

| Area | Status | Notes |
|---|---|---|
| Telemetry Polling | Functional | 500ms FFI polling; uses adaptive estimates when profiler empty. |
| UVM Map | Functional | Displays derived residency map from authoritative mprotect/VEH data. |
| Kernel Stats | Functional | Driven by runtime profiler JSON when enabled. |
| Device Selection | Missing | Single-device view only. |
| Background Compute | Robust | Dynamic compute-bound workload with configurable **duty-cycle scaling** (Load Factor 0.0-1.0). |
| Error Surfacing | Minimal | FFI failures log only to console. |
| History Export | Missing | No persistent history or export. |

## Known Gaps (Highest Impact)

- Full CUDA parsing and device function support (limited by regex meta-extractor).
- Scaling `__syncthreads` beyond the 1024-thread parallel worker pool.
- Multi-device visualization in Dashboard.
