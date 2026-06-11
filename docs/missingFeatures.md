# VGRE — Open Gaps, Enhancements, and Innovation Targets (Phase 2)

**Last Updated**: 2026-06-10

This is the authoritative, forward-looking ledger of what VGRE still needs to be a
genuinely production-deployable CPU GPU-emulation runtime. **Phase 1 (tracks A–W)
is complete and has been removed** — every item below is a *current, verified*
gap found by auditing the source tree (`grep` for stubs/heuristics/`NOT_SUPPORTED`,
reading the simplified code paths) and by researching the 2026 state of the art
(vLLM, TensorRT-LLM, ZLUDA/SCALE) for the features a production inference/HPC
runtime is expected to have.

> **Honesty note:** prior docs claimed "production-ready" and "validated across
> Linux/Windows/macOS". That is **not** true today: there is no CI, the
> Windows/macOS code paths are compile-guarded but **unverified**, and several
> compute paths are deliberately simplified (listed below). This document is the
> corrective: it states what must be true before "production-ready" is honest.

---

## 1. Real Implementation Gaps (simplified / partial code paths)

These are places where the code runs and returns a *result*, but the result is
approximate, slower than necessary, or only covers the easy case. Each cites the
actual source location.

### 1.1 Flash Attention — K/V recomputed instead of cached
- **Location**: `src/compiler/kernel_fusion_engine.cpp:588` (`// simplified; real impl caches`), `:683` (`// simplified per-element; full reduction would use shared memory`)
- **Gap**: the fused attention kernel recomputes K and V per query position and reduces per-element instead of tiling K/V into a cache-resident block and doing an online-softmax block reduction. Correct output, but O(N²) memory traffic — defeats the point of Flash Attention.
- **Fix**: implement true tiled online-softmax (FlashAttention-2 algorithm): load a K/V tile, accumulate running max/sum, rescale. Cache the K/V tile in an L2-sized buffer.

### 1.2 NCCL ring AllReduce — barrier-per-round, shared single buffer
- **Location**: `src/api/nccl/nccl_collectives.cpp:227` (`// In a real distributed system each rank would own its own buffer`), `:244` (`// barrier-per-round model`)
- **Gap**: the ring AllReduce shares one buffer and synchronizes with a barrier each step rather than each rank owning its slice and pipelining reduce-scatter + all-gather. Functionally correct, but serializes what should overlap.
- **Fix**: per-rank slice ownership; pipelined reduce-scatter/all-gather; keep the Kahan-compensated reduction (that part is already good).

### 1.3 WMMA / Tensor-core emulation — flat dot-product — ✅ FIXED (2026-06-11, Track 9)
- **Location**: `include/vgre/compiler/wmma_emulation.h` (raw-PTX `vgre_mma_*` helpers).
- **Was**: the raw-PTX `mma.sync` helpers computed a meaningless flat dot-product
  AND always threw (brace operands weren't flattened), so the PTX tensor-core path
  never ran. **Now**: a real warp-collective MMA — each lane deposits its fragment
  into a per-warp scratch buffer, barriers, the full tile GEMM is reconstructed via
  the PTX-ISA fragment→(row,col) maps, and per-lane outputs are scattered back.
  Implemented for FP16/BF16/TF32/INT8 (the precisions this gap names). Verified
  bit-exact (f16 0.00e+00, s8 exact) by `test_tensorcore_mma`. See
  implementationPlan.md Track 9.

### 1.4 MPS multi-client transport — "simplified" pipe handling — ✅ FIXED (2026-06-11, Track 14)
- **Location**: `src/advanced/mps_control.cpp` (`createSecurePipeInstance`, accept loop).
- **Was**: the accept loop re-created each next-client instance with a NULL DACL
  (dropping the CREATOR_OWNER+SYSTEM ACL — a security regression) and a `maxClients_`
  instance cap. **Now**: `createSecurePipeInstance()` builds the full DACL and uses
  `PIPE_UNLIMITED_INSTANCES`, and is shared by `start()` and the accept loop so every
  per-client instance is equally secured with no cap. See implementationPlan.md Track 14.

### 1.5 cuSPARSE format conversion — `sparse_view` incomplete
- **Location**: `src/api/cusparse/sparse_view.cpp:106` (`// This would need a full conversion implementation`)
- **Gap**: some sparse-format → sparse-format conversions in the generic view path are not implemented.
- **Fix**: implement the missing CSR↔CSC↔COO↔BSR↔ELL conversions (the per-format SpMV already exists; the conversions are the gap).

### 1.6 Graph optimizer — DCE "just removes" nodes
- **Location**: `src/core/graph_optimizer.cpp:461` (`// For simplicity in this v0.1.1 baseline, we'll just remove them`)
- **Gap**: dead-node elimination removes nodes without re-validating downstream dependency edges / reference counts in all cases.
- **Fix**: proper liveness analysis over the graph DAG before removal; preserve edges of any node with a live consumer.

### 1.7 PTX translator — carry propagation & memory operands simplistic
- **Location**: `src/compiler/ptx/ptx_translator_map.cpp:81` (`// Simplified: we propagate carry as part of the expression`), `:394` (`// Memory (simplistic: treat ptr operand as C pointer)`)
- **Gap**: `addc/subc/madc` carry-chain PTX and some addressed memory operands are translated approximately. Multi-word integer arithmetic (128-bit) and non-trivial addressing modes may be wrong.
- **Fix**: model the carry flag explicitly across the carry chain; handle the full PTX addressing grammar (`[reg+imm]`, vectorized `ld.v2/v4`).

### 1.8 Device cuRAND skip-ahead — bounded loop
- **Location**: `include/vgre/compiler/cuda_device_libs/curand_kernel.h:90` (`// Simplified: advance by n calls ... offset <= 2^20 is practical`)
- **Gap**: `curand_init` offset/skip-ahead is implemented as a loop, not the O(log n) matrix-power jump. Large offsets are impractically slow.
- **Fix**: implement the XORWOW/Philox logarithmic skip-ahead (matrix exponentiation over GF(2) for XORWOW; counter addition for Philox).

---

## 2. API Coverage Gaps (`NOT_SUPPORTED` returns that frameworks hit)

Audit target: each remaining `*_STATUS_NOT_SUPPORTED` / `cudaErrorNotSupported`
should either be implemented or carry a documented, framework-irrelevant reason.

### 2.1 cuSPARSE triangular solve & factorization variants
- **Location**: `src/api/cusparse/cusparse_triangular.cpp`, `cusparse_factorization.cpp` (2 each)
- **Gap**: some `cusparseSpSV`/`cusparseSpSM` analysis-policy variants and factorization paths return `NOT_SUPPORTED`.

### 2.2 cuSOLVER variants
- **Location**: `src/api/cusolver/cusolver_core.cpp` (2)
- **Gap**: specific decompositions/policies unimplemented.

### 2.3 CUDA Virtual Memory IPC
- **Gap**: `cuMemRetainAllocationHandle` (IPC export of VMM allocations) is a stub; `cuMemExportToShareableHandle`/`ImportFromShareableHandle` round-trip across processes is unverified.

### 2.4 `cudaStreamCaptureModeThreadLocal`
- **Gap**: only `cudaStreamCaptureModeGlobal` is enforced; thread-local capture mode captures all threads' calls.

### 2.5 `cuMemcpyAsync` with `CU_MEM_OPERATION_TYPE_MEMSET`
- **Gap**: the operation-type memset overload falls back to synchronous `memset` instead of routing to the scheduler.

---

## 3. Cross-Platform & Footprint (the "works on all three OSes" claim is unverified)

### 3.1 No CI — Windows/macOS support is compile-guarded but unproven
- **Reality**: there is no `.github/workflows`. The build and full test suite have only ever run on Linux. The `#if defined(_WIN32)` / `__APPLE__` branches compile in isolation but have **never been built or tested end-to-end** on those OSes.
- **Fix**: add a GitHub Actions matrix (`ubuntu-latest`, `windows-latest`, `macos-latest`) that configures, builds, and runs the test suite. Until that is green, the README/PROJECT_STATUS must say "Linux verified; Win/macOS unverified."

### 3.2 macOS NUMA / affinity is a stub — ✅ FIXED (2026-06-11, Track 21)
- **Location**: `src/core/scheduler_numa.cpp`.
- **Was**: the active macOS branch was a physical-CPU-count stub; a real
  `thread_policy_set` P/E-core implementation existed but was dead code (shadowed by
  a duplicate `#elif __APPLE__`). **Now**: the stub is removed so the real path
  compiles — workers are split by Apple-Silicon performance (`hw.perflevel0`) vs
  efficiency (`hw.perflevel1`) cores via `THREAD_AFFINITY_POLICY` tags. Documented
  boundary: macOS removed hard pinning, so these are scheduler hints, not hard pins.

### 3.3 Lightweight footprint audit (unmeasured)
- **Gap**: binary size, RSS at idle, and the dependency surface (LLVM-18, OpenBLAS/LAPACK, optional libibverbs/libsecret/tss2/OpenCL/grpc/protobuf/sqlite) have never been measured or minimized. A "lightweight" claim needs: a `-DVGRE_MINIMAL` build that drops every optional dep, a measured `.so` size, and a documented runtime memory floor.
- **Fix**: add a minimal build profile + a size/footprint report; statically assert no accidental hard dependency on optional libs.

---

## 4. Production-Deployment Readiness (what "production" actually requires)

Researched against the 2026 production-serving baseline (vLLM/TensorRT-LLM ops).

### 4.1 Live metrics endpoint (Prometheus)
- **Gap**: VGRE exports *post-hoc* traces (Chrome JSON, Perfetto, `.nsys-rep`) but has **no live `/metrics` endpoint**. Production serving requires a scrape target reporting: kernels/s, JIT-compile latency & cache hit-rate, queue depth, host-memory in use, per-virtual-device utilization.
- **Fix**: a tiny embedded HTTP server exposing Prometheus text format (`VGRE_METRICS_PORT`).

### 4.2 Health / readiness probes + graceful shutdown
- **Gap**: no `/healthz` / `/readyz` endpoints for Kubernetes liveness/readiness; shutdown drains in-flight work only via destructors.
- **Fix**: explicit health endpoints and a `vgre_drain()`/SIGTERM handler that finishes in-flight kernels then exits 0.

### 4.3 Versioned, reproducible packaging
- **Gap**: no published Docker image tags, no SemVer release process, no `vgre --version` reporting build hash + feature flags. The k8s device-plugin Dockerfile exists but isn't built/published by CI.
- **Fix**: SemVer tags, multi-arch (amd64/arm64) Docker images to GHCR pinned by digest, `vgre_get_build_info()` C API.

### 4.4 Config validation & safe defaults
- **Gap**: many `VGRE_*` env vars are read ad-hoc; an invalid value is sometimes silently ignored. No single schema/validator.
- **Fix**: one config registry that validates ranges, logs the effective config once at startup, and rejects unknown `VGRE_*` vars with a warning.

---

## 5. Missing Innovations (modern AI/ML the runtime can't yet serve)

Grounded in the 2026 inference stack (FP8 is "the single most impactful config",
paged attention + continuous batching are the vLLM core).

### 5.1 FP8 (E4M3 / E5M2) and FP4 quantized compute
- **Gap**: VGRE has FP16/BF16/INT8 paths but **no FP8 or FP4**. FP8 is the default production quantization on Hopper/Blackwell and the single biggest throughput lever; vLLM/TensorRT-LLM ship FP8 by default. Without it VGRE cannot emulate modern quantized inference.
- **Design**: FP8 E4M3/E5M2 software types + scaled matmul (per-tensor/row scales), FP4 (E2M1) with group scales (NVFP4/MXFP4). Wire into cuBLASLt `*_8F_E4M3` GEMM and the WMMA emulation.

### 5.2 Paged Attention + KV-cache manager (vLLM backend)
- **Gap**: no paged KV-cache, no block table, no continuous batching. These are what make VGRE usable as a *serving* backend rather than a single-kernel emulator.
- **Design**: a block-allocator KV-cache (PagedAttention layout), a `cudaMemcpy`-backed block table, and the paged-attention kernel in the fusion engine. Target: run a small LLM through vLLM's VGRE backend.

### 5.3 Continuous batching / request scheduler (serving layer)
- **Already done — do NOT re-implement:** CUDA Graph conditional `IF`/`WHILE`/`SWITCH` nodes exist *and execute* (`src/core/runtime_engine_graph.cpp:140`, `GraphCondType`, `cudaGraphAddConditionalNode`, with a WHILE iteration cap) — verified 2026-06-10.
- **Gap**: VGRE is a kernel/op emulator with no *serving* orchestration. Modern inference (vLLM/TGI) multiplexes many concurrent requests with continuous (in-flight) batching: new requests join a running batch at token boundaries, finished sequences are evicted, and the KV-cache (§5.2) is shared. VGRE has no request queue, batch manager, or token-boundary preemption.
- **Design**: a request queue + batch scheduler over the paged KV-cache (§5.2) admitting/evicting sequences per decode step — what turns VGRE from "runs a kernel" into "serves a model".

### 5.4 Large-PTX JIT compile speed (PyTorch/Triton)
- **Gap**: PyTorch/Triton ship multi-MB PTX/bitcode modules. The JIT is tuned for code quality, not compile speed (the same problem ZLUDA reports). First-call latency on large modules is unmeasured and likely poor.
- **Design**: a fast-path `-O1` tier for first compile + background re-JIT to `-O3`; parallel module compilation; measure & cap p99 compile latency.

### 5.5 End-to-end framework validation harness
- **Gap**: there is no test that actually runs **PyTorch** or **vLLM** against VGRE end-to-end. Unit/integration tests validate kernels in isolation; "runs unmodified CUDA workloads" is unproven for a real framework.
- **Design**: a gated CI job that installs CPU-only PyTorch + the VGRE interception shim and runs a tiny model fwd/bwd, asserting numerical agreement with a reference.

### 5.6 perf_event_open proxy counters (CUPTI fidelity)
- **Gap**: CUPTI counters are instruction-mix heuristics. Linux `perf_event_open` could provide real L1/L2/IPC proxies (documented as proxies).
- **Design**: optional `perf_event_open` integration behind `VGRE_USE_PERF_COUNTERS`.

---

## 6. Test Fidelity (passing ≠ real)

### 6.1 Parallel-execution flakiness (resource oversubscription)
- **Reality**: under `ctest -j$(nproc)`, a *different* clang/JIT test (ClangEnhanced, VectorAddition, ExternShared, IGPUBackend, Scheduler, …) intermittently fails/aborts each run from CPU + fork oversubscription (~N processes each forking `clang -O3 -march=native`). All pass standalone. This masks signal and would fail a CI gate.
- **Tried & reverted (2026-06-10)**: capping per-process threads (`VGRE_DEFAULT_THREAD_COUNT=4`) backfired (slower tests overlap more); a cross-process `flock` compile gate at a low limit made it worse. See `implementationPlan.md` Track 6 for the full post-mortem and better directions (bounded CI `-j` / `ctest --test-load`, `PROCESSORS` hints, measured per-test locks).
- **Status**: open — must be solved before CI (Track 1) can be a reliable gate.

### 6.2 iGPU OpenCL path is SKIPped, not tested
- **Location**: `tests/integration/test_igpu_backend.cpp` — SKIPs when the Mesa/i915 OpenCL driver can't dispatch (the common case). The transpiler's emitted OpenCL (warp shuffle/ballot, textures) is therefore **never executed** in CI.
- **Fix**: a host-side OpenCL-C compile check (clang `-cl-std`) of the generated kernel string so syntax/semantics are validated even without a working device.

### 6.3 No golden numerical-vector suite
- **Gap**: correctness is checked per-test with ad-hoc tolerances. There is no consolidated set of reference vectors (cuBLAS GEMM, cuDNN conv, FFT, RNG distributions) compared against a trusted oracle (NumPy/SciPy/OpenBLAS) at tight tolerance.
- **Fix**: a `test_golden_vectors` suite seeded from NumPy/SciPy references.
