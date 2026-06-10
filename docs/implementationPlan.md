# VGRE Implementation Plan — Phase 2

**Last Updated**: 2026-06-10

Phase 1 (tracks **A–W**) is complete and has been removed from this plan; see git
history (`feat(track-*)`, `fix(track-*)`) and `docs/missingFeatures.md`'s preamble.

Phase 2 below is the *real* remaining work to make VGRE honestly production-ready.
Every track maps to a verified gap in `docs/missingFeatures.md` (cited by §). Tracks
are ordered by priority: **P0** = blocks an honest "production-ready" claim, **P1**
= major capability/coverage, **P2** = fidelity/innovation.

---

## Completion Tracker

| # | Track | Pri | Maps to | Status |
|---|---|---|---|---|
| 1 | CI matrix (Linux/Windows/macOS) build+test | P0 | §3.1 | 🟡 Workflow added |
| 2 | Honest status docs (no false "validated") | P0 | §3.1 | 🟡 In progress |
| 3 | Prometheus `/metrics` endpoint | P0 | §4.1 | 🔴 Not started |
| 4 | Health/readiness probes + graceful drain | P0 | §4.2 | 🔴 Not started |
| 5 | Config registry + validation + `build_info` | P0 | §4.4, §4.3 | 🔴 Not started |
| 6 | Parallel-test stability (RESOURCE_LOCK) | P0 | §6.1 | 🔴 Not started |
| 7 | Flash Attention real tiled online-softmax | P1 | §1.1 | 🔴 Not started |
| 8 | NCCL pipelined ring (reduce-scatter/all-gather) | P1 | §1.2 | ✅ Done |
| 9 | WMMA real fragment-layout MMA | P1 | §1.3 | 🔴 Not started |
| 10 | cuSPARSE `sparse_view` conversions | P1 | §1.5 | ✅ Done |
| 11 | cuRAND logarithmic skip-ahead | P1 | §1.8 | ✅ Done |
| 12 | PTX carry-chain + full addressing | P1 | §1.7 | ✅ Done |
| 13 | Graph optimizer real liveness DCE | P1 | §1.6 | ✅ Done |
| 14 | MPS per-client pipe instances (Windows) | P1 | §1.4 | 🔴 Not started |
| 15 | `NOT_SUPPORTED` audit (cuSPARSE/cuSOLVER) | P1 | §2.1, §2.2 | ✅ Audited |
| 16 | VMM IPC + ThreadLocal capture + async memset | P1 | §2.3–2.5 | 🔴 Not started |
| 17 | FP8 (E4M3/E5M2) + FP4 quantized compute | P1 | §5.1 | 🔴 Not started |
| 18 | Paged Attention + KV-cache manager | P1 | §5.2 | 🔴 Not started |
| 19 | Versioned multi-arch Docker images | P1 | §4.3 | 🔴 Not started |
| 20 | Minimal build profile + footprint report | P2 | §3.3 | 🔴 Not started |
| 21 | macOS affinity (P/E cores) + documented boundary | P2 | §3.2 | 🔴 Not started |
| 22 | Continuous batching request scheduler | P2 | §5.3 | 🔴 Not started |
| 23 | Large-PTX fast-tier JIT (compile speed) | P2 | §5.4 | 🔴 Not started |
| 24 | PyTorch/vLLM end-to-end validation harness | P2 | §5.5 | 🔴 Not started |
| 25 | iGPU transpiler OpenCL-C compile check | P2 | §6.2 | 🔴 Not started |
| 26 | Golden numerical-vector suite | P2 | §6.3 | 🔴 Not started |
| 27 | perf_event_open proxy counters | P2 | §5.6 | 🔴 Not started |

---

## P0 — Blocks an honest "production-ready" claim

### Track 1 — CI matrix (Linux/Windows/macOS)
**New**: `.github/workflows/ci.yml`. Matrix over `ubuntu-latest`, `windows-latest`,
`macos-latest`: configure (CMake), build, run `ctest`. Cache LLVM. Until all three
are green, no doc may claim cross-platform "functional"/"validated". This is the
single most important production gate — it converts the unverified `#ifdef`
branches into a tested guarantee.
**Acceptance**: green CI on all three OSes; test artifacts uploaded.

### Track 2 — Honest status documentation
**Files**: `README.md`, `docs/PROJECT_STATUS.md`, `docs/README.md`.
Replace "Cross-Platform: all functional" / "validated across Win/macOS" with the
truth: *Linux verified; Windows/macOS code-complete but unverified pending Track 1.*
Update stale test counts. Soften "zero stubs" to enumerate the known simplified
paths (§1). State the production-readiness gates (Tracks 1,3,4,5,6).
**Acceptance**: no doc asserts a capability that isn't CI-verified.

### Track 3 — Prometheus `/metrics` endpoint
**New**: `src/api/metrics_server.cpp` + C API `vgre_start_metrics_server(port)`.
A minimal embedded HTTP/1.1 server (no new deps — raw socket) serving Prometheus
text format: `vgre_kernels_total`, `vgre_jit_compile_seconds`, `vgre_jit_cache_hits_total`,
`vgre_scheduler_queue_depth`, `vgre_host_memory_bytes`, per-device gauges. Enabled
by `VGRE_METRICS_PORT`.
**Acceptance**: `curl localhost:$PORT/metrics` returns valid Prometheus exposition.

### Track 4 — Health/readiness probes + graceful drain
**Files**: metrics server (`/healthz`, `/readyz`), `src/core/runtime_engine.cpp`
(`vgre_drain()`), worker CLI SIGTERM handler. `/readyz` returns 200 once the JIT
pipeline is warm and (if clustered) the master link is up. SIGTERM → stop
accepting work, drain in-flight kernels, exit 0.
**Acceptance**: k8s-style probe sequence works; SIGTERM mid-load loses no kernels.

### Track 5 — Config registry + validation + build info
**New**: `src/common/config_registry.cpp`. Single source of truth for every
`VGRE_*` var: name, type, range, default. Validate at startup, log the effective
config once, warn on unknown `VGRE_*`. Add `vgre_get_build_info()` → version,
git hash, enabled features (SIMD level, gRPC/RDMA/OpenCL/SQLite on/off).
**Acceptance**: invalid `VGRE_*` value is rejected with a clear message, not
silently ignored; `vgre --version` prints build info.

### Track 6 — Parallel-test stability
**Problem**: under `ctest -j$(nproc)` a *different* clang/JIT test (ClangEnhanced,
VectorAddition, ExternShared, …) intermittently fails/aborts each run from CPU +
fork oversubscription (~N processes each forking `clang -O3 -march=native` and
spawning a worker pool). All pass standalone.
**Approaches tried (2026-06-10) that did NOT work — do not repeat blindly**:
- `VGRE_DEFAULT_THREAD_COUNT=4` for tests: *backfired* — fewer threads made each
  test slower, increasing temporal overlap and thus concurrency pressure.
- A cross-process `flock` compile gate (`VGRE_MAX_CONCURRENT_COMPILES`): at a low
  limit it made failures *more* frequent (nested same-thread compiles + scarce
  slots; even with a reentrancy guard the suite was not stabilized and
  ClangEnhanced began failing every run). Reverted.
**Better directions to try next**: (a) CI runs at a *bounded* `-j` / `ctest
--test-load $(nproc)` rather than `-j$(nproc)`; (b) `PROCESSORS` property on the
heavy tests so CTest's load-aware scheduler accounts for them; (c) a per-test
`RESOURCE_LOCK` only on the few heaviest, measured to actually help (not a blanket
lock). Validate any approach with ≥10 consecutive full runs before claiming fixed.
**Acceptance**: 10 consecutive full runs (at the chosen CI parallelism) are 100% green.

---

## P1 — Major capability & coverage

### Track 7 — Flash Attention real tiled online-softmax
**File**: `src/compiler/kernel_fusion_engine.cpp` (lines ~588, ~683).
Replace the per-position K/V recompute + per-element reduction with the
FlashAttention-2 tiled algorithm: stream K/V tiles, maintain running max & sum,
rescale the accumulator. Keep the GQA detection from Phase 1.
**Acceptance**: fused output matches a naive softmax-attention reference at 1e-4;
memory traffic is O(N·d), not O(N²).

### Track 8 — NCCL pipelined ring AllReduce
**File**: `src/api/nccl/nccl_collectives.cpp` (~227, ~244). Per-rank slice
ownership; reduce-scatter then all-gather, overlapping steps (no global barrier
per round). Retain Kahan-compensated accumulation.
**Acceptance**: bit-identical to current results; measurable overlap vs barrier model.

### Track 9 — WMMA real fragment-layout MMA
**File**: `include/vgre/compiler/wmma_emulation.h` (~333, ~502).
**Scoping note (2026-06-10):** the high-level `nvcuda::wmma::mma_sync` C++ API is
already correct — `load_matrix_sync` materialises the full tile per serial thread
and the AVX-512 GEMM operates on it. The defect is confined to the low-level
raw-PTX `mma.sync.aligned.*` helpers (`vgre_mma_m16n8k16_*`, the "flat
dot-product" at ~333), used by Triton/CUTLASS-generated PTX. These are
**warp-collective**: a single lane holds only 8 of the 16 K-elements of A and 4
of B, so its 4 output elements cannot be computed correctly in isolation. A
correct fix is therefore NOT local — it requires buffering all 32 lanes' A/B/C
fragments for a warp, computing the full 16×8×16 (etc.) GEMM once, and scattering
the per-lane D fragments back, which needs warp-level coordination in the serial
execution model. **Deferred** until the warp-collective fragment buffer exists;
implementing the per-shape fragment→lane maps (PTX ISA 9.7.13.4.x) is the second
half. Do not "fix" the helper in place — any per-lane-only result is still wrong.
**Acceptance**: results bit-comparable to a reference MMA for each supported precision.

### Track 10 — cuSPARSE `sparse_view` conversions
**File**: `src/api/cusparse/sparse_view.cpp:106`. Implement the missing
CSR↔CSC↔COO↔BSR↔ELL conversions.
**Acceptance**: round-trip conversion preserves the matrix; `cusparseSparseToDense`/`DenseToSparse` agree.

### Track 11 — cuRAND logarithmic skip-ahead
**File**: `include/vgre/compiler/cuda_device_libs/curand_kernel.h:90`. XORWOW jump
via GF(2) matrix exponentiation; Philox via counter addition. O(log n), not O(n).
**Acceptance**: `curand_init(seed, seq, offset)` with offset=10^9 returns the same
stream as the reference, fast.

### Track 12 — PTX carry-chain + full addressing
**File**: `src/compiler/ptx/ptx_translator_map.cpp` (~81, ~394). Explicit carry
flag across `addc/subc/madc`; full addressing grammar (`[reg+imm]`, `ld.v2/v4`).
**Acceptance**: 128-bit add and vectorized loads produce correct results.

### Track 13 — Graph optimizer real liveness DCE
**File**: `src/core/graph_optimizer.cpp:461`. Liveness over the graph DAG before
removal; never drop a node with a live consumer.
**Acceptance**: a graph with a node feeding a live output is not mis-eliminated.

### Track 14 — MPS per-client pipe instances (Windows)
**File**: `src/advanced/mps_control.cpp:469`. `PIPE_UNLIMITED_INSTANCES` + accept
loop, per-client instance — matching the POSIX per-connection model.
**Acceptance**: ≥4 concurrent MPS clients on Windows without pipe contention.

### Track 15 — `NOT_SUPPORTED` audit (cuSPARSE / cuSOLVER) — ✅ audited 2026-06-10
**Files**: `cusparse_triangular.cpp`, `cusparse_factorization.cpp`, `cusolver_core.cpp`.
**Finding**: every remaining `*_NOT_SUPPORTED` return in these files is a
legitimate, self-documenting *validation guard*, not a stub:
- `cusolver_core.cpp:717,739` — `sygvd/hegvd` reject `itype ∉ {1,2,3}` (the only
  generalised-eigenproblem types CUDA defines); out-of-range is genuinely invalid.
- `cusparse_triangular.cpp:327,741` — SpSV/SpSM reject `computeType` other than
  `CUDA_R_32F`/`CUDA_R_64F`; the guard condition states the reason.
- `cusparse_factorization.cpp:557,749` — SpGEMM value copy handles
  F32/F64/C32/C64 and rejects only exotic types (F16/I8) it cannot represent.
No lazy stubs remain; the guard conditions are the precise documented reasons.
**Acceptance met**: each `NOT_SUPPORTED` is a documented validation guard. (Adding
complex/half SpSV or extra eigenproblem types is a future capability, not a stub
fix.)

### Track 16 — VMM IPC + ThreadLocal capture + async memset
**Files**: `cuda_virtual_memory.cpp`, stream capture, `cuMemcpyAsync`.
Implement `cuMemRetainAllocationHandle` IPC round-trip; enforce
`cudaStreamCaptureModeThreadLocal`; route `CU_MEM_OPERATION_TYPE_MEMSET` to the scheduler.
**Acceptance**: cross-process VMM handle round-trips; thread-local capture excludes other threads.

### Track 17 — FP8 (E4M3/E5M2) + FP4 quantized compute
**Files**: new `src/core/math/fp8.h`/`fp4.h`, cuBLASLt GEMM dispatch, WMMA emulation.
Software E4M3/E5M2 + E2M1 types; scaled GEMM (per-tensor/row/group scales);
NVFP4/MXFP4 group scaling. Wire into `*_8F_E4M3` cuBLASLt paths and attention.
**Acceptance**: FP8 GEMM matches a reference dequant→FP32-GEMM→requant at the format's epsilon.

### Track 18 — Paged Attention + KV-cache manager
**Files**: new `src/core/kv_cache.cpp`, fusion engine paged-attention kernel.
Block-allocator KV-cache (PagedAttention layout), block table, paged-attention kernel.
Foundation for serving (continuous batching builds on this).
**Acceptance**: paged attention matches contiguous attention; block table grows/frees correctly.

### Track 19 — Versioned multi-arch Docker images
**Files**: `Dockerfile`, CI release job. SemVer tags; amd64+arm64 images to GHCR
pinned by digest; image runs the smoke test.
**Acceptance**: `docker run ghcr.io/.../vgre:<tag> vgre --version` works on both arches.

---

## P2 — Fidelity & innovation

### Track 20 — Minimal build profile + footprint report
`-DVGRE_MINIMAL` dropping every optional dep; measured `.so` size + idle RSS report
in docs; static assert no hard dep on optional libs.

### Track 21 — macOS affinity (P/E cores) + documented boundary
`src/core/scheduler_numa.cpp` macOS path: use `thread_policy_set` affinity tags;
handle Apple-Silicon performance/efficiency core split; document the boundary.

### Track 22 — Continuous batching request scheduler
*(Conditional IF/WHILE/SWITCH graph nodes are already implemented — verified — so
this slot is the genuinely-missing serving layer.)* A request queue + batch
scheduler over the paged KV-cache (Track 18): admit new sequences and evict
finished ones at token boundaries (continuous/in-flight batching, vLLM-style).
This is what makes VGRE a model *server*, not just a kernel runner.

### Track 23 — Large-PTX fast-tier JIT
Fast `-O1` first-compile tier + background re-JIT to `-O3`; parallel module
compilation; measure & cap p99 first-call latency on multi-MB modules.

### Track 24 — PyTorch/vLLM end-to-end validation harness
Gated CI job: CPU-only PyTorch + VGRE shim runs a tiny model fwd/bwd; assert
numerical agreement with a reference. Proves "runs unmodified CUDA frameworks".

### Track 25 — iGPU transpiler OpenCL-C compile check
Host-side `clang -cl-std` compile of the generated OpenCL string so the
transpiler's output (shuffle/ballot/textures) is validated even without a device.

### Track 26 — Golden numerical-vector suite
`tests/test_golden_vectors`: cuBLAS GEMM, cuDNN conv, FFT, RNG distributions vs
NumPy/SciPy/OpenBLAS references at tight tolerance.

### Track 27 — perf_event_open proxy counters
Optional `VGRE_USE_PERF_COUNTERS`: real L1/L2/IPC via Linux `perf_event_open`,
documented as proxies for the GPU PMU heuristics.
