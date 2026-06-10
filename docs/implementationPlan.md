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
| 2 | Honest status docs (no false "validated") | P0 | §3.1 | ✅ Done |
| 3 | Prometheus `/metrics` endpoint | P0 | §4.1 | ✅ Done |
| 4 | Health/readiness probes + graceful drain | P0 | §4.2 | ✅ Done |
| 5 | Config registry + validation + `build_info` | P0 | §4.4, §4.3 | ✅ Done |
| 6 | Parallel-test stability (bounded -j + targeted lock) | P0 | §6.1 | ✅ Done |
| 7 | Flash Attention real tiled online-softmax | P1 | §1.1 | ✅ Done |
| 8 | NCCL pipelined ring (reduce-scatter/all-gather) | P1 | §1.2 | ✅ Done |
| 9 | WMMA real fragment-layout MMA | P1 | §1.3 | 🔴 Not started |
| 10 | cuSPARSE `sparse_view` conversions | P1 | §1.5 | ✅ Done |
| 11 | cuRAND logarithmic skip-ahead | P1 | §1.8 | ✅ Done |
| 12 | PTX carry-chain + full addressing | P1 | §1.7 | ✅ Done |
| 13 | Graph optimizer real liveness DCE | P1 | §1.6 | ✅ Done |
| 14 | MPS per-client pipe instances (Windows) | P1 | §1.4 | 🔴 Not started |
| 15 | `NOT_SUPPORTED` audit (cuSPARSE/cuSOLVER) | P1 | §2.1, §2.2 | ✅ Audited |
| 16 | VMM IPC handle (cuMemRetainAllocationHandle) | P1 | §2.3–2.5 | ✅ Done |
| 17 | FP8 (E4M3/E5M2) + FP4 quantized compute | P1 | §5.1 | ✅ Core done |
| 18 | Paged Attention + KV-cache manager | P1 | §5.2 | ✅ Done |
| 19 | Versioned multi-arch Docker images | P1 | §4.3 | 🟡 Artifacts ready |
| 20 | Minimal build profile + footprint report | P2 | §3.3 | ✅ Done |
| 21 | macOS affinity (P/E cores) + documented boundary | P2 | §3.2 | 🔴 Not started |
| 22 | Continuous batching request scheduler | P2 | §5.3 | ✅ Done |
| 23 | Large-PTX fast-tier JIT (compile speed) | P2 | §5.4 | 🔴 Not started |
| 24 | PyTorch end-to-end validation harness | P2 | §5.5 | ✅ Done |
| 25 | iGPU transpiler OpenCL-C compile check | P2 | §6.2 | ✅ Done |
| 26 | Golden numerical-vector suite | P2 | §6.3 | ✅ Done |
| 27 | perf_event_open proxy counters | P2 | §5.6 | ✅ Core present |

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
**Solution (2026-06-10, ✅ working)**: combine (a)+(c) from the directions below:
1. Run bounded — `ctest -j2 --test-load $(nproc)` (the CI workflow uses exactly
   this) rather than `-j$(nproc)`, so CTest's load-aware scheduler stops starting
   new tests when the machine is already saturated.
2. A *targeted* `RESOURCE_LOCK vgre_jit_heavy` on ONLY the five heaviest
   clang+multi-device tests (MultiDeviceCooperativeComprehensive,
   MultiDeviceIntegration, ClangEnhanced, SASSDetection, DeviceCurandIntegration)
   so two of them never run together. Not a blanket lock (that backfired).
Result: 3 consecutive full `ctest -j2 --test-load` runs (204 tests each) were
100% green after the change, vs ~1 flake/run before.
**Earlier approaches that did NOT work — do not repeat blindly**:
- `VGRE_DEFAULT_THREAD_COUNT=4` for tests: *backfired* — slower tests overlapped
  more, increasing concurrency pressure.
- A cross-process `flock` compile gate at a low limit: made failures *more*
  frequent (nested same-thread compiles + scarce slots). Reverted.
**Acceptance**: met — the bounded+targeted-lock invocation is deterministic
across consecutive runs (CI uses it).

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

### Track 16 — VMM IPC handle + capture/memset audit — ✅ done (2026-06-10)
**File**: `src/api/cuda_virtual_memory.cpp`.
- ✅ **`cuMemRetainAllocationHandle`** was genuinely missing (not a stub) —
  implemented: looks up the allocation handle backing a mapped address via the
  VA→handle map and increments a new `PhysAlloc.refcount`; `cuMemRelease` is now
  refcounted (frees the physical backing only on the last release).
  `test_virtual_memory::test_retain_handle` (7/7) proves the address→handle
  recovery, the unmapped-address rejection, and that the memory survives until
  the last of two references is released.
- **Doc corrections** (the other two items were inaccurate, like §5.3/§1.2):
  - `cudaStreamCaptureModeThreadLocal` *is* tracked per-thread
    (`cudart_shim_capture.cpp` `t_captureMode`, `cudaThreadExchangeStreamCaptureMode`).
    Full cross-thread enforcement (erroring on another thread's concurrent op
    during a thread-local capture) is a low-impact relaxed-safety semantic, not a
    correctness stub; left as a documented minor refinement.
  - `cuMemcpyAsync` with `CU_MEM_OPERATION_TYPE_MEMSET` is **not a real CUDA API**
    (no such operation-type overload exists); the gap was spurious — removed.
**Acceptance met**: the real missing VMM IPC handle API round-trips with correct
refcount lifetime.

### Track 17 — FP8 (E4M3/E5M2) + FP4 quantized compute — ✅ core done (2026-06-10)
**File**: new `include/vgre/core/math/fp_quant_gemm.h`.
- E4M3 codec with **round-to-nearest-even**, denormals, ±448 saturation, NaN
  (the pre-existing WMMA encoders truncated; this one rounds).
- **FP4 E2M1** codec (the genuinely-missing format) — magnitudes
  {0,.5,1,1.5,2,3,4,6}, nearest-with-ties-to-even.
- **Per-tensor scaled FP8 GEMM** (`D = sA·sB·dequant(A)·dequant(B)`) and
  **MXFP4 block-scaled FP4 GEMM** (one dequant scale per K-block, the
  micro-scaling contract used by NVFP4/MXFP4).
`test_fp_quant` (9/9): codec round-trips/rounding; the scaled FP8 GEMM equals the
dequant→fp32 GEMM exactly and is within 3% Frobenius error of the true fp32
product; the block-scaled FP4 GEMM matches its dequant reference exactly.
**Follow-up (not done)**: exposing this through a `cublasLtMatmul` FP8/FP4
descriptor API is a separate, larger surface — no such path exists today. The
WMMA emulation already has an (unscaled) FP8 MMA; the scaled kernels here are the
building block for a future cuBLASLt FP8 matmul.
**Acceptance met**: the FP8 GEMM matches dequant→FP32-GEMM at the format epsilon.

### Track 18 — Paged Attention + KV-cache manager — ✅ done (2026-06-10)
**Files**: new `include/vgre/core/kv_cache.h`, `src/core/kv_cache.cpp`.
`KVCacheManager`: a fixed-block pool with a free list; `appendToken` grows a
sequence's per-sequence **block table** (logical→physical), allocating one block
at a time, so sequences need no contiguous memory (vLLM PagedAttention layout).
`pagedAttention` runs an online-softmax over a sequence's KV by walking its block
table (full or causal-up-to-position). `freeSequence` returns blocks to the pool;
exhaustion is reported, not crashed.
`test_kv_cache` (9/9): deliberately fragments the pool so a sequence gets
**non-contiguous** blocks, then shows paged attention matches a contiguous
reference to 1.2e-7 (full and causal), the free list tracks alloc/free, and pool
exhaustion returns false. Continuous batching on top of this is Track 22.
**Acceptance met**: paged == contiguous; block table grows/frees correctly.

### Track 19 — Versioned multi-arch Docker images — 🟡 artifacts ready (2026-06-10)
**Files**: `Dockerfile`, `.dockerignore`, `.github/workflows/release.yml`,
`vgre-worker --version`.
- Multi-stage `Dockerfile`: a builder compiles the runtime + worker (Release,
  LLVM-18) and `cmake --install`s to `/opt/vgre`; a slim runtime stage carries
  only `lib/`, `bin/`, and the runtime deps (clang/llvm for the JIT, OpenMP,
  BLAS, SQLite), runs as a non-root user, OCI version/revision labels, and a
  HEALTHCHECK of `vgre-worker --version`.
- `release.yml`: on a `vX.Y.Z` tag, buildx builds **linux/amd64 + linux/arm64**
  and pushes to GHCR with SemVer tags + OCI labels, then prints the digest to pin.
- Added `vgre-worker --version` (prints the build-info JSON) — verified locally.
**Verified locally**: `cmake --install` produces exactly the `/opt/vgre/{lib,bin}`
layout the runtime stage copies, and the installed worker `--version` runs with
`LD_LIBRARY_PATH=lib`. The full multi-arch image build/push runs in CI (buildx is
not available on this host), so this is 🟡 like Track 1 until the first tagged
release is green.
**Acceptance (pending CI)**: `docker run ghcr.io/.../vgre:<tag> --version` works on
both arches.

---

## P2 — Fidelity & innovation

### Track 20 — Minimal build profile + footprint report — ✅ done (2026-06-10)
**Files**: `CMakeLists.txt` (`VGRE_MINIMAL` option + override block +
`vgre_footprint` target), `tools/footprint_report.sh`.
`-DVGRE_MINIMAL=ON` forces every optional integration OFF (gRPC, RDMA, OpenCL
backend, FFTW3, zlib, TPM2, libsecret, dashboard, examples, SLURM, tests),
retaining only the CUDA/cuBLAS/cuSOLVER(LAPACK)/JIT core. `cmake --build
--target vgre_footprint` prints the active feature profile and the runtime
library sizes (as built + stripped). **Measured**: the minimal profile compiles
cleanly and roughly halves the on-disk footprint — full build 196.9 MB
(libvgre 96.7 + cudart 100.2) vs minimal 97.8 MB (49.0 + 48.8), ~83 MB stripped.
**Acceptance met**: a lean profile drops the optional deps and the footprint is
measured/reported.

### Track 21 — macOS affinity (P/E cores) + documented boundary
`src/core/scheduler_numa.cpp` macOS path: use `thread_policy_set` affinity tags;
handle Apple-Silicon performance/efficiency core split; document the boundary.

### Track 22 — Continuous batching request scheduler — ✅ done (2026-06-10)
**Files**: `ContinuousBatchScheduler` in `include/vgre/core/kv_cache.h` +
`src/core/kv_cache.cpp`. A request queue + in-flight batch scheduler over the
paged KV-cache (Track 18): each `step()` retires finished sequences (reclaiming
their KV blocks), admits waiting requests while batch slots AND KV blocks allow
(prefilling the prompt), then advances every running request by one decode token.
New requests submitted mid-flight join at the next step — true continuous
batching, not a static batch. `test_kv_cache` (15/15) runs 4 requests of
different lengths through a small pool with `maxBatch=2` (so they cannot all run
at once), submits a 4th request *mid-run*, and verifies all four finish,
concurrency never exceeds the batch limit, it takes more than one batch's worth
of steps, and the KV pool is fully reclaimed at the end.

### Track 23 — Large-PTX fast-tier JIT
Fast `-O1` first-compile tier + background re-JIT to `-O3`; parallel module
compilation; measure & cap p99 first-call latency on multi-MB modules.

### Track 24 — PyTorch end-to-end validation harness — ✅ done (2026-06-10)
**File**: `tests/integration/pytorch_validation.py` (CTest `PyTorchValidation`).
**Constraint found**: this host has no NVIDIA driver (`libcuda.so.1` absent), so
torch+cu130 cannot enable its CUDA backend — running torch's *own* kernels on
VGRE would require faking the driver. Instead the harness uses **PyTorch as the
trusted oracle**: torch (CPU) generates inputs and the reference matmul, and the
SAME GEMM is executed through VGRE's exported cuBLAS C API via `ctypes`
(`cudaMalloc`/`cudaMemcpy`/`cublasSgemm_v2`, with the row-major→column-major
transpose mapping handled). Across 5 shapes incl. a 128×256×512 FFN slice, VGRE
matches PyTorch to ~1e-8 relative error. Self-SKIPs if torch/numpy are absent.
**Acceptance met**: VGRE reproduces PyTorch numerics through the public GPU API.
(A full vLLM/torch-on-VGRE run needs a driver shim — out of scope for this host.)

### Track 25 — iGPU transpiler OpenCL-C compile check — ✅ done (2026-06-10)
**Files**: `tests/integration/test_igpu_transpile.cpp`, public
`IGPUOpenCLExecutor::transpileToOpenCL`, transpiler fixes in
`src/runtime/igpu_opencl_executor.cpp`.
The test transpiles CUDA kernels (element-wise, warp-shuffle, ballot/vote,
dynamic shared memory) and compiles each with `clang -x cl -cl-std=CL1.2`
(Intel sub-group intrinsics stubbed so the structure validates without an Intel
runtime; SKIPs cleanly if the host clang lacks OpenCL). **It found and I fixed
three real transpiler bugs that emitted uncompilable OpenCL**:
1. Dynamic shared memory: the `__local TYPE* NAME` parameter was never added —
   the param-insertion searched `__kernel` from the start and hit the
   `#define __global__ __kernel` preamble instead of the kernel; now it scans
   backwards from the declaration to the enclosing kernel.
2. `unsigned*` / `unsigned int*` kernel pointers were left without an address
   space (the type list omitted `unsigned`) → invalid kernel arg.
3. The inserted `__local` shared-mem pointer also got `__global` prepended
   (`__local __global TYPE*`); the `__global` pass is now qualifier-aware
   (std::regex has no lookbehind, so it scans the preceding token by hand).
4/4 transpiled kernels now compile; the 3 iGPU/OpenCL tests stay green.

### Track 26 — Golden numerical-vector suite — ✅ done (2026-06-10)
`tests/api/test_golden_vectors.cpp` (7/7): fixed inputs checked against
INDEPENDENT hand/closed-form golden outputs (not VGRE's own naive reference):
cuFFT C2C forward vs the exact DFT of δ→[1,1,1,1], DC→[4,0,0,0], and a cosine
peaking 4 at bins 1&7; cuBLAS Sgemm of two diagonals (layout-independent golden
diag(10,21)); cuRAND uniform (in-range, mean≈0.5, var≈1/12) and normal(0,1)
(mean≈0, var≈1). A cuDNN conv golden can be added later, but the GEMM/FFT/RNG
oracles already give the consolidated tight-tolerance suite §6.3 asked for.

### Track 27 — perf_event_open proxy counters — ✅ core already present (2026-06-10)
**Finding**: the IPC half is already implemented. `cupti_shim.cpp` has a
`HwPmuSampler` (RAII, thread-local) that opens `perf_event_open(PERF_TYPE_HARDWARE,
PERF_COUNT_HW_INSTRUCTIONS)` with `ENABLE`/`RESET`/`read` and **auto-falls back to
the RuntimeProfiler instruction-mix proxies** when perf is unavailable — i.e. it
auto-detects rather than needing the `VGRE_USE_PERF_COUNTERS` opt-in the original
design proposed (auto-detect is strictly better). macOS uses thread CPU time.
**Remaining increment (not added)**: real L1D/LL cache hit-rate events
(`PERF_TYPE_HW_CACHE`) to back `l1_global_load_hit_rate`. Deliberately NOT added
here: this host has `perf_event_paranoid=4`, so perf-event reads cannot be
exercised at all — adding unverifiable counter code into the (passing, complex)
CUPTI shim would be speculative. Left as a CI/host-with-perf task.
**Status**: the perf_event_open IPC counter + documented proxy fallback exist and
satisfy the core of §5.6; cache-event proxies await a host where perf is readable.
