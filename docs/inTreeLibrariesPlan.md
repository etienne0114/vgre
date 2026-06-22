# VGRE In-Tree Libraries & Models Program

**Created**: 2026-06-22
**Purpose**: Stop waiting on external blockers (GPU hardware, gated checkpoint downloads,
system BLAS/FFTW). Build the missing pieces *from scratch, in-tree* so VGRE becomes a
**self-contained, lightweight, pip-installable CPU runtime that trains and serves its own
~100M-parameter model with no external math libraries and no gated downloads.**

This is the project's mission made literal: *use the resources we have (commodity CPU +
SIMD + the existing JIT/cluster) to build something extraordinary*, rather than depending on
complex external features to reach the advanced ones.

## Decisions (locked 2026-06-22)

| Decision | Choice |
|----------|--------|
| Lead build | **In-tree SIMD GEMM/BLAS** (prerequisite for fast training + inference) |
| Own model | **~100M-parameter** decoder LM, trained in-tree on public-domain text |
| Ship target | **Python wheel** (`pip install vgre`) |

## Why these, in this order

- The "82× GEMM" today hard-wraps system `cblas_sgemm`; with no OpenBLAS it drops to a *scalar*
  cache-tiled loop. So *fast* currently *requires an external dependency*. An in-tree SIMD
  micro-kernel makes GEMM fast **and** dependency-free — and **everything downstream (attention,
  autograd, the 100M training run) rides on it**, so it must come first.
- VGRE has **no autograd / optimizer / trainer** — the XLA stack is forward-only. That is the real
  reason the project leans on gated checkpoints: it cannot make its own. Building training unlocks a
  genuinely **own, real model** with zero external download.
- A **pip wheel** is the lowest-friction way for ML users to adopt a CPU runtime.

---

## Phase 1 — VGRE-GEMM: in-tree high-performance BLAS  ← LEAD

> **Status (2026-06-22): fp32 path DONE.** `src/xla/gemm/intree_gemm_f32.cpp` — BLIS-style
> packed, register-blocked **6×16 AVX2/FMA** micro-kernel with runtime ISA dispatch
> (`avx512`/`avx2`/`scalar`) and a portable scalar fallback. Wired as the **default** path in
> `blas_gemm.cpp` (`gemm_rows`); system cblas now only used when explicitly built with
> `VGRE_PREFER_SYSTEM_BLAS`. Verified: norm-wise error **~2e-7 vs a double-precision ground
> truth** (machine-epsilon fp32), **~35–44× faster than the naive triple loop** on a 4096³ GEMM
> (milestone **L2** criterion ≥10×), `blas_available()` reports the in-tree path, and the full
> XLA/model regression (incl. end-to-end JAX & TensorFlow StableHLO) passes with no change.
> The test (`tests/xla/test_blas_gemm.cpp`) was upgraded to a double-precision ground truth +
> norm-wise metric (the correct way to score a numerical GEMM).
>
> **Status (2026-06-22): bf16 path DONE.** `gemm_bf16_rows` (bf16 storage, f32 accumulation):
> the blocking driver is now templated on operand storage type, widening bf16→f32 only inside
> the cache-resident pack panels — so the big operands stay bf16 (half the memory/bandwidth) and
> the proven f32 micro-kernel is reused unchanged. Verified vs a double ground truth from the same
> bf16-rounded inputs (`tests/xla/test_gemm_bf16.cpp`, norm-wise error < 1e-4 = the f32-vs-double
> accumulation gap). This is the matmul training will run on.
>
> **Remaining in this phase:** fp16/int8/fp64 storage variants (trivial via the same template +
> `to_f32` overload / a VNNI int8 micro-kernel); an AVX-512 (16×N) micro-kernel; shared-Bpack
> threading to avoid per-thread repack; auto-tuned block sizes. Deferred — the autograd/trainer
> (Phase 2/3) is the higher-value next step toward the model.

**Goal**: replace both the `cblas_sgemm` fast path and the scalar fallback in
`src/xla/blas_gemm.cpp` with our own SIMD micro-kernel. Target ≥10× over the scalar fallback and
within ~1.5–2× of OpenBLAS single-threaded; scale to all cores via the existing `ThreadPool`.

**Design (BLIS-style, register-blocked + packed):**
- **Pack** A into `MR×KC` panels and B into `KC×NR` panels — contiguous, 64-byte-aligned, folding
  in `transA/transB` so the micro-kernel always sees the natural layout (no physical transpose at
  the call site).
- **Micro-kernel**: register-blocked `MR×NR` accumulation in vector registers using FMA.
  - AVX2/FMA fp32: `6×16` (6 rows × 2 `__m256` columns), the workhorse on this hardware
    (machine reports `AVX512=FALSE AVX2=TRUE`).
  - AVX-512 fp32: `14×32` / `16×32` when available.
  - Scalar reference micro-kernel for portability / non-x86.
- **Cache blocking**: `KC` sized to keep a B panel in L2, `MC` for A in L2/L3, `NC` for L3.
- **Runtime dispatch** on CPU caps (reuse `runtime::VectorEngine::getCapabilities()`), so one binary
  picks AVX-512 ▸ AVX2 ▸ scalar at load.
- **Threading**: keep the existing row-block / batch split over `ThreadPool` (already in
  `gemm_f32`); the micro-kernel slots into `gemm_rows`.

**Dtypes** (in order): fp32 → bf16 (AMX path already exists in `vector_engine_amx`) → fp16 → int8
(VNNI exists) → fp64. Each matches the existing `gemm_*` API surface.

**Wire-in**: a new `VGRE_USE_INTREE_GEMM` (default ON) routes `gemm_rows` through VGRE-GEMM;
system BLAS stays available behind `VGRE_USE_LAPACK` purely for A/B benchmarking. The goal is to
make the **default, no-dependency** build fast.

**Tests / bench**:
- Correctness vs a naive reference across shapes (incl. non-multiples of MR/NR), both transposes,
  batched — new `tests/xla/test_gemm.cpp`.
- Perf benchmark vs scalar fallback and (if present) cblas — `tests/xla/bench_gemm.cpp`.
- Regression: GPT-2 inference (existing path) must match bit-for-bit and not slow down.

**Deliverable**: `src/xla/gemm/` (`pack.cpp`, `microkernel_avx2.cpp`, `microkernel_avx512.cpp`,
`gemm.cpp`) + tests; `blas_gemm.cpp` calls into it.

---

## Phase 2 — VGRE-Autograd: reverse-mode autodiff

> **Status (2026-06-22): core DONE.** `include/vgre/xla/autograd.h` + `src/xla/autograd/autograd.cpp`
> — a tape-based ("define-by-run") reverse-mode engine over dense fp32 tensors. `backward()` does an
> iterative post-order DFS topo-sort and runs each node's local backward closure in reverse. The
> heavy op (`matmul`) runs on the in-tree GEMM, so training rides the same fast, dependency-free path
> as inference. Ops implemented + finite-difference-verified: `matmul`, `add` (same-shape + bias
> broadcast), `mul`, `scale`, `relu`, `gelu`, `silu`, `rms_norm`, `embedding`, fused
> `softmax_cross_entropy`, `mean`. All backward rules checked vs central finite differences in
> `tests/xla/test_autograd.cpp` (worst relative error ~2e-4, incl. the full embedding→matmul→CE LM
> loss).
>
> **Remaining in this phase:** `rope`, multi-head causal `attention` (built from the above), and
> `layer_norm` — then the transformer block is fully differentiable.

## Phase 3 — VGRE-Train: optimizers + loop + data

- Optimizers: SGD(+momentum), Adam, **AdamW**; global grad-norm clipping; cosine LR + warmup.
- **Mixed precision**: bf16 compute / fp32 master weights (halves memory, leans on AMX/the new GEMM).
- **Gradient checkpointing** to fit 100M activations in RAM.
- Data pipeline: a **public-domain corpus** (Project Gutenberg / TinyStories / enwik8) + the
  existing exact BPE tokenizer; sharded streaming, packed to context length.
- Checkpointing through the existing **safetensors / GGUF writers**, so trained weights load in the
  current inference path unchanged.
- Scale-out: reuse the built **tensor/pipeline parallelism + GSPMD + RDMA collectives** to spread a
  100M run across cluster CPUs (data + ZeRO-style optimizer-state sharding).

## Phase 4 — VGRE-LM: the ~100M model

- Architecture: Llama-style decoder — RMSNorm, RoPE, SwiGLU, GQA — ~100M params
  (e.g. 12 layers × d_model 768, or 16 × 640).
- Train in-tree to a **real, coherent checkpoint** on public-domain data; ship the checkpoint (or a
  fully-offline, reproducible recipe).
- Realism: a small `~10M` config trains in CPU-minutes as the CI/default model; the `~100M` target
  is a documented multi-core / multi-node recipe with expected wall-clock. No external download at
  any size.

## Phase 5 — Packaging: the pip wheel

- Thin Python package over the existing C-API (`vgre_c_api`) via pybind11/cffi exposing: init,
  malloc/memcpy, register/launch CUDA kernel, **and** tensor / train / generate.
- Bundle `libvgre` + `libvgre_cudart`; vendor only the needed LLVM runtime components (or static
  link) to keep the wheel lean; `manylinux` build via **cibuildwheel**.
- Acceptance: `pip install vgre && python -c "import vgre; ..."` runs a CUDA kernel, trains the tiny
  model, and generates text — **fully offline**.

---

## Cross-cutting rules

- Everything behind feature flags; `VGRE_MINIMAL` stays buildable; the 281/281 suite never regresses.
- Each phase gated by correctness tests **and** a benchmark before the next begins.
- No new external dependency may enter the default build — that is the whole point.

## Open risks

| Risk | Mitigation |
|------|------------|
| ~100M CPU training wall-clock | bf16 + the new GEMM + multi-node cluster; ship ~10M by default, ~100M as a cluster recipe |
| Wheel size from LLVM | vendor only required LLVM libs / static-strip; measure and document |
| Numerical drift in bf16 training | fp32 master weights + loss-scaling; finite-diff gated autograd |
