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

## Delivered — the whole program (Phases 1–5)

The five phases are **done and verified**; the ~100M-parameter, pip-installable,
train-and-serve-on-CPU model with no external math libraries is real (v0.1.0 wheels ship it).

| Phase | Delivered | Verified by |
|-------|-----------|-------------|
| **1 — VGRE-GEMM** (in-tree BLAS) | BLIS-style packed, register-blocked **6×16 AVX2/FMA** micro-kernel with runtime ISA dispatch (fp32 + **bf16** storage/f32-accum + weight-only **int8** GEMV); the **default** path in `blas_gemm.cpp` | `test_blas_gemm.cpp` (~2e-7 vs double truth), ~35–44× over the naive loop; serves fp32 / bf16 (½×) / int8 (¼×) |
| **2 — VGRE-Autograd** | tape-based reverse-mode autodiff over the in-tree GEMM: matmul/add/mul/relu/gelu/silu/rms_norm/layer_norm/embedding/rope/attention/**flash-attention**/conv2d/max_pool2d/softmax-CE | `test_autograd.cpp` / `test_conv.cpp` — all backward rules finite-diff verified; a full transformer block + a CNN train |
| **3 — VGRE-Train** | AdamW / SGD(+momentum), grad-norm clip, cosine-LR+warmup; `TokenStream` data pipeline; **safetensors** checkpoint I/O | `test_train.cpp` / `test_lm_pipeline.cpp` — trains a transformer end-to-end offline, save→reload identical logits, generates real Shakespeare |
| **4 — VGRE-LM** | configurable Llama-style decoder (RoPE + SwiGLU + RMSNorm), **KV-cached** generation, temperature/top-k/top-p/repetition sampling; `examples/train_lm.py` CLI | `test_model.cpp` — 85K→134M one `Config`; token-identical KV-cached decode; a 1.2M/600-step run drives val loss 6.06→0.09 |
| **5 — Packaging** | C-ABI + `vgre.LanguageModel`/`Tokenizer`; `build_wheel.sh` bundles the native libs into a self-contained wheel | `PythonLmBindings`; **v0.1.0 wheels** (Linux/macOS/Windows) verified: pip install → train an LM offline |

## Remaining (performance & breadth — the next release)

The correctness is done; what's left is **peak CPU performance** — the direct route to closing
the gap to a GPU (full analysis: [`zeroBurdenRoadmap.md`](zeroBurdenRoadmap.md) **§7**):

- **CPU tensor units**: an **AMX (bf16)** and **AVX-512-VNNI (int8)** micro-kernel in VGRE-GEMM
  (the AMX path exists in `vector_engine_amx`; make it the default bf16/int8 matmul) — the
  single biggest matmul speedup on modern CPUs (~4–16× over AVX-512 fp32).
- **AVX-512 (16×N) fp32 micro-kernel** + **auto-tuned block sizes** per CPU (cache-size-driven).
- **AVX2 unpack/decode** kernels for int8 / int4 / MXFP4 / ternary so dequant isn't the bottleneck.
- **fp16 / fp64 storage variants** (trivial via the same storage-templated driver).
- **Mixed-precision training** (bf16 compute / fp32 master weights) + **gradient checkpointing**
  to fit 100M activations in RAM; **ZeRO-style optimizer-state sharding** for a cluster run.
- **manylinux / cibuildwheel** redistributable wheels + **PyPI publish** (see `missingFeatures.md` §1.1).

---

## Cross-cutting rules

- Everything behind feature flags; `VGRE_MINIMAL` stays buildable; the full `ctest` suite never regresses (currently 399 with LLVM / 379 LLVM-free, 100% green).
- Each phase gated by correctness tests **and** a benchmark before the next begins.
- No new external dependency may enter the default build — that is the whole point.

## Open risks

| Risk | Mitigation |
|------|------------|
| ~100M CPU training wall-clock | bf16 + the new GEMM + multi-node cluster; ship ~10M by default, ~100M as a cluster recipe |
| Wheel size from LLVM | vendor only required LLVM libs / static-strip; measure and document |
| Numerical drift in bf16 training | fp32 master weights + loss-scaling; finite-diff gated autograd |
