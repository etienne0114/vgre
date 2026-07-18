# VGRE Implementation Plan — Advanced Feature Roadmap (2026)

**Last Updated**: 2026-07-18

The original Phase-4 enterprise roadmap (50 tracks) **and** the large-model programme (L1–L5)
are delivered to their software-implementable core — every track buildable to the *real,
no-stub* standard without external hardware/accounts is done, tested, and committed (see git
history and `missingFeatures.md`). This plan now describes the **next competitive frontier**:
six new capabilities that make VGRE *intelligent, modern, and lightweight*, each built **from
scratch, dependency-free**, extending code paths that already exist. Priority order follows the
core mission — **run/train large models on CPUs + clusters with no GPU, staying lightweight.**

Design rules (unchanged, non-negotiable):
- **No stubs / mocks / heuristics / placeholders.** Every method is real and fully functioning.
- **From scratch, in-tree.** No new third-party runtime dependency; ternary/MoE/SSM/spec-decode
  kernels are our own SIMD C++.
- **Verified numerically.** Each op ships with a test comparing against an independent reference
  (NumPy / a known checkpoint), executed end-to-end.
- **Lightweight by construction.** New kernels live in `libvgre_nn` (LLVM-free) where they belong
  so the Python wheel stays small.

---

## Track T1 — Ternary / 1-bit inference (BitNet b1.58) · P0

**Why.** Ternary weights {−1,0,+1} make the core matmul multiplication-free (add/sub only),
the single biggest lever for running huge models on a CPU (2.37×–6.17× x86 speedup, up to 82%
less energy, 100B on one CPU in the literature). This is the strongest realization of the
project's mission.

**Steps.**
1. **Ternary codec** (`src/xla/quant/ternary.{h,cpp}`): pack weights as 2-bit {−1,0,+1} with a
   per-group fp16 scale; encode/decode + a `Literal` storage kind next to bf16/int4/int8.
2. **Multiplication-free ternary GEMM** (`src/xla/gemm/ternary_gemm.cpp`): activation × ternary
   accumulate using masked add/sub; AVX2/AVX-512 (`_mm256_sign_epi8`-style) + scalar fallback,
   threaded via the existing pool. Add a **T-MAC-style LUT** kernel (precompute activation
   partial sums per packed byte) for sub-2-bit throughput.
3. **`BitLinear`** in the autograd/model stack (forward uses ternary-GEMM; training path keeps a
   latent fp master weight with straight-through estimation for the quantizer).
4. **Loader**: GGUF `I2_S` / TL1 / TL2 ternary tensor types → the ternary `Literal`, so real
   BitNet-b1.58 checkpoints load directly.
5. **Tests**: `XlaTernaryGemm` (ternary-GEMM == fp reference within scale tolerance);
   `PythonBitLinear` (a BitLinear MLP trains + matches a NumPy STE reference); a real
   BitNet-b1.58 GGUF forward matching the reference logits (skip-77 without the file).

---

## Track T2 — Mixture-of-Experts + expert-parallel · P1

**Why.** Sparse models activate only a few experts per token, so the *active* compute stays small
even at frontier scale — a natural fit for a cluster of ordinary CPUs, reusing our collectives.

**Steps.**
1. **Router/gate** (`src/xla/model/moe.{h,cpp}`): top-k softmax gating with load-balancing aux
   loss; differentiable in the autograd engine.
2. **Sparse dispatch/combine**: gather tokens per expert, run each expert's FFN, scatter-combine
   weighted by gate scores (no dense masking waste).
3. **`MoELayer`** integrated into the transformer block (drop-in for the dense FFN).
4. **Expert-parallel**: place experts on different cluster ranks; route tokens with an
   **all-to-all** built on the existing TCP/RDMA collective layer; combine with all-reduce.
5. **Tests**: `PythonMoE` (top-k routing gradient matches NumPy; dense-equivalent when k==E);
   `PythonMoEExpertParallel` (2-process expert-sharded forward == single-process reference).

---

## Track T3 — Speculative decoding · P1

**Why.** 2×–10× faster decode on CPU with **identical** output distribution — pure latency win.

**Steps.**
1. **Drafting** (`src/xla/generation.cpp`): pluggable drafters — n-gram/prompt-lookup (zero extra
   model), self-speculative (early-exit layers), and small-draft-model.
2. **Tree verification**: build a token tree, verify with **one** batched forward of the full
   model over paged KV (`KVCacheManager` already supports paged attention), accept the longest
   matching prefix; roll the KV back to the accepted length.
3. **Sampler-exact acceptance** so greedy and temperature/top-p output is provably unchanged.
4. **Tests**: `XlaSpeculativeDecode` (speculative output == vanilla greedy token-for-token on the
   in-tree LM; report mean accepted length / speedup).

---

## Track T4 — State-space models (Mamba-2 / Mamba-3) · P2

**Why.** Linear-time, constant-memory-per-step sequence modeling — a decisive CPU advantage at
long context (no growing KV cache), and an architecture class beyond transformers.

**Steps.**
1. **Selective scan** (`src/xla/autograd/scan.cpp`): a **parallel associative prefix-scan**
   (Blelloch, threaded + SIMD) for the input-dependent SSM recurrence, with its reverse-mode
   backward (the adjoint scan).
2. **Mamba block**: input/gate projections, depthwise short conv, selective SSM, output gate —
   using the MIMO (matrix-matrix) form so it reuses the existing GEMM.
3. **Loader**: Mamba safetensors/GGUF → the model stack.
4. **Tests**: `XlaSelectiveScan` (forward + gradient vs a NumPy sequential reference);
   `PythonMamba` (a small Mamba LM trains and generates).

---

## Track T5 — MXFP4 microscaling 4-bit (OCP) · P2

**Why.** The emerging interoperable 4-bit standard (shared block scale + FP4 elements) for both
weights and activations.

**Steps.**
1. **MXFP4 codec** (`src/xla/quant/mxfp4.{h,cpp}`): E2M1 elements + shared E8M0 block scale;
   encode/decode + `Literal` kind.
2. **Dequant-in-GEMM** path so MXFP4 weights run without a separate upcast (mirrors the int4
   path already in `blas_gemm` / `quant_gemm`).
3. **Tests**: `XlaMxfp4` (round-trip error bound; MXFP4-GEMM == fp reference within tolerance).

---

## Track T6 — QLoRA (quantized-base fine-tuning) · P3

**Why.** Fine-tune a large model on a laptop: frozen **int4/ternary** base + trainable fp LoRA
adapters; de-quant only inside the forward matmul. Extends the existing `LoRALinear` + quant.

**Steps.**
1. `QLoRALinear`: quantized frozen base (int4 or ternary) + LoRA A/B; backward flows only into
   the adapters; de-quant fused into the base matmul.
2. Adapter save/load + merge-to-(quantized)-base.
3. **Tests**: `PythonQLoRA` (adapter gradients match NumPy; frozen-base invariance; N-step AdamW
   convergence on a quantized base).

---

## Externally-blocked & physical-run tracks

Unchanged from `missingFeatures.md` §2–§3: security-enclave hardware, PJRT/MLIR wheels, live
cloud accounts, vendor SDKs, unreleased hardware, and the physical multi-machine / gated-download
large-model runs. The in-tree primitives for all of these already exist; only the external piece
remains. Every §1 track above *reduces* what a physical run needs (less memory, less compute).

---

## Success criteria

| Track | Criterion | Target |
|-------|-----------|--------|
| **T1** BitNet | ternary-GEMM matches fp reference; real BitNet-b1.58 GGUF forward matches reference logits | multiplication-free kernel, ≥2× vs int8 on the same matmul |
| **T2** MoE | top-k routing gradient == NumPy; 2-process expert-parallel forward == single-process | active-compute ∝ k/E; cluster expert sharding verified |
| **T3** Spec-decode | speculative output == vanilla greedy token-for-token | measured decode speedup > 1.5× on the in-tree LM |
| **T4** Mamba | selective-scan fwd+grad == sequential NumPy reference | linear-time; a Mamba LM trains + generates |
| **T5** MXFP4 | MXFP4-GEMM == fp reference within tolerance | weights run with no separate upcast pass |
| **T6** QLoRA | adapter grads == NumPy; frozen quantized-base invariance | fine-tune on a quantized base, adapters converge |

**Global gate for every track:** builds clean, `libvgre_nn` stays LLVM/BLAS/CUDA-free, the full
suite stays green, and each capability is exercised end-to-end (not just unit-detected).
