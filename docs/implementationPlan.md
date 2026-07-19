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

## Track T2 — Mixture-of-Experts + expert-parallel · P1  *(routing DONE; sparse/parallel next)*

**Why.** Sparse models activate only a few experts per token, so the *active* compute stays small
even at frontier scale — a natural fit for a cluster of ordinary CPUs, reusing our collectives.

**Status.** **Done:** a learned **top-k router** + **compute-sparse** gate-weighted combine as
`vgre.nn.MoELayer`. New `index_select` (gather) / `index_add` (scatter-add) autograd ops let each
expert's FFN run **only on the tokens routed to it** (active expert compute ∝ routed tokens, not
tokens×experts) while staying fully differentiable and trainable. Verified
(`test_nn.py::test_gather_scatter`, `test_moe`): gather/scatter fwd+bwd correct, the sparse
dispatch matches an independent dense NumPy reference to 2.4e-07, exactly k experts fire per
token, and an MoE network trains (loss 1.9 → 0.001).

**Also done:**
- **Expert-parallel** (`MoELayer(expert_parallel=True)`): experts are sharded across ranks
  (contiguous blocks, seeded by global index); each rank computes its local experts' partial and
  `all_reduce` sums the partials into the full output — the row-parallel trick over the
  differentiable collective. Verified across **real OS processes** (`PythonNnExpertParallel`):
  the 2-rank sharded forward is bit-identical to the single-process run, the replicated router's
  per-rank partial gradients sum to the full router gradient (1.8e-08), and sharded expert
  gradients match exactly.
- **Load-balancing aux loss** (`MoELayer.aux_loss()`, Switch-style ∝ Σ_e f_e·P_e) — a positive
  differentiable scalar; training with it added still converges (`test_moe`).

**Integration (done):** `TransformerBlock(moe_experts=…, moe_top_k=…, expert_parallel=…)` makes MoE
a drop-in for the dense FFN (2-D and 3-D inputs), with a per-block `aux_loss()`. Verified end-to-end
(`test_nn.py::test_moe_lm`): a 2-block GPT with MoE FFNs (embedding → attention → MoE-FFN → head)
overfits a sequence (loss 3.7 → 0.001) and greedily regenerates it token-for-token (12/12).

**T2 is complete** for the in-tree, single-/multi-process CPU target. (A distributed MoE-LM
*training* run over expert-parallel across machines is covered by the physical-cluster item in §3.)

---

## Track T3 — Speculative decoding · P1  *(algorithm DONE; C++ KV-cache throughput next)*

**Why.** 2×–10× faster decode on CPU with **identical** output distribution — pure latency win.

**Status — the decoding algorithm is complete and proven** over any `model(ids)→[T,vocab]`:
- **Greedy speculative** (`vgre.nn.speculative_generate`): the drafter proposes up to k tokens; the
  target verifies them in **one** forward and accepts the longest prefix equal to its own greedy
  argmax + one correction/bonus token → output provably identical to greedy decoding.
- **Sampler-exact speculative sampling** (`speculative_sample`, Leviathan/Chen): accept token d with
  prob min(1, q(d)/p(d)), else resample from the residual norm(relu(q−p)); bonus from q when all
  accepted → output **distribution-identical** to sampling the target at the same temperature/top_p.
- **Drafters**: `ngram_draft_fn` (zero-cost prompt-lookup) and `model_draft_fn` (a smaller draft
  model). `greedy_generate` is the baseline.

Verified (`test_nn.py::test_speculative_decoding`): on a memorized sequence, ngram speculative ==
greedy token-for-token at **4.0 tokens/forward**; draft-model speculative also == greedy; with an
untrained (diffuse, q_max≈0.27) target the sampler-exact empirical first-token distribution matches
the analytic target distribution (TV≈0.03, N=1500); and temperature→0 reduces to greedy exactly.

**Remaining (throughput optimization, not correctness):**
1. **KV-cache reuse + rollback** on the C++ `generate_cached` path so accepted tokens keep their
   cache and the batched verify runs on the raw (non-autograd) forward — the production-throughput
   version (the framework path already recomputes and is O(1) forwards/run). Tree verification for
   multi-branch drafts + self-speculative (early-exit) drafting.

---

## Track T4 — State-space models (Mamba-2 / Mamba-3) · P2  *(core DONE; parallel scan + loader next)*

**Why.** Linear-time, constant-memory-per-step sequence modeling — a decisive CPU advantage at
long context (no growing KV cache), and an architecture class beyond transformers.

**Status — the SSM core is complete and proven.**
- **Selective scan** (`selective_scan(a,b)` autograd op): the input-dependent recurrence
  h_t = a_t⊙h_{t-1} + b_t with the exact adjoint backward (reverse recurrence
  G_t = g_t + a_{t+1}·G_{t+1}, dL/db_t = G_t, dL/da_t = G_t·h_{t-1}). Exposed as
  `vgre.nn.selective_scan`.
- **Full selective Mamba (S6) block** (`vgre.nn.SSMBlock` / `MambaBlock`): a real
  d_state-dimensional diagonal SSM with input-dependent Δ/B/C —
  Δ_t = softplus(dt·x), Ā_t = exp(Δ_t⊙A) with A = −exp(A_log), h_t[:,n] = Ā_t[:,n]⊙h_{t-1}[:,n] +
  Δ_t⊙B_t[n]⊙u_t, y_t = Σ_n C_t[n]⊙h_t[:,n] + D⊙u_t. Built from `exp`/`softplus` (new autograd ops)
  + `selective_scan` per state channel; a pre-norm residual, attention-free sequence mixer.

Verified (`test_nn.py::test_mamba`): the scan forward matches a NumPy reference exactly and its
gradients match finite differences (~1e-5); the full SSM block matches an independent NumPy
reference of the exact Δ/A/B/C recurrence (3.8e-06); and an attention-free 2-block Mamba LM
memorizes a sequence (loss 3.4 → 0.000) and greedily regenerates it 14/14.

**Remaining (perf / breadth).**
1. **Parallel scan**: replace the sequential recurrence with the associative (Blelloch)
   prefix-scan, threaded + SIMD, for throughput on long sequences (the adjoint parallelizes too).
2. **Depthwise short conv** before the SSM + the **Mamba-3 MIMO** (matrix-matrix) state update.
3. **Loader**: Mamba safetensors/GGUF → the model stack.

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
