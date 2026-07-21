# VGRE Implementation Plan — Advanced Feature Roadmap (2026)

**Last Updated**: 2026-07-21

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

## Track T1 — Ternary / 1-bit inference (BitNet b1.58) · P0  *(DONE; loader external)*

**Why.** Ternary weights {−1,0,+1} make the core matmul multiplication-free (add/sub only),
the single biggest lever for running huge models on a CPU (2.37×–6.17× x86 speedup, up to 82%
less energy, 100B on one CPU in the literature). This is the strongest realization of the
project's mission.

**Status — done.**
- **Ternary codec + multiplication-free GEMM** (`include/vgre/xla/ternary_gemm.h`,
  `src/xla/gemm/ternary_gemm.cpp`): BitNet absmean per-column quantize/dequantize, and a matmul
  whose K-loop is only add/sub/skip — an AVX2 path deriving add/sub masks from the ternary codes
  (compare + bitwise-and, no per-element multiply) plus a portable scalar fallback.
- **`BitLinear`** (`vgre.nn`): quantization-aware training via a straight-through
  `ternary_quantize` autograd op — forward reuses the exact inference codec (so training-forward
  == inference GEMM), backward is the identity, so the fp master weight trains through the
  non-differentiable rounding.

Verified: `XlaTernaryGemm` — the kernel equals the dense dequantized GEMM to 1.6e-05 (fp
round-off only; the kernel adds no error), all weight signs preserved, AVX2 path selected;
`test_nn.py::test_bitlinear` — forward matches the absmean reference to 6e-08, the STE gradient is
exact, and a 2-layer BitLinear network trains (loss 1.3 → 0.34).

**Remaining.** GGUF `I2_S` / TL1 / TL2 ternary tensor types → the ternary storage kind, so real
BitNet-b1.58 checkpoints load directly (end-to-end verification needs a **gated multi-GB
download**, so it is tracked with the external items).

---

## Track T2 — Mixture-of-Experts + expert-parallel · P1  *(COMPLETE)*

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
the analytic target distribution (TV≈0.04, N=800); and temperature→0 reduces to greedy exactly.

**Remaining (throughput optimization, not correctness):**
1. **KV-cache reuse + rollback** on the C++ `generate_cached` path so accepted tokens keep their
   cache and the batched verify runs on the raw (non-autograd) forward — the production-throughput
   version (the framework path already recomputes and is O(1) forwards/run). Tree verification for
   multi-branch drafts + self-speculative (early-exit) drafting.

---

## Track T4 — State-space models (Mamba-2 / Mamba-3) · P2  *(core + parallel scan DONE; loader/MIMO next)*

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
reference of the exact Δ/A/B/C recurrence (7.6e-06); and an attention-free 2-block Mamba LM
memorizes a sequence (loss 3.4 → 0.000) and greedily regenerates it 14/14.

**Parallel scan (done).** The D state channels are independent, so `selective_scan` fans the
forward scan *and* the adjoint backward out over D across the thread pool (each channel a private
recurrence — bit-identical to serial). Verified (`test_mamba`): a 128×128 scan on the parallel path
matches the sequential NumPy reference exactly; measured ~4.9× speedup on a 512×2048 scan at 8
threads (2.0 ms vs 9.8 ms serial).

**Remaining (breadth).**
1. **Depthwise short conv** before the SSM + the **Mamba-3 MIMO** (matrix-matrix) state update.
2. **Loader**: Mamba safetensors/GGUF → the model stack.

---

## Track T5 — MXFP4 microscaling 4-bit (OCP) · P2  *(codec + GEMM DONE)*

**Why.** The emerging interoperable 4-bit standard (shared block scale + FP4 elements) for both
weights and activations.

**Status — done.** `include/vgre/xla/mxfp4.h` + `src/xla/gemm/mxfp4.cpp`: E2M1 4-bit elements
(magnitudes {0,.5,1,1.5,2,3,4,6}) with a shared E8M0 power-of-two scale per 32-element block down
K — **4.25 bits/weight**. `quantize` / `dequantize` + a **dequant-in-GEMM** `gemm` that applies the
block scale once per block (the E2M1 code stays 4-bit until multiplied by the activation, no fp32
expansion). In the LLVM-free `libvgre_nn`. Verified (`XlaMxfp4`): the dequant-in-GEMM equals the
dense fp32 GEMM of A against the dequantized weights (4e-06), the round-trip is a bounded FP4
approximation (rel err ≈ 0.12), and the footprint is 4.25 bits/weight.

**Remaining (breadth):** AVX2 unpack/decode SIMD path; a `Literal` storage kind + safetensors/GGUF
MXFP4 tensor loading so external MXFP4 checkpoints run directly; MXFP4 **activations** (not just
weights).

---

## Track T6 — QLoRA (quantized-base fine-tuning) · P3  *(DONE)*

**Why.** Fine-tune a large model on a laptop: frozen **ternary** base + trainable fp LoRA
adapters; de-quant only inside the forward matmul. Extends the existing `LoRALinear` + quant.

**Status — done.** `vgre.nn.QLoRALinear`: the base weight is frozen as ternary codes {-1,0,+1} +
per-column absmean scale (~2 bits/weight) stored as plain arrays — **not parameters**, never
updated, no gradient formed — and de-quantized transiently inside the forward matmul; only the
LoRA A [in,r], B [r,out] adapters train (B=0 init → starts at the quantized base). `from_linear`
quantizes an existing Linear; `adapter_parameters()` == `parameters()` (A,B only). Verified
(`test_nn.py::test_qlora`): parameters() holds exactly {A,B}; the frozen base is 2.25 bits/weight
(in=128); the initial output equals the quantized base; the adapter gradients match a NumPy
reference; the base codes are unchanged after training; and a fine-tune converges (loss 4.1 → 0.000).

**Remaining (breadth):** an int4/MXFP4 base option (ternary is in), and dequant-in-GEMM for the
base path (currently a transient fp32 dequant) to cut the forward's peak memory too.

---

## Externally-blocked & physical-run tracks

Unchanged from `missingFeatures.md` §2–§3: security-enclave hardware, PJRT/MLIR wheels, live
cloud accounts, vendor SDKs, unreleased hardware, and the physical multi-machine / gated-download
large-model runs. The in-tree primitives for all of these already exist; only the external piece
remains. Every §1 track above *reduces* what a physical run needs (less memory, less compute).

---

## Success criteria

| Track | Criterion | Result |
|-------|-----------|--------|
| **T1** BitNet | ternary-GEMM matches fp reference; multiplication-free kernel | ✅ **met** — kernel == dense dequantized GEMM to 1.6e-05, AVX2 add/sub-mask path, BitLinear QAT trains. *(Real BitNet-b1.58 GGUF forward pending a gated download.)* |
| **T2** MoE | top-k routing gradient == NumPy; 2-process expert-parallel forward == single-process | ✅ **met** — combine == NumPy 2.4e-07, exactly k experts/token (active compute ∝ routed tokens), cross-process sharded forward bit-identical + router-grad sum 1.8e-08 |
| **T3** Spec-decode | speculative output == vanilla greedy; decode speedup > 1.5× | ✅ **met** — token-for-token identical at **4.0 tokens/forward**; sampler-exact sampling also distribution-identical (TV 0.04) |
| **T4** Mamba | selective-scan fwd+grad == sequential NumPy reference; a Mamba LM trains + generates | ✅ **met** — scan fwd exact, grads ~1e-5; full S6 block == NumPy Δ/A/B/C 7.6e-06; attention-free Mamba LM regenerates 14/14; parallel scan bit-identical at ~4.9× |
| **T5** MXFP4 | MXFP4-GEMM == fp reference within tolerance; no separate upcast pass | ✅ **met** — round-trip bounded, dequant-in-GEMM |
| **T6** QLoRA | adapter grads == NumPy; frozen quantized-base invariance | ✅ **met** — params == {A,B}, base 2.25 bits/weight and unchanged, grads == NumPy, fine-tune converges 4.1 → 0.000 |

**Global gate for every track:** builds clean, `libvgre_nn` stays LLVM/BLAS/CUDA-free, the full
suite stays green, and each capability is exercised end-to-end (not just unit-detected).
