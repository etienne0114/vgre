# VGRE Implementation Plan — Advanced Feature Roadmap (2026)

**Last Updated**: 2026-09-25

The Phase-4 enterprise roadmap, the large-model programme (L1–L5), and the six advanced ML
tracks (T1–T6) below are delivered to their software-implementable core (see git history and
`missingFeatures.md`).

> ## 🟢 Track Z — The Zero-Burden Engine · **✅ DELIVERED**
>
> Removing the one hard build burden (LLVM 18) by building the CUDA-C →
> machine-code pipeline **from scratch, in-tree**. Delivered: `-DVGRE_ENABLE_JIT=OFF`
> builds and runs with **no LLVM** (and `-DVGRE_ENABLE_OPENMP=OFF` on the in-tree
> thread pool alone); a from-scratch CUDA-C front-end feeds a four-tier CPU backend
> (PTX interpreter → compiled-fiber → native x86-64 JIT → Tier-2 SSA), each held
> bit-exact against the others; v0.1.0 ships self-contained LLVM-free wheels.
> **Full plan, execution-tier detail, and the next perf frontier:**
> [`zeroBurdenRoadmap.md`](zeroBurdenRoadmap.md) (see its **§7 — Closing the
> performance gap to a real GPU**, the current top forward-looking priority).

The **T1–T6 ML tracks** below were built **from scratch, dependency-free** in the
LLVM-free `libvgre_nn` and are **all delivered** — see the **Success criteria** table at the
end for the verified result of each, and [`missingFeatures.md`](missingFeatures.md) §1–§2 for
the narrow breadth remainders and the external-download items. Priority order follows the core
mission: **run/train large models on CPUs + clusters with no GPU, staying lightweight.**

Design rules (unchanged, non-negotiable): **no stubs / mocks / heuristics / placeholders**;
**from scratch, in-tree** (no new third-party runtime dependency); **verified numerically**
against an independent reference (NumPy / a known checkpoint), end-to-end; **lightweight** —
new kernels live in `libvgre_nn` so the wheel stays small.

| Track | What it delivered (mission lever) | What's left (breadth/perf) |
|-------|-----------------------------------|----------------------------|
| **T1** Ternary / BitNet b1.58 | multiplication-free ternary GEMM (add/sub masks) + BitLinear QAT — the strongest CPU lever | GGUF `I2_S`/TL1/TL2 loader (needs a gated BitNet checkpoint) |
| **T2** Mixture-of-Experts | top-k router + compute-sparse dispatch (`index_select`/`index_add`) + expert-parallel across ranks | complete (a physical multi-machine MoE-LM training run is the external item) |
| **T3** Speculative decoding | greedy + sampler-exact speculative decode (identical output), ngram/model drafters | tree verification; early-exit self-speculative drafting (throughput) |
| **T4** State-space (Mamba/S6) | selective-scan autograd + full S6 block + parallel scan (~4.9×), attention-free LM | Mamba-3 MIMO state update; safetensors/GGUF Mamba loader |
| **T5** MXFP4 microscaling | E2M1 + E8M0 block-scale codec + dequant-in-GEMM (4.25 bits/wt) | AVX2 SIMD unpack; MXFP4 activations |
| **T6** QLoRA | frozen ternary base (~2 bits/wt) + trainable LoRA adapters | int4/MXFP4 base option; dequant-in-GEMM for the base path |

**Performance frontier (all tracks).** These tracks *cut the work* (quantization, sparsity,
linear-time SSM, fewer forward passes), which is the largest lever for running GPU-sized models
on a CPU. The complementary work — removing the emulation tax and using the CPU's tensor units
(AMX/VNNI) + warp-SIMD codegen — is the **performance roadmap in
[`zeroBurdenRoadmap.md`](zeroBurdenRoadmap.md) §7**.

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
