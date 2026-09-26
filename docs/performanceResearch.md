# VGRE Performance Research — Algorithms, Real Problems, Business

**Last updated:** 2026-09-26

VGRE executes GPU workloads on the CPU. That single fact defines both the
**problem** (a CPU has ~10–50× less compute and far less memory bandwidth than a
datacenter GPU) and the **opportunity** (no GPU capex, runs anywhere, and most
real inference is *memory-bound*, where the CPU/GPU gap is much smaller than the
raw-FLOP gap suggests). This document tracks the algorithmic directions that move
VGRE from "correct" to "fast enough to be a product," each tied to a concrete
problem and a business use it unlocks.

Status legend: **✅ delivered** · **▶ in progress** · **◇ proposed / researching**.

---

## The core insight: win where the CPU can win

A modern server CPU has ~50–200 GB/s of memory bandwidth and 1–4 TFLOP/s of
vectorized FP32 — a GPU has ~10× the bandwidth and ~20–50× the FLOP/s. So VGRE
should **not** try to beat the GPU on dense FLOP-bound work. It should win where:

1. **The work is memory-bound** (LLM decode, elementwise/normalization chains,
   attention at long context) — here fusion and cache-blocking close most of the gap.
2. **The work can be made mul-free or low-bit** (ternary / int4 / int8) — this
   converts the FLOP-bound part into cheap integer add/sub the CPU does well.
3. **The workload is latency-tolerant and cost-sensitive** (CI, batch, edge,
   on-prem) — where "no GPU required" beats "fastest."

Every item below is an instance of one of these three levers.

---

## 1. Low-bit, mul-free arithmetic  ▶

**Problem.** Dense GEMM is the FLOP-bound heart of every model; on a CPU it is the
bottleneck. **Algorithm.** BitNet-b1.58 ternary weights {−1, 0, +1} turn `w·x`
into **sign-flip + accumulate** — no multiplies — and int8/int4 activation
quantization keeps the accumulators cheap. **Delivered:** the GGUF `I2_S` ternary
loader and a SubLN transformer forward run the real bitnet-b1.58-2B on CPU
(`quant.h`, `hf_loader.cpp`, `tests/xla/test_bitnet_subln.cpp`); int8/int4
weight-only inference and int8/int4 KV cache are wired (`model.h`,
`kv_quant.h`). The ternary GEMM is already mul-free and never materializes fp32
weights (it operates directly on the int8 codes, add/sub/skip — `ternary_gemm.cpp`).

**Measured (`bench_ternary_gemm`, this AVX2 machine).** Correctness is *exact* vs
an fp32 same-arithmetic reference (max\|rel\| = 0). Speed vs the in-tree dense fp32
GEMM depends on shape: at a small GEMM (M=16, N=K=256) ternary is **~5.1× faster**
(the dense kernel's packing overhead dominates), but at a large GEMM (M=32,
N=K=2560) it is **~0.42×** — *slower* — because the dense path is a cache-blocked
BLIS-style kernel (~24 GFLOP/s) while the ternary kernel is memory-bound on the
K×N int8 codes. **Delivered (▶):** two-level (M×N) cache blocking, bit-exact throughout —
M-blocking (compute the masks once per `k`, stream the codes once per **panel of 8
rows**) lifted the large-shape kernel ~40% (0.26× → 0.42×) and the small shape to
5.1×; adding N-tiling (an `mb×512` accumulator kept L1-resident across the `k`-loop)
moved it only marginally further (0.44×).

**The key measured insight.** That N-tiling barely helped is the finding: the
kernel is **ALU-bound, not memory-bound**, because **"mul-free" is not a compute win
on a CPU with FMA**. A dense fp32 GEMM does multiply+add in ONE `vfma` per element;
the ternary path replaces that single FMA with compare+compare+and+and+add+sub
(~6 ops) to avoid a multiply the FMA unit already did for free. So on CPU the
ternary/low-bit advantage is **memory, not arithmetic**: 4× smaller weights today
(int8 codes vs fp32), and the real lever is **packing 4 codes/byte (2-bit) → 16×
smaller weights**, which wins in the *memory-bound* regime that matters for LLM
inference — large models whose weights do not fit in cache, at low batch, where
weight bandwidth dominates. **Next (◇):** the 2-bit packed format + a kernel that
unpacks in-register, benchmarked at the memory-bound shapes (large K·N, small M)
where it should beat fp32; an AVX-512 VNNI int8-activation path on hardware that
has it.

**Business.** The honest, measured value proposition: ternary/low-bit on CPU lets
you **fit a much bigger model in the same RAM and move less weight bandwidth** —
so a 2–8B model *runs at all* on a commodity box with no GPU — rather than
"faster dense compute." That is still the single biggest "transform into a
product" lever (private on-prem inference), but the pitch is *capacity and cost*,
not raw FLOP/s — a distinction the benchmark made concrete rather than assumed.

## 2. Operator fusion + a graph scheduler  ◇

**Problem.** On the CPU, the memory round-trips *between* kernels (write an
activation to DRAM, read it back for the next op) cost more than the ops
themselves. **Algorithm.** Fuse elementwise → reduction → elementwise chains into
one pass that keeps tiles in L1/L2 (the classic "fuse the epilogue" and
"norm+residual+activation" fusions), driven by a small dataflow graph over the
HLO/PTX the engine already builds. VGRE already has the IR (VGRE-HLO,
StableHLO import) and a four-tier backend to lower fused regions into.

**Business.** 2–4× on memory-bound inference with zero model changes — a
transparent speedup for every user, and the difference between "usable" and "too
slow" for interactive CPU inference.

## 3. Cache-blocked attention (online softmax)  ✅ / ▶

**Problem.** Naïve attention materializes the T×T score matrix — O(T²) memory,
DRAM-bound at long context. **Algorithm.** FlashAttention-style online-softmax
tiling computes attention in O(T) memory, streaming K/V tiles through cache.
**Delivered:** `flash_attention` in the training forward + the KV-cached decode
path. **Next (◇):** a blocked, multi-threaded CPU flash-attention with an L2-sized
K/V tile and int8 KV on the fly, and paged-attention-aware tiling for the serving
cache.

**Business.** Long-context (32k+) inference on CPU without the memory blowing up —
enables document/RAG workloads on-prem.

## 4. Sparsity-aware execution  ◇

**Problem.** A large fraction of LLM activations (post-ReLU/SwiGLU) and many
weights are zero, yet dense kernels still pay for them. **Algorithm.** Structured
2:4 weight sparsity (skip half the MACs) plus dynamic activation-sparsity gating
(skip rows whose activation is ~0), both cheap to detect on the CPU. Combine with
item 1 (ternary already *is* ~46% zeros — the `code == 1` case — so a
sparsity-skipping ternary kernel is nearly free to add).

**Business.** Another ~1.5–2× on top of low-bit, compounding the "LLM on a laptop"
story.

## 5. Autotuning + an analytical cost model  ◇

**Problem.** The best tile size / execution tier depends on the exact CPU (cache
sizes, ISA) and the kernel shape; hand-tuning does not port. **Algorithm.** A
lightweight analytical roofline cost model (bytes moved vs FLOPs vs measured
bandwidth) chooses tile sizes and picks among the four tiers per kernel, with an
optional measured autotuning pass cached to disk (the JIT cache already exists).
The dashboard already reports measured bandwidth and an estimated speedup — feed
that back into the chooser.

**Business.** Portable performance: the same wheel is near-optimal on a laptop, a
Xeon, and an ARM server — no per-machine tuning, which is what makes a
distributable product.

## 6. Distributed CPU scale-out  ✅ / ◇

**Problem.** One CPU is small; cost-sensitive users have *many* CPUs (a cluster of
cheap nodes) but no GPUs. **Algorithm (delivered ✅):** authenticated, LZ4-
compressed TCP transport; NCCL-style ring/tree AllReduce; tensor- and
pipeline-parallel executors; a GSPMD auto-partitioner; MoE grouped-GEMM — all
verified across real OS processes (`TensorParallel`, `NCCLRingAllReduce`,
`XlaGspmd`, `MoEGroupedGemm`, `test_cluster_exec`). **Next (◇):** overlap
compute with the compressed transfer (double-buffered, latency-hiding schedule)
and a bandwidth-aware partitioner that places shards to minimize cross-node
traffic; the only truly external gap is a physical multi-machine run
(needs networked hardware — `missingFeatures.md` §4).

**Business.** Turn a rack of commodity CPU boxes into a training/inference fabric
with no GPU capex — the "GPU-poor" cluster story, which is a real market for
education, national labs on a budget, and privacy-constrained on-prem.

## 7. A PTX/IR idiom superoptimizer  ◇ (research)

**Problem.** Emulated kernels lower generic PTX; the CPU could run hand-optimized
equivalents of common idioms (warp reductions, prefix scans, histogram, GEMM
epilogues) far faster than the generic lowering. **Algorithm.** Pattern-match
recognizable idioms in the IR and substitute a vectorized CPU implementation
(a small, verified rewrite library), falling back to the generic path otherwise —
a peephole superoptimizer scoped to GPU idioms. The differential-fuzzing harness
that already keeps the four tiers bit-exact is exactly the oracle needed to prove
each rewrite correct.

**Business.** Compounding, transparent speedups on the exact kernels real code
uses most — and a defensible technical moat (an idiom library + a correctness
oracle is hard to replicate).

---

## How this becomes a business

The through-line is **"run GPU software with no GPU, fast enough and cheap
enough to matter."** Concrete products the above unlock:

- **On-prem / edge LLM inference** (items 1, 2, 3, 4) — private models on a laptop
  or a single CPU server; the ternary path already runs a real 2B model.
- **GPU-free CI and CUDA education** (correctness + items 2, 5) — test CUDA/ML
  code on any runner; already the LLVM-free wheel's core value.
- **Commodity CPU clusters** (item 6) — cheap scale-out for the GPU-poor.

Each is measurable (tokens/s, $/token, latency) and each maps to a lever the CPU
can actually win on. The engineering discipline — every tier held bit-exact, every
feature test-gated — is what lets these ship as a trustworthy product rather than a
demo.

*This is a research/roadmap document. Items marked ◇ are proposals, not shipped
features; items marked ✅ are in the test suite today.*
