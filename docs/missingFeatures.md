# VGRE — Remaining Work & Advanced Roadmap

**Last Updated**: 2026-08-04

This file tracks **what is not yet done**. Everything implementable to the project's *real,
no-stub* standard **without external hardware/accounts has been delivered, tested, and removed
from this file** (see git history for the full set).

**Delivered baseline.** The complete in-tree large-model programme — bf16/fp16 + int4/int8
storage, mmap safetensors / GGUF (Q4_0/Q4_1/Q8_0/Q4_K/Q6_K) + GPTQ/AWQ, BLAS-backed GEMM (82×)
with dequant-in-GEMM, autoregressive generation + samplers, a byte-level BPE tokenizer matching
HF token-for-token (incl. unified `tokenizer.json`), tensor/pipeline parallelism + GSPMD
auto-partitioner over real RDMA/TCP sockets, CUDA-C→JIT with the full device-intrinsic surface, a
CUDA-GDB RSP debugger, LoRA fine-tuning, an HNSW vector index (local RAG), the lightweight
LLVM-free `libvgre_nn` with a real from-scratch TCP all-reduce, and a hardened C-ABI error
channel. Verified end-to-end on **real GPT-2 (124M)** matching Hugging Face, and on **real
multi-process distributed training** (data-, tensor-, pipeline-parallel).

**Plus the six 2026 advanced tracks (T1–T6), now all delivered — see §1.** **Suite: 303/303.**

The next-frontier in-tree items in §2 are now delivered. Remaining work is in **three** buckets:
1. **§1 — Narrow remainders inside the delivered T1–T6 tracks** (breadth/perf, not correctness).
2. **§3 — Externally-blocked tracks** (hardware / accounts / auditors / gated downloads).
3. **§4 — Physical large-model runs** (a real cluster or a license-gated multi-GB download).

---

## 1. Delivered: the 2026 advanced tracks (T1–T6) — and what's narrowly left

All six are **in-tree, from scratch, dependency-free, numerically verified against independent
references**, and live in the LLVM-free `libvgre_nn`. Build steps and success criteria are in
`implementationPlan.md`.

| Track | Delivered | Verified by |
|-------|-----------|-------------|
| **T1** Ternary / BitNet b1.58 | absmean ternary codec + **multiplication-free** GEMM (AVX2 add/sub masks + scalar); `BitLinear` quantization-aware training via a straight-through `ternary_quantize` op | kernel == dense dequantized GEMM to 1.6e-05; STE grad exact; BitLinear net trains (loss 1.3→0.34) |
| **T2** Mixture-of-Experts | top-k router; **compute-sparse** dispatch via new `index_select`/`index_add` gather-scatter ops; **expert-parallel** across cluster ranks (all_reduce of partials); Switch load-balance aux loss; MoE drop-in transformer FFN | combine == NumPy 2.4e-07; exactly k experts/token; **2 real OS processes**: forward bit-identical, router-grad sum == full (1.8e-08); MoE-LM regenerates 12/12 |
| **T3** Speculative decoding | lossless **greedy** speculative decode; **sampler-exact** speculative sampling (min(1,q/p) accept + residual resample); `ngram_draft_fn` and `model_draft_fn` drafters | == greedy token-for-token at 4.0 tokens/forward; sampler-exact empirical dist == analytic target (TV 0.038 on a diffuse q); temp→0 == greedy |
| **T4** State-space models (Mamba/S6) | `selective_scan` autograd op with exact adjoint, **parallelized over the state channels**; full multi-state selective SSM (Δ=softplus, Ā=exp(Δ⊙A), per-state scan, C-contraction, D-skip); attention-free Mamba LM | scan fwd exact + grads ~1e-5 vs finite differences; parallel path bit-identical (par_err 0.0), **~4.9× at 8 threads**; full block == NumPy Δ/A/B/C reference 7.6e-06; Mamba-LM regenerates 14/14 |
| **T5** MXFP4 microscaling | OCP E2M1 + shared E8M0 block-scale codec, dequant-in-GEMM | round-trip bound + MXFP4-GEMM == fp reference |
| **T6** QLoRA | frozen **ternary** base (~2 bits/weight, stored as codes, *not* a parameter, no gradient) + trainable LoRA adapters; `from_linear` | params == {A,B} exactly; base 2.25 bits/weight; init == quantized base; adapter grads == NumPy; base frozen; fine-tune converges (4.1→0.000) |

**Narrow remainders (breadth/perf — no correctness gaps):**

| Track | Left | Nature |
|-------|------|--------|
| T1 | GGUF `I2_S`/TL1/TL2 ternary tensor loader | needs a real BitNet-b1.58 checkpoint to verify end-to-end (**external download**) |
| T3 | KV-cache reuse + rollback on the raw C++ `generate_cached` path; tree verification; self-speculative (early-exit) drafting | **throughput optimization** — the algorithm is complete and proven |
| T4 | depthwise short conv before the SSM; **Mamba-3 MIMO** (matrix-matrix) state update; Mamba safetensors/GGUF loader | breadth / richer parameterization |
| T5 | SIMD unpack path for the 4-bit decode (the row-major block layout fights column-wise vectorization) | perf |
| T6 | int4/MXFP4 base option (ternary is in); dequant-in-GEMM for the base path | breadth / peak-memory |

---

## 2. Delivered: the next frontier

Each is in-tree, from scratch, and composes primitives we now have. Ordered by
mission impact (**run/train large models on CPUs + clusters, no GPU, lightweight**).

### 2.1 Hybrid Mamba–Transformer blocks (Jamba-style) — **DONE**
`vgre.nn.hybrid_blocks(dim, n_layers, num_heads, attn_every=4, d_state=8, moe_experts=…)` builds an
interleaved stack: every `attn_every`-th layer is a `TransformerBlock`, the rest are `MambaBlock`s
(and the attention layers can themselves be MoE). Only 1-in-`attn_every` layer carries a KV cache —
at `attn_every=4` that is a ~4× cut in KV memory versus an all-attention stack of the same depth,
while attention is retained where exact long-range recall matters.

Required making the SSM **batch-aware**: `SSMBlock` now accepts `[B,T,dim]` and scans each sequence
independently (a flattened `[B*T,dim]` scan would leak state across sequence boundaries — the test
asserts the batched result equals per-sequence scans **exactly (0.0)** *and* differs from the leaky
flat scan). Verified (`test_nn.py::test_hybrid_mamba_transformer`): batch-independence exact, the
stack interleaves as `MMMAMMMA`, and a 4-layer hybrid LM trains (3.4 → 0.000) and greedily
regenerates its sequence 14/14.

### 2.2 KV-cache quantization (int8) — **DONE**
At long context the KV cache, not the weights, dominates memory. `GPT::set_int8_kv_cache(true)`
(C-ABI `vgre_lm_set_int8_kv_cache`, Python `LanguageModel.set_int8_kv_cache()`) stores the
generation KV cache as int8 with a **per-(position, head) absmax scale** instead of fp32: per head
that is Dh bytes + one fp32 scale rather than 4·Dh bytes — a **4·Dh/(Dh+4)** reduction (3.2× at
Dh=16, 3.8× at Dh=64, → 4× as the head dim grows). Only one representation is allocated, so the
saving is real, and the fp32 path is untouched when the flag is off.

Quantization is lossy (symmetric absmax), but the perturbation sits far below the argmax margin:
verified (`test_lm_bindings.py` §7) that a trained LM's **greedy decoding is identical** with int8
KV, at 512 → 160 KV bytes/position/layer (3.20×). Composes with §2.1 — the hybrid stack
concentrates KV into the few attention layers, and this shrinks those.

**Left:** int4 KV (a second halving) and wiring the same scheme into `KVCacheManager`'s paged
pools for the serving path.

### 2.3 Multi-token prediction (MTP) heads — **DONE**
`vgre.nn.MTPHeads(dim, vocab, n_predict=K)` — K linear heads on a shared trunk, head j predicting
j tokens ahead; `mtp_loss(head_logits, targets)` sums the per-head cross-entropy (each head's
supervised range shrinks by j at the tail); `mtp_draft_fn(mtp_model, K)` uses the heads as the
model's **own draft** for speculative decoding — K draft tokens from ONE trunk forward, no second
model, no extra KV memory. Verified (`test_nn.py::test_mtp`): trained with the summed loss all
three heads predict their j-ahead token at 100% accuracy, and MTP-drafted speculative decoding is
lossless vs greedy at **4.0 tokens/forward** — compounding directly with T3.

### 2.4 Continuous batching / serving loop — **DONE**
`KVCacheManager` now owns a paged KV pool and `ContinuousBatchScheduler` admits queued requests,
prefills prompts, advances each running request one token per scheduler step, retires finished
requests immediately, and reclaims their KV blocks for new arrivals. Requests submitted mid-flight
join the next scheduler step; the batch cap and KV capacity are both enforced.

Verified (`test_kv_cache.cpp`): four requests, including one added mid-run, drain under a
`maxBatch=2` cap, never exceed the batch limit, and return the KV pool to its initial free-block
count after retirement. This is the serving lifecycle layer over the paged KV cache; model-specific
logit sampling can plug in where the test currently appends synthetic K/V rows.

### 2.5 Structured sparsity (2:4 / block) + sparse training path — **DONE**
`vgre.nn.StructuredSparseLinear(in_features, out_features, n=2, m=4)` adds a fixed N:M pruning mask
over the contraction dimension of Linear's `[in,out]` weights, physically zeros pruned entries,
applies `W * mask` in the autograd forward path, and exposes compact per-group bit metadata via
`metadata()` for sparse inference kernels. Helper APIs `structured_nm_mask()` and
`structured_nm_metadata()` are available for checkpoint conversion and inspection.

Verified (`test_nn.py::test_structured_sparse_linear`): every 2:4 group keeps exactly two weights,
metadata bit counts match the mask, forward output equals the dense masked reference exactly,
pruned weights receive zero gradient, and a 50%-sparse MLP trains to convergence. The older
core `block_sparse`/N:M math remains available for low-level sparse kernels; the Python training
path stays lightweight and dependency-free.

### 2.6 Quantization-aware distillation — **DONE**
The autograd engine now has a native fused `softmax_cross_entropy_soft(logits, soft_targets)` op,
exported through the C ABI and Python. `vgre.nn.distillation_loss()` implements
`T^2 * CE(softmax(student/T), softmax(teacher/T))`, with optional hard-label CE blending, and
`distill_step()` performs one optimizer step from a callable teacher or precomputed teacher logits.
Teacher probabilities are materialized as constants, so gradients update only the student.

Verified (`test_autograd.cpp::soft_sce`, `test_nn.py::test_distillation`): soft-target CE gradients
match central finite differences in C++; the Python distillation gradient matches the analytic
`T * (student_prob - teacher_prob) / batch` formula; and a ternary `BitLinear` student learns a
fixed teacher distribution with 0.88 teacher-argmax agreement.

---

## 3. Externally-blocked tracks (need hardware, an account, an auditor, or content)

The in-tree primitives already exist where applicable; only the externally-gated piece remains.

| § | Track | What's left | Blocker |
|---|-------|-------------|---------|
| 3.1 | GPU security framework | SEV-SNP/TDX enclaves, HSM, FIPS-140 cert | confidential-computing **hardware** + external **auditor** |
| 3.2 | Cryptography | homomorphic / threshold crypto, Intel QAT offload | research-grade scope / crypto-accelerator **hardware** |
| 3.3 | Windows deployment | DirectML backend, AD/Kerberos, Windows containers | Windows-specific **APIs/SDKs** (engine already builds+tests on windows-2022) |
| 3.4 | macOS / Apple Silicon | Metal Performance Shaders backend | **Apple Silicon + Metal** hardware (CPU JIT path is build-verified on macOS) |
| 3.5 | ML frameworks | device-level `jax.jit(backend='vgre')` PJRT plugin | upstream `pjrt_c_api.h` + MLIR C++ libs **not in the wheels** (StableHLO path runs JAX/TF/PyTorch) |
| 3.6 | Model serving | TensorRT-LLM / vLLM *compatibility layers* | those external **runtimes** / a live **fleet** |
| 3.7 | Multi-cloud | apply to live AWS/Azure/GCP | cloud **accounts + credentials** (Terraform module is built) |
| 3.8 | Multi-vendor | Intel oneAPI (SYCL/DPC++), Apple Metal; ROCm library shims | those **SDKs / hardware** (AMD HIP core runtime is done) |
| 3.9 | Edge / CDN | physical edge nodes, CDN providers | external **infrastructure** (latency-aware routing is built) |
| 3.10 | Post-Blackwell (Rubin) | Rubin/HBM4 emulation | **unreleased** hardware, no public ISA |

---

## 4. Physical large-model runs (external only)

The in-tree engineering is complete and verified over real loopback sockets **and across real OS
processes** (multi-step data-parallel with zero cross-step drift; cross-process tensor
parallelism; cross-process expert-parallel MoE). Only two things remain, both outside the code:

| Remaining | Why it isn't in-tree |
|-----------|----------------------|
| **Physical multi-node run** (Llama-3-70B tensor-parallel across N machines; 175B/405B pipeline) | Transport, collectives, the tensor/pipeline executor, and the GSPMD auto-partitioner are all built and verified across real OS processes; a true cross-machine run needs **actual networked machines**. |
| **Frontier-scale checkpoints** (Llama-3-8B/70B/405B, GPT-3) | Identical code path to the demonstrated GPT-2 run — it just needs a **license-gated, multi-GB download**. |

T1–T6 make these runs materially cheaper on commodity CPUs: ternary + MXFP4 + QLoRA shrink the
memory, MoE shrinks the active compute, speculative decoding shrinks the latency, and the SSM
path removes the KV cache entirely — which is the whole point.
