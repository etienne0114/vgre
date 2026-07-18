# VGRE — Remaining Work & Advanced Roadmap

**Last Updated**: 2026-07-18

This file tracks **what is not yet done** and the **new competitive frontier** we are
building toward. Everything implementable to the project's *real, no-stub* standard
**without external hardware/accounts has already been delivered, tested, and removed from
this file** (see git history for the full set). The delivered baseline now includes the
complete in-tree large-model programme — bf16/fp16 + int4/int8 storage, mmap safetensors /
GGUF (Q4_0/Q4_1/Q8_0/Q4_K/Q6_K) + GPTQ/AWQ, BLAS-backed GEMM (82×) with dequant-in-GEMM,
autoregressive generation + samplers, a byte-level BPE tokenizer matching HF token-for-token
(incl. unified `tokenizer.json`), tensor/pipeline parallelism + GSPMD auto-partitioner over
real RDMA/TCP sockets, CUDA-C→JIT with the full device-intrinsic surface, a CUDA-GDB RSP
debugger, LoRA fine-tuning, an HNSW vector index (local RAG), the lightweight LLVM-free
`libvgre_nn` with a real from-scratch TCP all-reduce, and a hardened C-ABI error channel.
Verified end-to-end on **real GPT-2 (124M)** matching Hugging Face, and on **real multi-process
distributed training** (data-, tensor-, pipeline-parallel). **Suite: 300/300.**

The remaining work is now in **three** buckets:
1. **§1 — New advanced feature roadmap (2026 frontier).** The competitive edge: what makes
   VGRE *intelligent, modern, and hard to beat* — all buildable in-tree, from scratch,
   dependency-free, to the same real-no-stub standard.
2. **§2 — Externally-blocked tracks** (hardware / accounts / auditors / gated downloads).
3. **§3 — Physical large-model runs** (a real cluster or a license-gated multi-GB download).

---

## 1. New advanced feature roadmap — the 2026 competitive frontier

These are **not yet in-tree** (audited 2026-07-18: no BitNet/ternary, MoE, Mamba/SSM,
speculative decoding, MXFP4, or QLoRA anywhere in `src/`, `include/`, `bindings/`). Each is
chosen to advance the core mission — **run and train large models on CPUs and clusters, with
no GPU, staying lightweight** — and each is built **from scratch** (no new third-party
dependency), extending code paths that already exist. Detailed build steps live in
`implementationPlan.md`; success criteria are at the bottom of that file.

### 1.1 — Ternary / 1-bit inference (BitNet b1.58) — **P0, highest-impact lightweight win**

Modern 1.58-bit LLMs constrain every weight to a ternary value {−1, 0, +1}. This turns the
dominant matmul into a **multiplication-free** accumulate (add/sub only), which is exactly
where a CPU is *not* disadvantaged versus a GPU. Microsoft's `bitnet.cpp` reports **2.37×–6.17×
CPU speedups on x86 and up to 82% lower energy**, and runs a **100B-parameter model on a single
CPU** at reading speed — the strongest possible statement of "eliminate the GPU barrier."

**In-tree, from scratch:**
- A ternary weight codec (2-bit packed {−1,0,+1}, per-group fp scale) as a new `Literal`
  storage kind next to the existing bf16/int4/int8 codecs.
- A **multiplication-free ternary GEMM** micro-kernel (add/sub accumulate, AVX2/AVX-512 +
  scalar fallback) beside `intree_gemm_f32.cpp`, plus a **T-MAC-style lookup-table** variant
  (precomputed activation partial sums indexed by packed ternary bytes) for sub-2-bit throughput.
- A `BitLinear` layer in the in-tree autograd/model stack, and a GGUF loader path for the
  `I2_S` / TL1 / TL2 ternary tensor types so real BitNet checkpoints load directly.

### 1.2 — Mixture-of-Experts (MoE) with expert-parallel over the cluster — **P1**

Frontier open models (Mixtral, DeepSeek-V3, Qwen-MoE) are sparse: only a few experts fire per
token, so the *active* compute is small even when the model is huge — ideal for CPU clusters.

**In-tree, from scratch:** a top-k **router/gate**, sparse expert dispatch/combine, an `MoELayer`
in the autograd stack, and **expert-parallel** sharding that places experts on different cluster
nodes and routes tokens over the **already-built** all-reduce / all-to-all collectives. This is
the piece that lets a cluster of ordinary machines serve a frontier-scale MoE.

### 1.3 — Speculative decoding — **P1, 2–10× faster CPU generation**

Draft several tokens cheaply, then verify them in one batched forward of the full model,
accepting the longest correct prefix. Reported **2×–10× decode latency reduction** with
identical output distribution.

**In-tree, from scratch:** self-speculative / n-gram / small-draft-model drafting + **tree
attention verification** wired into `generation.cpp` and `KVCacheManager` (paged KV already
exists). Lossless: output is provably identical to greedy/sampled decode.

### 1.4 — State-space models (Mamba-2 / Mamba-3) — **P2, linear-time long context**

SSMs run in **linear** time and **constant** memory per step (no growing KV cache), which is a
decisive CPU advantage at long context. Mamba-3 (2026) adds a MIMO recurrence that maps cleanly
onto matrix-matrix kernels we already have.

**In-tree, from scratch:** a **parallel selective-scan** (associative prefix-scan, threaded +
SIMD) op in the autograd engine with its reverse-mode backward, an SSM/Mamba block in the model
stack, and a GGUF/safetensors loader path for Mamba checkpoints — extending the engine beyond
transformers.

### 1.5 — MXFP4 microscaling 4-bit (OCP) — **P2**

The Open Compute microscaling FP4 format (shared 8-bit block scale + 4-bit elements) is becoming
the interop standard for 4-bit weights *and* activations.

**In-tree, from scratch:** an MXFP4 block codec + dequant-in-GEMM path alongside the existing
int4/GPTQ/AWQ codecs, so MXFP4 checkpoints load and run without a separate upcast pass.

### 1.6 — QLoRA (quantized-base fine-tuning) — **P3**

Combine the existing **LoRA** adapters with the existing **int4/ternary** quantized base so a
large model can be **fine-tuned on a laptop**: frozen quantized base + trainable fp adapters,
de-quant only in the forward matmul. Extends `LoRALinear` + the quant codecs already in-tree.

---

## 2. Externally-blocked tracks (need hardware, an account, an auditor, or content)

The in-tree primitives already exist where applicable; only the externally-gated piece remains.

| § | Track | What's left | Blocker |
|---|-------|-------------|---------|
| 2.1 | GPU security framework | SEV-SNP/TDX enclaves, HSM, FIPS-140 cert | confidential-computing **hardware** + external **auditor** |
| 2.2 | Cryptography | homomorphic / threshold crypto, Intel QAT offload | research-grade scope / crypto-accelerator **hardware** |
| 2.3 | Windows deployment | DirectML backend, AD/Kerberos, Windows containers | Windows-specific **APIs/SDKs** (engine already builds+tests on windows-2022) |
| 2.4 | macOS / Apple Silicon | Metal Performance Shaders backend | **Apple Silicon + Metal** hardware (CPU JIT path is build-verified on macOS) |
| 2.5 | ML frameworks | device-level `jax.jit(backend='vgre')` PJRT plugin | upstream `pjrt_c_api.h` + MLIR C++ libs **not in the wheels** (StableHLO path runs JAX/TF/PyTorch) |
| 2.6 | Model serving | TensorRT-LLM / vLLM *compatibility layers* | those external **runtimes** / a live **fleet** |
| 2.7 | Multi-cloud | apply to live AWS/Azure/GCP | cloud **accounts + credentials** (Terraform module is built) |
| 2.8 | Multi-vendor | Intel oneAPI (SYCL/DPC++), Apple Metal; ROCm library shims | those **SDKs / hardware** (AMD HIP core runtime is done) |
| 2.9 | Edge / CDN | physical edge nodes, CDN providers | external **infrastructure** (latency-aware routing is built) |
| 2.10 | Post-Blackwell (Rubin) | Rubin/HBM4 emulation | **unreleased** hardware, no public ISA |

---

## 3. Physical large-model runs (external only)

The in-tree engineering is complete and verified over real loopback sockets **and across real OS
processes** (multi-step data-parallel with zero cross-step drift; cross-process tensor
parallelism). Only two things remain, both outside the code:

| Remaining | Why it isn't in-tree |
|-----------|----------------------|
| **Physical multi-node run** (Llama-3-70B tensor-parallel across N machines; 175B/405B pipeline) | Transport, collectives, the tensor/pipeline executor, and the GSPMD auto-partitioner are all built and verified across real OS processes; a true cross-machine run needs **actual networked machines**. |
| **Frontier-scale checkpoints** (Llama-3-8B/70B/405B, GPT-3) | Identical code path to the demonstrated GPT-2 run — it just needs a **license-gated, multi-GB download**. |

Once §1 lands, the same distributed machinery makes these runs *cheaper* on commodity CPUs:
ternary + MoE + speculative decode shrink both the memory and the compute a physical cluster
needs, which is the whole point.
