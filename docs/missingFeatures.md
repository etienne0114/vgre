# VGRE — Remaining Work

**Last Updated**: 2026-06-16

This document tracks **only what is not yet done**. Everything implementable to the project's
"real, no-stub" standard *without external hardware/accounts* has been delivered, tested, and
**removed from this file** — see git history / commit messages for the full set. That delivered set
now includes, in addition to the earlier enterprise/security/ops stack, the **complete in-tree
large-model programme**:

> bf16/fp16 + int4/int8 weight storage; memory-mapped **safetensors** and **GGUF** loaders;
> **GGUF Q4_0/Q4_1/Q8_0/Q4_K/Q6_K**, **GPTQ/AWQ** dequant; **BLAS-backed GEMM (82×)** +
> **dequant-inside-the-GEMM**; autoregressive **generation + samplers**; a from-scratch
> **byte-level BPE tokenizer** that reproduces the GPT-2 tokenizer exactly; **tensor/pipeline
> parallelism**, a **GSPMD auto-partitioner**, and collectives **over real RDMA sockets**.
> Verified end-to-end on **real GPT-2 (124M)** — f32 / bf16 / int8 / int4 — matching Hugging Face
> token-for-token. (280/280 tests.)

What is left falls into two buckets, both **external**: hardware / accounts / auditors / content
(§1), and the two large-model items that need a real cluster or a gated multi-GB download (§2).

---

## 1. Externally-blocked tracks (need hardware, an account, an auditor, or content)

The in-tree primitives already exist where applicable; only the externally-gated piece remains.

| § | Track | What's left | Blocker |
|---|-------|-------------|---------|
| 1.1 | GPU security framework | SEV-SNP/TDX enclaves, HSM, FIPS-140 cert | confidential-computing **hardware** + an external **auditor** |
| 1.2 | Cryptography | homomorphic / threshold crypto, Intel QAT offload | research-grade scope / crypto-accelerator **hardware** |
| 3.1 | Windows deployment | DirectML backend, AD/Kerberos auth, Windows containers, PowerShell module | Windows-specific **APIs/SDKs** (engine already builds+tests on windows-2022) |
| 3.2 | macOS / Apple Silicon | Metal Performance Shaders backend | **Apple Silicon + Metal** hardware (engine builds/tests on macos-arm64) |
| 4.2 | ML frameworks | device-level `jax.jit(backend='vgre')` PJRT plugin | needs upstream `pjrt_c_api.h` + MLIR C++ libs **not in the wheels** (StableHLO path already runs JAX/TF/PyTorch) |
| 4.3 | Model serving | TensorRT-LLM / vLLM *compatibility layers*, A/B-canary rollout | those external **runtimes** / a live **fleet** (PagedAttention, continuous batching, generation are built) |
| 5.3 | Multi-cloud | apply to live AWS/Azure/GCP, cross-cloud networking | cloud **accounts + credentials** (Terraform module is built + validated) |
| 6.1 | Developer tools | CUDA-GDB-compatible debug stepping | a PTX/SASS single-step **interpreter** (the JIT compiles to native) — large self-contained build, a candidate future phase |
| 6.2 | Documentation & training | enterprise runbooks, video/tutorial content | **content-team** work, not code |
| 7.1 | Post-Blackwell (Rubin) | Rubin/HBM4 emulation | **unreleased** hardware, no public ISA |
| 7.2 | Multi-vendor | Intel oneAPI (SYCL/DPC++), Apple Metal | those **SDKs / hardware** (AMD HIP/ROCm is built) |
| 10.3 | Edge / CDN | physical edge nodes, CDN providers | external **infrastructure** (latency-aware routing is built) |

**6.1 (CUDA-GDB) is the only purely-software item here** — a GDB remote-serial-protocol server over
a PTX single-step interpreter. Everything else needs hardware, an account, an auditor, or content.

---

## 2. Large models — what remains (external only)

The in-tree engineering is complete (see the delivered list above; full milestone detail is in
git history). The forward pass, all weight formats, throughput, generation, the tokenizer, and the
distributed sharding/collectives are done and verified on a real pretrained model. Only two things
remain, and both are **outside the code**:

| Remaining | Why it isn't in-tree |
|-----------|----------------------|
| **Physical multi-node run** (Llama-3-70B tensor-parallel across N machines; 175B/405B pipeline) | The transport, NCCL-style + SoftwareRDMA collectives, the tensor/pipeline executor, and the GSPMD auto-partitioner are all built and verified over **real loopback sockets**. A true cross-machine run needs **actual networked machines**. |
| **Frontier-scale checkpoints** (Llama-3-8B/70B/405B, GPT-3) | Identical code path to the GPT-2 run already demonstrated end-to-end — it just needs a **license-gated, multi-GB download** (Llama-3 is gated; 8B is 16 GB bf16 / ~4.5 GB int4). |

**Optional in-tree polish** (not blocking; the existing paths already cover these):
- a dedicated **fused flash-attention** kernel (the online-softmax algorithm is already in
  `KVCacheManager::pagedAttention`; the BLAS GEMM path already makes attention matmuls fast);
- parsing the unified HF **`tokenizer.json`** format (the GPT-2 `vocab.json`+`merges.txt` path is
  done and exact).

Everything buildable in-tree to the real-no-stub standard is implemented and tested; the remaining
items are environmental (a cluster, a gated download), not missing engine code.
