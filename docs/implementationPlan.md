# VGRE Implementation Plan — Remaining Work

**Last Updated**: 2026-06-14

The original Phase-4 enterprise roadmap (50 tracks) has been **delivered down to its
software-implementable core** — every track buildable to the project's "real, no-stub" standard
without external hardware/accounts is done, tested, and committed (see `missingFeatures.md`
intro and git history for the full list). This plan now contains **only the remaining work**:
a short list of externally-blocked tracks, and the real engineering programme to run
**large models (GPT-3, Llama 3)** on the engine.

The "24-30 month / 50-60 engineer / $35-50M" estimates from the original plan were aspirational
research figures and are **not** a reflection of the remaining scope below.

---

## A. Externally-blocked tracks (not code we can write here)

These need hardware, an account/credential, an auditor, or content work. The in-tree primitives
already exist where applicable.

| Track | Remaining | Blocker |
|-------|-----------|---------|
| Security framework | SEV-SNP/TDX, HSM, FIPS-140 cert | confidential-computing hardware + auditor |
| Cryptography | homomorphic/threshold crypto, Intel QAT | research scope / crypto-accelerator hardware |
| Windows deployment | DirectML, AD/Kerberos, containers, PowerShell | Windows APIs/SDKs (engine already builds+tests on windows-2022) |
| macOS / Apple Silicon | Metal Performance Shaders backend | Apple Silicon + Metal hardware (engine builds+tests on macos-arm64) |
| Model serving | TensorRT-LLM/vLLM compat layers, A/B-canary | external runtimes / live fleet |
| Multi-cloud | apply to live AWS/Azure/GCP | cloud accounts + credentials |
| Multi-vendor | Intel oneAPI (SYCL), Apple Metal | those SDKs/hardware (AMD HIP/ROCm done) |
| Edge / CDN | physical edge nodes, CDN | external infrastructure |
| Docs & training | enterprise runbooks, video content | content-team work |
| Post-Blackwell (Rubin) | Rubin/HBM4 emulation | unreleased hardware, no public ISA |

**One purely-software remaining track:** **CUDA-GDB-compatible debugging** — a GDB
remote-serial-protocol server backed by a PTX/SASS single-step interpreter (the JIT currently
compiles to native, so there is no per-instruction interpreter to step). This is a large
self-contained build; it is a candidate for a dedicated future phase.

---

## B. The forward programme — run GPT-3 / Llama 3 on the engine

**Premise (verified, not aspirational):** every Llama-3 / GPT-3 building block already runs
correctly on the VGRE HLO engine and matches jax — **RMSNorm, RoPE, SwiGLU/SiLU,
Grouped-Query Attention, and a full Llama decoder layer**. Op coverage is *complete*. The work
left is **scale** — memory, numeric formats, weight loading, throughput, distribution. Full
technical detail is in `missingFeatures.md` §2; the milestone plan:

### Milestone L1 — bf16 storage + real checkpoint loading
- Add **bf16/fp16** as a stored tensor type (typed buffer in `Literal`), so weights load at native
  width (8B → 16 GB instead of 32 GB fp32).
- Feed weights as **parameters/external buffers**, not baked constants (the training pytree path
  already does this); add a memory-mapped **safetensors loader**.
- **Exit:** a real **Llama-3-8B** forward pass on VGRE numerically matches Hugging Face.

### Milestone L2 — BLAS-backed matmul (throughput)
- Route `Dot` / `DotGeneral` to **OpenBLAS / MKL `sgemm`** (or the emulated cuBLAS `sgemmEx`);
  cache-tile the remaining ops; add a fused flash-attention kernel (online softmax already in
  `KVCacheManager`).
- **Exit:** decode latency from "minutes/token" to "seconds/token".

### Milestone L3 — int4/int8 quantized weights
- Load **GGUF / GPTQ / AWQ** checkpoints; dequantize per-group inside the matmul (extend the
  existing int8 `dot_general` path to grouped int4).
- **Exit:** Llama-3-8B fits **4 GB** and runs on a laptop CPU.

### Milestone L4 — production generation loop
- Wire the engine's autoregressive loop to the built **paged KV-cache** and **continuous-batching
  scheduler**; add a tokenizer binding.
- **Exit:** long-context (8K–128K), multi-request LLM serving on CPU.

### Milestone L5 — distributed (models that don't fit one box)
- Add a **sharding pass** that splits `DotGeneral`/`Convolution` across ranks and inserts the
  collective (all-reduce/all-gather) a framework's `shard_map`/GSPMD requests, driven over the
  **existing RDMA transport + NCCL-style collectives**.
- **Exit:** **Llama-3-70B** tensor-parallel across N nodes; pipeline parallelism for 175B/405B.

**Honest scope:** L1–L4 are real, self-contained, CPU-only engineering achievable in-tree and
would let VGRE genuinely run Llama-3-8B without a GPU. L5 reuses the existing cluster substrate but
is a substantial distributed-systems effort. The whole programme keeps VGRE's core promise:
**large-model inference and training on commodity CPUs, no GPU required.**

---

## Success criteria (revised, concrete)

- **L1:** Llama-3-8B forward output matches HF reference to fp32 tolerance.
- **L2:** ≥10× matmul speedup vs the naive interpreter loop on 4096² GEMM.
- **L3:** Llama-3-8B int4 resident in ≤4 GB RSS; output perplexity within tolerance of the bf16 run.
- **L4:** sustained multi-request decode with paged KV; no per-token reallocation of the model.
- **L5:** 70B tensor-parallel across nodes with correctness matching a single-node bf16 reference.
