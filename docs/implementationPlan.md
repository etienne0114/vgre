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

### Milestone L1 — bf16 storage + real checkpoint loading  *(in-tree work DONE)*
- ✅ **Memory-mapped safetensors loader (2026-06-16):** `vgre::xla::SafeTensors`
  (`src/xla/safetensors.cpp`) mmaps a checkpoint once and materializes tensors into `Literal`s,
  dequantizing F16/BF16 (incl. subnormals) and widening F64/int. Cross-platform (`mmap` / Win32
  `MapViewOfFile`). `load(..., keepNative=true)` keeps BF16/F16 at 16-bit width. Test `XlaSafetensors`.
- ✅ **bf16/f16 native `Literal` storage (2026-06-16):** `Literal` is now dtype-tagged
  (`DType{F32,F16,BF16}`) with a 16-bit `half` buffer + `toF32/toBF16/toF16/at/storageBytes`
  (`include/vgre/xla/half.h` holds the RNE conversions). Weights stay at **native 16-bit width in
  RAM** (8B → 16 GB vs 32 GB); the evaluator decompresses a bf16 `Parameter`/`Constant` to f32 only
  as the consuming op runs, and liveness-based buffer reuse frees it after last use — so peak f32
  RAM is the **live set**, not the whole model. Compute stays f32 (no per-op change; CPU bf16
  arithmetic isn't faster — the win is memory). Test `XlaBf16Literal`: matmul with bf16 params
  matches fp32 within ~1.8%.
- **Exit (needs a 16 GB external checkpoint):** a real **Llama-3-8B** forward pass on VGRE
  numerically matches Hugging Face — the only remaining piece, and it is purely an external
  download + wiring, no further engine work for the bf16 path.

### Milestone L2 — BLAS-backed matmul (throughput) — ✅ DONE (2026-06-14)
- `Dot` / `DotGeneral` route to **cblas_sgemm** (OpenBLAS/MKL/reference) via
  `vgre::xla::gemm_f32` (`src/xla/blas_gemm.cpp`), with a built-in **cache-tiled** fallback when no
  BLAS is present — both parallelized over `vgre::xla::ThreadPool` (row-split for one large GEMM,
  batch-split for many small ones). `DotGeneral` maps the canonical batched-GEMM layout (leading
  batch dims, one contract + one free dim per side) to a single batched GEMM with transpose flags;
  other shapes keep the correct generic loop.
- **Exit met:** **82×** faster than the naive triple loop on a 4096³ GEMM (`XlaBlasGemm` test,
  ≥10× criterion); all JAX/TF/PyTorch/Keras/FlashAttention tests pass through it.
- *Remaining within throughput:* a dedicated fused flash-attention kernel (the online-softmax
  algorithm already exists in `KVCacheManager`) — optional, the BLAS path already makes attention
  matmuls fast.

### Milestone L3 — int4/int8 quantized weights  *(in-tree work DONE)*
- ✅ **GGUF loader + native block-quant `Literal` storage (2026-06-16):**
  `vgre::xla::GGUF` (`src/xla/gguf.cpp`) mmaps a llama.cpp `.gguf` checkpoint, parses the
  magic/version/typed-metadata/tensor-info header, and loads tensors as f32 or — with
  `keepNative` — at native quantized width. `Literal` gained `DType{Q8_0,Q4_0,Q4_1}` block storage
  (ggml layouts; dequant kernels in `include/vgre/xla/quant.h`), so an int4 weight is **~4.5
  bits/weight resident** (8B → ~4.5 GB) and the engine dequantizes it to f32 only when its consumer
  runs — same transient-dequant path as bf16, so a Q4_0 `Parameter` matmul is *identical* to the
  dequantized-f32 matmul, with no per-op change. Test `XlaGgufQuant`.
- *Remaining:* the super-block K-quants (Q4_K/Q6_K, used by many HF GGUFs) and GPTQ/AWQ packings;
  dequant-inside-the-GEMM-kernel (vs materialize-then-GEMM) for less transient f32.
- **Exit (needs an external int4 checkpoint):** Llama-3-8B fits ~4.5 GB and runs on a laptop CPU.

### Milestone L4 — production generation loop  *(in-tree work DONE)*
- ✅ **Autoregressive driver + samplers (2026-06-16):** `vgre::xla::generate` + `sampleToken`
  (`src/xla/generation.cpp`) — model-agnostic loop over a `step(prevToken,pos)→logits` callable
  with greedy / temperature / top-k / top-p (nucleus) sampling, EOS + length stop. Drives a real
  `HloModule` forward pass per token and integrates with the built **paged KV-cache**
  (`vgre::core::KVCacheManager`) — the cache grows one token/step and reclaims blocks on finish.
  The **continuous-batching scheduler** (`ContinuousBatchScheduler`) is already built for
  multi-request serving. Test `XlaGeneration`.
- ✅ **Byte-level BPE tokenizer (2026-06-16):** `vgre::xla::BpeTokenizer` (`src/xla/tokenizer.cpp`),
  built from scratch — 256-byte base vocab (every input round-trips, no UNK, UTF-8 implicit),
  `train()` learns merges from a corpus, `encode()`/`decode()` are inverses, and `loadMerges()`
  imports a real model's ordered merge list. Test `XlaTokenizer`. *(Glue remaining: parsing GPT-2
  `vocab.json`+`merges.txt` / HF `tokenizer.json` — the byte↔unicode remap + JSON — to ingest a
  specific shipped tokenizer file.)*
- **Exit (needs a real checkpoint + its tokenizer file):** long-context (8K–128K), multi-request
  LLM serving on CPU.

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
- **L2:** ✅ ≥10× matmul speedup vs the naive interpreter loop on 4096² GEMM — **met: 82×**.
- **L3:** Llama-3-8B int4 resident in ≤4 GB RSS; output perplexity within tolerance of the bf16 run.
- **L4:** sustained multi-request decode with paged KV; no per-token reallocation of the model.
- **L5:** 70B tensor-parallel across nodes with correctness matching a single-node bf16 reference.
