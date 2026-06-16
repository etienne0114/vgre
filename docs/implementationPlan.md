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
- ✅ **Exit MET on a real model (2026-06-16):** real **GPT-2 (124M)** runs end-to-end on VGRE —
  weights loaded from the actual Hugging Face `model.safetensors` via `vgre::xla::SafeTensors`,
  every matmul through the engine's `gemm_f32` (`tools/gpt2_infer.cpp`) — and its top-5 next-token
  logits **match Hugging Face transformers** (identical token order, agreement to ~1e-4, f32
  reduction-order rounding). The SafeTensors loader also reads all **160** real GPT-2 tensors with
  **0 mismatches** vs the reference `safetensors` library (`tools/validate_safetensors.py`). Llama-3-8B
  is the same path at scale (gated + 16 GB download); GPT-2 is the open, ungated proof that the
  loader + bf16/f32 engine run a real pretrained model. **bf16 verified on the real model too**
  (`VGRE_GPT2_BF16=1`): resident weights **548 MB → 274 MB (exactly half)** and the predicted next
  token is unchanged (" the") — the native-width memory win with correct output on real weights.

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
- ✅ **Loader verified on real GGUF (2026-06-16):** `vgre::xla::GGUF` reads a canonical `.gguf`
  written by the official `gguf` library and dequantizes **F32 / F16 / Q8_0 / Q4_0 bit-identically**
  to `gguf.quants.dequantize` (`tools/make_probe_gguf.py` + `checkpoint_probe`).
- ✅ **End-to-end quantized real model (2026-06-16):** real GPT-2 converted to a GGUF with the
  official writer (`tools/gpt2_to_gguf.py`), loaded back through `vgre::xla::GGUF`, and run on the
  engine (`tools/gpt2_infer <file>.gguf`). **Q8_0**: resident **548 → 220 MB**, top-1 token
  unchanged (" the"), top-5 identical to f32 (near-lossless). **Q4_0 (int4)**: resident **177 MB**
  (~1/3 of f32), top-1 still correct (" the") with the expected int4 logit perturbation. The int4
  weight path runs a real pretrained model.
- ✅ **K-quants Q4_K/Q6_K (2026-06-16):** the super-block formats most modern HF GGUFs use —
  Q4_K (256-weight super-block, 8×6-bit sub-scales/mins, 144 B) and Q6_K (210 B) dequant kernels
  (`include/vgre/xla/quant.h`), `Literal` `DType{Q4_K,Q6_K}` storage, and GGUF load — verified
  **bit-identical to `gguf.quants.dequantize`** over random super-blocks (`tools/validate_kquant.py`),
  plus hand-constructed in-suite checks (`XlaGgufQuant`).
- ✅ **Dequant-inside-the-GEMM (2026-06-16):** `gemm_q` (`src/xla/quant_gemm.cpp`) multiplies a
  block-quantized weight by dequantizing one column-block of one row at a time *inside* the matmul —
  peak transient f32 is O(N), not O(K·N), so the weight is never fully materialized. Identical to
  `gemm_f32(X, dequant(W))` for Q8_0/Q4_0/Q4_1/Q4_K/Q6_K incl. M=1 decode (`XlaQuantGemm`),
  thread-pool parallel over N-blocks. Wired into `gpt2_infer`'s GGUF path: real GPT-2 runs with its
  matmul weights staying quantized in RAM **and** through compute (≈298 MB vs 548 MB f32), logits
  identical to the materialized path.
- ✅ **GPTQ / AWQ 4-bit safetensors (2026-06-16):** `gptqDequantize` / `awqDequantize`
  (`src/xla/gptq.cpp`) turn the four GPTQ/AWQ tensors (qweight/qzeros/scales/g_idx) back into an
  f32 weight — GPTQ's nibble-along-`in` packing + `+1` zero convention + act-order `g_idx`, and AWQ's
  nibble-along-`out` `[0,2,4,6,1,3,5,7]` order. `SafeTensors::loadInt32` reads the packed int32
  tensors without f32 widening (packed values exceed f32's exact range). Validated by in-suite
  round-trip (`XlaGptq`) **and** against an independent numpy packer through a real safetensors file
  (`tools/validate_gptq.py` → sums match exactly).
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
  `train()` learns merges, `encode()`/`decode()` inverses, `loadMerges()` imports a merge list.
- ✅ **Real GPT-2 tokenizer file ingestion (2026-06-16):** `loadGpt2(vocab.json, merges.txt)` —
  builds GPT-2's byte↔unicode table, parses the vocab + ordered merges, and does GPT-2
  pre-tokenization (contractions / ` ?\p{L}+` / ` ?\p{N}+` / symbols / `\s+(?!\S)`).
  `encodeGpt2`/`decodeGpt2` emit/consume the model's exact ids — **11/11 exact vs the Hugging Face
  GPT-2 tokenizer** (multi-space, UTF-8 `naïve café`, contractions, digits/symbols). Tests
  `XlaTokenizer` (synthetic loadGpt2) + optional `RealGpt2Tokenizer`.
- ✅ **Full text→text on the real model (2026-06-16):** `gpt2_infer <model> --gen N "prompt"`
  pipes prompt → my GPT-2 tokenizer → engine forward (`gemm_f32`, real weights) → L4 `argmax` → my
  decoder, and its greedy continuation is **token-for-token identical to Hugging Face GPT-2**
  `model.generate` — from the f32 safetensors *and* the int4 Q4_0 GGUF. The whole CPU pipeline
  (load → tokenize → generate → detokenize) reproduces real GPT-2.
- **Exit (for very long context / many-request throughput):** the loop + paged-KV + tokenizer all
  exist; large-scale serving just needs a bigger model checkpoint.

### Milestone L5 — distributed (models that don't fit one box)  *(sharding DONE)*
- ✅ **Tensor + pipeline sharding (2026-06-16):** `vgre::xla` parallel module (`src/xla/parallel.cpp`)
  — Megatron-style **column-parallel** (shard W along N → per-rank GEMM → all-gather) and
  **row-parallel** (shard W along K + X along K → per-rank GEMM → all-reduce sum) matmul, each
  verified bit-identical / within float-reassociation tolerance of single-node for ranks 1–4 incl.
  uneven splits; the **Megatron column→row MLP** (one all-reduce per FFN, ReLU kept local on each
  rank's column slice) matches single-node; plus **pipeline partitioning** (balanced contiguous
  layer ranges) + staged execution. The `Communicator` (all-gather / all-reduce) is in-process for
  single-box TP + tests. Test `XlaParallel`.
- ✅ **Collectives over the real transport (2026-06-16):** `columnParallelMatmulRdma` /
  `rowParallelMatmulRdma` (`src/xla/parallel_rdma.cpp`) run the all-gather / all-reduce over the
  portable **SoftwareRDMA** layer — each rank registers its partial and the gathering rank pulls
  every partial with **one-sided RDMA reads over real sockets** (loopback in-process; identical code
  across machines). Result equals single-node; the test asserts the full output actually crossed
  sockets (`XlaParallelRdma`, verified under load). Tensor parallelism is now distributed over a real
  network transport, not in-process memcpy.
- ✅ **GSPMD auto-partitioner (2026-06-16):** `gspmdExecute` (`src/xla/gspmd.cpp`) takes an
  HloModule + a per-parameter `Sharding` annotation, **propagates the sharding through the graph**
  (elementwise ops flow it through; a Dot with a sharded contraction → row-parallel, a sharded
  output dim → column-parallel) and **auto-inserts the collectives** (all-reduce after a sharded
  contraction, all-gather to replicate at the root) — no hand-written calls. Verified == single-node
  for column/row-parallel single matmuls and the **Megatron MLP** (one auto-inserted all-reduce from
  the annotations alone) across ranks 1–4 (`XlaGspmd`).
- *Remaining:* exercising it across physical nodes (the loopback transport + the executor are done).
- **Exit (needs a real N-node cluster):** **Llama-3-70B** tensor-parallel across N nodes; pipeline
  parallelism for 175B/405B.

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
