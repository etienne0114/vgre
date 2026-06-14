# VGRE — Remaining Work & Large-Model Roadmap

**Last Updated**: 2026-06-14

This document tracks **only what is not yet done**. Everything that was implementable to the
project's "real, no-stub" standard *without external hardware/accounts* has been delivered and
**removed from this file** — see git history and `CHANGELOG`/commits for the full delivered set
(security: TPM2+CET+bounds-check, AES-GCM/ChaCha/X25519 + ML-KEM-768 PQC, tamper-evident audit
log + GDPR erasure, libFuzzer; ops: Prometheus/OTLP, roofline/flamegraph, Holt-Winters capacity
planning; K8s device-plugin + operator + MIG; RDMA/NCCL; backup/DR; secrets; OIDC/JWT+RBAC;
ETL; GitOps; circuit-breaker/rate-limit/load-balancer; consistent-hash/lock/leader-election;
Helm; chaos+mutation testing; AIOps anomaly detection; webhooks/OpenAPI; PCI/HIPAA/GDPR/SOX
policy engine; **the full ML stack — PyTorch + JAX + TensorFlow + Keras inference, autoregressive
generation, training/backprop, and int8 quantization on a thread-safe, multicore, governed,
observable HLO engine**; WASM + RISC-V cross-execution; a 3-OS CI matrix).

The remaining items fall into two buckets:
1. **§1 Remaining tracks** — each genuinely blocked on external hardware, an account/credential,
   an auditor, or content work (not code we can write honestly here).
2. **§2 Running large models (GPT-3, Llama 3)** — the one area with real *engineering* work left.
   Op coverage is already complete (a full Llama-3 decoder layer runs and matches jax); what's
   missing is **scale**: numeric formats, weight loading, throughput, and distribution.

---

## 1. Remaining tracks (blocked on something external)

| § | Track | What's left | Why it can't be finished in-tree |
|---|-------|-------------|-----------------------------------|
| 1.1 | GPU security framework | SEV-SNP/TDX enclaves, HSM, FIPS-140 cert | confidential-computing **hardware** + an external **auditor** |
| 1.2 | Cryptography | homomorphic / threshold crypto, Intel QAT offload | research-grade scope / crypto-accelerator **hardware** |
| 3.1 | Windows deployment | DirectML backend, Active-Directory/Kerberos auth, Windows containers, PowerShell module | Windows-specific **APIs/SDKs** (CI builds+tests the engine on windows-2022 already) |
| 3.2 | macOS / Apple Silicon | Metal Performance Shaders backend | **Apple Silicon + Metal** hardware (engine itself builds/tests on macos-arm64 in CI) |
| 4.2 | ML frameworks | device-level `jax.jit(backend='vgre')` PJRT plugin **+ large-model scale (§2)** | PJRT needs upstream `pjrt_c_api.h` + MLIR C++ libs **not in the wheels**; large-model work is real and tracked in §2 |
| 4.3 | Model serving | TensorRT-LLM / vLLM *compatibility layers*, cross-instance model sharding, A/B-canary rollout | those external **runtimes** / a live **fleet** (the primitives — PagedAttention, continuous batching, speculative decoding — are built) |
| 5.3 | Multi-cloud | apply to live AWS/Azure/GCP, cross-cloud networking | cloud **accounts + credentials** (Terraform module is built + validated) |
| 6.1 | Developer tools | CUDA-GDB-compatible debug stepping | needs a PTX/SASS single-step interpreter (the JIT compiles to native; no per-instruction interpreter) — a large self-contained build, candidate for a future phase |
| 6.2 | Documentation & training | enterprise runbooks, video/tutorial content | **content-team** work, not code |
| 7.1 | Post-Blackwell (Rubin) | Rubin/HBM4 emulation | **unreleased** hardware, no public ISA |
| 7.2 | Multi-vendor | Intel oneAPI (SYCL/DPC++), Apple Metal | those **SDKs / hardware** (AMD HIP/ROCm layer is built) |
| 10.3 | Edge / CDN | physical edge nodes, CDN providers | external **infrastructure** (latency-aware routing is built) |

**6.1 (CUDA-GDB) is the only purely-software remaining track** — and it is a major build (a GDB
remote-serial-protocol server over a PTX single-step interpreter). Everything else needs hardware,
an account, an auditor, or content.

---

## 2. Running large models — GPT-3 (175B) & Llama 3 (8B / 70B / 405B)

**Where we are.** The VGRE HLO engine already runs real neural networks end-to-end (CNN /
Transformer / RNN / GPT block / ResNet block), trains them (backprop + SGD), generates
autoregressively, and does int8 quantized inference — all validated against the framework.
Crucially, **every Llama-3 / GPT-3 building block has been verified to run correctly on the
engine** (matched against jax):

| Llama 3 component | Decomposes to | Status |
|-------------------|---------------|--------|
| **RMSNorm** | reduce(x²) + rsqrt + mul | ✅ runs, matches jax |
| **RoPE** (rotary embeddings) | iota + sin/cos + slice + concat + mul | ✅ runs, matches jax |
| **SwiGLU / SiLU** | logistic + mul + dot | ✅ runs, matches jax |
| **Grouped-Query Attention** | dot_general + broadcast + softmax + causal mask | ✅ runs, matches jax |
| **Full Llama decoder layer** | the above + residuals | ✅ runs, matches jax |

So **op coverage is not the blocker** — a Llama-3 forward pass is expressible today. The blockers
are all about **scale**. In priority order:

> **Delivered toward this (2026-06-14):** the engine now has (i) a **work-stealing fork-join
> thread pool** (`vgre::xla::ThreadPool`) replacing per-op thread spawning — reentrant,
> concurrency-safe, `VGRE_XLA_THREADS`-bounded; cut a chained 256×256 matmul **148 ms → 59 ms**
> by amortizing thread creation (`XlaThreadPool` test 6/6); and (ii) **liveness-based buffer reuse**
> in `evaluateMulti` — each value's buffer is freed after its last use, so peak memory is the live
> set, not the sum of all intermediates (a 40-layer×512 model and a 400-op chain verified correct).
>
> **L2 (throughput) — DONE (2026-06-14):** `Dot` / `DotGeneral` now route through a real GEMM
> (`vgre::xla::gemm_f32`, `src/xla/blas_gemm.cpp`): **cblas_sgemm** when a BLAS is present, a
> built-in **cache-tiled** kernel otherwise, both parallelized across the thread pool (row-split
> for one big GEMM, batch-split for many small ones). `DotGeneral` detects the canonical batched-
> GEMM layout (leading batch dims, one contract+one free dim per side) and maps it to one batched
> GEMM with the right transpose flags; other shapes keep the correct generic loop. Replaced the
> cache-naive triple loop — **82× faster on a 4096³ GEMM** (`XlaBlasGemm` test: all transpose combos
> + batched verified vs reference; ≥10× criterion met). All JAX/TF/PyTorch/Keras/FlashAttention
> tests pass through the new path. The bf16 and quantized-weight pieces (L1/L3) remain.

### 2.A — Numeric formats & memory  *(the hard wall — highest priority)*
The engine stores everything as **fp32**. Real checkpoints are bf16/fp16; large models are served
int8/int4. Footprint of the weights alone:

| Model | params | fp32 | bf16 | int8 | int4 |
|-------|--------|------|------|------|------|
| Llama-3-8B | 8.0 B | 32 GB | 16 GB | 8 GB | **4 GB** |
| Llama-3-70B | 70 B | 280 GB | 140 GB | 70 GB | 35 GB |
| GPT-3 / Llama-3-405B | 175–405 B | 0.7–1.6 TB | 0.35–0.8 TB | … | … |

**Add:**
- **bf16/fp16 storage** in the engine's tensor type (a typed buffer instead of `std::vector<float>`),
  so weights load at native width — halves the memory wall and the bandwidth cost.
- **Quantized-weight path (int4/int8)** — load **GGUF / GPTQ / AWQ** checkpoints and dequantize
  per group inside the matmul kernel. The int8 `dot_general` path already exists; extend it to
  grouped int4. *This is the single biggest unlock: it puts an 8B model in 4 GB — laptop RAM.*

### 2.B — Weight loading at scale
Today model weights are **baked into the StableHLO as constants** (fine for toy models; a 16 GB
constant blob is not — the serializer would have to write it all). **Add:**
- Feed weights as **parameters / external buffers** rather than constants — the pytree-argument
  flattening path already does exactly this for training, so the plumbing exists.
- A **memory-mapped `safetensors` / `GGUF` loader** so a checkpoint is `mmap`'d once, not copied.
- A **tokenizer** binding (tiktoken / SentencePiece — external lib) for text-in / text-out.

### 2.C — Throughput  *(matmul is the bottleneck)*
The interpreter's matmul is a cache-naive triple loop (~148 ms for a chained 256×256). Llama-3-8B
is 32 layers of 4096×{4096, 14336} matmuls **per token** — thousands× more FLOPs; at interpreter
speed a single token is minutes. **Add:**
- ✅ **BLAS-backed `Dot` / `DotGeneral`** — **DONE (2026-06-14)**. Routes to **cblas_sgemm**
  (OpenBLAS/MKL/reference), with a cache-tiled fallback, parallelized over the thread pool; **82×**
  on a 4096³ GEMM. See the L2 note above. *(The single biggest throughput unlock — done.)*
- **Cache-tiled / blocked** kernels for the ops that don't map to BLAS (conv, reduce_window).
- A **fused attention** kernel (flash-attention-style online softmax — the `KVCacheManager`
  already implements online-softmax `pagedAttention`, so the algorithm is in-tree).
- Keep weights **resident across the generation loop** (compile once, execute per token) — the
  `shared_ptr<const HloModule>` executable design already supports this.

### 2.D — KV-cache generation at scale
Greedy decode + `dynamic_update_slice` KV-cache already work for a toy GPT. For long context
(8K–128K), wire the engine's generation loop to the **paged KV-cache** (`vgre::core::KVCacheManager`,
already built) so cache memory grows in blocks, and to the **continuous-batching scheduler**
(`ContinuousBatchScheduler`, already built) for multi-request serving.

### 2.E — Distributed execution  *(70B / 175B / 405B — don't fit one box)*
Models beyond one machine need **tensor parallelism** (shard each matmul's columns across ranks,
all-reduce the partials) and/or **pipeline parallelism** (assign layer ranges to ranks). The
**cluster substrate already exists** — RDMA transport (`rdma_transport.cpp`), NCCL-style collectives
(`collective_ops_manager.cpp`), elastic membership — but **the XLA engine is single-node**. Add a
**sharding pass** that splits `DotGeneral` / `Convolution` across ranks and inserts the collective
(all-reduce / all-gather) that a framework's `shard_map` / GSPMD annotations request, then drives it
over the existing RDMA/NCCL transport.

### Concrete milestones (in order)
1. **bf16 storage** + load a real **Llama-3-8B** checkpoint as parameters (fits 16 GB) → forward
   pass numerically matches Hugging Face.
2. ✅ **BLAS-backed matmul** — **DONE (2026-06-14)**: 82× on 4096³; `Dot`/`DotGeneral` use
   cblas_sgemm + tiled fallback, thread-pool parallel.
3. **int4 (GGUF) weight path** → 8B fits **4 GB**, runs on a laptop.
4. **Paged-KV generation loop** → long-context autoregressive decode + multi-request batching.
5. **Tensor-parallel `DotGeneral`** over the existing RDMA/NCCL substrate → **70B across N nodes**;
   pipeline parallelism → 175B / 405B.

**Honest framing:** items 1–4 are real, self-contained engineering achievable in-tree (no external
hardware) and would make VGRE genuinely run Llama-3-8B on a CPU; item 5 reuses the existing cluster
but is a substantial distributed-systems build. None of it requires a GPU — that is the whole point.
