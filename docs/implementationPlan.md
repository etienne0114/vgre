# VGRE Implementation Plan — Remaining Work

**Last Updated**: 2026-06-16

The original Phase-4 enterprise roadmap (50 tracks) **and** the large-model programme (L1–L5) have
been delivered down to their software-implementable core — every track buildable to the project's
"real, no-stub" standard without external hardware/accounts is done, tested, and committed (see
`missingFeatures.md` and git history for the full delivered set). This plan now contains **only the
remaining work**: a short list of externally-blocked tracks, and the two large-model items that
need a real cluster or a gated download.

---

## A. Externally-blocked tracks (not code we can write here)

These need hardware, an account/credential, an auditor, or content work. The in-tree primitives
already exist where applicable.

| Track | Remaining | Blocker |
|-------|-----------|---------|
| Security framework | SEV-SNP/TDX, HSM, FIPS-140 cert | confidential-computing hardware + auditor |
| Cryptography | homomorphic/threshold crypto, Intel QAT | research scope / crypto-accelerator hardware |
| Windows deployment | DirectML, AD/Kerberos, containers, PowerShell | Windows APIs/SDKs (builds+tests on windows-2022) |
| macOS / Apple Silicon | Metal Performance Shaders backend | Apple Silicon + Metal hardware (builds+tests on macos-arm64) |
| ML frameworks | device-level `jax.jit(backend='vgre')` PJRT plugin | upstream `pjrt_c_api.h` + MLIR C++ libs not in the wheels |
| Model serving | TensorRT-LLM/vLLM compat layers, A/B-canary | external runtimes / live fleet |
| Multi-cloud | apply to live AWS/Azure/GCP | cloud accounts + credentials |
| Multi-vendor | Intel oneAPI (SYCL), Apple Metal | those SDKs/hardware (AMD HIP/ROCm done) |
| Edge / CDN | physical edge nodes, CDN | external infrastructure |
| Docs & training | enterprise runbooks, video content | content-team work |
| Post-Blackwell (Rubin) | Rubin/HBM4 emulation | unreleased hardware, no public ISA |

**One purely-software remaining track:** **CUDA-GDB-compatible debugging** — a GDB
remote-serial-protocol server backed by a PTX/SASS single-step interpreter (the JIT currently
compiles to native, so there is no per-instruction interpreter to step). A large self-contained
build; a candidate for a dedicated future phase.

---

## B. Large models — remaining (external only)

The L1–L5 programme is **complete in-tree** (bf16/int4 storage, safetensors + GGUF + GPTQ/AWQ
loaders, BLAS GEMM + dequant-in-GEMM, generation + samplers + GPT-2 tokenizer, tensor/pipeline
parallelism + GSPMD + RDMA collectives) and verified end-to-end on **real GPT-2 (124M)** matching
Hugging Face. The two remaining items are environmental:

| Remaining | Blocker |
|-----------|---------|
| **Physical multi-node run** — Llama-3-70B tensor-parallel across N machines (175B/405B pipeline) | Transport, collectives, the tensor/pipeline executor, and the GSPMD auto-partitioner are built and verified over real loopback sockets; a cross-machine run needs **actual networked machines**. |
| **Frontier-scale checkpoints** — Llama-3-8B/70B/405B, GPT-3 | The identical code path GPT-2 already runs end-to-end; it just needs a **license-gated, multi-GB download** (Llama-3 gated; 8B = 16 GB bf16 / ~4.5 GB int4). |

**Optional in-tree polish** (not blocking — existing paths already cover these): a dedicated fused
flash-attention kernel (online-softmax already in `KVCacheManager`; BLAS handles attention matmuls),
and parsing the unified HF `tokenizer.json` (the exact GPT-2 `vocab.json`+`merges.txt` path is done).

---

## Success criteria

| | Criterion | Status |
|--|-----------|--------|
| **L1** | Real model forward output matches HF reference | ✅ **met** — GPT-2 matches HF (f32 & bf16); Llama-3-8B is the same path behind a gated 16 GB download |
| **L2** | ≥10× matmul speedup vs the naive loop on a 4096² GEMM | ✅ **met — 82×** |
| **L3** | int4 weights run a real model at a fraction of fp32 RAM | ✅ **met** — GPT-2 from Q4_0/Q8_0/Q4_K GGUF + GPTQ/AWQ; dequant-in-GEMM keeps weights compressed through compute |
| **L4** | autoregressive decode with paged KV + a real tokenizer | ✅ **met** — text→text reproduces HF GPT-2 greedy generation token-for-token |
| **L5** | tensor-parallel correctness matching a single-node reference | ✅ **met in-process / over real RDMA sockets**; a physical N-node run needs real machines |
