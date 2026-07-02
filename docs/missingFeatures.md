# VGRE — Remaining Work

**Last Updated**: 2026-06-22

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

A **2026-06-22 deep audit of the CUDA-C JIT execution path** found one previously-undocumented
**in-tree** gap — the device-side intrinsic surface — now closed (§0). It is recorded here for
provenance even though the work is complete.

---

## 0. Device-side CUDA intrinsics in the JIT path — FOUND & FIXED (2026-06-22)

**What was found.** VGRE JIT-compiles a CUDA-C kernel by emitting
`#include "vgre/compiler/cpu_cuda_env.h"` ahead of the user source and compiling it as host C++
(`src/compiler/llvm_translation_engine.cpp:1300`). That environment header only declared four
atomics (`atomicAdd/Sub/Exch/CAS`) and a handful of helpers. A large slice of the **standard
device-intrinsic surface that real CUDA / ML / DL kernels depend on was missing**, so any kernel
using it compiled during AST analysis (the analysis stub forward-declared *some* of them) but then
**failed JIT compilation** with `use of undeclared identifier`. Confirmed empirically by compiling
probe kernels against the header. Missing set:

| Group | Missing intrinsics (now added) |
|-------|--------------------------------|
| Atomics | `atomicMax`, `atomicMin`, `atomicAnd`, `atomicOr`, `atomicXor`, `atomicInc`, `atomicDec` (CUDA wrap-around semantics) |
| Read-only / fences | `__ldg`, `__threadfence`, `__threadfence_block`, `__threadfence_system` |
| Block vote | `__syncthreads_count`, `__syncthreads_and`, `__syncthreads_or` (needed new runtime barrier-reduce) |
| Warp reduce | `__reduce_{add,min,max,and,or,xor}_sync` (sm_80+) |
| Bit reinterpret | `__int_as_float`, `__float_as_int`, `__uint_as_float`, `__float_as_uint`, `__double_as_longlong`, `__longlong_as_double`, `__double2hiint/loint`, `__hiloint2double` |
| Integer | `__mul24`, `__umul24`, `__mulhi`, `__umulhi`, `__mul64hi`, `__umul64hi`, `__sad`, `__usad`, `__byte_perm` |
| Rounded arith | `__fmaf_rn/rz/ru/rd`, `__fadd_rz/ru/rd`, `__fmul_*`, `__frcp_rn`, `__fdiv_rn`, `__fsqrt_rn`, `__frsqrt_rn`, and the `__d*_rn` double forms |
| Fast math | `__expf`, `__exp2f`, `__exp10f`, `__logf`, `__log2f`, `__log10f`, `__sinf`, `__cosf`, `__tanf`, `__powf`, `__sincosf` |
| CUDA-named math | `rsqrtf`, `rsqrt`, `rcbrtf`, `norm3df`, `rnorm3df`, `norm4df`, `rnorm4df`, `normf`, `rnormf`, `rhypotf`, `sinpif`, `cospif`, `sincospif` |
| Built-in vector types | `float1..4`, `double1..4`, `int1..4`, `uint1..4`, `char/uchar/short/ushort/long/ulong N`, `longlong/ulonglong N` + all `make_*` |
| Misc | `__saturatef`, `__nanosleep`, `__trap` |

**How it was fixed.**
- New header `include/vgre/compiler/cpu_cuda_intrinsics.h` providing exact CPU implementations of
  every intrinsic above plus the built-in vector types; included from `cpu_cuda_env.h`. Fast-math
  names that collide with glibc's (declared-but-unlinkable) internal libm symbols are aliased to the
  public libm function via guarded function-like macros, so they resolve on glibc, musl, macOS and
  Windows alike.
- Atomics (`Max/Min/And/Or/Xor/Inc/Dec`) added to `cpu_cuda_env.h` with correct CUDA semantics
  (CAS loops; `Inc/Dec` wrap-around).
- `__syncthreads_count/_and/_or` required true block-wide reduction: added
  `BlockBarrier::arrive_and_reduce` + `vgre_jit_block_barrier_reduce` runtime hook
  (`gpu_thread_context.{h,cpp}`), registered as a JIT symbol.
- `__reduce_*_sync` built as a butterfly over the existing `__shfl_xor_sync`.
- The AST-analysis stub (`clang_kernel_parser.cpp`) was extended with matching declarations + vector
  types so the analysis pass and the JIT pass agree.
- Verified by `tests/integration/test_device_intrinsics.cpp` (atomics, `__ldg`, fences, block-vote,
  vector-typed `float4` vectorized load, warp reduce) executed end-to-end through the JIT.

**Status: complete.** No remaining device-intrinsic gap is known in the JIT path.

### 0.1 Correctness/robustness bugs found & fixed in the same audit (2026-06-22)

- **AST-stub missing `sharedMem`** — the Clang AST-analysis stub never declared the
  `sharedMem` pseudo-variable, so any kernel using dynamic shared memory failed AST analysis on
  a *cold* cache (`use of undeclared identifier 'sharedMem'`). It only appeared green because CI
  ran with warm KernelIR caches. Declared `extern void* sharedMem;` in the stub.
- **Warp-buffer under-detection** — the per-block warp exchange buffer was only allocated when the
  source textually contained `__shfl*`; `__any_sync`/`__all_sync`/`__reduce_*_sync` route through
  shuffle/ballot internally and silently fell back to a thread's own value. Broadened the scan in
  both parser paths.
- **AST temp-file race** — `runClangAstDump` wrote every kernel to a single fixed temp path
  (`/tmp/vgre_kernel_tmp.cu`); concurrent AST analyses (separate processes/threads) clobbered each
  other, so clang read a half-written file and aborted. This surfaced as flaky
  `ClangParser/ClangEnhanced/KernelParserEnhanced` failures under `ctest -j`. Now keyed on
  pid + source-hash + atomic counter, matching the already-unique JIT codegen temp names.
  Full suite verified **281/281 green at `ctest -j6` with a fresh cache.**

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
| 7.2 | Multi-vendor | Intel oneAPI (SYCL/DPC++), Apple Metal; hipBLAS/hipDNN library shims | those **SDKs / hardware** (AMD HIP core runtime — device mgmt, streams, events, memory, module load + JIT kernel launch — is built and tested end-to-end as of 2026-07-02; the ROC library shims above remain) |
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
