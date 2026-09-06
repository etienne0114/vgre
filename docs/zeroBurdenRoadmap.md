# VGRE Zero-Burden Roadmap — a self-contained, lightweight engine (0 → 100%)

**Date:** 2026-09-06 · **Status:** master plan for the next major version.

## Mission (why this project exists)

Make advanced AI **accessible to everyone**: let anyone run and train big models,
and solve complex algorithms, on the computer they already own — no GPU, no
expensive hardware, no heavyweight toolchain to install first. VGRE should be
something a person **downloads and uses**, not something they must assemble from
a gigabyte of third-party SDKs. We hit that by building the pieces we need
**from scratch, in-tree, with lean algorithms and data structures**, so the
engine stays small and fast and depends on almost nothing.

The `libvgre_nn` library already proves the model is achievable: a complete
from-scratch ML stack (autograd, transformer/Mamba/MoE, quantization, tokenizer,
cluster all-reduce) that is **LLVM-free, BLAS-free, CUDA-free**. This roadmap
extends that same discipline to the **whole engine**.

---

## 1. The burden today (measured) — what a user/CI must install

| Dependency | Status in build | Weight | Verdict |
|---|---|---|---|
| **LLVM 18 + Clang** | **REQUIRED** (`find_package(LLVM REQUIRED)`) | ~1 GB; the slow/never-finishing Windows step | **eliminate** (this doc's core) |
| OpenMP | REQUIRED | small (ships with compiler) | replace with in-tree thread pool |
| A C++17 compiler | REQUIRED | unavoidable | keep (the one true dep) |
| BLAS/LAPACK, FFTW, SuiteSparse | optional, auto-detected | medium | in-tree fallbacks (GEMM done) |
| OpenSSL, zlib, SQLite | optional, auto-detected | small | in-tree fallbacks / vendored-lite |
| gRPC/Protobuf, ibverbs, OpenCL, TPM, Flutter, Go | optional feature-gated | large but off by default | leave optional |

**Conclusion:** the only *hard* burden worth eliminating is **LLVM**. Remove it
and VGRE builds and runs with **just a C++ compiler**. Everything else already
degrades gracefully or is off by default.

---

## 2. What LLVM does for us, and the from-scratch replacement

LLVM is used for exactly one capability: **turn a CUDA C kernel into executable
machine code** (Clang parses CUDA C → LLVM IR → ORC JIT → native). To replace it
from scratch we need three components, each of which has a proven lightweight
design in the literature and a partial head-start already in-tree.

### 2.1 Front-end: CUDA C → VGRE-IR (our own)
- **Have:** a regex `src/compiler/kernel_parser.cpp` (signature/body extraction)
  and a from-scratch **PTX translator** (`src/compiler/ptx/`).
- **Build:** a real hand-written **lexer + recursive-descent parser** for the
  CUDA-C subset we care about (types, `__global__/__device__`, `threadIdx/
  blockIdx/blockDim`, indexing, arithmetic, `if/for/while`, `__syncthreads`,
  shared memory, math intrinsics), lowering to a small typed **VGRE-IR** (SSA).
  Grow the supported subset test-first; publish a "supported CUDA-C" matrix.
- **Data structures:** arena-allocated AST + SSA value graph; string-interned
  identifiers; no third-party parser generator.

### 2.2 Middle/back-end: VGRE-IR → native, three tiers (ship in order)

The research maps cleanly onto a **layered execution backend** — a simple
universal tier first, faster tiers added without breaking correctness:

- **Tier 0 — SIMD interpreter (zero codegen).** Promote the existing
  `src/debug/ptx_interpreter.cpp` (693 LOC — registers, memory spaces, barriers,
  predication) to a first-class runtime `ExecutionBackend`, and vectorize the
  warp loop (execute 8/16 threads per SIMD lane). **No machine-code generation at
  all → works on every OS/arch instantly, zero deps.** This is the MVP and the
  permanent fallback.
- **Tier 1 — Copy-and-patch codegen.** Bake **stencils** (precompiled binary
  templates with holes) for each VGRE-IR op at **VGRE build time** using the
  ordinary compiler, ship them inside the library, and at runtime **copy + patch**
  the holes to stitch a kernel's machine code. This is exactly the technique in
  CPython 3.13's JIT: near-native speed, tiny runtime, and — crucially — **no
  LLVM on the user's machine and none at runtime** (Copy-and-Patch, Xu &
  Kjolstad, arXiv:2011.13127).
- **Tier 2 — Own SSA optimizing backend (long-term).** A MIR/QBE-class in-tree
  backend: VGRE-IR (SSA) → a few classic passes (const-fold, DCE, GVN, LICM) →
  **linear-scan register allocation** (Poletto & Sarkar, O(n), JIT-grade) → a
  direct **x86-64 / AArch64 machine-code emitter**. MIR shows a complete such
  backend is ~10K LOC with ~70% of GCC-O2 performance; QBE shows the SSA+
  optimizer core stays small. Optional, for peak throughput on hot kernels.

Backend chosen at runtime (`VGRE_EXEC_BACKEND=interp|cp|ssa`), Tier 0 as the
guaranteed fallback. **Correctness is shared** by running the existing kernel
test-suite against every tier and diffing outputs.

### 2.3 Why this satisfies the mission
- **User installs nothing but VGRE** — no LLVM, no CUDA SDK. Tier 0 needs only a
  C++ compiler to have built VGRE; Tier 1 stencils are baked once in CI.
- **Windows build stops being painful** — the ~1 GB LLVM download is gone.
- **Still fast** — Tier 1/2 recover near-native speed for the kernels that matter.

---

## 3. Other from-scratch pieces (extend the `libvgre_nn` discipline)

| Area | Have | Build from scratch |
|---|---|---|
| **Threading** | uses OpenMP + an in-tree pool | make the in-tree work-stealing thread pool the only requirement; drop the REQUIRED OpenMP |
| **Linear algebra** | SIMD GEMM (fp32/bf16), 82× BLAS path optional | keep the in-tree SIMD GEMM as the default; BLAS strictly optional accelerator |
| **Compression** | zlib/LZ4 optional | small in-tree LZ4/RLE for checkpoints + cluster transfers |
| **Hashing/JSON/crypto** | in-tree `common/json`, some OpenSSL | finish in-tree JSON everywhere (drop `llvm::json`); vendored-lite SHA/AES for the non-TLS paths |
| **Model I/O** | mmap safetensors + GGUF (many quant types) | keep; add streaming/partial-load for models bigger than RAM |
| **Data structures** | HNSW (RAG), paged KV, arena allocators | the theme: cache-aware, SIMD-friendly, allocation-light structures everywhere |

**Extreme-algorithm levers already in-tree that make big models fit small
machines** (keep pushing these — they are the mission): ternary/BitNet
multiplication-free GEMM, MoE compute-sparsity, MXFP4/int4 microscaling, QLoRA,
Mamba/SSM (no KV cache), speculative + multi-token decoding, int4/int8 KV cache,
2:4 structured sparsity. See `missingFeatures.md` for their verified status.

---

## 4. The 0 → 100% program (phased, test-first, no stubs)

**Phase A — Decouple & prove the fallback (foundation).**
1. Drop `llvm::json` → in-tree `vgre::common::json` (removes LLVM from cluster).
2. Promote `ptx_interpreter` to a runtime `ExecutionBackend`; route
   `runtime_engine` kernel launch through a backend interface.
3. Add `VGRE_EXEC_BACKEND`; run the full kernel suite on the interpreter tier.
   *Exit:* a build with **LLVM absent** passes the kernel tests (interpreter).

**Phase B — Own front-end.**
4. Hand-written CUDA-C lexer/parser → VGRE-IR for the documented subset.
5. Publish the "supported CUDA-C subset" matrix; grow it test-first.
   *Exit:* the example kernels (vector add, GEMM, reductions, stencil, the docs'
   samples) compile through VGRE-IR with no Clang.

**Phase C — Fast codegen (copy-and-patch).**
6. Stencil generator run in CI; runtime copy-and-patch emitter for VGRE-IR.
7. Benchmark Tier 1 vs the old LLVM JIT on the kernel corpus.
   *Exit:* Tier 1 within a target factor of the LLVM JIT on the corpus; **LLVM
   removed from the default build.**

**Phase D — Peak backend (optional) + threading.**
8. In-tree thread pool becomes the only threading requirement (OpenMP optional).
9. Tier 2 SSA backend for hot kernels (linear-scan regalloc + native emitter).

**Phase E — Packaging the zero-burden promise.**
10. Single-command install that needs only a compiler; prebuilt wheels/binaries
    with baked stencils so end users compile nothing. Refresh the docs site.

---

## 5. Success criteria (global gate)

- VGRE **builds and runs the kernel + model suites with LLVM absent** (Tier 0),
  and with copy-and-patch for speed (Tier 1).
- **No new third-party runtime dependency** is introduced (from-scratch only);
  `libvgre_nn` stays LLVM/BLAS/CUDA-free and the whole engine trends that way.
- Every new component ships with a numerical/behavioural test vs an independent
  reference and is exercised end-to-end (no stubs, no placeholders).
- A non-technical user can install VGRE and run a model **without installing any
  compiler toolchain SDK** (prebuilt binary + baked stencils).

---

## 6. Research basis (internet)

- **Copy-and-Patch Compilation**, Xu & Kjolstad — fast codegen by stitching
  precompiled stencils; the basis of CPython 3.13's JIT (PEP 744). arXiv:2011.13127.
- **MIR** (V. Makarov, Red Hat) — a complete lightweight JIT/back-end in ~10K
  LOC, ~70% of GCC-O2 performance, ~100× faster compile. Design reference for
  Tier 2.
- **QBE** — a "mini-LLVM": small SSA IR + compact optimizer + x86-64/arm64.
- **Linear-scan register allocation**, Poletto & Sarkar (TOPLAS 1999) — O(n)
  single-pass allocation, JIT-grade, simple to implement.
- **TinyCC** — a whole C compiler with its own assembler/linker as evidence a
  self-contained front-to-back toolchain is feasible in a small codebase.
