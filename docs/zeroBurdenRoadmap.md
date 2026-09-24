# VGRE Zero-Burden Roadmap — a self-contained, lightweight engine (0 → 100%)

**Date:** 2026-09-06 (progress updated 2026-09-22) · **Status:** master plan for
the next major version — **Phases A, B, C and the threading half of D are done**:
`-DVGRE_ENABLE_JIT=OFF` builds and runs with **no LLVM**, `-DVGRE_ENABLE_OPENMP=OFF`
builds green on the in-tree thread pool alone, and the from-scratch CUDA-C
front-end + Tier-0/Tier-1 backends are held bit-exact by a differential-fuzzing
suite. **Phase C is now delivered too**: a from-scratch **native x86-64 machine-code
JIT** (`src/compiler/frontend/native_kernel_x64.cpp`) is the default execution tier —
it emits real machine code for the scalar CUDA-C subset (all int/float arithmetic,
comparisons, ternary select, casts, scalar locals, general gather/scatter, bounded
loops, GEMM in both flattened and true-2-D form, and the full math-function surface
incl. transcendentals via libm calls), is differential-fuzzed bit-exact against the
Tier-1 compiled backend, and measures **~18–118× faster** than it (saxpy, `NativePerf`).
Anything outside the subset (e.g. `__syncthreads`) falls back to the compiled tier,
then the interpreter. **What's left:** the *optional* Tier-2 SSA backend (peak speed
for hot kernels) and Phase E packaging.

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
| **LLVM 18 + Clang** | **now OPTIONAL** — `option(VGRE_ENABLE_JIT)`; `OFF` skips `find_package(LLVM)` entirely | ~1 GB; the slow/never-finishing Windows step | ✅ **eliminated** in the LLVM-free build (this doc's core) |
| OpenMP | **now OPTIONAL** — `option(VGRE_ENABLE_OPENMP)`; `OFF` uses the in-tree thread pool | small (ships with compiler) | ✅ replaced by the in-tree work-stealing thread pool |
| A C++17 compiler | REQUIRED | unavoidable | keep (the one true dep) |
| BLAS/LAPACK, FFTW, SuiteSparse | optional, auto-detected | medium | in-tree fallbacks (GEMM done) |
| OpenSSL, zlib, SQLite | optional, auto-detected | small | in-tree fallbacks / vendored-lite |
| gRPC/Protobuf, ibverbs, OpenCL, TPM, Flutter, Go | optional feature-gated | large but off by default | leave optional |

**Conclusion (achieved):** the only *hard* burden worth eliminating was **LLVM**,
and the LLVM-free build now runs with **just a C++ compiler** (OpenMP dropped
too). Everything else already degrades gracefully or is off by default. The
*speed* work is now largely done too: the native x86-64 JIT (Tier 1b) is the
default and runs ~18–118× faster than the portable compiled tier. Only the
*optional* Tier-2 SSA backend (peak throughput on hot kernels) remains.

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
- **Tier 1 — Compiled backend (DONE).** The CUDA-C AST is lowered **once** to
  slot-based bound closures (`include/vgre/compiler/frontend/compiled_kernel.h`)
  and executed per-thread with zero string parsing — **~32–200× faster than the
  interpreter**, portable (pure C++, every arch), LLVM-free. Barrier-free kernels run
  the closures directly; **cooperative kernels** (`__shared__`, `__syncthreads`,
  `__syncwarp`, `__shfl_*`, warp vote/reduce/match, `__syncthreads_{count,and,or}`,
  `__activemask`) run on a **fiber executor** — each CUDA thread a stackful fiber,
  a per-block scheduler releasing barriers/warp ops — so tiled GEMM, block & warp
  reductions and shared-memory attention no longer fall back to Tier 0. The executor
  is portable to every host (`ucontext` on POSIX incl. macOS, the Win32 Fibers API on
  Windows), with no interpreter fallback (bit-exact vs the interpreter,
  `test_cuda_coop.cpp`). Selected via `VGRE_EXEC_BACKEND=compiled`.
  - **Tier 1b — native x86-64 JIT (DONE).** `native_kernel_x64.cpp`: a from-scratch
    hand-written machine-code emitter (not CI-baked stencils) that mmaps W^X memory
    and emits real x86-64 for the scalar subset — all int/float arithmetic, bitwise/
    shift, comparisons (as conditions and values), ternary select, casts (round +
    saturate/quantize), scalar locals + reassignment, general-index gather/scatter,
    bounded `for`-loop reductions, GEMM (flattened *and* true-2-D `threadIdx.y`), and
    the full math surface (`sqrtf`/`rsqrtf`/`fabsf`/`fminf`/`fmaxf`/`min`/`max`/`powf`
    + `expf`/`logf`/`sinf`/`cosf`/`floorf`/`ceilf` via libm calls). Arch-specific
    (x86-64 Linux; other hosts fall back), so it layers on top of the portable
    compiled tier. **Default tier** in the no-LLVM build; ~18–118× over Tier 1,
    differential-fuzzed bit-exact. Copy-and-patch stencils were the original plan;
    the direct emitter reaches the same goal with no build-time stencil step.
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

**Phase A — Decouple & prove the fallback (foundation). ✅ DONE.**
1. Drop `llvm::json` → in-tree `vgre::common::json` (removes LLVM from cluster). ✅
2. Promote `ptx_interpreter` to a runtime `ExecutionBackend`; route
   `runtime_engine` kernel launch through a backend interface. ✅
3. Add `VGRE_EXEC_BACKEND`; run the full kernel suite on the interpreter tier. ✅
   *Exit met:* a build with **LLVM absent** passes the kernel tests (interpreter).

**Phase B — Own front-end. ✅ DONE.**
4. Hand-written CUDA-C lexer/parser → PTX for the documented subset. ✅
5. Publish the "supported CUDA-C subset" matrix; grow it test-first. ✅
   (`supportedCudaSubset.md`; a full flash-attention kernel compiles with no Clang,
   and both tiers are differential-fuzzed bit-exact.)

**Phase C — Fast codegen (native machine code). ✅ DONE.**
6. Built a from-scratch **native x86-64 JIT** (`native_kernel_x64.cpp`) rather than a
   CI-baked copy-and-patch stencil set: a hand-written machine-code emitter that
   mmaps W^X memory and emits real x86-64 for the scalar elementwise/reduction/GEMM
   subset. LLVM-free, no build-time stencil step. It is wired in as the **default**
   tier (`RuntimeEngine::makeBackendKernel` / the C-ABI dispatch try it first, then
   the compiled tier, then the interpreter), with `VGRE_DISABLE_NATIVE=1` to opt out.
7. Benchmarked (`test_native_perf.cpp`, `NativePerf`): native is **~18–118× faster
   than the Tier-1 compiled backend** and ~1800–5700× faster than the interpreter on
   saxpy, and every native op is held **bit-exact** against the compiled tier by the
   `CudaNative` differential-fuzzing suite (float / int / quant / locals / gather /
   scatter / reduce / 2-D sweeps, 800 kernels each).
   *Exit met:* a native-code speed tier exists, is the default, and is far faster than
   the portable Tier-1 backend — without reintroducing LLVM.

**Phase D — Peak backend (optional) + threading.**
8. In-tree thread pool becomes the only threading requirement (OpenMP optional). ✅ DONE
9. Tier 2 SSA backend for hot kernels (linear-scan regalloc + native emitter). 🚧 STARTED
   - **Increment 1 ✅ DONE**: VGRE-IR (typed SSA: basic blocks + br/condbr/ret) +
     AST→SSA lowering for the scalar elementwise subset (`if`-guarded bodies,
     ternary→select) + a well-formedness verifier + a reference evaluator. Held
     bit-exact against the Tier-1 compiled backend by `test_ssa_ir.cpp`
     (`src/compiler/frontend/ssa_ir.cpp`). This is the foundation the passes and
     the emitter build on.
   - **Increment 2 ✅ DONE**: full structured control flow — `if`/`if-else`, `for`,
     `while`, inc/dec — with on-demand **SSA phi insertion** (Braun et al.: per-block
     current definitions, incomplete phis in unsealed loop headers). Loop-carried
     values and if/else merges lower to real phi nodes; the evaluator resolves them
     by arrival edge. Held bit-exact vs the compiled tier by `test_ssa_ir.cpp`
     (for-sum / while / if-else / nested loop+conditional).
   - **Increment 2b ✅ DONE**: the rest of structured control flow — `do-while`,
     `break`, `continue` — via a loop-context stack + latch/exit blocks (phi
     construction handles the extra edges). Diffed vs the Tier-0 interpreter (the
     compiled tier lacks these), bit-exact.
   - **Increment 3 ✅ DONE**: the optimizer passes — **constant folding**, **local
     value numbering** (safe in-block GVN/CSE), **dead-code elimination**, and
     **loop-invariant code motion** (dominator sets by iterative dataflow → natural
     loops via back-edges → hoist pure invariants to the preheader). Pure IR→IR,
     run in `SsaProgram::compile`. `test_ssa_ir.cpp` checks each stays bit-exact vs
     the reference tiers AND shrinks/hoists (fold 32→22, cse-dce 28→23, licm hoists).
   - **Increment 4 ✅ DONE**: **native x86-64 machine-code emission**. A slot-per-value
     emitter lowers the SSA to real machine code (mmap W^X): hot ops (int/float
     arithmetic, compares, load/store, branches, phi edge-copies, const/param/tid,
     select) inlined; the tricky ones (float↔int saturating casts, integer div/mod,
     float compares, math intrinsics) delegated to bit-exact C-ABI helpers. Control
     flow is block-structured with phi destruction, so ifs and loops both compile.
     `SsaProgram::launch()` runs the emitted code (`usedNative()` reports it), matching
     the compiled tier bit-for-bit (`test_ssa_ir.cpp`, native probe). x86-64/Linux;
     other hosts keep running the portable evaluator, so Tier-2 works everywhere.
   - **Increment 5 ✅ DONE**: register allocation. (a) A **redundant-load-elimination
     peephole** keeps each op's result live in rax/xmm0 and skips reloading it as the
     next op's first operand. (b) **Global linear-scan register allocation** (Poletto &
     Sarkar): integer/pointer values get a callee-saved register **r12–r15** for their
     whole live range **across blocks and loop iterations** — computed from a full
     backward liveness dataflow + conservative live intervals — so e.g. a pointer param
     stays in a register through a loop instead of being reloaded each iteration.
     Callee-saved ⇒ safe across the emitter's helper calls (no spill-around-call). The
     emitter reaches every value through the same register-or-slot accessor, so the
     allocator alone drives it. Verified bit-exact across the whole `test_ssa_ir` suite
     (nested loops, break/continue, do-while, if/else, optimizer, LICM), native running.
   - **Remaining**: **phi register allocation** (loop-carried accumulators/counters into
     registers — the interval must also cover the edge-copy writes at every predecessor
     terminator; kept in slots for now pending a fix for a nested-loop overlap case);
     float/XMM allocation (spill-around-call); an AArch64 emitter; cross-block GVN;
     `switch`; and wiring Tier-2 into the runtime as `VGRE_EXEC_BACKEND=ssa` (a small
     AST→SSA adapter; the backend is complete via `SsaProgram`).

**Phase E — Packaging the zero-burden promise.**
10. Single-command install that needs only a compiler; prebuilt wheels/binaries
    with baked stencils so end users compile nothing. Refresh the docs site.

---

## 5. Success criteria (global gate)

- VGRE **builds and runs the kernel + model suites with LLVM absent** (Tier 0),
  with the compiled closure tier and the native x86-64 JIT (Tier 1 / 1b) for speed.
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
