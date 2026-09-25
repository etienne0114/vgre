# VGRE Zero-Burden Roadmap — a self-contained, lightweight engine (0 → 100%)

**Date:** 2026-09-06 (progress updated 2026-09-25) · **Status:** master plan for
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
Anything outside the subset falls back to the compiled tier, then the interpreter.
**What's left:** the *optional* Tier-2 SSA backend is now feature-complete for the
scalar + shared-memory + warp subset (native x86-64, opt-in AArch64, portable
evaluator elsewhere) and wired as `VGRE_EXEC_BACKEND=ssa`; the whole subset — including
shared memory and every warp intrinsic — now emits **native x86-64 machine code**. Its
remaining items are hardware-gated only (flipping ARM native on after real-HW
validation; native AArch64 codegen for the cooperative ops). Then Phase E packaging.

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
default and runs ~18–118× faster than the portable compiled tier. The *optional*
Tier-2 SSA backend is now feature-complete for the scalar + shared-memory + warp
subset — native on x86-64 (shared memory and warp intrinsics included), opt-in AArch64,
portable evaluator elsewhere — wired as `VGRE_EXEC_BACKEND=ssa`, tier 3; only
hardware-gated items remain (flipping ARM native on after real-HW validation).

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
- **Tier 2 — Own SSA optimizing backend (feature-complete for its subset).** A
  MIR/QBE-class in-tree backend: VGRE-IR (SSA) → classic passes (const-fold, DCE,
  cross-block/dominator-scoped GVN, LICM) → **linear-scan register allocation**
  (Poletto & Sarkar, O(n), JIT-grade; GPR + XMM, loop-carried phis) → a direct
  **x86-64 / AArch64 machine-code emitter**. Covers the scalar + shared-memory +
  warp subset (full control flow incl. `switch`, `__device__` inlining, local
  arrays, `__shared__`/`__syncthreads`, and the full warp-intrinsic surface); the
  cooperative shared/warp ops run on a per-block evaluator everywhere and as **native
  ucontext-fiber machine code on x86-64** (`vgre_ssa_barrier`/`vgre_ssa_warp` + a block +
  per-warp selective-release scheduler), AArch64 shared/barrier native (opt-in). Wired as
  `VGRE_EXEC_BACKEND=ssa`. Optional, for peak throughput on hot kernels.

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
9. Tier 2 SSA backend for hot kernels (linear-scan regalloc + native emitter). ✅ FEATURE-COMPLETE for the scalar + shared-memory + warp subset (native x86-64; AArch64 opt-in; portable evaluator elsewhere), wired as `VGRE_EXEC_BACKEND=ssa` (tier 3).
   - **Delivered (increments 1–20)**, held bit-exact vs the interpreter/compiled tiers by
     `test_ssa_ir.cpp` / `test_ssa_backend.cpp`: VGRE-IR + Braun-phi lowering + verifier +
     evaluator; full control flow (`if`/`for`/`while`/`do-while`/`break`/`continue`/`switch`);
     optimizer (const-fold, cross-block/dominator-scoped GVN, LICM, DCE); linear-scan
     register allocation (GPR **r12–r15** + XMM **xmm2–xmm7**, loop-carried phis); native
     **x86-64** emitter + opt-in **AArch64** (`Arm64Asm`, encodings llvm-mc-verified);
     `__device__` inlining, per-thread local arrays, extended math; and **native x86-64
     machine code for shared memory + the full warp-intrinsic surface** (`__shfl_*`, vote,
     `__reduce_*_sync`, `__match_{any,all}_sync`, `__syncwarp`, `__activemask`) via
     `vgre_ssa_barrier`/`vgre_ssa_warp`/`vgre_ssa_activemask` on ucontext fibers with a
     block + per-warp selective-release scheduler. (Full increment log: git history and the
     `tier2_ssa_backend` project memory.)
   - **Remaining (hardware-gated / breadth, not correctness):** flip `VGRE_SSA_ARM_NATIVE`
     on by default after validating native AArch64 execution on real ARM hardware; native
     AArch64 codegen for the warp/shared cooperative ops (x86-64 is fully native, ARM uses
     the evaluator); and the tiered fallbacks for out-of-subset constructs (multi-dim
     arrays, struct params, dynamic `extern __shared__`).

**Phase E — Packaging the zero-burden promise.**
10. Single-command install that needs only a compiler; prebuilt wheels/binaries
    with baked stencils so end users compile nothing. Refresh the docs site. ✅ (v0.1.0
    ships self-contained LLVM-free wheels for Linux/macOS/Windows; PyPI publish + an
    x86-64 macOS wheel are the remaining polish — see `missingFeatures.md`.)

**Phase F — Closing the performance gap to a real GPU (the next frontier).**

See **§7** below. In one line: remove the *emulation tax* so kernels hit the CPU's own
roofline, use the CPU's tensor units + reduced precision to raise that roofline, and use
algorithmic levers to cut the *work* — so a commodity CPU finishes a job sized for a GPU.

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

---

## 7. Closing the performance gap to a real GPU

VGRE runs on CPUs, so it cannot match a datacenter GPU's raw silicon. But the gap has
**two separable parts**, and only one of them is physics.

### 7.1 The gap, measured honestly

| | Datacenter GPU (A100) | AVX-512 server CPU (32c) | Desktop (AVX2, 8c) |
|---|---|---|---|
| fp32 peak | ~19.5 TFLOP/s | ~2–4 TFLOP/s | ~0.5–1 TFLOP/s |
| bf16/int8 "tensor" peak | ~312 TFLOP/s (bf16), ~624 TOPS (int8) | ~8 TFLOP/s (AMX bf16), ~32 TOPS (VNNI int8) | (no tensor unit) |
| memory bandwidth | ~2 TB/s (HBM2e) | ~300–400 GB/s (8-ch DDR5) | ~50–100 GB/s (2-ch DDR5) |

1. **The hardware ceiling — unavoidable.** Even a *perfectly tuned* CPU kernel is
   ~5–40× below a GPU on bandwidth and ~10–100× on raw fp32. **This cannot be "fixed."**
2. **The emulation tax — removable.** On top of the ceiling, an emulator adds per-thread
   scalar dispatch, PTX interpretation, no SIMD-across-threads, and poor cache use — another
   ~10–1000×. **This is exactly what VGRE attacks.** Closing it makes VGRE run at the CPU's
   own roofline; then reduced precision + algorithmic levers make the *workload* fit that
   roofline.

**Strategy:** (A) remove the emulation tax → hit the CPU roofline; (B) use the CPU's best
units + reduced precision → raise the ML roofline; (C) cut the *work* algorithmically → a
smaller machine still finishes. The realistic target is **not** "as fast as an A100" — it is
"as fast as this CPU can possibly be for this workload, and small enough to fit."

### 7.2 A — Remove the emulation tax (hit the CPU roofline)

1. **SIMT → SIMD warp execution — the #1 lever (~8–16×).** A CUDA warp is 32 threads
   running the same instruction on different data — that is *literally* a SIMD vector. VGRE
   today mostly runs threads **scalar, one at a time**; executing a warp as **AVX-512
   (16 fp32 lanes) / AVX2 (8 lanes)** processes 8–16 threads per instruction. Needs
   warp-vectorized codegen: per-lane predicate masks for divergence, `vgather`/`vscatter`
   for non-contiguous access, and warp shuffle/vote as vector permutes/movemask. Designed in
   the Tier-0 note, **not built — the single highest-value perf item.**
2. **Native codegen for the whole hot path.** Interpretation is ~1000× off; the native
   x86-64 JIT + SSA backend already remove that for their subset (~18–118× over the compiled
   tier). Extend native coverage to 64-bit-heavy and the cooperative ops on **all** arches
   (native AArch64 warp/shared) so nothing hot falls to the evaluator.
3. **Cache-blocking + fusion (raise arithmetic intensity).** Most ML kernels are
   *memory-bound* on a CPU. Tile to L1/L2, **fuse** elementwise→GEMM→activation→norm so
   intermediates never reach DRAM, use non-temporal/streaming stores for write-only data,
   and software-prefetch. The kernel-fusion engine + flash-attention do the big ones; make
   fusion the default for elementwise chains.
4. **Thread-pool & NUMA efficiency.** Keep every core saturated with low dispatch overhead
   (persistent workers + CTA-range partitioning are done); add **first-touch NUMA placement
   + affinity** so bandwidth scales with sockets, and avoid oversubscription.

### 7.3 B — Raise the CPU's ML roofline (tensor units + reduced precision)

5. **CPU tensor units — Intel AMX / AVX-512-VNNI (~4–16× on matmul).** AMX does bf16/int8
   *tile* matmul at ~8× the AVX-512 fp32 rate; VNNI does int8 4-element dot-products at ~4×.
   Route GEMM/attention through **AMX (bf16)** and **VNNI (int8)** — the AMX path already
   exists in `vector_engine_amx`; make it the default bf16/int8 matmul and add an AMX
   micro-kernel to the in-tree GEMM. (ARM equivalents: SVE/SME, i8mm/bf16 dot.)
6. **Reduced precision everywhere — the CPU's biggest lever.** Lower precision cuts *both*
   compute and bandwidth, the two things the CPU is short on: int8 GEMV is ~4× fp32,
   **ternary/BitNet is multiplication-free** (add/sub only), MXFP4/int4 quarter the bytes.
   VGRE has all of these (T1/T5/T6) — the win is making them the **default serving path** and
   adding **AVX2 unpack kernels** (MXFP4/int4/ternary decode) so dequant isn't the bottleneck.
7. **KV-cache & activation quantization.** At long context the KV cache, not the weights,
   dominates bandwidth; int4 KV is 5× smaller (done for generation — wire into the paged
   serving path).

### 7.4 C — Cut the work (algorithmic — fit the model to the machine)

8. **Sparsity:** MoE activates only *k* experts (active compute ∝ routed tokens, not
   tokens×experts); 2:4 structured sparsity halves the matmul. *(done — push utilization.)*
9. **Linear-time sequence models:** Mamba/SSM removes O(T²) attention **and** the growing KV
   cache — a decisive CPU win at long context. *(done.)*
10. **Fewer forward passes:** speculative + multi-token decoding emit *k* tokens per forward
    (~4× decode throughput at identical output). *(done.)*
11. **Flash-attention:** O(T) memory instead of O(T²) — the difference between fitting long
    context in cache and thrashing DRAM. *(done.)*

### 7.5 What "closing the gap" means, and how we measure it

Target the **CPU's roofline, not the GPU's**: a tuned VGRE kernel should sit within ~1.5–2×
of hand-optimized OpenBLAS/oneDNN on the same CPU (the in-tree GEMM already does), and the
*whole model* should run at the CPU's peak for its precision. Against a GPU the residual gap
is then just the silicon ratio — which the algorithmic levers (quantization + sparsity + SSM
+ speculative decode) shrink by **reducing the work**, so a commodity CPU finishes a
GPU-sized job. That is the mission: make *the computer you already own* enough.

**Perf backlog (next release), each gated by a benchmark against the roofline:**
- [ ] **Warp-SIMD codegen** — execute 8/16 threads per AVX2/AVX-512 vector (the #1 lever).
- [ ] **AMX bf16 + VNNI int8 GEMM micro-kernels** as the default bf16/int8 matmul.
- [ ] **AVX2 unpack kernels** for MXFP4 / int4 / ternary so dequant isn't the bottleneck.
- [ ] **Fusion by default** for elementwise→GEMM→activation→norm chains (cut DRAM passes).
- [ ] **NUMA-aware allocation + affinity** so bandwidth scales across sockets.
- [ ] **Autotuned tile sizes** per CPU (cache-size-driven), like the GEMM autotuner.
- [ ] A **roofline / bandwidth benchmark** (`bench_*`) reporting achieved-vs-peak FLOP/s and
      GB/s per kernel, so every optimization is measured against the hardware ceiling.

### 7.6 Research basis (performance)
- **Roofline model**, Williams, Waterman & Patterson (CACM 2009) — the achieved-vs-peak
  framework that separates compute- from memory-bound and defines the target.
- **Anatomy of High-Performance Matrix Multiplication (BLIS)**, Goto & van de Geijn / Van
  Zee & van de Geijn — the register-blocked, cache-packed micro-kernel VGRE-GEMM follows.
- **Intel AMX / AVX-512-VNNI** ISA references — CPU tile/dot tensor units for bf16/int8.
- **BitNet b1.58 / ternary LLMs** (Ma et al., 2024) — multiplication-free inference, the
  strongest CPU lever (2.4–6.2× x86 speedup, up to ~82% less energy in the literature).
- **FlashAttention** (Dao et al., 2022) and **Mamba/S6** (Gu & Dao, 2023) — reduce the
  *work* (O(T) memory; no KV cache) rather than the per-op cost.
- **Speculative decoding** (Leviathan et al.; Chen et al., 2023) — fewer target forwards at
  identical output distribution.
