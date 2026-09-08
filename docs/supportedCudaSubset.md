# Supported CUDA-C subset (the from-scratch front-end)

**Updated:** 2026-09-08

VGRE's own CUDA-C front-end (`src/compiler/frontend/`: lexer → parser → PTX
codegen) compiles kernels with **no Clang/LLVM**. This is the compiler used when
`-DVGRE_ENABLE_JIT=OFF`, and whenever `VGRE_EXEC_BACKEND` selects a non-JIT
backend. It targets the practical subset of CUDA-C that real compute kernels use;
anything outside it returns a **located error** (never wrong code), and — in a
JIT-enabled build — falls back to the LLVM path.

Every feature below is verified end-to-end on **both** execution tiers
(Tier-0 PTX interpreter and Tier-1 compiled) unless noted.

**Both** execution tiers run a grid's thread-blocks (CTAs) **in parallel** across
the in-tree work-stealing thread pool (`include/vgre/xla/thread_pool.h`) — no
OpenMP, so the multi-core path is live in *every* build, including the bare
no-LLVM/no-OpenMP one:

* **Tier-1 compiled** parallelises over CTAs directly in `CompiledKernel::launch`.
* **Tier-0 interpreter** partitions the grid into contiguous CTA ranges, each run
  by its own interpreter instance (`PtxInterpreter::runCtaRange`) with private
  shared memory + thread state, so `__shared__`/`__syncthreads` kernels
  (tiled GEMM, reductions) scale too. The single-instance debugger surface
  (`resume`/`stepThread`) is untouched.

Barrier-free CTAs are independent, so this is safe; the only cross-CTA sharing,
`atomicAdd`, is a real hardware atomic RMW on both tiers — the compiled tier via
`__atomic_fetch_add` / a bit-CAS loop for floats, the interpreter via a real
`atom.global.add` (`__atomic` builtins, locked fallback for other compilers) —
so results are correct regardless of CTA scheduling. Measured **~5.4×**
(compiled, compute-bound elementwise) and **~4.9×** (interpreter, tiled GEMM) on
an 8-worker pool, both bit-identical to the serial result.

## Types
| Type | Interpreter tier | Compiled tier |
|---|---|---|
| `int`, `unsigned`, `bool`, `char`, `short` (32-bit) | ✅ | ✅ |
| `float` | ✅ | ✅ |
| `double`, `long` (64-bit) | ⛔ rejected cleanly (use compiled tier / JIT) | ✅ full f64 / i64 |
| pointers (`T*`, `const T* __restrict__`) | ✅ | ✅ |
| by-value `struct` params (scalar members, `.member` reads) | ✅ | defers to interpreter |

## Expressions
| Feature | Status |
|---|---|
| int/float literals (hex, suffixes) | ✅ |
| arithmetic `+ - * / %` | ✅ |
| comparison `== != < <= > >=` | ✅ |
| logical `&& || !`, bitwise `& \| ^ ~ << >>` | ✅ |
| assignment `=` and compound `+= -= *= /= %=` | ✅ |
| pre/post increment/decrement `++ --` | ✅ |
| ternary `cond ? a : b` (nested) | ✅ |
| C-style casts `(int)x`, `(float)y` | ✅ |
| indexing `p[i]`, member `threadIdx.x` | ✅ |
| `atomicAdd(&p[i], v)` (int + float) | ✅ (real atomic RMW on both tiers — correct under parallel CTAs) |
| address-of `&p[i]` (as an atomicAdd target) | ✅ |

## Statements / control flow
| Feature | Status |
|---|---|
| variable declarations (with init) | ✅ |
| `if` / `else` (chained) | ✅ |
| `for`, `while` | ✅ |
| `return` | ✅ |
| blocks / scopes | ✅ (single flat scope) |
| `__device__` helper functions | ✅ (inlined; nested calls OK; recursion rejected) |

## Memory & builtins
| Feature | Status |
|---|---|
| global load/store | ✅ |
| `__shared__` arrays + `__syncthreads()` | ✅ (Tier-0 interpreter, now parallel across CTAs; the compiled tier defers barrier kernels to Tier-0) |
| `threadIdx/blockIdx/blockDim/gridDim.{x,y,z}` | ✅ (full 3D) |

## Intrinsics
`sqrtf`, `rsqrtf`, `fabsf`, `abs` (int/float), `expf`/`__expf`, `logf`/`__logf`,
`sinf`, `cosf`, `floorf`, `ceilf`, `fminf`, `fmaxf`, `min`, `max`, `fmaf`, `powf`.
On the interpreter tier the transcendentals use PTX approximate ops
(`sin.approx`, `ex2.approx`, …); the compiled tier uses libm.

## Not yet supported (returns an error, falls back to JIT when available)
- local (non-`__shared__`) arrays
- templates, recursion
- texture/surface and warp-shuffle intrinsics
- `double`/`long` on the interpreter tier (compiled tier + JIT cover them)

Grow this set test-first: add a kernel test under `tests/compiler/`, implement it
in `src/compiler/frontend/{parser,codegen}.cpp` **and** the compiled tier
(`compiled_kernel.cpp`), and verify both tiers against a reference.

## Test status of the two builds (2026-09-08)

| Build | Result |
|---|---|
| `-DVGRE_ENABLE_JIT=ON` (default) | **318 / 318 pass** — full LLVM JIT + from-scratch backends |
| **bare: `-DVGRE_ENABLE_JIT=OFF -DVGRE_ENABLE_OPENMP=OFF`** | **298 / 298 pass, 0 crashes** — VGRE built with **nothing but a C++17 compiler** (no LLVM, no OpenMP; the compiled-kernel tier still parallelises CTAs via the in-tree thread pool). The lone `-j`-load flake, `Phase3ExtAPI`, passes in isolation. |
| `-DVGRE_ENABLE_JIT=OFF` (no LLVM, OpenMP on) | **297 / 297 pass, 0 crashes/aborts** — every test that runs, passes (`PythonNn` is a documented `-j`-load flake: passes 3/3 in isolation and with CI's `--repeat until-pass`) |

The whole engine kernel path is routed through the from-scratch backends when
LLVM is absent (`RuntimeEngine::registerKernel`/`launchKernel` +
`runtime_engine_backend.cpp`), through a **single unified kernel registry** — the
no-LLVM C-ABI (`vgre_register_kernel`/`vgre_launch_kernel`) registers with the
engine rather than a side store — so **`cudaLaunchKernel`, the C++ engine API,
UVM, stream concurrency, capture-based CUDA graphs, AND C-API graphs
(`GraphCAPIIntegration`) all execute correctly with no LLVM**.

The tests **excluded** from the no-LLVM build (guarded on `VGRE_ENABLE_JIT` in
`tests/CMakeLists.txt`) genuinely require the JIT and fall into three groups:

1. **LLVM-only machinery** — bitcode modules (`LLVMBitcodeModuleDirect`), Clang
   AST analysis (`ClangEnhanced`, `KernelParserEnhanced`, `VectorizationHints`,
   `FLOPCounting`), IR-level kernel fusion (`KernelFusionIntegration`), FLOP/
   profiling export (`NsightExport`), and the JIT warp/device intrinsics
   (`WarpShuffleJIT`, `DeviceIntrinsics`, `DeviceCurand`, `wgmma`, `tensorcore`,
   `ptx_translate`, `warp_shuffle`, `cluster_exec`, `block_threads_toggle`).
2. **Interpreter-incompatible** — cooperative groups with grid-wide sync
   (`CooperativeGroupsPartition`, `MultiDeviceCooperativeComprehensive`): the
   sequential interpreter cannot provide a resident-grid barrier.
3. **Broader-CUDA-C-subset follow-ons** — multi-feature kernels like
   `FlashAttention` that the from-scratch front-end does not yet compile.
   (`struct` kernel params and `__device__` helpers are now supported; the
   dual-registry split is fixed — both graph paths work with no LLVM.)
