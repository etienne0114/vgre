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

## Types
| Type | Interpreter tier | Compiled tier |
|---|---|---|
| `int`, `unsigned`, `bool`, `char`, `short` (32-bit) | ✅ | ✅ |
| `float` | ✅ | ✅ |
| `double`, `long` (64-bit) | ⛔ rejected cleanly (use compiled tier / JIT) | ✅ full f64 / i64 |
| pointers (`T*`, `const T* __restrict__`) | ✅ | ✅ |

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
| `atomicAdd(&p[i], v)` (int + float) | ✅ |
| address-of `&p[i]` (as an atomicAdd target) | ✅ |

## Statements / control flow
| Feature | Status |
|---|---|
| variable declarations (with init) | ✅ |
| `if` / `else` (chained) | ✅ |
| `for`, `while` | ✅ |
| `return` | ✅ |
| blocks / scopes | ✅ (single flat scope) |

## Memory & builtins
| Feature | Status |
|---|---|
| global load/store | ✅ |
| `__shared__` arrays + `__syncthreads()` | ✅ (Tier-0 interpreter; the compiled tier defers barrier kernels to Tier-0) |
| `threadIdx/blockIdx/blockDim/gridDim.{x,y,z}` | ✅ (full 3D) |

## Intrinsics
`sqrtf`, `rsqrtf`, `fabsf`, `abs` (int/float), `expf`/`__expf`, `logf`/`__logf`,
`sinf`, `cosf`, `floorf`, `ceilf`, `fminf`, `fmaxf`, `min`, `max`, `fmaf`, `powf`.
On the interpreter tier the transcendentals use PTX approximate ops
(`sin.approx`, `ex2.approx`, …); the compiled tier uses libm.

## Not yet supported (returns an error, falls back to JIT when available)
- `struct` kernel parameters and user `__device__` functions
- local (non-`__shared__`) arrays
- templates, recursion
- texture/surface and warp-shuffle intrinsics
- `double`/`long` on the interpreter tier (compiled tier + JIT cover them)

Grow this set test-first: add a kernel test under `tests/compiler/`, implement it
in `src/compiler/frontend/{parser,codegen}.cpp` **and** the compiled tier
(`compiled_kernel.cpp`), and verify both tiers against a reference.
