# Supported CUDA-C subset (the from-scratch front-end)

**Updated:** 2026-09-08

VGRE's own CUDA-C front-end (`src/compiler/frontend/`: lexer → parser → PTX
codegen) compiles kernels with **no Clang/LLVM**. This is the compiler used when
`-DVGRE_ENABLE_JIT=OFF`, and whenever `VGRE_EXEC_BACKEND` selects a non-JIT
backend. It targets the practical subset of CUDA-C that real compute kernels use;
anything outside it returns a **located error** (never wrong code). Integer constant
subexpressions are **constant-folded** to a single immediate (`5*5*5*5` → `625`;
array indices and integer casts fold too). Every emitted kernel is run through a
**structural PTX verifier** (labels resolve, registers are declared and in range,
no malformed operands) as a self-check, so a codegen bug surfaces as an internal
error instead of bad PTX at runtime. The PTX header's
`.target` (SM arch), `.version` (PTX ISA) and `.address_size` are configurable via
`CodegenOptions` (defaults `sm_52` / `7.0` / `64`). And — in a JIT-enabled build —
falls back to the LLVM path.

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
| `int`, `unsigned`, `bool`, `char`, `short` | ✅ — **integer promotions**: `char`/`short`/`bool` promote to `int` in arithmetic (`(char)100 + (char)100 == 200`, no 8-bit wrap); **unsigned semantics** honored (ordered compares, `/ % >>`, `min`/`max`, widening use unsigned rules when an operand is unsigned; the rank rule makes `unsigned int + long` a *signed* long); `char`/`short`/`bool` load and store at their **real width** (1/2 bytes) with signed loads **sign-extended** and unsigned/bool zero-extended (`ld.s8`/`u8`/`s16`/`u16`, `st.u8`/`u16`) — so `char*`/`short*` buffers and narrow struct members are correct | ✅ |
| `float` | ✅ | ✅ |
| `double`, `long` (64-bit) | ✅ full f64 / i64 (real `.f64`/`.s64` PTX, `cvt`, 64-bit literals) | ✅ full f64 / i64 |
| pointers (`T*`, `const T* __restrict__`) | ✅ incl. pointer arithmetic `p ± i` (element-scaled, 64-bit); **64-bit comparisons** `== != < <= > >=` and `== nullptr` (full address width, `setp.*.u64`), pointer truthiness `if (p)`, and pointer-typed `cond ? p : q`; and cache-hinted loads/stores `__ldg`/`__ldca`/`__ldcs`/`__ldcg`/`__ldlu`/`__ldcv`/`__ldg_nc` and `__stwb`/`__stcg`/`__stcs`/`__stwt` (plain `ld.global`/`st.global` on the CPU — no cache hierarchy) | ✅ |
| `__half` (fp16, 16-bit storage) | ✅ `__half` variables + `__half*` load/store (`ld/st.b16`); `__float2half`/`__half2float` (`cvt.rn.f16.f32` / `cvt.f32.f16`, IEEE binary16 round-to-nearest-even); and half **arithmetic** `__hadd`/`__hsub`/`__hmul`/`__hdiv`/`__hfma`/`__hneg`/`__hmax`/`__hmin` + comparisons `__heq`/`__hne`/`__hlt`/`__hle`/`__hgt`/`__hge`, each computed in float and narrowed back to fp16 (result rounds to fp16, comparisons return `int`); and the half **unary math** family `hsqrt`/`hrsqrt`/`hrcp`/`__habs`/`hceil`/`hfloor`/`htrunc`/`hrint`/`hexp`/`hexp2`/`hexp10`/`hlog`/`hlog2`/`hlog10`/`hsin`/`hcos` (promote to float, apply the f32 op, narrow to fp16). **Implicit conversions** work too: `float f = h;` and `__half h = f;` convert through f32 (not a bit reinterpret), and a plain operator on `__half` (`a + b`) promotes to float, computes, and narrows back on assignment. | defers to interpreter |
| by-value `struct` params (scalar members, `.member` reads) | ✅ | defers to interpreter |
| **local** `struct` variables (`Vec p;`) | ✅ register-per-field: `p.x` read, `p.x = …`/`p.x += …` write, and whole-struct copy `Vec q = p;` (from a local or by-value-param struct). Scalar members. | defers to interpreter |
| `struct*` in memory — `p->field`, struct arrays (`data[i]`, `data + i`) | ✅ `p->x` loads / `p->y = v` stores at `ptr + member_offset`; struct-array indexing and `struct*` arithmetic stride by the struct size (from the layout table). | defers to interpreter |

## Expressions
| Feature | Status |
|---|---|
| int/float literals (hex, suffixes) | ✅ |
| character literals `'a'`, `'\n'`, `'\x41'`; `true` / `false` / `nullptr` | ✅ (char → its int value; `true`/`false` → 1/0; `nullptr` → null pointer constant) |
| arithmetic `+ - * / %` | ✅ |
| comparison `== != < <= > >=` | ✅ |
| logical `&& || !`, bitwise `& \| ^ ~ << >>` | ✅ — `&&`/`||` are true C logical ops (each side tested for truthiness `!= 0`, result 0/1) and **short-circuit** the RHS (`p && p->x` won't deref a null `p`; `a != 0 && 100/a > 3` won't divide by zero) |
| assignment `=` and compound `+= -= *= /= %= &= \|= ^= <<= >>=` | ✅ (compound op is honored for the LHS's type, incl. unsigned) |
| pre/post increment/decrement `++ --` | ✅ incl. **pointers** (`p++`/`p--` advance/retreat by one element) |
| pointer `-` pointer (ptrdiff) | ✅ element-count difference (byte difference ÷ element size) |
| ternary `cond ? a : b` (nested) | ✅ |
| C-style casts `(int)x`, `(float)y` | ✅ numeric conversions, `int`↔pointer, and pointer↔pointer. **Validated:** casts to/from a `struct`, or between a floating type and a pointer (which need a bit reinterpret like `__float_as_int`, not a value cast), are located errors. |
| `sizeof(type)` | ✅ folds to the type's byte size at parse time (`sizeof(float)`→4, `sizeof(double)`→8, any pointer→8); `sizeof expr` is a located error (no type inference) |
| declaration qualifiers `static` / `inline` / `volatile`; function qualifiers `__host__` / `__forceinline__` / `__noinline__` / `__inline_hint__` / `__launch_bounds__(…)` | ✅ accepted (no effect on lowering) — so `static __shared__`, `volatile` locals, and `__host__ __device__` helpers compile verbatim |
| indexing `p[i]`, member `threadIdx.x` | ✅ |
| unary dereference `*p` / `*(a+i)` and address-of `&a[i]` | ✅ — `*p` loads (and `*p = v` / `*p += v` store) through a pointer; `&a[i]` yields the element's 64-bit address (global arrays; `&scalar` and `&shared[i]` are located errors) |
| `atomicAdd`, `atomicSub`, `atomicMin`, `atomicMax`, `atomicExch`, `atomicAnd`, `atomicOr`, `atomicXor` (`&p[i], v`) and `atomicCAS(&p[i], cmp, v)` | ✅ real lock-free RMW on both tiers — correct under parallel CTAs; each returns the OLD value. add/sub take int **or** float; min/max are integer (signed/unsigned by the pointee); exch/and/or/xor/cas are 32/64-bit. On `__shared__`/local memory they lower to a sequential ld/compute/st (already atomic within a CTA). **Validated:** atomics require a 32- or 64-bit type, and `min`/`max`/`and`/`or`/`xor` require an integer type — invalid combinations (e.g. `atomicMin` on a `float*`, or any atomic on a `char*`) are located errors, not invalid PTX. |
| address-of `&p[i]` (as an atomic target) | ✅ |

## Statements / control flow
| Feature | Status |
|---|---|
| variable declarations (with init) | ✅ incl. multiple declarators (`int a, b = 1, c;`) |
| `if` / `else` (chained) | ✅ |
| `for`, `while`, `do`-`while` | ✅ |
| `switch` / `case` / `default` | ✅ integer selector; C fall-through (no implicit break); `default` optional; `break` exits the switch. **Validated:** `case` labels are constant-folded (constant expressions like `case 1+2:` work); duplicate case values, a non-constant label, or more than one `default` are located errors. |
| `break`, `continue` | ✅ `break` exits the nearest loop **or** switch; `continue` targets the nearest enclosing loop (`for`-increment / `while`-retest), skipping switches; misuse outside a loop/switch is a located error |
| `return` | ✅ |
| blocks / scopes | ✅ (single flat scope) |
| `__device__` / `__host__ __device__` helper functions | ✅ (inlined; nested calls OK; recursion rejected). Hygienic: each inline gets its own variable/array scope and unique PTX symbols; a non-void helper that falls through without a `return` yields a defined 0. A `__host__`-only function is **not** device-callable (calling it from a kernel is a located error); the launch entry must be a `__global__` kernel. |

**Token vocabulary:** the lexer recognizes the **complete** C++ reserved-keyword set (through C++26, incl. the alternative operator spellings `and`/`or`/…), every CUDA qualifier/annotation (`__host__`/`__device__`/`__global__`/`__shared__`/`__constant__`/`__managed__`/`__restrict__`/`__forceinline__`/`__launch_bounds__`/`__grid_constant__`/`__cluster_dims__`/…), all operators and punctuators (`-> :: ... .* ->* <=> <<< >>>` included), character literals and `true`/`false`/`nullptr`. Each token carries a source byte span; the keyword table, spellings, names and classifiers are centralized in `token.cpp`. Tokens outside the parsed subset lex cleanly and produce a **located** error if used — never wrong code.

## Memory & builtins
| Feature | Status |
|---|---|
| global load/store | ✅ |
| per-thread **local** arrays (`float tmp[4];`, register scratch) | ✅ **both tiers** — Tier-0: PTX `.local` per-thread arena; Tier-1 compiled: a private run of Cell slots (so no-barrier local-array kernels stay on the fast path). Constant + dynamic indexing. |
| **multi-dimensional** arrays (`float As[16][16]`, `As[ty][tx]`) | ✅ N-D declarations + N-D indexing flatten to row-major offsets, so textbook tiled kernels compile verbatim. Works for both `__shared__` and local arrays. |
| `__shared__` arrays + `__syncthreads()` | ✅ (Tier-0 interpreter, now parallel across CTAs; the compiled tier defers barrier/`__shared__` kernels to Tier-0) |
| scalar `__shared__` variables (`__shared__ int flag;`) | ✅ a single block-wide cell (`.shared` symbol, `ld.shared`/`st.shared` on bare read/write), so a write by one thread is seen by the whole CTA — the flag/leader-election pattern used by single-pass grid reductions. Distinct scalars occupy distinct cells; compound assignment (`+=`) and `++`/`--` (prefix/postfix, via a shared read-modify-write) are supported. |
| **dynamic** shared memory — `extern __shared__ T s[];` | ✅ byte size comes from the launch (`LaunchConfig.sharedBytes`), not the source → `.extern .shared`; the interpreter sizes the per-CTA arena as static shared **+** the launch's dynamic bytes (which alias the region after the static area). Textbook launch-sized reductions/scans compile verbatim. |
| `threadIdx/blockIdx/blockDim/gridDim.{x,y,z}` | ✅ (full 3D) |
| **warp shuffle** — `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync` | ✅ Tier-0 interpreter (32-lane warp rendezvous; optional power-of-two `width`). **32-bit and 64-bit** values: `double`/`long` shuffle as two `.b32` words (low/high) recombined — bit-exact (a `double` moves through an integer register, shuffling its raw IEEE-754 bits). Warp-cooperative, so the compiled tier defers these to Tier-0. |
| **warp vote** — `__ballot_sync`, `__any_sync`, `__all_sync` | ✅ Tier-0 interpreter. The predicate becomes a real `.pred` (`setp.ne`); a warp-wide rendezvous collects every active lane's bit into a ballot mask, then each lane reads its result (`ballot` = the masked lane bitmask; `any`/`all` = reductions over the membership mask). Warp-cooperative → Tier-0. |
| **warp sync** — `__syncwarp([mask])`, `__activemask()` | ✅ Tier-0 interpreter. `__syncwarp` → `bar.warp.sync`, a 32-lane-scoped barrier (all active lanes rendezvous before continuing); `__activemask` → `activemask.b32`, the bitmask of the warp's non-exited lanes. Warp-cooperative → Tier-0. |
| **block barrier reductions** — `__syncthreads_count(pred)`, `__syncthreads_and(pred)`, `__syncthreads_or(pred)` | ✅ Tier-0 interpreter. A CTA-wide barrier fused with a reduction over the predicate: `__syncthreads_count` → `bar.red.popc.u32` (number of threads with a nonzero predicate); `__syncthreads_and`/`_or` → `bar.red.and`/`or.pred` (nonzero iff the predicate holds for **all** / **any** thread). The whole block rendezvouses, the parked threads' predicates are reduced, and the result is broadcast to every thread. Block-cooperative → Tier-0. |
| **memory fences** — `__threadfence_block()`, `__threadfence()`, `__threadfence_system()` | ✅ Lowered to PTX `membar.cta`/`membar.gl`/`membar.sys`. The interpreter issues a **real** CPU barrier (`std::atomic_thread_fence`): within a CTA the cooperative scheduler already orders a thread's memory ops, but the backend runs different CTAs on parallel host threads, so a device/system-scope fence does real work — it orders global writes a concurrently-scheduled block observes. This makes the canonical single-pass grid reduction (last block, chosen by an atomic counter, sums every block's fenced partial) correct, not merely accepted. |

## Intrinsics
`sqrt`, `rsqrt`, `fabs`, `abs` (int/float, width-preserving), `exp`/`__expf`,
`log`/`__logf`, `exp2`, `log2`, `tanh`, `sin`, `cos`, `floor`, `ceil`, `trunc`,
`rint`, `nearbyint`, `hypot`, `fmod`, `fdividef`, `copysign`, `__saturatef`,
`fmin`,
`fmax`, `min`, `max`, `fma`, `pow` — each in both the **f32** `…f`-suffixed
spelling and the **f64** (double) bare-C spelling (`sqrtf`↔`sqrt`, `fmaf`↔`fma`,
`tanhf`↔`tanh`, …), picked by the name. (`exp2`/`log2` map directly to PTX
`ex2.approx`/`lg2.approx`; `tanh` is a dedicated interpreter op — together they
cover activation kernels like tanh-GELU.)
On the interpreter tier the transcendentals use PTX approximate ops
(`sin.approx`, `ex2.approx`, …) evaluated at the operand's width; the compiled
tier uses libm.

**Extended math (built by composition, so both tiers support them):** `exp10`,
`log10`, `sinh`, `cosh`, `expm1`, `log1p` — each in the f32 (`…f`) and f64 spelling
— lowered via `ex2`/`lg2` (e.g. `sinh(x)=(e^x−e^−x)/2`, `log1p(x)=log2(1+x)·ln2`).

**Inverse trig / special (interpreter-tier — the interpreter computes these via
libm; no PTX approx op exists):** `atan`, `asin`, `acos`, `atan2`, `cbrt`, `erf`
(f32 and f64). Emitted as `<op>.approx.<ty>`; usable on the always-available Tier-0
interpreter (the compiled tier does not provide these).

**Explicit-rounding arithmetic:** `__fadd`/`__fsub`/`__fmul`/`__fdiv`,
`__fmaf`, `__frcp`, `__fsqrt`, `__frsqrt` (f32) and `__dadd`/`__dsub`/`__dmul`/
`__ddiv`, `__fma`, `__drcp`, `__dsqrt` (f64), each in the `_rn`/`_rz`/`_ru`/`_rd`
rounding-mode spellings. These name the rounding a plain operator would leave to
the compiler — e.g. `__fmul_rn(a,b)` then `__fadd_rn(_,c)` stays un-contracted,
while `__fmaf_rn(a,b,c)` forces a single-rounding fused multiply-add. Lowered to
the rounding-tagged PTX op (`add.rn.f32`, `fma.rn.f32`, `div.rn.f32`, …).
**Fidelity note:** the Tier-0 interpreter evaluates round-to-nearest-even
exactly (it computes in double, then rounds to the result type), so the `_rn`
forms are bit-exact; the directed `_rz`/`_ru`/`_rd` forms are accepted for source
compatibility but currently also round to nearest.

**Bit-reinterpret (type-punning):** `__float_as_int`/`__float_as_uint`,
`__int_as_float`/`__uint_as_float`, `__double_as_longlong`, `__longlong_as_double`
— copy the raw bits between a float and a same-width integer register (`mov.b32`/
`mov.b64`). Used for fast-math bit tricks and float atomics via `atomicCAS`.

**Rounding-mode conversions (value casts):** `__float2int_{rn,rz,ru,rd}` and
`__float2uint_{rn,rz,ru,rd}` → PTX `cvt.{rni,rzi,rpi,rmi}.{s32,u32}.f32`
(nearest-even / toward-zero / +∞ / −∞); `__int2float_rn` / `__uint2float_rn` →
`cvt.rn.f32.{s32,u32}`. Common in quantization / mixed-precision code.

**Bit intrinsics:** `__popc`/`__popcll` (population count), `__clz`/`__clzll`
(count leading zeros; `clz(0)` = bit width), `__brev`/`__brevll` (bit reversal,
width-preserving), and `__ffs`/`__ffsll` (1-indexed lowest set bit, `0` for a zero
operand). `popc`/`clz`/`brev` map to PTX `*.b{32,64}`; `ffs` is composed as
`clz(brev(x)) + 1` guarded for `x == 0`. All return a 32-bit `int` except `__brev`
(width-preserving). `__popc(__ballot_sync(mask, pred))` — the canonical count of
lanes whose predicate is set — works end to end.

## Worked example: FlashAttention

These pieces compose. `tests/compiler/test_cuda_flash_attention.cpp` compiles a
real **flash-attention** kernel — `__shared__` K/V tiles + `__syncthreads()`
cooperative loading, a per-thread local `acc[D]` accumulator, the online-softmax
running-max/running-sum rescale (no N² score matrix), and `expf`/`fmaxf` — with
**no LLVM**, runs it on the Tier-0 interpreter, and matches a CPU
softmax-attention reference to ~5e-8 (float rounding). The canonical transformer
workload runs on the from-scratch front-end.

## Not yet supported (returns an error, falls back to JIT when available)
- templates, recursion
- texture/surface intrinsics
- `__shfl` predicate/return variants beyond the four `__shfl_*_sync` forms, and
  `__ballot`-style vote **without** an explicit membership mask (the `_sync`
  forms — `__ballot_sync`/`__any_sync`/`__all_sync` — and 64-bit `__shfl_*_sync`
  of `double`/`long` **are** supported)

Grow this set test-first: add a kernel test under `tests/compiler/`, implement it
in `src/compiler/frontend/{parser,codegen}.cpp` **and** the compiled tier
(`compiled_kernel.cpp`), and verify both tiers against a reference.

## Test status of the two builds (2026-09-08)

| Build | Result |
|---|---|
| `-DVGRE_ENABLE_JIT=ON` (default) | **322 / 322 pass** — full LLVM JIT + from-scratch backends (`XlaBlasGemm` is a heavy `-j`-load timing flake: passes in isolation) |
| **bare: `-DVGRE_ENABLE_JIT=OFF -DVGRE_ENABLE_OPENMP=OFF`** | **302 / 302 pass, 0 crashes** — VGRE built with **nothing but a C++17 compiler** (no LLVM, no OpenMP; the compiled-kernel tier still parallelises CTAs via the in-tree thread pool). |
| `-DVGRE_ENABLE_JIT=OFF` (no LLVM, OpenMP on) | **302 / 302 pass, 0 crashes/aborts** (`Phase3ExtAPI` is an occasional `-j`-load flake — passes in isolation) |

The whole engine kernel path is routed through the from-scratch backends when
LLVM is absent (`RuntimeEngine::registerKernel`/`launchKernel` +
`runtime_engine_backend.cpp`), through a **single unified kernel registry** — the
no-LLVM C-ABI (`vgre_register_kernel`/`vgre_launch_kernel`) registers with the
engine rather than a side store — so **`cudaLaunchKernel`, the C++ engine API,
UVM, stream concurrency, capture-based CUDA graphs, AND C-API graphs
(`GraphCAPIIntegration`) all execute correctly with no LLVM**.

The tests **excluded** from the no-LLVM build (guarded on `VGRE_ENABLE_JIT` in
`tests/CMakeLists.txt`) genuinely require the JIT and fall into two groups:

1. **LLVM-only machinery** — bitcode modules (`LLVMBitcodeModuleDirect`), Clang
   AST analysis (`ClangEnhanced`, `KernelParserEnhanced`, `VectorizationHints`,
   `FLOPCounting`), IR-level kernel fusion (`KernelFusionIntegration`), FLOP/
   profiling export (`NsightExport`), and the JIT warp/device intrinsics
   (`WarpShuffleJIT`, `DeviceIntrinsics`, `DeviceCurand`, `wgmma`, `tensorcore`,
   `ptx_translate`, `warp_shuffle`, `cluster_exec`, `block_threads_toggle`).
2. **Interpreter-incompatible** — cooperative groups with grid-wide sync
   (`CooperativeGroupsPartition`, `MultiDeviceCooperativeComprehensive`): the
   sequential interpreter cannot provide a resident-grid barrier.

(`struct` kernel params, `__device__` helpers, per-thread local arrays, and a
full **flash-attention** kernel are now supported on the from-scratch front-end;
the dual-registry split is fixed — both graph paths work with no LLVM.)
