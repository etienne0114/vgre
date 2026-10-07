# VGRE Project Status & Gap Analysis

**Last Updated**: 2026-10-07

**Latest hosted verification**: ✅ [GitHub Actions run 37042616142](https://github.com/etienne0114/vgre/actions/runs/37042616142) passed Linux x86-64, Linux LLVM-free, macOS ARM64, and Windows x86-64. Each OS job completed successfully.

- **Linux**: ✅ Full `ctest` suite passed in the latest hosted Linux x86-64 job.
- **Linux, LLVM-free**: ✅ The dedicated `linux-x86_64-llvm-free` job passed with `VGRE_ENABLE_JIT=OFF` and `VGRE_ENABLE_OPENMP=OFF`.
- **macOS**: ✅ Apple Silicon build, CTest, Python integration, and SSA native-execution checks passed in the latest hosted job.
- **Windows**: ✅ Windows x86-64 build, CTest, pretrained GGUF CLI smoke, and Python integration passed in the latest hosted job.
- **PyPI**: Latest published package is [0.1.4](https://pypi.org/project/vgre/0.1.4/), with Linux, macOS universal2, and Windows wheels.

**2026-10-07 math verification:** Commit `2857b114` contains the expanded full-fragment
FP16/BF16/TF32 `mma.sync` references, TF32 conversion tests, FP16/TF32 WGMMA references,
all mapped tcgen05 BF16/FP16/TF32/FP8 shape/type dispatch checks, and the FP16 WGMMA
`m64n128k16` output-bound guard. The latest reported Linux, Linux LLVM-free, macOS, and
Windows CI jobs are green.

**Local FP8 follow-on (2026-10-07):** E4M3/E5M2 conversion now implements round-to-nearest-even,
round-toward-zero, ReLU, separate `.satfinite` behavior, and PTX packed-source ordering.
Exhaustive byte and midpoint checks pass, along with `PTXTranslate` (31/31),
`PTXFp8Translation` (14/14), `TensorMemoryTcgen05` (23/23), `TensorCoreMMA` (14/14),
and `WgmmaWarpGroup` (7/7). The follow-on subsequently passed hosted CI across the
Linux x86-64, Linux LLVM-free, macOS ARM64, and Windows x86-64 lanes.
Linux `MatMulAMX` and the portable scalar `MatMulAMX` checks passed in the prior local
verification. The full local 410-test CTest run was
stopped at test 78 after `IdentityMetricsJwt` could not start its loopback HTTP server;
tests after that point were not run. `VGRE_CACHE_DIR` now overrides
the kernel AST cache directory on all platforms, and stale-cache eviction handles
read-only directories without throwing; the CUDA Graph integration test passes against
the default read-only cache path.

**2026-10-07 PTX integer-conversion follow-on (CI-green):** The inline PTX
translator now emits explicit nearest-even, toward-zero, downward, and upward
float-to-integer conversions for signed and unsigned 32-bit and 64-bit destinations
from both f32 and f64 sources. Conversion clamps to the destination range and follows
PTX NaN results without relying on undefined out-of-range C++ casts. The user reports
this follow-on passed CI across the platform matrix.

**2026-10-07 floating narrowing follow-on (current workspace):** The shared `__half`
codec now handles ties-to-even and subnormal rounding correctly; PTX f32-to-f16
`.rn` and `.rz` use explicit modes. f64-to-f32 conversions now round with explicit
IEEE modes, and widening f32-to-f64 no longer rounds the input to an integer first.
Floating `.sat` conversions map NaN to positive zero. Clang C++17 syntax checks pass;
runtime tests and hosted CI for this follow-on remain pending.

**Public demo**: 🌐 free CPU demo live at **https://vgrengine.streamlit.app** (Streamlit Community Cloud — HF now requires PRO for server-side Spaces)  
**Production Readiness**: core emulation is stable and verified on Linux; macOS ARM64 and Windows x86-64 are both CI-green (full `ctest` suite) — all three platforms now pass in CI.

> **Historical CI status (2026-09):** CI was restored on the public repository.
> At that point some OS lanes were informational; since then the workflow was
> updated so Linux, macOS, and Windows are required, and all four latest matrix
> jobs (including Linux LLVM-free) passed run 37042616142.

> **LLVM is now optional (delivered):** kernel execution no longer requires LLVM —
> a from-scratch CUDA-C front-end feeds a four-tier CPU backend (PTX interpreter,
> compiled-fiber tier, native x86-64 JIT, and an optional SSA optimizing backend),
> and the full suite passes LLVM-free (`VGRE_ENABLE_JIT=OFF`). See
> [`zeroBurdenRoadmap.md`](zeroBurdenRoadmap.md) for the plan and what remains.

> **2026-07-03 macOS bring-up** (in-tree, not yet full CI-green):
> - `cmake/VGREPlatform.cmake` auto-detects Homebrew `llvm@18`, `libomp`, SDK,
>   and libc++ rpath — no hardcoded install prefixes.
> - OpenMP, CTest `DYLD_LIBRARY_PATH`, JIT clang discovery, Keychain `SecItem*`,
>   `dispatch_semaphore` external semaphores, and IOKit SMC temperature are real
>   implementations (not stubs).
> - Verified locally: full `cmake --build`, `examples/vector_addition` (1M elems),
>   integration/JIT ctest subset green. Run full suite serially:
>   `VGRE_LOG_LEVEL=ERROR ctest -j1 --timeout 300`.

> **2026-07-02 audit deltas** (see git log for the commits):
> - HIP/ROCm layer expanded from a 9-function memcpy veneer to the real core
>   runtime surface (streams, events, async memory, device props/attrs, module
>   load + JIT kernel launch), with an end-to-end numerically-verified test.
> - `vgre_set_block_threads` was a logged no-op returning success; it is now a
>   real runtime flag consulted by every JIT-generated launcher (tested).
> - `cuModuleGetFunction` on a module whose image had no extractable source
>   crashed the process inside ORC (uncaught `std::length_error`); foreign
>   handles are now validated and return `CUDA_ERROR_INVALID_VALUE`.
> - `getActiveKernelCount()` always returned 0 (counter never incremented);
>   live and cumulative kernel-launch counters are now maintained and the gRPC
>   `kernels_launched` metric reports the cumulative total.
> - Distributed-training verification deepened: multi-step (8-iteration)
>   2-process data-parallel AdamW training (zero cross-step drift; matches the
>   single-process full-batch trajectory) and cross-process tensor parallelism
>   (sharded forward/backward + sharded SGD reassembling the full-model run)
>   now run over the real TCP collective in CI-shaped tests
>   (`PythonNnDistributedMultistep`, `PythonNnTensorParallel`).

> **Historical correction (2026-06-10):** the CI/platform half below is out of date.
> The latest CI status is in the header: Linux, macOS, and Windows all pass the
> full `ctest` suite, and each is a required lane. The simplified-compute-path caveats
> it lists still stand where `missingFeatures.md` §1 marks them. Original text:
> earlier revisions of this file claimed
> "CI/CD-Ready", "validated across Linux, Windows, macOS", and "zero stubs".
> That was inaccurate at the time. The then-truth: the build and tests ran on **Linux only**;
> Windows/macOS code was compile-guarded but **unverified** (there was no CI). A
> handful of compute paths are deliberately **simplified** (Flash Attention
> recomputes K/V and NCCL ring uses a barrier-per-round model; the earlier WMMA flat
> dot-product caveat is detailed below). This file now tracks the real
> path to production rather than asserting it.

> **2026-10-07 math correction:** The register-based `mma.sync` helpers reconstruct
> complete per-warp tiles from the PTX fragments. The current local regression set
> also checks FP16/BF16/TF32 `mma.sync`, mapped WGMMA formats, and all mapped tcgen05
> dispatch shapes against matrix references. This does not claim support for PTX
> shapes or modifiers that are absent from the translator maps.

VGRE (Virtual GPU Runtime Engine) is a high-fidelity CUDA emulation runtime designed to execute unmodified CUDA, cuBLAS, cuDNN, cuSPARSE, cuSolver, cuRAND, and NCCL workloads on standard x86-64 and ARM64 CPU architectures. It intercepts GPU API calls at load time and runs them on host hardware using an LLVM-18 JIT compilation pipeline and a thread-safe parallel execution model.

### 2026-06-07 Session Fixes
- **Zero compiler warnings** — strict-aliasing type-punning replaced with `std::memcpy`, all `warn_unused_result` captures added, `static thread_local` cross-TU semantic bug resolved via `extern thread_local`. Warnings-as-errors (`-Werror`) enabled globally.
- **`g_current_ctx` cross-TU bug** — `static thread_local` in a shared header gave each TU its own independent per-thread CUDA context pointer. `cuCtxSetCurrent` in one TU was invisible to `cuCtxGetCurrent` in another. Fixed: `extern thread_local` + single definition in `cuda_driver_device_context.cpp`.
- **K8s deployment** — `--is-master` flag added to `vgre-worker` for headless container master mode. K8s device plugin rewired: `context.Context` in all gRPC handlers, `grpc.NewClient` + `insecure.NewCredentials`, `VGRE_DEVICE_PLUGIN_PATH` env var, proper `<-stream.Context().Done()`. Distroless base image; C++ runtime libraries added to Dockerfiles.
- **Auto-detected build features** — CMake probes libibverbs, libtss2-esys, libsecret-1, and CPU SIMD at configure time; all four default ON when installed; build never aborts on missing optional deps.
- **Cross-platform FFI errors** — `vgre_ffi.dart` emits OS-specific hints on library load failure (Linux/macOS/Windows).

---

## 1. Core Platform Verification

### 1.1 Linux (x86-64) — canonical

**Verified on Linux (x86-64).** The full `ctest` suite passes on Linux, exercised with property-based
exploration, differential fuzzing of the CUDA-C front-end against both execution
tiers, ThreadSanitizer race analysis, and static-destruction verification.

### 1.2 macOS (Apple Silicon) — CI-green

**The full `ctest` suite runs green in CI on `macos-14` (Apple Silicon).** The native
engine (`libvgre`, `libvgre_cudart`, examples, integration tests) compiles with
Homebrew `llvm@18` auto-selected by CMake; JIT kernel compilation discovers Clang at
runtime (`VGRE_CLANG_PATH` → `llvm-config-18` → `brew --prefix llvm@18`). A prebuilt
macOS-arm64 wheel ships with each release. The latest macOS CI job is required and
passed the full suite plus Python integration and native SSA checks.

**macOS-specific real implementations** (not stubs): UVM via `SIGSEGV`/`SIGBUS`,
Keychain via `SecItemAdd`/`SecItemCopyMatching`, thermal via IOKit SMC,
external semaphores via `dispatch_semaphore`, cluster TCP via BSD sockets.

**Documented macOS approximations** (see `missingFeatures.md`): NUMA pinning
(Mach hints vs Linux affinity), no Metal Performance Shaders backend, no NVIDIA
PMU/CUPTI counters without physical GPU hardware.

### 1.3 Windows — CI-green

**Windows builds and runs the full `ctest` suite in CI** (`windows-2022`,
clang-cl, LLVM-18 tarball cached). The latest required Windows job also passed
the pretrained GGUF CLI smoke and Python integration checks. Earlier bring-up
fixes included AVX2 `rsqrt`/GEMM numerical accuracy, `vgre.dll` dependency
loading under Python 3.8+, and cp1252 console encoding of non-ASCII test output.

- **Linux core passes**: full regression + integration + platform suite green on x86-64.
- **Mostly real compute, with documented exceptions**: nearly every path runs real
  CPU math (AVX/ARM64 SIMD + OpenBLAS/LAPACK). Known software approximations include
  Flash Attention K/V recomputation and the NCCL barrier-per-round ring; other
  tensor-core shapes still need ISA-by-ISA audit. Hardware-boundary proxies include
  CUPTI PMU counters.
- **Teardown stability**: thread lifecycles, socket listeners, and memory heaps are
  managed deterministically; the JIT-worker static-destruction crash is fixed.
- **Known test caveat**: under `ctest -j`, heavy clang/JIT tests can intermittently
  time out from CPU oversubscription (all pass standalone). Phase 2, Track 6
  serializes them so CI is deterministic.

---

## 2. Genuinely Missing or Partially Implemented Features

The CUDA-emulation feature surface is complete; remaining gaps are hardware-level
constraints where CPU emulation falls back to a high-fidelity proxy or reports
platform limits. **Superseded (2026-09):** the earlier "all software-emulatable
features are done" framing predates the **zero-burden program** — making LLVM
optional (done) and building the CUDA-C→PTX pipeline from scratch. That pipeline's
own speed step is now done too: a from-scratch **native x86-64 JIT**
(`native_kernel_x64.cpp`, Tier 1b) is the default execution tier, emitting real
machine code for the scalar subset (all int/float ops, casts, comparisons, ternary,
locals, gather/scatter, bounded-loop reductions, flattened + true-2-D GEMM, and the
full math surface incl. transcendentals) at ~18–118× the compiled tier, held
bit-exact by the `CudaNative` fuzzers. The *optional* Tier-2 SSA backend
(`VGRE_EXEC_BACKEND=ssa`, tier 3) is now feature-complete for the scalar +
shared-memory + warp subset (native x86-64, AArch64 (native by default), portable evaluator
elsewhere; held bit-exact by `test_ssa_ir.cpp`/`test_ssa_backend.cpp`); its remaining
items are perf/hardware-gated (native codegen for the cooperative warp ops; ARM native
default flip). Remaining zero-burden work is tracked in
[`zeroBurdenRoadmap.md`](zeroBurdenRoadmap.md) and [`missingFeatures.md`](missingFeatures.md).

For the comprehensive, definitive list of boundary conditions (such as physical PMU counters, SASS binary execution, and GPUDirect RDMA), please see [missingFeatures.md](missingFeatures.md).

---

## 3. Verified Capabilities & Library Coverage

The following components are fully implemented, verified via regression tests, and stable for production deployment:

### 3.1 Kernel Compilation & Execution
- **From-scratch CUDA-C front-end + four-tier CPU backend (no LLVM required)**: an in-tree lexer/parser lowers kernels to an AST, then to the fastest of four bit-exact tiers — a PTX interpreter, a compiled-closure tier with a cooperative fiber executor, a hand-emitted native x86-64 JIT, and an optional own SSA optimizing backend (`VGRE_EXEC_BACKEND=interp|cp|ssa`). This is the default in the LLVM-free build, which passes its dedicated hosted CI lane.
- **LLVM JIT Compiler (optional, high-performance path)**: when LLVM dev libs are present, dynamically JITs PTX to native assembly via Clang and LLVM ORC JIT, optimized with `-O3 -march=native`.
- **Persistent Disk Caching**: Stores JIT compilations in `~/.vgre/cache/` using an LRU cache with AST collision eviction and integrity check.
- **Block Worker Pool**: Emulates GPU grid execution using a pre-warmed thread pool (1024-2048 threads) and sense-reversing barrier objects for `__syncthreads()`.
- **CUDA Dynamic Parallelism**: Fully supports recursive child kernel launches from JIT kernels.
- **CUDA Graphs**: Real DAG with topological sorting, kernel fusion, conditional SWITCH/IF nodes, and external semaphore synchronization.

### 3.2 Memory Management
- **Unified Virtual Memory (UVM)**: Fully emulated UVM via standard virtual memory tools (`mmap(PROT_NONE)` + `mprotect` + `madvise()` on Linux/macOS, Vectored Exception Handlers on Windows). Runs a background page-migration manager.
- **Async Allocations**: Stream-ordered memory pools (`cudaMallocAsync`) to eliminate OS allocation bottlenecks.
- **Multi-Process Shared Memory (IPC)**: Supports sharing buffers between local processes (`cudaIpcGetMemHandle`/`cudaIpcOpenMemHandle`) via POSIX shared memory segments.

### 3.3 Compute Libraries
- **cuBLAS & cuBLASLt**: Supports L1/L2/L3 operations, cache-blocked GEMM, CBLAS delegation, `cublasGemmEx` widening fallbacks, and custom algo heuristics (`cublasLtMatmulAlgoGetHeuristic`).
- **cuDNN**: Conv, Max/Avg Pooling, Activations (ReLU, Sigmoid, Tanh, ELU, Swish), Softmax, Dropout, Attention, LRN, CTC Loss, and RNN (LSTM/GRU forward + BPTT backward and weight gradients).
- **cuSPARSE**: Supports CSR, BSR, and COO formats, batched SpMM, sparse triangular solve with matrix RHS (`cusparseSpSM` for both real and complex types), Sampled Dense-Dense Matrix Multiplication (`cusparseSDDMM`), and format descriptors. Zero-copy sparse view system (CSR↔CSC, CSR→BSR) via `sparse_view.{h,cpp}`.
- **cuSolver**: QR, LU, SVD, Eigen, and batched solvers backed by LAPACK/OpenBLAS. Includes 64-bit type-erased APIs (`cusolverDnX*`). Full mathematically rigorous generalized eigenvalue (`cusolverDnXsygvd`) with L⁻¹AL⁻ᵀ congruence reduction and back-projection.
- **cuRAND**: Thread-safe host-side generators and device-side JIT kernels supporting XORWOW, Philox, MRG32K3A, Sobol, and MTGP32.
- **NCCL**: Ring and barrier-based collective operations (AllReduce, Broadcast, AllGather, ReduceScatter) with upcast/downcast support for half-precision formats.

### 3.4 Advanced Mathematical Hardening
- **Mixed Precision**: Full FP16, BF16, FP8 (E4M3/E5M2), INT8, and INT4 conversion and vectorized compute via `src/core/math/mixed_precision.{h,cpp}`.
- **Cache-Oblivious Algorithms**: Recursive divide-and-conquer MatMul, Transpose, 2D Conv, and SpMV for optimal performance across all cache hierarchies via `src/core/math/cache_oblivious.{h,cpp}`.
- **Block Sparse SIMD**: SELLPACK/VBSF block-sparse SpMV and SpMM with AVX-512/AVX2 vectorization via `src/core/math/block_sparse.{h,cpp}`.
- **Tensor Core Emulation**: AVX-512 VNNI (INT8), AVX-512 BF16, and Intel AMX matrix emulation via `src/core/math/tensor_core_emulation.{h,cpp}`.

### 3.5 Data Structure & Scheduling Hardening
- **3-Level TLB Cache**: L1 thread-local (256 sets × 8 ways, CLOCK replacement, AVX2-vectorized tag comparison), L2 thread-local (1024 sets × 16 ways, LRU), and a shared sharded L2 (16 shards × 256 sets × 4 ways, atomic LRU) for O(1) virtual→region translation in the SIGSEGV handler.
- **SPSC Rings**: Lock-free Single-Producer Single-Consumer task rings per stream as a fast path on top of the Chase-Lev work-stealing deques.
- **Dynamic Heuristics Eliminated**: All previously hardcoded magic numbers (bandwidth ceilings, IPC estimates, thread search patterns, workspace sizes) replaced with CPUID-probed hardware detection, Z-score bandwidth classification, and problem-size-based computation.

---

## 4. Test Suite Summary

The VGRE test suite covers memory management, compiler translation (incl. the from-scratch
CUDA-C front-end + eight differential fuzzers against both execution tiers),
compute libraries, and clustering. The latest hosted cross-platform verification
is linked at the top of this document. The per-area breakdown below is
**historical/indicative** and does not assert current test counts.

| Suite | Focus | Result |
|---|---|---|
| Unit | TLB, Scheduler, Concurrency, Security, Data Structures | ✅ Passed |
| Integration | JIT, Graphs, UVM, Streams, Multi-Device, Cluster | ✅ Passed |
| API | cuBLAS, cuDNN, cuFFT, cuSPARSE, cuSolver, cuRAND, CUDA RT | ✅ Passed |
| Compiler | CUDA-C front-end (lexer/parser/codegen), differential fuzzers, PTX interpreter + compiled tier | ✅ Passed |
| Core | Dirty Page Tracking, Radix Sort, Texture, VEB Tree | ✅ Passed |
| Advanced | TCP Cluster Security, Hybrid Auth, Diagnostic Logger | ✅ Passed |
| Platform | Cross-Platform Worker | ✅ Passed |

---

## 5. Build & Test Commands

To build and run the test suite locally:

```bash
# Create build directory
mkdir -p build && cd build

# Configure CMAKE with optimized Release flags
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build targets in parallel
cmake --build . -j$(nproc)

# Execute full test suite
ctest --output-on-failure -j$(nproc)
```
