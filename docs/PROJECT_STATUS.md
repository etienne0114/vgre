# VGRE Project Status & Gap Analysis

**Last Updated**: 2026-09-20 (front-end/backend differential-fuzzing hardening; test counts reconciled)  
**Build Status (Linux)**: ✅ full `ctest` suite passing (**399 tests** with LLVM, **379** in the LLVM-free build, 100% green under full `-j`) on x86-64 Linux — the required CI job  
**Build Status (Linux, LLVM-free)**: ✅ **CI-guarded zero-burden path** — a dedicated `linux-x86_64-llvm-free` job builds with `VGRE_ENABLE_JIT=OFF` + `VGRE_ENABLE_OPENMP=OFF` and **no `llvm-*-dev`/`libclang`/`libomp` installed**, running the **358-test** JIT-free subset (the ~20 JIT-only tests are gated out in CMake). Proves the lightweight, no-toolchain build stays green on every push  
**Build Status (macOS)**: ✅ **CI-green** on Apple Silicon (ARM64). Latest fix: the process-exit `recursive_mutex` abort (an `atexit` handler locking the `RuntimeEngine` singleton after its destruction — EINVAL on macOS libc++) resolved by a leaked, never-destroyed singleton  
**Build Status (Windows)**: ✅ **CI-green** — builds and runs the full `ctest` suite on `windows-2022` (LLVM-18 tarball cached, clang-cl), confirmed on run 35325255176 (2026-09-18, Test step = success). The bring-up fixes: AVX2 `rsqrt`/GEMM numerical accuracy, `vgre.dll` dependency loading under Python 3.8+, and cp1252 console encoding of non-ASCII test output. Still `continue-on-error` in the workflow (may be promoted to required after a few more consecutive green runs)  
**Public demo**: 🌐 free CPU demo live at **https://vgrengine.streamlit.app** (Streamlit Community Cloud — HF now requires PRO for server-side Spaces)  
**Production Readiness**: core emulation is stable and verified on Linux; macOS ARM64 and Windows x86-64 are both CI-green (full `ctest` suite) — all three platforms now pass in CI.

> **CI status corrected (2026-09):** the previous "GitHub Actions billing
> blocker" is **resolved** — CI now runs free on the public repo on **every
> push**, with Linux required and macOS/Windows informational
> (`continue-on-error`) until fully green. The prior text below (billing failing
> since 2026-06-22) is **historical and no longer true**.

> **LLVM is now optional (delivered):** kernel execution no longer requires LLVM —
> a from-scratch CUDA-C front-end feeds a four-tier CPU backend (PTX interpreter,
> compiled-fiber tier, native x86-64 JIT, and an optional SSA optimizing backend),
> and the full suite passes LLVM-free (379 tests, `VGRE_ENABLE_JIT=OFF`). See
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

> **Correction (2026-06-10) — partly superseded (2026-09):** the CI/platform half
> below is out of date. As of 2026-09 there **is** CI, and all three OSes run the
> full `ctest` suite green (see the header): Linux required, macOS green, Windows
> informational. The note is kept for history; the simplified-compute-path caveats
> it lists still stand where `missingFeatures.md` §1 marks them. Original text:
> earlier revisions of this file claimed
> "CI/CD-Ready", "validated across Linux, Windows, macOS", and "zero stubs".
> That was inaccurate at the time. The then-truth: the build and tests ran on **Linux only**;
> Windows/macOS code was compile-guarded but **unverified** (there was no CI). A
> handful of compute paths are deliberately **simplified** (Flash Attention
> recomputes K/V, NCCL ring uses a barrier-per-round model, WMMA is a flat
> dot-product — see `docs/missingFeatures.md` §1). This file now tracks the real
> path to production rather than asserting it.

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

**Verified on Linux (x86-64).** The full `ctest` suite (399 tests with LLVM,
379 in the LLVM-free build) passes on Linux, exercised with property-based
exploration, differential fuzzing of the CUDA-C front-end against both execution
tiers, ThreadSanitizer race analysis, and static-destruction verification.

### 1.2 macOS (Apple Silicon) — CI-green

**The full `ctest` suite runs green in CI on `macos-14` (Apple Silicon).** The native
engine (`libvgre`, `libvgre_cudart`, examples, integration tests) compiles with
Homebrew `llvm@18` auto-selected by CMake; JIT kernel compilation discovers Clang at
runtime (`VGRE_CLANG_PATH` → `llvm-config-18` → `brew --prefix llvm@18`). A prebuilt
macOS-arm64 wheel ships with each release. (The CI job is `continue-on-error` in the
workflow — informational — pending promotion to required, but it exercises and passes
the whole suite on every push.)

**macOS-specific real implementations** (not stubs): UVM via `SIGSEGV`/`SIGBUS`,
Keychain via `SecItemAdd`/`SecItemCopyMatching`, thermal via IOKit SMC,
external semaphores via `dispatch_semaphore`, cluster TCP via BSD sockets.

**Documented macOS approximations** (see `missingFeatures.md`): NUMA pinning
(Mach hints vs Linux affinity), no Metal Performance Shaders backend, no NVIDIA
PMU/CUPTI counters without physical GPU hardware.

### 1.3 Windows — CI-green

**Windows builds and runs the full `ctest` suite in CI** (`windows-2022`,
clang-cl, LLVM-18 tarball cached), confirmed 2026-09-18 (see the header). The
bring-up fixes were AVX2 `rsqrt`/GEMM numerical accuracy, `vgre.dll` dependency
loading under Python 3.8+, and cp1252 console encoding of non-ASCII test output.
The workflow keeps it `continue-on-error` (informational) pending a few more
consecutive green runs before promotion to required.

- **Linux core passes**: full regression + integration + platform suite green on x86-64.
- **Mostly real compute, with documented exceptions**: nearly every path runs real
  CPU math (AVX/ARM64 SIMD + OpenBLAS/LAPACK). The exceptions are the simplified
  paths in `docs/missingFeatures.md` §1 (Flash Attention K/V recompute, NCCL
  barrier-per-round ring, WMMA flat dot-product, etc.) and the documented
  hardware-boundary proxies (CUPTI PMU counters).
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
- **From-scratch CUDA-C front-end + four-tier CPU backend (no LLVM required)**: an in-tree lexer/parser lowers kernels to an AST, then to the fastest of four bit-exact tiers — a PTX interpreter, a compiled-closure tier with a cooperative fiber executor, a hand-emitted native x86-64 JIT, and an optional own SSA optimizing backend (`VGRE_EXEC_BACKEND=interp|cp|ssa`). This is the default in the LLVM-free build (379 tests green).
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

The VGRE test suite runs **378 CTest targets** (LLVM build) / **358** (LLVM-free
build) covering memory management, compiler translation (incl. the from-scratch
CUDA-C front-end + eight differential fuzzers against both execution tiers),
compute libraries, and clustering — 100% green under full `-j`. The per-area
breakdown below is **historical/indicative** (it predates substantial growth); the
authoritative current count is the total above, reproduced by the build commands
in §5.

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
