# VGRE — Missing Features & Known Limitations

**Research Date**: 2026-05-15 (deep source-code audit)  
**Audit Methodology**: Full `src/` and `include/` grep + manual file inspection; every claim verified against actual source code.  
**Scope**: Only genuinely missing, incomplete, or architecturally limited features are listed below. All implemented features are documented in `PROJECT_STATUS.md` and `api_reference.md`.

---

## 1. Coverage Reality Check

| Category | Claimed | Actual | Notes |
|---|---|---|---|
| CUDA Runtime API | ~101 funcs | ~95 funcs | Most core functions implemented; some legacy/debug APIs return no-op stubs |
| CUDA Driver API | ~56 funcs | ~55 funcs | NvSciSync not applicable on Linux; module loading fully implemented |
| CUDA Graphs | All node types | All 11 node types | Exec update v2 exists in interceptor, not a separate file |
| PTX ISA | ~120 instructions | ~110 instructions | Core arithmetic, warp, atomics, MMA (Ampere/Hopper/Blackwell), TMA, texture, grid.sync all present |
| cuBLAS (real) | Level 1/2/3 complete | Level 1/2/3 complete | S/D variants + Hermitian (Cherk/Zherk/Cher2k/Zher2k/Chemm/Zhemm) all implemented |
| cuBLAS (complex) | Listed in docs | **NOT IMPLEMENTED** | No C/Z Level-1/2/3 GEMM/GEMV/AXPY/DOT/SCAL/NRM2/ROT |
| cuDNN (legacy) | All major ops | All major ops | Fwd/bwd for conv, pool, activation, softmax, BN, dropout, RNN, attention, LRN, divisive norm, CTC loss, tensor ops |
| cuDNN Backend API | Wired to legacy | Partially wired | Handles conv, activation, pool, softmax, reduction, matmul, BN, norm, RNN, concat, signal, gen_stats, bn_bwd_weights. **No attention routing.** |
| NCCL | P2P + collectives | P2P + collectives | Single-node shared-memory; multi-node routes through TCPCluster (functional but not RDMA-optimized) |
| cuFFT | Reference DFT | Reference DFT | No FFTW3/MKL delegation; O(n²) DFT only — too slow for production large transforms |
| cuRAND | 4 generators | 4 generators | All use sequential `std::mt19937_64`; no parallel stream-safe generator partitioning |
| cuSOLVER | 4 dense routines | 4 dense routines | `potrf`, `geqrf`, `gesvd`, `syevd` only. **Missing**: `getrf` (LU), `getrs`, `ormqr`, `gelsd`, sparse solvers |
| cuSPARSE | CSR SpMV/SpMM | CSR SpMV/SpMM | Missing: format conversions (CSR↔CSC↔COO), sparse triangular solve, sparse factorization |
| cuBLASLt | Basic matmul + epilogues | Basic matmul + ReLU/GELU/Bias | Missing: more complex fused epilogues, heuristic selection, algorithm caching |
| Profiling | Kernel timeline + Chrome trace | Kernel timeline + Chrome trace | `InstructionSample` is heuristic-only (no hardware PC counter). No separate `src/advanced/profiling/` directory — all in `runtime_profiler.cpp` |
| K8s Plugin | Go gRPC | Go gRPC | Basic daemonset; no dynamic device discovery |
| SLURM GRES | C shared lib | C shared lib | Basic vGPU allocation tracker |

---

## 2. Tier 1 — Critical Missing (limits framework compatibility)

### 2.1 Complex BLAS (cuBLAS C/Z precisions)

**Status**: **COMPLETELY MISSING**. No `cublasC*` or `cublasZ*` routines exist in any source file.

| Missing API | Impact | Notes |
|---|---|---|
| `cublasCgemm` / `cublasZgemm` | Blocks PyTorch complex linear layers | No complex GEMM in `cublas_level3.cpp` |
| `cublasCgemv` / `cublasZgemv` | Blocks complex matrix-vector ops | No complex GEMV |
| `cublasCaxpy` / `cublasZaxpy` | Blocks complex vector ops | No complex Level-1 |
| `cublasCdotc` / `cublasZdotc` | Blocks complex dot products | Hermitian dot product not implemented |
| `cublasCscal` / `cublasZscal` | Blocks complex vector scaling | |
| `cublasCcopy` / `cublasZcopy` | Blocks complex vector copy | |
| `cublasCtrsv` / `cublasZtrsv` | Blocks complex triangular solve | |
| `cublasCsyrk` / `cublasZsyrk` | Blocks complex symmetric rank-k | |
| `cublasChemm`/`cublasCherk`/`cublasCher2k` | Hermitian variants **ARE implemented** | `src/api/cublas/cublas_hermitian.cpp` |

**Implementation path**: Add `cuComplex`/`cuDoubleComplex` reference loops alongside existing S/D paths. Can reuse existing CBLAS if linked against complex LAPACK.

---

### 2.2 cuSOLVER — Advanced Dense + Sparse

| Missing API | Impact | Notes |
|---|---|---|
| `cusolverDnSgetrf` / `cusolverDnDgetrf` (LU factorization) | Blocks general matrix solve | Only Cholesky (`potrf`) exists |
| `cusolverDnSgetrs` / `cusolverDnDgetrs` (triangular solve from LU) | Needs `getrf` first | |
| `cusolverDnSormqr` / `cusolverDnDormqr` (apply Q from QR) | Needs `geqrf` result | |
| `cusolverDnSgelsd` / `cusolverDnDgelsd` (least squares) | Blocks `torch.linalg.lstsq` | |
| Sparse solvers (`cusparseSpSV`, sparse Cholesky, etc.) | Blocks sparse ML workflows | `src/api/cusparse/` only has dense-style CSR SpMV/SpMM |

---

### 2.3 cuFFT — No Optimized Backend

| Missing | Impact | Notes |
|---|---|---|
| FFTW3/MKL delegation | Large transforms are O(n²) slow | Reference DFT is correct but unusable for >1K points |
| Real-world plan caching | Repeated plans recompute twiddle factors | `cufftPlan1d` always rebuilds from scratch |
| Half-precision FFT | `CUFFT_STATUS_NOT_SUPPORTED` returned | No `__half` DFT path |

---

### 2.4 cuSPARSE — Format Conversions & Sparse Solvers

| Missing API | Impact | Notes |
|---|---|---|
| `cusparseCreateCsr` / `cusparseCreateCoo` (generic API descriptors) | Modern sparse API | Only legacy `cusparseSpMV` with CSR + COO hardcoded exists |
| `cusparseSparseToDense` / `cusparseDenseToSparse` | Format conversion | Not implemented |
| `cusparseSpSV` (sparse triangular solve) | Sparse preconditioners | Not implemented |
| Sparse factorization (Cholesky, LU, QR) | Scientific computing | Not implemented |

---

## 3. Tier 2 — Medium Priority Missing (performance/usability gaps)

### 3.1 cuBLASLt — Missing Epilogues & Heuristics

| Missing | Impact | Notes |
|---|---|---|
| `CUBLASLT_EPILOGUE_RELU_AUX` | Needs aux mask output | Only plain ReLU exists |
| `CUBLASLT_EPILOGUE_GELU_AUX` | Needs aux mask output | Only plain GELU exists |
| `CUBLASLT_EPILOGUE_DRELU_DGELU` | Bwd epilogues | Not implemented |
| `CUBLASLT_EPILOGUE_BGRADIENT` | Bias gradient output | Not implemented |
| Heuristic selection (`cublasLtMatmulAlgoGetHeuristic`) | Always returns 1 algo | No actual perf tuning; always falls back to reference |
| Algorithm caching | Repeated matmuls rebuild plan | No cache in `cublasLtMatmul` |

---

### 3.2 cuDNN Backend API — Attention Routing

| Missing | Impact | Notes |
|---|---|---|
| `CUDNN_BACKEND_OPERATION_ATTENTION_DESCRIPTOR` | Backend API for MultiHeadAttn | `cudnnMultiHeadAttnForward` exists as legacy API, but Backend `Execute` returns `NOT_SUPPORTED` for attention descriptors |

---

### 3.3 cuRAND — Parallel Stream Safety

| Missing | Impact | Notes |
|---|---|---|
| Per-stream generator state | Race conditions if multiple host threads call `curandGenerate` concurrently | All calls share one `std::mt19937_64` engine in the handle; no thread-local or stream-local partitioning |
| Device-side `curand_*` API | Kernel-side RNG | Returns `CURAND_STATUS_NOT_SUPPORTED` by design (no device model in CPU emulation) |

---

### 3.4 Profiling — Hardware Sampling Caveat

| Missing | Impact | Notes |
|---|---|---|
| True instruction-level hardware sampling | `InstructionSample` is pure heuristic | `runtime_profiler.cpp` estimates instruction mix from kernel source heuristics, not hardware PMU counters. No separate `src/advanced/profiling/` directory. |
| CUPTI-equivalent API surface | PyTorch profiler integration | Only basic kernel timeline + Chrome trace export exists |

---

### 3.5 Texture / Surface — Mipmapping Gaps

| Missing | Impact | Notes |
|---|---|---|
| Trilinear filtering (`LINEAR` + mipmaps) | `TextureFilterMode::LINEAR` on mipmapped textures | `boxFilter2D` generates mipmaps, but sample path in `texture_manager.cpp` does not do trilinear interpolation |
| Anisotropic filtering | `maxAnisotropy` stored but unused | `cuTexRefSetMaxAnisotropy` stores value; sampler never reads it |
| SRGB gamma decode | `CU_TRSF_SRGB` accepted but ignored | No gamma pipeline in CPU model |

---

## 4. Tier 3 — Low Priority / Architectural Limitations

### 4.1 NCCL — Multi-Node Performance

| Limitation | Impact | Notes |
|---|---|---|
| No RDMA transport | High-latency for multi-node | All multi-node traffic goes through TCPCluster TCP sockets |
| No NVLink-style P2P | Single-node only uses shared memory | `ncclSend/Recv` use shared-memory slots on single node; TCP routing is functional but not optimized |
| No topology-aware tree/ring | AllReduce uses flat all-to-all | Missing ring/tree algorithm selection based on topology |

---

### 4.2 CUDA Runtime — Legacy/Debug Stubs

| API | Status | Notes |
|---|---|---|
| `cudaThreadExit` / `cudaThreadSynchronize` | No-op wrappers | Legacy 1.x APIs; thin wrappers around device APIs |
| `cudaDeviceFlushGPUDirectRDMAWrites` | No-op | RDMA not applicable in CPU model |
| `cudaProfilerStart` / `cudaProfilerStop` | No-op | No external profiler attached |
| `cudaGraphDebugDotPrint` | Basic DOT output | Emits GraphViz; no per-node attribute detail |

---

### 4.3 PTX ISA — Edge Cases

| Limitation | Impact | Notes |
|---|---|---|
| `tex` vector variants (v2/v4) return replicated scalar | Apps expecting per-channel data get duplicates | `TextureManager` is single-channel float; vector ops replicate scalar to all channels |
| `txq` returns conservative defaults | 1024×1024×1, 1 channel | No texture metadata query support in `TextureManager` |
| `cp.async.bulk.tensor.{3,4,5}d` → serial loop | TMA is emulated as strided memcpy | Correctness preserved; performance is serial |
| `tcgen05.mma` delegates to `wgmma` | Blackwell tensor core emulated as Hopper | Correct for BF16/FP16/TF32 shapes; no true SM100-specific behavior |

---

## 5. Documentation vs. Source Discrepancies

The following items were **previously claimed as separate files** in `implementationPlan.md` but are actually **implemented within existing files**:

| Claimed File | Actual Location | Status |
|---|---|---|
| `src/core/graph/graph_manager_exec_update_v2.cpp` | `src/api/cuda_interceptor_graphs.cpp::graphExecUpdateV2()` | Implemented |
| `src/api/cudnn/cudnn_int8_packed.cpp` | `src/api/cudnn/cudnn_{convolution,pooling,activation}.cpp` (INT8x4/INT8x32 branches) | Implemented |
| `src/api/cudart/cudart_shim_cdp.cpp` | `src/runtime/cdp_executor.cpp` | Implemented |
| `src/advanced/profiling/cupti_equivalent.cpp` | `src/advanced/runtime_profiler.cpp` | Implemented |
| `src/advanced/profiling/kernel_timeline.cpp` | `src/advanced/runtime_profiler.cpp` | Implemented |
| `src/advanced/profiling/instruction_sampler.cpp` | `src/advanced/runtime_profiler.cpp` | Implemented |

---

## 6. Recommendations

1. **Complex BLAS** is the largest functional gap for PyTorch/TensorFlow complex tensor support. Priority: **High**.
2. **cuFFT FFTW3 delegation** would make the library usable for real workloads. Priority: **High**.
3. **cuSOLVER LU/least-squares** (`getrf`, `getrs`, `gelsd`) are needed for general linear algebra. Priority: **Medium**.
4. **cuSPARSE format conversions + sparse triangular solve** would unblock sparse ML. Priority: **Medium**.
5. **cuBLASLt heuristic selection** would improve matmul performance. Priority: **Low**.
6. **cuDNN Backend attention routing** is needed for frameworks using Backend API exclusively. Priority: **Low** (legacy attention API works).
7. **Cross-platform discipline**: New OS-dependent code must use `*_linux.cpp` / `*_macos.cpp` / `*_win32.cpp` split pattern.
8. **Memory discipline**: Every new memory allocation goes through `MemoryManager`. Every new compute path integrates with `AdaptiveExecutionEngine`.

---

*Last updated: 2026-05-15. Verified against git commit `803b76f`.*
