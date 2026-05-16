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
| cuBLAS (complex) | Listed in docs | **IMPLEMENTED** | C/Z Level-1/2/3: GEMM, GEMV, AXPY, DOT, SCAL, NRM2, ROT, COPY, SWAP, TRSV, GER, TRSM, SYRK, SYR2K, SYMM, TRMM all present |
| cuDNN (legacy) | All major ops | All major ops | Fwd/bwd for conv, pool, activation, softmax, BN, dropout, RNN, attention, LRN, divisive norm, CTC loss, tensor ops |
| cuDNN Backend API | Wired to legacy | Partially wired | Handles conv, activation, pool, softmax, reduction, matmul, BN, norm, RNN, concat, signal, gen_stats, bn_bwd_weights. **No attention routing.** |
| NCCL | P2P + collectives | P2P + collectives | Single-node shared-memory; multi-node routes through TCPCluster (functional but not RDMA-optimized) |
| cuFFT | O(n log n) FFT | O(n log n) FFT | Built-in Cooley-Tukey + Bluestein. Optional FFTW3 delegation via `VGRE_HAS_FFTW3`. Plan caching via handle map. |
| cuRAND | 4 generators | 4 generators | All use sequential `std::mt19937_64`; no parallel stream-safe generator partitioning |
| cuSOLVER | 8 dense routines | 8 dense routines | `potrf`, `geqrf`, `gesvd`, `syevd`, `getrf`, `getrs`, `ormqr`, `gelsd` — all via LAPACK delegation with proper workspace queries. Missing: sparse solvers |
| cuSPARSE | CSR SpMV/SpMM | Full generic API | CSR, COO, CSC descriptors; SpMV, SpMM; SparseToDense; DenseToSparse; SpSV (tri solve); SpGEMM; ILU0; IC0 |
| cuBLASLt | Basic matmul + epilogues | Full epilogue set + LRU cache | ReLU, GELU, Bias, DRELU, DGELU, DRELU_BGRAD, DGELU_BGRAD, BGRADA/BGRADB; AlgoGetHeuristic + LRU cache |
| Profiling | Kernel timeline + Chrome trace | Kernel timeline + Chrome trace | `InstructionSample` is heuristic-only (no hardware PC counter). No separate `src/advanced/profiling/` directory — all in `runtime_profiler.cpp` |
| File Organization | All large files split | **DONE** | 8 monolithic files (>800 lines) split into 22 smaller files. All 353 functions verified present. 110/110 tests pass |
| K8s Plugin | Go gRPC | Go gRPC | Basic daemonset; no dynamic device discovery |
| SLURM GRES | C shared lib | C shared lib | Basic vGPU allocation tracker |

---

## 2. Tier 1 — Critical Missing (limits framework compatibility)

### 2.1 Complex BLAS (cuBLAS C/Z precisions)

**Status**: **IMPLEMENTED** (2026-05-15). All core C/Z routines are in `src/api/cublas/cublas_complex.cpp`.

| Implemented API | Level | Notes |
|---|---|---|
| `cublasCgemm_v2` / `cublasZgemm_v2` | L3 | Full complex GEMM with op(A), op(B) support incl. conjugate transpose |
| `cublasCgemv_v2` / `cublasZgemv_v2` | L2 | Complex matrix-vector multiply with N/T/C transpose |
| `cublasCaxpy_v2` / `cublasZaxpy_v2` | L1 | Complex vector AXPY |
| `cublasCdotc_v2` / `cublasZdotc_v2` | L1 | Conjugated dot product |
| `cublasCdotu_v2` / `cublasZdotu_v2` | L1 | Unconjugated dot product |
| `cublasCscal_v2` / `cublasZscal_v2` / `cublasCsscal_v2` / `cublasZdscal_v2` | L1 | Complex and real-scaled complex |
| `cublasCcopy_v2` / `cublasZcopy_v2` | L1 | Complex vector copy |
| `cublasCswap_v2` / `cublasZswap_v2` | L1 | Complex vector swap |
| `cublasScnrm2_v2` / `cublasDznrm2_v2` | L1 | Complex Euclidean norm |
| `cublasIcamax_v2` / `cublasIzamax_v2` | L1 | Index of max absolute value |
| `cublasScasum_v2` / `cublasDzasum_v2` | L1 | Sum of absolute values |
| `cublasCrot_v2` / `cublasZrot_v2` | L1 | Complex plane rotation |
| `cublasCtrsv_v2` / `cublasZtrsv_v2` | L2 | Complex triangular solve |
| `cublasCgeru_v2` / `cublasZgeru_v2` | L2 | Unconjugated rank-1 update |
| `cublasCgerc_v2` / `cublasZgerc_v2` | L2 | Conjugated rank-1 update |
| `cublasCsyrk_v2` / `cublasZsyrk_v2` | L3 | Complex symmetric rank-k |
| `cublasCsyr2k_v2` / `cublasZsyr2k_v2` | L3 | Complex symmetric rank-2k |
| `cublasCtrsm_v2` / `cublasZtrsm_v2` | L3 | Complex triangular solve (matrix) |
| `cublasCsymm_v2` / `cublasZsymm_v2` | L3 | Complex symmetric matrix multiply |
| `cublasCtrmm_v2` / `cublasZtrmm_v2` | L3 | Complex triangular matrix multiply |
| `cublasChemm`/`cublasCherk`/`cublasCher2k` | L3 | Hermitian variants (pre-existing in `cublas_hermitian.cpp`) |

All 21 test cases pass. OpenMP parallelization enabled for CGEMM/ZGEMM.

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

### 2.3 cuFFT — O(n log n) FFT Backend

**Status**: **IMPLEMENTED** (2026-05-15). Built-in Cooley-Tukey radix-2 + Bluestein for arbitrary sizes. Optional FFTW3 delegation.

| Implemented | Notes |
|---|---|
| Cooley-Tukey radix-2 FFT | O(n log n) for power-of-2 sizes, OpenMP-parallelized butterfly |
| Bluestein's algorithm | O(n log n) for arbitrary sizes (prime, non-power-of-2) |
| FFTW3 delegation | Compile with `VGRE_HAS_FFTW3` for fftw3/fftw3f backend (auto-detected by CMake) |
| 1D/2D/3D C2C, R2C, C2R | Float and double precision (C2C, Z2Z, R2C, C2R, D2Z, Z2D) |
| Batched transforms | `cufftPlan1d` with batch > 1, parallelized over batches |
| Plan handle caching | Plans stored in global map, reusable across exec calls |

**Remaining**: Half-precision FFT (`__half` DFT) — returns `CUFFT_NOT_SUPPORTED`.

All 11 test cases pass. 112/112 full suite pass.

---

### 2.4 cuSPARSE — Format Conversions & Sparse Solvers

**Status**: **PARTIALLY IMPLEMENTED** (2026-05-16).

| API | Status | Notes |
|---|---|---|
| `cusparseCreateCsr` / `cusparseCreateCoo` (generic API descriptors) | **DONE** | Implemented |
| `cusparseCreateCsc` | **DONE** | Added in `cusparse_format.cpp` — stored as transposed CSR |
| `cusparseSparseToDense` / `cusparseSparseToDense_bufferSize` | **DONE** | Added in `cusparse_format.cpp` |
| `cusparseDenseToSparse_bufferSize` / `_analysis` / `_compress` | **DONE** | Added in `cusparse_format.cpp` |
| `cusparseSpMatGetAttribute` / `cusparseSpMatSetAttribute` | **DONE** | Fill mode, diagonal type |
| `cusparseSpMatGetSize` / `cusparseCsrSetPointers` | **DONE** | Required by DenseToSparse workflow |
| `cusparseSpSV_createDescr` / `_destroyDescr` / `_bufferSize` / `_analysis` / `_solve` | **DONE** | Full forward/backward/transposed triangular solve via gather+scatter in `cusparse_triangular.cpp` |
| `cusparseSpGEMM` (sparse × sparse) | **DONE** (2026-05-16) | 3-phase workEstimation+compute+copy in `cusparse_factorization.cpp`; float + double |
| `cusparseScsrilu02` / `cusparseDcsrilu02` (ILU0) | **DONE** (2026-05-16) | In-place ILU with zero fill-in; marker-based O(nnz * avg_row) |
| `cusparseScsric02` / `cusparseDcsric02` (IC0) | **DONE** (2026-05-16) | In-place incomplete Cholesky with zero fill-in for SPD matrices |
| Sparse direct factorization (UMFPACK-style full fill-in) | **MISSING** | Requires external sparse LAPACK library |

---

## 3. Tier 2 — Medium Priority Missing (performance/usability gaps)

### 3.1 cuBLASLt — Epilogues & Heuristics

**Status**: **IMPLEMENTED** (2026-05-16).

| API | Status | Notes |
|---|---|---|
| `CUBLASLT_EPILOGUE_RELU_AUX` / `GELU_AUX` | **DONE** | Implemented — same as plain ReLU/GELU (no separate aux mask allocation needed for CPU model) |
| `CUBLASLT_EPILOGUE_DRELU` / `DGELU` | **DONE** | Backward pass gradients: element-wise ReLU/GELU derivative applied in-place |
| `CUBLASLT_EPILOGUE_DRELU_BGRAD` / `DGELU_BGRAD` | **DONE** | Bias gradient (column-sum) + ReLU/GELU gradient |
| `CUBLASLT_EPILOGUE_BGRADA` / `BGRADB` | **DONE** | Pure bias gradient column-sum written to `epilogueAuxPtr` |
| `cublasLtMatmulAlgoGetHeuristic` (singular) | **DONE** | Added singular form; plural `AlgoGetHeuristics` also exists |
| A/B/C/D scale pointers, amaxD | **DONE** | Stored in MatmulDesc; quantization scale semantics deferred to future |
| Algorithm caching | **DONE** (2026-05-16) | LRU cache (`kAlgoCacheMax=128` entries) keyed on M/N/K/dtype/epilogue/transA/transB; `cublasLtMatmulAlgoGetHeuristic` populates it |

---

### 3.2 cuDNN Backend API — Attention Routing

**Status**: **IMPLEMENTED** (2026-05-16).

`CUDNN_BACKEND_OPERATION_ATTENTION_DESCRIPTOR = 36` added. The Execute handler performs scaled dot-product attention directly: `softmax(Q·K^T / scale) · V` with float data. Tensor layout: `(batch, heads, seqLen, head_dim)`.

---

### 3.3 cuRAND — Thread Safety

**Status**: **IMPLEMENTED** (2026-05-16).

Registry changed to `unique_ptr<GeneratorState>` with `std::mutex genMutex` per handle. All generate calls (`curandGenerate`, `GenerateUniform`, `GenerateNormal`, `GenerateLogNormal`, `GeneratePoisson`, etc.) lock `st->genMutex` before accessing `st->engine`. Concurrent calls on different handles are fully parallel; concurrent calls on the same handle serialize correctly.

| Remaining limitation |
|---|
| Device-side `curand_*` API returns `CURAND_STATUS_NOT_SUPPORTED` by design (no device model in CPU emulation) |

---

### 3.4 Profiling — Hardware Sampling Caveat

| Missing | Impact | Notes |
|---|---|---|
| True instruction-level hardware sampling | `InstructionSample` is pure heuristic | `runtime_profiler.cpp` estimates instruction mix from kernel source heuristics, not hardware PMU counters. No separate `src/advanced/profiling/` directory. |
| CUPTI-equivalent API surface | PyTorch profiler integration | Only basic kernel timeline + Chrome trace export exists |

---

### 3.5 Texture / Surface — Filter Gaps

| Feature | Status | Notes |
|---|---|---|
| Trilinear filtering (`LINEAR` + mipmaps) | **DONE** | `tex2DLod` performs bilinear sampling at each mip level then blends linearly between adjacent levels |
| Anisotropic filtering | **DONE** | `maxAnisotropy` used in `tex2D` — up to 16× samples averaged along U and V axes |
| SRGB gamma decode (`CU_TRSF_SRGB`) | **DONE** (2026-05-16) | `TextureDescriptor::srgbDecode` flag; when true, UINT8/UINT16 texels decoded via IEC 61966-2-1 (`srgbToLinear`) before sampling |

---

## 4. Tier 3 — Low Priority / Architectural Limitations

### 4.1 NCCL — Multi-Node Performance

| Limitation | Impact | Notes |
|---|---|---|
| No RDMA transport | High-latency for multi-node | All multi-node traffic goes through TCPCluster TCP sockets |
| No NVLink-style P2P | Single-node only uses shared memory | `ncclSend/Recv` use shared-memory slots on single node; TCP routing is functional but not optimized |
| No topology-aware tree/ring | AllReduce uses flat barrier for small tensors | Ring reduce-scatter + all-gather is used for tensors > 1 MB (`kRingThreshold`); flat barrier for smaller tensors (latency-optimal) |

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

## 6. Resolved Items (2026-05-15)

The following items that were previously potential concerns have been resolved:

| Item | Resolution | Date |
|---|---|---|
| Monolithic file maintainability | 8 files (>800 lines) split into 22 smaller files with zero method loss | 2026-05-15 |
| `cudart_shim.cpp` too large (1429 lines) | Split into 4 files: core (994), memory_pool (240), cooperative (201), mipmap (90) | 2026-05-15 |
| `cublas_level2.cpp` too large (1018 lines) | Split into standard (455) + packed/banded (570) | 2026-05-15 |
| `cublas_level3.cpp` too large (1173 lines) | Split into GEMM/core (1025) + batched BLAS3 (154) | 2026-05-15 |
| `cuda_interceptor.cpp` too large (1147 lines) | Split into core (869) + device attrs (309) | 2026-05-15 |
| `vector_engine.cpp` too large (1163 lines) | Split into core (460) + float ops (371) + double ops (412) | 2026-05-15 |
| `graph_manager.cpp` too large (975 lines) | Split into lifecycle (228) + nodes (519) + serde (265) | 2026-05-15 |
| `adaptive_execution_engine.cpp` too large (1082 lines) | Split into core (489) + record (251) + tune (535) | 2026-05-15 |
| `hybrid_compute_manager.cpp` too large (930 lines) | Split into core (342) + remote (288) + workload (383) | 2026-05-15 |

---

## 7. Recommendations

1. ~~**Complex BLAS**~~ — **DONE** (2026-05-15). C/Z Level-1/2/3 implemented.
2. ~~**cuFFT FFTW3 delegation**~~ — **DONE** (2026-05-15). O(n log n) Cooley-Tukey + Bluestein + optional FFTW3.
3. ~~**cuSOLVER LU/least-squares**~~ — **DONE** (2026-05-15). getrf, getrs, ormqr, gelsd all via LAPACK delegation.
4. **cuSPARSE format conversions + sparse triangular solve** would unblock sparse ML. Priority: **Medium**.
5. **cuBLASLt heuristic selection** would improve matmul performance. Priority: **Low**.
6. **cuDNN Backend attention routing** is needed for frameworks using Backend API exclusively. Priority: **Low** (legacy attention API works).
7. **Cross-platform discipline**: New OS-dependent code must use `*_linux.cpp` / `*_macos.cpp` / `*_win32.cpp` split pattern.
8. **Memory discipline**: Every new memory allocation goes through `MemoryManager`. Every new compute path integrates with `AdaptiveExecutionEngine`.
9. **File size discipline**: No source file should exceed 800 lines for shims or 600 lines for core logic. Split by concern when approaching limits.

---

*Last updated: 2026-05-16 (second pass). 113/113 tests passing. Sparse factorization (ILU0/IC0/SpGEMM), caching, and parallel-test serialization added.*
