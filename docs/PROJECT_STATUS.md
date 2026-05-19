# VGRE Project Status

**Last Updated**: 2026-05-19 (code-verified audit)  
**Build**: 117/117 tests pass — but see "Test Coverage Gaps" below  
**Status**: Advanced prototype. Core numerical paths are real. Several critical ML APIs produce wrong results or are absent.

---

## What Actually Works (Code-Verified)

### Kernel Execution
- LLVM-18 ORC JIT: Clang AST parse → LLVM IR → native code. Full PTX translator.
- Block worker pool: real `__syncthreads()` support via barrier objects.
- CUDA Dynamic Parallelism: recursive child kernel dispatch through host bridge.
- CUDA Graphs: real DAG, topological sort, kernel fusion.
- KernelCache: now includes sourceHash integrity check + AST collision eviction.

### Memory
- `cudaMalloc/Free`, pinned memory, pool allocator, copy engine.
- UVM: `cudaMallocManaged`, `cudaMemAdvise` (real `madvise()`), `cudaMemPrefetchAsync` (real `mbind()` on Linux).
- Virtual memory: `cuMemCreate/Map/SetAccess` via `mmap(PROT_NONE)` + `mprotect`.
- Texture: 1D/2D/3D, bilinear, mipmap generation (box filter).

### Compute APIs
- cuBLAS L1/L2/L3: real cache-blocked GEMM, CBLAS delegation.
- cuFFT: Cooley-Tukey radix-2 + Bluestein for all sizes; FFTW3 optional.
- cuDNN: convolution, batchnorm, activation, pooling, softmax, dropout, attention, CTC loss, LRN — all real CPU implementations with OpenMP.
- cuSPARSE: SpMV/SpMM (CSR), ILU0/IC0, triangular solve.
- cuSolver: LU, QR, SVD, least-squares — LAPACK-backed or built-in.
- NCCL: AllReduce (3 algorithms), Broadcast, ReduceScatter.
- cuRAND: host-side (all generators + distributions) and device-side (XORWOW, Philox — added 2026-05-19).

### Cluster / Transport
- TCP cluster: peer discovery (UDP + proactive), full mesh, HMAC-SHA256 auth.
- Ring all-reduce, RDMA delta-sync (requires InfiniBand), TCP fallback.
- AES-256-GCM secure channel (PBKDF2 600K iterations).
- WebSocket (RFC 6455), gRPC (optional).
- GPU passthrough: real NVRTC + CUDA execution if NVIDIA GPU present.

---

## What Returns Wrong Results (Silent Failures)

**These are the most dangerous — they compile, link, and produce output, but the output is incorrect.**

| API | Symptom |
|---|---|
| `cudnnRNNForwardInference` with `CUDNN_LSTM` | Returns tanh-RNN output; cell state ignored. LSTM networks produce garbage. |
| `cudnnRNNForwardInference` with `CUDNN_GRU` | Returns tanh-RNN output; no gating. GRU networks produce garbage. |
| `cuOccupancyMaxActiveBlocksPerMultiprocessor` | Returns heuristic based on fixed 2048 threads/SM. Auto-tuned block sizes may be suboptimal. |
| `cudnnFindConvolutionForwardAlgorithm` returning WINOGRAD | Falls back silently to GEMM. Output is correct but performance contract is violated. |

---

## What Returns NOT_SUPPORTED (Hard Failures)

| API | Notes |
|---|---|
| `cusparseSpGEMM_compute` | Requires UMFPACK. Graph neural networks fail. |
| `curandStateMtgp32` in-kernel | Header not implemented. Kernels using MTGP32 fail to JIT-compile. |
| `cudnnBackendExecute` | cuDNN v8 backend (PyTorch ≥ 2.0). |
| cuSolver batched APIs | `cusolverDnSgetrfBatched`, etc. |
| macOS Keychain, Linux libsecret, Windows DPAPI | Token manager backends fall back to encrypted file. |
| `cuModuleGetFunction` on SASS-only cubins | Only PTX-embedded binaries supported. |

---

## What Is Missing Entirely

| Missing Feature | Impact |
|---|---|
| PTX multi-module symbol linking | Separate-compilation CUDA workflows fail |
| CUDA TMA (Tensor Memory Accelerator) PTX | Hopper TMA kernels fail to JIT-compile |
| cuDNN RNN backward (BPTT) | RNN training broken |
| Multi-GPU P2P memory | Frameworks using `cudaMemcpyPeer` get slow host-staged copies |
| CUPTI / hardware performance counters | No profiling capability |
| MPS (multi-process service) | Only one process per virtual device |
| SASS disassembler | Pre-compiled library binaries cannot run |
| cuFFT CUDA_C_16BF | Bfloat16 complex FFT absent |

---

## Test Coverage Truthful Assessment

117 tests pass. They cover:
- Core memory and stream semantics ✓
- Kernel JIT for standard patterns ✓
- TCP cluster + security ✓
- cuBLAS, cuFFT, cuSPARSE, cuSolver ✓ (unbatched)
- cuDNN convolution, BN, activation ✓
- cuRAND host-side ✓

They do **NOT** cover:
- LSTM/GRU training — would expose Section 2 wrong-result bugs
- Cross-module PTX linking
- cuDNN v8 backend execution
- MTGP32 device-side cuRAND
- cuSolver batched
- TMA instruction execution
- Multi-process / MPS scenarios

**Passing 117/117 does not mean these gaps are fixed. It means these gaps are not tested.**

---

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
cd tests && ctest --output-on-failure -j$(nproc)
```

Expected output: `100% tests passed, 0 tests failed out of 117`
Expected runtime: ~64 seconds with -j$(nproc) on a modern CPU.
