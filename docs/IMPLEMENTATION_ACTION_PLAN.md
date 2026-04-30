# Phase 2 Implementation Action Plan

**Date**: 2026-04-30
**Goal**: Expand VGRE capability by adding advanced CUDA hardware support and closing high-impact operational gaps.

> **Status**: **Phase 1 Complete**. All original 16 tasks (stabilization, zero-simulation verification, and cross-platform completion) are 100% complete as of 2026-04-22. This document now tracks **Phase 2**.

## Phase 2 Features (Missing API/Architecture Support)

The following 5 features have been identified through deep codebase analysis as essential missing capabilities required for advanced compute scenarios:

### 1. Warp-Level Intrinsics (`__shfl`, `__ballot`)
*   **Gap**: Modern CUDA algorithms (like reductions and AI sorting) use warp shuffles (`__shfl_sync`, `__popc`, `__ballot_sync`) to share data without hitting shared memory. These are completely missing from `cpu_cuda_env.h`.
*   **Implementation Strategy**: Map warp intrinsics to CPU SIMD lane shuffling instructions (e.g., AVX2 `_mm256_permutevar8x32_ps`) or provide software fallbacks via shared memory arrays.

### 2. GPU Passthrough for Cluster Workers
*   **Gap**: The VGRE TCP cluster currently pools CPU resources but ignores physical GPUs on worker nodes (it JIT-compiles everything to CPU code).
*   **Implementation Strategy**: Create a `vgre_gpu_worker` role. Have it detect physical NVIDIA GPUs, dynamically load the real `libcuda.so` runtime, and forward the execution directly to the hardware instead of the LLVM compiler.

### 3. Native Deep Learning Library Shims (cuBLAS, cuDNN)
*   **Gap**: VGRE handles custom kernel compilation well, but AI frameworks (PyTorch/TensorFlow) rely heavily on pre-compiled libraries. Calling `cublasSgemm` directly will fail.
*   **Implementation Strategy**: Build library shims that intercept `cublas*`, `cufft*`, and `cudnn*` calls and route them to highly optimized native CPU equivalents (e.g., OpenBLAS, Intel MKL, or FFTW).

### 4. Full FP16 (`__half`) Hardware Support
*   **Gap**: `bfloat16` is supported via SIMD mapping, but the standard `__half` (FP16) type lacks native mapping.
*   **Implementation Strategy**: Map the `__half` data type to the C11 `_Float16` type, utilizing AVX-512 FP16 hardware instructions where available, and software emulation on older CPUs.

### 5. Hardware AES-NI Acceleration Integration
*   **Gap**: A software AES-256-CTR implementation is currently used for cluster channel encryption. AES-NI intrinsics are wired in code but not fully integrated or defaulted.
*   **Implementation Strategy**: Ensure `__builtin_ia32_aesenc128` (AES-NI) is the default path when `-maes` is detected, drastically reducing CPU overhead during large memory transfers across the cluster.

### 6. CUDA Dynamic Parallelism (CDP)
*   **Gap**: Zero support for `cudaGetParameterBuffer` or device-side kernel launches. VGRE cannot execute kernels that spawn other kernels.
*   **Implementation Strategy**: Implement device-side enqueueing and a background CPU dispatcher thread that consumes dynamically spawned kernel grids.

### 7. Tensor Core Emulation (WMMA)
*   **Gap**: Zero support for the `nvcuda::wmma` namespace. Modern AI models (like LLaMA) use Warp Matrix Multiply Accumulate instructions for massive mixed-precision speedups.
*   **Implementation Strategy**: Map `nvcuda::wmma` APIs to CPU-side Advanced Matrix Extensions (AMX) or AVX-512 BFLOAT16 vectorized block operations.

### 8. CUDA IPC API Routing
*   **Gap**: Missing interception for `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`, and `cudaIpcGetEventHandle`. PyTorch DDP relies on these for multi-process distributed training.
*   **Implementation Strategy**: Intercept `cudaIpc*` APIs and map them directly into VGRE's existing POSIX/Win32 shared memory manager.

### 9. Inline PTX Assembly Translator
*   **Gap**: Highly optimized kernels often bypass C++ and write raw inline PTX assembly (`asm("...")`). VGRE's JIT compiler currently fails to parse these blocks.
*   **Implementation Strategy**: Build a basic PTX-to-LLVM-IR translation pass inside the Clang AST parser for the most common inline assembly operations found in AI libraries.

---

## Tracking

- `[ ]` 1. Warp-level intrinsics (`__shfl`)
- `[ ]` 2. GPU passthrough worker
- `[ ]` 3. Native library shims (cuBLAS/cuDNN)
- `[ ]` 4. Full FP16 (`__half`) support
- `[ ]` 5. AES-NI optimization default
- `[ ]` 6. CUDA Dynamic Parallelism (CDP)
- `[ ]` 7. Tensor Core Emulation (WMMA)
- `[ ]` 8. CUDA IPC API routing
- `[ ]` 9. Inline PTX Assembly translator
