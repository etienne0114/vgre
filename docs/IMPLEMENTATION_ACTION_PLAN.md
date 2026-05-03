# Phase 2 Implementation Action Plan

**Date**: 2026-05-03
**Goal**: Expand VGRE capability by adding advanced CUDA hardware support and closing high-impact operational gaps.

> **Status**: **Phase 2 Complete + Hardened**. All 9 Phase 2 features implemented and production-hardened. 64/64 tests passing.
> **Last hardening pass (2026-05-03)**: cuBLAS transpose bug fixed; cuDNN Winograd conditions corrected (dilation+alignment); cuDNN pooling implemented (MaxPool/AvgPool); warp shuffle mask respected; 100+ PTX opcodes; OpenCL warp shuffle wg-size-safe; singleton leaks fixed; iGPU GFLOPS from real OpenCL device query.

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
*   **Implementation Strategy**: Build a advanced PTX-to-LLVM-IR translation pass inside the Clang AST parser for the most common inline assembly operations found in AI libraries.

---

## Tracking

- `[x]` 1. Warp-level intrinsics (`__shfl`) — `include/vgre/compiler/cpu_cuda_warp.h`; warp exchange buffer in JIT wrapper; auto-parallel for warp-shuffle kernels
- `[x]` 2. GPU passthrough worker — `src/advanced/gpu_passthrough.cpp`; libcuda.so + NVRTC dlopen; integrated into dispatch_manager before CPU fallback
- `[x]` 3. Native library shims (cuBLAS/cuDNN) — `src/api/cublas_shim.cpp` + `cudnn_shim.cpp`; reference GEMM + optional OpenBLAS; conv/BN/softmax/activation
- `[x]` 4. Full FP16 (`__half`) support — `include/vgre/compiler/cpu_cuda_fp16.h`; IEEE-754 FP16 encode/decode; full operator set + `__half2`
- `[x]` 5. AES-NI runtime detection — `src/advanced/secure_channel_crypto.cpp`; `__builtin_cpu_supports("aes")` guard before HW path
- `[x]` 6. CUDA Dynamic Parallelism (CDP) — `src/runtime/cdp_executor.cpp`; `cudaGetParameterBuffer` + `cudaLaunchDevice` in `cpu_cuda_env.h`; drain after each block
- `[x]` 7. Tensor Core Emulation (WMMA) — `include/vgre/compiler/wmma_emulation.h`; `nvcuda::wmma` namespace; scalar FP32 fallback for all tile sizes
- `[x]` 8. CUDA IPC API routing — `src/api/cuda_ipc_memory.cpp`; POSIX SHM-backed `cudaIpcGetMemHandle`/`OpenMemHandle`/`CloseMemHandle`
- `[x]` 9. Inline PTX Assembly translator — `src/compiler/ptx_translator.cpp`; 100+ opcodes: FP64, i64, shared/local/global vectorized mem, atomics, vote, shuffle, setp, selp, cvt, FP intrinsics; called before JIT compilation

## Implementation Files

| Feature | New Files | Modified Files |
|---------|-----------|----------------|
| Warp intrinsics | `include/vgre/compiler/cpu_cuda_warp.h` | `llvm_translation_engine.cpp`, `llvm_translation_codegen.cpp`, `kernel_parser.cpp`, `cpu_cuda_env.h`, `types.h` |
| GPU passthrough | `src/advanced/gpu_passthrough.cpp`, `include/vgre/advanced/gpu_passthrough.h` | `dispatch_manager.cpp`, `advanced/CMakeLists.txt` |
| cuBLAS/cuDNN | `src/api/cublas_shim.cpp`, `src/api/cudnn_shim.cpp` | `src/api/CMakeLists.txt` |
| FP16 | `include/vgre/compiler/cpu_cuda_fp16.h` | `cpu_cuda_env.h` |
| AES-NI | — | `src/advanced/secure_channel_crypto.cpp` |
| CDP | `src/runtime/cdp_executor.cpp`, `include/vgre/runtime/cdp_executor.h` | `cpu_cuda_env.h`, `llvm_translation_engine.cpp`, `cpu_parallel_executor.cpp`, `runtime/CMakeLists.txt` |
| WMMA | `include/vgre/compiler/wmma_emulation.h` | `cpu_cuda_env.h` |
| CUDA IPC | `src/api/cuda_ipc_memory.cpp` | `src/api/CMakeLists.txt` |
| PTX | `src/compiler/ptx_translator.cpp`, `include/vgre/compiler/ptx_translator.h` | `llvm_translation_codegen.cpp`, `src/compiler/CMakeLists.txt` |
