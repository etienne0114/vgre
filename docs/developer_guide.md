# VGRE Developer Guide

Welcome to the Virtual GPU Runtime Engine (VGRE) Developer Guide. This document provides a deep dive into the internal architecture, subsystems, and execution model of VGRE. It is intended for developers who want to extend VGRE, add new backends, or understand how virtual CUDA execution works under the hood.

## Architecture Overview

VGRE is designed to intercept CUDA-like compute requests from high-level frameworks (via Python bindings or `LD_PRELOAD` C-API shims) and transparently execute them on the host CPU. It achieves near-native performance by leveraging Multithreading, JIT compilation (via LLVM), and highly optimized memory bridging.

The system is composed of four primary layers:
1. **API Layer (`bindings/python`, `src/api`)**: Exposes the `vgre.Runtime` for Python integration and `libvgre_cudart.so` for binary interception of compiled CUDA applications.
2. **Runtime Engine (`src/core/runtime_engine.cpp`)**: The central coordinator that manages devices, registers kernels, and handles stream-based asynchronous execution dispatching.
3. **Execution Engine (`src/core/scheduler.cpp`, `src/runtime/cpu_parallel_executor.cpp`)**: The OpenMP-backed multithreaded task dispatcher that schedules and executes grid/block dimensions across available CPU cores authoritatively.
4. **LLVM JIT Compiler (`src/compiler/llvm_translation_engine.cpp`)**: Dynamically translates virtual kernel operations or parsed Python kernel source into executable machine code at runtime.

---

## 1. API and Interception Layer

### C-API (`libvgre.so`)
The `vgre_c_api.cpp` provides a flat extern "C" struct mapping. This is the sole boundary between external runtimes (like ctypes in Python) and the internal C++ system. It exposes familiar CUDA-like primitives:
- `vgre_malloc`
- `vgre_memcpy`
- `vgre_launch_kernel`
- `vgre_stream_synchronize`

### Python Bindings
The Python `kernel.py` and `runtime.py` modules use `ctypes` to map standard NumPy arrays into the C-API. **Crucial detail:** Memory pointers passed to `vgre_launch_kernel` via `ctypes` *must* be rigorously deep-copied inside the C++ runtime to prevent Python's garbage collector from invalidating the memory before background threads execute the kernel. 

---

## 2. The Runtime Engine

The `RuntimeEngine` is a singleton that orchestrates the entire application lifecycle. 

### Kernel Registration
When a kernel is registered via `RuntimeEngine::registerKernel()`, it passes through the `KernelParser`. VGRE supports dynamically parsing C-style kernel strings or pre-compiled `.bc` LLVM Bitcode Modules. The engine assigns a unique `KernelId` and caches the interpreted representation.

### Asynchronous Dispatch
When `RuntimeEngine::launchKernel()` is invoked, it:
1. Validates the `KernelId`.
2. Explicitly deep-copies the argument pointers passed from the API layer into `std::shared_ptr<std::vector<void*>>` thread-safe containers. This guarantees the pointers remain valid across the async boundary.
3. Submits a closure to the `Scheduler` specifying the Stream ID and the target `CPUParallelExecutor`.

---

## 3. The Execution Subsystem

VGRE's execution architecture is deeply multithreaded to replicate GPU concurrency on standard host CPUs.

### The Scheduler (`scheduler.cpp`)
The Scheduler manages multiple independent CUDA streams. Each stream maintains its own queue of pending tasks. The Scheduler uses a persistent pool of `std::thread` workers (scaled to the CPU's hardware concurrency limit). 
When a thread picks up a stream task, it evaluates the dependencies and unblocks the execution queue serially for that specific stream.

### OpenMP CPU Parallel Executor (`cpu_parallel_executor.cpp`)
Once a kernel reaches the front of the scheduler queue, it is handed off to the `CPUParallelExecutor`. 
Instead of serializing execution, this executor maps the virtual CUDA `(GridDim.x * GridDim.y * GridDim.z)` dimensionality into a flat loop. 
It then utilizes `#pragma omp parallel for schedule(dynamic)` to scatter these block executions across all available CPU sockets instantly, mimicking massive parallel SM dispatch.

---

## 4. LLVM Translation Engine (JIT)

The `LLVMTranslationEngine` bridges the gap between parsed kernel source and machine instructions.

### Dynamic Compilation Flow
1. **Dynamic JIT Compilation**: All kernels are dynamically compiled through the LLVM ORC JIT pipeline with host-native CPU feature detection (AVX2/AVX-512/FMA) and aggressive O3 optimization.
2. **LLVM IR Generation**: If dynamic JIT is enabled, the Engine explicitly builds LLVM IR strings mapping the `blockIdx`, `gridDim`, and `args[]` arrays.
3. **ORC JIT**: The IR is parsed into an `llvm::Module` and pushed into an LLVM ORC JIT Engine (`llvmState_->jit`). The engine returns an executable function pointer (`CompiledKernelFn`).

### Pointer Casting Rules
The translated JIT function is always assumed to have the signature:
```cpp
void kernel_inner(void** args, uint32_t bx, uint32_t by, uint32_t bz, ...)
```
The translation engine is responsible for unpacking the generic `void** args` array and `bitcast`ing each index to the respective floating-point or integer pointer required by the destination data block.

---

## 5. Memory Management and UVM

The `MemoryManager` creates an emulated `std::malloc` pool mimicking VRAM. 
- **Pool Size**: Default is 4GB (managed via `MAX_MANAGED_REGIONS = 4096`).
- **UVM Implementation**: VGRE fully supports **Unified Virtual Memory (UVM)**. If managed memory is requested (`cudaMallocManaged`), VGRE utilizes POSIX `mmap()` to allocate host pages and sets up a global `SIGSEGV` signal handler.
- **Page Faulting**: When the CPU or a kernel accesses a managed page that isn't resident, the signal handler intercepts the violation and triggers a "Page Fault" event, migrating the page from the managed pool to host residence.

---

## 6. Performance Authority & Profiling

### Authoritative Calibration
VGRE does not "guess" performance. The `AdaptiveExecutionEngine` performs ground-truth calibration by running micro-benchmarks on the host CPU at startup. This determines the peak GFLOPS and memory bandwidth possible on the specific hardware, which are then used as the 100% baseline for telemetry.

### Runtime Profiling
The `RuntimeProfiler` records every kernel launch, memory copy, and synchronization event. 
- **Event Recording**: Uses a lock-free buffer to avoid profiling overhead.
- **Trace Export**: Supports exporting data to the Chrome Trace format (`chrome://tracing`). This allows developers to visualize overlapping streams and kernel execution timelines with microsecond precision.

## Conclusion
Extending VGRE typically involves modifying the `LLVMTranslationEngine` for new instructions or the `AdaptiveExecutionEngine` for more granular hardware sensing. The system is designed to be highly modular, with clear boundaries between the JIT compiler, the memory pool, and the OpenMP parallel executor.

---

## Approved Future Innovations Roadmap

As part of transforming VGRE into an extraordinary, sophisticated, real-functioning tool, the following enhancements are upcoming:

1. **Full Texture & Surface API Implementation**: Complete hardware-backed realization of `cudaArray_t` APIs.
2. **Dynamic JIT Kernel Fusion**: A runtime pass to fuse consecutive PTX kernels into a single LLVM IR module, drastically cutting launch overhead.
3. **Interactive 3D Hardware Topology Viewer**: A Flutter frontend update featuring a fully interactive 3D representation of cluster and Node PCIe/Memory topology.
4. **Agentic "AI Tuner" Interface**: An embedded AI chat UI in the dashboard enabling real-time, natural-language tuning commands routed securely to the backend.
