# VGRE API Reference

This document outlines the public interface for the Virtual GPU Runtime Engine (VGRE). It covers both the high-level Python bindings and the low-level C-API shim used for framework interception.

---

## 1. Python Bindings API

The Python API provides an interface analogous to PyCUDA, designed for direct scripting execution.

### `vgre.Runtime`

The Core manager for initiating devices and profiling code execution.

- **`__init__()`**
  Creates an uninitialized runtime context.
- **`init(device_id: int = 0, enable_profiling: bool = False)`**
  Boots the native C++ VGRE Engine. Allocates the memory manager and OpenMP thread schedulers.
- **`shutdown()`**
  Gracefully destroys the context and releases all resources.
- **`launch(kernel: Kernel, grid_dim: Dim3, block_dim: Dim3, args: list, parallel: bool = True)`**
  Dispatches a kernel for execution on the virtual engine. `args` can include standard `numpy.ndarray` elements. They are automatically translated to `ctypes` data addresses under the hood. 
- **`synchronize()`**
  Blocks the host thread until all currently executing stream kernels have finished running completely.

### `vgre.Kernel`

Represents an executable logic unit dynamically compiled via the JIT engine.

- **`__init__(name: str, function: Callable, source: str = "")`**
  Creates a kernel representation. The `source` argument is parsed by the C++ `KernelParser` and JIT compiled into an LLVM executable closure during `Runtime.launch()`.
- **Factory Functions**:
  - `vector_add_kernel()`
  - `vector_mul_kernel()`
  - `vector_scale_kernel()`

### `vgre.VirtualDevice`

Represents the emulated generic compute instance.

- **`get_properties() -> DeviceProperties`**
  Returns maximum threads per block, core counts, and VRAM memory properties detected from the host hardware.

### `vgre.Dim3`
CUDA-style multi-dimensional Grid and Block wrapper logic.
- `Dim3(x: int, y: int = 1, z: int = 1)`

---

## 2. C-API Interface Header (`vgre_c_api.h`)

This is the exported interface loaded dynamically by `ctypes`, or utilized by native C/C++ applications intercepting `libvgre.so`. All methods return an integer status condition (`0` represents `VGRE_SUCCESS`).

### Initialization
- **`int vgre_init(void)`**
- **`int vgre_shutdown(void)`**

### Memory Managment
- **`int vgre_malloc(void **ptr, size_t size)`**
  Allocates unmanaged VRAM memory on the virtual device scope.
- **`int vgre_free(void *ptr)`**
- **`int vgre_memcpy(void *dst, const void *src, size_t count, int direction)`**
  Emulates `cudaMemcpy`. Supports mapping between Host vectors, and the Virtual Device memory pool.
- **`int vgre_memset(void *ptr, int value, size_t count)`**

### Execution Dispatch
- **`int vgre_register_kernel(const char *name, const char *source, uint64_t *out_kernel_id)`**
  Submits a C-style string to the LLVM Engine for runtime compilation. Returns an integer `KernelID`.
- **`int vgre_launch_kernel(uint64_t kernel_id, const uint32_t grid_dim[3], const uint32_t block_dim[3], void **args, int num_args, size_t shared_mem, uint64_t stream_id)`**
  Schedules an asynchronous execute routine on the OpenMP thread pool utilizing the specified KernelID mappings. 

### Synchronization
- **`int vgre_stream_create(uint64_t *out_stream_id)`**
- **`int vgre_stream_synchronize(uint64_t stream_id)`**
- **`int vgre_stream_destroy(uint64_t stream_id)`**
- **`int vgre_synchronize(void)`**
  A global blocking barrier resolving all pending enqueued kernels.

---

## 3. OpenCL Compatibility Layer

VGRE provides a minimal OpenCL 1.2 compatible facade for applications using `libOpenCL.so`.

### Platform & Device
- **`clGetPlatformIDs`**: Returns the `VGRE Virtual Platform`.
- **`clGetDeviceIDs`**: Returns the `VGRE Virtual Device`.
- **Robust IDs**: Platform and Device IDs are generated based on a machine-specific hash to ensure stability across restarts.

### Context & Queues
- **`clCreateContext`**: Maps an OpenCL context to a VGRE device.
- **`clCreateCommandQueue`**: Maps an OpenCL queue to a VGRE asynchronous stream.

### Kernel Dispatch
- **`clEnqueueNDRangeKernel`**: Automatically translates OpenCL C source to VGRE Kernel IR and dispatches via the LLVM JIT engine.
