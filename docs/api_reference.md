# VGRE API Reference

This document outlines the public interface for the Virtual GPU Runtime Engine (VGRE). It covers the Python bindings, the C API (`vgre_c_api.h`), the dashboard telemetry schema, and the OpenCL compatibility layer.

---

## 1. Python Bindings API

The Python API provides an interface analogous to PyCUDA, designed for direct scripting execution.

### `vgre.Runtime`

The core manager for initiating devices and profiling code execution.

- **`__init__()`** — Creates an uninitialized runtime context.
- **`init(device_id: int = 0, enable_profiling: bool = False)`** — Boots the native C++ VGRE Engine. Allocates the memory manager and OpenMP thread schedulers.
- **`shutdown()`** — Gracefully destroys the context and releases all resources.
- **`launch(kernel: Kernel, grid_dim: Dim3, block_dim: Dim3, args: list, parallel: bool = True)`** — Dispatches a kernel for execution. `args` may include `numpy.ndarray` elements; they are automatically translated to `ctypes` data addresses.
- **`synchronize()`** — Blocks the host thread until all currently executing stream kernels have finished.

### `vgre.Kernel`

Represents an executable logic unit dynamically compiled via the JIT engine.

- **`__init__(name: str, function: Callable, source: str = "")`** — Creates a kernel representation. The `source` argument is parsed by `ClangKernelParser` and JIT-compiled to native code during `Runtime.launch()`.
- **Factory Functions**: `vector_add_kernel()`, `vector_mul_kernel()`, `vector_scale_kernel()`

### `vgre.VirtualDevice`

Represents the emulated compute instance.

- **`get_properties() -> DeviceProperties`** — Returns max threads, core count, and VRAM memory properties detected from the host hardware.

### `vgre.Dim3`
CUDA-style multi-dimensional grid and block wrapper.
- `Dim3(x: int, y: int = 1, z: int = 1)`

---

## 2. C API Interface (`vgre_c_api.h`)

Exported C interface loaded dynamically by `ctypes` or used by native C/C++ applications. All functions return `int` (0 = `VGRE_SUCCESS`).

### Initialization
```c
int vgre_init(void);
int vgre_shutdown(void);
```

### Device Management
```c
int vgre_get_device_count(int *count);
int vgre_set_device(int device_id);
int vgre_get_device(int *device_id);
int vgre_get_device_properties(int device_id, vgre_device_properties_t *props);
int vgre_synchronize(void);  // global barrier: wait for all enqueued kernels
```

`vgre_device_properties_t` fields: `name[256]`, `total_memory`, `shared_mem_per_block`, `max_threads_per_block`, `warp_size`, `multiprocessor_count`, `compute_capability_major/minor`, `pci_bus_id/device_id`.

### Memory Management
```c
int vgre_malloc(void **ptr, size_t size);
    // Allocates device memory (no page protection; like cudaMalloc)

int vgre_malloc_managed(void **ptr, size_t size);
    // Allocates UVM managed memory (mmap PROT_NONE + SIGSEGV handler)

int vgre_free(void *ptr);

int vgre_memcpy(void *dst, const void *src, size_t count, int direction);
    // direction: 0=H2D, 1=D2H, 2=D2D, 3=H2H

int vgre_memset(void *ptr, int value, size_t count);
```

### P2P Peer Device Access
```c
int vgre_device_can_access_peer(int *can_access, int device, int peer_device);
int vgre_device_enable_peer_access(int peer_device);
int vgre_device_disable_peer_access(int peer_device);
```

### Kernel Execution
```c
int vgre_register_kernel(const char *name, const char *source,
                          uint64_t *out_kernel_id);
    // Submits source to Clang+LLVM JIT pipeline; returns KernelId

int vgre_launch_kernel(uint64_t kernel_id,
                        const uint32_t grid_dim[3],
                        const uint32_t block_dim[3],
                        void **args, int num_args,
                        size_t shared_mem,
                        uint64_t stream_id);
    // Asynchronous dispatch; stream_id=0 for default stream
```

### Stream Management
```c
int vgre_stream_create(uint64_t *out_stream_id);
int vgre_stream_create_with_priority(uint64_t *out_stream_id, int priority);
    // priority: higher value = higher scheduling priority
int vgre_stream_synchronize(uint64_t stream_id);
int vgre_stream_destroy(uint64_t stream_id);
```

### CUDA Graphs
```c
int vgre_graphCreate(uint64_t *out_graph);
int vgre_graphClone(uint64_t src_graph, uint64_t *out_clone);
    // Deep-copies src_graph into a new independent graph (all nodes + deps).
int vgre_graphDestroy(uint64_t graph);
int vgre_graphInstantiate(uint64_t graph, uint64_t *out_exec);
int vgre_graphExecDestroy(uint64_t exec);
int vgre_graphLaunch(uint64_t exec, uint64_t stream);

int vgre_graphAddKernelNodeEx(uint64_t graph, uint64_t kernel_id,
                               const char *name,
                               const uint32_t grid_dim[3],
                               const uint32_t block_dim[3],
                               void **args, const uint8_t *arg_types, int num_args,
                               const uint64_t *deps, int num_deps,
                               uint64_t *out_node_id);
    // arg_types: array of ArgType (0=POINTER, 1=INT32, 2=INT64, 3=FLOAT32, ...)

int vgre_graphAddMemcpyNodeEx(uint64_t graph, void *dst, void *src,
                               size_t count, int kind,
                               const uint64_t *deps, int num_deps,
                               uint64_t *out_node_id);

int vgre_graphAddDependency(uint64_t graph, uint64_t node_id,
                             uint64_t depends_on);
int vgre_graphUpdateKernelNode(uint64_t graph, uint64_t node_id,
                                void **args, const uint8_t *arg_types, int num_args);
int vgre_graphUpdateMemcpyNode(uint64_t graph, uint64_t node_id,
                                void *dst, void *src, size_t count, int kind);
```

## 2.5 CUDA Runtime API Shim (`libvgre_cudart.so`)

VGRE provides a comprehensive binary-compatible layer for application interception. Supported functions include:

### Device & Context
- `cudaGetDeviceCount`, `cudaSetDevice`, `cudaGetDevice`
- `cudaGetDeviceProperties`, `cudaDeviceSynchronize`, `cudaDeviceReset`
- `cudaSetDeviceFlags`, `cudaGetDeviceFlags`
- `cudaDeviceGetPCIBusId`, `cudaDeviceGetByPCIBusId`

### Memory Management
- `cudaMalloc`, `cudaFree`, `cudaMemset`, `cudaMemsetAsync`
- `cudaMemcpy`, `cudaMemcpyAsync`, `cudaMemcpy2D`, `cudaMemcpy2DAsync`
- `cudaMallocPitch`, `cudaMemGetInfo`, `cudaGetSymbolAddress`
- **Managed Memory**: `cudaMallocManaged`, `cudaMemAdvise`, `cudaMemPrefetchAsync`
- **Host Memory**: `cudaHostAlloc`, `cudaFreeHost`, `cudaHostRegister`, `cudaHostUnregister`
- **Async Pools**: `cudaMallocAsync`, `cudaFreeAsync`, `cudaMemPoolCreate`, `cudaMemPoolDestroy`

### Streams & Events
- `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamCreateWithPriority`
- `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamQuery`
- `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaEventSynchronize`
- `cudaEventElapsedTime`, `cudaEventDestroy`

### Advanced Interop
- `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess`, `cudaDeviceDisablePeerAccess`
- `cudaMemcpyPeer`, `cudaMemcpyPeerAsync`
- **CUDA Graphs**: `cudaGraphCreate`, `cudaGraphClone`, `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaGraphInstantiate`, `cudaGraphLaunch`, `cudaGraphDestroy`, `cudaGraphExecDestroy`

---

## 3. Dashboard Telemetry API

### Telemetry & Profiling
```c
int vgre_get_telemetry(vgre_telemetry_t *telemetry);
    // Fields: gflops, memory_bandwidth_gbps, page_fault_rate,
    //         resident_pages, kernel_launches, total_bytes_transferred

int vgre_set_profiler_enabled(int enabled);

int vgre_get_profiler_json(char **out_json, int top_n);
    // Returns Chrome trace JSON; caller must call vgre_free_string(out_json)

int vgre_get_memory_info_json(char **out_json);
    // Returns JSON with pool stats, UVM map, dirty pages

int vgre_get_kernel_history_json(const char *kernel_name, char **out_json);
    // Per-kernel invocation timeline JSON

int vgre_get_logs(char ***buffer, int *count);
    // Returns last 100 log messages; caller must call vgre_free_logs(buffer, count)

const char* vgre_get_version(void);
void vgre_free_string(char *str);
void vgre_free_logs(char **buffer, int count);
```

### Cluster & Node Management
```c
int vgre_get_cluster_nodes(vgre_cluster_node_t *nodes, int *count);
    // Fills nodes[] with up to *count remote worker node descriptors

int vgre_cluster_set_security(int enabled);
    // Enables authenticated encrypted channel (HMAC-SHA256 + AES-256-CTR).
    // Requires VGRE_TCP_AUTH_TOKEN environment variable.
    // NOTE: cipher is AES-256-CTR (software implementation), not AES-256-GCM.

int vgre_cluster_get_security_info(vgre_security_info_t *info);
    // info.cipher_name = "VGRE-HMAC-SHA256-AES256-CTR"
    // info.key_fingerprint = hex(SHA256(session_key))
    // info.packets_sent, packets_received, bytes_sent, bytes_received

int vgre_cluster_wait(uint64_t kernel_id, int timeout_ms);
    // Block until remote kernel result available (timeout_ms=-1 = infinite)

int vgre_set_service_mode(int is_master);
    // 1 = this process is the cluster master; 0 = worker
```

### Credit / Billing System (Compute-Unit-Seconds)
```c
int vgre_credits_get_balance(const char *node_address,
                              vgre_credit_info_t *info);
    // info.credits_earned, credits_consumed, net_balance (all in CUS)

int vgre_credits_get_all(vgre_credit_info_t *nodes, int *count);
    // All nodes in the resource ledger

int vgre_credits_reset(void);
    // Clear the ledger (admin use only)
```

### Configuration
```c
int vgre_set_background_compute(int enabled);
    // Enable/disable background GEMM workload for dashboard GFLOPS demo

int vgre_set_block_threads(int enabled);
    // 0 = serial/OpenMP hybrid; 1 = persistent WorkerPool for syncthreads correctness.
    // This updates the internal engine state.

const char* vgre_get_version(void);
    // Returns version string "0.1.2".
```

---

## 4. Dashboard Telemetry API (polling schema)

The VGRE Dashboard uses an isolate-based background thread to poll the C API every 500 ms. The following telemetry fields are presented:

| Field | Source | Meaning |
|-------|--------|---------|
| `gflops` | AdaptiveExecutionEngine | Ground-truth GFLOPS from LLVM-calibrated instruction counts |
| `memoryBandwidthGbps` | MemoryManager | Measured h2d/d2h/d2d transfer bandwidth |
| `pageFaultRate` | UVM Handler | SIGSEGV-triggered page migrations per second |
| `uvmMap` | MemoryManager | bitset represented as `uint8_t[1024]` residency map |
| `deviceTemperature` | AdaptiveExecutionEngine | Real-time thermal sensor reading (Linux/TPM) or calibrated heuristic |
| `activeKernels` | RuntimeEngine | Number of kernels currently executing in the BlockWorkerPool |
| `kernelLaunches` | RuntimeProfiler | Cumulative kernel invocation count |
| `totalBytesTransferred` | MemoryManager | Cumulative bytes across all memcpy operations |
| `clusterNodeCount` | TCPClusterManager | Number of connected worker nodes |

---

## 4. OpenCL Compatibility Layer

VGRE provides a minimal OpenCL 1.2 compatible facade for applications using `libOpenCL.so`.

### Platform & Device
- **`clGetPlatformIDs`**: Returns the `VGRE Virtual Platform`.
- **`clGetDeviceIDs`**: Returns the `VGRE Virtual Device`.
- **Robust IDs**: Platform and Device IDs are generated from machine-specific entropy for stability across restarts.

### Context & Queues
- **`clCreateContext`**: Maps an OpenCL context to a VGRE device.
- **`clCreateCommandQueue`**: Maps an OpenCL queue to a VGRE asynchronous stream.

### Memory
- **`clCreateBuffer`** / **`clReleaseMemObject`**: Map to `vgre_malloc` / `vgre_free`.
- **`clEnqueueWriteBuffer`** / **`clEnqueueReadBuffer`**: H2D and D2H memcpy.

### Kernel Dispatch
- **`clEnqueueNDRangeKernel`**: Translates OpenCL C source to VGRE KernelIR and dispatches via the LLVM JIT engine.

### Events & Profiling
- **`clWaitForEvents`**, **`clGetEventProfilingInfo`**: Wraps VGRE event timing.
- **`clFinish`**: Equivalent to `vgre_stream_synchronize`.
