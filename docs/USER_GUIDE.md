# VGRE User Guide

Welcome to the Virtual GPU Runtime Engine (VGRE). This guide will help you set up and use VGRE to run CUDA-accelerated applications on non-NVIDIA hardware.

---

## 1. Getting Started

### 1.1 Prerequisites
- **Linux**: Ubuntu 22.04+ (Recommended) or Fedora 38+.
- **Compiler**: LLVM/Clang 16+.
- **Dependencies**: `cmake`, `libssl-dev`, `libtpm2-tss-dev` (optional for hardware auth).

### 1.2 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/vgre-org/vgre-runtime.git
   cd vgre-runtime
   ```
2. Build the project:
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   sudo make install
   ```
   This installs the core libraries to `/usr/local/lib/vgre/` and headers to `/usr/local/include/vgre/`.

---

## 2. Using VGRE with Applications

VGRE supports two primary modes of operation: **Native Integration** and **Framework Interception**.

### 2.1 Framework Interception (PyTorch, TensorFlow, etc.)
To run an existing CUDA application (like a Python script using PyTorch) on VGRE, use `LD_PRELOAD` to redirect CUDA calls:

```bash
export LD_PRELOAD=/usr/local/lib/vgre/libvgre_cudart.so
export VGRE_LOG_LEVEL=INFO
python my_pytorch_script.py
```

### 2.2 Native Python Integration
For custom scripts, use the `vgre` Python package:

```python
import vgre
import numpy as np

# Initialize the VGRE Runtime
rt = vgre.Runtime()
rt.init(enable_profiling=True)

# Launch a kernel
kernel = vgre.Kernel("vector_add", my_source_code)
rt.launch(kernel, grid=(1024, 1, 1), block=(256, 1, 1), args=[a_ptr, b_ptr, c_ptr])

rt.synchronize()
```

---

## 3. VGRE Dashboard & Monitoring

VGRE includes a real-time monitor to visualize compute and memory utilization.

1. **Launch the Dashboard**:
   ```bash
   vgre-dashboard
   ```
2. **Connect**: The dashboard automatically connects to the active VGRE session via IPC.
3. **Features**:
   - **GFLOPS Meter**: Real-time compute performance.
   - **UVM Residency Map**: Visualizes page migrations in managed memory.
   - **3D Topology**: View your virtual device cluster in a 3D orbital space.

---

## 4. Distributed Cluster Setup

VGRE allows you to aggregate multiple network-connected CPUs into a single Virtual GPU cluster.

### 4.1 Start a Worker Node
On a remote machine:
```bash
vgre-worker --port 9090 --auth-token my-secret-key
```

### 4.2 Connect the Master Node
On your local machine, set the environment variable:
```bash
export VGRE_CLUSTER_NODES="192.168.1.50:9090"
export VGRE_TCP_AUTH_TOKEN="my-secret-key"
./my_cuda_app
```

---

## 5. Troubleshooting

| Issue | Potential Cause | Solution |
|---|---|---|
| `cudaErrorNoDevice` | VGRE Driver not found | Ensure `LD_LIBRARY_PATH` includes `/usr/local/lib/vgre/`. |
| `cudaErrorInvalidValue` | Pointer out of bounds | Check your kernel arguments; VGRE performs strict bounds checking. |
| Dashboard not updating | IPC Socket error | Ensure you haven't exceeded the max IPC connections or check `/tmp/vgre_ipc`. |
| Low Performance | JIT Cache miss | The first run is always slower due to Clang AST parsing. Subsequent runs use the cache. |

---

For more details, refer to the [Technical Architecture](architecture.md) or the [API Reference](api_reference.md).
