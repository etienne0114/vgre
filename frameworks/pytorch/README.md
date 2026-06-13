# VGRE PyTorch backend (§4.2)

Runs unmodified PyTorch programs on VGRE's CPU-based CUDA emulation via a
PyTorch out-of-tree **PrivateUse1** backend named `vgre`:

- tensor storage is allocated by VGRE's emulated `cudaMalloc`,
- host↔device copies go through `cudaMemcpy`,
- **`mm` (matmul) is routed through VGRE's emulated cuBLAS** (`cublasSgemm`),
- every other ATen op is covered by a CPU fallback (transfer → run → copy back).

## Build

Requires `libvgre_cudart.so` (build the project first) and PyTorch installed.

```bash
# from frameworks/pytorch/
VGRE_BUILD_DIR=/path/to/virtual-gpu-runtime/build python setup.py build_ext --inplace
```

## Use

```python
import torch, vgre_torch        # registers the "vgre" device
x = torch.randn(4, 8).to("vgre")
W = torch.randn(8, 3).to("vgre")
y = torch.relu(x @ W)           # x@W runs on VGRE's emulated cuBLAS
print(y.cpu())
```

## Notes / scope

- `float32` matmul goes through emulated cuBLAS; other dtypes/ops use the CPU
  fallback (correct, but staged through host memory).
- The symbols are resolved from `libvgre_cudart.so` via `dlsym` so a CUDA-enabled
  PyTorch build's real `cudaMalloc` does not interpose.
- TensorFlow / JAX integration would instead require a PJRT/XLA backend (an HLO
  compiler), a substantially larger effort — not provided here.
