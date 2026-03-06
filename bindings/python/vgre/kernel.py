"""
VGRE Kernel — Python kernel definition and execution helpers.

Provides a pythonic API for defining GPU-like kernels that execute
on CPU via the VGRE engine.
"""

import numpy as np
from typing import Callable, List, Optional, Tuple
import multiprocessing
import time


class Dim3:
    """3D dimension descriptor (mirrors CUDA dim3)."""

    def __init__(self, x: int = 1, y: int = 1, z: int = 1):
        self.x = x
        self.y = y
        self.z = z

    def total(self) -> int:
        return self.x * self.y * self.z

    def __repr__(self) -> str:
        return f"Dim3({self.x}, {self.y}, {self.z})"


class Kernel:
    """
    Represents a GPU-like kernel that can be executed on CPU.

    Usage:
        def vector_add(a, b, c, block_idx, block_dim, grid_dim):
            start = block_idx.x * block_dim.x
            end = min(start + block_dim.x, len(a))
            for i in range(start, end):
                c[i] = a[i] + b[i]

        k = Kernel("vector_add", vector_add)
        k.launch(grid_dim=Dim3(4), block_dim=Dim3(256), args=[a, b, c])
    """

    def __init__(self, name: str, source: str = ""):
        self.name = name
        self.source = source
        self._kernel_id = None
        self._launch_count = 0
        self._total_time_ms = 0.0

    @property
    def launch_count(self) -> int:
        return self._launch_count

    @property
    def avg_time_ms(self) -> float:
        if self._launch_count == 0:
            return 0.0
        return self._total_time_ms / self._launch_count

    def __repr__(self) -> str:
        return (f"Kernel(name='{self.name}', "
                f"launches={self._launch_count}, "
                f"avg_time={self.avg_time_ms:.2f}ms)")


# Convenience factory functions
def vector_add_kernel() -> Kernel:
    source = """
    extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            c[i] = a[i] + b[i];
        }
    }
    """
    return Kernel("vector_add", source)

def vector_mul_kernel() -> Kernel:
    source = """
    extern "C" __global__ void vector_mul(float* a, float* b, float* c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            c[i] = a[i] * b[i];
        }
    }
    """
    return Kernel("vector_mul", source)

def vector_scale_kernel() -> Kernel:
    source = """
    extern "C" __global__ void vector_scale(float* a, float scalar, float* c, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            c[i] = a[i] * scalar;
        }
    }
    """
    return Kernel("vector_scale", source)
