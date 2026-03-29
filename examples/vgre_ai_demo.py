#!/usr/bin/env python3
"""
VGRE AI Demo — Matrix Multiplication (GEMM)
Demonstrates running a core AI kernel on CPU via VGRE's CUDA emulation.
"""

import numpy as np
import os
import sys
import time

# Add project root to sys.path to find vgre package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "bindings", "python"))

import vgre
from vgre.kernel import Kernel, Dim3
from vgre.runtime import Runtime

def main():
    print("VGRE AI Module Demo: Matrix Multiplication")
    print("=" * 45)

    # 1. Initialize VGRE Runtime
    rt = Runtime()
    try:
        # We need to tell the native loader where the library is
        # Since we just deployed it to ~/.local/share/VGRE
        os.environ["VGRE_LIB_PATH"] = os.path.expanduser("~/.local/share/VGRE/lib/libvgre.so")
        rt.init(enable_profiling=True)
    except Exception as e:
        print(f"Error initializing VGRE: {e}")
        return

    # 2. Define GEMM Kernel (C = A * B)
    # This is a naive GEMM kernel. In real AI frameworks, these are highly optimized.
    gemm_source = """
    extern "C" __global__ void gemm(float* A, float* B, float* C, int N) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row < N && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    """
    kernel = Kernel("gemm", gemm_source)

    # 3. Prepare Data (N x N matrices)
    N = 256  # Size 256x256
    print(f"Preparing {N}x{N} matrices...")
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    C_vgre = np.zeros((N, N), dtype=np.float32)

    # 4. Launch Kernel
    print("Launching GEMM kernel on VGRE (CPU-based CUDA)...")
    block_size = 16
    grid_size = (N + block_size - 1) // block_size
    
    grid_dim = Dim3(grid_size, grid_size)
    block_dim = Dim3(block_size, block_size)

    # Synchronous launch for verification
    elapsed_ms = rt.launch(kernel, grid_dim, block_dim, [A, B, C_vgre, N], parallel=False)
    
    # 5. Verify with NumPy
    print(f"Kernel execution took: {elapsed_ms:.2f} ms")
    
    print("Verifying results with NumPy...")
    start_cpu = time.perf_counter()
    C_numpy = np.dot(A, B)
    end_cpu = time.perf_counter()
    print(f"NumPy execution took: {(end_cpu - start_cpu)*1000:.2f} ms")

    # Check accuracy
    diff = np.abs(C_vgre - C_numpy).max()
    if diff < 1e-3:
        print(f"SUCCESS: Results match (Max diff: {diff:.6f})")
    else:
        print(f"FAILURE: Results mismatch (Max diff: {diff:.6f})")

    # 6. Performance Metrics
    total_ops = 2 * (N ** 3)  # Multiply-adds
    gflops = (total_ops / 1e9) / (elapsed_ms / 1000.0)
    print(f"Performance: {gflops:.3f} GFLOPS")

    print("\n" + rt.get_profiling_report().summary())
    
    rt.shutdown()

if __name__ == "__main__":
    main()
