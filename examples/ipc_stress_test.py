import sys
import os
import time
import numpy as np

# Ensure we use libvgre from the build directory
# Normally this would be handled by LD_LIBRARY_PATH
vgre_path = os.path.abspath("../build")
sys.path.append(os.path.abspath("../bindings/python"))

import vgre
from vgre.kernel import Kernel, Dim3

# Define a matrix multiply kernel source for registration
matmul_source = """
extern "C" __global__ void matmul(float* A, float* B, float* C, int N) {
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

def run_external_workload():
    print("--- VGRE External Workload (IPC Test) ---")
    try:
        # Initialize the runtime
        runtime = vgre.Runtime()
        runtime.init()

        dev = runtime.get_device()
        props = dev.get_properties()
        print(f"Connecting to VGRE Service on {props.name}")

        # Matrix size: 512x512 (reasonable for CPU emulation)
        N = 512
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
        C = np.zeros((N, N), dtype=np.float32)

        # Create a kernel for matrix multiplication
        kernel = Kernel("matmul", matmul_source)

        # Grid/block dimensions for 2D kernel
        block = Dim3(16, 16)
        grid = Dim3((N + 15) // 16, (N + 15) // 16)

        print("Starting continuous workload loop...")
        print("Switch to the VGRE Dashboard to see aggregated GFLOPS!")

        loop_count = 0
        while True:
            # Launch matrix multiplication via the proper API
            n_val = np.int32(N)
            runtime.launch(kernel, grid, block, [A, B, C, n_val])
            runtime.synchronize()

            loop_count += 1
            if loop_count % 10 == 0:
                print(f"Completed {loop_count} iterations...")

            # small sleep to avoid 100% CPU starvation of the UI
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nWorkload stopped.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        runtime.shutdown()

if __name__ == "__main__":
    run_external_workload()
