import sys
import time
import math
import numpy as np

# Add VGRE bindings to path
sys.path.insert(0, "./bindings/python")

try:
    from vgre import VirtualDevice, Runtime
    from vgre.kernel import Dim3, vector_add_kernel, vector_mul_kernel
    VGRE_AVAILABLE = True
except ImportError:
    VGRE_AVAILABLE = False


def run_benchmark(name, func, warmup=5, iters=20):
    # Warmup
    for _ in range(warmup):
        func()
    
    # Measure
    start_time = time.perf_counter()
    for _ in range(iters):
        func()
    end_time = time.perf_counter()
    
    avg_time_ms = ((end_time - start_time) / iters) * 1000.0
    print(f"{name:.<50} {avg_time_ms:7.3f} ms")
    return avg_time_ms


def benchmark_vgre(N):
    if not VGRE_AVAILABLE:
        print("VGRE bindings not found. Skipping VGRE benchmark.")
        return

    print(f"\n========================================")
    print(f"   Benchmark Mode: VGRE Native Engine")
    print(f"========================================")

    dev = VirtualDevice()
    print(f"VGRE Device: {dev.get_properties().name}")

    grid_size = math.ceil(N / 256)
    block_size = 256
    padded_N = grid_size * block_size

    # Use padded size so the JIT kernel doesn't overshoot
    a = np.random.randn(padded_N).astype(np.float32)
    b = np.random.randn(padded_N).astype(np.float32)
    c = np.zeros(padded_N, dtype=np.float32)

    rt = Runtime()
    rt.init(enable_profiling=False)

    kernel = vector_add_kernel()
    grid_size = (N + block_size - 1) // block_size

    def vgre_func():
        rt.launch(kernel, Dim3(grid_size), Dim3(block_size), [a, b, c, N], parallel=True)
        rt.synchronize()

    avg_ms = run_benchmark(f"[VGRE] VectorAdd {N//1000000}M elements", vgre_func)
    gb_s = (3.0 * N * 4) / (avg_ms * 1e-3) / 1e9
    print(f"  -> {gb_s:.2f} GB/s")

    rt.shutdown()


def benchmark_numpy(N):
    print(f"\n========================================")
    print(f"   Benchmark Mode: Native NumPy")
    print(f"========================================")

    A = np.random.randn(N).astype(np.float32)
    B = np.random.randn(N).astype(np.float32)
    
    def func():
        return A + B  # Will be optimized C-execution in numpy
        
    avg_ms = run_benchmark(f"[NumPy] VectorAdd {N//1000000}M elements", func)
    gb_s = (3.0 * N * 4) / (avg_ms * 1e-3) / 1e9 
    print(f"  -> {gb_s:.2f} GB/s")


if __name__ == "__main__":
    TEST_SIZE = 10 * 1000000  # 10 Million elements
    
    print("Running Real VGRE Backend Benchmarking Suite...")
    benchmark_numpy(TEST_SIZE)
    benchmark_vgre(TEST_SIZE)
