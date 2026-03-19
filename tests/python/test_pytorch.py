"""
VGRE Python Test — PyTorch-like Integration

Tests vector operations using the VGRE Python runtime,
simulating how PyTorch tensor ops would flow through the engine.
"""
import sys
from pathlib import Path
import numpy as np

# Add bindings to path (repo-root relative)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bindings/python"))

from vgre import VirtualDevice, Kernel, Runtime
from vgre.kernel import Dim3, vector_add_kernel, vector_mul_kernel


def test_device_detection():
    """Test virtual GPU device detection."""
    print("--- Test: Device Detection ---")
    dev = VirtualDevice()
    props = dev.get_properties()

    assert props.multi_processor_count > 0
    assert props.total_global_mem > 0
    assert VirtualDevice.get_device_count() >= 1

    print(f"  Device: {props.name}")
    print(f"  Cores:  {props.multi_processor_count}")
    print(f"  VRAM:   {props.total_global_mem // (1024*1024)} MB")
    print("[PASS] Device detection\n")


def test_vector_addition():
    """Test vector addition kernel execution."""
    print("--- Test: Vector Addition (PyTorch-style) ---")
    N = 4096

    # Simulate PyTorch tensors as numpy arrays
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    c = np.zeros(N, dtype=np.float32)

    # Launch via VGRE runtime
    rt = Runtime()
    rt.init(enable_profiling=True)

    kernel = vector_add_kernel()
    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    elapsed = rt.launch(
        kernel,
        Dim3(grid_size), Dim3(block_size),
        [a, b, c, N]
    )
    rt.synchronize()

    # Verify
    expected = a + b
    max_err = np.max(np.abs(c - expected))
    assert max_err < 1e-5, f"Max error: {max_err}"

    print(f"  N={N}, time={elapsed:.3f}ms, max_error={max_err:.2e}")
    print("[PASS] Vector addition\n")

    rt.shutdown()


def test_vector_multiplication():
    """Test element-wise multiplication."""
    print("--- Test: Vector Multiplication ---")
    N = 2048

    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    c = np.zeros(N, dtype=np.float32)

    rt = Runtime()
    rt.init()
    kernel = vector_mul_kernel()
    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    elapsed = rt.launch(kernel, Dim3(grid_size), Dim3(block_size), [a, b, c, N])
    rt.synchronize()

    expected = a * b
    max_err = np.max(np.abs(c - expected))
    assert max_err < 1e-5, f"Max error: {max_err}"

    print(f"  N={N}, time={elapsed:.3f}ms, max_error={max_err:.2e}")
    print("[PASS] Vector multiplication\n")
    rt.shutdown()


def test_profiling():
    """Test profiling report generation."""
    print("--- Test: Profiling ---")
    N = 1024

    rt = Runtime()
    rt.init(enable_profiling=True)

    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    c = np.zeros(N, dtype=np.float32)

    kernel = vector_add_kernel()

    for _ in range(5):
        rt.launch(kernel, Dim3(4), Dim3(256), [a, b, c, N])
    rt.synchronize()

    report = rt.get_profiling_report()
    print(report.summary())

    json_str = report.to_json()
    assert '"profiler"' in json_str
    assert '"total_events": 5' in json_str

    print("[PASS] Profiling\n")
    rt.shutdown()


def test_large_tensor():
    """Test with a large tensor (simulating AI workload size)."""
    print("--- Test: Large Tensor (1M elements) ---")
    N = 1_000_000

    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    c = np.zeros(N, dtype=np.float32)

    rt = Runtime()
    rt.init()
    kernel = vector_add_kernel()
    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    elapsed = rt.launch(
        kernel, Dim3(grid_size), Dim3(block_size), [a, b, c, N]
    )
    rt.synchronize()

    expected = a + b
    max_err = np.max(np.abs(c - expected))
    assert max_err < 1e-5, f"Max error: {max_err}"

    throughput = (3 * N * 4) / (elapsed / 1000) / 1e9  # GB/s
    print(f"  N={N}, time={elapsed:.1f}ms, throughput={throughput:.2f} GB/s")
    print("[PASS] Large tensor\n")
    rt.shutdown()


if __name__ == "__main__":
    print("=" * 50)
    print("  VGRE PyTorch Integration Test")
    print("=" * 50 + "\n")

    test_device_detection()
    test_vector_addition()
    test_vector_multiplication()
    test_profiling()
    test_large_tensor()

    print("✓ All PyTorch integration tests passed!")
