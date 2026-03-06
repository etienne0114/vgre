"""
VGRE Python Test — TensorFlow-like Integration

Tests tensor operations using the VGRE Python runtime,
simulating how TensorFlow eager-mode ops would flow through the engine.
"""
import sys
import numpy as np

sys.path.insert(0, "../bindings/python")

from vgre import VirtualDevice, Runtime
from vgre.kernel import Kernel, Dim3


MATMUL_SOURCE = """
extern "C" __global__ void matmul(float* a, float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}
"""

RELU_SOURCE = """
extern "C" __global__ void relu(float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (x[i] > 0.0f) ? x[i] : 0.0f;
    }
}
"""


def test_matmul():
    """Test matrix multiplication (dense layer simulation)."""
    print("--- Test: Matrix Multiply (TF-like dense layer) ---")
    N = 64

    a = np.random.randn(N, N).astype(np.float32)
    b = np.random.randn(N, N).astype(np.float32)
    c = np.zeros((N, N), dtype=np.float32)

    rt = Runtime()
    rt.init()

    kernel = Kernel("matmul", MATMUL_SOURCE)
    block_size = 16
    grid_size = (N + block_size - 1) // block_size

    elapsed = rt.launch(
        kernel,
        Dim3(grid_size, grid_size),
        Dim3(block_size, block_size),
        [a, b, c, N]
    )
    rt.synchronize()

    expected = a @ b
    max_err = np.max(np.abs(c - expected))
    assert max_err < 0.1, f"Max error: {max_err}"

    print(f"  N={N}x{N}, time={elapsed:.1f}ms, max_error={max_err:.2e}")
    print("[PASS] Matrix multiply\n")
    rt.shutdown()


def test_relu_activation():
    """Test ReLU activation function."""
    print("--- Test: ReLU Activation ---")
    N = 4096

    x = np.random.randn(N).astype(np.float32)
    out = np.zeros(N, dtype=np.float32)

    rt = Runtime()
    rt.init()

    kernel = Kernel("relu", RELU_SOURCE)
    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    elapsed = rt.launch(kernel, Dim3(grid_size), Dim3(block_size), [x, out, N])
    rt.synchronize()

    expected = np.maximum(0, x)
    max_err = np.max(np.abs(out - expected))
    assert max_err < 1e-6, f"Max error: {max_err}"

    print(f"  N={N}, time={elapsed:.3f}ms")
    print("[PASS] ReLU activation\n")
    rt.shutdown()


def test_dense_layer_forward():
    """Test a full dense layer: output = ReLU(X @ W + bias)."""
    print("--- Test: Dense Layer Forward Pass ---")
    batch = 32
    in_features = 128
    out_features = 64

    X = np.random.randn(batch, in_features).astype(np.float32)
    W = np.random.randn(in_features, out_features).astype(np.float32)
    bias = np.random.randn(out_features).astype(np.float32)

    rt = Runtime()
    rt.init()

    # Step 1: matmul (batch x in) @ (in x out) -> (batch x out)
    Z = np.zeros((batch, out_features), dtype=np.float32)
    
    # Custom matmul source for batch processing
    DENSE_MATMUL = """
    extern "C" __global__ void dense_matmul(float* x, float* w, float* z, int batch, int in_f, int out_f) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < batch && col < out_f) {
            float sum = 0.0f;
            for (int k = 0; k < in_f; ++k) {
                sum += x[row * in_f + k] * w[k * out_f + col];
            }
            z[row * out_f + col] = sum;
        }
    }
    """
    
    matmul_kernel = Kernel("dense_matmul", DENSE_MATMUL)
    block = 16
    grid_x = (out_features + block - 1) // block
    grid_y = (batch + block - 1) // block

    rt.launch(
        matmul_kernel,
        Dim3(grid_x, grid_y), Dim3(block, block),
        [X, W, Z, batch, in_features, out_features]
    )
    rt.synchronize()

    # Step 2: add bias (Python side as simulate-framework logic)
    Z = Z + bias

    # Step 3: ReLU
    out_flat = np.zeros(batch * out_features, dtype=np.float32)
    z_flat = Z.flatten()
    relu_kernel = Kernel("dense_relu", RELU_SOURCE)
    total = batch * out_features
    rt.launch(
        relu_kernel,
        Dim3((total + 255) // 256), Dim3(256),
        [z_flat, out_flat, total]
    )
    rt.synchronize()

    output = out_flat.reshape(batch, out_features)

    # Verify against numpy
    expected = np.maximum(0, X @ W + bias)
    max_err = np.max(np.abs(output - expected))

    print(f"  Shape: ({batch},{in_features}) → ({batch},{out_features})")
    print(f"  Max error: {max_err:.2e}")
    assert max_err < 1e-4

    print("[PASS] Dense layer forward\n")
    rt.shutdown()


def test_runtime_context_manager():
    """Test Runtime as a context manager."""
    print("--- Test: Runtime Context Manager ---")

    with Runtime() as rt:
        assert rt.is_initialized()
        dev = rt.get_device()
        assert dev.has_context()
        print(f"  Device: {dev}")

    print("[PASS] Context manager\n")


if __name__ == "__main__":
    print("=" * 50)
    print("  VGRE TensorFlow Integration Test")
    print("=" * 50 + "\n")

    test_matmul()
    test_relu_activation()
    test_dense_layer_forward()
    test_runtime_context_manager()

    print("✓ All TensorFlow integration tests passed!")
