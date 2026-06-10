#!/usr/bin/env python3
# Track 24 — PyTorch end-to-end validation harness.
#
# A real NVIDIA driver (libcuda.so.1) is absent on this host, so torch+cu130
# cannot enable its CUDA backend (torch.cuda.is_available() == False) — running
# torch's own kernels *on* VGRE would need a faked driver. Instead we use PyTorch
# as the trusted ORACLE: torch (CPU) generates the inputs and the reference, and
# the SAME computation is executed through VGRE's exported CUDA/cuBLAS C API via
# ctypes (real device allocations, real cublasSgemm). The two are compared — this
# validates that VGRE reproduces PyTorch's numerics through the public GPU API,
# which is what an "end-to-end PyTorch validation" actually needs to assert.

import ctypes
import os
import sys

try:
    import numpy as np
    import torch
except ImportError as e:
    print(f"=== PyTorch validation harness (Track 24) ===\n  [SKIP] {e} not installed")
    sys.exit(0)  # not a failure — environment lacks PyTorch/NumPy

HOST_TO_DEVICE = 1
DEVICE_TO_HOST = 2


def load_vgre():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    for cand in (
        os.path.join(root, "build", "libvgre_cudart.so"),
        os.path.join(root, "build", "libvgre_cudart.so.0"),
    ):
        if os.path.exists(cand):
            return ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
    raise FileNotFoundError("libvgre_cudart.so not found under build/")


def bind(lib):
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int
    lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cublasCreate_v2.restype = ctypes.c_int
    lib.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    lib.cublasDestroy_v2.restype = ctypes.c_int
    # cublasSgemm_v2(handle, transa, transb, m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc)
    lib.cublasSgemm_v2.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int,
    ]
    lib.cublasSgemm_v2.restype = ctypes.c_int


def vgre_to_device(lib, host_np):
    nbytes = host_np.nbytes
    dptr = ctypes.c_void_p()
    assert lib.cudaMalloc(ctypes.byref(dptr), nbytes) == 0, "cudaMalloc failed"
    src = host_np.ctypes.data_as(ctypes.c_void_p)
    assert lib.cudaMemcpy(dptr, src, nbytes, HOST_TO_DEVICE) == 0, "H2D failed"
    return dptr, nbytes


def vgre_sgemm(lib, A_rm, B_rm):
    """Compute A_rm @ B_rm (row-major) through VGRE cuBLAS and return the result.

    cuBLAS is column-major; a row-major MxK buffer read column-major is its
    transpose. So to get row-major R = A@B we ask cuBLAS for R^T = B^T @ A^T,
    i.e. sgemm(N,N, m=N, n=M, k=K, B, ldb=N, A, lda=K, C, ldc=N), and read C as
    the row-major MxN result.
    """
    M, K = A_rm.shape
    K2, N = B_rm.shape
    assert K == K2
    A = np.ascontiguousarray(A_rm, dtype=np.float32)
    B = np.ascontiguousarray(B_rm, dtype=np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    dA, _ = vgre_to_device(lib, A)
    dB, _ = vgre_to_device(lib, B)
    dC, csz = vgre_to_device(lib, C)

    handle = ctypes.c_void_p()
    assert lib.cublasCreate_v2(ctypes.byref(handle)) == 0, "cublasCreate failed"
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)
    OP_N = 0
    # R^T(N×M) = B^T(N×K) · A^T(K×M); operands are the row-major buffers read
    # column-major (already the transposes), so pass them untransposed.
    rc = lib.cublasSgemm_v2(handle, OP_N, OP_N, N, M, K,
                            ctypes.byref(alpha), dB, N, dA, K,
                            ctypes.byref(beta), dC, N)
    assert rc == 0, f"cublasSgemm returned {rc}"
    lib.cublasDestroy_v2(handle)

    out = np.empty((M, N), dtype=np.float32)
    assert lib.cudaMemcpy(out.ctypes.data_as(ctypes.c_void_p), dC, csz, DEVICE_TO_HOST) == 0
    for d in (dA, dB, dC):
        lib.cudaFree(d)
    return out


def main():
    print("=== PyTorch end-to-end validation harness (Track 24) ===")
    print(f"  torch {torch.__version__}; cuda.is_available()={torch.cuda.is_available()} "
          f"(no driver → using torch CPU as the oracle)")
    lib = load_vgre()
    bind(lib)

    torch.manual_seed(0)
    failures = 0
    for (M, K, N) in [(32, 32, 32), (64, 48, 16), (1, 128, 1), (17, 5, 23)]:
        a = torch.randn(M, K)
        b = torch.randn(K, N)
        ref = (a @ b).numpy()                       # PyTorch oracle (CPU)
        got = vgre_sgemm(lib, a.numpy(), b.numpy())  # VGRE cuBLAS via ctypes
        rel = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-8)
        ok = rel < 1e-4
        print(f"  [{'PASS' if ok else 'FAIL'}] GEMM {M}x{K}x{N}: "
              f"VGRE vs PyTorch rel err = {rel:.2e}")
        failures += (0 if ok else 1)

    # A larger, more realistic linear-layer shape (transformer FFN slice).
    a = torch.randn(128, 256)
    b = torch.randn(256, 512)
    ref = (a @ b).numpy()
    got = vgre_sgemm(lib, a.numpy(), b.numpy())
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    ok = rel < 1e-4
    print(f"  [{'PASS' if ok else 'FAIL'}] FFN 128x256x512: rel err = {rel:.2e}")
    failures += (0 if ok else 1)

    print(f"\n{'ALL PASSED' if failures == 0 else str(failures) + ' FAILED'}")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
