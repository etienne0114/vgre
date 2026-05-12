#!/usr/bin/env python3
"""
End-to-end test: Run CUDA vector-addition on CPU via VGRE.

This script loads libvgre.so through ctypes and exercises:
  - vgre_init / vgre_shutdown
  - vgre_malloc / vgre_free
  - vgre_memcpy (H2D, D2H)
  - vgre_register_kernel (CUDA C source)
  - vgre_launch_kernel
  - Result verification

Usage:
    LD_LIBRARY_PATH=build python3 test_cuda_on_cpu.py
"""

import ctypes
import ctypes.util
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Locate libvgre.so
# ---------------------------------------------------------------------------
BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
LIB_PATH = os.path.join(BUILD_DIR, "libvgre.so")

if not os.path.exists(LIB_PATH):
    print(f"ERROR: {LIB_PATH} not found. Build first: cmake --build build")
    sys.exit(1)

# Prepend build dir so the dynamic linker finds libvgre_cudart.so as well.
if BUILD_DIR not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = BUILD_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")

lib = ctypes.CDLL(LIB_PATH)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VGRE_SUCCESS = 0
VGRE_MEMCPY_HOST_TO_DEVICE = 0
VGRE_MEMCPY_DEVICE_TO_HOST = 1

# ---------------------------------------------------------------------------
# Helper: ctypes wrappers
# ---------------------------------------------------------------------------
def _check(rc, msg):
    if rc != VGRE_SUCCESS:
        raise RuntimeError(f"{msg} failed with code {rc}")

class VgreContext:
    def __init__(self):
        _check(lib.vgre_init(), "vgre_init")
        self._closed = False

    def close(self):
        if not self._closed:
            lib.vgre_shutdown()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_device_count(self):
        count = ctypes.c_int()
        _check(lib.vgre_get_device_count(ctypes.byref(count)), "vgre_get_device_count")
        return count.value

    def malloc(self, size):
        ptr = ctypes.c_void_p()
        _check(lib.vgre_malloc(ctypes.byref(ptr), size), "vgre_malloc")
        return ptr.value

    def free(self, ptr):
        _check(lib.vgre_free(ctypes.c_void_p(ptr)), "vgre_free")

    def memcpy_h2d(self, dst, src, count):
        _check(lib.vgre_memcpy(ctypes.c_void_p(dst), src, count, VGRE_MEMCPY_HOST_TO_DEVICE),
               "vgre_memcpy H2D")

    def memcpy_d2h(self, dst, src, count):
        _check(lib.vgre_memcpy(dst, ctypes.c_void_p(src), count, VGRE_MEMCPY_DEVICE_TO_HOST),
               "vgre_memcpy D2H")

    def register_kernel(self, name, source):
        kid = ctypes.c_uint64()
        _check(lib.vgre_register_kernel(name.encode(), source.encode(), ctypes.byref(kid)),
               "vgre_register_kernel")
        return kid.value

    def launch_kernel(self, kernel_id, grid, block, args, shared_mem=0, stream=0):
        # grid / block are [x, y, z] uint32 arrays
        grid_arr = (ctypes.c_uint32 * 3)(*grid)
        block_arr = (ctypes.c_uint32 * 3)(*block)
        _check(lib.vgre_launch_kernel(kernel_id, grid_arr, block_arr, args, len(args),
                                     shared_mem, stream),
               "vgre_launch_kernel")


# ---------------------------------------------------------------------------
# Simple vector-addition CUDA kernel
# ---------------------------------------------------------------------------
KERNEL_SOURCE = r'''
extern "C" __global__ void vecAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
'''

# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("VGRE CUDA-on-CPU End-to-End Test")
    print("=" * 60)

    os.environ["VGRE_IPC_MODE"] = "OFF"

    with VgreContext() as vgre:
        dev_count = vgre.get_device_count()
        print(f"\n[1] Devices detected: {dev_count}")

        N = 1024
        nbytes = N * ctypes.sizeof(ctypes.c_float)

        # Host arrays
        h_A = np.random.rand(N).astype(np.float32)
        h_B = np.random.rand(N).astype(np.float32)
        h_C = np.empty(N, dtype=np.float32)

        print(f"[2] Array size: {N} elements ({nbytes} bytes)")

        # Device allocation
        d_A = vgre.malloc(nbytes)
        d_B = vgre.malloc(nbytes)
        d_C = vgre.malloc(nbytes)
        print(f"[3] Device memory allocated at 0x{d_A:x}, 0x{d_B:x}, 0x{d_C:x}")

        # H2D copy
        vgre.memcpy_h2d(d_A, h_A.ctypes.data, nbytes)
        vgre.memcpy_h2d(d_B, h_B.ctypes.data, nbytes)
        print("[4] Host -> Device copy done")

        # Register kernel
        kid = vgre.register_kernel("vecAdd", KERNEL_SOURCE)
        print(f"[5] Kernel registered, ID = {kid}")

        # Build argument array: (void**)[&d_A, &d_B, &d_C, &n]
        # Each element is the ADDRESS OF the variable holding the value.
        # The runtime dereferences each element to get the actual argument.
        n_arg = ctypes.c_int32(N)
        d_A_ptr = ctypes.c_void_p(d_A)
        d_B_ptr = ctypes.c_void_p(d_B)
        d_C_ptr = ctypes.c_void_p(d_C)
        arg_ptrs = (ctypes.c_void_p * 4)(
            ctypes.addressof(d_A_ptr),
            ctypes.addressof(d_B_ptr),
            ctypes.addressof(d_C_ptr),
            ctypes.addressof(n_arg),
        )

        # Launch
        block = (256, 1, 1)
        grid = ((N + block[0] - 1) // block[0], 1, 1)
        vgre.launch_kernel(kid, grid, block, arg_ptrs)
        print(f"[6] Kernel launched: grid={grid}, block={block}")

        # Synchronize to ensure kernel completes before readback
        lib.vgre_synchronize()
        print("[6b] Synchronized")

        # D2H copy
        vgre.memcpy_d2h(h_C.ctypes.data, d_C, nbytes)
        print("[7] Device -> Host copy done")

        # Verify
        expected = h_A + h_B
        max_err = np.max(np.abs(h_C - expected))
        rel_err = max_err / (np.max(np.abs(expected)) + 1e-9)

        print(f"[8] Verification: max absolute error = {max_err:.6e}, relative = {rel_err:.6e}")
        print(f"    Input A (first 5): {h_A[:5]}")
        print(f"    Input B (first 5): {h_B[:5]}")
        print(f"    Output  (first 5): {h_C[:5]}")
        print(f"    Expected(first 5): {expected[:5]}")

        if max_err < 1e-5:
            print("\n" + "=" * 60)
            print("PASS — CUDA vector-add executed correctly on CPU!")
            print("=" * 60)
            rc = 0
        else:
            print("\n" + "=" * 60)
            print(f"FAIL — max error {max_err} exceeds threshold")
            print("=" * 60)
            rc = 1

        # Cleanup
        vgre.free(d_A)
        vgre.free(d_B)
        vgre.free(d_C)
        print("[9] Device memory freed")

        return rc


if __name__ == "__main__":
    sys.exit(main())
