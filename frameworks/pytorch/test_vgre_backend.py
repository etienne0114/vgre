#!/usr/bin/env python3
"""Exercise the VGRE PyTorch backend (docs/missingFeatures.md §4.2).

JIT-compiles csrc/vgre_backend.cpp against libtorch + libvgre_cudart, registers
VGRE as the "vgre" PrivateUse1 device, then runs real PyTorch tensors on it:
allocation + host<->device copies go through VGRE's emulated CUDA runtime, and
matmul through its emulated cuBLAS; everything else via the CPU fallback.
Exit code 0 = all checks passed.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BUILD = os.environ.get("VGRE_BUILD_DIR", os.path.join(ROOT, "build"))


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


def main():
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print("SKIP: PyTorch not importable:", e)
        return 0

    lib = os.path.join(BUILD, "libvgre_cudart.so")
    if not os.path.exists(lib):
        print("SKIP: libvgre_cudart.so not built at", lib)
        return 0

    from torch.utils.cpp_extension import load
    mod = load(
        name="vgre_torch_backend",
        sources=[os.path.join(HERE, "csrc", "vgre_backend.cpp")],
        extra_include_paths=[os.path.join(ROOT, "include")],
        extra_cflags=["-std=c++17", "-O2"],
        extra_ldflags=[f"-L{BUILD}", "-lvgre_cudart", f"-Wl,-rpath,{BUILD}"],
        verbose=False,
    )
    assert mod is not None
    torch.utils.rename_privateuse1_backend("vgre")

    # Minimal device runtime module so torch can resolve the "vgre" device.
    import types
    devmod = types.ModuleType("torch.vgre")
    devmod.current_device = lambda: 0
    devmod.device_count = lambda: 1
    devmod.is_available = lambda: True
    devmod.is_initialized = lambda: True
    devmod._is_in_bad_fork = lambda: False
    devmod.get_rng_state = lambda device=0: torch.zeros(1, dtype=torch.uint8)
    devmod.set_rng_state = lambda *a, **k: None
    torch._register_device_module("vgre", devmod)
    try:
        torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=False,
                                                             for_storage=False)
    except Exception:
        pass
    dev = "vgre"

    npass = 0

    def check(name, cond):
        nonlocal npass
        print(("  PASS  " if cond else "  FAIL  ") + name)
        if cond:
            npass += 1
        elif "--keep-going" not in sys.argv:
            pass
        return cond

    # ── move a CPU tensor onto the VGRE device (empty + cudaMemcpy H2D) ──
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    av = a.to(dev)
    check("tensor.to('vgre') lands on the vgre device", av.device.type == "vgre")
    check("round-trip to('vgre').cpu() preserves data (H2D+D2H)", torch.allclose(av.cpu(), a))

    # ── elementwise ops via the CPU fallback ─────────────────────────────
    check("add (fallback) correct", torch.allclose((av + av).cpu(), a + a))
    check("scalar mul (fallback) correct", torch.allclose((av * 2.0).cpu(), a * 2.0))
    check("relu (fallback) correct", torch.allclose(torch.relu(av - 3.0).cpu(), torch.relu(a - 3.0)))
    check("sum reduction (fallback) correct", torch.allclose(av.sum().cpu(), a.sum()))

    # ── matmul through VGRE's emulated cuBLAS ────────────────────────────
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3x2
    bv = b.to(dev)
    cref = a @ b
    cv = av @ bv  # routed to vgre_mm → cublasSgemm
    check("matmul via emulated cuBLAS matches reference", torch.allclose(cv.cpu(), cref, atol=1e-4))
    check("matmul result is on the vgre device", cv.device.type == "vgre")

    # ── a tiny end-to-end: y = relu(x @ W) on vgre ───────────────────────
    x = torch.randn(4, 8)
    W = torch.randn(8, 3)
    yref = torch.relu(x @ W)
    yv = torch.relu(x.to(dev) @ W.to(dev))
    check("relu(x@W) end-to-end on vgre matches CPU", torch.allclose(yv.cpu(), yref, atol=1e-4))

    total = 9
    print(f"\n{npass} / {total} passed")
    return 0 if npass == total else 1


if __name__ == "__main__":
    sys.exit(main())
