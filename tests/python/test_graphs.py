import sys
from pathlib import Path
try:
    import numpy as np
except Exception:
    print("SKIP: numpy not available")
    raise SystemExit(0)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bindings/python"))

from vgre import Runtime
from vgre.kernel import vector_add_kernel, Dim3
from vgre.memory import DeviceArray
from vgre.graph import Graph, ARG_POINTER, ARG_INT32
from vgre._native import native_register_kernel


def main():
    rt = Runtime()
    rt.init()

    N = 512
    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)

    d_a = DeviceArray.from_numpy(a)
    d_b = DeviceArray.from_numpy(b)
    d_c = DeviceArray.zeros(N, dtype=np.float32)

    kernel = vector_add_kernel()
    kernel_id = native_register_kernel(kernel.name, kernel.source)

    g = Graph()
    node = g.add_kernel_node(
        kernel_id,
        kernel.name,
        (2, 1, 1),
        (256, 1, 1),
        [d_a, d_b, d_c, N],
        [ARG_POINTER, ARG_POINTER, ARG_POINTER, ARG_INT32],
    )
    g.instantiate()
    g.launch()
    rt.synchronize()

    out = d_c.to_numpy()
    if not np.allclose(out[:N], a + b, atol=1e-5):
        raise SystemExit("Graph output mismatch")

    d_a.free()
    d_b.free()
    d_c.free()
    rt.shutdown()
    print("PASS: python graph")


if __name__ == "__main__":
    main()
