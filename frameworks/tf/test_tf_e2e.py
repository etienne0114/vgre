"""End-to-end: run real TensorFlow (XLA) graphs on the VGRE HLO engine.

For each case a TF subprocess (tf_worker.py) XLA-compiles a tf.function and dumps
its StableHLO + inputs + TF's own reference output. This driver — which never
imports TensorFlow — parses that StableHLO and executes it on VGRE, then asserts
the result matches TF. Real TF lowering, real translation, real execution; the
two-process split avoids the TF/jaxlib MLIR co-load segfault.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jax"))

try:
    from jax.interpreters import mlir as jmlir
    from jaxlib.mlir import ir
    import vgre_jax
    from vgre_jax.translate import Translator
except ImportError as e:
    print(f"SKIP: prerequisites missing ({e})")
    sys.exit(0)

PY = sys.executable
CASES = [
    "matmul", "dense_relu", "add_mul", "tanh", "sigmoid", "softmax",
    "reduce_sum", "reduce_mean", "reduce_max", "transpose", "exp_log_abs",
    "rsqrt", "concat", "relu_conv2d", "maxpool2d", "avgpool2d",
]


def run_case(name, tmp):
    out = os.path.join(tmp, name)
    r = subprocess.run([PY, os.path.join(HERE, "tf_worker.py"), name, out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"worker failed: {r.stderr.strip().splitlines()[-1:] }"
    txt = open(os.path.join(out, "m.mlir")).read()
    with jmlir.make_ir_context():
        mod = ir.Module.parse(txt)
        exe = Translator().compile_module(mod)
    inputs = []
    i = 0
    while os.path.exists(os.path.join(out, f"in_{i}.npy")):
        inputs.append(np.load(os.path.join(out, f"in_{i}.npy")))
        i += 1
    got = exe(*inputs)
    ref = np.load(os.path.join(out, "ref.npy"))
    ok = got.shape == ref.shape and np.allclose(got, ref, rtol=1e-3, atol=1e-3)
    return ok, ("" if ok else f"got{got[:4]} ref{ref[:4]} (maxdiff {np.max(np.abs(got-ref)):.3g})")


def main():
    # quick availability probe: can the worker import TF at all?
    probe = subprocess.run([PY, "-c", "import tensorflow"], capture_output=True, text=True)
    if probe.returncode != 0:
        print("SKIP: tensorflow not importable")
        return 0
    print("=== TensorFlow (XLA) → StableHLO → VGRE HLO (vs TF) ===")
    npass = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name in CASES:
            try:
                ok, msg = run_case(name, tmp)
            except Exception as e:
                ok, msg = False, f"{type(e).__name__}: {str(e)[:90]}"
            print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {msg}" if not ok else ""))
            npass += ok
    print(f"\n{npass} / {len(CASES)} passed")
    return 0 if npass == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
