"""End-to-end: run real Keras 3 models on the VGRE HLO engine.

Each model (CNN, BatchNorm/LayerNorm MLPs, a Transformer block, and the recurrent
family SimpleRNN/LSTM/GRU/stacked/Bidirectional) is built + lowered to StableHLO
in an isolated subprocess (keras_worker.py) — some layers pull in TensorFlow,
whose LLVM collides with libvgre's if co-loaded. This driver, which never imports
Keras, parses that StableHLO, runs it on VGRE and asserts the output matches
Keras' own inference. The recurrent models exercise the engine's control flow
(while / dynamic-slice / dynamic-update-slice / reverse).
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
CASES = ["mlp", "cnn", "batchnorm", "layernorm_gelu", "transformer",
         "simple_rnn", "lstm", "gru", "stacked_lstm", "bidirectional"]


def run_case(name, tmp):
    out = os.path.join(tmp, name)
    r = subprocess.run([PY, os.path.join(HERE, "keras_worker.py"), name, out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or ["?"])[-1]
        return False, f"worker failed: {tail}"
    with jmlir.make_ir_context():
        mod = ir.Module.parse(open(os.path.join(out, "m.mlir")).read())
        exe = Translator().compile_module(mod)
    x = np.load(os.path.join(out, "in_0.npy"))
    got = exe(x)
    ref = np.load(os.path.join(out, "ref.npy"))
    ok = got.shape == ref.shape and np.allclose(got, ref, rtol=2e-3, atol=2e-3)
    return ok, ("" if ok else f"maxdiff {np.max(np.abs(got-ref)):.3g}")


def main():
    probe = subprocess.run([PY, "-c", "import keras"], capture_output=True, text=True,
                           env={**os.environ, "KERAS_BACKEND": "jax"})
    if probe.returncode != 0:
        print("SKIP: keras not importable")
        return 0
    print("=== Keras 3 models → StableHLO → VGRE HLO (vs Keras) ===")
    npass = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name in CASES:
            try:
                ok, msg = run_case(name, tmp)
            except Exception as e:
                ok, msg = False, f"{type(e).__name__}: {str(e)[:90]}"
            print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({msg})" if not ok else ""))
            npass += ok
    print(f"\n{npass} / {len(CASES)} passed")
    return 0 if npass == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
