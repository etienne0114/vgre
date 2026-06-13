"""End-to-end: run real Keras 3 models on the VGRE HLO engine.

Builds complete networks with the high-level Keras API — CNN, BatchNorm/LayerNorm
MLPs, a Transformer block, and the recurrent family SimpleRNN/LSTM/GRU/stacked/
Bidirectional — lowers each to StableHLO (Keras JAX backend) and runs it on VGRE,
asserting the output matches Keras' own inference.

Runs in one process: libvgre is loaded with RTLD_DEEPBIND (see vgre_jax.runtime),
so its statically-linked LLVM and its protobuf/abseil deps bind to their own
copies rather than the ones TensorFlow bundles (Bidirectional pulls in TF). The
recurrent models exercise the engine's control flow (while / dynamic-slice /
dynamic-update-slice / reverse).
"""
import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jax"))
sys.path.insert(0, HERE)

try:
    import jax
    import keras  # noqa: F401
    from jax.interpreters import mlir as jmlir  # noqa: F401
    import vgre_jax
    import keras_worker  # model builders
except ImportError as e:
    print(f"SKIP: prerequisites missing ({e})")
    sys.exit(0)

CASES = ["mlp", "cnn", "batchnorm", "layernorm_gelu", "transformer",
         "simple_rnn", "lstm", "gru", "stacked_lstm", "bidirectional"]


def main():
    print("=== Keras 3 models → StableHLO → VGRE HLO (in-process, vs Keras) ===")
    npass = 0
    for name in CASES:
        try:
            model, x = keras_worker.build(name)
            model(x)  # initialise variables
            call = jax.jit(lambda z: model(z, training=False))
            ref = np.asarray(call(x), np.float32).ravel()
            got = vgre_jax.run(call, x)
            ok = got.shape == ref.shape and np.allclose(got, ref, rtol=2e-3, atol=2e-3)
            msg = "" if ok else f"  (maxdiff {np.max(np.abs(got-ref)):.3g})"
        except Exception as e:
            ok, msg = False, f": {type(e).__name__}: {str(e)[:80]}"
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{msg}")
        npass += ok
    print(f"\n{npass} / {len(CASES)} passed")
    return 0 if npass == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
