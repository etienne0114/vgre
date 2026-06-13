"""End-to-end: run real TensorFlow (XLA) graphs on the VGRE HLO engine.

Each case XLA-compiles a tf.function, lowers it to StableHLO, translates that to
VGRE HLO and executes it, then asserts the result matches TensorFlow's own.
Everything runs in one process: libvgre is loaded with RTLD_DEEPBIND (see
vgre_jax.runtime), so its statically-linked LLVM and its protobuf/abseil deps
bind to their own copies instead of TensorFlow's bundled ones — no subprocess
isolation needed.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jax"))
sys.path.insert(0, HERE)

try:
    import tensorflow as tf
    from jax.interpreters import mlir as jmlir
    from jaxlib.mlir import ir
    import vgre_jax  # noqa: F401  (loads libvgre with RTLD_DEEPBIND)
    from vgre_jax.translate import Translator
    import tf_worker  # case definitions
except ImportError as e:
    print(f"SKIP: prerequisites missing ({e})")
    sys.exit(0)


def main():
    print("=== TensorFlow (XLA) → StableHLO → VGRE HLO (in-process, vs TF) ===")
    cases = tf_worker.cases()
    npass = 0
    for name, (fn, inputs) in cases.items():
        try:
            g = tf.function(fn, jit_compile=True)
            targs = [tf.constant(a) for a in inputs]
            txt = g.experimental_get_compiler_ir(*targs)(stage="stablehlo")
            with jmlir.make_ir_context():
                mod = ir.Module.parse(txt)
                exe = Translator().compile_module(mod)
            got = exe(*inputs)
            ref = np.asarray(g(*targs), np.float32).ravel()
            ok = got.shape == ref.shape and np.allclose(got, ref, rtol=2e-3, atol=2e-3)
            msg = "" if ok else f"  (maxdiff {np.max(np.abs(got-ref)):.3g})"
        except Exception as e:
            ok, msg = False, f": {type(e).__name__}: {str(e)[:80]}"
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{msg}")
        npass += ok
    print(f"\n{npass} / {len(cases)} passed")
    return 0 if npass == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
