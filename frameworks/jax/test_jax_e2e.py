"""End-to-end: run real JAX-jitted functions on the VGRE HLO engine.

For each case we (1) lower a jax.jit function to StableHLO, (2) translate +
execute it on VGRE, and (3) assert the result matches jax.jit(f)(*args) computed
by JAX's own CPU backend. Nothing is mocked: real lowering, real translation of
the emitted StableHLO ops, real execution on VGRE.
"""
import sys

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("SKIP: jax not installed")
    sys.exit(0)

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import vgre_jax

jax.config.update("jax_enable_x64", False)

PASS = FAIL = 0


def case(name, f, *args):
    global PASS, FAIL
    args = [np.asarray(a, dtype=np.float32) for a in args]
    jit = jax.jit(f)
    ref = np.asarray(jit(*args), dtype=np.float32).ravel()
    try:
        got = vgre_jax.run(jit, *args)
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        FAIL += 1
        return
    ok = got.shape == ref.shape and np.allclose(got, ref, rtol=1e-4, atol=1e-4)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + ("" if ok else f"\n        vgre={got}\n        ref ={ref}"))
    PASS += ok
    FAIL += (not ok)


def main():
    print("=== JAX → StableHLO → VGRE HLO (end-to-end vs jax CPU) ===")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3)).astype(np.float32)
    W = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((4,)).astype(np.float32)

    case("elementwise add", lambda a, b: a + b, x, x)
    case("mul+sub", lambda a, b: a * b - a, x, x)
    case("matmul", lambda a, b: a @ b, x, W)
    case("dense relu(x@W+b)", lambda a, w, c: jnp.maximum(a @ w + c, 0.0), x, W, b)
    case("affine + tanh", lambda a, w, c: jnp.tanh(a @ w + c), x, W, b)
    case("exp/log/abs", lambda a: jnp.abs(jnp.log(jnp.exp(a) + 2.0)), x)
    case("reduce sum axis1", lambda a: jnp.sum(a, axis=1), x)
    case("reduce max axis0", lambda a: jnp.max(a, axis=0), x)
    case("transpose", lambda a: a.T, x)
    case("reshape", lambda a: a.reshape(3, 2), x)
    case("where(>0)", lambda a: jnp.where(a > 0, a, -a), x)
    case("softmax-ish", lambda a: jnp.exp(a - jnp.max(a, axis=1, keepdims=True)), x)
    case("rsqrt norm", lambda a: a * jax.lax.rsqrt(jnp.sum(a * a, axis=1, keepdims=True) + 1e-6), x)

    print(f"\n{PASS} / {PASS + FAIL} passed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
