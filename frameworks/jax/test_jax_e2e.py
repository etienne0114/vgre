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

    # ── real-model ops ───────────────────────────────────────────────────────
    q = rng.standard_normal((2, 4, 8)).astype(np.float32)
    k = rng.standard_normal((2, 4, 8)).astype(np.float32)
    v = rng.standard_normal((2, 4, 8)).astype(np.float32)
    img = rng.standard_normal((1, 2, 7, 7)).astype(np.float32)
    ker = rng.standard_normal((3, 2, 3, 3)).astype(np.float32)
    tbl = rng.standard_normal((10, 5)).astype(np.float32)
    ids = np.array([2, 7, 1, 9], np.int32)

    case("batched matmul", lambda a, b: jnp.einsum("bij,bjk->bik", a, b),
         rng.standard_normal((3, 2, 5)).astype(np.float32),
         rng.standard_normal((3, 5, 4)).astype(np.float32))
    case("self-attention", lambda Q, K, V: jax.nn.softmax(
        (Q @ K.transpose(0, 2, 1)) / np.float32(np.sqrt(8.0)), axis=-1) @ V, q, k, v)
    case("softmax(nn)", lambda a: jax.nn.softmax(a, axis=-1), x)
    case("sigmoid(nn)", lambda a: jax.nn.sigmoid(a), x)
    case("layernorm", lambda a: (a - a.mean(-1, keepdims=True))
         / jnp.sqrt(a.var(-1, keepdims=True) + 1e-5), x)
    x4 = rng.standard_normal((2, 4)).astype(np.float32)
    vec3 = rng.standard_normal((3,)).astype(np.float32)
    case("concatenate", lambda a, b: jnp.concatenate([a, b], axis=1), x, x4)
    case("slice", lambda a: a[:, 1:3], x)
    case("pad", lambda a: jnp.pad(a, ((1, 0), (0, 2))), x)
    case("matvec", lambda a, v: a @ v, x, vec3)
    case("dot 1D", lambda a, b: a @ b, b, b)
    case("sqrt|x|", lambda a: jnp.sqrt(jnp.abs(a)), x)
    case("conv2d SAME", lambda im, kr: jax.lax.conv_general_dilated(
        im, kr, (1, 1), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW")), img, ker)
    case("conv2d VALID s2", lambda im, kr: jax.lax.conv_general_dilated(
        im, kr, (2, 2), "VALID", dimension_numbers=("NCHW", "OIHW", "NCHW")), img, ker)
    case("conv2d grouped", lambda im, kr: jax.lax.conv_general_dilated(
        im, kr, (1, 1), "SAME", feature_group_count=2,
        dimension_numbers=("NCHW", "OIHW", "NCHW")),
        rng.standard_normal((1, 4, 5, 5)).astype(np.float32),
        rng.standard_normal((4, 2, 3, 3)).astype(np.float32))
    case("maxpool2d", lambda im: jax.lax.reduce_window(
        im, -jnp.inf, jax.lax.max, (1, 1, 2, 2), (1, 1, 2, 2), "VALID"), img)
    case("avgpool2d", lambda im: jax.lax.reduce_window(
        im, 0.0, jax.lax.add, (1, 1, 2, 2), (1, 1, 2, 2), "VALID") / 4.0, img)
    case("embedding gather", lambda t, i: t[i.astype(jnp.int32)], tbl, ids.astype(np.float32))
    case("2-layer MLP", lambda a, w1, b1, w2, b2: jax.nn.sigmoid(
        jnp.maximum(a @ w1 + b1, 0.0) @ w2 + b2),
        x, W, b, rng.standard_normal((4, 2)).astype(np.float32),
        rng.standard_normal((2,)).astype(np.float32))

    print(f"\n{PASS} / {PASS + FAIL} passed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
