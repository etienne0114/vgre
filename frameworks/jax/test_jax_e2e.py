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

    # ── control flow (while / dynamic-slice / scan) ──────────────────────────
    seq = rng.standard_normal((6,)).astype(np.float32)
    case("scan cumsum", lambda xs: jax.lax.scan(lambda c, v: (c + v, c + v),
         np.float32(0), xs)[1], seq)
    case("scan final carry", lambda xs: jax.lax.scan(lambda c, v: (c * 0.9 + v, c),
         np.float32(0), xs)[0], seq)
    case("fori_loop sum", lambda s: jax.lax.fori_loop(
        0, 6, lambda i, acc: acc + s[i], np.float32(0)), seq)
    case("while_loop", lambda s: jax.lax.while_loop(
        lambda st: st[0] < 5, lambda st: (st[0] + 1, st[1] + s[st[0]]),
        (0, np.float32(0)))[1], seq)

    # a hand-written LSTM cell rolled over time with lax.scan (real recurrence)
    def lstm_seq(xs, Wi, Wh, bvec):
        def step(carry, xt):
            h, c = carry
            z = xt @ Wi + h @ Wh + bvec
            i, f, g, o = jnp.split(z, 4, axis=-1)
            i, f, o = jax.nn.sigmoid(i), jax.nn.sigmoid(f), jax.nn.sigmoid(o)
            c = f * c + i * jnp.tanh(g)
            h = o * jnp.tanh(c)
            return (h, c), h
        h0 = jnp.zeros((xs.shape[0], 4)); c0 = jnp.zeros((xs.shape[0], 4))
        (hN, _), _ = jax.lax.scan(step, (h0, c0), xs.transpose(1, 0, 2))
        return hN
    xs = rng.standard_normal((2, 5, 3)).astype(np.float32)
    Wi = rng.standard_normal((3, 16)).astype(np.float32)
    Wh = rng.standard_normal((4, 16)).astype(np.float32)
    bv = rng.standard_normal((16,)).astype(np.float32)
    case("lstm cell via scan", lstm_seq, xs, Wi, Wh, bv)

    # ── sort / top-k / argmax (generation & sampling) ────────────────────────
    s2 = rng.standard_normal((3, 6)).astype(np.float32)
    case("sort ascending", lambda a: jnp.sort(a, -1), s2)
    case("argsort", lambda a: jnp.argsort(a, -1).astype(jnp.float32), s2)
    case("top_k values", lambda a: jax.lax.top_k(a, 3)[0], s2)
    case("top_k indices", lambda a: jax.lax.top_k(a, 3)[1].astype(jnp.float32), s2)
    case("argmax (greedy decode)", lambda a: jnp.argmax(a, -1).astype(jnp.float32), s2)
    case("argmin", lambda a: jnp.argmin(a, -1).astype(jnp.float32), s2)
    case("logical and>0", lambda a, b: jnp.logical_and(a > 0, b > 0).astype(jnp.float32), s2, s2)

    # ── scaled models: a GPT transformer block (causal) and a ResNet block ───
    def gpt_block(z, Wq, Wk, Wv, Wo, Wf1, Wf2):
        B, T, D = z.shape; H = 2; dh = D // H
        def ln(u):
            m = u.mean(-1, keepdims=True); v = u.var(-1, keepdims=True)
            return (u - m) / jnp.sqrt(v + 1e-5)
        hn = ln(z)
        q = (hn @ Wq).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        k = (hn @ Wk).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        v = (hn @ Wv).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        s = (q @ k.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(dh))
        mask = jnp.tril(jnp.ones((T, T)))
        s = jnp.where(mask[None, None] > 0, s, np.float32(-1e9))
        o = (jax.nn.softmax(s, -1) @ v).transpose(0, 2, 1, 3).reshape(B, T, D) @ Wo
        z = z + o
        return z + jax.nn.gelu(ln(z) @ Wf1) @ Wf2
    D = 8
    gpt_args = [rng.standard_normal((2, 4, D)).astype(np.float32)] + \
        [rng.standard_normal((D, D)).astype(np.float32) for _ in range(4)] + \
        [rng.standard_normal((D, 16)).astype(np.float32), rng.standard_normal((16, D)).astype(np.float32)]
    case("GPT block (causal attention)", gpt_block, *gpt_args)

    def resnet_block(z, w1, w2, g1, b1, g2, b2):
        def bn(u, g, bb):
            m = u.mean((0, 2, 3), keepdims=True); v = u.var((0, 2, 3), keepdims=True)
            return (u - m) / jnp.sqrt(v + 1e-5) * g.reshape(1, -1, 1, 1) + bb.reshape(1, -1, 1, 1)
        def cv(u, w):
            return jax.lax.conv_general_dilated(u, w, (1, 1), "SAME",
                                                dimension_numbers=("NCHW", "OIHW", "NCHW"))
        h = jax.nn.relu(bn(cv(z, w1), g1, b1))
        return jax.nn.relu(z + bn(cv(h, w2), g2, b2))
    C = 4
    rn_args = [rng.standard_normal((2, C, 8, 8)).astype(np.float32),
               rng.standard_normal((C, C, 3, 3)).astype(np.float32),
               rng.standard_normal((C, C, 3, 3)).astype(np.float32)] + \
        [rng.standard_normal((C,)).astype(np.float32) for _ in range(4)]
    case("ResNet block (conv-bn-residual)", resnet_block, *rn_args)

    # ── scatter (.at[].set/add) + KV-cache + autoregressive generation ───────
    grid = rng.standard_normal((2, 5)).astype(np.float32)
    col = rng.standard_normal((2,)).astype(np.float32)
    case("scatter .at[:,2].set", lambda a, v: a.at[:, 2].set(v), grid, col)
    case("scatter .at[:,2].add", lambda a, v: a.at[:, 2].add(v), grid, col)
    case("KV-cache dynamic_update_slice",
         lambda c, u: jax.lax.dynamic_update_slice(c, u, (0, 0, 3, 0)),
         rng.standard_normal((1, 2, 6, 4)).astype(np.float32),
         rng.standard_normal((1, 2, 1, 4)).astype(np.float32))

    # a real small GPT doing greedy autoregressive decoding (grow-by-concat)
    Vv, Dd, Hh = 16, 8, 2
    dh2 = Dd // Hh
    Wemb = jnp.asarray(rng.standard_normal((Vv, Dd)).astype(np.float32))
    Wpos = jnp.asarray(rng.standard_normal((16, Dd)).astype(np.float32))
    Wqk = [jnp.asarray(rng.standard_normal((Dd, Dd)).astype(np.float32)) for _ in range(4)]
    Wff = [jnp.asarray(rng.standard_normal((Dd, 16)).astype(np.float32)),
           jnp.asarray(rng.standard_normal((16, Dd)).astype(np.float32))]

    def _ln(z):
        m = z.mean(-1, keepdims=True); v = z.var(-1, keepdims=True)
        return (z - m) / jnp.sqrt(v + 1e-5)

    def _logits(tokens):
        B, Tt = tokens.shape
        h = Wemb[tokens.astype(jnp.int32)] + Wpos[:Tt]
        hn = _ln(h)
        q = (hn @ Wqk[0]).reshape(B, Tt, Hh, dh2).transpose(0, 2, 1, 3)
        k = (hn @ Wqk[1]).reshape(B, Tt, Hh, dh2).transpose(0, 2, 1, 3)
        v = (hn @ Wqk[2]).reshape(B, Tt, Hh, dh2).transpose(0, 2, 1, 3)
        s = (q @ k.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(dh2))
        mask = jnp.tril(jnp.ones((Tt, Tt)))
        s = jnp.where(mask[None, None] > 0, s, np.float32(-1e9))
        o = (jax.nn.softmax(s, -1) @ v).transpose(0, 2, 1, 3).reshape(B, Tt, Dd) @ Wqk[3]
        h = h + o
        h = h + jax.nn.gelu(_ln(h) @ Wff[0]) @ Wff[1]
        return _ln(h) @ Wemb.T

    def generate(prompt, n_new):
        toks = prompt
        for _ in range(n_new):
            nxt = jnp.argmax(_logits(toks)[:, -1, :], -1).astype(jnp.float32)
            toks = jnp.concatenate([toks, nxt[:, None]], axis=1)
        return toks
    prompt = jnp.asarray(rng.integers(1, Vv, size=(2, 3)).astype(np.float32))
    case("GPT greedy autoregressive (6 tokens)", lambda p: generate(p, 6), prompt)

    print(f"\n{PASS} / {PASS + FAIL} passed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
