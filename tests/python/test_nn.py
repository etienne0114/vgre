#!/usr/bin/env python3
"""
vgre.nn — build and train arbitrary models from pure Python on VGRE's CPU engine.

  1. An MLP learns XOR (needs a hidden nonlinearity) → ~100% accuracy.
  2. A CNN (conv→relu→max-pool→linear) classifies the bright-patch quadrant of an
     8×8 image → high accuracy.

Both run entirely through the in-tree autograd C ABI — no GPU, no PyTorch/JAX.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bindings", "python"))

try:
    import numpy as np
    import vgre
    import vgre.nn as nn
except Exception as e:  # pragma: no cover
    print(f"SKIP: cannot import vgre/numpy ({e})")
    sys.exit(77)

if not vgre.NATIVE_AVAILABLE:
    print("SKIP: libvgre not found")
    sys.exit(77)

rng = np.random.default_rng(0)


def test_mlp_xor() -> bool:
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = [0, 1, 1, 0]
    W1 = nn.tensor(rng.standard_normal((2, 8)) * 0.8, requires_grad=True)
    b1 = nn.tensor(np.zeros(8), requires_grad=True)
    W2 = nn.tensor(rng.standard_normal((8, 2)) * 0.8, requires_grad=True)
    b2 = nn.tensor(np.zeros(2), requires_grad=True)
    opt = nn.AdamW([W1, b1, W2, b2], lr=5e-2, weight_decay=0.0)
    x = nn.tensor(X)
    for _ in range(400):
        opt.zero_grad()
        h = nn.relu(nn.matmul(x, W1) + b1)
        logits = nn.matmul(h, W2) + b2
        loss = nn.softmax_cross_entropy(logits, Y)
        loss.backward()
        opt.step()
    logits = (nn.matmul(nn.relu(nn.matmul(x, W1) + b1), W2) + b2).numpy()
    acc = float(np.mean(np.argmax(logits, axis=1) == np.array(Y)))
    print(f"[mlp] xor accuracy = {acc*100:.0f}%  loss = {loss.item():.4f}")
    return acc == 1.0


def make_batch(B):
    X = rng.standard_normal((B, 1, 8, 8)).astype(np.float32) * 0.05
    y = rng.integers(0, 4, size=B)
    for b in range(B):
        r0, c0 = (y[b] // 2) * 4, (y[b] % 2) * 4
        X[b, 0, r0 + 1:r0 + 3, c0 + 1:c0 + 3] = 1.0
    return X, list(y)


def test_cnn_quadrant() -> bool:
    Wc = nn.tensor(rng.standard_normal((4, 1, 3, 3)) * 0.3, requires_grad=True)
    bc = nn.tensor(np.zeros(4), requires_grad=True)
    Wf = nn.tensor(rng.standard_normal((4 * 4 * 4, 4)) * 0.1, requires_grad=True)
    bf = nn.tensor(np.zeros(4), requires_grad=True)
    opt = nn.AdamW([Wc, bc, Wf, bf], lr=5e-3, weight_decay=0.0)
    B = 16
    for _ in range(300):
        X, y = make_batch(B)
        x = nn.tensor(X)
        opt.zero_grad()
        h = nn.max_pool2d(nn.relu(nn.conv2d(x, Wc, bc, stride=1, pad=1)), 2, 2)   # [B,4,4,4]
        logits = nn.matmul(nn.reshape(h, (B, 4 * 4 * 4)), Wf) + bf
        loss = nn.softmax_cross_entropy(logits, y)
        loss.backward()
        opt.step()
    # eval
    Xe, ye = make_batch(64)
    xe = nn.tensor(Xe)
    h = nn.max_pool2d(nn.relu(nn.conv2d(xe, Wc, bc, stride=1, pad=1)), 2, 2)
    logits = (nn.matmul(nn.reshape(h, (64, 4 * 4 * 4)), Wf) + bf).numpy()
    acc = float(np.mean(np.argmax(logits, axis=1) == np.array(ye)))
    print(f"[cnn] quadrant accuracy = {acc*100:.0f}%")
    return acc > 0.9


def test_module_api() -> bool:
    # Same XOR task, built with the PyTorch-like module API.
    nn.seed(0)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    opt = nn.AdamW(model.parameters(), lr=5e-2, weight_decay=0.0)
    X = nn.tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32))
    Y = [0, 1, 1, 0]
    loss = None
    for _ in range(400):
        opt.zero_grad()
        loss = nn.softmax_cross_entropy(model(X), Y)
        loss.backward()
        opt.step()
    acc = float(np.mean(np.argmax(model(X).numpy(), axis=1) == np.array(Y)))
    print(f"[module] Sequential MLP xor accuracy = {acc*100:.0f}%  loss = {loss.item():.4f}")

    # Checkpoint round-trip: save, load into a fresh model, identical output.
    before = model(X).numpy()
    nn.save(model, "/tmp/vgre_nn_model.safetensors")
    nn.seed(999)
    fresh = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    nn.load(fresh, "/tmp/vgre_nn_model.safetensors")
    after = fresh(X).numpy()
    same = bool(np.max(np.abs(before - after)) < 1e-5)
    print(f"[module] checkpoint round-trip identical = {same}")
    return acc == 1.0 and same


def test_bn_cnn() -> bool:
    # A Sequential conv→BN→relu→pool→flatten→linear CNN with batch norm.
    nn.seed(1)
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, stride=1, pad=1), nn.BatchNorm2d(4), nn.ReLU(),
        nn.MaxPool2d(2), nn.Flatten(), nn.Linear(4 * 4 * 4, 4))
    opt = nn.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    model.train()
    for _ in range(300):
        X, y = make_batch(16)
        opt.zero_grad()
        loss = nn.softmax_cross_entropy(model(nn.tensor(X)), y)
        loss.backward()
        opt.step()
    model.eval()   # BN now uses running statistics
    Xe, ye = make_batch(64)
    logits = model(nn.tensor(Xe)).numpy()
    acc = float(np.mean(np.argmax(logits, axis=1) == np.array(ye)))
    print(f"[bn-cnn] eval accuracy = {acc*100:.0f}%")
    return acc > 0.9


def test_transformer_block() -> bool:
    # A transformer block built from pure-Python vgre.nn ops, trained to memorize
    # a fixed token sequence — proves the framework covers transformers too.
    V, T, D, Hh = 16, 12, 32, 4
    seq = [(i * 5 + 2) % V for i in range(T + 1)]
    ids, tgt = seq[:-1], seq[1:]

    def P(*shape, scale=0.02):
        return nn.tensor(rng.standard_normal(shape) * scale, requires_grad=True)
    E = P(V, D); g1 = nn.tensor(np.ones(D), requires_grad=True)
    Wq, Wk, Wv, Wo = P(D, D), P(D, D), P(D, D), P(D, D)
    g2 = nn.tensor(np.ones(D), requires_grad=True)
    W1, W2 = P(D, 4 * D), P(4 * D, D)
    gf = nn.tensor(np.ones(D), requires_grad=True); head = P(D, V)
    params = [E, g1, Wq, Wk, Wv, Wo, g2, W1, W2, gf, head]
    opt = nn.AdamW(params, lr=3e-3, weight_decay=0.0)

    def forward():
        x = nn.embedding(E, ids)
        h = nn.rms_norm(x, g1)
        q = nn.rope(nn.matmul(h, Wq), Hh); k = nn.rope(nn.matmul(h, Wk), Hh); v = nn.matmul(h, Wv)
        a = nn.attention(q, k, v, Hh, causal=True)
        x = x + nn.matmul(a, Wo)
        h2 = nn.rms_norm(x, g2)
        x = x + nn.matmul(nn.relu(nn.matmul(h2, W1)), W2)
        return nn.matmul(nn.rms_norm(x, gf), head)

    first = last = None
    for _ in range(250):
        opt.zero_grad()
        loss = nn.softmax_cross_entropy(forward(), tgt)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    print(f"[transformer] loss {first:.3f} -> {last:.3f}")
    return last < 0.05


def test_gpt_module() -> bool:
    # A small GPT assembled from vgre.nn modules — the ergonomic transformer path.
    nn.seed(3)
    V, T, D = 16, 12, 32
    seq = [(i * 5 + 2) % V for i in range(T + 1)]
    ids, tgt = seq[:-1], seq[1:]

    class GPT(nn.Module):
        def __init__(self):
            self.embed = nn.Embedding(V, D)
            self.blocks = [nn.TransformerBlock(D, num_heads=4) for _ in range(2)]
            self.norm = nn.RMSNorm(D)
            self.head = nn.Linear(D, V, bias=False)

        def forward(self, ids):
            x = self.embed(ids)
            for b in self.blocks:
                x = b(x)
            return self.head(self.norm(x))

    model = GPT()
    opt = nn.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    first = last = None
    for _ in range(250):
        opt.zero_grad()
        loss = nn.softmax_cross_entropy(model(ids), tgt)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    nparams = sum(t.size for _, t in model.named_parameters())
    print(f"[gpt-module] {nparams} params, loss {first:.3f} -> {last:.3f}")
    return last < 0.05


def test_dropout_and_tied() -> bool:
    x = nn.tensor(np.ones((4, 8), np.float32))
    d = nn.Dropout(0.5)
    d.train()
    out_train = d(x).numpy()
    d.eval()
    out_eval = d(x).numpy()
    # Train mode drops ~half the units (and scales survivors by 2); eval = identity.
    frac_zero = float(np.mean(out_train == 0.0))
    eval_identity = bool(np.allclose(out_eval, 1.0))
    # linear_tied: x[2,3] · Wᵀ where W is [4,3] -> [2,4]
    W = nn.tensor(np.arange(12, dtype=np.float32).reshape(4, 3))
    xt = nn.tensor(np.ones((2, 3), np.float32))
    tied = nn.linear_tied(xt, W).numpy()
    tied_ok = tied.shape == (2, 4) and np.allclose(tied[0], W.numpy().sum(axis=1))
    print(f"[dropout] train zero-frac={frac_zero:.2f} eval-identity={eval_identity}  linear_tied={tied_ok}")
    return 0.2 < frac_zero < 0.8 and eval_identity and tied_ok


def test_leaky_relu() -> bool:
    # leaky_relu is composed from existing ops (relu/scale/add), not a new C
    # kernel, so this checks both that the composition is forward-correct and
    # that autograd differentiates through the composition correctly.
    slope = 0.1
    rng2 = np.random.default_rng(7)
    x_np = rng2.standard_normal(64).astype(np.float32)
    x_np[np.abs(x_np) < 1e-2] += 0.5   # keep clear of the x=0 kink
    x = nn.tensor(x_np, requires_grad=True)
    y = nn.leaky_relu(x, slope)
    ref = np.where(x_np > 0, x_np, slope * x_np)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    w_np = rng2.standard_normal(64).astype(np.float32)
    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    grad_ref = np.where(x_np > 0, 1.0, slope) * w_np / x_np.size
    grad_err = float(np.max(np.abs(x.grad() - grad_ref)))

    mod = nn.LeakyReLU(slope)
    mod_ok = bool(np.allclose(mod(nn.tensor(x_np)).numpy(), ref))

    print(f"[leaky_relu] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} module_ok={mod_ok}")
    return fwd_err < 1e-6 and grad_err < 1e-5 and mod_ok


def test_activation_modules() -> bool:
    # sigmoid/tanh/silu/gelu/softplus already exist as functional ops (used
    # directly inside SSMBlock/TransformerBlock) but had no Module wrapper like
    # ReLU/LeakyReLU; this checks the new Sigmoid/Tanh/SiLU/GELU/Softplus
    # modules against a from-scratch NumPy/math reference for both forward and
    # gradient.
    import math
    rng2 = np.random.default_rng(11)
    x_np = rng2.standard_normal(64).astype(np.float32)
    w_np = rng2.standard_normal(64).astype(np.float32)

    def sigmoid_ref(a): return 1.0 / (1.0 + np.exp(-a))
    erf = np.vectorize(math.erf)
    def phi_ref(a): return 0.5 * (1.0 + erf(a / math.sqrt(2.0)))            # Phi(x)
    def pdf_ref(a): return np.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)  # phi(x)

    cases = {
        "sigmoid": (nn.Sigmoid(), nn.sigmoid, sigmoid_ref,
                    lambda a: sigmoid_ref(a) * (1.0 - sigmoid_ref(a))),
        "tanh":    (nn.Tanh(), nn.tanh, np.tanh,
                    lambda a: 1.0 - np.tanh(a) ** 2),
        "silu":    (nn.SiLU(), nn.silu, lambda a: a * sigmoid_ref(a),
                    lambda a: sigmoid_ref(a) + a * sigmoid_ref(a) * (1.0 - sigmoid_ref(a))),
        "gelu":    (nn.GELU(), nn.gelu, lambda a: a * phi_ref(a),
                    lambda a: phi_ref(a) + a * pdf_ref(a)),
        "softplus": (nn.Softplus(), nn.softplus, lambda a: np.logaddexp(0.0, a),
                     lambda a: sigmoid_ref(a)),
    }

    ok = True
    for name, (mod, fn, fwd_ref, grad_ref) in cases.items():
        ref = fwd_ref(x_np.astype(np.float64)).astype(np.float32)
        x = nn.tensor(x_np, requires_grad=True)
        y = fn(x)
        fwd_err = float(np.max(np.abs(y.numpy() - ref)))

        nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
        g_ref = (grad_ref(x_np.astype(np.float64)).astype(np.float32) * w_np) / x_np.size
        grad_err = float(np.max(np.abs(x.grad() - g_ref)))

        mod_ok = bool(np.allclose(mod(nn.tensor(x_np)).numpy(), ref, atol=1e-5))
        print(f"[{name}] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} module_ok={mod_ok}")
        ok = ok and fwd_err < 1e-5 and grad_err < 1e-4 and mod_ok
    return ok


def test_mish() -> bool:
    # mish(x) = x*tanh(softplus(x)) is composed from existing ops (mul/tanh/
    # softplus), not a new C kernel, so this checks both forward correctness
    # against a from-scratch NumPy reference and that autograd differentiates
    # through the composition correctly.
    rng2 = np.random.default_rng(13)
    x_np = rng2.standard_normal(64).astype(np.float32)
    w_np = rng2.standard_normal(64).astype(np.float32)

    def softplus_ref(a): return np.logaddexp(0.0, a)
    def mish_ref(a): return a * np.tanh(softplus_ref(a))

    x = nn.tensor(x_np, requires_grad=True)
    y = nn.mish(x)
    ref = mish_ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    # d/dx [x*tanh(softplus(x))] = tanh(sp) + x*sigmoid(x)*(1-tanh(sp)^2)
    sp = softplus_ref(x_np.astype(np.float64))
    t = np.tanh(sp)
    sig = 1.0 / (1.0 + np.exp(-x_np.astype(np.float64)))
    grad_ref = (t + x_np.astype(np.float64) * sig * (1.0 - t * t)).astype(np.float32)
    grad_err = float(np.max(np.abs(x.grad() - (grad_ref * w_np) / x_np.size)))

    mod_ok = bool(np.allclose(nn.Mish()(nn.tensor(x_np)).numpy(), ref, atol=1e-5))

    print(f"[mish] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} module_ok={mod_ok}")
    return fwd_err < 1e-5 and grad_err < 1e-4 and mod_ok


def test_log_sigmoid() -> bool:
    # log_sigmoid(x) = -softplus(-x) is composed from existing ops (scale/
    # softplus), not a new C kernel, so this checks both forward correctness
    # against a from-scratch NumPy reference (the numerically-stable
    # -logaddexp(0,-x) form) and that autograd differentiates through the
    # composition correctly.
    rng2 = np.random.default_rng(17)
    x_np = rng2.standard_normal(64).astype(np.float32) * 5.0  # wide range incl. large |x|
    w_np = rng2.standard_normal(64).astype(np.float32)

    def log_sigmoid_ref(a): return -np.logaddexp(0.0, -a)

    x = nn.tensor(x_np, requires_grad=True)
    y = nn.log_sigmoid(x)
    ref = log_sigmoid_ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    # d/dx log(sigmoid(x)) = sigmoid(-x) = 1 - sigmoid(x)
    grad_ref = (1.0 / (1.0 + np.exp(x_np.astype(np.float64)))).astype(np.float32)
    grad_err = float(np.max(np.abs(x.grad() - (grad_ref * w_np) / x_np.size)))

    mod_ok = bool(np.allclose(nn.LogSigmoid()(nn.tensor(x_np)).numpy(), ref, atol=1e-5))
    print(f"[log_sigmoid] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} module_ok={mod_ok}")
    return fwd_err < 1e-5 and grad_err < 1e-4 and mod_ok


def test_elu() -> bool:
    # elu(x) = x for x>0, alpha*(exp(x)-1) for x<=0 is composed from existing
    # ops (relu/exp/scale/add), not a new C kernel, so this checks both
    # forward correctness against a from-scratch NumPy reference and that
    # autograd differentiates through the composition correctly, including
    # at alpha != 1 and across the x=0 boundary.
    rng2 = np.random.default_rng(19)
    x_np = rng2.standard_normal(64).astype(np.float32) * 3.0  # wide range incl. large |x|
    w_np = rng2.standard_normal(64).astype(np.float32)
    alpha = 1.5

    def elu_ref(a): return np.where(a > 0, a, alpha * (np.exp(a) - 1.0))
    def elu_grad_ref(a): return np.where(a > 0, 1.0, alpha * np.exp(a))

    x = nn.tensor(x_np, requires_grad=True)
    y = nn.elu(x, alpha)
    ref = elu_ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    grad_ref = elu_grad_ref(x_np.astype(np.float64)).astype(np.float32)
    grad_err = float(np.max(np.abs(x.grad() - (grad_ref * w_np) / x_np.size)))

    mod_ok = bool(np.allclose(nn.ELU(alpha)(nn.tensor(x_np)).numpy(), ref, atol=1e-5))
    print(f"[elu] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} module_ok={mod_ok}")
    return fwd_err < 1e-5 and grad_err < 1e-4 and mod_ok


def test_celu() -> bool:
    # celu(x, alpha) = x for x>0, alpha*(exp(x/alpha)-1) for x<=0 is composed
    # from existing ops (relu/exp/scale/add), not a new C kernel, so this
    # checks both forward correctness against a from-scratch NumPy reference
    # and that autograd differentiates through the composition correctly,
    # including at alpha != 1 (where CELU diverges from ELU) and across the
    # x=0 boundary.
    rng2 = np.random.default_rng(29)
    x_np = rng2.standard_normal(64).astype(np.float32) * 3.0  # wide range incl. large |x|
    w_np = rng2.standard_normal(64).astype(np.float32)
    alpha = 2.0

    def celu_ref(a): return np.where(a > 0, a, alpha * (np.exp(a / alpha) - 1.0))
    def celu_grad_ref(a): return np.where(a > 0, 1.0, np.exp(a / alpha))

    x = nn.tensor(x_np, requires_grad=True)
    y = nn.celu(x, alpha)
    ref = celu_ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    grad_ref = celu_grad_ref(x_np.astype(np.float64)).astype(np.float32)
    grad_err = float(np.max(np.abs(x.grad() - (grad_ref * w_np) / x_np.size)))

    mod_ok = bool(np.allclose(nn.CELU(alpha)(nn.tensor(x_np)).numpy(), ref, atol=1e-5))
    # alpha=1 must exactly match the existing ELU implementation.
    elu_match = bool(np.allclose(nn.celu(x, 1.0).numpy(), nn.elu(x, 1.0).numpy(), atol=1e-6))
    print(f"[celu] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} module_ok={mod_ok} "
          f"elu_match={elu_match}")
    return fwd_err < 1e-5 and grad_err < 1e-4 and mod_ok and elu_match


def test_hardswish() -> bool:
    # hardsigmoid(x) = clip((x+3)/6, 0, 1) and hardswish(x) = x*hardsigmoid(x)
    # are composed from existing ops (relu/add/scale/mul), not a new C
    # kernel, so this checks both forward correctness against a from-scratch
    # NumPy reference and that autograd differentiates through the
    # composition correctly, including at the x=-3 and x=3 kinks.
    rng2 = np.random.default_rng(23)
    x_np = rng2.standard_normal(64).astype(np.float32) * 3.0  # wide range incl. |x|>3
    w_np = rng2.standard_normal(64).astype(np.float32)

    def hsig_ref(a): return np.clip((a + 3.0) / 6.0, 0.0, 1.0)
    def hswish_ref(a): return a * hsig_ref(a)
    def hswish_grad_ref(a):
        # hsig is piecewise-linear, but hswish = x*hsig(x) is piecewise-quadratic
        # on (-3,3): d/dx = hsig(x) + x/6 = x/3 + 0.5, which is NOT bounded by
        # [0,1] (it reaches 1.5 at x->3-), with a genuine kink down to 1 at x=3.
        return np.where(a <= -3.0, 0.0, np.where(a >= 3.0, 1.0, a / 3.0 + 0.5))

    x = nn.tensor(x_np, requires_grad=True)
    y = nn.hardswish(x)
    ref = hswish_ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    grad_ref = hswish_grad_ref(x_np.astype(np.float64)).astype(np.float32)
    grad_err = float(np.max(np.abs(x.grad() - (grad_ref * w_np) / x_np.size)))

    hsig_ok = bool(np.allclose(nn.Hardsigmoid()(nn.tensor(x_np)).numpy(),
                                hsig_ref(x_np), atol=1e-5))
    mod_ok = bool(np.allclose(nn.Hardswish()(nn.tensor(x_np)).numpy(), ref, atol=1e-5))
    print(f"[hardswish] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} "
          f"hardsigmoid_ok={hsig_ok} module_ok={mod_ok}")
    return fwd_err < 1e-5 and grad_err < 1e-4 and hsig_ok and mod_ok


def test_relu6() -> bool:
    # relu6(x) = min(relu(x), 6) is composed from existing ops (relu/add/scale),
    # not a new C kernel, so this checks both forward correctness against a
    # from-scratch NumPy reference and that autograd differentiates through
    # the composition correctly, including at the x=0 and x=6 kinks.
    rng2 = np.random.default_rng(31)
    x_np = rng2.standard_normal(64).astype(np.float32) * 5.0  # wide range incl. |x|>6
    w_np = rng2.standard_normal(64).astype(np.float32)

    def relu6_ref(a): return np.clip(a, 0.0, 6.0)
    def relu6_grad_ref(a): return np.where((a > 0.0) & (a < 6.0), 1.0, 0.0)

    x = nn.tensor(x_np, requires_grad=True)
    y = nn.relu6(x)
    ref = relu6_ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    grad_ref = relu6_grad_ref(x_np.astype(np.float64)).astype(np.float32)
    grad_err = float(np.max(np.abs(x.grad() - (grad_ref * w_np) / x_np.size)))

    mod_ok = bool(np.allclose(nn.ReLU6()(nn.tensor(x_np)).numpy(), ref, atol=1e-5))
    print(f"[relu6] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} module_ok={mod_ok}")
    return fwd_err < 1e-5 and grad_err < 1e-4 and mod_ok


def test_softshrink() -> bool:
    # softshrink(x, lambd) = x-lambd for x>lambd, x+lambd for x<-lambd, else 0
    # is composed from existing ops (relu/scale/add), not a new C kernel, so
    # this checks both forward correctness against a from-scratch NumPy
    # reference and that autograd differentiates through the composition
    # correctly, including across the dead zone at the x=-lambd/x=lambd kinks.
    rng2 = np.random.default_rng(37)
    x_np = rng2.standard_normal(64).astype(np.float32) * 2.0  # wide range incl. |x|>lambd
    w_np = rng2.standard_normal(64).astype(np.float32)
    lambd = 0.5

    def softshrink_ref(a):
        return np.where(a > lambd, a - lambd, np.where(a < -lambd, a + lambd, 0.0))
    def softshrink_grad_ref(a):
        return np.where((a > lambd) | (a < -lambd), 1.0, 0.0)

    x = nn.tensor(x_np, requires_grad=True)
    y = nn.softshrink(x, lambd)
    ref = softshrink_ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    grad_ref = softshrink_grad_ref(x_np.astype(np.float64)).astype(np.float32)
    grad_err = float(np.max(np.abs(x.grad() - (grad_ref * w_np) / x_np.size)))

    mod_ok = bool(np.allclose(nn.Softshrink(lambd)(nn.tensor(x_np)).numpy(), ref, atol=1e-5))
    zero_ok = bool(np.allclose(nn.softshrink(x, 0.0).numpy(), x_np, atol=1e-6))
    print(f"[softshrink] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} module_ok={mod_ok} "
          f"zero_ok={zero_ok}")
    return fwd_err < 1e-5 and grad_err < 1e-4 and mod_ok and zero_ok


def test_tanhshrink() -> bool:
    # tanhshrink(x) = x - tanh(x) is composed from existing ops (tanh/add/scale),
    # not a new C kernel, so this checks both forward correctness against a
    # from-scratch NumPy reference and that autograd differentiates through the
    # composition correctly (d/dx = tanh(x)^2, i.e. 1 - sech(x)^2).
    rng2 = np.random.default_rng(41)
    x_np = rng2.standard_normal(64).astype(np.float32) * 2.0
    w_np = rng2.standard_normal(64).astype(np.float32)

    def tanhshrink_ref(a):
        return a - np.tanh(a)
    def tanhshrink_grad_ref(a):
        return np.tanh(a) ** 2

    x = nn.tensor(x_np, requires_grad=True)
    y = nn.tanhshrink(x)
    ref = tanhshrink_ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    grad_ref = tanhshrink_grad_ref(x_np.astype(np.float64)).astype(np.float32)
    grad_err = float(np.max(np.abs(x.grad() - (grad_ref * w_np) / x_np.size)))

    mod_ok = bool(np.allclose(nn.Tanhshrink()(nn.tensor(x_np)).numpy(), ref, atol=1e-5))
    zero_ok = bool(np.allclose(nn.tanhshrink(nn.tensor(np.zeros(4, dtype=np.float32))).numpy(),
                                np.zeros(4), atol=1e-6))
    print(f"[tanhshrink] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} module_ok={mod_ok} "
          f"zero_ok={zero_ok}")
    return fwd_err < 1e-5 and grad_err < 1e-4 and mod_ok and zero_ok


def test_hardtanh() -> bool:
    # hardtanh(x, min_val, max_val) = clamp(x, min_val, max_val) is composed
    # from existing ops (relu/add/scale, the same max(a,b)==b+relu(a-b) trick
    # relu6 uses twice), not a new C kernel, so this checks both forward
    # correctness against a from-scratch NumPy reference and that autograd
    # differentiates through the composition correctly, including at the
    # min_val/max_val kinks, plus that relu6 is the min_val=0/max_val=6 case.
    rng2 = np.random.default_rng(43)
    x_np = rng2.standard_normal(64).astype(np.float32) * 3.0  # wide range incl. |x|>1
    w_np = rng2.standard_normal(64).astype(np.float32)
    min_val, max_val = -2.0, 1.5

    def hardtanh_ref(a): return np.clip(a, min_val, max_val)
    def hardtanh_grad_ref(a): return np.where((a > min_val) & (a < max_val), 1.0, 0.0)

    x = nn.tensor(x_np, requires_grad=True)
    y = nn.hardtanh(x, min_val, max_val)
    ref = hardtanh_ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    grad_ref = hardtanh_grad_ref(x_np.astype(np.float64)).astype(np.float32)
    grad_err = float(np.max(np.abs(x.grad() - (grad_ref * w_np) / x_np.size)))

    mod_ok = bool(np.allclose(nn.Hardtanh(min_val, max_val)(nn.tensor(x_np)).numpy(), ref, atol=1e-5))
    relu6_match = bool(np.allclose(nn.hardtanh(x, 0.0, 6.0).numpy(), nn.relu6(x).numpy(), atol=1e-6))
    print(f"[hardtanh] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} module_ok={mod_ok} "
          f"relu6_match={relu6_match}")
    return fwd_err < 1e-5 and grad_err < 1e-4 and mod_ok and relu6_match


def test_huber_loss() -> bool:
    # huber_loss(pred, target, delta) = 0.5*e^2 for |e|<=delta,
    # delta*(|e|-0.5*delta) for |e|>delta, composed from existing ops
    # (relu/hardtanh/mul/add/scale/mean, no new C kernel, no division), so
    # this checks both forward correctness against a from-scratch NumPy
    # reference and that autograd differentiates through the composition
    # correctly, including across the delta kink and both reduction modes.
    rng2 = np.random.default_rng(59)
    pred_np = rng2.standard_normal(64).astype(np.float32) * 2.0
    target_np = rng2.standard_normal(64).astype(np.float32)
    delta = 0.75

    def huber_ref(p, t):
        e = p - t
        ae = np.abs(e)
        return np.where(ae <= delta, 0.5 * e * e, delta * (ae - 0.5 * delta))
    def huber_grad_ref(p, t):
        e = p - t
        return np.clip(e, -delta, delta)

    pred = nn.tensor(pred_np, requires_grad=True)
    target = nn.tensor(target_np)
    y = nn.huber_loss(pred, target, delta, reduction="none")
    ref = huber_ref(pred_np.astype(np.float64), target_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    loss = nn.huber_loss(pred, target, delta, reduction="mean")
    mean_ok = bool(np.allclose(loss.numpy(), np.mean(ref), atol=1e-5))
    loss.backward()
    grad_ref = huber_grad_ref(pred_np.astype(np.float64), target_np.astype(np.float64)).astype(np.float32)
    grad_err = float(np.max(np.abs(pred.grad() - grad_ref / pred_np.size)))

    sum_ok = bool(np.allclose(nn.huber_loss(pred, target, delta, reduction="sum").numpy(),
                               np.sum(ref), atol=1e-3))
    print(f"[huber_loss] fwd_err={fwd_err:.1e} mean_ok={mean_ok} grad_err={grad_err:.1e} sum_ok={sum_ok}")
    return fwd_err < 1e-5 and mean_ok and grad_err < 1e-4 and sum_ok


def test_mse_loss() -> bool:
    # mse_loss(pred, target) = (pred-target)^2, composed from existing ops
    # (add/scale/mul/mean, no new C kernel), so this checks both forward
    # correctness against a from-scratch NumPy reference and that autograd
    # differentiates through the composition correctly, across all three
    # reduction modes.
    rng2 = np.random.default_rng(61)
    pred_np = rng2.standard_normal(64).astype(np.float32) * 2.0
    target_np = rng2.standard_normal(64).astype(np.float32)

    def mse_ref(p, t):
        e = p - t
        return e * e
    def mse_grad_ref(p, t):
        return 2.0 * (p - t)

    pred = nn.tensor(pred_np, requires_grad=True)
    target = nn.tensor(target_np)
    y = nn.mse_loss(pred, target, reduction="none")
    ref = mse_ref(pred_np.astype(np.float64), target_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    loss = nn.mse_loss(pred, target, reduction="mean")
    mean_ok = bool(np.allclose(loss.numpy(), np.mean(ref), atol=1e-5))
    loss.backward()
    grad_ref = mse_grad_ref(pred_np.astype(np.float64), target_np.astype(np.float64)).astype(np.float32)
    grad_err = float(np.max(np.abs(pred.grad() - grad_ref / pred_np.size)))

    sum_ok = bool(np.allclose(nn.mse_loss(pred, target, reduction="sum").numpy(),
                               np.sum(ref), atol=1e-3))
    print(f"[mse_loss] fwd_err={fwd_err:.1e} mean_ok={mean_ok} grad_err={grad_err:.1e} sum_ok={sum_ok}")
    return fwd_err < 1e-5 and mean_ok and grad_err < 1e-4 and sum_ok


def test_l1_loss() -> bool:
    # l1_loss(pred, target) = |pred-target|, composed from existing ops
    # (add/scale/relu/mean, no new C kernel), so this checks both forward
    # correctness against a from-scratch NumPy reference and that autograd
    # differentiates through the composition correctly, across all three
    # reduction modes. The gradient is the sign of the error, undefined at
    # e==0, so the random inputs avoid exact ties.
    rng2 = np.random.default_rng(71)
    pred_np = rng2.standard_normal(64).astype(np.float32) * 2.0
    target_np = rng2.standard_normal(64).astype(np.float32)

    def l1_ref(p, t):
        return np.abs(p - t)
    def l1_grad_ref(p, t):
        return np.sign(p - t)

    pred = nn.tensor(pred_np, requires_grad=True)
    target = nn.tensor(target_np)
    y = nn.l1_loss(pred, target, reduction="none")
    ref = l1_ref(pred_np.astype(np.float64), target_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    loss = nn.l1_loss(pred, target, reduction="mean")
    mean_ok = bool(np.allclose(loss.numpy(), np.mean(ref), atol=1e-5))
    loss.backward()
    grad_ref = l1_grad_ref(pred_np.astype(np.float64), target_np.astype(np.float64)).astype(np.float32)
    grad_err = float(np.max(np.abs(pred.grad() - grad_ref / pred_np.size)))

    sum_ok = bool(np.allclose(nn.l1_loss(pred, target, reduction="sum").numpy(),
                               np.sum(ref), atol=1e-3))
    print(f"[l1_loss] fwd_err={fwd_err:.1e} mean_ok={mean_ok} grad_err={grad_err:.1e} sum_ok={sum_ok}")
    return fwd_err < 1e-5 and mean_ok and grad_err < 1e-4 and sum_ok


def test_bce_with_logits_loss() -> bool:
    # binary_cross_entropy_with_logits(x,t) = -[t*log(sigmoid(x)) +
    # (1-t)*log(sigmoid(-x))], composed from existing ops (log_sigmoid/
    # add/scale/mul/mean, no new C kernel). Checks forward against the
    # standard numerically-stable NumPy reference
    # max(x,0) - x*t + log(1+exp(-|x|)), the gradient against the analytic
    # sigmoid(x)-t formula, all three reduction modes, and that soft targets
    # (not just 0/1) work since nothing here assumes binary targets.
    rng2 = np.random.default_rng(81)
    pred_np = rng2.standard_normal(64).astype(np.float32) * 3.0
    target_np = rng2.uniform(0.0, 1.0, size=64).astype(np.float32)

    def bce_ref(x, t):
        return np.maximum(x, 0) - x * t + np.log1p(np.exp(-np.abs(x)))
    def bce_grad_ref(x, t):
        return 1.0 / (1.0 + np.exp(-x)) - t

    pred = nn.tensor(pred_np, requires_grad=True)
    target = nn.tensor(target_np)
    y = nn.binary_cross_entropy_with_logits(pred, target, reduction="none")
    ref = bce_ref(pred_np.astype(np.float64), target_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    loss = nn.binary_cross_entropy_with_logits(pred, target, reduction="mean")
    mean_ok = bool(np.allclose(loss.numpy(), np.mean(ref), atol=1e-5))
    loss.backward()
    grad_ref = bce_grad_ref(pred_np.astype(np.float64), target_np.astype(np.float64)).astype(np.float32)
    grad_err = float(np.max(np.abs(pred.grad() - grad_ref / pred_np.size)))

    sum_ok = bool(np.allclose(nn.binary_cross_entropy_with_logits(pred, target, reduction="sum").numpy(),
                               np.sum(ref), atol=1e-3))
    print(f"[bce_with_logits_loss] fwd_err={fwd_err:.1e} mean_ok={mean_ok} grad_err={grad_err:.1e} sum_ok={sum_ok}")
    return fwd_err < 1e-4 and mean_ok and grad_err < 1e-4 and sum_ok


def test_label_smoothing_cross_entropy() -> bool:
    # label_smoothing_cross_entropy spreads `smoothing` mass off the true
    # class uniformly over all V classes, then reuses softmax_cross_entropy_soft.
    # Checks: forward matches a from-scratch NumPy CE-with-smoothing reference,
    # the gradient matches the standard (softmax(logits)-soft_target)/M formula,
    # and smoothing=0 collapses to ordinary softmax_cross_entropy exactly.
    rng3 = np.random.default_rng(37)
    m, v = 5, 6
    logits_np = rng3.standard_normal((m, v)).astype(np.float32) * 1.5
    targets = [2, 0, 5, 3, 1]
    smoothing = 0.1

    def smoothed_ref(z, tgts, eps):
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        t = np.full_like(z, eps / v)
        t[np.arange(m), tgts] += 1.0 - eps
        loss = -np.sum(t * np.log(p)) / m
        grad = (p - t) / m
        return loss, grad

    logits = nn.tensor(logits_np, requires_grad=True)
    loss = nn.label_smoothing_cross_entropy(logits, targets, smoothing=smoothing)
    ref_loss, ref_grad = smoothed_ref(logits_np.astype(np.float64), targets, smoothing)
    fwd_err = abs(float(loss.item()) - ref_loss)
    loss.backward()
    grad_err = float(np.max(np.abs(logits.grad() - ref_grad.astype(np.float32))))

    logits0 = nn.tensor(logits_np, requires_grad=True)
    hard = nn.softmax_cross_entropy(logits0, targets)
    smooth0 = nn.label_smoothing_cross_entropy(logits0, targets, smoothing=0.0)
    zero_smoothing_ok = bool(np.allclose(hard.numpy(), smooth0.numpy(), atol=1e-5))

    print(f"[label_smoothing_ce] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} "
          f"zero_smoothing_ok={zero_smoothing_ok}")
    return fwd_err < 1e-4 and grad_err < 1e-5 and zero_smoothing_ok


def test_kl_div_loss() -> bool:
    # kl_div_loss(logits, target_probs) = KL(target_probs || softmax(logits)),
    # built as softmax_cross_entropy_soft(logits, target_probs) minus the
    # (constant, gradient-free) target entropy computed in NumPy. Checks:
    # forward matches a from-scratch NumPy KL reference, the gradient matches
    # the analytic (softmax(logits)-target_probs)/M formula (identical to
    # softmax_cross_entropy_soft's, since the entropy offset is constant),
    # "sum" reduction is M times "mean", and target_probs==softmax(logits)
    # (self-KL) is ~0.
    rng4 = np.random.default_rng(83)
    m, v = 5, 6
    logits_np = rng4.standard_normal((m, v)).astype(np.float32) * 1.5
    raw = rng4.uniform(0.1, 1.0, size=(m, v)).astype(np.float64)
    target_np = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)

    def kl_ref(z, t):
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        kl = np.sum(t * (np.log(np.clip(t, 1e-12, 1.0)) - np.log(p)), axis=1)
        grad = (p - t) / z.shape[0]
        return float(kl.mean()), grad

    logits = nn.tensor(logits_np, requires_grad=True)
    loss = nn.kl_div_loss(logits, target_np, reduction="mean")
    ref_loss, ref_grad = kl_ref(logits_np.astype(np.float64), target_np.astype(np.float64))
    fwd_err = abs(float(loss.item()) - ref_loss)
    loss.backward()
    grad_err = float(np.max(np.abs(logits.grad() - ref_grad.astype(np.float32))))

    sum_ok = bool(np.allclose(nn.kl_div_loss(logits, target_np, reduction="sum").numpy(),
                               ref_loss * m, atol=1e-3))

    z0 = rng4.standard_normal((3, 4)).astype(np.float32)
    p0 = np.exp(z0 - z0.max(1, keepdims=True))
    p0 /= p0.sum(1, keepdims=True)
    self_kl = float(nn.kl_div_loss(nn.tensor(z0), p0, reduction="mean").item())
    self_ok = abs(self_kl) < 1e-4

    print(f"[kl_div_loss] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} sum_ok={sum_ok} self_ok={self_ok}")
    return fwd_err < 1e-4 and grad_err < 1e-5 and sum_ok and self_ok


def test_selu() -> bool:
    # SELU(x) = scale*(x for x>0, alpha*(exp(x)-1) for x<=0) with the fixed
    # self-normalizing constants (Klambauer et al.), composed from existing
    # ops (elu/scale), not a new C kernel, so this checks both forward
    # correctness against a from-scratch NumPy reference and that autograd
    # differentiates through the composition correctly, plus that it is
    # exactly elu(x, alpha) rescaled by the fixed SELU scale constant.
    rng2 = np.random.default_rng(53)
    x_np = rng2.standard_normal(64).astype(np.float32) * 2.0  # wide range incl. x<=0
    w_np = rng2.standard_normal(64).astype(np.float32)
    alpha, scale_ = 1.6732632423543772, 1.0507009873554804

    def selu_ref(a):
        return scale_ * np.where(a > 0, a, alpha * (np.exp(a) - 1.0))
    def selu_grad_ref(a):
        return scale_ * np.where(a > 0, 1.0, alpha * np.exp(a))

    x = nn.tensor(x_np, requires_grad=True)
    y = nn.selu(x)
    ref = selu_ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    grad_ref = selu_grad_ref(x_np.astype(np.float64)).astype(np.float32)
    grad_err = float(np.max(np.abs(x.grad() - (grad_ref * w_np) / x_np.size)))

    mod_ok = bool(np.allclose(nn.SELU()(nn.tensor(x_np)).numpy(), ref, atol=1e-5))
    elu_match = bool(np.allclose(nn.selu(x).numpy(), scale_ * nn.elu(x, alpha).numpy(), atol=1e-6))
    print(f"[selu] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} module_ok={mod_ok} "
          f"elu_match={elu_match}")
    return fwd_err < 1e-5 and grad_err < 1e-4 and mod_ok and elu_match


def test_softmin() -> bool:
    # Softmin(x) = softmax(-x) over the last dim, composed from existing ops
    # (scale/softmax), not a new C kernel, so this checks both forward
    # correctness against a from-scratch NumPy reference (row-wise, one row
    # per batch element) and that autograd differentiates through the
    # composition correctly, using the same softmax jacobian-vector-product
    # formula softmax_cross_entropy is built on.
    rng2 = np.random.default_rng(59)
    M, N = 5, 7
    x_np = rng2.standard_normal((M, N)).astype(np.float32) * 2.0
    w_np = rng2.standard_normal((M, N)).astype(np.float32)

    def softmin_ref(a):
        z = -a
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def softmin_grad_ref(a, w):
        p = softmin_ref(a)
        dot = (p * w).sum(axis=-1, keepdims=True)
        dz = p * (w - dot)  # dL/dz where z = -x
        return -dz  # dL/dx = -dL/dz

    x = nn.tensor(x_np, requires_grad=True)
    y = nn.softmin(x)
    ref = softmin_ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))
    sum_ok = bool(np.allclose(y.numpy().sum(axis=-1), 1.0, atol=1e-5))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    grad_ref = softmin_grad_ref(x_np.astype(np.float64), w_np.astype(np.float64))
    grad_err = float(np.max(np.abs(x.grad() - (grad_ref.astype(np.float32) / x_np.size))))

    mod_ok = bool(np.allclose(nn.Softmin()(nn.tensor(x_np)).numpy(), ref, atol=1e-5))
    print(f"[softmin] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} sum_ok={sum_ok} "
          f"module_ok={mod_ok}")
    return fwd_err < 1e-5 and grad_err < 1e-4 and sum_ok and mod_ok


def test_gaussian() -> bool:
    # Gaussian(x) = exp(-x^2), composed from existing ops (mul/scale/exp), not
    # a new C kernel, so this checks forward correctness against a
    # from-scratch NumPy reference, that autograd differentiates through the
    # composition correctly, and that the output is bounded in (0,1] and
    # peaks at x=0 (the defining bell-curve shape).
    rng2 = np.random.default_rng(61)
    x_np = rng2.standard_normal(64).astype(np.float32) * 2.0
    w_np = rng2.standard_normal(64).astype(np.float32)

    def gaussian_ref(a): return np.exp(-a * a)
    def gaussian_grad_ref(a): return -2.0 * a * np.exp(-a * a)

    x = nn.tensor(x_np, requires_grad=True)
    y = nn.gaussian(x)
    ref = gaussian_ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref)))
    range_ok = bool(np.all(y.numpy() > 0.0) and np.all(y.numpy() <= 1.0 + 1e-6))
    peak_ok = bool(np.isclose(nn.gaussian(nn.tensor(np.zeros(1, np.float32))).numpy()[0], 1.0, atol=1e-6))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    grad_ref = gaussian_grad_ref(x_np.astype(np.float64))
    grad_err = float(np.max(np.abs(x.grad() - (grad_ref.astype(np.float32) * w_np) / x_np.size)))

    mod_ok = bool(np.allclose(nn.Gaussian()(nn.tensor(x_np)).numpy(), ref, atol=1e-5))
    print(f"[gaussian] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} range_ok={range_ok} "
          f"peak_ok={peak_ok} module_ok={mod_ok}")
    return fwd_err < 1e-5 and grad_err < 1e-4 and range_ok and peak_ok and mod_ok


def test_swiglu_mlp() -> bool:
    # SwiGLUMLP: down(silu(gate(x)) * up(x)), the Llama/Mistral/Qwen gated FFN,
    # composed entirely from existing ops (Linear/silu/mul) -- no new C kernel.
    # Checks forward correctness against a from-scratch NumPy reference built
    # from the module's own weight matrices, that autograd's gradient wrt the
    # input matches a central finite difference through the full
    # gate/up/down/silu composition, and that a small SwiGLU regressor learns.
    nn.seed(11)
    dim, hidden, N = 6, 10, 8
    mlp = nn.SwiGLUMLP(dim, hidden)
    rng2 = np.random.default_rng(41)
    x_np = rng2.standard_normal((N, dim)).astype(np.float32)
    w_np = rng2.standard_normal((N, dim)).astype(np.float32)

    Wg, Wu, Wd = mlp.gate.W.numpy(), mlp.up.W.numpy(), mlp.down.W.numpy()

    def silu_np(z): return z / (1.0 + np.exp(-z))
    def ref(a):
        g, u = a @ Wg, a @ Wu
        return (silu_np(g) * u) @ Wd

    x = nn.tensor(x_np, requires_grad=True)
    y = mlp(x)
    ref_y = ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref_y)))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    analytic_gx = x.grad().copy()

    def loss_np(a):
        return float(np.sum(ref(a.astype(np.float64)) * w_np) / w_np.size)
    eps, i, j = 1e-3, 2, 3
    x_plus, x_minus = x_np.copy(), x_np.copy()
    x_plus[i, j] += eps; x_minus[i, j] -= eps
    fd_gx = (loss_np(x_plus) - loss_np(x_minus)) / (2 * eps)
    grad_err = abs(float(analytic_gx[i, j]) - fd_gx)

    # a tiny SwiGLU regressor must actually learn a fixed random target.
    nn.seed(5)
    reg = nn.SwiGLUMLP(4, 12)
    xt = nn.tensor(rng.standard_normal((20, 4)).astype(np.float32))
    target = nn.tensor(rng.standard_normal((20, 4)).astype(np.float32))
    opt = nn.AdamW(reg.parameters(), lr=1e-2)
    first = last = None
    for s in range(200):
        opt.zero_grad()
        diff = nn.add(reg(xt), nn.scale(target, -1.0))
        loss = nn.mean(nn.mul(diff, diff))
        loss.backward(); opt.step()
        if s == 0:
            first = loss.item()
        last = loss.item()

    print(f"[swiglu] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} "
          f"train {first:.3f} -> {last:.3f}")
    return fwd_err < 1e-4 and grad_err < 5e-3 and last < first * 0.5


def test_geglu_mlp() -> bool:
    # GeGLUMLP: down(gelu(gate(x)) * up(x)), the T5 v1.1/PaLM gated FFN,
    # composed entirely from existing ops (Linear/gelu/mul) -- no new C kernel.
    # Same checks as SwiGLUMLP: forward vs a from-scratch NumPy reference built
    # from the module's own weight matrices (exact erf-based GELU), input
    # gradient vs a central finite difference through the full
    # gate/up/down/gelu composition, and that a small GeGLU regressor learns.
    nn.seed(13)
    dim, hidden, N = 6, 10, 8
    mlp = nn.GeGLUMLP(dim, hidden)
    rng2 = np.random.default_rng(43)
    x_np = rng2.standard_normal((N, dim)).astype(np.float32)
    w_np = rng2.standard_normal((N, dim)).astype(np.float32)

    Wg, Wu, Wd = mlp.gate.W.numpy(), mlp.up.W.numpy(), mlp.down.W.numpy()

    _erf = np.vectorize(math.erf)
    def gelu_np(z): return 0.5 * z * (1.0 + _erf(z / math.sqrt(2.0)))
    def ref(a):
        g, u = a @ Wg, a @ Wu
        return (gelu_np(g) * u) @ Wd

    x = nn.tensor(x_np, requires_grad=True)
    y = mlp(x)
    ref_y = ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref_y)))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    analytic_gx = x.grad().copy()

    def loss_np(a):
        return float(np.sum(ref(a.astype(np.float64)) * w_np) / w_np.size)
    eps, i, j = 1e-3, 2, 3
    x_plus, x_minus = x_np.copy(), x_np.copy()
    x_plus[i, j] += eps; x_minus[i, j] -= eps
    fd_gx = (loss_np(x_plus) - loss_np(x_minus)) / (2 * eps)
    grad_err = abs(float(analytic_gx[i, j]) - fd_gx)

    # a tiny GeGLU regressor must actually learn a fixed random target. Uses a
    # local RNG (not the shared module-level `rng`) so this test doesn't shift
    # the draw sequence seen by later tests that depend on it (e.g. xor init).
    nn.seed(7)
    reg = nn.GeGLUMLP(4, 12)
    rng3 = np.random.default_rng(17)
    xt = nn.tensor(rng3.standard_normal((20, 4)).astype(np.float32))
    target = nn.tensor(rng3.standard_normal((20, 4)).astype(np.float32))
    opt = nn.AdamW(reg.parameters(), lr=1e-2)
    first = last = None
    for s in range(200):
        opt.zero_grad()
        diff = nn.add(reg(xt), nn.scale(target, -1.0))
        loss = nn.mean(nn.mul(diff, diff))
        loss.backward(); opt.step()
        if s == 0:
            first = loss.item()
        last = loss.item()

    print(f"[geglu] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} "
          f"train {first:.3f} -> {last:.3f}")
    return fwd_err < 1e-4 and grad_err < 5e-3 and last < first * 0.5


def test_glu() -> bool:
    # GLU: (xW+b) * sigmoid(xV+c), the Dauphin et al. 2016 gated linear unit,
    # composed entirely from existing ops (Linear/sigmoid/mul) -- no new C
    # kernel. Checks forward correctness against a from-scratch NumPy
    # reference built from the module's own weight matrices, that autograd's
    # gradient wrt the input matches a central finite difference through the
    # full value/gate/sigmoid/mul composition, and that a small GLU regressor
    # learns.
    nn.seed(19)
    din, dout, N = 6, 5, 8
    glu = nn.GLU(din, dout)
    rng2 = np.random.default_rng(47)
    x_np = rng2.standard_normal((N, din)).astype(np.float32)
    w_np = rng2.standard_normal((N, dout)).astype(np.float32)

    Wv, bv = glu.value.W.numpy(), glu.value.b.numpy()
    Wg, bg = glu.gate.W.numpy(), glu.gate.b.numpy()

    def sigmoid_np(z): return 1.0 / (1.0 + np.exp(-z))
    def ref(a):
        return (a @ Wv + bv) * sigmoid_np(a @ Wg + bg)

    x = nn.tensor(x_np, requires_grad=True)
    y = glu(x)
    ref_y = ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref_y)))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    analytic_gx = x.grad().copy()

    def loss_np(a):
        return float(np.sum(ref(a.astype(np.float64)) * w_np) / w_np.size)
    eps, i, j = 1e-3, 2, 3
    x_plus, x_minus = x_np.copy(), x_np.copy()
    x_plus[i, j] += eps; x_minus[i, j] -= eps
    fd_gx = (loss_np(x_plus) - loss_np(x_minus)) / (2 * eps)
    grad_err = abs(float(analytic_gx[i, j]) - fd_gx)

    # a tiny GLU regressor must actually learn a fixed random target. Uses a
    # local RNG (not the shared module-level `rng`) so this test doesn't
    # shift the draw sequence seen by later tests that depend on it.
    nn.seed(23)
    reg = nn.GLU(4, 4)
    rng3 = np.random.default_rng(29)
    xt = nn.tensor(rng3.standard_normal((20, 4)).astype(np.float32))
    target = nn.tensor(rng3.standard_normal((20, 4)).astype(np.float32))
    opt = nn.AdamW(reg.parameters(), lr=1e-2)
    first = last = None
    for s in range(200):
        opt.zero_grad()
        diff = nn.add(reg(xt), nn.scale(target, -1.0))
        loss = nn.mean(nn.mul(diff, diff))
        loss.backward(); opt.step()
        if s == 0:
            first = loss.item()
        last = loss.item()

    print(f"[glu] fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} "
          f"train {first:.3f} -> {last:.3f}")
    return fwd_err < 1e-4 and grad_err < 5e-3 and last < first * 0.5


def test_maxout() -> bool:
    # maximum(a,b)/minimum(a,b): elementwise max/min composed from existing
    # ops (add/relu/scale), not a new C kernel. Checks forward vs NumPy and
    # that gradient routes entirely to whichever input was larger (the loser
    # gets zero gradient, matching relu's non-differentiability handling).
    rng2 = np.random.default_rng(71)
    a_np = rng2.standard_normal(32).astype(np.float32)
    b_np = rng2.standard_normal(32).astype(np.float32)
    a, b = nn.tensor(a_np, requires_grad=True), nn.tensor(b_np, requires_grad=True)
    ymax = nn.maximum(a, b)
    fwd_max_err = float(np.max(np.abs(ymax.numpy() - np.maximum(a_np, b_np))))
    nn.mean(ymax).backward()
    a_wins = a_np >= b_np
    grad_max_ok = bool(np.allclose(a.grad(), a_wins.astype(np.float32) / a_np.size) and
                        np.allclose(b.grad(), (~a_wins).astype(np.float32) / a_np.size))

    a2, b2 = nn.tensor(a_np, requires_grad=True), nn.tensor(b_np, requires_grad=True)
    ymin = nn.minimum(a2, b2)
    fwd_min_err = float(np.max(np.abs(ymin.numpy() - np.minimum(a_np, b_np))))

    # Maxout(in,out,num_pieces): elementwise max over `num_pieces` independent
    # Linear projections (Goodfellow et al. 2013), built from Linear +
    # maximum() -- no new C kernel. Checks forward vs a from-scratch NumPy
    # reference built from the module's own per-piece weights, the input
    # gradient vs a central finite difference through the full
    # pieces/max composition, and that a tiny Maxout regressor learns.
    nn.seed(23)
    dim, out, pieces, N = 5, 4, 3, 8
    mo = nn.Maxout(dim, out, num_pieces=pieces)
    rng3 = np.random.default_rng(73)
    x_np = rng3.standard_normal((N, dim)).astype(np.float32)
    w_np = rng3.standard_normal((N, out)).astype(np.float32)
    Ws = [p.W.numpy() for p in mo.pieces]
    bs = [p.b.numpy() for p in mo.pieces]

    def maxout_ref(xa):
        outs = [xa @ W + bv for W, bv in zip(Ws, bs)]
        r = outs[0]
        for o in outs[1:]:
            r = np.maximum(r, o)
        return r

    x = nn.tensor(x_np, requires_grad=True)
    y = mo(x)
    ref_y = maxout_ref(x_np.astype(np.float64)).astype(np.float32)
    fwd_err = float(np.max(np.abs(y.numpy() - ref_y)))

    nn.mean(nn.mul(y, nn.tensor(w_np))).backward()
    analytic_gx = x.grad().copy()

    def loss_np(xa):
        return float(np.sum(maxout_ref(xa.astype(np.float64)) * w_np) / w_np.size)
    eps, i, j = 1e-3, 1, 2
    x_plus, x_minus = x_np.copy(), x_np.copy()
    x_plus[i, j] += eps; x_minus[i, j] -= eps
    fd_gx = (loss_np(x_plus) - loss_np(x_minus)) / (2 * eps)
    grad_err = abs(float(analytic_gx[i, j]) - fd_gx)

    nn.seed(29)
    reg = nn.Maxout(4, 4, num_pieces=3)
    rng4 = np.random.default_rng(79)
    xt = nn.tensor(rng4.standard_normal((20, 4)).astype(np.float32))
    target = nn.tensor(rng4.standard_normal((20, 4)).astype(np.float32))
    opt = nn.AdamW(reg.parameters(), lr=1e-2)
    first = last = None
    for s in range(200):
        opt.zero_grad()
        diff = nn.add(reg(xt), nn.scale(target, -1.0))
        loss = nn.mean(nn.mul(diff, diff))
        loss.backward(); opt.step()
        if s == 0:
            first = loss.item()
        last = loss.item()

    print(f"[maxout] max_fwd_err={fwd_max_err:.1e} max_grad_ok={grad_max_ok} "
          f"min_fwd_err={fwd_min_err:.1e} module_fwd_err={fwd_err:.1e} "
          f"module_grad_err={grad_err:.1e} train {first:.3f} -> {last:.3f}")
    return (fwd_max_err < 1e-6 and grad_max_ok and fwd_min_err < 1e-6 and
            fwd_err < 1e-4 and grad_err < 5e-3 and last < first * 0.5)


def test_sgd() -> bool:
    # SGD (with momentum) also minimizes a simple quadratic.
    w = nn.tensor(np.zeros(4, np.float32), requires_grad=True)
    target = nn.tensor(np.array([3, -2, 0.5, 7], np.float32))
    opt = nn.SGD([w], lr=0.1, momentum=0.9)
    loss = None
    for _ in range(300):
        opt.zero_grad()
        diff = nn.add(w, nn.scale(target, -1.0))
        loss = nn.mean(nn.mul(diff, diff))
        loss.backward()
        opt.step(clip=0.0)
    print(f"[sgd] quadratic loss -> {loss.item():.2e}")
    return loss.item() < 1e-3


def test_checkpoint() -> bool:
    # Gradient checkpointing: identical grads/loss to the non-checkpointed graph.
    def run(use_ckpt):
        r = np.random.default_rng(0)
        W1 = nn.tensor(r.standard_normal((4, 6)).astype(np.float32), requires_grad=True)
        W2 = nn.tensor(r.standard_normal((6, 4)).astype(np.float32), requires_grad=True)
        x = nn.tensor(r.standard_normal((3, 4)).astype(np.float32), requires_grad=True)
        seg = lambda p: nn.matmul(nn.relu(nn.matmul(p[0], p[1])), p[2])
        z = nn.checkpoint(seg, [x, W1, W2]) if use_ckpt else seg([x, W1, W2])
        loss = nn.mean(nn.mul(z, z))
        loss.backward()
        return W1.grad(), W2.grad(), x.grad(), loss.item()
    a = run(False)
    b = run(True)
    d = max(float(np.max(np.abs(a[i] - b[i]))) for i in range(3))
    d = max(d, abs(a[3] - b[3]))
    # And a checkpointed TransformerBlock trains (forward+backward+step).
    nn.seed(0)
    blk = nn.TransformerBlock(32, num_heads=4)
    opt = nn.AdamW(blk.parameters(), lr=1e-3, weight_decay=0.0)
    xb = nn.tensor(np.random.default_rng(1).standard_normal((4, 8, 32)).astype(np.float32))
    f0 = fN = None
    for _ in range(15):
        opt.zero_grad()
        loss = nn.mean(nn.mul(nn.checkpoint(lambda p: blk(p[0]), [xb]),
                              nn.checkpoint(lambda p: blk(p[0]), [xb])))
        loss.backward(); opt.step()
        f0 = loss.item() if f0 is None else f0; fN = loss.item()
    print(f"[checkpoint] grad/out max|diff|={d:.2e}  ckpt-block loss {f0:.4f}->{fN:.4f}")
    return d < 1e-5 and fN < f0


def test_distributed_dp() -> bool:
    # Data-parallel correctness: averaging per-shard gradients == the full-batch
    # gradient (because the loss is a mean). all_reduce_gradients does exactly
    # this averaging across nodes, so distributed DP training matches single-node.
    r = np.random.default_rng(0)
    W = nn.tensor(r.standard_normal((6, 4)).astype(np.float32), requires_grad=True)
    X = r.standard_normal((8, 6)).astype(np.float32)      # 8 samples
    y = [int(v) for v in r.integers(0, 4, size=8)]

    def grad_on(rows):
        for p in [W]:
            p.set_(p.numpy())                              # (no-op; keep value)
        # zero grad via an optimizer-less reset: re-create not needed; use a step
        # of all_reduce? simpler: read grads after a fresh backward on a clone.
        Wc = nn.tensor(W.numpy(), requires_grad=True)
        xb = nn.tensor(X[rows])
        loss = nn.softmax_cross_entropy(nn.matmul(xb, Wc), [y[i] for i in rows])
        loss.backward()
        return Wc.grad()

    g_full = grad_on(list(range(8)))
    g_a = grad_on(list(range(0, 4)))
    g_b = grad_on(list(range(4, 8)))
    dp = 0.5 * (g_a + g_b)
    diff = float(np.max(np.abs(g_full - dp)))

    # Single-node all_reduce_gradients must be a no-op (world_size==1).
    ws = nn.world_size()
    Wn = nn.tensor(W.numpy(), requires_grad=True)
    loss = nn.softmax_cross_entropy(nn.matmul(nn.tensor(X), Wn), y)
    loss.backward()
    before = Wn.grad().copy()
    nn.all_reduce_gradients([Wn])
    noop_diff = float(np.max(np.abs(before - Wn.grad())))

    print(f"[distributed] world_size={ws}  DP-identity max|diff|={diff:.2e}  "
          f"single-node all_reduce no-op diff={noop_diff:.2e}")
    return diff < 1e-5 and noop_diff < 1e-6


def test_tensor_parallel() -> bool:
    # Row-parallel (tensor) parallelism: the weight's contraction dim is sharded
    # across ranks. Each rank computes x_shard·W_shard; all_reduce sums the
    # partials into the full output. Verify the composition equals the unsharded
    # matmul (world=1 → all_reduce is identity, so this is the in-process sum;
    # on a cluster each rank holds ONE shard and all_reduce sums across nodes).
    r = np.random.default_rng(0)
    X = r.standard_normal((2, 6)).astype(np.float32)
    W = r.standard_normal((6, 4)).astype(np.float32)
    full = nn.matmul(nn.tensor(X), nn.tensor(W)).numpy()
    parts = nn.add(
        nn.all_reduce(nn.matmul(nn.tensor(X[:, :3]), nn.tensor(W[:3]))),
        nn.all_reduce(nn.matmul(nn.tensor(X[:, 3:]), nn.tensor(W[3:]))))
    d = float(np.max(np.abs(parts.numpy() - full)))
    print(f"[tensor-parallel] row-parallel shard-sum vs full max|diff|={d:.2e}  ws={nn.world_size()}")
    return d < 1e-5


def test_error_reporting() -> bool:
    # Production hardening: a native failure must surface as a Python exception
    # with a message, not silently corrupt training. Previously the C ABI
    # swallowed exceptions (catch(...){}); now it records them in a thread-local
    # channel that nn raises from.
    raised = False
    try:
        # inner dims 3 vs 5 — matmul must reject this.
        nn.matmul(nn.tensor(np.zeros((2, 3), np.float32)),
                  nn.tensor(np.zeros((5, 4), np.float32)))
    except RuntimeError as e:
        raised = bool(str(e).strip()) and "op failed" in str(e)

    # A subsequent valid op must succeed and leave the error channel clear.
    W = nn.tensor(np.random.default_rng(0).standard_normal((4, 3)).astype(np.float32),
                  requires_grad=True)
    loss = nn.softmax_cross_entropy(
        nn.matmul(nn.tensor(np.random.default_rng(1).standard_normal((2, 4)).astype(np.float32)), W),
        [0, 1])
    loss.backward()
    cleared = nn._last_error() == "" and np.isfinite(W.grad()).all()

    print(f"[error-reporting] invalid op raised={raised}  valid-after cleared={cleared}")
    return raised and cleared


def test_bitlinear() -> bool:
    # BitNet b1.58: the STE ternary_quantize must reproduce the per-column
    # absmean dequantized weight (so training-forward == inference ternary GEMM)
    # with an identity straight-through gradient, and a BitLinear network must
    # actually learn despite ternary weights.
    W = nn.tensor(np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32),
                  requires_grad=True)
    Wq = nn.ternary_quantize(W)
    Wn = W.numpy(); scale = np.abs(Wn).mean(axis=0)
    inv = np.where(scale == 0, 1.0, scale)
    codes = np.where(Wn / inv > 0.5, 1, np.where(Wn / inv < -0.5, -1, 0))
    ref = codes * scale
    fwd_err = float(np.max(np.abs(Wq.numpy() - ref)))
    nn.mean(Wq).backward()
    ste_ok = np.allclose(W.grad(), 1.0 / W.numpy().size, atol=1e-6)   # identity STE

    rng = np.random.default_rng(1)
    X = nn.tensor(rng.standard_normal((32, 8)).astype(np.float32))
    y = [int(v) for v in rng.integers(0, 3, size=32)]
    net = nn.Sequential(nn.BitLinear(8, 16), nn.ReLU(), nn.BitLinear(16, 3))
    opt = nn.AdamW(net.parameters(), lr=5e-2)
    first = last = None
    for step in range(60):
        opt.zero_grad()
        loss = nn.softmax_cross_entropy(net(X), y)
        loss.backward(); opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    print(f"[bitlinear] fwd_err={fwd_err:.1e} ste={ste_ok}  train loss {first:.3f} -> {last:.3f}")
    return fwd_err < 1e-5 and ste_ok and last < first * 0.6


def test_structured_sparse_linear() -> bool:
    # N:M structured sparsity: every 4-weight contraction group keeps the top-2
    # magnitudes per output column, exposes compact bit metadata, masks the
    # forward exactly, and blocks gradients into pruned weights.
    rng = np.random.default_rng(14)
    W = rng.standard_normal((8, 5)).astype(np.float32)
    b = rng.standard_normal(5).astype(np.float32) * 0.1
    layer = nn.StructuredSparseLinear(8, 5, n=2, m=4, base_weight=W, base_bias=b)
    mask = layer.mask
    meta = layer.metadata()
    group_counts = [int(mask[start:start + 4, c].sum())
                    for c in range(mask.shape[1]) for start in range(0, mask.shape[0], 4)]
    mask_ok = all(v == 2 for v in group_counts)
    meta_ok = all(int(x).bit_count() == 2 for x in meta.reshape(-1))
    density_ok = abs(layer.density() - 0.5) < 1e-6

    Xn = rng.standard_normal((6, 8)).astype(np.float32)
    x = nn.tensor(Xn)
    out = layer(x).numpy()
    ref = Xn @ (W * mask) + b[None, :]
    fwd_err = float(np.max(np.abs(out - ref)))

    loss = nn.mean(layer(x))
    loss.backward()
    grad = layer.W.grad()
    expected = np.outer(Xn.sum(axis=0), np.ones(5, dtype=np.float32)) / float(out.size)
    grad_ok = (float(np.max(np.abs(grad - expected * mask))) < 1e-6 and
               np.allclose(grad[mask == 0.0], 0.0))

    nn.seed(15)
    X = nn.tensor(rng.standard_normal((40, 8)).astype(np.float32))
    y = [int(v) for v in rng.integers(0, 3, size=40)]
    net = nn.Sequential(nn.StructuredSparseLinear(8, 16), nn.ReLU(),
                        nn.StructuredSparseLinear(16, 3))
    opt = nn.AdamW(net.parameters(), lr=4e-2, weight_decay=0.0)
    first = last = None
    for step in range(80):
        opt.zero_grad(); tr = nn.softmax_cross_entropy(net(X), y); tr.backward(); opt.step()
        if step == 0:
            first = tr.item()
        last = tr.item()

    print(f"[structured-sparse] mask={mask_ok} meta={meta_ok} density={layer.density():.2f} "
          f"fwd_err={fwd_err:.1e} grad_mask={grad_ok} train {first:.3f}->{last:.3f}")
    return mask_ok and meta_ok and density_ok and fwd_err < 1e-6 and grad_ok and last < first * 0.7


def test_distillation() -> bool:
    # Soft-target distillation: the native CE-soft gradient must match the
    # analytic T*(softmax(student/T)-softmax(teacher/T))/M formula, and a
    # ternary BitLinear student must learn a fixed teacher distribution.
    rng = np.random.default_rng(16)
    Ttemp = 2.5
    z_np = rng.standard_normal((3, 4)).astype(np.float32)
    t_np = rng.standard_normal((3, 4)).astype(np.float32)
    z = nn.tensor(z_np, requires_grad=True)
    loss = nn.distillation_loss(z, t_np, temperature=Ttemp)
    loss.backward()
    sp = np.exp(z_np / Ttemp - (z_np / Ttemp).max(1, keepdims=True)); sp /= sp.sum(1, keepdims=True)
    tp = np.exp(t_np / Ttemp - (t_np / Ttemp).max(1, keepdims=True)); tp /= tp.sum(1, keepdims=True)
    grad_ref = Ttemp * (sp - tp) / z_np.shape[0]
    grad_err = float(np.max(np.abs(z.grad() - grad_ref)))

    nn.seed(16)
    Xn = rng.standard_normal((48, 6)).astype(np.float32)
    teacher_W = rng.standard_normal((6, 4)).astype(np.float32) * 1.2
    teacher_logits = Xn @ teacher_W
    X = nn.tensor(Xn)
    student = nn.Sequential(nn.BitLinear(6, 18), nn.ReLU(), nn.BitLinear(18, 4))
    opt = nn.AdamW(student.parameters(), lr=4e-2, weight_decay=0.0)
    first = last = None
    for step in range(120):
        opt.zero_grad()
        dl = nn.distillation_loss(student(X), teacher_logits, temperature=2.0)
        dl.backward(); opt.step()
        if step == 0:
            first = dl.item()
        last = dl.item()
    agree = float(np.mean(np.argmax(student(X).numpy(), axis=1) == np.argmax(teacher_logits, axis=1)))
    print(f"[distill] grad_err={grad_err:.1e}  ternary student {first:.3f}->{last:.3f} agree={agree:.2f}")
    return grad_err < 2e-6 and last < first * 0.7 and agree > 0.75


def test_mtp() -> bool:
    # Multi-Token Prediction: K heads on a shared trunk, head j predicting j
    # tokens ahead. Trained with the summed CE, each head must learn its
    # look-ahead, and the heads must serve as the model's OWN draft for
    # speculative decoding — lossless vs greedy of head 0, several tokens/forward.
    nn.seed(3)
    V, T, D, K = 12, 24, 32, 3
    seq = [i % 4 for i in range(T + 1)]
    ids, tgt = seq[:-1], seq[1:]

    class Trunk(nn.Module):
        def __init__(self):
            self.e = nn.Embedding(V, D)
            self.blocks = [nn.TransformerBlock(D, num_heads=4) for _ in range(2)]
            self.n = nn.RMSNorm(D)

        def forward(self, ids):
            x = self.e(ids)
            for b in self.blocks:
                x = b(x)
            return self.n(x)

    trunk = Trunk(); heads = nn.MTPHeads(D, V, n_predict=K)
    def mtp_model(ids):
        return heads(trunk(ids))
    opt = nn.AdamW(trunk.parameters() + heads.parameters(), lr=3e-3)
    first = last = None
    for st in range(300):
        opt.zero_grad(); loss = nn.mtp_loss(mtp_model(ids), tgt); loss.backward(); opt.step()
        if st == 0:
            first = loss.item()
        last = loss.item()

    hl = [h.numpy() for h in mtp_model(ids)]
    accs = []
    for j in range(K):
        n = T - j
        accs.append(float((np.argmax(hl[j][:n], 1) == np.array(tgt[j:j + n])).mean()))

    target = lambda c: mtp_model(c)[0]
    greedy = nn.greedy_generate(target, [0, 1, 2], 12)
    spec, fw = nn.speculative_generate(target, [0, 1, 2], 12,
                                       nn.mtp_draft_fn(mtp_model, K), k=K)
    lossless = (greedy == spec)
    tpf = 12 / fw
    print(f"[mtp] train {first:.3f} -> {last:.3f}  head-acc={[round(a,2) for a in accs]}  "
          f"self-spec lossless={lossless} tok/fwd={tpf:.2f}")
    return last < first * 0.1 and min(accs) >= 0.99 and lossless and tpf > 1.5


def test_hybrid_mamba_transformer() -> bool:
    # Hybrid Mamba-Transformer (Jamba-style). Three properties:
    #  (a) batched SSM is per-sequence — a [B,T,D] scan must equal scanning each
    #      sequence alone, and must NOT equal a flattened [B*T,D] scan (which
    #      would leak state across sequence boundaries);
    #  (b) hybrid_blocks interleaves the two mixers at the requested ratio, so
    #      only 1-in-attn_every layer carries a KV cache;
    #  (c) a hybrid LM trains and greedily regenerates its sequence.
    nn.seed(0)
    B, T, D = 3, 6, 16
    ssm = nn.SSMBlock(D, d_state=4)
    xn = np.random.default_rng(0).standard_normal((B, T, D)).astype(np.float32)
    y3 = ssm(nn.tensor(xn)).numpy()
    per = np.stack([ssm(nn.tensor(xn[b])).numpy() for b in range(B)])
    batch_indep = float(np.max(np.abs(y3 - per)))
    leaky = ssm(nn.tensor(xn.reshape(B * T, D))).numpy().reshape(B, T, D)
    differs_from_leaky = float(np.max(np.abs(y3 - leaky)))

    blocks = nn.hybrid_blocks(D, n_layers=8, num_heads=4, attn_every=4, d_state=4)
    kinds = ["A" if isinstance(b, nn.TransformerBlock) else "M" for b in blocks]
    ratio_ok = (kinds == ["M", "M", "M", "A", "M", "M", "M", "A"])

    nn.seed(11)
    V, TT, Dm = 16, 14, 32
    seq = [(i * 3 + 1) % V for i in range(TT + 1)]
    ids, tgt = seq[:-1], seq[1:]

    class Hybrid(nn.Module):
        def __init__(self):
            self.e = nn.Embedding(V, Dm)
            self.blocks = nn.hybrid_blocks(Dm, n_layers=4, num_heads=4,
                                           attn_every=2, d_state=8)
            self.n = nn.RMSNorm(Dm); self.h = nn.Linear(Dm, V, bias=False)

        def forward(self, ids):
            x = self.e(ids)
            for b in self.blocks:
                x = b(x)
            return self.h(self.n(x))

    m = Hybrid(); opt = nn.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for st in range(300):
        opt.zero_grad(); loss = nn.softmax_cross_entropy(m(ids), tgt); loss.backward(); opt.step()
        if st == 0:
            first = loss.item()
        last = loss.item()
    cur = [seq[0]]; gen = []
    for _ in range(TT):
        gen.append(int(np.argmax(m(cur).numpy()[-1]))); cur = cur + [gen[-1]]
    match = sum(int(a == b) for a, b in zip(gen, tgt))

    print(f"[hybrid] batch_indep={batch_indep:.1e} (vs leaky flat scan {differs_from_leaky:.2f}) "
          f"stack={''.join(kinds)}  LM train {first:.3f} -> {last:.3f}  match {match}/{TT}")
    return (batch_indep < 1e-6 and differs_from_leaky > 1e-3 and ratio_ok
            and last < first * 0.2 and match >= TT - 1)


def test_qlora() -> bool:
    # QLoRA: a frozen ternary-quantized base (~2 bits/weight, not a parameter) +
    # a trainable LoRA adapter. Only A,B train; the base never changes; B=0 means
    # the layer starts exactly at the quantized base; grads match NumPy; and it
    # fine-tunes a task to convergence.
    nn.seed(0)
    inf, outf, r = 128, 32, 4
    q = nn.QLoRALinear(inf, outf, r=r, alpha=8, bias=True)

    only_ab = (len(q.parameters()) == 2)
    bits = q.base_bits_per_weight()

    x = nn.tensor(np.random.default_rng(1).standard_normal((16, inf)).astype(np.float32))
    base = q._codes.astype(np.float32) * q._scale[None, :]
    init_ok = float(np.max(np.abs(q(x).numpy() - (x.numpy() @ base + q._bias[None, :])))) < 1e-4

    codes0 = q._codes.copy()
    tgt = [int(v) for v in np.random.default_rng(2).integers(0, outf, 16)]
    nn.softmax_cross_entropy(q(x), tgt).backward()
    gA, gB = q.A.grad().copy(), q.B.grad().copy()
    Xn, A, B, sc = x.numpy(), q.A.numpy(), q.B.numpy(), q.scaling
    logits = Xn @ base + sc * (Xn @ A) @ B + q._bias[None, :]
    pr = np.exp(logits - logits.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    oh = np.zeros_like(pr); oh[np.arange(16), tgt] = 1; dl = (pr - oh) / 16
    grad_ok = (float(np.max(np.abs(gA - sc * Xn.T @ (dl @ B.T)))) < 1e-4 and
               float(np.max(np.abs(gB - sc * (Xn @ A).T @ dl))) < 1e-4)

    opt = nn.AdamW(q.adapter_parameters(), lr=5e-2)
    first = last = None
    for s in range(80):
        opt.zero_grad(); loss = nn.softmax_cross_entropy(q(x), tgt); loss.backward(); opt.step()
        if s == 0:
            first = loss.item()
        last = loss.item()
    frozen = np.array_equal(codes0, q._codes)
    # int4 base variant: ~4 bits/weight, but a strictly lower reconstruction
    # error than the 2-bit ternary base (finer quantization), still frozen +
    # params == {A,B}, and fine-tunes.
    nn.seed(0)
    W = np.random.default_rng(7).standard_normal((inf, outf)).astype(np.float32)
    qt = nn.QLoRALinear(inf, outf, r=r, base_weight=W.copy(), base_format="ternary")
    qi = nn.QLoRALinear(inf, outf, r=r, base_weight=W.copy(), base_format="int4")
    err_t = float(np.max(np.abs(W - qt._codes * qt._scale[None, :])))
    err_i = float(np.max(np.abs(W - qi._codes * qi._scale[None, :])))
    int4_ok = (len(qi.parameters()) == 2 and 3.0 <= qi.base_bits_per_weight() < 5.0
               and err_i < err_t and set(np.unique(qi._codes).tolist()) <= set(range(-7, 8)))
    codes_i0 = qi._codes.copy()
    opt2 = nn.AdamW(qi.adapter_parameters(), lr=5e-2)
    for _ in range(40):
        opt2.zero_grad(); nn.softmax_cross_entropy(qi(x), tgt).backward(); opt2.step()
    int4_ok = int4_ok and np.array_equal(codes_i0, qi._codes)   # base still frozen

    # mxfp4 base variant: OCP microscaling — 4-bit E2M1 codes sharing an 8-bit
    # E8M0 power-of-two scale per 32-row block (~4.25 bits/weight). Finer than
    # the 2-bit ternary base, still frozen + params == {A,B}, and fine-tunes.
    qm = nn.QLoRALinear(inf, outf, r=r, base_weight=W.copy(), base_format="mxfp4")
    err_m = float(np.max(np.abs(W - qm._base().numpy())))
    mxfp4_ok = (len(qm.parameters()) == 2 and 4.0 <= qm.base_bits_per_weight() < 4.5
                and err_m < err_t
                and set(np.unique(np.abs(qm._codes)).tolist()) <= set(range(8)))
    codes_m0 = qm._codes.copy()
    opt3 = nn.AdamW(qm.adapter_parameters(), lr=5e-2)
    for _ in range(40):
        opt3.zero_grad(); nn.softmax_cross_entropy(qm(x), tgt).backward(); opt3.step()
    mxfp4_ok = mxfp4_ok and np.array_equal(codes_m0, qm._codes)   # base still frozen

    print(f"[qlora] params={len(q.parameters())} base_bits={bits:.2f} init_ok={init_ok} "
          f"grad_ok={grad_ok} frozen={frozen}  fine-tune {first:.3f} -> {last:.3f}  "
          f"int4(bits={qi.base_bits_per_weight():.2f}, recon_err {err_i:.3f}<{err_t:.3f}={err_i<err_t}) "
          f"mxfp4(bits={qm.base_bits_per_weight():.2f}, recon_err {err_m:.3f}<{err_t:.3f}={err_m<err_t})")
    return (only_ab and bits < 3.0 and init_ok and grad_ok and frozen
            and last < first * 0.2 and int4_ok and mxfp4_ok)


def test_mamba() -> bool:
    # Selective scan (the SSM recurrence h_t = a_t*h_{t-1}+b_t): forward matches a
    # NumPy reference exactly, and its analytic gradients match finite differences.
    # Then an attention-free Mamba LM must memorize + regenerate a sequence.
    T, Dd = 6, 3
    rng = np.random.default_rng(0)
    a_np = (0.5 + 0.3 * rng.standard_normal((T, Dd))).astype(np.float32)
    b_np = rng.standard_normal((T, Dd)).astype(np.float32)
    a = nn.tensor(a_np, requires_grad=True); b = nn.tensor(b_np, requires_grad=True)
    h = nn.selective_scan(a, b)

    href = np.zeros((T, Dd), np.float32); p = np.zeros(Dd, np.float32)
    for t in range(T):
        p = a_np[t] * p + b_np[t]; href[t] = p
    fwd_err = float(np.max(np.abs(h.numpy() - href)))

    w = rng.standard_normal((T, Dd)).astype(np.float32)
    nn.mean(nn.mul(h, nn.tensor(w))).backward()
    ga, gb = a.grad().copy(), b.grad().copy()

    def L(av, bv):
        hh = np.zeros((T, Dd), np.float32); q = np.zeros(Dd, np.float32)
        for t in range(T):
            q = av[t] * q + bv[t]; hh[t] = q
        return float(np.mean(hh * w))
    eps = 1e-3; na = np.zeros_like(a_np); nb = np.zeros_like(b_np)
    for t in range(T):
        for d in range(Dd):
            ap = a_np.copy(); ap[t, d] += eps; am = a_np.copy(); am[t, d] -= eps
            na[t, d] = (L(ap, b_np) - L(am, b_np)) / (2 * eps)
            bp = b_np.copy(); bp[t, d] += eps; bm = b_np.copy(); bm[t, d] -= eps
            nb[t, d] = (L(a_np, bp) - L(a_np, bm)) / (2 * eps)
    grad_err = max(float(np.max(np.abs(ga - na))), float(np.max(np.abs(gb - nb))))

    # Parallel path: a large scan (T*D ≥ 8192 fans out over the D channels on the
    # thread pool) must be bit-identical to the sequential NumPy reference.
    Tp, Dp = 128, 128
    ap = (0.9 + 0.05 * rng.standard_normal((Tp, Dp))).astype(np.float32)
    bp = rng.standard_normal((Tp, Dp)).astype(np.float32)
    hp = nn.selective_scan(nn.tensor(ap), nn.tensor(bp)).numpy()
    hpr = np.zeros((Tp, Dp), np.float32); pp = np.zeros(Dp, np.float32)
    for t in range(Tp):
        pp = ap[t] * pp + bp[t]; hpr[t] = pp
    par_err = float(np.max(np.abs(hp - hpr)))

    # Full multi-state selective SSM (Mamba/S6): the block's forward must match an
    # independent NumPy reference of the exact Δ/A/B/C recurrence.
    nn.seed(1)
    Tt, dim, Ns = 7, 16, 8
    xx = nn.tensor(rng.standard_normal((Tt, dim)).astype(np.float32))
    ssm = nn.SSMBlock(dim, d_state=Ns, conv_kernel=0)   # isolate the SSM math
    out = ssm(xx).numpy()
    xn = xx.numpy()

    def lin(L, v):
        y = v @ L.W.numpy()
        return y + L.b.numpy() if L.b is not None else y
    u = lin(ssm.in_proj, xn)
    raw = lin(ssm.dt_proj, xn)
    delta = np.maximum(raw, 0) + np.log1p(np.exp(-np.abs(raw)))     # softplus
    Bm, Cm = lin(ssm.B_proj, xn), lin(ssm.C_proj, xn)
    A = -np.exp(ssm.A_log.numpy()); Dsk = ssm.Dskip.numpy()
    yy = np.zeros((Tt, dim), np.float32); hs = np.zeros((dim, Ns), np.float32)
    for t in range(Tt):
        hs = np.exp(delta[t][:, None] * A) * hs + (delta[t] * u[t])[:, None] * Bm[t][None, :]
        yy[t] = (hs * Cm[t][None, :]).sum(1) + Dsk * u[t]
    ref = yy @ ssm.out_proj.W.numpy() + ssm.out_proj.b.numpy()
    ssm_err = float(np.max(np.abs(out - ref)))

    # Depthwise short conv: matches a NumPy causal conv+bias+SiLU, and is causal —
    # perturbing a future input must not change any earlier output.
    kk = 4
    cssm = nn.SSMBlock(dim, d_state=Ns, conv_kernel=kk)
    uu = rng.standard_normal((Tt, dim)).astype(np.float32)
    cy = cssm._causal_conv1d(nn.tensor(uu)).numpy()
    cw, cb = cssm.conv_w.numpy(), cssm.conv_b.numpy()
    up = np.concatenate([np.zeros((kk - 1, dim), np.float32), uu], 0)
    cref = np.zeros((Tt, dim), np.float32)
    for t in range(Tt):
        for i in range(kk):
            cref[t] += cw[:, i] * up[t + i]
    cref += cb; cref = cref / (1 + np.exp(-cref))
    conv_err = float(np.max(np.abs(cy - cref)))
    uu2 = uu.copy(); uu2[Tt - 1] += 5.0
    cy2 = cssm._causal_conv1d(nn.tensor(uu2)).numpy()
    conv_causal = float(np.max(np.abs(cy[:Tt - 1] - cy2[:Tt - 1])))   # must be 0

    # Attention-free Mamba LM: memorize + regenerate.
    nn.seed(5)
    V, TT, D = 16, 14, 32
    seq = [(i * 3 + 1) % V for i in range(TT + 1)]
    ids, tgt = seq[:-1], seq[1:]

    class MambaLM(nn.Module):
        def __init__(self):
            self.embed = nn.Embedding(V, D)
            self.blocks = [nn.MambaBlock(D) for _ in range(2)]
            self.norm = nn.RMSNorm(D); self.head = nn.Linear(D, V, bias=False)

        def forward(self, ids):
            x = self.embed(ids)
            for bl in self.blocks:
                x = bl(x)
            return self.head(self.norm(x))

    m = MambaLM(); opt = nn.AdamW(m.parameters(), lr=3e-3)
    first = last = None
    for s in range(300):
        opt.zero_grad(); loss = nn.softmax_cross_entropy(m(ids), tgt); loss.backward(); opt.step()
        if s == 0:
            first = loss.item()
        last = loss.item()
    cur = [seq[0]]; gen = []
    for _ in range(TT):
        gen.append(int(np.argmax(m(cur).numpy()[-1]))); cur = cur + [gen[-1]]
    match = sum(int(x == y) for x, y in zip(gen, tgt))

    print(f"[mamba] scan fwd_err={fwd_err:.1e} grad_err={grad_err:.1e} par_err={par_err:.1e} "
          f"ssm_block_err={ssm_err:.1e} conv_err={conv_err:.1e} conv_causal={conv_causal:.1e}  "
          f"LM train {first:.3f} -> {last:.3f}  greedy match {match}/{TT}")
    # par_err is a float-rounding tolerance vs the NumPy reference over a 128-step
    # recurrence, not a bit-identity check: on macOS-ARM clang's FMA contraction
    # makes it ~1.7e-6 (Linux x86 ~0). A real parallel-scan bug (cross-channel
    # leakage) would be O(1), so 1e-4 still catches it while staying portable.
    return (fwd_err < 1e-6 and grad_err < 1e-3 and par_err < 1e-4 and ssm_err < 1e-4
            and conv_err < 1e-5 and conv_causal == 0.0
            and last < first * 0.2 and match >= TT - 1)


def test_speculative_decoding() -> bool:
    # Lossless speculative decoding: on a GPT that memorized a periodic sequence,
    # an n-gram drafter proposes the repeat and the target verifies it in one
    # forward. The speculative output must be TOKEN-FOR-TOKEN identical to plain
    # greedy decoding, while accepting several tokens per forward (the speedup).
    nn.seed(7)
    V, D, P = 12, 32, 4                          # period-4 sequence: 0,1,2,3,0,1,...
    train = [i % P for i in range(40)]
    ids, tgt = train[:-1], train[1:]

    class GPT(nn.Module):
        def __init__(self):
            self.embed = nn.Embedding(V, D)
            self.blocks = [nn.TransformerBlock(D, num_heads=4) for _ in range(2)]
            self.norm = nn.RMSNorm(D)
            self.head = nn.Linear(D, V, bias=False)

        def forward(self, ids):
            x = self.embed(ids)
            for b in self.blocks:
                x = b(x)
            return self.head(self.norm(x))

    model = GPT()
    opt = nn.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(160):
        opt.zero_grad()
        loss = nn.softmax_cross_entropy(model(ids), tgt)
        loss.backward(); opt.step()

    prompt = [0, 1, 2, 3, 0, 1]
    n_new = 16
    greedy = nn.greedy_generate(model, prompt, n_new)

    # (a) n-gram drafter: lossless + multi-token-per-forward speedup.
    spec, forwards = nn.speculative_generate(model, prompt, n_new,
                                             nn.ngram_draft_fn(k=4, n=2), k=4)
    ngram_ok = (greedy == spec) and (n_new / forwards) > 1.5

    # (b) draft-MODEL drafter: still identical to greedy of the TARGET.
    spec_m, _ = nn.speculative_generate(model, prompt, n_new,
                                        nn.model_draft_fn(model, k=4), k=4)
    model_ok = (greedy == spec_m)

    # (c) sampler-exact speculative sampling is distribution-identical to sampling
    #     the target directly. Use UNTRAINED target+draft (diffuse, DIFFERENT
    #     distributions) so q is non-degenerate and the accept/reject +
    #     residual-resample paths are genuinely exercised; the empirical
    #     first-token histogram must match the target's analytic q.
    nn.seed(21); target_u = GPT()
    nn.seed(84); draft_u = GPT()
    q = _softmax_np(np.asarray(target_u(prompt).numpy()[-1], np.float64))
    spread_ok = float(q.max()) < 0.5             # distribution genuinely non-degenerate
    N = 800
    counts = np.zeros(V)
    for i in range(N):
        toks, _ = nn.speculative_sample(target_u, draft_u, prompt, 1, k=2,
                                        temperature=1.0, seed=i)
        counts[toks[0]] += 1
    tv = 0.5 * float(np.sum(np.abs(counts / N - q)))   # total-variation distance
    dist_ok = spread_ok and tv < 0.07

    # (d) temperature→0 reduces to greedy of the target exactly.
    cold, _ = nn.speculative_sample(model, GPT(), prompt, n_new, k=4,
                                    temperature=1e-6, seed=1)
    cold_ok = (cold == greedy)

    print(f"[spec] ngram(lossless={greedy==spec}, tok/fwd={n_new/forwards:.2f}) "
          f"model_draft_lossless={model_ok}  sampler-exact(q_max={q.max():.2f}, TV={tv:.3f})  "
          f"temp0==greedy={cold_ok}")
    return ngram_ok and model_ok and dist_ok and cold_ok


def _softmax_np(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def test_moe_lm() -> bool:
    # End-to-end MoE language model: a GPT whose transformer FFN is a sparse
    # Mixture-of-Experts (drop-in). It must overfit a short sequence (train loss
    # collapses) and then greedily generate that exact sequence — proving MoE
    # composes through embedding → attention → MoE-FFN → head and trains.
    nn.seed(4)
    V, T, D, E = 16, 12, 32, 4
    seq = [(i * 5 + 2) % V for i in range(T + 1)]
    ids, tgt = seq[:-1], seq[1:]

    class MoEGPT(nn.Module):
        def __init__(self):
            self.embed = nn.Embedding(V, D)
            self.blocks = [nn.TransformerBlock(D, num_heads=4, moe_experts=E, moe_top_k=2)
                           for _ in range(2)]
            self.norm = nn.RMSNorm(D)
            self.head = nn.Linear(D, V, bias=False)

        def forward(self, ids):
            x = self.embed(ids)
            for b in self.blocks:
                x = b(x)
            return self.head(self.norm(x))

        def aux(self):
            a = None
            for b in self.blocks:
                bl = b.aux_loss()
                a = bl if a is None else nn.add(a, bl)
            return a

    model = MoEGPT()
    opt = nn.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    first = last = None
    for step in range(200):
        opt.zero_grad()
        logits = model(ids)
        loss = nn.add(nn.softmax_cross_entropy(logits, tgt), nn.scale(model.aux(), 0.001))
        loss.backward(); opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    # Greedy autoregression from the first token must reproduce the sequence:
    # feed the growing prefix, take argmax of the last position (predicts the
    # next token), which should regenerate tgt = seq[1:].
    cur = [seq[0]]
    gen = []
    for _ in range(T):
        nxt = int(np.argmax(model(cur).numpy()[-1]))
        gen.append(nxt); cur = cur + [nxt]
    match = sum(int(a == b) for a, b in zip(gen, tgt))
    print(f"[moe-lm] train loss {first:.3f} -> {last:.3f}  greedy match {match}/{T}")
    return last < first * 0.2 and match >= T - 1


def test_gather_scatter() -> bool:
    # index_select (gather) forward + scatter-add backward, and index_add (scatter)
    # forward — the primitives that make MoE dispatch compute-sparse.
    x = nn.tensor(np.arange(20, dtype=np.float32).reshape(5, 4), requires_grad=True)
    idx = [3, 1, 3]
    g = nn.index_select(x, idx)
    gather_ok = np.allclose(g.numpy(), x.numpy()[idx])
    nn.mean(g).backward()
    gexp = np.zeros((5, 4), np.float32)
    for i in idx:
        gexp[i] += 1.0 / g.numpy().size
    bwd_ok = np.allclose(x.grad(), gexp)          # duplicated index → summed grad

    s = nn.tensor(np.ones((3, 4), np.float32))
    sc = nn.index_add(5, s, [0, 2, 2]).numpy()
    ref = np.zeros((5, 4), np.float32); ref[0] += 1; ref[2] += 2
    scatter_ok = np.allclose(sc, ref)
    print(f"[gather] gather={gather_ok} scatter_add_bwd={bwd_ok} scatter={scatter_ok}")
    return gather_ok and bwd_ok and scatter_ok


def test_moe() -> bool:
    # Mixture-of-Experts top-k routing: the compute-sparse dispatch (each expert
    # runs only on its routed tokens via gather/scatter) must match an independent
    # NumPy reference that evaluates all experts densely, exactly k experts fire
    # per token, and a network with an MoE layer must learn.
    nn.seed(0)
    T, d, E, k = 12, 8, 4, 2
    x = nn.tensor(np.random.default_rng(3).standard_normal((T, d)).astype(np.float32))
    moe = nn.MoELayer(d, E, d_ff=16, top_k=k)
    out = moe(x).numpy()

    logits = np.asarray(nn.matmul(x, moe.Wg).numpy())
    topk = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    mask = np.full(logits.shape, -1e9, np.float32); np.put_along_axis(mask, topk, 0.0, axis=1)
    g = np.exp(logits + mask - (logits + mask).max(1, keepdims=True)); g /= g.sum(1, keepdims=True)
    ref = np.zeros((T, d), np.float32)
    for e in range(E):
        ref += g[:, e:e + 1] * moe.experts[e](x).numpy()
    combine_err = float(np.max(np.abs(out - ref)))
    active = {int(v) for v in (g > 1e-6).sum(1)}          # experts active per token

    rng = np.random.default_rng(5)
    X = nn.tensor(rng.standard_normal((48, d)).astype(np.float32))
    y = [int(v) for v in rng.integers(0, 3, size=48)]
    net = nn.Sequential(nn.MoELayer(d, E, d_ff=16, top_k=k), nn.ReLU(), nn.Linear(d, 3))
    opt = nn.AdamW(net.parameters(), lr=3e-2)
    first = last = None
    for s in range(80):
        opt.zero_grad(); loss = nn.softmax_cross_entropy(net(X), y); loss.backward(); opt.step()
        if s == 0:
            first = loss.item()
        last = loss.item()
    # Load-balancing aux loss: a positive differentiable scalar; adding it to the
    # task loss must still let the network train.
    _ = moe(x)
    aux_val = moe.aux_loss().item()
    net2 = nn.Sequential(nn.MoELayer(d, E, d_ff=16, top_k=k), nn.ReLU(), nn.Linear(d, 3))
    opt2 = nn.AdamW(net2.parameters(), lr=3e-2)
    f2 = l2 = None
    for s in range(80):
        opt2.zero_grad()
        logits = net2(X)
        loss = nn.add(nn.softmax_cross_entropy(logits, y),
                      nn.scale(net2.layers[0].aux_loss(), 0.01))
        loss.backward(); opt2.step()
        if s == 0:
            f2 = loss.item()
        l2 = loss.item()

    print(f"[moe] combine_err={combine_err:.1e} active/token={active}  train loss {first:.3f} -> {last:.3f}"
          f"  aux={aux_val:.3f} aux-train {f2:.3f} -> {l2:.3f}")
    return (combine_err < 1e-5 and active == {k} and last < first * 0.6
            and aux_val > 0.0 and l2 < f2 * 0.7)


def main() -> int:
    # Run each test, catch exceptions, and record which ones returned False so a
    # failure names itself (essential when a threshold tips over only on one
    # platform, e.g. macOS-ARM vs Linux-x86 float differences).
    tests = [
        test_structured_sparse_linear, test_distillation, test_mtp,
        test_hybrid_mamba_transformer, test_qlora, test_mamba,
        test_speculative_decoding, test_gather_scatter, test_moe, test_moe_lm,
        test_bitlinear, test_error_reporting, test_tensor_parallel,
        test_distributed_dp, test_checkpoint, test_sgd, test_leaky_relu,
        test_activation_modules, test_mish, test_log_sigmoid, test_elu,
        test_celu, test_hardswish, test_relu6, test_softshrink, test_tanhshrink, test_hardtanh,
        test_huber_loss, test_mse_loss, test_l1_loss, test_bce_with_logits_loss,
        test_label_smoothing_cross_entropy,
        test_kl_div_loss,
        test_selu, test_softmin, test_gaussian, test_swiglu_mlp, test_geglu_mlp,
        test_glu, test_maxout,
        test_dropout_and_tied,
        test_mlp_xor, test_cnn_quadrant, test_module_api, test_bn_cnn,
        test_transformer_block, test_gpt_module,
    ]
    failed = []
    for fn in tests:
        try:
            if not fn():
                failed.append(fn.__name__)
        except Exception as e:  # a crash is a failure, and must name itself too
            failed.append(f"{fn.__name__} (exception: {e})")
    if not failed:
        print("PASS — vgre.nn trains MLP + CNN from pure Python")
        return 0
    print("FAIL — the following tests did not pass: " + ", ".join(failed))
    return 1


if __name__ == "__main__":
    sys.exit(main())
