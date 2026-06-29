#!/usr/bin/env python3
"""
vgre.nn — build and train arbitrary models from pure Python on VGRE's CPU engine.

  1. An MLP learns XOR (needs a hidden nonlinearity) → ~100% accuracy.
  2. A CNN (conv→relu→max-pool→linear) classifies the bright-patch quadrant of an
     8×8 image → high accuracy.

Both run entirely through the in-tree autograd C ABI — no GPU, no PyTorch/JAX.
"""
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


def main() -> int:
    ok = True
    ok &= test_checkpoint()
    ok &= test_sgd()
    ok &= test_dropout_and_tied()
    ok &= test_mlp_xor()
    ok &= test_cnn_quadrant()
    ok &= test_module_api()
    ok &= test_bn_cnn()
    ok &= test_transformer_block()
    ok &= test_gpt_module()
    if ok:
        print("PASS — vgre.nn trains MLP + CNN from pure Python")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
