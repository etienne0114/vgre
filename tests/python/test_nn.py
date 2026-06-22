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


def main() -> int:
    ok = True
    ok &= test_mlp_xor()
    ok &= test_cnn_quadrant()
    if ok:
        print("PASS — vgre.nn trains MLP + CNN from pure Python")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
