#!/usr/bin/env python3
"""LoRA adapters (vgre.nn.LoRALinear) + HNSW vector index (vgre.vector).

The two additions that make the lightweight stack a complete local-AI engine:
CPU fine-tuning via low-rank adapters, and dependency-free ANN retrieval.

LoRA checks (all compute in the C++ autograd):
  1. gradient correctness: dL/dA and dL/dB match the NumPy analytic gradients
     of  y = x @ W + (alpha/r)·(x @ A) @ B  under softmax-CE
  2. the frozen base weight W never changes during training
  3. 30 AdamW steps on (A, B) only → loss decreases
  4. merge() folds the adapter into W with identical forward output
  5. adapter save/load roundtrip restores the exact same function

VectorIndex checks:
  6. recall@5 ≥ 0.9 vs NumPy brute force (cosine, 500×32)
  7. save/load returns identical results; default ids count upward
"""
import os
import sys
import tempfile

import numpy as np

# Self-bootstrap (the suite's global ENVIRONMENT setter strips PYTHONPATH):
# put the in-tree bindings on sys.path and point VGRE_LIB_PATH at the built
# libvgre_nn.so before importing vgre.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "bindings", "python"))
if not os.environ.get("VGRE_LIB_PATH"):
    for _cand in (os.environ.get("VGRE_NN_LIB", ""),
                  os.path.join(_ROOT, "build", "src", "xla", "libvgre_nn.so"),
                  os.path.join(os.getcwd(), "src", "xla", "libvgre_nn.so")):
        if _cand and os.path.isfile(_cand):
            os.environ["VGRE_LIB_PATH"] = os.path.abspath(_cand)
            break


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def main():
    try:
        import vgre.nn as nn
        from vgre.vector import VectorIndex
        nn._require()
    except Exception as e:
        print(f"SKIP: native library unavailable ({e})")
        return 77

    rng = np.random.default_rng(3)
    D_IN, D_OUT, R, ALPHA, N = 10, 5, 4, 8.0, 16
    X = rng.standard_normal((N, D_IN)).astype(np.float32)
    y = [int(v) for v in rng.integers(0, D_OUT, size=N)]

    nn.seed(11)
    layer = nn.LoRALinear(D_IN, D_OUT, r=R, alpha=ALPHA, bias=False)
    W0 = layer.W.numpy().copy()
    A0 = layer.A.numpy().copy()
    assert np.all(layer.B.numpy() == 0.0), "B must start at zero"
    s = layer.scaling

    # ── 1. Gradient correctness vs NumPy analytic ────────────────────────────
    xb = nn.tensor(X)
    loss = nn.softmax_cross_entropy(layer(xb), y)
    loss.backward()
    gA, gB = layer.A.grad(), layer.B.grad()

    logits = X @ W0 + s * (X @ A0) @ layer.B.numpy()
    p = _softmax(logits)
    onehot = np.zeros_like(p)
    onehot[np.arange(N), y] = 1.0
    G = (p - onehot) / N                       # dL/dlogits (mean reduction)
    gA_ref = s * X.T @ G @ layer.B.numpy().T   # dL/dA
    gB_ref = s * (X @ A0).T @ G                # dL/dB
    eA = float(np.max(np.abs(gA - gA_ref)))
    eB = float(np.max(np.abs(gB - gB_ref)))
    print(f"[lora] grad-vs-numpy max|diff|  dA={eA:.2e}  dB={eB:.2e}")
    if eA > 1e-5 or eB > 1e-5:
        print("FAIL: LoRA gradients wrong")
        return 1

    # ── 2+3. Train the adapter only; base stays frozen; loss decreases ──────
    opt = nn.AdamW(layer.adapter_parameters(), lr=0.05, weight_decay=0.0)
    opt.zero_grad()
    losses = []
    for _ in range(30):
        loss = nn.softmax_cross_entropy(layer(nn.tensor(X)), y)
        loss.backward()
        losses.append(loss.item())
        opt.step(clip=0.0)
        opt.zero_grad()
    print(f"[lora] 30-step adapter training loss {losses[0]:.4f} → {losses[-1]:.4f}")
    if not losses[-1] < 0.5 * losses[0]:
        print("FAIL: adapter training did not reduce the loss")
        return 1
    if float(np.max(np.abs(layer.W.numpy() - W0))) != 0.0:
        print("FAIL: frozen base weight changed during training")
        return 1
    if float(np.max(np.abs(layer.A.numpy() - A0))) == 0.0:
        print("FAIL: adapter A never updated")
        return 1

    # ── 4. merge(): identical forward, delta folded into W ──────────────────
    y_adapter = layer(nn.tensor(X)).numpy()
    layer.merge()
    y_merged = layer(nn.tensor(X)).numpy()
    e_merge = float(np.max(np.abs(y_adapter - y_merged)))
    changed = float(np.max(np.abs(layer.W.numpy() - W0)))
    print(f"[lora] merge forward max|diff|={e_merge:.2e}  (W moved by {changed:.3f})")
    if e_merge > 1e-5 or changed == 0.0:
        print("FAIL: merge broke the forward pass")
        return 1

    # ── 5. Adapter save/load restores the exact function ────────────────────
    nn.seed(22)
    donor = nn.LoRALinear(D_IN, D_OUT, r=R, alpha=ALPHA, bias=False,
                          base_weight=W0)
    opt2 = nn.AdamW(donor.adapter_parameters(), lr=0.05, weight_decay=0.0)
    for _ in range(5):
        l2 = nn.softmax_cross_entropy(donor(nn.tensor(X)), y)
        l2.backward()
        opt2.step(clip=0.0)
        opt2.zero_grad()
    tmp = tempfile.mkdtemp(prefix="vgre_lora_")
    apath = os.path.join(tmp, "adapter.npz")
    donor.save_adapter(apath)
    clone = nn.LoRALinear(D_IN, D_OUT, r=R, alpha=1.0, bias=False,
                          base_weight=W0)
    clone.load_adapter(apath)
    e_rt = float(np.max(np.abs(donor(nn.tensor(X)).numpy()
                               - clone(nn.tensor(X)).numpy())))
    print(f"[lora] adapter save/load forward max|diff|={e_rt:.2e}")
    if e_rt > 1e-6:
        print("FAIL: adapter roundtrip mismatch")
        return 1

    # ── 6. VectorIndex recall vs brute force ─────────────────────────────────
    NV, DV, K = 500, 32, 5
    base = rng.standard_normal((NV, DV)).astype(np.float32)
    ix = VectorIndex(dim=DV, metric="cosine")
    ix.add(base)                                # default ids 0..NV-1
    assert len(ix) == NV
    bn = base / np.linalg.norm(base, axis=1, keepdims=True)
    hits = total = 0
    queries = rng.standard_normal((20, DV)).astype(np.float32)
    for q in queries:
        qn = q / np.linalg.norm(q)
        truth = set(np.argsort(1.0 - bn @ qn)[:K].tolist())
        ids, dists = ix.search(q, k=K, ef=100)
        assert np.all(np.diff(dists) >= 0)
        hits += len(truth.intersection(ids.tolist()))
        total += K
    recall = hits / total
    print(f"[vector] recall@{K} = {recall:.3f} (n={NV}, dim={DV})")
    if recall < 0.9:
        print("FAIL: vector recall too low")
        return 1

    # ── 7. Save/load identical results ───────────────────────────────────────
    vpath = os.path.join(tmp, "index.vgvx")
    ix.save(vpath)
    ix2 = VectorIndex.load(vpath)
    assert len(ix2) == NV and ix2.dim == DV
    for q in queries[:5]:
        a_ids, a_d = ix.search(q, k=K, ef=100)
        b_ids, b_d = ix2.search(q, k=K, ef=100)
        assert np.array_equal(a_ids, b_ids) and np.array_equal(a_d, b_d)
    print("[vector] save/load roundtrip identical")

    print("LORA + VECTOR-INDEX PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
