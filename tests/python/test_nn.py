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
    print(f"[qlora] params={len(q.parameters())} base_bits={bits:.2f} init_ok={init_ok} "
          f"grad_ok={grad_ok} frozen={frozen}  fine-tune {first:.3f} -> {last:.3f}")
    return only_ab and bits < 3.0 and init_ok and grad_ok and frozen and last < first * 0.2


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
    ssm = nn.SSMBlock(dim, d_state=Ns)
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
          f"ssm_block_err={ssm_err:.1e}  LM train {first:.3f} -> {last:.3f}  greedy match {match}/{TT}")
    return (fwd_err < 1e-6 and grad_err < 1e-3 and par_err < 1e-6 and ssm_err < 1e-4
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
    ok = True
    ok &= test_mtp()
    ok &= test_hybrid_mamba_transformer()
    ok &= test_qlora()
    ok &= test_mamba()
    ok &= test_speculative_decoding()
    ok &= test_gather_scatter()
    ok &= test_moe()
    ok &= test_moe_lm()
    ok &= test_bitlinear()
    ok &= test_error_reporting()
    ok &= test_tensor_parallel()
    ok &= test_distributed_dp()
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
