#!/usr/bin/env python3
"""Tensor parallelism across REAL processes over the libvgre_nn TCP collective.

The TP identity (per-shard partial products sum to the full matmul via the
differentiable all_reduce) was previously verified only in-process (world=1,
where all_reduce is an identity). This test proves it across two OS processes
with the actual TCP sum in the forward graph.

Row-parallel layer: the contraction dim K is sharded — rank r holds
X[:, cols_r] and W[rows_r, :], computes its partial product, and
Y = all_reduce(matmul(x_shard, W_shard)) reconstructs the full X @ W on every
rank. Backward through all_reduce is identity, so each rank's W_shard.grad()
must equal the corresponding row-block of the full dL/dW.

Checks (world=2, real TCP):
  1. Forward identity: Y on both ranks equals the NumPy full X @ W.
  2. Sharded backward: each rank's first-step W_shard gradient equals the
     matching row-block of the NumPy full softmax-CE gradient.
  3. Sharded training: STEPS iterations of SGD+momentum on the shards; the
     reassembled [W_shard0; W_shard1] must match a single-process (world=1)
     full-model run of the same loop, and the loss must decrease. Both ranks
     see identical losses every step (they share the all-reduced logits).
"""
import os
import socket
import subprocess
import sys
import tempfile

import numpy as np

# ── Shared problem definition (identical, seeded, on every process) ──────────
K_IN, D_OUT, N = 6, 4, 8          # K_IN is the sharded contraction dim
K_HALF = K_IN // 2
STEPS = 3
LR, MOMENTUM = 0.1, 0.9


def _problem():
    r = np.random.default_rng(7)
    W = r.standard_normal((K_IN, D_OUT)).astype(np.float32)
    X = r.standard_normal((N, K_IN)).astype(np.float32)
    y = np.asarray([int(v) for v in r.integers(0, D_OUT, size=N)], dtype=np.int64)
    return W, X, y


def _reference_forward_and_grad(W, X, y):
    """NumPy full-model forward logits and softmax-CE dL/dW (mean reduction)."""
    logits = X @ W
    p = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(p)
    p /= p.sum(axis=1, keepdims=True)
    onehot = np.zeros_like(p)
    onehot[np.arange(N), y] = 1.0
    grad_logits = (p - onehot) / N
    return logits.astype(np.float32), (X.T @ grad_logits).astype(np.float32)


def _find_lib():
    lib = os.environ.get("VGRE_NN_LIB")
    if lib and os.path.isfile(lib):
        return lib
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(os.getcwd(), "src", "xla", "libvgre_nn.so"),
        os.path.join(here, "..", "..", "build", "src", "xla", "libvgre_nn.so"),
        os.path.join(os.getcwd(), "build", "src", "xla", "libvgre_nn.so"),
    ):
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


# ── Worker: sharded (world=2) or full-model reference (world=1) ──────────────
def _run_worker(out_path):
    rank = int(os.environ.get("VGRE_NN_RANK", "0"))
    import vgre.nn as nn

    ws = nn.world_size()
    W_init, X, y = _problem()
    targets = [int(v) for v in y]

    if ws == 2:
        rows = slice(0, K_HALF) if rank == 0 else slice(K_HALF, K_IN)
        x_np = np.ascontiguousarray(X[:, rows])     # [N, K/2]
        w_np = np.ascontiguousarray(W_init[rows])   # [K/2, D_OUT]
    elif ws == 1:
        x_np, w_np = X, W_init                      # full model; all_reduce = id
    else:
        sys.stderr.write(f"rank {rank}: world_size={ws}, expected 1 or 2\n")
        sys.exit(3)

    W = nn.tensor(w_np, requires_grad=True)
    xb = nn.tensor(x_np)
    opt = nn.SGD([W], lr=LR, momentum=MOMENTUM)

    losses = []
    y0 = g0 = None
    for k in range(STEPS):
        out = nn.all_reduce(nn.matmul(xb, W))       # real TCP sum when world=2
        loss = nn.softmax_cross_entropy(out, targets)
        loss.backward()
        if k == 0:
            y0 = out.numpy().copy()                 # reconstructed full logits
            g0 = W.grad().copy()                    # this rank's shard gradient
        losses.append(loss.item())
        opt.step(clip=0.0)                          # clip<=0 → no clipping
        opt.zero_grad()

    np.savez(out_path, y0=y0, g0=g0, w_final=W.numpy().copy(),
             losses=np.asarray(losses, dtype=np.float64))
    sys.stderr.write(
        f"rank {rank}: world={ws} steps={STEPS} "
        f"loss {losses[0]:.4f} → {losses[-1]:.4f}\n")


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _spawn(lib, pypath, out, world, rank=0, port=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = pypath
    env["VGRE_LIB_PATH"] = lib
    env["VGRE_NN_WORLD_SIZE"] = str(world)
    env["VGRE_NN_RANK"] = str(rank)
    if port is not None:
        env["VGRE_NN_MASTER_HOST"] = "127.0.0.1"
        env["VGRE_NN_MASTER_PORT"] = str(port)
    return subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--worker", out], env=env)


# ── Launcher ──────────────────────────────────────────────────────────────────
def _run_launcher():
    lib = _find_lib()
    if lib is None:
        print("SKIP: libvgre_nn.so not found (set VGRE_NN_LIB)")
        sys.exit(77)  # ctest SKIP_RETURN_CODE

    tmp = tempfile.mkdtemp(prefix="vgre_tp_")
    outs = [os.path.join(tmp, f"rank{r}.npz") for r in (0, 1)]
    ref_out = os.path.join(tmp, "ref.npz")

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    bindings = os.path.join(root, "bindings", "python")
    pypath = os.pathsep.join(
        p for p in (bindings, os.environ.get("PYTHONPATH", "")) if p)

    port = _free_port()
    procs = [_spawn(lib, pypath, outs[r], world=2, rank=r, port=port)
             for r in (0, 1)]
    procs.append(_spawn(lib, pypath, ref_out, world=1))
    rcs = [p.wait() for p in procs]
    if any(rc != 0 for rc in rcs):
        print(f"FAIL: worker exit codes = {rcs}")
        sys.exit(1)

    d0, d1, ref = np.load(outs[0]), np.load(outs[1]), np.load(ref_out)
    W_init, X, y = _problem()
    logits_ref, grad_ref = _reference_forward_and_grad(W_init, X, y)

    # 1. Forward identity: both ranks reconstructed the full X @ W over TCP.
    fwd0 = float(np.max(np.abs(d0["y0"] - logits_ref)))
    fwd1 = float(np.max(np.abs(d1["y0"] - logits_ref)))

    # 2. Sharded backward: each rank's grad == its row-block of the full grad.
    bwd0 = float(np.max(np.abs(d0["g0"] - grad_ref[:K_HALF])))
    bwd1 = float(np.max(np.abs(d1["g0"] - grad_ref[K_HALF:])))

    # 3. Sharded SGD+momentum training == single-process full-model training
    #    (elementwise optimizer → the shard trajectories must reassemble the
    #    full trajectory), losses identical across ranks and decreasing.
    w_assembled = np.concatenate([d0["w_final"], d1["w_final"]], axis=0)
    train = float(np.max(np.abs(w_assembled - ref["w_final"])))
    loss_agree = float(np.max(np.abs(d0["losses"] - d1["losses"])))
    dec = d0["losses"][-1] < d0["losses"][0] and ref["losses"][-1] < ref["losses"][0]

    print(f"[dist-tp] forward vs NumPy max|diff| r0={fwd0:.2e} r1={fwd1:.2e}   "
          f"shard-grad vs NumPy r0={bwd0:.2e} r1={bwd1:.2e}")
    print(f"[dist-tp] {STEPS}-step sharded-SGD vs world=1 full model "
          f"max|diff|={train:.2e}   rank-loss agreement={loss_agree:.2e}   "
          f"loss {d0['losses'][0]:.4f}→{d0['losses'][-1]:.4f}")

    ok = (fwd0 < 1e-5 and fwd1 < 1e-5 and bwd0 < 1e-5 and bwd1 < 1e-5
          and train < 1e-4 and loss_agree < 1e-6 and dec)
    print("CROSS-PROCESS TENSOR-PARALLEL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _run_worker(sys.argv[2])
    else:
        _run_launcher()
