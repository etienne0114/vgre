#!/usr/bin/env python3
"""Real multi-process data-parallel training over the lightweight libvgre_nn
collective (cluster_collective.cpp).

This is the end-to-end proof that the LLVM-free ML library can train across a
CPU cluster: two OS processes (rank 0 + rank 1) each run a forward+backward on
their OWN data shard, then all_reduce_gradients() averages the gradients across
the two processes over a real TCP socket. After the sync, both ranks must hold
identical gradients, and those must equal the full-batch (8-sample) gradient
computed independently in NumPy.

Launcher mode (default): spawns two `--worker` subprocesses, each pointed at
libvgre_nn.so via VGRE_LIB_PATH with VGRE_NN_WORLD_SIZE=2 / RANK / MASTER_PORT,
then compares their synced gradients to the NumPy reference.
"""
import os
import socket
import subprocess
import sys
import tempfile

import numpy as np

# ── Shared problem definition (identical, seeded, on every rank) ─────────────
D_IN, D_OUT, N = 6, 4, 8


def _problem():
    r = np.random.default_rng(0)
    W = r.standard_normal((D_IN, D_OUT)).astype(np.float32)
    X = r.standard_normal((N, D_IN)).astype(np.float32)
    y = np.asarray([int(v) for v in r.integers(0, D_OUT, size=N)], dtype=np.int64)
    return W, X, y


def _reference_full_grad(W, X, y):
    """Full-batch softmax-cross-entropy gradient w.r.t. W (mean reduction)."""
    logits = X @ W
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    onehot = np.zeros_like(p)
    onehot[np.arange(N), y] = 1.0
    grad_logits = (p - onehot) / N            # mean over batch
    return (X.T @ grad_logits).astype(np.float32)


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


# ── Worker: one rank's shard of the data-parallel step ───────────────────────
def _run_worker(out_path):
    rank = int(os.environ["VGRE_NN_RANK"])
    import vgre.nn as nn

    ws = nn.world_size()
    if ws != 2:
        sys.stderr.write(f"rank {rank}: world_size={ws}, expected 2\n")
        sys.exit(3)

    W_init, X, y = _problem()
    rows = range(0, N // 2) if rank == 0 else range(N // 2, N)
    rows = list(rows)

    W = nn.tensor(W_init, requires_grad=True)
    xb = nn.tensor(X[rows])
    loss = nn.softmax_cross_entropy(nn.matmul(xb, W), [int(y[i]) for i in rows])
    loss.backward()

    # Real cross-process gradient sync (sum over ranks, ÷ world_size).
    nn.all_reduce_gradients([W])
    g = W.grad()
    np.save(out_path, g)
    sys.stderr.write(
        f"rank {rank}: shard={rows} world={ws} synced-grad-norm={np.linalg.norm(g):.5f}\n"
    )


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ── Launcher: spawn 2 ranks, compare their synced grads to NumPy reference ───
def _run_launcher():
    lib = _find_lib()
    if lib is None:
        print("SKIP: libvgre_nn.so not found (set VGRE_NN_LIB)")
        sys.exit(77)  # ctest SKIP_RETURN_CODE

    port = _free_port()
    tmp = tempfile.mkdtemp(prefix="vgre_dist_")
    outs = [os.path.join(tmp, f"grad{r}.npy") for r in (0, 1)]

    # Ensure workers can import the vgre package regardless of how PYTHONPATH is
    # passed: prepend the in-tree bindings dir (…/bindings/python) derived from
    # this file's location (…/tests/python/test_nn_distributed.py).
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    bindings = os.path.join(root, "bindings", "python")
    pypath = os.pathsep.join(p for p in (bindings, os.environ.get("PYTHONPATH", "")) if p)

    procs = []
    for rank in (0, 1):
        env = os.environ.copy()
        env["PYTHONPATH"] = pypath
        env["VGRE_LIB_PATH"] = lib
        env["VGRE_NN_WORLD_SIZE"] = "2"
        env["VGRE_NN_RANK"] = str(rank)
        env["VGRE_NN_MASTER_HOST"] = "127.0.0.1"
        env["VGRE_NN_MASTER_PORT"] = str(port)
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker", outs[rank]],
            env=env))

    rcs = [p.wait() for p in procs]
    if any(rc != 0 for rc in rcs):
        print(f"FAIL: worker exit codes = {rcs}")
        sys.exit(1)

    g0, g1 = np.load(outs[0]), np.load(outs[1])
    W, X, y = _problem()
    g_ref = _reference_full_grad(W, X, y)

    agree = float(np.max(np.abs(g0 - g1)))                 # both ranks identical?
    vs_ref = float(np.max(np.abs(g0 - g_ref)))             # == full-batch grad?
    print(f"[dist-dp] rank0 vs rank1 max|diff|={agree:.2e}   "
          f"synced vs full-batch ref max|diff|={vs_ref:.2e}")

    ok = agree < 1e-6 and vs_ref < 1e-5
    print("DISTRIBUTED DATA-PARALLEL TRAINING PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _run_worker(sys.argv[2])
    else:
        _run_launcher()
