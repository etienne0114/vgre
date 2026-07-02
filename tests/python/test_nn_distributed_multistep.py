#!/usr/bin/env python3
"""Multi-step distributed data-parallel training over the libvgre_nn TCP
collective — the cross-step extension of test_nn_distributed.py (which proves
one synced step).

Two OS processes (rank 0 + rank 1) each run STEPS full training iterations on
their own data shard: forward → backward → all_reduce_gradients() (real TCP
sum + ÷world) → AdamW.step() → zero_grad(). This exercises the optimizer's
moment state under gradient sync and catches cross-step drift.

Checks:
  1. No drift: after every step, rank 0 and rank 1 hold identical weights
     (the collective broadcasts one summed buffer, and AdamW is deterministic,
     so the trajectories must never diverge).
  2. Equivalence: the final distributed weights match a single-process
     (world=1) full-batch run of the *same* vgre.nn training loop.
  3. Training happens: the loss decreases on both ranks and the reference.

Launcher mode (default) spawns the two rank workers plus the world=1 reference
worker, then compares their saved trajectories.
"""
import os
import socket
import subprocess
import sys
import tempfile

import numpy as np

# ── Shared problem definition (identical, seeded, on every process) ──────────
D_IN, D_OUT, N = 6, 4, 8
STEPS = 8
LR = 0.05


def _problem():
    r = np.random.default_rng(0)
    W = r.standard_normal((D_IN, D_OUT)).astype(np.float32)
    X = r.standard_normal((N, D_IN)).astype(np.float32)
    y = np.asarray([int(v) for v in r.integers(0, D_OUT, size=N)], dtype=np.int64)
    return W, X, y


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


# ── Worker: STEPS data-parallel training iterations on this rank's shard ─────
# Also used for the world=1 reference (full batch, all-reduce is identity).
def _run_worker(out_path):
    rank = int(os.environ.get("VGRE_NN_RANK", "0"))
    import vgre.nn as nn

    ws = nn.world_size()
    W_init, X, y = _problem()
    if ws == 1:
        rows = list(range(N))                       # reference: full batch
    elif ws == 2:
        rows = list(range(0, N // 2) if rank == 0 else range(N // 2, N))
    else:
        sys.stderr.write(f"rank {rank}: world_size={ws}, expected 1 or 2\n")
        sys.exit(3)

    W = nn.tensor(W_init, requires_grad=True)
    opt = nn.AdamW([W], lr=LR, weight_decay=0.0)
    targets = [int(y[i]) for i in rows]

    losses, w_hist = [], []
    for _ in range(STEPS):
        xb = nn.tensor(X[rows])
        loss = nn.softmax_cross_entropy(nn.matmul(xb, W), targets)
        loss.backward()
        nn.all_reduce_gradients([W])                # real TCP sync (no-op world=1)
        losses.append(loss.item())
        opt.step(clip=0.0)                          # clip<=0 → no clipping
        opt.zero_grad()
        w_hist.append(W.numpy().copy())

    np.savez(out_path, losses=np.asarray(losses, dtype=np.float64),
             w_hist=np.stack(w_hist))
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

    tmp = tempfile.mkdtemp(prefix="vgre_dist_ms_")
    outs = [os.path.join(tmp, f"rank{r}.npz") for r in (0, 1)]
    ref_out = os.path.join(tmp, "ref.npz")

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    bindings = os.path.join(root, "bindings", "python")
    pypath = os.pathsep.join(
        p for p in (bindings, os.environ.get("PYTHONPATH", "")) if p)

    # 2-process distributed run + single-process full-batch reference.
    port = _free_port()
    procs = [_spawn(lib, pypath, outs[r], world=2, rank=r, port=port)
             for r in (0, 1)]
    procs.append(_spawn(lib, pypath, ref_out, world=1))
    rcs = [p.wait() for p in procs]
    if any(rc != 0 for rc in rcs):
        print(f"FAIL: worker exit codes = {rcs}")
        sys.exit(1)

    d0, d1, ref = np.load(outs[0]), np.load(outs[1]), np.load(ref_out)

    # 1. Cross-step drift: ranks must hold identical weights after EVERY step.
    drift = float(np.max(np.abs(d0["w_hist"] - d1["w_hist"])))

    # 2. Final distributed weights == single-process full-batch trajectory.
    vs_ref = float(np.max(np.abs(d0["w_hist"][-1] - ref["w_hist"][-1])))

    # 3. The loss went down everywhere (the loop actually trains).
    dec0 = d0["losses"][-1] < d0["losses"][0]
    dec1 = d1["losses"][-1] < d1["losses"][0]
    decr = ref["losses"][-1] < ref["losses"][0]

    print(f"[dist-dp-multistep] steps={STEPS}  "
          f"cross-step drift max|diff|={drift:.2e}  "
          f"final W vs world=1 ref max|diff|={vs_ref:.2e}  "
          f"loss r0 {d0['losses'][0]:.4f}→{d0['losses'][-1]:.4f} "
          f"r1 {d1['losses'][0]:.4f}→{d1['losses'][-1]:.4f} "
          f"ref {ref['losses'][0]:.4f}→{ref['losses'][-1]:.4f}")

    ok = drift < 1e-7 and vs_ref < 1e-4 and dec0 and dec1 and decr
    print("MULTI-STEP DISTRIBUTED DP TRAINING PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _run_worker(sys.argv[2])
    else:
        _run_launcher()
