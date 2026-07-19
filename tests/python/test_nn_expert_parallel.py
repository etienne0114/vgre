#!/usr/bin/env python3
"""Expert-parallel Mixture-of-Experts across REAL processes over the libvgre_nn
TCP collective.

Experts are sharded across ranks: with E experts and world W, rank r holds the
block [r*E/W, (r+1)*E/W). Each rank runs the router (replicated), computes only
its local experts' gate-weighted contributions (a partial [T,d]), and
out = all_reduce(partial) sums the per-rank partials into the full MoE output —
the same trick as row-parallel tensor parallelism, reusing the differentiable
collective. Every expert is seeded by its GLOBAL index, so a 2-rank sharded run
must reproduce a single-process (world=1) run exactly.

Checks (world=2 sharded vs world=1 full, real TCP):
  1. Forward: each rank's all_reduced output equals the single-process output.
  2. Router gradient: Wg is replicated, so each rank holds only a partial
     gradient; their SUM (what all_reduce_gradients would produce) must equal the
     single-process full router gradient.
  3. Sharded expert gradients: each rank's local experts' gradients equal the
     matching experts' gradients from the single-process run.
"""
import os
import socket
import subprocess
import sys
import tempfile

import numpy as np

D, E, K, N, SEED = 5, 4, 2, 10, 3        # d_model==num_classes so out feeds CE


def _problem():
    r = np.random.default_rng(11)
    X = r.standard_normal((N, D)).astype(np.float32)
    y = np.asarray([int(v) for v in r.integers(0, D, size=N)], dtype=np.int64)
    return X, y


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


def _run_worker(out_path):
    import vgre.nn as nn
    X, y = _problem()
    targets = [int(v) for v in y]

    moe = nn.MoELayer(D, E, d_ff=12, top_k=K, expert_parallel=True, seed=SEED)
    xb = nn.tensor(X)
    out = moe(xb)                                 # all_reduce'd full output [N, D]
    loss = nn.softmax_cross_entropy(out, targets)
    loss.backward()

    # Save the reconstructed output, the (partial or full) router gradient, and
    # each LOCAL expert's first-layer weight gradient keyed by its global id.
    saved = {"out": out.numpy().copy(),
             "wg_grad": moe.Wg.grad().copy(),
             "local_ids": np.asarray(moe.local_ids, dtype=np.int64)}
    for li, gid in enumerate(moe.local_ids):
        saved[f"e{gid}"] = moe.experts[li].layers[0].W.grad().copy()
    np.savez(out_path, **saved)
    sys.stderr.write(f"rank {os.environ.get('VGRE_NN_RANK','0')}: "
                     f"world={nn.world_size()} experts={moe.local_ids} loss={loss.item():.4f}\n")


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
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


def _run_launcher():
    lib = _find_lib()
    if lib is None:
        print("SKIP: libvgre_nn.so not found (set VGRE_NN_LIB)")
        sys.exit(77)

    tmp = tempfile.mkdtemp(prefix="vgre_ep_")
    outs = [os.path.join(tmp, f"rank{r}.npz") for r in (0, 1)]
    ref_out = os.path.join(tmp, "ref.npz")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    bindings = os.path.join(root, "bindings", "python")
    pypath = os.pathsep.join(p for p in (bindings, os.environ.get("PYTHONPATH", "")) if p)

    port = _free_port()
    procs = [_spawn(lib, pypath, outs[r], world=2, rank=r, port=port) for r in (0, 1)]
    procs.append(_spawn(lib, pypath, ref_out, world=1))
    rcs = [p.wait() for p in procs]
    if any(rc != 0 for rc in rcs):
        print(f"FAIL: worker exit codes = {rcs}")
        sys.exit(1)

    d0, d1, ref = np.load(outs[0]), np.load(outs[1]), np.load(ref_out)

    # 1. Forward: sharded all_reduced output == single-process output.
    fwd0 = float(np.max(np.abs(d0["out"] - ref["out"])))
    fwd1 = float(np.max(np.abs(d1["out"] - ref["out"])))

    # 2. Router gradient: replicated Wg → each rank partial; their sum == full.
    router = float(np.max(np.abs((d0["wg_grad"] + d1["wg_grad"]) - ref["wg_grad"])))

    # 3. Sharded expert gradients match the single-process experts.
    exp_err = 0.0
    for dat in (d0, d1):
        for gid in dat["local_ids"].tolist():
            exp_err = max(exp_err, float(np.max(np.abs(dat[f"e{gid}"] - ref[f"e{gid}"]))))

    print(f"[dist-ep] forward r0={fwd0:.2e} r1={fwd1:.2e}   "
          f"router-grad sum vs full={router:.2e}   expert-grad={exp_err:.2e}")
    ok = fwd0 < 1e-5 and fwd1 < 1e-5 and router < 1e-5 and exp_err < 1e-5
    print("CROSS-PROCESS EXPERT-PARALLEL PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _run_worker(sys.argv[2])
    else:
        _run_launcher()
