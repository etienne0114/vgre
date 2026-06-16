#!/usr/bin/env python3
# Validate vgre::xla GPTQ/AWQ dequant against an independent numpy packer that
# follows the documented AutoGPTQ / AWQ layouts. Packs random int4 data into a
# real safetensors file, runs the loader+dequant tool, and compares.
#
#   validate_gptq.py <gptq_dequant_bin>
import subprocess, sys, tempfile, os, numpy as np
from safetensors.numpy import save_file

binp = sys.argv[1]
rng = np.random.default_rng(11)
IN, OUT, GS = 64, 32, 16
G, OUT8 = IN // GS, OUT // 8
AWQ_INV = [0, 4, 1, 5, 2, 6, 3, 7]

qw = rng.integers(0, 16, size=(IN, OUT), dtype=np.int32)
z  = rng.integers(0, 16, size=(G, OUT), dtype=np.int32)
scales = rng.uniform(0.01, 0.2, size=(G, OUT)).astype(np.float32)

def run(mode, tensors, ref):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "q.safetensors")
        save_file(tensors, path)
        out = subprocess.run([binp, path, "", mode, str(IN), str(OUT), str(G)],
                             capture_output=True, text=True)
        f = out.stdout.split("\t")
        n, s = int(f[0]), float(f[1])
        rsum = float(np.float64(ref).sum())
        ok = (n == ref.size) and abs(s - rsum) < 1e-2 + 1e-4 * abs(rsum)
        print(f"{mode}: n={n} sum={s:.4f} ref={rsum:.4f}  {'MATCH' if ok else 'DIFF'}")
        return ok

# ── GPTQ: qweight [in/8,out] nibbles along in; qzeros [G,out/8]; W=s*(qw-(z+1)) ─
qweight = np.zeros((IN // 8, OUT), dtype=np.int32)
for r in range(IN // 8):
    for t in range(8):
        qweight[r] |= (qw[r * 8 + t] & 0xF) << (4 * t)
qzeros = np.zeros((G, OUT8), dtype=np.int32)
for g in range(G):
    for jj in range(OUT8):
        for t in range(8):
            qzeros[g, jj] |= (z[g, jj * 8 + t] & 0xF) << (4 * t)
ref_gptq = scales[np.repeat(np.arange(G), GS)] * (qw - (z[np.repeat(np.arange(G), GS)] + 1))
ok1 = run("gptq", {"qweight": qweight, "qzeros": qzeros, "scales": scales}, ref_gptq)

# ── AWQ: nibbles along out with AWQ order; W=(qw-z)*s ───────────────────────────
qweight_a = np.zeros((IN, OUT8), dtype=np.int32)
for i in range(IN):
    for jj in range(OUT8):
        for t in range(8):
            qweight_a[i, jj] |= (qw[i, jj * 8 + t] & 0xF) << (4 * AWQ_INV[t])
qzeros_a = np.zeros((G, OUT8), dtype=np.int32)
for g in range(G):
    for jj in range(OUT8):
        for t in range(8):
            qzeros_a[g, jj] |= (z[g, jj * 8 + t] & 0xF) << (4 * AWQ_INV[t])
ref_awq = (qw - z[np.repeat(np.arange(G), GS)]) * scales[np.repeat(np.arange(G), GS)]
ok2 = run("awq", {"qweight": qweight_a, "qzeros": qzeros_a, "scales": scales}, ref_awq)

sys.exit(0 if (ok1 and ok2) else 1)
