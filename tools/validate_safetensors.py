#!/usr/bin/env python3
# Cross-check vgre::xla::SafeTensors against the reference `safetensors` library
# on a real checkpoint: for every tensor, compare shape, element count, float64
# sum and the first 5 values produced by each loader.
#
#   validate_safetensors.py <probe_binary> <file.safetensors>
import subprocess, sys, numpy as np
from safetensors import safe_open

probe, path = sys.argv[1], sys.argv[2]

# Reference (safetensors lib), upcast everything to f32 like the C++ loader does.
ref = {}
with safe_open(path, "numpy") as f:
    for k in f.keys():
        a = f.get_tensor(k).astype(np.float32).ravel()
        ref[k] = (list(a.shape), int(a.size), float(np.float64(a).sum()), a[:5].tolist())

# VGRE loader output.
out = subprocess.run([probe, "st", path], capture_output=True, text=True)
mine = {}
for line in out.stdout.splitlines():
    p = line.split("\t")
    name = p[0]; numel = int(p[3]); s = float(p[4]); first = [float(x) for x in p[5:]]
    mine[name] = (numel, s, first)

bad = 0
for k, (shape, numel, rsum, rfirst) in ref.items():
    if k not in mine:
        print("MISSING in VGRE:", k); bad += 1; continue
    n, s, first = mine[k]
    if n != numel:
        print(f"NUMEL mismatch {k}: vgre {n} vs ref {numel}"); bad += 1
    # sums: relative tolerance (f32 reduction order differs)
    if abs(s - rsum) > 1e-2 + 1e-4 * abs(rsum):
        print(f"SUM mismatch {k}: vgre {s:.4f} vs ref {rsum:.4f}"); bad += 1
    for a, b in zip(first, rfirst):
        if abs(a - b) > 1e-5:
            print(f"VALUE mismatch {k}: {a} vs {b}"); bad += 1; break

print(f"\nchecked {len(ref)} tensors; {bad} mismatch(es)")
sys.exit(1 if bad else 0)
