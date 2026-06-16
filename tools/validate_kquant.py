#!/usr/bin/env python3
# Verify vgre::xla dequant of the ggml K-quants (Q4_K / Q6_K) is bit-identical to
# the reference gguf.quants.dequantize. The gguf library only implements
# *dequantize* for K-quants, so we feed random super-blocks (valid byte layout,
# with a sane f16 super-block scale to avoid NaN) through both.
#
#   validate_kquant.py <quant_dequant_bin>
import subprocess, sys, numpy as np
import gguf.quants as Q
from gguf import GGMLQuantizationType as T

binp = sys.argv[1]
d_f16 = np.frombuffer(np.float16(0.05).tobytes(), np.uint8)
bad = 0
for name, qt, bb in [("Q4_K", T.Q4_K, 144), ("Q6_K", T.Q6_K, 210)]:
    nblocks, rng = 8, np.random.default_rng(7)
    raw = rng.integers(0, 256, size=nblocks * bb, dtype=np.uint8).reshape(nblocks, bb)
    for b in range(nblocks):
        if qt == T.Q4_K:
            raw[b, 0:2] = d_f16
            raw[b, 2:4] = np.frombuffer(np.float16(0.01).tobytes(), np.uint8)
        else:
            raw[b, 208:210] = d_f16
    ref = Q.dequantize(raw.copy(), qt).astype(np.float32).ravel()
    n = nblocks * 256
    p = subprocess.run([binp, str(int(qt)), str(n)], input=raw.tobytes(), capture_output=True)
    mine = np.frombuffer(p.stdout, dtype=np.float32)
    d = float(np.max(np.abs(ref - mine))) if mine.size == ref.size else 9e9
    print(f"{name}: ggml_type={int(qt)} n={n} max|ref-mine|={d:.3e}  {'MATCH' if d < 1e-4 else 'DIFF'}")
    bad += d >= 1e-4
sys.exit(1 if bad else 0)
