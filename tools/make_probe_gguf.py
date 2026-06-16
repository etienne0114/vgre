#!/usr/bin/env python3
# Write a canonical GGUF (via the official `gguf` library) containing F32 / F16 /
# Q8_0 / Q4_0 tensors, and print the reference dequantized stats. Used to verify
# vgre::xla::GGUF reads real GGUF-format files and dequantizes identically to the
# reference implementation:
#
#   python tools/make_probe_gguf.py out.gguf      # writes file + prints REF rows
#   ./build/checkpoint_probe gguf out.gguf        # VGRE loader rows must match
import sys, numpy as np, gguf
from gguf import GGUFWriter, GGMLQuantizationType as T
import gguf.quants as Q

path = sys.argv[1] if len(sys.argv) > 1 else "probe.gguf"
rng = np.random.default_rng(0)
tensors = {
    "t_f32": (rng.standard_normal((4, 8)).astype(np.float32),  T.F32),
    "t_f16": (rng.standard_normal((2, 16)).astype(np.float32), T.F16),
    "t_q8":  (rng.standard_normal((3, 32)).astype(np.float32), T.Q8_0),
    "t_q4":  (rng.standard_normal((2, 64)).astype(np.float32), T.Q4_0),
}
w = GGUFWriter(path, "probe-arch")
w.add_uint32("answer", 42)
ref = {}
for name, (arr, qt) in tensors.items():
    if qt == T.F32:
        data = arr;                       ref[name] = arr.astype(np.float64)
    elif qt == T.F16:
        data = arr.astype(np.float16);    ref[name] = data.astype(np.float64)
    else:
        data = Q.quantize(arr, qt);       ref[name] = Q.dequantize(data, qt).astype(np.float64).reshape(arr.shape)
    w.add_tensor(name, data, raw_dtype=qt)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

for name in tensors:
    r = ref[name].ravel()
    print(f"REF\t{name}\t{r.size}\t{r.sum():.6f}\t" + "\t".join(f"{v:.6f}" for v in r[:5]))
