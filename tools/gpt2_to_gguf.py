#!/usr/bin/env python3
# Convert the real GPT-2 safetensors checkpoint to a GGUF with quantized weights,
# using the official `gguf` library — so it can be loaded back through
# vgre::xla::GGUF and run on the engine (proves the int4/int8 path end-to-end on
# a real model). The large 2-D matmul weights are quantized (default Q8_0);
# embeddings / norms / biases stay F16/F32. Tensor names are kept identical to
# the safetensors checkpoint so tools/gpt2_infer can load either.
#
#   python tools/gpt2_to_gguf.py <gpt2.safetensors> <out.gguf> [Q8_0|Q4_0]
import sys, numpy as np, gguf
from gguf import GGUFWriter, GGMLQuantizationType as T
import gguf.quants as Q
from safetensors import safe_open

src, out = sys.argv[1], sys.argv[2]
quant = getattr(T, sys.argv[3]) if len(sys.argv) > 3 else T.Q8_0

w = GGUFWriter(out, "gpt2")
n_quant = 0
with safe_open(src, "numpy") as f:
    for name in f.keys():
        a = f.get_tensor(name).astype(np.float32)
        # Quantize only 2-D matmul weights whose last dim is a multiple of 32;
        # keep everything else (embeddings handled as F16, norms/biases F32).
        is_2d_weight = (a.ndim == 2 and name.endswith(".weight")
                        and "ln_" not in name)
        if name in ("wte.weight", "wpe.weight"):
            w.add_tensor(name, a.astype(np.float16), raw_dtype=T.F16)
        elif is_2d_weight and a.shape[1] % 32 == 0:
            w.add_tensor(name, Q.quantize(a, quant), raw_dtype=quant); n_quant += 1
        elif a.ndim >= 1:
            w.add_tensor(name, a, raw_dtype=T.F32)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"wrote {out}: {n_quant} weights quantized to {quant.name}")
