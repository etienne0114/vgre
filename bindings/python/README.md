# VGRE — Python bindings

Run CUDA kernels on CPU, **and** train/run VGRE's own in-tree language model —
all from Python, with no external BLAS, no ML framework, and no gated checkpoint
download. Everything runs through VGRE's in-tree SIMD GEMM + autograd + AdamW.

## Install

```bash
# Build a self-contained wheel (compiles the native libs + bundles them):
bindings/python/build_wheel.sh
pip install bindings/python/dist/vgre-*.whl
```

Or use it in place against a local CMake build:

```bash
export VGRE_LIB_PATH=/path/to/build/libvgre.so
export PYTHONPATH=/path/to/virtual-gpu-runtime/bindings/python
```

## Train and generate with VGRE-LM

```python
import vgre

text = open("corpus.txt").read()                 # any public-domain text
tok = vgre.Tokenizer().train(text, num_merges=512)
ids = tok.encode(text)

lm = vgre.LanguageModel(vocab=tok.vocab_size, n_layer=4, d_model=256,
                        n_head=8, max_seq=256)
print("parameters:", lm.num_parameters)

import random
T = 128
for step in range(2000):
    s = random.randint(0, len(ids) - T - 1)
    loss = lm.train_step(ids[s:s+T], ids[s+1:s+1+T], lr=3e-3)

out = lm.generate(tok.encode("Once upon a time"), n_new=64, temperature=0.8)
print(tok.decode(out))

lm.save("model.safetensors")     # standard safetensors — loads on the VGRE engine
```

## CUDA on CPU

The `Runtime`, `Kernel`, `DeviceArray`, `Stream`, and `Graph` classes drive the
CUDA-emulation runtime (JIT-compile CUDA C, launch on CPU). See the top-level
README and `test_cuda_on_cpu.py`.
