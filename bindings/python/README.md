# VGRE — Virtual GPU Runtime

**Run unmodified CUDA on any CPU — no GPU required — and train/run real language
models in the same package.** Self-contained and LLVM-free: the wheel bundles the
native engine, so the only dependency is NumPy.

📖 **Docs:** https://vgre.vercel.app · **Source:** https://github.com/etienne0114/vgre

```bash
python3 -m venv ~/.venvs/vgre && source ~/.venvs/vgre/bin/activate   # PEP 668
pip install vgre
vgre --version
vgre generate --prompt "the "
```

## Run a real pretrained model — fluent output, no training

Load a pretrained **Llama-family** GGUF checkpoint and generate on the CPU. VGRE
dequantizes the weights and runs the forward pass — verified with
SmolLM2-135M-Instruct at **~28 tokens/s** (weight-only int8):

```python
import vgre
tok = vgre.Tokenizer().load_hf("tokenizer.json")          # the model's own tokenizer
lm  = vgre.LanguageModel(vocab=49152, n_layer=30, d_model=576, n_head=9,
                         n_kv_head=3, d_ff=1536, max_seq=2048, tie_embeddings=True)
lm.load_gguf("SmolLM2-135M-Instruct.Q8_0.gguf")           # dequantizes Q8_0 -> f32
lm.set_int8_inference()                                   # ~4x smaller, faster on CPU
ids = tok.encode("Explain what a GPU does in two sentences.")
print(tok.decode(lm.generate(ids, n_new=120, temperature=0.7, top_k=50)))
# → "A GPU (Graphics Processing Unit) is an electronic component that executes
#    calculations on data stored in memory, allowing for faster and more
#    efficient processing of graphics and other digital content."
```

Constraints: a **Llama-architecture** checkpoint, the model's **exact dims** (from
its `config.json`), a **legacy quant** (`Q8_0` / `Q4_0` / `Q4_1`), and rope-theta
10000 (SmolLM2, TinyLlama). Full recipe:
[`pretrained_demo.py`](https://github.com/etienne0114/vgre/blob/main/pretrained_demo.py).

## Train a small model from scratch

```python
import vgre, random
text = open("corpus.txt").read()            # a REAL corpus — 100K+ tokens, not a sentence
tok = vgre.Tokenizer().train(text, num_merges=1024)
ids = tok.encode(text)
lm  = vgre.LanguageModel(vocab=tok.vocab_size, n_layer=4, d_model=256, n_head=8, max_seq=256)

T = min(128, len(ids) - 1)
for step in range(3000):
    lr = vgre.cosine_lr(step, 150, 3000, 3e-4, 3e-5)
    lm.train_batch([(lambda s: (ids[s:s+T], ids[s+1:s+1+T]))(random.randint(0, len(ids)-T-1))
                    for _ in range(16)], lr=lr)
print(tok.decode(lm.generate(tok.encode("Once upon a time"), n_new=100, temperature=0.8, top_k=40)))
```

> A from-scratch model needs **scale** — a tiny corpus produces gibberish because
> the model can only memorize it. For fluent output, load a pretrained model
> (above); to *train*, use a real corpus and enough steps.

## CUDA on CPU

The `Runtime`, `Kernel`, `DeviceArray`, `Stream`, and `Graph` classes JIT-compile
CUDA C and launch it on the CPU (no GPU, no CUDA toolkit). See the
[docs](https://vgre.vercel.app/running-cuda.html).

## Why VGRE (vs. a dedicated inference engine)

For running *one* model as fast as possible, a specialized engine like llama.cpp
is typically faster. VGRE's value is **one self-contained, GPU-free stack** that
emulates CUDA **and** trains **and** runs inference:

| | VGRE | llama.cpp | PyTorch + transformers |
|---|---|---|---|
| Install | `pip install vgre` (+ NumPy) | build a C++ binary | multi-GB |
| Pretrained GGUF on CPU | ✅ ~28 tok/s (135M) | ✅ fastest | ✅ heavy |
| Train / fine-tune | ✅ autograd, AdamW, LoRA | ❌ | ✅ heavyweight |
| **Unmodified CUDA on CPU** | ✅ core purpose | ❌ | ❌ |
| GPU / CUDA / LLVM required | ❌ none | ❌ none | GPU optional |

Reach for VGRE when you want no GPU, minimal dependencies, and one transparent
toolchain for CUDA development, training, and inference — for learning, CI, and
research. Full comparison and guides at **https://vgre.vercel.app**.
