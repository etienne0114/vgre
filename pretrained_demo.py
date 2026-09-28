"""Run a REAL pretrained language model on the CPU with VGRE — fluent output,
no training. Loads SmolLM2-135M-Instruct (a small Llama-architecture model) from
a GGUF checkpoint and generates text.

The model + tokenizer are cached in ~/.cache/vgre_models (outside the repo).
Download them once with:

    mkdir -p ~/.cache/vgre_models && cd ~/.cache/vgre_models
    curl -L -o tokenizer.json \
      https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/tokenizer.json
    curl -L -o SmolLM2-135M-Instruct.Q8_0.gguf \
      https://huggingface.co/QuantFactory/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct.Q8_0.gguf
"""
import os
import time

import vgre

CACHE = os.path.expanduser("~/.cache/vgre_models")
GGUF = os.path.join(CACHE, "SmolLM2-135M-Instruct.Q8_0.gguf")
TOKJSON = os.path.join(CACHE, "tokenizer.json")

for f in (GGUF, TOKJSON):
    if not os.path.exists(f):
        raise SystemExit(f"missing {f} — see the download commands in this file's docstring")

# ── Tokenizer: the model's exact byte-level BPE (with its special tokens) ─────
tok = vgre.Tokenizer()
tok.load_hf(TOKJSON)
print(f"tokenizer: vocab {tok.vocab_size}")

# ── Model: SmolLM2-135M-Instruct config (must match the checkpoint) ───────────
#   30 layers · d_model 576 · 9 heads · 3 KV heads (GQA) · d_ff 1536 ·
#   tied embeddings · rope theta 10000 (VGRE's default).
lm = vgre.LanguageModel(
    vocab=tok.vocab_size, n_layer=30, d_model=576, n_head=9,
    n_kv_head=3, d_ff=1536, max_seq=2048, tie_embeddings=True,
)
t0 = time.time()
lm.load_gguf(GGUF)                     # dequantizes Q8_0 → f32
lm.set_int8_inference()                # weight-only int8: ~4x smaller, faster on CPU
print(f"loaded {lm.num_parameters:,} params in {time.time()-t0:.1f}s")

# ── Prompt (SmolLM2-Instruct chat format) ────────────────────────────────────
def chat(user):
    return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")

prompt = "Explain what a GPU does in two sentences."
ids = tok.encode(chat(prompt))
t0 = time.time()
out = lm.generate(ids, n_new=120, temperature=0.7, top_k=50, top_p=0.9,
                  repetition_penalty=1.1, seed=1234)
dt = time.time() - t0
reply = tok.decode(out[len(ids):]).split("<|im_end|>")[0].strip()

print("\n--- prompt ---")
print(prompt)
print("\n--- generated ---")
print(reply)
print(f"\n({len(out)-len(ids)} tokens in {dt:.1f}s = {(len(out)-len(ids))/dt:.1f} tok/s on CPU)")
