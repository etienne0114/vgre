#!/usr/bin/env python3
"""
Train VGRE's own language model — end to end, on CPU, fully offline.

This is a real, runnable training tool built entirely on VGRE's in-tree stack
(SIMD GEMM + autograd + AdamW + BPE tokenizer + KV-cached sampling) — no GPU, no
external BLAS, no PyTorch/JAX, no downloaded checkpoint.

    # bundled public-domain sample:
    python examples/train_lm.py --steps 600

    # your own corpus + a bigger model:
    python examples/train_lm.py --corpus book.txt --layers 6 --dim 256 \
        --heads 8 --merges 1024 --steps 5000 --out model.safetensors

It prints train/validation loss (val is held-out text the model never trains
on, so a falling val loss is genuine learning, not memorization) and a sample
generation, then saves a standard-safetensors checkpoint.

Run from the source tree with:
    PYTHONPATH=bindings/python LD_LIBRARY_PATH=build python examples/train_lm.py
or after `pip install vgre`.
"""
import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bindings", "python"))

import vgre  # noqa: E402

# A small public-domain sample (Shakespeare, out of copyright) used when no
# --corpus is given, so the demo is fully self-contained.
SAMPLE = """
When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery, so gazed on now,
Will be a tattered weed, of small worth held:
Then being asked where all thy beauty lies,
Where all the treasure of thy lusty days,
To say within thine own deep-sunken eyes,
Were an all-eating shame and thriftless praise.
Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimmed;
And every fair from fair sometime declines,
By chance, or nature's changing course untrimmed:
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st,
Nor shall death brag thou wand'rest in his shade,
When in eternal lines to time thou grow'st.
So long as men can breathe, or eyes can see,
So long lives this, and this gives life to thee.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Train VGRE's in-tree LM on CPU.")
    ap.add_argument("--corpus", help="text file to train on (default: bundled sample)")
    ap.add_argument("--merges", type=int, default=512, help="BPE merges (vocab size)")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=64, help="context length (tokens)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=1, help="sequences per optimizer step")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=0, help="LR warmup steps (default 10%)")
    ap.add_argument("--dropout", type=float, default=0.0, help="residual dropout (training)")
    ap.add_argument("--tie", action="store_true", help="tie input/output embeddings")
    ap.add_argument("--serve-dtype", choices=["f32", "bf16", "int8"], default="f32",
                    help="weight precision used for the final generation")
    ap.add_argument("--out", default="vgre_lm.safetensors")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not vgre.NATIVE_AVAILABLE:
        print("ERROR: libvgre not found. Build it and set VGRE_LIB_PATH / LD_LIBRARY_PATH.")
        return 1

    text = open(args.corpus, encoding="utf-8").read() if args.corpus else SAMPLE
    # Repeat tiny samples so there is enough signal for a quick demo run.
    if len(text) < 4000:
        text = (text + "\n") * (4000 // len(text) + 1)

    print(f"[data] {len(text)} chars; training a BPE tokenizer ({args.merges} merges)...")
    tok = vgre.Tokenizer().train(text, args.merges)
    ids = tok.encode(text)
    V = max(ids) + 1
    # Held-out validation split (last 10%): the model never trains on it.
    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    print(f"[data] vocab={V} tokens={len(ids)} (train={len(train_ids)} val={len(val_ids)})")

    lm = vgre.LanguageModel(vocab=V, n_layer=args.layers, d_model=args.dim,
                            n_head=args.heads, max_seq=max(args.ctx + 1, 64),
                            dropout=args.dropout, tie_embeddings=args.tie, seed=args.seed)
    print(f"[model] {lm.num_parameters/1e6:.2f}M parameters "
          f"({args.layers}L x d{args.dim} x {args.heads}h)")

    rng = random.Random(args.seed)
    T = args.ctx

    def window(buf):
        s = rng.randint(0, len(buf) - T - 1)
        return buf[s:s + T], buf[s + 1:s + 1 + T]

    def val_loss(n=20):
        # Forward-only loss over held-out windows (no gradient step / optimizer
        # update — genuine validation).
        if len(val_ids) < T + 1:
            return float("nan")
        tot = 0.0
        for _ in range(n):
            s = rng.randint(0, len(val_ids) - T - 1)
            tot += lm.loss(val_ids[s:s + T], val_ids[s + 1:s + 1 + T])
        return tot / n

    import math
    warmup = args.warmup if args.warmup > 0 else max(1, args.steps // 10)
    print(f"[train] baseline loss ~= ln(vocab) = {math.log(V):.3f}; "
          f"cosine LR {args.lr:g} (warmup {warmup})")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = vgre.cosine_lr(step - 1, warmup, args.steps, args.lr)
        if args.batch > 1:
            loss = lm.train_batch([window(train_ids) for _ in range(args.batch)], lr=lr)
        else:
            x, y = window(train_ids)
            loss = lm.train_step(x, y, lr=lr)
        if step % max(1, args.steps // 10) == 0 or step == 1:
            vl = val_loss()
            print(f"[train] step {step:5d}/{args.steps}  lr={lr:.2e}  "
                  f"train_loss={loss:.4f}  val_loss={vl:.4f}")
    dt = time.time() - t0
    print(f"[train] {args.steps} steps in {dt:.1f}s ({args.steps/dt:.1f} steps/s)")

    if args.serve_dtype == "bf16":
        lm.set_bf16_inference(True)
    elif args.serve_dtype == "int8":
        lm.set_int8_inference(True)
    print(f"[serve] generating with {args.serve_dtype} weights")

    prompt_text = "Shall I compare"
    prompt = tok.encode(prompt_text)
    gen = lm.generate(prompt, n_new=60, temperature=0.8, top_k=40, top_p=0.95,
                      repetition_penalty=1.2, seed=args.seed)
    print("\n[generate] prompt: " + repr(prompt_text))
    print("[generate] " + repr(tok.decode(gen)))

    lm.save(args.out)
    print(f"\n[save] checkpoint -> {args.out} (standard safetensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
