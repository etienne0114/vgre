#!/usr/bin/env python3
"""Honest CPU generation benchmark for VGRE's language model.

Measures end-to-end **tokens/second** for KV-cached autoregressive decoding on
the CPU across a few model sizes, plus a memory-mode comparison (fp32 vs bf16 vs
int8 weights, and fp32 vs int8/int4 KV cache). No GPU. Prints a plain table and
machine-readable JSON so results can be tracked over time or in CI.

This exists so the project's claims are grounded in measured numbers rather than
adjectives. It reports what it actually observes on the machine it runs on —
nothing more.

Usage:
    VGRE_LIB_PATH=build/src/xla/libvgre_nn.so python benchmarks/bench_generate.py
    python benchmarks/bench_generate.py --json results.json
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "bindings", "python")))

import vgre  # noqa: E402


def _params(vocab, n_layer, d_model, d_ff):
    # Rough parameter count (embeddings + per-layer attention + MLP), enough to
    # label a size class; not an exact accounting.
    per_layer = 4 * d_model * d_model + 2 * d_model * d_ff
    return vocab * d_model + n_layer * per_layer


def _time_generate(lm, prompt, n_new, reps=3):
    lm.generate(prompt, n_new=4, temperature=0.0)          # warm the caches
    best = 1e30
    for _ in range(reps):
        t0 = time.perf_counter()
        lm.generate(prompt, n_new=n_new, temperature=0.0)
        best = min(best, time.perf_counter() - t0)
    return n_new / best                                     # tokens/sec


def bench_sizes(n_new=64):
    configs = [
        ("tiny",   dict(vocab=256, n_layer=2, d_model=128, n_head=4,  d_ff=512,  max_seq=256)),
        ("small",  dict(vocab=256, n_layer=4, d_model=256, n_head=8,  d_ff=1024, max_seq=256)),
        ("medium", dict(vocab=256, n_layer=6, d_model=512, n_head=8,  d_ff=2048, max_seq=256)),
    ]
    prompt = list(range(16))
    rows = []
    for name, cfg in configs:
        lm = vgre.LanguageModel(seed=0, **cfg)
        tps = _time_generate(lm, prompt, n_new)
        p = _params(cfg["vocab"], cfg["n_layer"], cfg["d_model"], cfg["d_ff"])
        rows.append({"size": name, "params_M": round(p / 1e6, 2),
                     "d_model": cfg["d_model"], "n_layer": cfg["n_layer"],
                     "tokens_per_sec": round(tps, 1)})
        lm.close()
    return rows


def bench_modes(n_new=64):
    cfg = dict(vocab=256, n_layer=4, d_model=256, n_head=8, d_ff=1024, max_seq=256)
    prompt = list(range(16))
    rows = []

    def run(label, setup):
        lm = vgre.LanguageModel(seed=0, **cfg)
        setup(lm)
        rows.append({"mode": label, "tokens_per_sec": round(_time_generate(lm, prompt, n_new), 1)})
        lm.close()

    run("fp32 weights, fp32 KV", lambda lm: None)
    run("bf16 weights",          lambda lm: lm.set_bf16_inference(True))
    run("int8 weights",          lambda lm: lm.set_int8_inference(True))
    run("fp32 weights, int8 KV", lambda lm: lm.set_int8_kv_cache(True))
    run("fp32 weights, int4 KV", lambda lm: lm.set_int4_kv_cache(True))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    print(f"VGRE CPU generation benchmark  (n_new={args.n_new}, no GPU)\n")
    sizes = bench_sizes(args.n_new)
    print(f"  {'size':7} {'params(M)':>10} {'d_model':>8} {'layers':>7} {'tok/s':>8}")
    for r in sizes:
        print(f"  {r['size']:7} {r['params_M']:>10} {r['d_model']:>8} "
              f"{r['n_layer']:>7} {r['tokens_per_sec']:>8}")
    print("\n  memory mode (small model):")
    modes = bench_modes(args.n_new)
    for r in modes:
        print(f"    {r['mode']:24} {r['tokens_per_sec']:>8} tok/s")

    result = {"n_new": args.n_new, "sizes": sizes, "modes": modes}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
