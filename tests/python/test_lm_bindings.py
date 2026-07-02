#!/usr/bin/env python3
"""
End-to-end test of the VGRE Python LM bindings (vgre.Tokenizer / vgre.LanguageModel).

Trains a BPE tokenizer + a tiny GPT on an embedded public-domain corpus, checks
the loss drops, that greedy generation reproduces learned text, and that a
checkpoint round-trips — all through the ctypes wheel API, fully offline.

Run via ctest (sets LD_LIBRARY_PATH + PYTHONPATH) or:
    PYTHONPATH=bindings/python LD_LIBRARY_PATH=build python3 tests/python/test_lm_bindings.py
"""
import os
import random
import sys

# Allow running from the source tree without installing the wheel.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "bindings", "python"))

try:
    import vgre
except Exception as e:  # pragma: no cover
    print(f"SKIP: cannot import vgre ({e})")
    sys.exit(77)

if not vgre.NATIVE_AVAILABLE:
    print("SKIP: libvgre not found (set VGRE_LIB_PATH / LD_LIBRARY_PATH)")
    sys.exit(77)


def main() -> int:
    text = ("Shall I compare thee to a summer's day? "
            "Thou art more lovely and more temperate: ") * 24

    tok = vgre.Tokenizer().train(text, num_merges=150)
    ids = tok.encode(text)
    assert tok.decode(ids) == text, "tokenizer must round-trip"
    V = max(ids) + 1
    print(f"[1] vocab={tok.vocab_size} tokens={len(ids)}")

    lm = vgre.LanguageModel(vocab=V, n_layer=2, d_model=64, n_head=4,
                            d_ff=128, max_seq=32, seed=3)
    print(f"[2] params={lm.num_parameters}")

    random.seed(0)
    T, first, last = 24, None, None
    for _ in range(150):
        s = random.randint(0, len(ids) - T - 1)
        loss = lm.train_step(ids[s:s + T], ids[s + 1:s + 1 + T], lr=3e-3)
        first = loss if first is None else first
        last = loss
    print(f"[3] loss {first:.3f} -> {last:.3f}")
    assert last < first * 0.5, "model must learn"

    # Forward-only loss() must not perturb training (no optimizer update) and
    # should be low after training.
    vl = lm.loss(ids[:T], ids[1:T + 1])
    print(f"[3b] forward-only loss = {vl:.4f}")
    assert vl >= 0.0 and vl < first, "loss() should report a valid, post-training loss"

    # Mini-batch training (gradient accumulation) also reduces loss.
    bf, bl = None, None
    for _ in range(20):
        batch = [(ids[s:s + T], ids[s + 1:s + 1 + T])
                 for s in (random.randint(0, len(ids) - T - 1) for _ in range(4))]
        bl = lm.train_batch(batch, lr=2e-3)
        bf = bl if bf is None else bf
    print(f"[3c] mini-batch loss {bf:.4f} -> {bl:.4f}")
    assert bl <= bf, "train_batch should not increase loss"

    gen = lm.generate(tok.encode("Shall I compare"), n_new=12, temperature=0.0)
    decoded = tok.decode(gen)
    print(f"[4] generated: {decoded!r}")
    assert "summer" in decoded, "greedy generation should reproduce learned text"

    ckpt = "/tmp/vgre_lm_bindings.safetensors"
    lm.save(ckpt)
    lm2 = vgre.LanguageModel(vocab=V, n_layer=2, d_model=64, n_head=4,
                             d_ff=128, max_seq=32, seed=999)
    lm2.load(ckpt)
    g2 = lm2.generate(tok.encode("Shall I compare"), n_new=12, temperature=0.0)
    assert gen == g2, "checkpoint reload must reproduce generation"
    print("[5] checkpoint round-trip OK")

    # HF tokenizer.json binding (byte-level BPE family): exact ids vs the
    # checked-in HF `tokenizers` reference, if the real fixture is present.
    hf = os.path.expanduser("~/.vgre_cache/hf_gpt2/tokenizer.json")
    if os.path.exists(hf):
        import json
        t2 = vgre.Tokenizer().load_hf(hf)
        assert t2.vocab_size == 50257, "gpt2 vocab size via load_hf"
        assert t2.special_id("<|endoftext|>") == 50256, "gpt2 special id"
        with open(os.path.join(ROOT, "tests", "data",
                               "hf_tokenizer_ref_gpt2.json")) as f:
            ref = json.load(f)
        for case in ref["cases"]:
            got = t2.encode(case["text"])
            assert got == case["ids"], f"HF id mismatch on {case['text']!r}"
            assert t2.decode(got) == case["text"], "HF decode roundtrip"
        print(f"[6] load_hf: {len(ref['cases'])} GPT-2 reference cases exact")
    else:
        print("[6] load_hf fixture absent — skipped")

    print("PASS — VGRE-LM Python bindings train, generate, and checkpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
