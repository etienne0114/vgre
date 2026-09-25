#!/usr/bin/env python3
"""Post-install smoke test for a VGRE wheel: prove the bundled native library
loads and the in-tree LM trains on CPU. Run after `pip install vgre-*.whl`."""
import sys

import vgre

if not getattr(vgre, "NATIVE_AVAILABLE", False):
    print("FAIL: native library did not load from the wheel", file=sys.stderr)
    sys.exit(1)

tok = vgre.Tokenizer().train("the quick brown fox. " * 40, num_merges=48)
lm = vgre.LanguageModel(vocab=tok.vocab_size, n_layer=2, d_model=48, n_head=4)
seq = tok.encode("the quick brown fox. " * 8)

loss0 = lm.train_step(seq[:-1], seq[1:], lr=3e-3)
loss1 = loss0
for _ in range(10):
    loss1 = lm.train_step(seq[:-1], seq[1:], lr=3e-3)

if not loss1 < loss0:
    print(f"FAIL: loss did not decrease ({loss0} -> {loss1})", file=sys.stderr)
    sys.exit(1)

print(f"wheel OK: native loaded, vocab={tok.vocab_size}, LM trains {loss0:.3f} -> {loss1:.3f}")
