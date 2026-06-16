#!/usr/bin/env python3
# Verify vgre::xla::BpeTokenizer::loadGpt2 reproduces the Hugging Face GPT-2
# tokenizer exactly over a battery of strings (whitespace runs, UTF-8,
# contractions, digits, symbols).
#
#   validate_tokenizer.py <gpt2_tokenize_bin> <vocab.json> <merges.txt>
import subprocess, sys
from transformers import GPT2TokenizerFast

binp, vocab, merges = sys.argv[1], sys.argv[2], sys.argv[3]
tok = GPT2TokenizerFast.from_pretrained("gpt2")

tests = [
    "The capital of France is", "Hello, world!", "don't stop",
    "It's 2024, and pi≈3.14.", "  multiple   spaces",
    "new\nline\ttab", "naïve café", "   leading", "trailing   ",
    "Mix3d Numb3rs and CAPS!!!", "a",
]
bad = 0
for s in tests:
    ref = tok(s)["input_ids"]
    out = subprocess.run([binp, vocab, merges, s], capture_output=True, text=True)
    mine = list(map(int, out.stdout.split()))
    if ref != mine:
        bad += 1
        print("DIFF", repr(s), "\n  hf  :", ref, "\n  mine:", mine)
print(f"{len(tests) - bad}/{len(tests)} exact vs Hugging Face GPT-2 tokenizer")
sys.exit(1 if bad else 0)
