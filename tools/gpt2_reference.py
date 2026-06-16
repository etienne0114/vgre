#!/usr/bin/env python3
# Hugging Face ground truth for the VGRE GPT-2 run. Prints, for a prompt, the
# token ids and the top-5 next-token logits at the last position — to diff
# against tools/gpt2_infer (which loads the same checkpoint via vgre::xla).
import sys, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

ids = tok(prompt)["input_ids"]
with torch.no_grad():
    logits = model(torch.tensor([ids])).logits[0, -1]  # [vocab]
top = torch.topk(logits, 5)

print("IDS " + " ".join(str(i) for i in ids))
for tid, val in zip(top.indices.tolist(), top.values.tolist()):
    print(f"{tid}\t{val:.5f}\t{tok.decode([tid])!r}")
