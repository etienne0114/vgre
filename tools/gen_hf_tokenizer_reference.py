#!/usr/bin/env python3
"""Generate tests/data/hf_tokenizer_ref_*.json — authoritative token ids from
the real Hugging Face `tokenizers` library, used by tests/xla/test_tokenizer_hf.cpp
to verify VGRE's from-scratch tokenizer.json engine token-for-token.

Usage (needs `pip install tokenizers`):

    python3 tools/gen_hf_tokenizer_reference.py <name> <tokenizer.json> <out.json>

e.g.
    python3 tools/gen_hf_tokenizer_reference.py gpt2 \
        ~/.vgre_cache/hf_gpt2/tokenizer.json tests/data/hf_tokenizer_ref_gpt2.json
"""
import json
import sys

from tokenizers import Tokenizer

# Deliberately nasty corpus: contractions (both cases), long digit runs,
# Latin/Greek/Cyrillic/CJK letters, Arabic-Indic digits, emoji, punctuation
# runs, tabs, NBSP, multi-space, multi-newline, and mixed-script words.
CASES = [
    "Hello, world!",
    "I'm sure it's John's — isn't it? He'LL win. THEY'RE here. we've I'd",
    "The year 2026 has 365 days; pi is 3.14159265358979, and 1234567890 is big.",
    "café naïve façade Zürich groß œuvre",
    "Ελληνικά русский текст 中文分词测试 日本語のテキスト 한국어 텍스트",
    "٠١٢٣٤٥٦٧٨٩ arabic digits and १२३ devanagari",
    "hi🚀there 🎉🎉 emoji✨run",
    "func(x, y) -> {return x*y;} // comment #hashtag @mention $100 <tag>",
    "line one\nline two\n\nparagraph two\r\nwindows line\n\n\n",
    "  leading spaces\tand\ttabs   multiple   spaces  ",
    "non breaking thin　ideographic spaces",
    "MixedCASE wordsWith2Numbers and snake_case and kebab-case",
    "...!!!???;;;:::((()))[[]]{{}}",
    "a\nb c\n d\n\ne  \n  f",
]


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    name, tok_path, out_path = sys.argv[1:4]
    tok = Tokenizer.from_file(tok_path)

    cases = list(CASES)
    # Special-token handling (added tokens are model-specific).
    if name == "gpt2":
        cases.append("Document one.<|endoftext|>Document two.")
    elif name.startswith("qwen"):
        cases.append("<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n")

    out = {"name": name, "cases": []}
    for text in cases:
        ids = tok.encode(text).ids
        out["cases"].append({"text": text, "ids": ids})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=True, indent=1)
    print(f"{name}: {len(cases)} cases -> {out_path}")


if __name__ == "__main__":
    main()
