// gpt2_tokenize — encode text with a real GPT-2 tokenizer loaded by
// vgre::xla::BpeTokenizer::loadGpt2, printing the token ids (and decoding back
// to stderr for a round-trip check). Used to validate against the Hugging Face
// tokenizer (tools/validate_tokenizer.py).
//
//   gpt2_tokenize <vocab.json> <merges.txt> <text...>
#include "vgre/xla/tokenizer.h"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: gpt2_tokenize <vocab.json> <merges.txt> <text>\n"); return 2; }
    vgre::xla::BpeTokenizer t;
    if (!t.loadGpt2(argv[1], argv[2])) { std::fprintf(stderr, "load failed\n"); return 1; }
    std::string text = argv[3];
    for (int i = 4; i < argc; ++i) { text += ' '; text += argv[i]; }
    auto ids = t.encodeGpt2(text);
    for (size_t i = 0; i < ids.size(); ++i) std::printf("%d%c", ids[i], i + 1 < ids.size() ? ' ' : '\n');
    std::string dec = t.decodeGpt2(ids);
    std::fprintf(stderr, "roundtrip: %s\n", dec == text ? "OK" : "MISMATCH");
    return 0;
}
