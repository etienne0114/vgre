// Byte-level BPE tokenizer (L4 text front end): exact round-trips for arbitrary
// bytes incl. UTF-8, training that actually compresses, deterministic merges,
// and importing an external ordered merge list.

#include "vgre/xla/tokenizer.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using vgre::xla::BpeTokenizer;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

int main() {
    // ── Untrained tokenizer = raw bytes; still round-trips everything ───────
    {
        BpeTokenizer t;
        CHECK(t.vocabSize() == 256, "base vocab = 256 bytes");
        for (const std::string& s : {std::string("hello world"),
                                     std::string("café \xF0\x9F\x9A\x80 \t\n"),  // UTF-8 + emoji
                                     std::string("\x00\x01\xFF binary", 11)}) {
            auto ids = t.encode(s);
            CHECK(ids.size() == s.size(), "untrained: one id per byte");
            CHECK(t.decode(ids) == s, "untrained round-trip exact");
        }
    }

    // ── Training learns merges and compresses repeated text ─────────────────
    {
        std::string corpus;
        for (int i = 0; i < 200; ++i) corpus += "the quick brown fox ";
        BpeTokenizer t;
        t.train(corpus, /*numMerges=*/50);
        CHECK(t.vocabSize() > 256, "training added merged tokens");

        std::string sample = "the quick brown fox";
        auto ids = t.encode(sample);
        CHECK(t.decode(ids) == sample, "trained round-trip exact");
        CHECK((int)ids.size() < (int)sample.size(), "merges compress trained text");
        // "the" recurs constantly → should collapse to very few tokens.
        CHECK(t.encode("the").size() <= 2, "frequent word merged to <=2 tokens");

        // Determinism: same input → same ids.
        CHECK(t.encode(sample) == ids, "encode deterministic");

        // Unseen text still round-trips (byte fallback, no UNK).
        std::string unseen = "ZZZ \xC3\xA9 123!";
        CHECK(t.decode(t.encode(unseen)) == unseen, "unseen text round-trips");
    }

    // ── Re-training is reproducible across instances ────────────────────────
    {
        std::string corpus;
        for (int i = 0; i < 100; ++i) corpus += "ababab abab ";
        BpeTokenizer a, b;
        a.train(corpus, 20);
        b.train(corpus, 20);
        CHECK(a.vocabSize() == b.vocabSize(), "training reproducible (vocab size)");
        CHECK(a.encode("ababab") == b.encode("ababab"), "training reproducible (encoding)");
    }

    // ── Importing an external ordered merge list (real-model vocab path) ────
    {
        BpeTokenizer t;
        // Merge 'a'+'b' -> "ab", then "ab"+'c' -> "abc" (operands are byte exprs).
        int n = t.loadMerges({{"a", "b"}, {"ab", "c"}});
        CHECK(n == 2, "two merges accepted");
        auto ids = t.encode("abc");
        CHECK(ids.size() == 1 && t.tokenBytes(ids[0]) == "abc", "imported merges apply (abc→1 token)");
        CHECK(t.decode(ids) == "abc", "imported-merge round-trip");
        // A merge whose operands aren't known yet is skipped, not fatal.
        BpeTokenizer t2;
        CHECK(t2.loadMerges({{"xy", "z"}}) == 0, "merge with unknown operand skipped");
    }

    // ── GPT-2 file ingestion (loadGpt2): ASCII letters map to themselves in
    //    GPT-2's byte→unicode table, so a synthetic vocab.json/merges.txt needs
    //    no remapping. Verifies the file parser + vocab-id mapping + merges.
    {
        namespace fs = std::filesystem;
        fs::path vj = fs::temp_directory_path() / "vgre_vocab.json";
        fs::path mt = fs::temp_directory_path() / "vgre_merges.txt";
        { std::ofstream f(vj); f << R"({"a":10,"b":11,"c":12,"ab":20,"abc":30})"; }
        { std::ofstream f(mt); f << "#version: 0.2\na b\nab c\n"; }  // a+b→ab, ab+c→abc

        BpeTokenizer t;
        CHECK(t.loadGpt2(vj.string(), mt.string()), "loadGpt2 parses vocab+merges");
        CHECK(t.gpt2Ready(), "gpt2 mode active");
        // "abc": a+b→ab, ab+c→abc → single vocab id 30.
        auto ids = t.encodeGpt2("abc");
        CHECK(ids == std::vector<int>{30}, "encodeGpt2 applies merges → model vocab id");
        CHECK(t.decodeGpt2(ids) == "abc", "decodeGpt2 round-trip via model ids");
        // "ab" → id 20; a lone "a" → its byte id 10.
        CHECK(t.encodeGpt2("ab") == std::vector<int>{20}, "encodeGpt2 partial merge");
        CHECK(t.encodeGpt2("a") == std::vector<int>{10}, "encodeGpt2 single token");
        fs::remove(vj); fs::remove(mt);
    }

    if (g_fail == 0) { std::printf("test_tokenizer: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_tokenizer: %d FAILURE(S)\n", g_fail);
    return 1;
}
