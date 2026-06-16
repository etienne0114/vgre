// Byte-level BPE tokenizer — the text-in/text-out front end for generation (L4).
//
// Built from scratch (no external tokenizer lib). The base vocabulary is the 256
// raw bytes, so *any* input round-trips exactly — there is no unknown token and
// UTF-8 is handled implicitly. Merges (learned by train() or imported via
// loadMerges()) compose frequent adjacent token pairs into new tokens, exactly
// like GPT-2 / tiktoken / SentencePiece-BPE. Pre-tokenization splits text into
// "(leading spaces)(non-spaces)" pieces so merges never cross word boundaries.
//
// This gives a real, testable BPE: train() learns merges from a corpus;
// encode()/decode() are inverses; loadMerges() ingests a real model's ordered
// merge list (its byte-pair operands) so the same engine can use that model's
// vocabulary.
#ifndef VGRE_XLA_TOKENIZER_H
#define VGRE_XLA_TOKENIZER_H

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace vgre {
namespace xla {

class BpeTokenizer {
public:
    BpeTokenizer();  // base vocab = 256 single-byte tokens

    // Learn `numMerges` BPE merges from `corpus` (greedy most-frequent-pair).
    void train(const std::string& corpus, int numMerges);

    // Import an ordered merge list — each entry is the raw byte expansions of the
    // two operands to merge (rank = position). Lets the tokenizer adopt a real
    // model's BPE vocabulary. Returns the number of merges accepted.
    int loadMerges(const std::vector<std::pair<std::string, std::string>>& merges);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

    int vocabSize() const { return (int)vocab_.size(); }
    // Raw byte expansion of a token id (empty if out of range).
    std::string tokenBytes(int id) const;

    // ── Real GPT-2 / GPT-style tokenizer file ingestion ──────────────────────
    // Load a shipped GPT-2 tokenizer: vocab.json (token→id, in GPT-2's byte→
    // unicode remapping) + merges.txt (ordered BPE merge rules). Sets up the
    // byte↔unicode table, the model's own vocab-id mapping, GPT-2 pre-tokenization
    // (regex-equivalent word splitting), and the merges. After this,
    // encodeGpt2/decodeGpt2 emit/consume the model's exact token ids.
    bool loadGpt2(const std::string& vocabJsonPath, const std::string& mergesTxtPath);
    bool gpt2Ready() const { return gpt2_; }
    std::vector<int> encodeGpt2(const std::string& text) const;
    std::string decodeGpt2(const std::vector<int>& ids) const;

private:
    // Each token id maps to its byte expansion; ids 0..255 are single bytes.
    std::vector<std::string> vocab_;
    // (left id, right id) -> merged id, in creation order (smaller id = higher
    // priority during encode).
    std::map<std::pair<int, int>, int> merge_;

    int addMerge(int a, int b);                       // create/return merged id
    std::vector<std::vector<int>> pretokenize(const std::string& text) const;
    void applyMerges(std::vector<int>& piece) const;  // greedy BPE on one piece

    // GPT-2 mode state (populated by loadGpt2).
    bool gpt2_ = false;
    std::vector<int> byteToCp_;                       // byte → remap codepoint (256)
    std::map<int, int> cpToByte_;                     // remap codepoint → byte
    std::map<std::string, int> bytesToVocabId_;       // token byte-expansion → gpt2 id
    std::map<int, std::string> vocabIdToBytes_;
    std::vector<std::vector<int>> pretokenizeGpt2(const std::string& text) const;
};

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_TOKENIZER_H
