// Byte-level BPE tokenizer — see include/vgre/xla/tokenizer.h.

#include "vgre/xla/tokenizer.h"

#include <algorithm>
#include <unordered_map>

namespace vgre {
namespace xla {

namespace {
inline bool isSpace(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
}  // namespace

BpeTokenizer::BpeTokenizer() {
    vocab_.resize(256);
    for (int i = 0; i < 256; ++i) vocab_[i] = std::string(1, (char)(unsigned char)i);
}

std::string BpeTokenizer::tokenBytes(int id) const {
    return (id >= 0 && id < (int)vocab_.size()) ? vocab_[id] : std::string();
}

int BpeTokenizer::addMerge(int a, int b) {
    auto key = std::make_pair(a, b);
    auto it = merge_.find(key);
    if (it != merge_.end()) return it->second;
    int id = (int)vocab_.size();
    vocab_.push_back(vocab_[a] + vocab_[b]);
    merge_[key] = id;
    return id;
}

// Split into pieces of "(leading spaces)(following non-spaces)", so BPE merges
// stay within a word (GPT-2 'Ġword' style). Every byte is preserved.
std::vector<std::vector<int>> BpeTokenizer::pretokenize(const std::string& text) const {
    std::vector<std::vector<int>> pieces;
    size_t i = 0, n = text.size();
    while (i < n) {
        std::vector<int> piece;
        while (i < n && isSpace((unsigned char)text[i])) piece.push_back((unsigned char)text[i++]);
        while (i < n && !isSpace((unsigned char)text[i])) piece.push_back((unsigned char)text[i++]);
        pieces.push_back(std::move(piece));
    }
    return pieces;
}

// Greedy BPE: repeatedly merge the adjacent pair with the highest priority
// (smallest merged id = learned earliest), merging all its occurrences, until no
// known pair remains.
void BpeTokenizer::applyMerges(std::vector<int>& piece) const {
    if (piece.size() < 2) return;
    while (true) {
        int bestId = -1;
        std::pair<int, int> bestPair;
        for (size_t i = 0; i + 1 < piece.size(); ++i) {
            auto it = merge_.find({piece[i], piece[i + 1]});
            if (it != merge_.end() && (bestId < 0 || it->second < bestId)) {
                bestId = it->second;
                bestPair = {piece[i], piece[i + 1]};
            }
        }
        if (bestId < 0) break;
        std::vector<int> out;
        out.reserve(piece.size());
        for (size_t i = 0; i < piece.size();) {
            if (i + 1 < piece.size() && piece[i] == bestPair.first && piece[i + 1] == bestPair.second) {
                out.push_back(bestId);
                i += 2;
            } else {
                out.push_back(piece[i]);
                ++i;
            }
        }
        piece.swap(out);
    }
}

void BpeTokenizer::train(const std::string& corpus, int numMerges) {
    std::vector<std::vector<int>> pieces = pretokenize(corpus);
    for (int m = 0; m < numMerges; ++m) {
        // Count adjacent pairs across all pieces.
        std::map<std::pair<int, int>, int64_t> counts;
        for (const auto& p : pieces)
            for (size_t i = 0; i + 1 < p.size(); ++i) counts[{p[i], p[i + 1]}]++;
        if (counts.empty()) break;

        // Most frequent pair; ties broken by the (a,b) ordering (std::map) for
        // deterministic training.
        std::pair<int, int> best{};
        int64_t bestCnt = 0;
        for (const auto& kv : counts)
            if (kv.second > bestCnt) { bestCnt = kv.second; best = kv.first; }
        if (bestCnt < 2) break;  // no repeated pair left to compress

        int newId = addMerge(best.first, best.second);
        for (auto& p : pieces) {
            std::vector<int> out;
            out.reserve(p.size());
            for (size_t i = 0; i < p.size();) {
                if (i + 1 < p.size() && p[i] == best.first && p[i + 1] == best.second) {
                    out.push_back(newId);
                    i += 2;
                } else {
                    out.push_back(p[i]);
                    ++i;
                }
            }
            p.swap(out);
        }
    }
}

int BpeTokenizer::loadMerges(const std::vector<std::pair<std::string, std::string>>& merges) {
    // Map an existing token's byte expansion back to its id (base bytes + any
    // merges already created). Built incrementally as merges are accepted.
    std::unordered_map<std::string, int> byBytes;
    for (int i = 0; i < (int)vocab_.size(); ++i) byBytes.emplace(vocab_[i], i);

    int accepted = 0;
    for (const auto& mp : merges) {
        auto la = byBytes.find(mp.first);
        auto lb = byBytes.find(mp.second);
        if (la == byBytes.end() || lb == byBytes.end()) continue;  // operands not yet known
        int id = addMerge(la->second, lb->second);
        byBytes.emplace(vocab_[id], id);
        ++accepted;
    }
    return accepted;
}

std::vector<int> BpeTokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    for (auto& piece : pretokenize(text)) {
        applyMerges(piece);
        ids.insert(ids.end(), piece.begin(), piece.end());
    }
    return ids;
}

std::string BpeTokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids)
        if (id >= 0 && id < (int)vocab_.size()) out += vocab_[id];
    return out;
}

}  // namespace xla
}  // namespace vgre
