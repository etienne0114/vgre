// Byte-level BPE tokenizer — see include/vgre/xla/tokenizer.h.

#include "vgre/xla/tokenizer.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "vgre/common/json.h"

namespace vgre {
namespace xla {

namespace {
inline bool isSpace(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
inline bool isAsciiLetter(unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
inline bool isAsciiDigit(unsigned char c) { return c >= '0' && c <= '9'; }
// GPT-2 \p{L}: ASCII letters plus any non-ASCII byte (UTF-8 letters stay together).
inline bool isLetterByte(unsigned char c) { return isAsciiLetter(c) || c >= 0x80; }

// Decode a UTF-8 string to codepoints.
std::vector<int> utf8Decode(const std::string& s) {
    std::vector<int> cps;
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        int cp, len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
        else { cp = c & 0x07; len = 4; }
        for (int k = 1; k < len && i + k < n; ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        cps.push_back(cp);
        i += len;
    }
    return cps;
}
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

// ── Real GPT-2 tokenizer ingestion ───────────────────────────────────────────

bool BpeTokenizer::loadGpt2(const std::string& vocabJsonPath, const std::string& mergesTxtPath) {
    // GPT-2 byte→unicode table (bytes_to_unicode): printable byte ranges map to
    // themselves; the rest map to 256+n. Build byte↔codepoint maps.
    byteToCp_.assign(256, 0);
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
    std::vector<int> cs = bs;
    int n = 0;
    std::vector<char> inBs(256, 0);
    for (int b : bs) inBs[b] = 1;
    for (int b = 0; b < 256; ++b)
        if (!inBs[b]) { bs.push_back(b); cs.push_back(256 + n); ++n; }
    for (size_t i = 0; i < bs.size(); ++i) { byteToCp_[bs[i]] = cs[i]; cpToByte_[cs[i]] = bs[i]; }

    // Codepoints (a token string) → raw byte expansion.
    auto cpsToBytes = [&](const std::string& s) {
        std::string out;
        for (int cp : utf8Decode(s)) {
            auto it = cpToByte_.find(cp);
            if (it != cpToByte_.end()) out += (char)(unsigned char)it->second;
        }
        return out;
    };

    // vocab.json: token(remapped) → id.
    std::ifstream vf(vocabJsonPath, std::ios::binary);
    if (!vf) return false;
    std::stringstream ss; ss << vf.rdbuf();
    common::json::Value root;
    if (!common::json::parse(ss.str(), root) || !root.isObject()) return false;
    for (const auto& kv : root.obj) {
        int id = (int)kv.second.asNumber(-1);
        if (id < 0) continue;
        std::string bytes = cpsToBytes(kv.first);
        bytesToVocabId_[bytes] = id;
        vocabIdToBytes_[id] = bytes;
    }

    // merges.txt: ordered "A B" lines (skip a leading #version comment).
    std::ifstream mf(mergesTxtPath);
    if (!mf) return false;
    std::vector<std::pair<std::string, std::string>> merges;
    std::string line;
    while (std::getline(mf, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        merges.emplace_back(cpsToBytes(line.substr(0, sp)), cpsToBytes(line.substr(sp + 1)));
    }
    loadMerges(merges);
    gpt2_ = true;
    return true;
}

// GPT-2 pre-tokenization: contractions, ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`,
// and whitespace runs (a single space attaches to the following word). ASCII-exact;
// UTF-8 letters stay within a word via the >=0x80 rule.
std::vector<std::vector<int>> BpeTokenizer::pretokenizeGpt2(const std::string& text) const {
    std::vector<std::vector<int>> pieces;
    const std::string& t = text;
    size_t i = 0, n = t.size();
    auto emit = [&](size_t a, size_t b) {
        std::vector<int> p;
        for (size_t k = a; k < b; ++k) p.push_back((unsigned char)t[k]);
        pieces.push_back(std::move(p));
    };
    while (i < n) {
        unsigned char c = (unsigned char)t[i];
        // contractions: 's 't 're 've 'm 'll 'd
        if (c == '\'' && i + 1 < n) {
            std::string r = t.substr(i, std::min<size_t>(3, n - i));
            const char* cons[] = {"'re", "'ve", "'ll", "'s", "'t", "'m", "'d"};
            bool done = false;
            for (const char* k : cons)
                if (r.compare(0, std::string(k).size(), k) == 0) { emit(i, i + std::string(k).size()); i += std::string(k).size(); done = true; break; }
            if (done) continue;
        }
        size_t start = i;
        size_t k = i;
        if (c == ' ' && i + 1 < n && !isSpace((unsigned char)t[i + 1])) k = i + 1;  // optional leading space
        unsigned char d = (k < n) ? (unsigned char)t[k] : 0;
        if (k < n && isLetterByte(d)) {
            size_t j = k; while (j < n && isLetterByte((unsigned char)t[j])) ++j; emit(start, j); i = j;
        } else if (k < n && isAsciiDigit(d)) {
            size_t j = k; while (j < n && isAsciiDigit((unsigned char)t[j])) ++j; emit(start, j); i = j;
        } else if (k < n && !isSpace(d)) {
            size_t j = k; while (j < n && !isSpace((unsigned char)t[j]) && !isLetterByte((unsigned char)t[j]) && !isAsciiDigit((unsigned char)t[j])) ++j;
            emit(start, j); i = j;
        } else {  // whitespace run (\s+(?!\S)): a trailing space 0x20 before a word
                  // is left for the next ` ?\X+` rule to attach to that word.
            size_t j = i; while (j < n && isSpace((unsigned char)t[j])) ++j;
            size_t end = j;
            if (j < n && (unsigned char)t[j - 1] == ' ' && j - 1 > i) end = j - 1;
            emit(i, end); i = end;
        }
    }
    return pieces;
}

std::vector<int> BpeTokenizer::encodeGpt2(const std::string& text) const {
    std::vector<int> ids;
    for (auto& piece : pretokenizeGpt2(text)) {
        std::vector<int> p = piece;
        applyMerges(p);
        for (int internalId : p) {
            const std::string& bytes = vocab_[internalId];
            auto it = bytesToVocabId_.find(bytes);
            if (it != bytesToVocabId_.end()) ids.push_back(it->second);
            else  // fallback: emit each byte's vocab id
                for (unsigned char b : bytes) {
                    auto bi = bytesToVocabId_.find(std::string(1, (char)b));
                    if (bi != bytesToVocabId_.end()) ids.push_back(bi->second);
                }
        }
    }
    return ids;
}

std::string BpeTokenizer::decodeGpt2(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        auto it = vocabIdToBytes_.find(id);
        if (it != vocabIdToBytes_.end()) out += it->second;
    }
    return out;
}

}  // namespace xla
}  // namespace vgre
