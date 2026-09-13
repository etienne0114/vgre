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

// As above, but also records each codepoint's starting byte offset (plus a
// final sentinel = text size) so pre-tokenized pieces can be sliced as bytes.
void utf8DecodeOffsets(const std::string& s, std::vector<uint32_t>& cps,
                       std::vector<size_t>& off) {
    cps.clear();
    off.clear();
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        uint32_t cp;
        int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
        else { cp = c & 0x07; len = 4; }
        for (int k = 1; k < len && i + k < n; ++k)
            cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
        cps.push_back(cp);
        off.push_back(i);
        i += (size_t)len;
        if (i > n) i = n;
    }
    off.push_back(n);
}

// ── Codepoint classification: real UCD range tables (see the generator at
// tools/gen_unicode_tables.py) — exact \p{L} / \p{N}, matching the regex
// engines behind Hugging Face `tokenizers`. ────────────────────────────────
struct CpRange { uint32_t lo, hi; };
#include "vgre/xla/unicode_tables.inc"

bool inRanges(uint32_t cp, const CpRange* r, size_t n) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp > r[mid].hi) lo = mid + 1;
        else hi = mid;
    }
    return lo < n && cp >= r[lo].lo && cp <= r[lo].hi;
}
inline bool isUL(uint32_t cp) { return inRanges(cp, kUnicodeLetterRanges, kUnicodeLetterRangesCount); }
inline bool isUN(uint32_t cp) { return inRanges(cp, kUnicodeNumberRanges, kUnicodeNumberRangesCount); }
// Unicode White_Space (regex \s).
inline bool isUWs(uint32_t cp) {
    switch (cp) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20:
        case 0x85: case 0xA0: case 0x1680: case 0x2028: case 0x2029:
        case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return cp >= 0x2000 && cp <= 0x200A;
    }
}
}  // namespace

const char* tokenizerErrorString(TokenizerError e) {
    switch (e) {
        case TokenizerError::Ok:                  return "ok";
        case TokenizerError::InvalidUtf8:         return "invalid UTF-8 byte sequence";
        case TokenizerError::OverlongUtf8:        return "overlong (non-minimal) UTF-8 encoding";
        case TokenizerError::SurrogateCodepoint:  return "UTF-16 surrogate codepoint in UTF-8";
        case TokenizerError::CodepointOutOfRange: return "codepoint above U+10FFFF";
        case TokenizerError::InvalidTokenId:      return "token id not in vocabulary";
        case TokenizerError::ByteNotInVocab:      return "raw byte missing from model vocabulary";
        case TokenizerError::FileNotFound:        return "tokenizer file not found";
        case TokenizerError::ParseError:          return "tokenizer file parse error";
        case TokenizerError::UnsupportedModel:    return "unsupported tokenizer model/normalizer/split";
        case TokenizerError::InvalidVocabId:      return "invalid vocabulary id";
        case TokenizerError::DuplicateVocabId:    return "duplicate vocabulary id";
        case TokenizerError::InvalidMerge:        return "malformed merge rule";
        case TokenizerError::InconsistentState:   return "inconsistent loaded tokenizer state";
        case TokenizerError::NotLoaded:           return "tokenizer not loaded for this API";
    }
    return "unknown error";
}

// Strict RFC 3629 UTF-8 validation via the Unicode "well-formed byte sequences"
// table: rejects overlong forms, surrogate codepoints and anything > U+10FFFF,
// each with a distinct error code and the byte offset of the first offender.
bool BpeTokenizer::isValidUtf8(const std::string& s, TokenizerError* why, size_t* atByte) {
    auto reject = [&](TokenizerError e, size_t at) {
        if (why) *why = e;
        if (atByte) *atByte = at;
        return false;
    };
    const unsigned char* p = reinterpret_cast<const unsigned char*>(s.data());
    const size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        const unsigned char b0 = p[i];
        if (b0 < 0x80) { ++i; continue; }             // ASCII

        int len;
        unsigned char lo1, hi1;                        // legal range of the 1st cont. byte
        if (b0 == 0xC0 || b0 == 0xC1) return reject(TokenizerError::OverlongUtf8, i);
        else if (b0 >= 0xC2 && b0 <= 0xDF) { len = 2; lo1 = 0x80; hi1 = 0xBF; }
        else if (b0 == 0xE0)               { len = 3; lo1 = 0xA0; hi1 = 0xBF; }  // <A0 overlong
        else if (b0 >= 0xE1 && b0 <= 0xEC) { len = 3; lo1 = 0x80; hi1 = 0xBF; }
        else if (b0 == 0xED)               { len = 3; lo1 = 0x80; hi1 = 0x9F; }  // >9F surrogate
        else if (b0 >= 0xEE && b0 <= 0xEF) { len = 3; lo1 = 0x80; hi1 = 0xBF; }
        else if (b0 == 0xF0)               { len = 4; lo1 = 0x90; hi1 = 0xBF; }  // <90 overlong
        else if (b0 >= 0xF1 && b0 <= 0xF3) { len = 4; lo1 = 0x80; hi1 = 0xBF; }
        else if (b0 == 0xF4)               { len = 4; lo1 = 0x80; hi1 = 0x8F; }  // >8F out of range
        else if (b0 >= 0xF5)               return reject(TokenizerError::CodepointOutOfRange, i);
        else                               return reject(TokenizerError::InvalidUtf8, i);  // lone continuation

        if (i + (size_t)len > n) return reject(TokenizerError::InvalidUtf8, i);   // truncated
        const unsigned char b1 = p[i + 1];
        if (b1 < lo1 || b1 > hi1) {
            // Classify the boundary violation precisely for E0/ED/F0/F4 leads.
            if (b0 == 0xE0 && b1 >= 0x80 && b1 <= 0x9F) return reject(TokenizerError::OverlongUtf8, i);
            if (b0 == 0xED && b1 >= 0xA0 && b1 <= 0xBF) return reject(TokenizerError::SurrogateCodepoint, i);
            if (b0 == 0xF0 && b1 >= 0x80 && b1 <= 0x8F) return reject(TokenizerError::OverlongUtf8, i);
            if (b0 == 0xF4 && b1 >= 0x90 && b1 <= 0xBF) return reject(TokenizerError::CodepointOutOfRange, i);
            return reject(TokenizerError::InvalidUtf8, i);
        }
        for (int k = 2; k < len; ++k)
            if (p[i + k] < 0x80 || p[i + k] > 0xBF) return reject(TokenizerError::InvalidUtf8, i);
        i += (size_t)len;
    }
    if (why) *why = TokenizerError::Ok;
    return true;
}

bool BpeTokenizer::setError(TokenizerError e, std::string msg) const {
    err_ = e;
    errMsg_ = std::move(msg);
    return e == TokenizerError::Ok;
}

BpeTokenizer::BpeTokenizer() {
    vocab_.resize(256);
    for (int i = 0; i < 256; ++i) vocab_[i] = std::string(1, (char)(unsigned char)i);
}

void BpeTokenizer::reset() {
    vocab_.assign(256, std::string());
    for (int i = 0; i < 256; ++i) vocab_[i] = std::string(1, (char)(unsigned char)i);
    merge_.clear();
    gpt2_ = false;
    hf_ = false;
    byteToCp_.clear();
    cpToByte_.clear();
    bytesToVocabId_.clear();
    vocabIdToBytes_.clear();
    split_ = SplitRule{};
    specials_.clear();
    specialById_.clear();
    added_.clear();
    hfVocabSize_ = 0;
    clearError();
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

// ── Real GPT-2 / Hugging Face tokenizer ingestion ────────────────────────────

// GPT-2 byte→unicode table (bytes_to_unicode): printable byte ranges map to
// themselves; the rest map to 256+n. Builds the byte↔codepoint maps shared by
// every byte-level BPE model (GPT-2, Llama-3, Qwen, …).
void BpeTokenizer::buildByteLevelTable() {
    byteToCp_.assign(256, 0);
    cpToByte_.clear();
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
}

// Remapped codepoints (a token string as stored in vocab/merges) → raw bytes.
std::string BpeTokenizer::cpsToBytes(const std::string& s) const {
    std::string out;
    for (int cp : utf8Decode(s)) {
        auto it = cpToByte_.find(cp);
        if (it != cpToByte_.end()) out += (char)(unsigned char)it->second;
    }
    return out;
}

bool BpeTokenizer::loadGpt2(const std::string& vocabJsonPath, const std::string& mergesTxtPath) {
    // Phase 1: read + validate both files without mutating state.
    std::ifstream vf(vocabJsonPath, std::ios::binary);
    if (!vf) return setError(TokenizerError::FileNotFound, "cannot open " + vocabJsonPath);
    std::stringstream ss; ss << vf.rdbuf();
    common::json::Value root;
    if (!common::json::parse(ss.str(), root) || !root.isObject())
        return setError(TokenizerError::ParseError, "vocab.json is not a JSON object");

    std::map<std::string, int> b2id;  // remapped token string -> id
    std::map<int, std::string> id2b;
    for (const auto& kv : root.obj) {
        double num = kv.second.asNumber(-1);
        int id = (int)num;
        if (num < 0 || (double)id != num)
            return setError(TokenizerError::InvalidVocabId,
                            "vocab entry '" + kv.first + "' has a non-integer/negative id");
        b2id.emplace(kv.first, id);
        auto ins = id2b.emplace(id, kv.first);
        if (!ins.second && ins.first->second != kv.first)
            return setError(TokenizerError::DuplicateVocabId,
                            "vocab id " + std::to_string(id) + " assigned to two tokens");
    }
    if (b2id.empty()) return setError(TokenizerError::ParseError, "vocab.json is empty");

    std::ifstream mf(mergesTxtPath);
    if (!mf) return setError(TokenizerError::FileNotFound, "cannot open " + mergesTxtPath);
    std::vector<std::pair<std::string, std::string>> mergesRemapped;
    std::string line;
    while (std::getline(mf, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF tolerance
        if (line.empty() || line[0] == '#') continue;               // skip #version header
        size_t sp = line.find(' ');
        if (sp == std::string::npos || sp == 0 || sp + 1 >= line.size() ||
            line.find(' ', sp + 1) != std::string::npos)
            return setError(TokenizerError::InvalidMerge,
                            "merges.txt line is not two space-separated operands: '" + line + "'");
        mergesRemapped.emplace_back(line.substr(0, sp), line.substr(sp + 1));
    }

    // Phase 2: commit from a clean base.
    reset();
    buildByteLevelTable();
    for (const auto& kv : b2id) {
        std::string bytes = cpsToBytes(kv.first);
        bytesToVocabId_[bytes] = kv.second;
        vocabIdToBytes_[kv.second] = bytes;
    }
    std::vector<std::pair<std::string, std::string>> merges;
    merges.reserve(mergesRemapped.size());
    for (const auto& mp : mergesRemapped)
        merges.emplace_back(cpsToBytes(mp.first), cpsToBytes(mp.second));
    loadMerges(merges);
    split_ = SplitRule{};  // GPT-2 pattern: exact-case contractions, ' ?' prefixes,
                           // unbounded digit runs, plain punct, no newline rule
    gpt2_ = true;
    hf_ = false;
    if (!validate()) { TokenizerError e = err_; std::string m = errMsg_; reset(); return setError(e, m); }
    clearError();
    return true;
}

// Pre-tokenization: the model's split regex, executed codepoint-exactly.
// GPT-2 (split_ defaults):
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// cl100k / Llama-3 / Qwen family (knobs from loadHf):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
//    ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Alternatives are tried in pattern order at each position, exactly like the
// regex alternation; `\s+(?!\S)` leaves the run's last whitespace codepoint to
// attach to the following token.
std::vector<std::vector<int>> BpeTokenizer::pretokenizeMapped(const std::string& text) const {
    std::vector<uint32_t> cps;
    std::vector<size_t> off;
    utf8DecodeOffsets(text, cps, off);
    const size_t m = cps.size();

    std::vector<std::vector<int>> pieces;
    auto emit = [&](size_t a, size_t b) {  // codepoint indices [a, b)
        std::vector<int> p;
        p.reserve(off[b] - off[a]);
        for (size_t k = off[a]; k < off[b]; ++k) p.push_back((unsigned char)text[k]);
        pieces.push_back(std::move(p));
    };

    size_t i = 0;
    while (i < m) {
        uint32_t c = cps[i];

        // 1. contractions: 's 't 're 've 'm 'll 'd (case-insensitive iff (?i:)).
        if (c == '\'' && i + 1 < m) {
            auto low = [&](uint32_t x) {
                return (split_.ciContractions && x >= 'A' && x <= 'Z') ? x + 32 : x;
            };
            uint32_t c1 = low(cps[i + 1]);
            uint32_t c2 = (i + 2 < m) ? low(cps[i + 2]) : 0;
            size_t len = 0;
            if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
                (c1 == 'l' && c2 == 'l'))
                len = 3;
            else if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd')
                len = 2;
            if (len) { emit(i, i + len); i += len; continue; }
        }

        // 2. letters with an optional one-codepoint prefix:
        //    GPT-2 ' ?\p{L}+' vs cl100k '[^\r\n\p{L}\p{N}]?\p{L}+'.
        {
            size_t k = i;
            bool prefixOk = split_.anyLetterPrefix
                                ? (!isUL(c) && !isUN(c) && c != '\r' && c != '\n')
                                : (c == ' ');
            if (prefixOk && i + 1 < m && isUL(cps[i + 1])) k = i + 1;
            if (k < m && isUL(cps[k])) {
                size_t j = k + 1;
                while (j < m && isUL(cps[j])) ++j;
                emit(i, j);
                i = j;
                continue;
            }
        }

        // 3. digits: GPT-2 ' ?\p{N}+' vs cl100k '\p{N}{1,max}' (no prefix).
        {
            size_t k = i;
            if (split_.digitMax == 0 && c == ' ' && i + 1 < m && isUN(cps[i + 1]))
                k = i + 1;
            if (k < m && isUN(cps[k])) {
                size_t j = k + 1;
                while (j < m && isUN(cps[j]) &&
                       (split_.digitMax == 0 || j - k < (size_t)split_.digitMax))
                    ++j;
                emit(i, j);
                i = j;
                continue;
            }
        }

        // 4. ' ?[^\s\p{L}\p{N}]+' with an optional '[\r\n]*' tail (cl100k).
        {
            size_t k = i;
            if (c == ' ' && i + 1 < m) k = i + 1;
            if (k < m && !isUWs(cps[k]) && !isUL(cps[k]) && !isUN(cps[k])) {
                size_t j = k + 1;
                while (j < m && !isUWs(cps[j]) && !isUL(cps[j]) && !isUN(cps[j])) ++j;
                if (split_.punctNewlineTail)
                    while (j < m && (cps[j] == '\r' || cps[j] == '\n')) ++j;
                emit(i, j);
                i = j;
                continue;
            }
        }

        // 5–7. whitespace: ['\s*[\r\n]+' |] '\s+(?!\S)' | '\s+'.
        if (isUWs(c)) {
            size_t j = i + 1;
            while (j < m && isUWs(cps[j])) ++j;
            if (split_.newlineRun) {
                size_t lastNl = (size_t)-1;
                for (size_t t2 = i; t2 < j; ++t2)
                    if (cps[t2] == '\r' || cps[t2] == '\n') lastNl = t2;
                if (lastNl != (size_t)-1) { emit(i, lastNl + 1); i = lastNl + 1; continue; }
            }
            size_t end = j;
            if (j < m && j - i >= 2) end = j - 1;  // \s+(?!\S): leave the last ws
            emit(i, end);
            i = end;
            continue;
        }

        emit(i, i + 1);  // unreachable in practice (rule 4 covers non-ws symbols)
        ++i;
    }
    return pieces;
}

// BPE + model-vocab mapping over one specials-free segment.
void BpeTokenizer::encodeMappedSegment(const std::string& segment, std::vector<int>& out) const {
    for (auto& piece : pretokenizeMapped(segment)) {
        std::vector<int> p = piece;
        applyMerges(p);
        for (int internalId : p) {
            const std::string& bytes = vocab_[internalId];
            auto it = bytesToVocabId_.find(bytes);
            if (it != bytesToVocabId_.end()) out.push_back(it->second);
            else  // fallback: emit each byte's vocab id
                for (unsigned char b : bytes) {
                    auto bi = bytesToVocabId_.find(std::string(1, (char)b));
                    if (bi != bytesToVocabId_.end()) out.push_back(bi->second);
                }
        }
    }
}

std::vector<int> BpeTokenizer::encodeGpt2(const std::string& text) const {
    std::vector<int> ids;
    encodeMappedSegment(text, ids);
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

// ── Unified Hugging Face tokenizer.json ──────────────────────────────────────

namespace {

// Known split patterns → knobs. Exact string match keeps this honest: an
// unrecognized pattern fails loadHf instead of silently mis-tokenizing.
const char* kPatGpt2 =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
const char* kPatGpt2Alt =
    "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
const char* kPatCl100k =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const char* kPatQwen =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

}  // namespace

int BpeTokenizer::specialTokenId(const std::string& content) const {
    for (const auto& sp : specials_)
        if (sp.first == content) return sp.second;
    return -1;
}

bool BpeTokenizer::loadHf(const std::string& tokenizerJsonPath) {
    // ── Phase 1: parse + validate the file WITHOUT mutating any state, so a bad
    // or unsupported file leaves the tokenizer exactly as it was. ─────────────
    std::ifstream f(tokenizerJsonPath, std::ios::binary);
    if (!f) return setError(TokenizerError::FileNotFound, "cannot open " + tokenizerJsonPath);
    std::stringstream ss;
    ss << f.rdbuf();
    common::json::Value root;
    if (!common::json::parse(ss.str(), root) || !root.isObject())
        return setError(TokenizerError::ParseError, "tokenizer.json is not a JSON object");

    const common::json::Value* model = root.find("model");
    if (!model || !model->isObject())
        return setError(TokenizerError::ParseError, "missing model object");
    const common::json::Value* mtype = model->find("type");
    if (mtype && mtype->asString("BPE") != "BPE")
        return setError(TokenizerError::UnsupportedModel,
                        "model.type is '" + mtype->asString() + "', only BPE is supported");

    const common::json::Value* norm = root.find("normalizer");
    if (norm && !norm->isNull()) {
        std::string nt = norm->find("type") ? norm->find("type")->asString() : "";
        if (nt != "NFC")
            return setError(TokenizerError::UnsupportedModel,
                            "normalizer '" + nt + "' (SentencePiece-style) is unsupported");
    }

    const common::json::Value* pre = root.find("pre_tokenizer");
    if (!pre || pre->isNull())
        return setError(TokenizerError::UnsupportedModel, "missing pre_tokenizer");
    bool byteLevel = false;
    std::string splitPat;
    auto scanPre = [&](const common::json::Value& p) {
        std::string pt = p.find("type") ? p.find("type")->asString() : "";
        if (pt == "ByteLevel") {
            byteLevel = true;
            if (p.find("add_prefix_space") && p.find("add_prefix_space")->asBool(false))
                byteLevel = false;  // prefix-space models (RoBERTa-era) unsupported
        } else if (pt == "Split") {
            const common::json::Value* pat = p.find("pattern");
            if (pat && pat->find("Regex")) splitPat = pat->find("Regex")->asString();
        }
    };
    std::string preType = pre->find("type") ? pre->find("type")->asString() : "";
    if (preType == "Sequence") {
        const common::json::Value* subs = pre->find("pretokenizers");
        if (!subs || !subs->isArray())
            return setError(TokenizerError::ParseError, "pre_tokenizer.Sequence has no pretokenizers[]");
        for (const auto& p : subs->arr) scanPre(p);
    } else {
        scanPre(*pre);
    }
    if (!byteLevel)
        return setError(TokenizerError::UnsupportedModel, "pre_tokenizer is not byte-level");

    SplitRule sr;  // ByteLevel's built-in regex = the GPT-2 pattern (defaults)
    if (!splitPat.empty()) {
        if (splitPat == kPatCl100k) {
            sr.ciContractions = true; sr.anyLetterPrefix = true; sr.digitMax = 3;
            sr.punctNewlineTail = true; sr.newlineRun = true;
        } else if (splitPat == kPatQwen) {
            sr.ciContractions = true; sr.anyLetterPrefix = true; sr.digitMax = 1;
            sr.punctNewlineTail = true; sr.newlineRun = true;
        } else if (splitPat == kPatGpt2 || splitPat == kPatGpt2Alt) {
            sr = SplitRule{};
        } else {
            return setError(TokenizerError::UnsupportedModel,
                            "unrecognized Split regex — refusing to approximate");
        }
    }

    const common::json::Value* vocab = model->find("vocab");
    if (!vocab || !vocab->isObject())
        return setError(TokenizerError::ParseError, "missing model.vocab object");
    const common::json::Value* merges = model->find("merges");
    if (!merges || !merges->isArray())
        return setError(TokenizerError::ParseError, "missing model.merges array");

    // Strict vocab-id validation + duplicate detection, into locals.
    std::map<std::string, int> b2id;
    std::map<int, std::string> id2b;
    int maxId = -1;
    for (const auto& kv : vocab->obj) {
        double num = kv.second.asNumber(-1);
        int id = (int)num;
        if (num < 0 || (double)id != num)
            return setError(TokenizerError::InvalidVocabId,
                            "vocab entry '" + kv.first + "' has a non-integer/negative id");
        // Keyed by the remapped token string for now; decoded to raw bytes at
        // commit (Phase 2), after the byte-level table exists.
        b2id.emplace(kv.first, id);
        auto ins = id2b.emplace(id, kv.first);
        if (!ins.second && ins.first->second != kv.first)
            return setError(TokenizerError::DuplicateVocabId,
                            "vocab id " + std::to_string(id) + " assigned to both '" +
                            ins.first->second + "' and '" + kv.first + "'");
        if (id > maxId) maxId = id;
    }
    if (maxId < 0)
        return setError(TokenizerError::ParseError, "model.vocab is empty");

    // Strict merge validation: every entry must be a well-formed pair.
    std::vector<std::pair<std::string, std::string>> msRemapped;
    msRemapped.reserve(merges->arr.size());
    for (const auto& mv : merges->arr) {
        if (mv.isString()) {
            size_t sp = mv.str.find(' ');
            // Exactly one space: byte-level operands never contain a space (it is
            // remapped to 'Ġ'), so "A B" is well-formed but "A B C" or "A " is not.
            if (sp == std::string::npos || sp == 0 || sp + 1 >= mv.str.size() ||
                mv.str.find(' ', sp + 1) != std::string::npos)
                return setError(TokenizerError::InvalidMerge,
                                "merge '" + mv.str + "' is not two space-separated operands");
            msRemapped.emplace_back(mv.str.substr(0, sp), mv.str.substr(sp + 1));
        } else if (mv.isArray() && mv.arr.size() == 2 &&
                   mv.arr[0].isString() && mv.arr[1].isString() &&
                   !mv.arr[0].str.empty() && !mv.arr[1].str.empty()) {
            msRemapped.emplace_back(mv.arr[0].str, mv.arr[1].str);
        } else {
            return setError(TokenizerError::InvalidMerge, "malformed merge entry (wrong arity/type)");
        }
    }

    // ── Phase 2: commit. All checks passed; build clean state from scratch so a
    // reused instance carries nothing over and any (now unlikely) failure below
    // leaves the pristine base tokenizer, never a half-built one. ─────────────
    reset();
    buildByteLevelTable();
    for (const auto& kv : b2id) {  // keyed by remapped token string
        std::string bytes = cpsToBytes(kv.first);
        bytesToVocabId_[bytes] = kv.second;
        vocabIdToBytes_[kv.second] = bytes;
    }
    std::vector<std::pair<std::string, std::string>> ms;
    ms.reserve(msRemapped.size());
    for (const auto& mp : msRemapped)
        ms.emplace_back(cpsToBytes(mp.first), cpsToBytes(mp.second));
    loadMerges(ms);

    // added_tokens: full metadata + fast literal-match list + reverse map, with
    // duplicate-id detection.
    const common::json::Value* added = root.find("added_tokens");
    if (added && added->isArray()) {
        for (const auto& tv : added->arr) {
            const common::json::Value* content = tv.find("content");
            const common::json::Value* idv = tv.find("id");
            if (!content || !idv) continue;
            int id = (int)idv->asNumber(-1);
            if (content->str.empty() || id < 0) continue;
            auto ins = specialById_.emplace(id, content->str);
            if (!ins.second && ins.first->second != content->str) {
                reset();
                return setError(TokenizerError::DuplicateVocabId,
                                "added-token id " + std::to_string(id) + " used twice");
            }
            AddedToken at;
            at.content = content->str;
            at.id = id;
            auto flag = [&](const char* k) {
                const common::json::Value* v = tv.find(k);
                return v && v->asBool(false);
            };
            at.special    = flag("special");
            at.lstrip     = flag("lstrip");
            at.rstrip     = flag("rstrip");
            at.singleWord = flag("single_word");
            at.normalized = flag("normalized");
            added_.push_back(std::move(at));
            specials_.emplace_back(content->str, id);
            if (id > maxId) maxId = id;
        }
    }
    // Deterministic special ordering: longest content first (greedy match), ties
    // broken by ascending id then content, so encode() is reproducible run-to-run.
    std::sort(specials_.begin(), specials_.end(), [](const auto& a, const auto& b) {
        if (a.first.size() != b.first.size()) return a.first.size() > b.first.size();
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
    });
    std::sort(added_.begin(), added_.end(),
              [](const AddedToken& a, const AddedToken& b) { return a.id < b.id; });

    split_ = sr;
    hfVocabSize_ = maxId + 1;
    hf_ = true;
    gpt2_ = false;

    if (!validate()) { TokenizerError e = err_; std::string m = errMsg_; reset(); return setError(e, m); }
    clearError();
    return true;
}

std::vector<int> BpeTokenizer::encodeHf(const std::string& text) const {
    std::vector<int> ids;
    size_t pos = 0;
    while (pos < text.size()) {
        // Earliest special occurrence wins; specials_ is longest-first, so at
        // equal positions the longest token matches (HF added-token semantics).
        size_t best = std::string::npos, bi = 0;
        for (size_t s = 0; s < specials_.size(); ++s) {
            size_t at = text.find(specials_[s].first, pos);
            if (at < best) { best = at; bi = s; }
        }
        if (best == std::string::npos) {
            encodeMappedSegment(text.substr(pos), ids);
            break;
        }
        if (best > pos) encodeMappedSegment(text.substr(pos, best - pos), ids);
        ids.push_back(specials_[bi].second);
        pos = best + specials_[bi].first.size();
    }
    return ids;
}

std::string BpeTokenizer::decodeHf(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        auto sp = specialById_.find(id);
        if (sp != specialById_.end()) { out += sp->second; continue; }
        auto it = vocabIdToBytes_.find(id);
        if (it != vocabIdToBytes_.end()) out += it->second;
    }
    return out;
}

// ── Loaded-state validation ──────────────────────────────────────────────────
bool BpeTokenizer::validate() const {
    if (!hf_ && !gpt2_) { clearError(); return true; }  // base mode: lossless by construction

    // Byte<->codepoint table complete and invertible.
    if (byteToCp_.size() != 256)
        return setError(TokenizerError::InconsistentState, "byte->codepoint table is not 256 entries");
    for (int b = 0; b < 256; ++b) {
        auto it = cpToByte_.find(byteToCp_[b]);
        if (it == cpToByte_.end() || it->second != b)
            return setError(TokenizerError::InconsistentState,
                            "byte<->codepoint table is not invertible at byte " + std::to_string(b));
    }

    // bytes<->id maps are exact inverses (this also catches a duplicate id, which
    // would leave more byte-strings than distinct ids).
    if (bytesToVocabId_.size() != vocabIdToBytes_.size())
        return setError(TokenizerError::DuplicateVocabId,
                        "bytes<->id maps disagree in size (duplicate id or token)");
    for (const auto& kv : bytesToVocabId_) {
        auto it = vocabIdToBytes_.find(kv.second);
        if (it == vocabIdToBytes_.end() || it->second != kv.first)
            return setError(TokenizerError::InconsistentState, "bytes<->id map is not invertible");
    }

    // Special-token ids must not collide with each other.
    if (specialById_.size() != specials_.size())
        return setError(TokenizerError::DuplicateVocabId, "duplicate special-token id");

    // NB: byte-level *completeness* (all 256 raw bytes representable) is not
    // required here — minimal/partial vocabularies are legal to load. Instead the
    // checked encoders (encode*Checked) report ByteNotInVocab at encode time, so a
    // byte that cannot be mapped is surfaced rather than silently dropped.
    clearError();
    return true;
}

bool BpeTokenizer::mappedIdKnown(int id) const {
    return specialById_.find(id) != specialById_.end() ||
           vocabIdToBytes_.find(id) != vocabIdToBytes_.end();
}

// ── Checked encode/decode ────────────────────────────────────────────────────
bool BpeTokenizer::encodeChecked(const std::string& text, std::vector<int>& out) const {
    out.clear();
    TokenizerError why;
    size_t at;
    if (!isValidUtf8(text, &why, &at))
        return setError(why, "invalid UTF-8 at byte " + std::to_string(at));
    out = encode(text);  // base byte-level path is inherently lossless
    clearError();
    return true;
}

bool BpeTokenizer::decodeChecked(const std::vector<int>& ids, std::string& out) const {
    out.clear();
    for (size_t i = 0; i < ids.size(); ++i)
        if (ids[i] < 0 || ids[i] >= (int)vocab_.size())
            return setError(TokenizerError::InvalidTokenId,
                            "token id " + std::to_string(ids[i]) + " at position " +
                            std::to_string(i) + " is out of range [0," +
                            std::to_string(vocab_.size()) + ")");
    out = decode(ids);
    clearError();
    return true;
}

// Byte-loss-checked mapped encoder: appends ids for `segment`, or fails with
// ByteNotInVocab. Mirrors encodeMappedSegment but never silently drops a byte.
bool BpeTokenizer::encodeMappedSegmentChecked(const std::string& segment,
                                              std::vector<int>& out) const {
    for (auto& piece : pretokenizeMapped(segment)) {
        std::vector<int> p = piece;
        applyMerges(p);
        for (int internalId : p) {
            const std::string& bytes = vocab_[internalId];
            auto it = bytesToVocabId_.find(bytes);
            if (it != bytesToVocabId_.end()) { out.push_back(it->second); continue; }
            for (unsigned char b : bytes) {
                auto bi = bytesToVocabId_.find(std::string(1, (char)b));
                if (bi == bytesToVocabId_.end())
                    return setError(TokenizerError::ByteNotInVocab,
                                    "byte 0x" +
                                    std::string(1, "0123456789abcdef"[(b >> 4) & 0xF]) +
                                    std::string(1, "0123456789abcdef"[b & 0xF]) +
                                    " has no model vocabulary id");
                out.push_back(bi->second);
            }
        }
    }
    return true;
}

bool BpeTokenizer::encodeGpt2Checked(const std::string& text, std::vector<int>& out) const {
    out.clear();
    if (!gpt2_ && !hf_) return setError(TokenizerError::NotLoaded, "encodeGpt2Checked before loadGpt2");
    TokenizerError why;
    size_t at;
    if (!isValidUtf8(text, &why, &at))
        return setError(why, "invalid UTF-8 at byte " + std::to_string(at));
    if (!encodeMappedSegmentChecked(text, out)) { out.clear(); return false; }
    clearError();
    return true;
}

bool BpeTokenizer::decodeGpt2Checked(const std::vector<int>& ids, std::string& out) const {
    out.clear();
    if (!gpt2_) return setError(TokenizerError::NotLoaded, "decodeGpt2Checked before loadGpt2");
    for (size_t i = 0; i < ids.size(); ++i)
        if (vocabIdToBytes_.find(ids[i]) == vocabIdToBytes_.end())
            return setError(TokenizerError::InvalidTokenId,
                            "token id " + std::to_string(ids[i]) + " at position " +
                            std::to_string(i) + " is not in the model vocabulary");
    out = decodeGpt2(ids);
    clearError();
    return true;
}

bool BpeTokenizer::encodeHfChecked(const std::string& text, std::vector<int>& out) const {
    out.clear();
    if (!hf_) return setError(TokenizerError::NotLoaded, "encodeHfChecked before loadHf");
    TokenizerError why;
    size_t at;
    if (!isValidUtf8(text, &why, &at))
        return setError(why, "invalid UTF-8 at byte " + std::to_string(at));
    // Special-aware segmentation identical to encodeHf, but each non-special
    // segment goes through the byte-loss-checked mapped encoder.
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best = std::string::npos, bi = 0;
        for (size_t s = 0; s < specials_.size(); ++s) {
            size_t a = text.find(specials_[s].first, pos);
            if (a < best) { best = a; bi = s; }
        }
        const size_t end = (best == std::string::npos) ? text.size() : best;
        if (end > pos) {
            if (!encodeMappedSegmentChecked(text.substr(pos, end - pos), out)) { out.clear(); return false; }
        }
        if (best == std::string::npos) break;
        out.push_back(specials_[bi].second);
        pos = best + specials_[bi].first.size();
    }
    clearError();
    return true;
}

bool BpeTokenizer::decodeHfChecked(const std::vector<int>& ids, std::string& out) const {
    out.clear();
    if (!hf_) return setError(TokenizerError::NotLoaded, "decodeHfChecked before loadHf");
    for (size_t i = 0; i < ids.size(); ++i)
        if (!mappedIdKnown(ids[i]))
            return setError(TokenizerError::InvalidTokenId,
                            "token id " + std::to_string(ids[i]) + " at position " +
                            std::to_string(i) + " is neither a vocabulary nor a special id");
    out = decodeHf(ids);
    clearError();
    return true;
}

}  // namespace xla
}  // namespace vgre
