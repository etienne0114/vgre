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
    buildByteLevelTable();

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
    split_ = SplitRule{};  // GPT-2 pattern: exact-case contractions, ' ?' prefixes,
                           // unbounded digit runs, plain punct, no newline rule
    gpt2_ = true;
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
    std::ifstream f(tokenizerJsonPath, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    common::json::Value root;
    if (!common::json::parse(ss.str(), root) || !root.isObject()) return false;

    // model: must be BPE.
    const common::json::Value* model = root.find("model");
    if (!model || !model->isObject()) return false;
    const common::json::Value* mtype = model->find("type");
    if (mtype && mtype->asString("BPE") != "BPE") return false;

    // normalizer: only the identity-on-NFC-text kinds these models use.
    const common::json::Value* norm = root.find("normalizer");
    if (norm && !norm->isNull()) {
        std::string nt = norm->find("type") ? norm->find("type")->asString() : "";
        if (nt != "NFC") return false;  // Metaspace/Prepend/… = SentencePiece → unsupported
    }

    // pre_tokenizer: must contain ByteLevel; may contain one Split pattern.
    const common::json::Value* pre = root.find("pre_tokenizer");
    if (!pre || pre->isNull()) return false;
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
        if (!subs || !subs->isArray()) return false;
        for (const auto& p : subs->arr) scanPre(p);
    } else {
        scanPre(*pre);
    }
    if (!byteLevel) return false;

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
            return false;  // unknown split family — refuse rather than approximate
        }
    }

    buildByteLevelTable();
    bytesToVocabId_.clear();
    vocabIdToBytes_.clear();

    // model.vocab: token(remapped) → id.
    const common::json::Value* vocab = model->find("vocab");
    if (!vocab || !vocab->isObject()) return false;
    int maxId = -1;
    for (const auto& kv : vocab->obj) {
        int id = (int)kv.second.asNumber(-1);
        if (id < 0) continue;
        std::string bytes = cpsToBytes(kv.first);
        bytesToVocabId_[bytes] = id;
        vocabIdToBytes_[id] = bytes;
        if (id > maxId) maxId = id;
    }
    if (maxId < 0) return false;

    // model.merges: legacy "A B" strings or newer ["A","B"] pairs.
    const common::json::Value* merges = model->find("merges");
    if (!merges || !merges->isArray()) return false;
    std::vector<std::pair<std::string, std::string>> ms;
    ms.reserve(merges->arr.size());
    for (const auto& mv : merges->arr) {
        if (mv.isString()) {
            size_t sp = mv.str.find(' ');
            if (sp == std::string::npos) continue;
            ms.emplace_back(cpsToBytes(mv.str.substr(0, sp)),
                            cpsToBytes(mv.str.substr(sp + 1)));
        } else if (mv.isArray() && mv.arr.size() == 2) {
            ms.emplace_back(cpsToBytes(mv.arr[0].asString()),
                            cpsToBytes(mv.arr[1].asString()));
        }
    }
    loadMerges(ms);

    // added_tokens: literal-match specials with reserved ids.
    specials_.clear();
    specialById_.clear();
    const common::json::Value* added = root.find("added_tokens");
    if (added && added->isArray()) {
        for (const auto& tv : added->arr) {
            const common::json::Value* content = tv.find("content");
            const common::json::Value* idv = tv.find("id");
            if (!content || !idv) continue;
            int id = (int)idv->asNumber(-1);
            if (content->str.empty() || id < 0) continue;
            specials_.emplace_back(content->str, id);
            specialById_[id] = content->str;
            if (id > maxId) maxId = id;
        }
    }
    std::sort(specials_.begin(), specials_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    split_ = sr;
    hfVocabSize_ = maxId + 1;
    hf_ = true;
    gpt2_ = false;
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

}  // namespace xla
}  // namespace vgre
