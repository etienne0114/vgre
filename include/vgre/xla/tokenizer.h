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

// Structured error codes for the checked APIs and the model loaders. Every
// failure path sets one of these plus a human-readable message (lastError() /
// lastErrorMessage()), so callers get a precise reason instead of a bare false.
enum class TokenizerError {
    Ok = 0,
    InvalidUtf8,          // input is not well-formed UTF-8 (bad length/continuation)
    OverlongUtf8,         // a non-minimal (overlong) UTF-8 encoding
    SurrogateCodepoint,   // a UTF-16 surrogate (U+D800..U+DFFF) encoded as UTF-8
    CodepointOutOfRange,  // a codepoint above U+10FFFF
    InvalidTokenId,       // a token id absent from the active vocabulary (decode)
    ByteNotInVocab,       // a raw byte has no id in the model vocabulary (encode)
    FileNotFound,         // a tokenizer file could not be opened
    ParseError,           // JSON / merges-file parse failure
    UnsupportedModel,     // non-byte-level model, or normalizer/split not supported
    InvalidVocabId,       // a vocab entry has a negative / non-numeric id
    DuplicateVocabId,     // two vocab entries share an id (ambiguous decode)
    InvalidMerge,         // a merge rule is malformed (wrong arity / empty operand)
    InconsistentState,    // validate() found the loaded state self-inconsistent
    NotLoaded,            // a checked HF/GPT-2 API was called before a successful load
};

// Stable, human-readable name for an error code (never null).
const char* tokenizerErrorString(TokenizerError e);

// Full metadata for a Hugging Face added/special token (from added_tokens[]).
// Kept alongside the fast match list so callers can introspect the exact HF
// semantics of each special without re-parsing tokenizer.json.
struct AddedToken {
    std::string content;         // exact literal text of the token
    int         id = -1;         // reserved vocabulary id
    bool        special = false; // "special": rendered/handled as a control token
    bool        lstrip = false;  // strip one run of whitespace to its left on match
    bool        rstrip = false;  // strip one run of whitespace to its right on match
    bool        singleWord = false;  // only match on word boundaries
    bool        normalized = false;  // token participates in normalization
};

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

    // ── Unified Hugging Face `tokenizer.json` ingestion ──────────────────────
    // Parses the single-file format shipped by every modern HF model repo and
    // configures the tokenizer to emit that model's exact ids. Supports the
    // dominant modern family — byte-level BPE — which covers GPT-2/Whisper,
    // Llama-3, Qwen2/3, Phi, DeepSeek, Mistral-NeMo, …:
    //   • model.vocab + model.merges (both the legacy "A B" string form and the
    //     newer ["A","B"] pair form)
    //   • added_tokens (special tokens: matched literally before BPE, encoded
    //     to their reserved ids, emitted verbatim on decode)
    //   • the pre_tokenizer: either ByteLevel's built-in GPT-2 split regex or a
    //     Split pattern of the cl100k/Llama-3 family, recognized and executed
    //     with codepoint-exact \p{L}/\p{N}/\s classes (real UCD range tables,
    //     see unicode_tables.inc) — contraction case-sensitivity, digit-run
    //     limits ({1,3} / single / unbounded), optional non-letter prefixes,
    //     punctuation newline tails, and newline-run rules are all honored.
    // Returns false (no silent fallback) for non-byte-level models (Unigram /
    // SentencePiece-Metaspace, WordPiece) or an unrecognized split pattern.
    // Text is assumed NFC-normalized (the only normalizer these models use).
    bool loadHf(const std::string& tokenizerJsonPath);
    bool hfReady() const { return hf_; }
    std::vector<int> encodeHf(const std::string& text) const;  // special-token aware
    std::string decodeHf(const std::vector<int>& ids) const;
    int hfVocabSize() const { return hfVocabSize_; }
    // Id of an added/special token (exact content match), -1 if absent.
    int specialTokenId(const std::string& content) const;

    // ── Added/special-token metadata (HF) ────────────────────────────────────
    // Full metadata for every added token parsed from tokenizer.json, in a
    // deterministic order (see the ordering note on encodeHf). Empty until a
    // successful loadHf.
    const std::vector<AddedToken>& addedTokens() const { return added_; }

    // ── Structured error reporting ───────────────────────────────────────────
    // The code/message of the most recent operation on this tokenizer. Set by
    // every loader and every *Checked API; TokenizerError::Ok after success.
    TokenizerError     lastError() const { return err_; }
    const std::string& lastErrorMessage() const { return errMsg_; }

    // ── Explicit reset / state isolation ─────────────────────────────────────
    // Return to the pristine construction state: base vocabulary of 256 single
    // byte tokens, no merges, no GPT-2/HF mapping, no special tokens, no error.
    // A reused instance therefore loads a new model with zero carryover, and a
    // failed load leaves the tokenizer in this clean, still-usable base state.
    void reset();

    // ── Loaded-state validation ──────────────────────────────────────────────
    // Verify the currently-loaded mapped vocabulary is self-consistent: the
    // byte<->codepoint table is complete and invertible, bytes<->id maps are
    // exact inverses with no duplicate ids, and (byte-level requirement) all 256
    // single-byte tokens are representable so encoding can never silently drop a
    // byte. Runs automatically at the end of each successful load; also callable
    // directly. Returns true, or false with lastError() set. No-op (true) in the
    // untrained base mode, whose losslessness is structural.
    bool validate() const;

    // ── Strict RFC 3629 UTF-8 validation ─────────────────────────────────────
    // True iff `s` is well-formed UTF-8 with NO overlong encodings, NO UTF-16
    // surrogate codepoints (U+D800..U+DFFF), and nothing above U+10FFFF. On
    // failure, *why (if given) receives the specific TokenizerError and *atByte
    // (if given) the byte offset of the first offending sequence.
    static bool isValidUtf8(const std::string& s, TokenizerError* why = nullptr,
                            size_t* atByte = nullptr);

    // ── Checked encode/decode (validating, non-throwing) ─────────────────────
    // Same tokenization as the plain encode()/decode()/…Hf/…Gpt2 methods, but:
    //   • checked encoders first require `text` to be valid UTF-8 (see above) and
    //     guarantee no byte is silently lost during model-vocab mapping;
    //   • checked decoders require every id to be resolvable in the active
    //     vocabulary (no silent skips of invalid ids).
    // Each returns true and fills `out` on success, or false with lastError()/
    // lastErrorMessage() set and `out` left empty. The HF/GPT-2 variants also
    // report NotLoaded if called before a successful load.
    bool encodeChecked(const std::string& text, std::vector<int>& out) const;
    bool decodeChecked(const std::vector<int>& ids, std::string& out) const;
    bool encodeGpt2Checked(const std::string& text, std::vector<int>& out) const;
    bool decodeGpt2Checked(const std::vector<int>& ids, std::string& out) const;
    bool encodeHfChecked(const std::string& text, std::vector<int>& out) const;
    bool decodeHfChecked(const std::vector<int>& ids, std::string& out) const;

private:
    // Set/clear the structured error state (mutable: usable from const methods).
    bool setError(TokenizerError e, std::string msg) const;
    void clearError() const { err_ = TokenizerError::Ok; errMsg_.clear(); }
    // True if `id` resolves in the active mapped (HF/GPT-2) vocabulary or is a
    // known special id.
    bool mappedIdKnown(int id) const;

    mutable TokenizerError err_ = TokenizerError::Ok;
    mutable std::string    errMsg_;
    std::vector<AddedToken> added_;   // full HF added-token metadata (deterministic)

    // Each token id maps to its byte expansion; ids 0..255 are single bytes.
    std::vector<std::string> vocab_;
    // (left id, right id) -> merged id, in creation order (smaller id = higher
    // priority during encode).
    std::map<std::pair<int, int>, int> merge_;

    int addMerge(int a, int b);                       // create/return merged id
    std::vector<std::vector<int>> pretokenize(const std::string& text) const;
    void applyMerges(std::vector<int>& piece) const;  // greedy BPE on one piece

    // Mapped-vocab mode state (populated by loadGpt2 / loadHf).
    bool gpt2_ = false;
    bool hf_ = false;
    std::vector<int> byteToCp_;                       // byte → remap codepoint (256)
    std::map<int, int> cpToByte_;                     // remap codepoint → byte
    std::map<std::string, int> bytesToVocabId_;       // token byte-expansion → model id
    std::map<int, std::string> vocabIdToBytes_;

    // Pre-tokenizer knobs, extracted from the model's split regex.
    struct SplitRule {
        bool ciContractions = false;   // (?i:'s|…) vs case-sensitive
        bool anyLetterPrefix = false;  // [^\r\n\p{L}\p{N}]?\p{L}+ vs ' ?\p{L}+'
        int  digitMax = 0;             // \p{N}{1,max}; 0 = ' ?\p{N}+' (GPT-2)
        bool punctNewlineTail = false; // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
        bool newlineRun = false;       // '\s*[\r\n]+' alternative present
    };
    SplitRule split_;                                 // GPT-2 defaults

    // Added/special tokens (longest-first for greedy matching) + reverse map.
    std::vector<std::pair<std::string, int>> specials_;
    std::map<int, std::string> specialById_;
    int hfVocabSize_ = 0;

    void buildByteLevelTable();                       // GPT-2 bytes_to_unicode
    std::string cpsToBytes(const std::string& s) const;
    // Codepoint-exact split (unicode_tables.inc) driven by split_; returns
    // byte-value pieces of `text`.
    std::vector<std::vector<int>> pretokenizeMapped(const std::string& text) const;
    void encodeMappedSegment(const std::string& segment, std::vector<int>& out) const;
    // Byte-loss-checked variant of encodeMappedSegment: appends to `out`, or
    // returns false with ByteNotInVocab set if any byte has no model id.
    bool encodeMappedSegmentChecked(const std::string& segment, std::vector<int>& out) const;
};

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_TOKENIZER_H
