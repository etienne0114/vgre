// Dependency-free, RFC 8259-strict JSON parser + serializer.  Recursive-descent
// with a nesting-depth guard, UTF-8 validation, exact-integer preservation,
// structural limits, and located errors.  Objects preserve insertion order and
// are looked up by key.  Values can be built programmatically and serialized, so
// this fully replaces the read AND write sides of a third-party JSON library
// (e.g. llvm::json) — nothing in the engine needs an external JSON dependency.
#ifndef VGRE_COMMON_JSON_H
#define VGRE_COMMON_JSON_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vgre {
namespace common {
namespace json {

// Why a parse or a dump failed (Ok on success).
enum class ErrorCode {
    Ok = 0,
    UnexpectedEnd,             // input ended mid-value
    TrailingGarbage,           // extra non-whitespace after the top-level value
    InvalidLiteral,            // malformed true/false/null
    InvalidNumber,             // number not matching the RFC 8259 grammar
    NonFiniteNumber,           // number overflows to +/-inf, or dump of inf/NaN
    NumberTooLarge,            // number token exceeds ParseOptions::maxNumberBytes
    ControlCharacterInString,  // raw control char (< 0x20) inside a string
    InvalidStringEscape,       // unknown backslash escape
    InvalidUnicodeEscape,      // bad \uXXXX or bad surrogate pairing
    InvalidUtf8,               // malformed UTF-8 bytes (parse or dump)
    ExpectedValue,             // a value was required here
    ExpectedString,           // an object key (string) was required here
    ExpectedColon,             // ':' expected between key and value
    ExpectedCommaOrEnd,        // ',' or a closing bracket/brace expected
    DuplicateKey,              // repeated object key (when rejected)
    DepthLimitExceeded,        // nesting deeper than the guard limit
    InputTooLarge,             // input exceeds ParseOptions::maxInputBytes
    StringTooLarge,            // a string exceeds ParseOptions::maxStringBytes
    ArrayTooLarge,             // an array exceeds ParseOptions::maxArrayElements
    ObjectTooLarge,            // an object exceeds ParseOptions::maxObjectMembers
    ValueCountLimitExceeded,   // total values exceed ParseOptions::maxTotalValues
    OutputTooLarge,            // dump exceeds DumpOptions::maxOutputBytes
};

const char* errorCodeName(ErrorCode c);

struct Error {
    ErrorCode   code = ErrorCode::Ok;
    std::string message;
    int line = 0;    // 1-based (0 if not applicable)
    int column = 0;  // 1-based
};

struct ParseOptions {
    bool   rejectDuplicateKeys = true;
    int    maxDepth = 200;
    size_t maxInputBytes    = static_cast<size_t>(-1);
    size_t maxStringBytes   = static_cast<size_t>(-1);
    size_t maxNumberBytes   = static_cast<size_t>(-1);
    size_t maxArrayElements = static_cast<size_t>(-1);
    size_t maxObjectMembers = static_cast<size_t>(-1);
    size_t maxTotalValues   = static_cast<size_t>(-1);
};

struct DumpOptions {
    bool   pretty = false;       // 2-space (indentWidth) indented multi-line
    int    indentWidth = 2;
    bool   escapeNonAscii = false;  // emit \uXXXX (surrogate pairs) for cp >= 0x80
    size_t maxOutputBytes = static_cast<size_t>(-1);
};

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;

    bool        b = false;
    double      num = 0.0;
    std::string numRaw;   // exact source token for parsed numbers ("" if built)
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;

    bool isNull()   const { return type == Null; }
    bool isBool()   const { return type == Bool; }
    bool isNumber() const { return type == Number; }
    bool isString() const { return type == String; }
    bool isArray()  const { return type == Array; }
    bool isObject() const { return type == Object; }

    // Object member lookup (nullptr if absent or not an object).
    const Value* find(const std::string& key) const;

    // Typed accessors with safe defaults.
    std::string asString(const std::string& def = {}) const { return type == String ? str : def; }
    double      asNumber(double def = 0.0) const { return type == Number ? num : def; }
    bool        asBool(bool def = false) const { return type == Bool ? b : def; }
    // Integer accessors: exact for parsed integer tokens (beyond 2^53), and they
    // reject fractional values (returning `def`) rather than truncating.
    int64_t     asInt64(int64_t def = 0) const;
    uint64_t    asUint64(uint64_t def = 0) const;

    // ── Builders (construct a value to serialize with dump()) ─────────────────
    static Value object() { Value v; v.type = Object; return v; }
    static Value array()  { Value v; v.type = Array;  return v; }
    static Value string(std::string s) { Value v; v.type = String; v.str = std::move(s); return v; }
    static Value number(double n)       { Value v; v.type = Number; v.num = n; return v; }
    static Value boolean(bool x)        { Value v; v.type = Bool;   v.b = x;   return v; }
    static Value null()                 { return Value{}; }

    // Object: set/overwrite `key` (turns this into an Object if it was Null).
    Value& set(const std::string& key, Value val);
    // Array: append `val` (turns this into an Array if it was Null).
    Value& append(Value val);
};

// ── Parse ─────────────────────────────────────────────────────────────────────
// All overloads return true on success.  On failure `out` is left in an
// unspecified-but-valid state and, when provided, `err` describes the failure.
bool parse(const std::string& text, Value& out);
bool parse(const std::string& text, Value& out, Error* err);
bool parse(const std::string& text, Value& out, const ParseOptions& opt);
bool parse(const std::string& text, Value& out, const ParseOptions& opt, Error* err);

// ── Serialize ─────────────────────────────────────────────────────────────────
// Convenience form: returns the JSON text (best-effort; empty on an unrepresentable
// value such as a non-finite number).  pretty=true is 2-space indented.
std::string dump(const Value& v, bool pretty = false);
// Full form: writes to `out`, honouring DumpOptions; returns false and fills
// `err` if the value cannot be represented (non-finite number, invalid UTF-8) or
// exceeds the output-byte limit.
bool dump(const Value& v, std::string& out, const DumpOptions& opt, Error* err = nullptr);

}  // namespace json
}  // namespace common
}  // namespace vgre

#endif  // VGRE_COMMON_JSON_H
