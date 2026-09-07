// Dependency-free, RFC 8259-strict JSON parser + serializer.
// See include/vgre/common/json.h.

#include "vgre/common/json.h"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <utility>

namespace vgre {
namespace common {
namespace json {

const char* errorCodeName(ErrorCode c) {
    switch (c) {
        case ErrorCode::Ok: return "Ok";
        case ErrorCode::UnexpectedEnd: return "UnexpectedEnd";
        case ErrorCode::TrailingGarbage: return "TrailingGarbage";
        case ErrorCode::InvalidLiteral: return "InvalidLiteral";
        case ErrorCode::InvalidNumber: return "InvalidNumber";
        case ErrorCode::NonFiniteNumber: return "NonFiniteNumber";
        case ErrorCode::NumberTooLarge: return "NumberTooLarge";
        case ErrorCode::ControlCharacterInString: return "ControlCharacterInString";
        case ErrorCode::InvalidStringEscape: return "InvalidStringEscape";
        case ErrorCode::InvalidUnicodeEscape: return "InvalidUnicodeEscape";
        case ErrorCode::InvalidUtf8: return "InvalidUtf8";
        case ErrorCode::ExpectedValue: return "ExpectedValue";
        case ErrorCode::ExpectedString: return "ExpectedString";
        case ErrorCode::ExpectedColon: return "ExpectedColon";
        case ErrorCode::ExpectedCommaOrEnd: return "ExpectedCommaOrEnd";
        case ErrorCode::DuplicateKey: return "DuplicateKey";
        case ErrorCode::DepthLimitExceeded: return "DepthLimitExceeded";
        case ErrorCode::InputTooLarge: return "InputTooLarge";
        case ErrorCode::StringTooLarge: return "StringTooLarge";
        case ErrorCode::ArrayTooLarge: return "ArrayTooLarge";
        case ErrorCode::ObjectTooLarge: return "ObjectTooLarge";
        case ErrorCode::ValueCountLimitExceeded: return "ValueCountLimitExceeded";
        case ErrorCode::OutputTooLarge: return "OutputTooLarge";
    }
    return "?";
}

const Value* Value::find(const std::string& key) const {
    if (type != Object) return nullptr;
    for (const auto& kv : obj)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

Value& Value::set(const std::string& key, Value val) {
    if (type != Object) *this = Value::object();
    for (auto& kv : obj)
        if (kv.first == key) { kv.second = std::move(val); return *this; }
    obj.emplace_back(key, std::move(val));
    return *this;
}

Value& Value::append(Value val) {
    if (type != Array) *this = Value::array();
    arr.push_back(std::move(val));
    return *this;
}

namespace {
// A numRaw token is a pure integer (no fraction/exponent) → parse it exactly.
bool tokenIsInteger(const std::string& t) {
    return !t.empty() && t.find_first_of(".eE") == std::string::npos;
}
}  // namespace

int64_t Value::asInt64(int64_t def) const {
    if (type != Number) return def;
    if (!numRaw.empty() && tokenIsInteger(numRaw)) {
        errno = 0;
        char* endp = nullptr;
        long long v = std::strtoll(numRaw.c_str(), &endp, 10);
        if (endp && *endp == '\0' && errno == 0) return static_cast<int64_t>(v);
        return def;
    }
    double v = num;
    if (std::isfinite(v) && v == std::floor(v) &&
        v >= -9223372036854775808.0 && v < 9223372036854775808.0)
        return static_cast<int64_t>(v);
    return def;
}

uint64_t Value::asUint64(uint64_t def) const {
    if (type != Number) return def;
    if (!numRaw.empty() && tokenIsInteger(numRaw)) {
        if (numRaw[0] == '-') return def;
        errno = 0;
        char* endp = nullptr;
        unsigned long long v = std::strtoull(numRaw.c_str(), &endp, 10);
        if (endp && *endp == '\0' && errno == 0) return static_cast<uint64_t>(v);
        return def;
    }
    double v = num;
    if (std::isfinite(v) && v >= 0 && v == std::floor(v) && v < 18446744073709551616.0)
        return static_cast<uint64_t>(v);
    return def;
}

// ── UTF-8 ─────────────────────────────────────────────────────────────────────
namespace {

// Decode one well-formed UTF-8 scalar at [p,end). Returns bytes consumed (1-4),
// or 0 if the sequence is malformed (overlong, bad continuation, surrogate, or
// out of range). Rejects the encodings RFC 8259 / Unicode forbid.
int utf8Decode(const unsigned char* p, const unsigned char* end, uint32_t& cp) {
    if (p >= end) return 0;
    unsigned char c = p[0];
    if (c < 0x80) { cp = c; return 1; }
    if ((c & 0xE0) == 0xC0) {
        if (c < 0xC2 || end - p < 2) return 0;               // overlong / short
        unsigned char c1 = p[1];
        if ((c1 & 0xC0) != 0x80) return 0;
        cp = (uint32_t(c & 0x1F) << 6) | (c1 & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0) {
        if (end - p < 3) return 0;
        unsigned char c1 = p[1], c2 = p[2];
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) return 0;
        cp = (uint32_t(c & 0x0F) << 12) | (uint32_t(c1 & 0x3F) << 6) | (c2 & 0x3F);
        if (cp < 0x800) return 0;                            // overlong
        if (cp >= 0xD800 && cp <= 0xDFFF) return 0;          // surrogate
        return 3;
    }
    if ((c & 0xF8) == 0xF0) {
        if (c > 0xF4 || end - p < 4) return 0;
        unsigned char c1 = p[1], c2 = p[2], c3 = p[3];
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return 0;
        cp = (uint32_t(c & 0x07) << 18) | (uint32_t(c1 & 0x3F) << 12) |
             (uint32_t(c2 & 0x3F) << 6) | (c3 & 0x3F);
        if (cp < 0x10000 || cp > 0x10FFFF) return 0;         // overlong / range
        return 4;
    }
    return 0;
}

void appendUtf8(std::string& s, uint32_t cp) {
    if (cp <= 0x7F) {
        s.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// ── Parser ─────────────────────────────────────────────────────────────────────
struct Parser {
    const char* p;
    const char* end;
    const ParseOptions& opt;
    Error* err;
    int line = 1, col = 1;
    int depth = 0;
    size_t values = 0;

    Parser(const std::string& s, const ParseOptions& o, Error* e)
        : p(s.data()), end(s.data() + s.size()), opt(o), err(e) {}

    bool fail(ErrorCode code, const char* msg) {
        if (err) { err->code = code; err->message = msg; err->line = line; err->column = col; }
        return false;
    }

    void adv() {  // consume one byte, tracking line/column
        if (p < end) {
            if (*p == '\n') { ++line; col = 1; } else { ++col; }
            ++p;
        }
    }

    void skipWs() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) adv();
    }

    bool countValue() {
        if (++values > opt.maxTotalValues) return fail(ErrorCode::ValueCountLimitExceeded, "too many values");
        return true;
    }

    bool parseValue(Value& out) {
        if (!countValue()) return false;
        skipWs();
        if (p >= end) return fail(ErrorCode::UnexpectedEnd, "unexpected end of input");
        switch (*p) {
            case '{': return parseObject(out);
            case '[': return parseArray(out);
            case '"': out.type = Value::String; return parseString(out.str);
            case 't': case 'f': return parseBool(out);
            case 'n': return parseNull(out);
            case '-': case '0': case '1': case '2': case '3': case '4':
            case '5': case '6': case '7': case '8': case '9':
                return parseNumber(out);
            default: return fail(ErrorCode::ExpectedValue, "expected a value");
        }
    }

    struct Depth {
        Parser& s; bool ok;
        explicit Depth(Parser& s_) : s(s_) { ok = (++s.depth <= s.opt.maxDepth); }
        ~Depth() { --s.depth; }
    };

    bool parseObject(Value& out) {
        Depth d(*this);
        if (!d.ok) return fail(ErrorCode::DepthLimitExceeded, "nesting too deep");
        out.type = Value::Object;
        adv();  // '{'
        skipWs();
        if (p < end && *p == '}') { adv(); return true; }
        std::unordered_set<std::string> seen;
        for (;;) {
            skipWs();
            if (p >= end || *p != '"') return fail(ErrorCode::ExpectedString, "expected a string key");
            std::string key;
            if (!parseString(key)) return false;
            if (opt.rejectDuplicateKeys && !seen.insert(key).second)
                return fail(ErrorCode::DuplicateKey, "duplicate object key");
            skipWs();
            if (p >= end || *p != ':') return fail(ErrorCode::ExpectedColon, "expected ':'");
            adv();
            Value v;
            if (!parseValue(v)) return false;
            out.obj.emplace_back(std::move(key), std::move(v));
            if (out.obj.size() > opt.maxObjectMembers)
                return fail(ErrorCode::ObjectTooLarge, "too many object members");
            skipWs();
            if (p >= end) return fail(ErrorCode::UnexpectedEnd, "unterminated object");
            if (*p == ',') { adv(); continue; }
            if (*p == '}') { adv(); return true; }
            return fail(ErrorCode::ExpectedCommaOrEnd, "expected ',' or '}'");
        }
    }

    bool parseArray(Value& out) {
        Depth d(*this);
        if (!d.ok) return fail(ErrorCode::DepthLimitExceeded, "nesting too deep");
        out.type = Value::Array;
        adv();  // '['
        skipWs();
        if (p < end && *p == ']') { adv(); return true; }
        for (;;) {
            Value v;
            if (!parseValue(v)) return false;
            out.arr.push_back(std::move(v));
            if (out.arr.size() > opt.maxArrayElements)
                return fail(ErrorCode::ArrayTooLarge, "too many array elements");
            skipWs();
            if (p >= end) return fail(ErrorCode::UnexpectedEnd, "unterminated array");
            if (*p == ',') { adv(); continue; }
            if (*p == ']') { adv(); return true; }
            return fail(ErrorCode::ExpectedCommaOrEnd, "expected ',' or ']'");
        }
    }

    bool hex4(uint32_t& v) {
        if (end - p < 4) return false;
        v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = p[i];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (c - 'A' + 10);
            else return false;
        }
        for (int i = 0; i < 4; ++i) adv();
        return true;
    }

    bool parseString(std::string& out) {
        adv();  // opening quote
        out.clear();
        for (;;) {
            if (p >= end) return fail(ErrorCode::UnexpectedEnd, "unterminated string");
            unsigned char c = static_cast<unsigned char>(*p);
            if (c == '"') { adv(); return true; }
            if (c == '\\') {
                adv();
                if (p >= end) return fail(ErrorCode::UnexpectedEnd, "unterminated escape");
                char e = *p; adv();
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        uint32_t cp = 0;
                        if (!hex4(cp)) return fail(ErrorCode::InvalidUnicodeEscape, "bad \\u escape");
                        if (cp >= 0xD800 && cp <= 0xDBFF) {              // high surrogate
                            if (end - p < 2 || p[0] != '\\' || p[1] != 'u')
                                return fail(ErrorCode::InvalidUnicodeEscape, "expected low surrogate");
                            adv(); adv();
                            uint32_t lo = 0;
                            if (!hex4(lo)) return fail(ErrorCode::InvalidUnicodeEscape, "bad low surrogate");
                            if (lo < 0xDC00 || lo > 0xDFFF)
                                return fail(ErrorCode::InvalidUnicodeEscape, "invalid low surrogate");
                            cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
                        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                            return fail(ErrorCode::InvalidUnicodeEscape, "lone low surrogate");
                        }
                        appendUtf8(out, cp);
                        break;
                    }
                    default: return fail(ErrorCode::InvalidStringEscape, "unknown escape");
                }
            } else if (c < 0x20) {
                return fail(ErrorCode::ControlCharacterInString, "control character in string");
            } else if (c < 0x80) {
                out.push_back(static_cast<char>(c));
                adv();
            } else {
                uint32_t cp = 0;
                int n = utf8Decode(reinterpret_cast<const unsigned char*>(p),
                                   reinterpret_cast<const unsigned char*>(end), cp);
                if (n == 0) return fail(ErrorCode::InvalidUtf8, "malformed UTF-8");
                for (int i = 0; i < n; ++i) { out.push_back(*p); adv(); }
            }
            if (out.size() > opt.maxStringBytes)
                return fail(ErrorCode::StringTooLarge, "string too large");
        }
    }

    bool parseNumber(Value& out) {
        const char* start = p;
        if (p < end && *p == '-') adv();
        if (p >= end) return fail(ErrorCode::InvalidNumber, "bad number");
        if (*p == '0') {
            adv();  // a leading zero must be the whole integer part
        } else if (*p >= '1' && *p <= '9') {
            while (p < end && *p >= '0' && *p <= '9') adv();
        } else {
            return fail(ErrorCode::InvalidNumber, "bad number");
        }
        if (p < end && *p == '.') {
            adv();
            if (p >= end || *p < '0' || *p > '9') return fail(ErrorCode::InvalidNumber, "empty fraction");
            while (p < end && *p >= '0' && *p <= '9') adv();
        }
        if (p < end && (*p == 'e' || *p == 'E')) {
            adv();
            if (p < end && (*p == '+' || *p == '-')) adv();
            if (p >= end || *p < '0' || *p > '9') return fail(ErrorCode::InvalidNumber, "empty exponent");
            while (p < end && *p >= '0' && *p <= '9') adv();
        }
        std::string tok(start, p);
        if (tok.size() > opt.maxNumberBytes)
            return fail(ErrorCode::NumberTooLarge, "number token too large");
        errno = 0;
        char* endp = nullptr;
        double v = std::strtod(tok.c_str(), &endp);
        if (!endp || *endp != '\0') return fail(ErrorCode::InvalidNumber, "bad number");
        if (!std::isfinite(v)) return fail(ErrorCode::NonFiniteNumber, "non-finite number");
        out.type = Value::Number;
        out.num = v;
        out.numRaw = std::move(tok);
        return true;
    }

    bool parseBool(Value& out) {
        if (end - p >= 4 && std::memcmp(p, "true", 4) == 0) {
            for (int i = 0; i < 4; ++i) adv();
            out.type = Value::Bool; out.b = true; return true;
        }
        if (end - p >= 5 && std::memcmp(p, "false", 5) == 0) {
            for (int i = 0; i < 5; ++i) adv();
            out.type = Value::Bool; out.b = false; return true;
        }
        return fail(ErrorCode::InvalidLiteral, "invalid literal");
    }

    bool parseNull(Value& out) {
        if (end - p >= 4 && std::memcmp(p, "null", 4) == 0) {
            for (int i = 0; i < 4; ++i) adv();
            out.type = Value::Null; return true;
        }
        return fail(ErrorCode::InvalidLiteral, "invalid literal");
    }

    bool run(Value& out) {
        if (!parseValue(out)) return false;
        skipWs();
        if (p != end) return fail(ErrorCode::TrailingGarbage, "trailing content after value");
        return true;
    }
};

}  // namespace

bool parse(const std::string& text, Value& out, const ParseOptions& opt, Error* err) {
    if (err) *err = Error{};
    out = Value{};  // parsing appends into out.obj/out.arr — start from empty
    if (text.size() > opt.maxInputBytes) {
        if (err) { err->code = ErrorCode::InputTooLarge; err->message = "input too large"; }
        return false;
    }
    Parser parser(text, opt, err);
    return parser.run(out);
}

bool parse(const std::string& text, Value& out, const ParseOptions& opt) {
    return parse(text, out, opt, nullptr);
}
bool parse(const std::string& text, Value& out, Error* err) {
    static const ParseOptions kDefault;
    return parse(text, out, kDefault, err);
}
bool parse(const std::string& text, Value& out) {
    static const ParseOptions kDefault;
    return parse(text, out, kDefault, nullptr);
}

// ── Serialization ─────────────────────────────────────────────────────────────
namespace {

struct Emitter {
    std::string& out;
    const DumpOptions& opt;
    Error* err;
    bool ok = true;

    bool push(char c) {
        if (out.size() + 1 > opt.maxOutputBytes) return setErr(ErrorCode::OutputTooLarge, "output too large");
        out.push_back(c);
        return true;
    }
    bool append(const char* s, size_t n) {
        if (out.size() + n > opt.maxOutputBytes) return setErr(ErrorCode::OutputTooLarge, "output too large");
        out.append(s, n);
        return true;
    }
    bool append(const std::string& s) { return append(s.data(), s.size()); }
    bool setErr(ErrorCode c, const char* m) {
        ok = false;
        if (err) { err->code = c; err->message = m; }
        return false;
    }
};

bool emitString(Emitter& e, const std::string& s) {
    if (!e.push('"')) return false;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(s.data());
    const unsigned char* end = p + s.size();
    char buf[8];
    while (p < end) {
        unsigned char c = *p;
        if (c == '"') { if (!e.append("\\\"", 2)) return false; ++p; }
        else if (c == '\\') { if (!e.append("\\\\", 2)) return false; ++p; }
        else if (c == '\b') { if (!e.append("\\b", 2)) return false; ++p; }
        else if (c == '\f') { if (!e.append("\\f", 2)) return false; ++p; }
        else if (c == '\n') { if (!e.append("\\n", 2)) return false; ++p; }
        else if (c == '\r') { if (!e.append("\\r", 2)) return false; ++p; }
        else if (c == '\t') { if (!e.append("\\t", 2)) return false; ++p; }
        else if (c < 0x20) {
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            if (!e.append(buf, 6)) return false;
            ++p;
        } else if (c < 0x80) {
            if (!e.push(static_cast<char>(c))) return false;
            ++p;
        } else {
            uint32_t cp = 0;
            int n = utf8Decode(p, end, cp);
            if (n == 0) return e.setErr(ErrorCode::InvalidUtf8, "invalid UTF-8 in string");
            if (e.opt.escapeNonAscii) {
                if (cp <= 0xFFFF) {
                    std::snprintf(buf, sizeof(buf), "\\u%04x", cp);
                    if (!e.append(buf, 6)) return false;
                } else {
                    uint32_t v = cp - 0x10000;
                    uint32_t hi = 0xD800 + (v >> 10), lo = 0xDC00 + (v & 0x3FF);
                    std::snprintf(buf, sizeof(buf), "\\u%04x", hi);
                    if (!e.append(buf, 6)) return false;
                    std::snprintf(buf, sizeof(buf), "\\u%04x", lo);
                    if (!e.append(buf, 6)) return false;
                }
            } else {
                if (!e.append(reinterpret_cast<const char*>(p), n)) return false;
            }
            p += n;
        }
    }
    return e.push('"');
}

bool emitNumber(Emitter& e, const Value& v) {
    if (!v.numRaw.empty()) return e.append(v.numRaw);  // exact parsed token
    if (!std::isfinite(v.num)) return e.setErr(ErrorCode::NonFiniteNumber, "non-finite number");
    char buf[32];
    if (v.num == static_cast<double>(static_cast<int64_t>(v.num)) &&
        v.num >= -9223372036854775808.0 && v.num < 9223372036854775808.0) {
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(static_cast<int64_t>(v.num)));
    } else {
        std::snprintf(buf, sizeof(buf), "%.17g", v.num);
    }
    return e.append(buf, std::strlen(buf));
}

bool emitValue(Emitter& e, const Value& v, int indent) {
    auto newline = [&](int ind) -> bool {
        if (!e.opt.pretty) return true;
        if (!e.push('\n')) return false;
        for (int i = 0; i < ind * e.opt.indentWidth; ++i) if (!e.push(' ')) return false;
        return true;
    };
    switch (v.type) {
        case Value::Null:   return e.append("null", 4);
        case Value::Bool:   return v.b ? e.append("true", 4) : e.append("false", 5);
        case Value::Number: return emitNumber(e, v);
        case Value::String: return emitString(e, v.str);
        case Value::Array:
            if (v.arr.empty()) return e.append("[]", 2);
            if (!e.push('[')) return false;
            for (size_t i = 0; i < v.arr.size(); ++i) {
                if (i && !e.push(',')) return false;
                if (!newline(indent + 1)) return false;
                if (!emitValue(e, v.arr[i], indent + 1)) return false;
            }
            if (!newline(indent)) return false;
            return e.push(']');
        case Value::Object:
            if (v.obj.empty()) return e.append("{}", 2);
            if (!e.push('{')) return false;
            for (size_t i = 0; i < v.obj.size(); ++i) {
                if (i && !e.push(',')) return false;
                if (!newline(indent + 1)) return false;
                if (!emitString(e, v.obj[i].first)) return false;
                if (!e.push(':')) return false;
                if (e.opt.pretty && !e.push(' ')) return false;
                if (!emitValue(e, v.obj[i].second, indent + 1)) return false;
            }
            if (!newline(indent)) return false;
            return e.push('}');
    }
    return true;
}

}  // namespace

bool dump(const Value& v, std::string& out, const DumpOptions& opt, Error* err) {
    if (err) *err = Error{};
    out.clear();
    Emitter e{out, opt, err};
    if (!emitValue(e, v, 0)) { out.clear(); return false; }
    return true;
}

std::string dump(const Value& v, bool pretty) {
    DumpOptions opt;
    opt.pretty = pretty;
    std::string out;
    dump(v, out, opt, nullptr);  // best-effort; out is cleared on failure
    return out;
}

}  // namespace json
}  // namespace common
}  // namespace vgre
