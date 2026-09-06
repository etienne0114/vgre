// Dependency-free JSON parser + serializer.  See include/vgre/common/json.h.

#include "vgre/common/json.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace vgre {
namespace common {
namespace json {

const Value* Value::find(const std::string& key) const {
    if (type != Object) return nullptr;
    for (const auto& kv : obj)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

Value& Value::set(const std::string& key, Value val) {
    if (type != Object) *this = Value::object();  // coerce Null/other to an object
    for (auto& kv : obj)
        if (kv.first == key) { kv.second = std::move(val); return *this; }
    obj.emplace_back(key, std::move(val));
    return *this;
}

Value& Value::append(Value val) {
    if (type != Array) *this = Value::array();     // coerce Null/other to an array
    arr.push_back(std::move(val));
    return *this;
}

namespace {

// Cap nesting so a hostile deeply-nested document can't overflow the stack.
static const int kMaxDepth = 200;

struct Parser {
    const char* p;
    const char* end;
    int depth = 0;
    explicit Parser(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}

    // RAII depth counter: ok is false when the nesting limit is exceeded.
    struct Nest {
        int& d;
        bool ok;
        explicit Nest(int& d_) : d(d_) { ok = (++d <= kMaxDepth); }
        ~Nest() { --d; }
    };

    void skipWs() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }

    bool parseValue(Value& out) {
        skipWs();
        if (p >= end) return false;
        switch (*p) {
            case '{': return parseObject(out);
            case '[': return parseArray(out);
            case '"': { out.type = Value::String; return parseString(out.str); }
            case 't': case 'f': return parseBool(out);
            case 'n': return parseNull(out);
            default:  return parseNumber(out);
        }
    }

    bool parseObject(Value& out) {
        Nest nest(depth);
        if (!nest.ok) return false;
        out.type = Value::Object;
        ++p; // '{'
        skipWs();
        if (p < end && *p == '}') { ++p; return true; }
        while (p < end) {
            skipWs();
            if (p >= end || *p != '"') return false;
            std::string key;
            if (!parseString(key)) return false;
            skipWs();
            if (p >= end || *p != ':') return false;
            ++p;
            Value v;
            if (!parseValue(v)) return false;
            out.obj.emplace_back(std::move(key), std::move(v));
            skipWs();
            if (p >= end) return false;
            if (*p == ',') { ++p; continue; }
            if (*p == '}') { ++p; return true; }
            return false;
        }
        return false;
    }

    bool parseArray(Value& out) {
        Nest nest(depth);
        if (!nest.ok) return false;
        out.type = Value::Array;
        ++p; // '['
        skipWs();
        if (p < end && *p == ']') { ++p; return true; }
        while (p < end) {
            Value v;
            if (!parseValue(v)) return false;
            out.arr.push_back(std::move(v));
            skipWs();
            if (p >= end) return false;
            if (*p == ',') { ++p; continue; }
            if (*p == ']') { ++p; return true; }
            return false;
        }
        return false;
    }

    static void appendUtf8(std::string& s, uint32_t cp) {
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

    int hex4(uint32_t& v) {
        if (p + 4 > end) return 0;
        v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = p[i];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (c - 'A' + 10);
            else return 0;
        }
        p += 4;
        return 1;
    }

    bool parseString(std::string& out) {
        ++p; // opening quote
        out.clear();
        while (p < end) {
            char c = *p++;
            if (c == '"') return true;
            if (c == '\\') {
                if (p >= end) return false;
                char e = *p++;
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
                        if (!hex4(cp)) return false;
                        if (cp >= 0xD800 && cp <= 0xDBFF) { // high surrogate
                            if (p + 2 > end || p[0] != '\\' || p[1] != 'u') return false;
                            p += 2;
                            uint32_t lo = 0;
                            if (!hex4(lo)) return false;
                            if (lo < 0xDC00 || lo > 0xDFFF) return false;
                            cp = 0x10000 + (((cp - 0xD800) << 10) | (lo - 0xDC00));
                        }
                        appendUtf8(out, cp);
                        break;
                    }
                    default: return false;
                }
            } else {
                out.push_back(c);
            }
        }
        return false;
    }

    bool parseNumber(Value& out) {
        const char* start = p;
        if (p < end && (*p == '-' || *p == '+')) ++p;
        bool any = false;
        while (p < end && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' ||
                           *p == '+' || *p == '-')) { ++p; any = true; }
        if (!any) return false;
        std::string tok(start, p);
        char* endp = nullptr;
        out.type = Value::Number;
        out.num = std::strtod(tok.c_str(), &endp);
        return endp && *endp == '\0';
    }

    bool parseBool(Value& out) {
        if (end - p >= 4 && std::memcmp(p, "true", 4) == 0) { p += 4; out.type = Value::Bool; out.b = true; return true; }
        if (end - p >= 5 && std::memcmp(p, "false", 5) == 0) { p += 5; out.type = Value::Bool; out.b = false; return true; }
        return false;
    }

    bool parseNull(Value& out) {
        if (end - p >= 4 && std::memcmp(p, "null", 4) == 0) { p += 4; out.type = Value::Null; return true; }
        return false;
    }
};

} // namespace

bool parse(const std::string& text, Value& out) {
    Parser parser(text);
    if (!parser.parseValue(out)) return false;
    parser.skipWs();
    return parser.p == parser.end; // trailing garbage => invalid
}

// ── Serialization ─────────────────────────────────────────────────────────────

namespace {

void escapeTo(std::string& o, const std::string& s) {
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\b': o += "\\b";  break;
            case '\f': o += "\\f";  break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) {  // other control chars → \u00XX
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o.push_back(static_cast<char>(c));  // UTF-8 bytes pass through
                }
        }
    }
}

void numberTo(std::string& o, double n) {
    // Integral values within int64 range print without a decimal point so
    // round-tripping an integer stays an integer; otherwise use enough precision
    // to round-trip the double exactly.
    char buf[32];
    if (n == static_cast<double>(static_cast<int64_t>(n)) &&
        n >= -9.223372036854775e18 && n <= 9.223372036854775e18) {
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(static_cast<int64_t>(n)));
    } else {
        std::snprintf(buf, sizeof(buf), "%.17g", n);
    }
    o += buf;
}

void dumpTo(std::string& o, const Value& v, bool pretty, int indent) {
    auto newline = [&](int ind) {
        if (pretty) { o.push_back('\n'); o.append(static_cast<size_t>(ind) * 2, ' '); }
    };
    switch (v.type) {
        case Value::Null:   o += "null"; break;
        case Value::Bool:   o += v.b ? "true" : "false"; break;
        case Value::Number: numberTo(o, v.num); break;
        case Value::String: o.push_back('"'); escapeTo(o, v.str); o.push_back('"'); break;
        case Value::Array:
            if (v.arr.empty()) { o += "[]"; break; }
            o.push_back('[');
            for (size_t i = 0; i < v.arr.size(); ++i) {
                if (i) o.push_back(',');
                newline(indent + 1);
                dumpTo(o, v.arr[i], pretty, indent + 1);
            }
            newline(indent);
            o.push_back(']');
            break;
        case Value::Object:
            if (v.obj.empty()) { o += "{}"; break; }
            o.push_back('{');
            for (size_t i = 0; i < v.obj.size(); ++i) {
                if (i) o.push_back(',');
                newline(indent + 1);
                o.push_back('"');
                escapeTo(o, v.obj[i].first);
                o.push_back('"');
                o.push_back(':');
                if (pretty) o.push_back(' ');
                dumpTo(o, v.obj[i].second, pretty, indent + 1);
            }
            newline(indent);
            o.push_back('}');
            break;
    }
}

} // namespace

std::string dump(const Value& v, bool pretty) {
    std::string out;
    dumpTo(out, v, pretty, 0);
    return out;
}

} // namespace json
} // namespace common
} // namespace vgre
