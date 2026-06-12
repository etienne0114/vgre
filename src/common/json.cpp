// Minimal recursive-descent JSON parser.  See include/vgre/common/json.h.

#include "vgre/common/json.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace vgre {
namespace common {
namespace json {

const Value* Value::find(const std::string& key) const {
    if (type != Object) return nullptr;
    for (const auto& kv : obj)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

namespace {

struct Parser {
    const char* p;
    const char* end;
    explicit Parser(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}

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

} // namespace json
} // namespace common
} // namespace vgre
