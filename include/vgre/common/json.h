// Minimal, dependency-free JSON parser (RFC 8259 subset sufficient for JWT/JWKS,
// config, and report payloads).  Recursive-descent; objects preserve insertion
// order and are looked up by key.  Not a serializer — read-only.
#ifndef VGRE_COMMON_JSON_H
#define VGRE_COMMON_JSON_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vgre {
namespace common {
namespace json {

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;

    bool        b = false;
    double      num = 0.0;
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
    uint64_t    asUint64(uint64_t def = 0) const {
        return type == Number && num >= 0 ? static_cast<uint64_t>(num) : def;
    }
};

// Parse `text`.  Returns true and fills `out` on success; false on malformed
// input (out is left in an unspecified-but-valid state).
bool parse(const std::string& text, Value& out);

} // namespace json
} // namespace common
} // namespace vgre

#endif // VGRE_COMMON_JSON_H
