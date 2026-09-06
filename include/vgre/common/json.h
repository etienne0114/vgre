// Dependency-free JSON parser + serializer (RFC 8259).  Recursive-descent parse
// with a nesting-depth guard; objects preserve insertion order and are looked up
// by key.  Values can also be built programmatically and serialized with dump(),
// so this fully replaces the read AND write sides of a third-party JSON library
// (e.g. llvm::json) — nothing needs an external JSON dependency.
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
    int64_t     asInt64(int64_t def = 0) const {
        return type == Number ? static_cast<int64_t>(num) : def;
    }
    uint64_t    asUint64(uint64_t def = 0) const {
        return type == Number && num >= 0 ? static_cast<uint64_t>(num) : def;
    }

    // ── Builders (construct a value to serialize with dump()) ─────────────────
    static Value object() { Value v; v.type = Object; return v; }
    static Value array()  { Value v; v.type = Array;  return v; }
    static Value string(std::string s) { Value v; v.type = String; v.str = std::move(s); return v; }
    static Value number(double n)       { Value v; v.type = Number; v.num = n; return v; }
    static Value boolean(bool x)        { Value v; v.type = Bool;   v.b = x;   return v; }
    static Value null()                 { return Value{}; }

    // Object: set/overwrite `key` (turns this into an Object if it was Null).
    // Returns *this for chaining. No-op preconditions are asserted by isObject().
    Value& set(const std::string& key, Value val);
    // Array: append `val` (turns this into an Array if it was Null).
    Value& append(Value val);
};

// Parse `text`.  Returns true and fills `out` on success; false on malformed
// input or nesting deeper than the guard limit (out is left in an
// unspecified-but-valid state).
bool parse(const std::string& text, Value& out);

// Serialize `v` to a JSON string.  pretty=false is compact (no spaces);
// pretty=true is 2-space-indented multi-line.  Round-trips with parse().
std::string dump(const Value& v, bool pretty = false);

} // namespace json
} // namespace common
} // namespace vgre

#endif // VGRE_COMMON_JSON_H
