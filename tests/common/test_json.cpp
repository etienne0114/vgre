// In-tree JSON (vgre::common::json): parser + serializer completeness.
// Track Z — the in-tree JSON must fully replace a third-party library (parse
// AND serialize) so nothing needs an external JSON dependency.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/common/json.h"

#include <cstdio>
#include <string>

using vgre::common::json::Value;
using vgre::common::json::parse;
using vgre::common::json::dump;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

int main() {
    // ── 1. Parse a rich document ──────────────────────────────────────────────
    const std::string src =
        R"({"name":"vgre","n":42,"pi":3.5,"ok":true,"nil":null,)"
        R"("list":[1,2,3],"nested":{"a":[true,false]},"big":9007199254740992})";
    Value root;
    CHECK(parse(src, root), "parse rich document");
    CHECK(root.isObject(), "root is object");
    CHECK(root.find("name") && root.find("name")->asString() == "vgre", "string field");
    CHECK(root.find("n") && root.find("n")->asInt64() == 42, "int field via asInt64");
    CHECK(root.find("pi") && root.find("pi")->asNumber() == 3.5, "float field");
    CHECK(root.find("ok") && root.find("ok")->asBool() == true, "bool field");
    CHECK(root.find("nil") && root.find("nil")->isNull(), "null field");
    CHECK(root.find("list") && root.find("list")->isArray() &&
          root.find("list")->arr.size() == 3, "array field");
    CHECK(root.find("nested") && root.find("nested")->find("a") &&
          root.find("nested")->find("a")->arr.size() == 2, "nested field");
    CHECK(root.find("big") && root.find("big")->asUint64() == 9007199254740992ull,
          "2^53 integer exact (double-representable)");

    // ── 2. Round-trip: parse(dump(v)) preserves the tree ─────────────────────
    const std::string out = dump(root);
    Value reparsed;
    CHECK(parse(out, reparsed), "dump() output re-parses");
    CHECK(reparsed.find("name")->asString() == "vgre", "round-trip string");
    CHECK(reparsed.find("n")->asInt64() == 42, "round-trip int (no .0)");
    CHECK(out.find("42") != std::string::npos && out.find("42.0") == std::string::npos,
          "integer serialized without a decimal point");
    CHECK(dump(reparsed) == out, "dump is stable across a round-trip");

    // pretty form must also round-trip
    Value pv;
    CHECK(parse(dump(root, /*pretty=*/true), pv), "pretty dump re-parses");
    CHECK(pv.find("nested")->find("a")->arr.size() == 2, "pretty round-trip structure");

    // ── 3. Builders + serialize ──────────────────────────────────────────────
    Value obj = Value::object();
    obj.set("id", Value::number(7))
       .set("tags", Value::array().append(Value::string("a")).append(Value::string("b")))
       .set("enabled", Value::boolean(true));
    Value b;
    CHECK(parse(dump(obj), b), "built object serializes + re-parses");
    CHECK(b.find("id")->asInt64() == 7, "built int");
    CHECK(b.find("tags")->arr.size() == 2 && b.find("tags")->arr[1].asString() == "b",
          "built array");
    CHECK(b.find("enabled")->asBool(), "built bool");
    // set() overwrites an existing key in place (order preserved)
    obj.set("id", Value::number(9));
    CHECK(obj.find("id")->asInt64() == 9 && obj.obj.size() == 3, "set() overwrites, no dup");

    // ── 4. String escaping round-trips (quotes, control, unicode) ────────────
    Value s;
    CHECK(parse(R"("tab\tquote\"slash\/é")", s), "parse escaped string");
    CHECK(s.asString() == "tab\tquote\"slash/\xc3\xa9", "escapes decode (incl \\u00e9 -> UTF-8)");
    Value s2;
    CHECK(parse(dump(s), s2), "re-serialized escaped string re-parses");
    CHECK(s2.asString() == s.asString(), "escaped string round-trips exactly");

    // ── 5. Malformed inputs rejected ─────────────────────────────────────────
    Value junk;
    CHECK(!parse("{", junk), "reject truncated object");
    CHECK(!parse("[1,2", junk), "reject truncated array");
    CHECK(!parse("{\"a\":1} trailing", junk), "reject trailing garbage");
    CHECK(!parse("nul", junk), "reject bad literal");

    // ── 6. Depth guard: pathological nesting is rejected, not a crash ────────
    std::string deep;
    for (int i = 0; i < 5000; ++i) deep.push_back('[');
    Value d;
    CHECK(!parse(deep, d), "deeply-nested input rejected by depth guard (no overflow)");

    if (g_fail == 0)
        std::printf("PASS: vgre::common::json parser + serializer — all checks green\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
