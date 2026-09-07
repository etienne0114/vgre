// In-tree JSON (vgre::common::json): parser + serializer completeness.
// Track Z — the in-tree JSON must fully replace a third-party library (parse
// AND serialize) so nothing needs an external JSON dependency.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/common/json.h"

#include <cstdio>
#include <limits>
#include <string>

using vgre::common::json::DumpOptions;
using vgre::common::json::Error;
using vgre::common::json::ErrorCode;
using vgre::common::json::ParseOptions;
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

    // Unicode escapes: paired surrogates decode, lone/misordered surrogates do not.
    Value junk;
    Value music;
    CHECK(parse(R"("\uD834\uDD1E")", music), "parse paired UTF-16 surrogate escape");
    CHECK(music.asString() == std::string("\xF0\x9D\x84\x9E", 4),
          "surrogate pair decodes to one UTF-8 scalar");
    CHECK(!parse(R"("\uDD1E")", junk), "reject lone low surrogate");
    CHECK(!parse(R"("\uD834x")", junk), "reject high surrogate not followed by low surrogate");
    CHECK(!parse(R"("\uD834\u0041")", junk), "reject high surrogate followed by non-low surrogate");
    CHECK(!parse(R"("\u12xz")", junk), "reject non-hex unicode escape");

    DumpOptions escaped;
    escaped.escapeNonAscii = true;
    std::string escapedOut;
    Error dumpErr;
    CHECK(dump(music, escapedOut, escaped, &dumpErr), "dump with non-ASCII escaping succeeds");
    CHECK(escapedOut == R"("\ud834\udd1e")", "dump emits supplementary scalar as surrogate pair");

    // ── 5. Malformed inputs rejected ─────────────────────────────────────────
    CHECK(!parse("{", junk), "reject truncated object");
    CHECK(!parse("[1,2", junk), "reject truncated array");
    CHECK(!parse("{\"a\":1} trailing", junk), "reject trailing garbage");
    CHECK(!parse("nul", junk), "reject bad literal");
    CHECK(!parse("+1", junk), "reject leading plus number");
    CHECK(!parse("01", junk), "reject leading zero number");
    CHECK(!parse("1.", junk), "reject number with empty fraction");
    CHECK(!parse("1e", junk), "reject number with empty exponent");
    CHECK(!parse("1e999", junk), "reject non-finite number");

    Error err;
    CHECK(!parse(std::string("\"bad\nstring\""), junk, &err) &&
          err.code == ErrorCode::ControlCharacterInString,
          "reject unescaped control character in string with useful error code");
    std::string invalidUtf8 = "\"";
    invalidUtf8.push_back(static_cast<char>(0xC0));
    invalidUtf8.push_back(static_cast<char>(0x80));
    invalidUtf8.push_back('"');
    CHECK(!parse(invalidUtf8, junk, &err) && err.code == ErrorCode::InvalidUtf8,
          "reject overlong UTF-8 in string");

    // Duplicate keys are rejected by default, including after escape decoding.
    CHECK(!parse(R"({"a":1,"a":2})", junk, &err) &&
          err.code == ErrorCode::DuplicateKey, "reject duplicate object key");
    CHECK(!parse(R"({"a":1,"\u0061":2})", junk), "reject escaped duplicate object key");
    ParseOptions allowDup;
    allowDup.rejectDuplicateKeys = false;
    CHECK(parse(R"({"a":1,"a":2})", junk, allowDup) && junk.obj.size() == 2 &&
          junk.find("a")->asInt64() == 1,
          "duplicate keys can be preserved only when explicitly allowed");

    // Strict integer accessors avoid accidental truncation and double rounding.
    Value huge;
    CHECK(parse("9007199254740993", huge), "parse integer above 2^53");
    CHECK(huge.asUint64() == 9007199254740993ull, "uint64 accessor uses exact token");
    CHECK(dump(huge) == "9007199254740993", "dump preserves exact parsed integer token");
    Value fractional;
    CHECK(parse("3.5", fractional), "parse fractional number");
    CHECK(fractional.asInt64(123) == 123 && fractional.asUint64(456) == 456,
          "integer accessors reject fractional values");
    Value exponentInt;
    CHECK(parse("1.25e3", exponentInt) && exponentInt.asUint64() == 1250,
          "integer accessor handles integral exponent form exactly");

    // Memory and input limits are enforced before unbounded growth.
    ParseOptions tiny;
    tiny.maxInputBytes = 4;
    CHECK(!parse("[1,2]", junk, tiny, &err) && err.code == ErrorCode::InputTooLarge,
          "input byte limit enforced");
    tiny = ParseOptions{};
    tiny.maxStringBytes = 3;
    CHECK(!parse(R"("abcd")", junk, tiny, &err) && err.code == ErrorCode::StringTooLarge,
          "string byte limit enforced");
    tiny = ParseOptions{};
    tiny.maxArrayElements = 2;
    CHECK(!parse("[1,2,3]", junk, tiny, &err) && err.code == ErrorCode::ArrayTooLarge,
          "array element limit enforced");
    tiny = ParseOptions{};
    tiny.maxObjectMembers = 1;
    CHECK(!parse(R"({"a":1,"b":2})", junk, tiny, &err) &&
          err.code == ErrorCode::ObjectTooLarge,
          "object member limit enforced");
    tiny = ParseOptions{};
    tiny.maxTotalValues = 2;
    CHECK(!parse("[1,2]", junk, tiny, &err) &&
          err.code == ErrorCode::ValueCountLimitExceeded,
          "total value-count limit enforced");

    CHECK(!parse("{\n  \"a\" 1}", junk, &err) && err.code == ErrorCode::ExpectedColon &&
          err.line == 2 && err.column == 7,
          "error reporting includes line and column");

    // Serialization rejects values that cannot be represented as RFC 8259 JSON.
    Value inf = Value::number(std::numeric_limits<double>::infinity());
    CHECK(!dump(inf, escapedOut, DumpOptions{}, &dumpErr) &&
          dumpErr.code == ErrorCode::NonFiniteNumber,
          "dump rejects infinity/NaN");
    Value badString = Value::string(std::string(1, static_cast<char>(0xFF)));
    CHECK(!dump(badString, escapedOut, DumpOptions{}, &dumpErr) &&
          dumpErr.code == ErrorCode::InvalidUtf8,
          "dump rejects invalid UTF-8 string");
    DumpOptions tinyDump;
    tinyDump.maxOutputBytes = 4;
    CHECK(!dump(Value::string("abcd"), escapedOut, tinyDump, &dumpErr) &&
          dumpErr.code == ErrorCode::OutputTooLarge,
          "dump output byte limit enforced");

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
