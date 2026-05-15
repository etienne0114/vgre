// PTX Translator internal shared types, helpers, and the translation map.
#pragma once

#include "vgre/compiler/ptx_translator.h"
#include "vgre/compiler/wmma_emulation.h"
#include "vgre/common/logger.h"
#include <regex>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <functional>

namespace vgre {
namespace compiler {

namespace {

// vgre_lop3_b32: 3-operand logic via 8-bit LUT.
// Bit i of result = bit ((a[i]<<2)|(b[i]<<1)|c[i]) of lut.
static inline unsigned vgre_lop3_b32(unsigned a, unsigned b, unsigned c, unsigned char lut) {
    unsigned r = 0;
    for (int i = 31; i >= 0; --i) {
        int idx = (((a >> i) & 1) << 2) | (((b >> i) & 1) << 1) | ((c >> i) & 1);
        r |= ((unsigned)((lut >> idx) & 1)) << i;
    }
    return r;
}

// vgre_prmt_b32: PTX prmt.b32 — byte permutation.
// Selects 4 output bytes from [b:a] (b in high 4, a in low 4) using 16-bit
// selector c (4 nibbles).  Bits[2:0] = source byte index 0-7; bit[3] = sign-extend.
static inline unsigned vgre_prmt_b32(unsigned a, unsigned b, unsigned c) {
    unsigned char src[8];
    for (int i = 0; i < 4; ++i) { src[i]   = (unsigned char)((a >> (8*i)) & 0xFF); }
    for (int i = 0; i < 4; ++i) { src[4+i] = (unsigned char)((b >> (8*i)) & 0xFF); }
    unsigned result = 0;
    for (int i = 0; i < 4; ++i) {
        unsigned nib  = (c >> (4*i)) & 0xF;
        unsigned bidx = nib & 0x7;
        bool sext     = (nib & 0x8) != 0;
        unsigned char sel = (bidx < 8) ? src[bidx] : 0;
        unsigned expanded = sext ? ((sel & 0x80) ? 0xFFu : 0x00u) : (unsigned)sel;
        result |= (expanded & 0xFF) << (8*i);
    }
    return result;
}

// Trim whitespace
static inline std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Strip %r0, %f1, etc. -> "r0", "f1"; map to C operands by index.
// Parse "output : input : clobber" constraint string.
struct AsmConstraints {
    std::vector<std::string> outputs;   // e.g. "=r"
    std::vector<std::string> inputs;    // e.g. "r"
    std::vector<std::string> clobbers;
};

static inline AsmConstraints parseConstraints(const std::string& cs) {
    AsmConstraints r;
    // Split by ':' respecting quotes
    std::vector<std::string> parts;
    std::string cur;
    int depth = 0;
    for (char c : cs) {
        if (c == '"') depth ^= 1;
        if (c == ':' && depth == 0) { parts.push_back(cur); cur.clear(); }
        else cur += c;
    }
    parts.push_back(cur);

    auto parseList = [](const std::string& s) {
        std::vector<std::string> out;
        std::regex re("\"([^\"]+)\"");
        auto b = std::sregex_iterator(s.begin(), s.end(), re);
        auto e = std::sregex_iterator();
        for (auto it = b; it != e; ++it) out.push_back((*it)[1].str());
        return out;
    };
    if (parts.size() > 0) r.outputs  = parseList(parts[0]);
    if (parts.size() > 1) r.inputs   = parseList(parts[1]);
    if (parts.size() > 2) r.clobbers = parseList(parts[2]);
    return r;
}

// Map from PTX opcode to C++ lambda (string substitution).
// %0, %1, ... refer to operands in order (outputs first, then inputs).
using TranslateMap = std::unordered_map<std::string,
    std::function<std::string(const std::vector<std::string>&)>>;

} // anonymous namespace


// Translation maps (defined in respective .cpp files).
const TranslateMap& getMap();              // core arithmetic, memory, control flow
const TranslateMap& getTextureMap();       // tex, tld4, txq, suld, sust
const TranslateMap& getSharedAtomicMap();  // atom.shared.*
const TranslateMap& getConversionMap();    // cvt.*, sqrt.rn.f32, match.sync, elect.sync, grid.sync
std::vector<std::string> splitOperands(const std::string& s);
} // namespace compiler
} // namespace vgre
