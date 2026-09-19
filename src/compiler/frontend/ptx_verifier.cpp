// Structural PTX verifier — see include/vgre/compiler/frontend/ptx_verifier.h.

#include "vgre/compiler/frontend/ptx_verifier.h"

#include <cctype>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vgre {
namespace compiler {
namespace frontend {

namespace {

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

// A virtual register token starts with '%' followed by a letter prefix and then,
// for a numbered register, decimal digits (%r10, %fd3, %p0). A special register
// (%tid.x, %ntid.y, …) has a non-digit after the prefix and is not range-checked.
// Returns true and fills prefix/index for a numbered register.
bool parseReg(const std::string& tok, std::string& prefix, long& index) {
    if (tok.size() < 2 || tok[0] != '%') return false;
    size_t i = 1;
    while (i < tok.size() && std::isalpha((unsigned char)tok[i])) ++i;
    if (i == 1 || i >= tok.size() || !std::isdigit((unsigned char)tok[i])) return false;  // special reg
    prefix = tok.substr(1, i - 1);
    index = std::stol(tok.substr(i));
    return true;
}

}  // namespace

PtxVerifyResult verifyPtx(const std::string& ptx) {
    PtxVerifyResult r;
    auto fail = [&](const std::string& why, const std::string& line) {
        r.ok = false;
        r.error = "PTX verification failed (internal codegen error): " + why +
                  "  in: '" + trim(line) + "'";
        return r;
    };

    std::unordered_map<std::string, long> regCount;   // prefix -> declared count (%r<N>)
    std::unordered_set<std::string> labels;           // "$L1" etc.

    // Pass 1: collect declared register ranges and label definitions.
    {
        std::istringstream in(ptx);
        std::string line;
        while (std::getline(in, line)) {
            std::string t = trim(line);
            if (t.empty()) continue;
            // ".reg .b32 %r<11>;"  -> prefix "r", count 11.
            if (t.rfind(".reg", 0) == 0) {
                size_t pct = t.find('%');
                size_t lt = t.find('<', pct);
                size_t gt = t.find('>', lt);
                if (pct != std::string::npos && lt != std::string::npos && gt != std::string::npos) {
                    std::string prefix;
                    size_t i = pct + 1;
                    while (i < lt && std::isalpha((unsigned char)t[i])) ++i;
                    prefix = t.substr(pct + 1, i - (pct + 1));
                    long n = std::stol(t.substr(lt + 1, gt - lt - 1));
                    regCount[prefix] = n;
                }
                continue;
            }
            // A label definition: "$L3:" (ends with ':' and no spaces).
            if (t.back() == ':' && t[0] == '$') labels.insert(t.substr(0, t.size() - 1));
        }
    }

    // Pass 2: check instructions.
    std::istringstream in(ptx);
    std::string line;
    while (std::getline(in, line)) {
        std::string t = trim(line);
        if (t.empty()) continue;
        // Skip declarations, labels, header/signature lines, braces.
        if (t[0] == '.' || t[0] == '{' || t[0] == '}' || t[0] == '(' || t[0] == ')' ||
            t[0] == ',' || t.back() == ':' || t.rfind(".reg", 0) == 0)
            continue;
        if (t.back() != ';') continue;   // not an instruction (e.g. signature fragment)
        std::string stmt = t.substr(0, t.size() - 1);   // drop ';'

        // Strip an optional predicate guard: "@%p3 " or "@!%p3 ".
        if (!stmt.empty() && stmt[0] == '@') {
            size_t sp = stmt.find(' ');
            if (sp == std::string::npos) return fail("predicated instruction has no body", line);
            stmt = trim(stmt.substr(sp + 1));
        }

        // Split mnemonic and operand list.
        size_t sp = stmt.find(' ');
        std::string mnem = (sp == std::string::npos) ? stmt : stmt.substr(0, sp);
        std::string opsStr = (sp == std::string::npos) ? "" : trim(stmt.substr(sp + 1));

        // Branch target must be a defined label.
        if (mnem == "bra") {
            if (labels.find(opsStr) == labels.end())
                return fail("branch to undefined label '" + opsStr + "'", line);
            continue;
        }

        if (opsStr.empty()) continue;   // operand-less op (ret, bar.sync 0 handled below)

        // Split operands on commas (bracketed memory operands contain no commas here).
        std::vector<std::string> ops;
        std::string cur;
        for (char c : opsStr) {
            if (c == ',') { ops.push_back(trim(cur)); cur.clear(); }
            else cur.push_back(c);
        }
        ops.push_back(trim(cur));

        for (const std::string& op : ops) {
            if (op.empty())
                return fail("empty operand in '" + mnem + "'", line);
            // Extract register tokens (also from inside [ ... ] memory operands).
            std::string tok;
            auto flush = [&]() {
                if (tok.empty()) return true;
                std::string prefix; long idx = 0;
                if (parseReg(tok, prefix, idx)) {
                    auto it = regCount.find(prefix);
                    if (it == regCount.end())
                        return false;   // used a register class that was never declared
                    if (idx >= it->second) return false;   // out of declared range
                }
                tok.clear();
                return true;
            };
            bool bad = false;
            for (char c : op) {
                if (c == '%' || std::isalnum((unsigned char)c) || c == '_' || c == '$' || c == '.') {
                    tok.push_back(c);
                } else {
                    if (!flush()) { bad = true; break; }
                }
            }
            if (!bad && !flush()) bad = true;
            if (bad)
                return fail("use of an undeclared or out-of-range register in '" + op + "'", line);
        }
    }

    return r;
}

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre
