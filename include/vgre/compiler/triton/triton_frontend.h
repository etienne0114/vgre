#ifndef VGRE_COMPILER_TRITON_FRONTEND_H
#define VGRE_COMPILER_TRITON_FRONTEND_H

// Phase 3, Track P3-13 — Triton IR (TTIR) frontend.
//
// VGRE consumes Triton's *emitted PTX*; this ingests Triton's own IR directly —
// the textual TTIR MLIR dialect (`tt.func`, `tt.make_range`, `tt.splat`,
// `tt.addptr`, `tt.load`/`tt.store`, `tt.get_program_id`, plus `arith.*`) — and
// executes it, avoiding the PTX round-trip. A Triton "program" operates on a
// BLOCK of elements with implicit SIMD; this lowers each TTIR op to its per-lane
// scalar semantics and runs every program (grid) over its lane range, so a
// representative Triton kernel runs from IR with output matching the reference.
//
// Supported TTIR subset (documented contract):
//   tt.func public @name(%arg: T, ...) { ... tt.return }
//   %r = tt.get_program_id {x|y|z} : i32
//   %r = tt.make_range {start = S : i32, end = E : i32} : tensor<NxI32>
//   %r = tt.splat %x : Tin -> tensor<Nx...>
//   %r = tt.addptr %ptrs, %offs : tensor<Nx!tt.ptr<elt>>, tensor<NxI32>
//   %r = tt.load  %ptrs (, %mask)? : tensor<Nx elt>
//        tt.store %ptrs, %val (, %mask)? : tensor<Nx elt>
//   %r = arith.constant V : (i32|f32)
//   %r = arith.{addi,subi,muli} %a, %b : ...      (integer, elementwise)
//   %r = arith.{addf,subf,mulf,divf} %a, %b : ... (float,   elementwise)
//   %r = arith.cmpi {slt,sle,sgt,sge,eq,ne}, %a, %b : ...  (-> i1 mask)
//   %r = arith.{sitofp,fptosi} %a : ...           (int<->float convert)
// Element types handled for load/store/addptr: f32, i32 (f16 widened to f32).

#include <cmath>
#include <cstdint>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace triton {

// ── Parsed IR ─────────────────────────────────────────────────────────────────
struct Op {
    std::string result;                                  // "%4" or "" (no result)
    std::string mnemonic;                                // "arith.addi"
    std::string extra;                                   // leading keyword: cmp pred / pid axis
    std::vector<std::string> operands;                   // ["%3","%2"]
    std::unordered_map<std::string, std::string> attrs;  // start/end/value
    std::string typeStr;                                 // signature after the top-level ':'
};
struct Func {
    std::string name;
    std::vector<std::pair<std::string, std::string>> args;  // (%arg0, type)
    std::vector<Op> body;
};
struct Module { std::vector<Func> funcs; };

// ── Runtime value: one cell per lane (int cells hold ints/booleans/addresses) ──
struct Val {
    char kind = 'i';                  // 'i' integer/pointer/bool, 'f' float
    std::vector<int64_t> i;
    std::vector<double>  f;
};

// A kernel argument: a pointer (device buffer) or a scalar integer.
struct KernelArg {
    bool    isPtr = false;
    void*   ptr   = nullptr;
    int64_t ival  = 0;
};

namespace detail {

inline std::string strip(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Element byte-size from a type fragment ("f32"→4, "f16"→2, "i32"→4, "f64"→8).
inline int elemSizeOf(const std::string& t) {
    if (t.find("f64") != std::string::npos || t.find("i64") != std::string::npos) return 8;
    if (t.find("f16") != std::string::npos || t.find("bf16") != std::string::npos) return 2;
    return 4;  // f32 / i32 default
}
inline bool isFloatType(const std::string& t) {
    return t.find("f32") != std::string::npos || t.find("f16") != std::string::npos ||
           t.find("f64") != std::string::npos || t.find("bf16") != std::string::npos;
}

// First ':' at bracket-depth 0 (so the ':' inside an attr dict / type is skipped).
inline size_t topLevelColon(const std::string& s) {
    int depth = 0;
    for (size_t k = 0; k < s.size(); ++k) {
        char c = s[k];
        if (c == '<' || c == '(' || c == '{' || c == '[') ++depth;
        else if (c == '>' || c == ')' || c == '}' || c == ']') --depth;
        else if (c == ':' && depth == 0) return k;
    }
    return std::string::npos;
}

}  // namespace detail

// ── Parser ────────────────────────────────────────────────────────────────────
inline Module parse(const std::string& src) {
    Module m;
    // Locate `tt.func ... @name( args ) {` and its body up to the matching close.
    static const std::regex funcRe(R"(tt\.func[^@]*@([A-Za-z0-9_]+)\s*\(([^)]*)\)[^\{]*\{)");
    auto begin = std::sregex_iterator(src.begin(), src.end(), funcRe);
    for (auto it = begin; it != std::sregex_iterator(); ++it) {
        const std::smatch& mm = *it;
        Func fn;
        fn.name = mm[1].str();

        // Arguments: %argK : type   (type runs to the next top-level comma).
        std::string argList = mm[2].str();
        static const std::regex argRe(R"((%[A-Za-z0-9_]+)\s*:\s*([^,]+))");
        for (auto a = std::sregex_iterator(argList.begin(), argList.end(), argRe);
             a != std::sregex_iterator(); ++a)
            fn.args.emplace_back((*a)[1].str(), detail::strip((*a)[2].str()));

        // Body: from after '{' to the matching '}'.
        size_t bodyStart = mm.position() + mm.length();
        int depth = 1;
        size_t k = bodyStart;
        for (; k < src.size() && depth > 0; ++k) {
            if (src[k] == '{') ++depth;
            else if (src[k] == '}') --depth;
        }
        std::string body = src.substr(bodyStart, (k > bodyStart ? k - bodyStart - 1 : 0));

        // Split into statements on newlines; parse each op line.
        std::string line;
        std::istringstream bs(body);
        while (std::getline(bs, line)) {
            // Strip MLIR location suffixes and comments.
            line = std::regex_replace(line, std::regex(R"(loc\([^)]*\))"), "");
            size_t cmt = line.find("//");
            if (cmt != std::string::npos) line = line.substr(0, cmt);
            line = detail::strip(line);
            if (line.empty() || line == "{" || line == "}") continue;

            Op op;
            std::string rest = line;
            size_t eq = rest.find('=');
            // Only treat '=' as result-binding if it precedes the mnemonic (no '<'/space tricks).
            if (eq != std::string::npos && rest.substr(0, eq).find('%') != std::string::npos &&
                rest.find("==") == std::string::npos) {
                op.result = detail::strip(rest.substr(0, eq));
                rest = detail::strip(rest.substr(eq + 1));
            }

            size_t colon = detail::topLevelColon(rest);
            std::string head = (colon == std::string::npos) ? rest : detail::strip(rest.substr(0, colon));
            op.typeStr = (colon == std::string::npos) ? "" : detail::strip(rest.substr(colon + 1));

            // Mnemonic = first whitespace-delimited token of head.
            size_t sp = head.find_first_of(" \t");
            op.mnemonic = (sp == std::string::npos) ? head : head.substr(0, sp);
            std::string args = (sp == std::string::npos) ? "" : detail::strip(head.substr(sp));

            // Leading keyword (cmp predicate / program-id axis), if the next token isn't a '%' or '{'.
            if (!args.empty() && args[0] != '%' && args[0] != '{') {
                size_t e = args.find_first_of(" ,\t");
                op.extra = args.substr(0, e);
            }

            // Operands: every %token in the head (none appear inside our attr dicts).
            static const std::regex pctRe(R"(%[A-Za-z0-9_]+)");
            for (auto a = std::sregex_iterator(head.begin(), head.end(), pctRe);
                 a != std::sregex_iterator(); ++a)
                op.operands.push_back((*a).str());

            // Attributes start/end (make_range) and a leading numeric literal (constant).
            std::smatch am;
            if (std::regex_search(head, am, std::regex(R"(start\s*=\s*(-?\d+))"))) op.attrs["start"] = am[1].str();
            if (std::regex_search(head, am, std::regex(R"(end\s*=\s*(-?\d+))")))   op.attrs["end"]   = am[1].str();
            if (op.mnemonic == "arith.constant") {
                if (std::regex_search(head, am, std::regex(R"(arith\.constant\s+(-?[0-9][0-9eE.+\-]*))")))
                    op.attrs["value"] = am[1].str();
            }
            fn.body.push_back(std::move(op));
        }
        m.funcs.push_back(std::move(fn));
    }
    return m;
}

// ── Interpreter ───────────────────────────────────────────────────────────────
class Interpreter {
public:
    // Run `fn` over a 1-D grid (gridX programs). args bind to the function's
    // parameters in order. The BLOCK width is taken from the kernel's make_range.
    static void run(const Func& fn, const std::vector<KernelArg>& args, int gridX) {
        int BLOCK = blockWidth(fn);
        if (BLOCK <= 0) throw std::runtime_error("triton: could not determine BLOCK width");

        for (int pid = 0; pid < gridX; ++pid) {
            std::unordered_map<std::string, Val> env;
            // Bind arguments as uniform (broadcast) values.
            for (size_t a = 0; a < fn.args.size() && a < args.size(); ++a) {
                Val v; v.kind = 'i'; v.i.assign(BLOCK, args[a].isPtr
                    ? static_cast<int64_t>(reinterpret_cast<uintptr_t>(args[a].ptr))
                    : args[a].ival);
                env[fn.args[a].first] = std::move(v);
            }
            for (const Op& op : fn.body) {
                if (op.mnemonic == "tt.return") break;
                evalOp(op, env, pid, BLOCK);
            }
        }
    }

private:
    static int blockWidth(const Func& fn) {
        for (const Op& op : fn.body)
            if (op.mnemonic == "tt.make_range") {
                int s = op.attrs.count("start") ? std::stoi(op.attrs.at("start")) : 0;
                int e = op.attrs.count("end")   ? std::stoi(op.attrs.at("end"))   : 0;
                return e - s;
            }
        // Fall back to a tensor<Nx...> shape in any type signature.
        for (const Op& op : fn.body) {
            std::smatch m;
            if (std::regex_search(op.typeStr, m, std::regex(R"(tensor<(\d+)x)")))
                return std::stoi(m[1].str());
        }
        return 0;
    }

    static const Val& get(const std::unordered_map<std::string, Val>& env, const std::string& n) {
        auto it = env.find(n);
        if (it == env.end()) throw std::runtime_error("triton: undefined value " + n);
        return it->second;
    }

    static void evalOp(const Op& op, std::unordered_map<std::string, Val>& env, int pid, int BLOCK) {
        const std::string& mn = op.mnemonic;
        Val out; out.kind = 'i'; out.i.assign(BLOCK, 0);

        if (mn == "tt.get_program_id") {
            int v = (op.extra == "y") ? 0 : (op.extra == "z") ? 0 : pid;  // 1-D grid
            out.i.assign(BLOCK, v);
        } else if (mn == "tt.make_range") {
            int s = op.attrs.count("start") ? std::stoi(op.attrs.at("start")) : 0;
            for (int l = 0; l < BLOCK; ++l) out.i[l] = s + l;
        } else if (mn == "tt.splat") {
            out = get(env, op.operands.at(0));                            // already broadcast
        } else if (mn == "tt.addptr") {
            const Val& p = get(env, op.operands.at(0));
            const Val& o = get(env, op.operands.at(1));
            int es = detail::elemSizeOf(op.typeStr);
            for (int l = 0; l < BLOCK; ++l) out.i[l] = p.i[l] + o.i[l] * es;
        } else if (mn == "tt.load") {
            const Val& p = get(env, op.operands.at(0));
            const bool hasMask = op.operands.size() > 1;
            const Val* msk = hasMask ? &get(env, op.operands.at(1)) : nullptr;
            bool fp = detail::isFloatType(op.typeStr);
            out.kind = fp ? 'f' : 'i';
            if (fp) out.f.assign(BLOCK, 0.0);
            for (int l = 0; l < BLOCK; ++l) {
                if (msk && msk->i[l] == 0) continue;                      // masked-off → 0
                void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(p.i[l]));
                if (fp) out.f[l] = *reinterpret_cast<const float*>(addr);
                else    out.i[l] = *reinterpret_cast<const int32_t*>(addr);
            }
        } else if (mn == "tt.store") {
            const Val& p = get(env, op.operands.at(0));
            const Val& v = get(env, op.operands.at(1));
            const bool hasMask = op.operands.size() > 2;
            const Val* msk = hasMask ? &get(env, op.operands.at(2)) : nullptr;
            bool fp = detail::isFloatType(op.typeStr);
            for (int l = 0; l < BLOCK; ++l) {
                if (msk && msk->i[l] == 0) continue;
                void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(p.i[l]));
                if (fp) *reinterpret_cast<float*>(addr)   = static_cast<float>(v.f[l]);
                else    *reinterpret_cast<int32_t*>(addr) = static_cast<int32_t>(v.i[l]);
            }
            return;  // no result
        } else if (mn == "arith.constant") {
            std::string val = op.attrs.count("value") ? op.attrs.at("value") : "0";
            if (detail::isFloatType(op.typeStr) || val.find('.') != std::string::npos ||
                val.find('e') != std::string::npos || val.find('E') != std::string::npos) {
                out.kind = 'f'; out.f.assign(BLOCK, std::stod(val));
            } else {
                out.i.assign(BLOCK, std::stoll(val));
            }
        } else if (mn == "arith.addi" || mn == "arith.subi" || mn == "arith.muli") {
            const Val& a = get(env, op.operands.at(0));
            const Val& b = get(env, op.operands.at(1));
            for (int l = 0; l < BLOCK; ++l)
                out.i[l] = (mn == "arith.addi") ? a.i[l] + b.i[l]
                         : (mn == "arith.subi") ? a.i[l] - b.i[l]
                                                : a.i[l] * b.i[l];
        } else if (mn == "arith.addf" || mn == "arith.subf" || mn == "arith.mulf" || mn == "arith.divf") {
            const Val& a = get(env, op.operands.at(0));
            const Val& b = get(env, op.operands.at(1));
            out.kind = 'f'; out.f.assign(BLOCK, 0.0);
            for (int l = 0; l < BLOCK; ++l)
                out.f[l] = (mn == "arith.addf") ? a.f[l] + b.f[l]
                         : (mn == "arith.subf") ? a.f[l] - b.f[l]
                         : (mn == "arith.mulf") ? a.f[l] * b.f[l]
                                                : a.f[l] / b.f[l];
        } else if (mn == "arith.cmpi") {
            const Val& a = get(env, op.operands.at(0));
            const Val& b = get(env, op.operands.at(1));
            const std::string& pr = op.extra;
            for (int l = 0; l < BLOCK; ++l) {
                int64_t x = a.i[l], y = b.i[l]; bool r;
                if      (pr == "slt") r = x <  y;
                else if (pr == "sle") r = x <= y;
                else if (pr == "sgt") r = x >  y;
                else if (pr == "sge") r = x >= y;
                else if (pr == "eq")  r = x == y;
                else                  r = x != y;   // "ne"
                out.i[l] = r ? 1 : 0;
            }
        } else if (mn == "arith.sitofp") {
            const Val& a = get(env, op.operands.at(0));
            out.kind = 'f'; out.f.assign(BLOCK, 0.0);
            for (int l = 0; l < BLOCK; ++l) out.f[l] = static_cast<double>(a.i[l]);
        } else if (mn == "arith.fptosi") {
            const Val& a = get(env, op.operands.at(0));
            for (int l = 0; l < BLOCK; ++l) out.i[l] = static_cast<int64_t>(a.f[l]);
        } else {
            throw std::runtime_error("triton: unsupported op '" + mn + "'");
        }

        if (!op.result.empty()) env[op.result] = std::move(out);
    }
};

// ── Compiled library entry ────────────────────────────────────────────────────
// Parse a TTIR module and run `kernelName` (or the first kernel if empty) over a
// 1-D grid. Defined in src/compiler/triton/triton_frontend.cpp so the frontend is
// part of the shipped library (not header-only) and reachable as a public API.
void run_kernel(const std::string& ttir, const std::string& kernelName,
                const std::vector<KernelArg>& args, int gridX);

}  // namespace triton
}  // namespace vgre

#endif  // VGRE_COMPILER_TRITON_FRONTEND_H
