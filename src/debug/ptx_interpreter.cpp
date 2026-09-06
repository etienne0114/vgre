// PTX single-step interpreter — see include/vgre/debug/ptx_interpreter.h.

#include "vgre/debug/ptx_interpreter.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#include <sys/mman.h>
#endif

namespace vgre {
namespace debug {

namespace {

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Size in bytes of a PTX type suffix ("u32" → 4, "f64" → 8, …).
int typeSize(const std::string& t) {
    if (t.size() < 2) return 0;
    if (t == "pred") return 1;
    int bits = std::atoi(t.c_str() + 1);
    return bits / 8;
}

bool isFloatType(const std::string& t) { return !t.empty() && t[0] == 'f'; }
bool isSignedType(const std::string& t) { return !t.empty() && t[0] == 's'; }

// Split "mad.lo.s32" → {"mad","lo","s32"}.
std::vector<std::string> splitDots(const std::string& op) {
    std::vector<std::string> out;
    std::stringstream ss(op);
    std::string part;
    while (std::getline(ss, part, '.')) out.push_back(part);
    return out;
}

// Last dotted component that looks like a type (s32/u64/f32/b16/pred).
std::string lastType(const std::vector<std::string>& parts) {
    for (auto it = parts.rbegin(); it != parts.rend(); ++it) {
        const std::string& p = *it;
        if (p == "pred") return p;
        if (p.size() >= 2 && (p[0] == 's' || p[0] == 'u' || p[0] == 'f' || p[0] == 'b') &&
            std::isdigit((unsigned char)p[1]))
            return p;
    }
    return "";
}

// Operand splitting: commas at bracket depth 0.
std::vector<std::string> splitArgs(const std::string& s) {
    std::vector<std::string> out;
    int depth = 0;
    std::string cur;
    for (char c : s) {
        if (c == '[' || c == '{' || c == '(') ++depth;
        if (c == ']' || c == '}' || c == ')') --depth;
        if (c == ',' && depth == 0) { out.push_back(trim(cur)); cur.clear(); }
        else cur += c;
    }
    cur = trim(cur);
    if (!cur.empty()) out.push_back(cur);
    return out;
}

float  asF32(uint64_t v) { float f;  uint32_t u = (uint32_t)v; std::memcpy(&f, &u, 4); return f; }
double asF64(uint64_t v) { double d; std::memcpy(&d, &v, 8); return d; }
uint64_t fromF32(float f)  { uint32_t u; std::memcpy(&u, &f, 4); return u; }
uint64_t fromF64(double d) { uint64_t u; std::memcpy(&u, &d, 8); return u; }

int64_t signExtend(uint64_t v, int bytes) {
    switch (bytes) {
        case 1: return (int8_t)v;
        case 2: return (int16_t)v;
        case 4: return (int32_t)v;
        default: return (int64_t)v;
    }
}
uint64_t zeroExtend(uint64_t v, int bytes) {
    switch (bytes) {
        case 1: return v & 0xFFull;
        case 2: return v & 0xFFFFull;
        case 4: return v & 0xFFFFFFFFull;
        default: return v;
    }
}

}  // namespace

// ── Parsing ───────────────────────────────────────────────────────────────────

PtxInterpreter::PtxInterpreter(const std::string& ptx, const std::string& entry) {
    parse(ptx, entry);
}

void PtxInterpreter::parse(const std::string& ptx, const std::string& entry) {
    // Locate ".entry <name>" (with or without .visible).
    size_t ep = std::string::npos;
    for (size_t at = ptx.find(".entry"); at != std::string::npos;
         at = ptx.find(".entry", at + 6)) {
        size_t ns = ptx.find_first_not_of(" \t\r\n", at + 6);
        if (ns != std::string::npos && ptx.compare(ns, entry.size(), entry) == 0) {
            char after = ns + entry.size() < ptx.size() ? ptx[ns + entry.size()] : ' ';
            if (!std::isalnum((unsigned char)after) && after != '_') { ep = ns; break; }
        }
    }
    if (ep == std::string::npos)
        throw std::runtime_error("PTX kernel not found: " + entry);
    kernel_.name = entry;

    // Parameters: "(.param .u64 name, .param .f32 name, …)".
    size_t po = ptx.find('(', ep);
    size_t body = ptx.find('{', ep);
    if (body == std::string::npos) throw std::runtime_error("PTX: missing kernel body");
    int paramOff = 0;
    if (po != std::string::npos && po < body) {
        size_t pc = ptx.find(')', po);
        for (const std::string& p : splitArgs(ptx.substr(po + 1, pc - po - 1))) {
            // ".param .u64 x" or ".param .align 8 .b8 x[16]" (aggregate: treat as bytes)
            std::stringstream ss(p);
            std::string tok, type, name;
            while (ss >> tok) {
                if (tok == ".param" || tok == ".align") { if (tok == ".align") ss >> tok; continue; }
                if (tok[0] == '.') type = tok.substr(1);
                else name = tok;
            }
            int sz = typeSize(type);
            size_t br = name.find('[');
            if (br != std::string::npos) {           // byte-array param
                sz *= std::atoi(name.c_str() + br + 1);
                name = name.substr(0, br);
            }
            if (name.empty() || sz <= 0) throw std::runtime_error("PTX: bad param: " + p);
            paramOff = (paramOff + sz - 1) / sz * sz;    // natural alignment
            kernel_.params.push_back({name, sz, paramOff});
            paramOff += sz;
        }
    }

    // Body: match braces from `body`.
    int depth = 0;
    size_t end = body;
    for (; end < ptx.size(); ++end) {
        if (ptx[end] == '{') ++depth;
        if (ptx[end] == '}') { if (--depth == 0) break; }
    }
    std::string src = ptx.substr(body + 1, end - body - 1);

    // Line-by-line: declarations, labels, instructions.
    std::istringstream lines(src);
    std::string raw;
    int lineNo = 0;
    std::set<std::string> regSeen;
    auto declReg = [&](const std::string& name) {
        if (regSeen.insert(name).second) regNames_.push_back(name);
    };
    while (std::getline(lines, raw)) {
        ++lineNo;
        size_t cm = raw.find("//");
        if (cm != std::string::npos) raw = raw.substr(0, cm);
        std::string line = trim(raw);
        if (line.empty()) continue;

        // Labels (possibly followed by an instruction on the same line).
        while (true) {
            size_t col = line.find(':');
            if (col == std::string::npos) break;
            std::string lbl = trim(line.substr(0, col));
            bool plain = !lbl.empty();
            for (char c : lbl)
                if (!std::isalnum((unsigned char)c) && c != '_' && c != '$') plain = false;
            if (!plain) break;
            kernel_.labels[lbl] = (int)kernel_.code.size();
            line = trim(line.substr(col + 1));
        }
        if (line.empty()) continue;

        if (line[0] == '.') {
            // ".reg .f32 %f<4>;" | ".reg .u32 %r0;" | ".shared .align 4 .b8 s[128];"
            std::string decl = line;
            if (!decl.empty() && decl.back() == ';') decl.pop_back();
            std::stringstream ss(decl);
            std::string kind; ss >> kind;
            if (kind == ".reg") {
                std::string type, name; ss >> type >> name;
                size_t lt = name.find('<');
                if (lt != std::string::npos) {
                    int n = std::atoi(name.c_str() + lt + 1);
                    std::string base = name.substr(0, lt);
                    for (int i = 0; i < n; ++i) declReg(base + std::to_string(i));
                } else declReg(name);
            } else if (kind == ".shared") {
                std::string tok, type, name;
                while (ss >> tok) {
                    if (tok == ".align") { ss >> tok; continue; }
                    if (tok[0] == '.') type = tok.substr(1);
                    else name = tok;
                }
                size_t br = name.find('[');
                int count = 1;
                if (br != std::string::npos) {
                    count = std::atoi(name.c_str() + br + 1);
                    name = name.substr(0, br);
                }
                int bytes = typeSize(type) * count;
                kernel_.sharedBytes = (kernel_.sharedBytes + 15) / 16 * 16;
                kernel_.sharedVars[name] = kernel_.sharedBytes;
                kernel_.sharedBytes += bytes;
            }
            // .local/.maxntid/… are accepted and ignored for execution.
            continue;
        }

        // One or more instructions separated by ';'.
        std::stringstream stmts(line);
        std::string stmt;
        while (std::getline(stmts, stmt, ';')) {
            stmt = trim(stmt);
            if (stmt.empty()) continue;
            PtxInstr ins;
            ins.line = lineNo;
            ins.text = stmt;
            if (stmt[0] == '@') {
                size_t sp = stmt.find_first_of(" \t");
                std::string p = stmt.substr(1, sp - 1);
                if (!p.empty() && p[0] == '!') { ins.predNeg = true; p = p.substr(1); }
                ins.pred = p;
                stmt = trim(stmt.substr(sp + 1));
            }
            size_t sp = stmt.find_first_of(" \t");
            ins.op = (sp == std::string::npos) ? stmt : stmt.substr(0, sp);
            if (sp != std::string::npos) ins.args = splitArgs(trim(stmt.substr(sp + 1)));
            kernel_.code.push_back(std::move(ins));
        }
    }
    if (kernel_.code.empty()) throw std::runtime_error("PTX: empty kernel body");
}

// ── Launch / scheduling ───────────────────────────────────────────────────────

void PtxInterpreter::launch(int gridX, int blockX, void* const* args, int numArgs) {
    // 1D convenience: {gridX,1,1} × {blockX,1,1}.
    launch(Dim3{gridX, 1, 1}, Dim3{blockX, 1, 1}, args, numArgs);
}

void PtxInterpreter::launch(const Dim3& grid, const Dim3& block, void* const* args, int numArgs) {
    if (grid.x <= 0 || grid.y <= 0 || grid.z <= 0 ||
        block.x <= 0 || block.y <= 0 || block.z <= 0)
        throw std::runtime_error("PTX: bad launch config");
    if (numArgs != (int)kernel_.params.size())
        throw std::runtime_error("PTX: kernel expects " +
                                 std::to_string(kernel_.params.size()) + " args");
    gridDim_[0] = grid.x;   gridDim_[1] = grid.y;   gridDim_[2] = grid.z;
    blockDim_[0] = block.x; blockDim_[1] = block.y; blockDim_[2] = block.z;
    gridTotal_  = grid.total();
    blockTotal_ = block.total();
    int total = 0;
    for (const auto& p : kernel_.params) total = std::max(total, p.offset + p.sizeBytes);
    paramBlock_.assign((size_t)total, 0);
    for (int i = 0; i < numArgs; ++i)
        std::memcpy(paramBlock_.data() + kernel_.params[i].offset, args[i],
                    (size_t)kernel_.params[i].sizeBytes);
    exited_ = false;
    startCta(0);
}

void PtxInterpreter::startCta(int cta) {
    cta_ = cta;
    // Decompose the linear CTA index into %ctaid.{x,y,z} (x fastest).
    ctaIdx_[0] = cta % gridDim_[0];
    ctaIdx_[1] = (cta / gridDim_[0]) % gridDim_[1];
    ctaIdx_[2] = cta / (gridDim_[0] * gridDim_[1]);
    shared_.assign((size_t)std::max(kernel_.sharedBytes, 1), 0);
    threads_.assign((size_t)blockTotal_, Thread{});
}

bool PtxInterpreter::ctaFinished() const {
    for (const auto& t : threads_)
        if (!t.done) return false;
    return true;
}

int PtxInterpreter::pcOf(int thread) const {
    return (thread >= 0 && thread < (int)threads_.size()) ? threads_[thread].pc : 0;
}
bool PtxInterpreter::threadExited(int thread) const {
    return thread < 0 || thread >= (int)threads_.size() || threads_[thread].done;
}

void PtxInterpreter::releaseBarrierIfReady() {
    for (const auto& t : threads_)
        if (!t.done && !t.atBarrier) return;
    for (auto& t : threads_) t.atBarrier = false;
}

StopReason PtxInterpreter::resume() {
    if (exited_) return StopReason::Exited;
    // Step the previously-stopped thread over its breakpoint first (gdb also
    // does the z0/s/Z0 dance, but a plain 'c' at a bp must still make progress).
    if (stoppedThread_ < (int)threads_.size()) {
        Thread& st = threads_[stoppedThread_];
        if (!st.done && !st.atBarrier && breakpoints_.count(st.pc)) execOne(st, stoppedThread_);
    }
    while (true) {
        releaseBarrierIfReady();
        bool progressed = false;
        for (int tid = 0; tid < (int)threads_.size(); ++tid) {
            Thread& t = threads_[tid];
            while (!t.done && !t.atBarrier) {
                if (breakpoints_.count(t.pc)) { stoppedThread_ = tid; return StopReason::Breakpoint; }
                if (!execOne(t, tid)) break;
                progressed = true;
            }
        }
        if (ctaFinished()) {
            if (cta_ + 1 < gridTotal_) { startCta(cta_ + 1); continue; }
            exited_ = true;
            return StopReason::Exited;
        }
        if (!progressed) {
            releaseBarrierIfReady();
            for (const auto& t : threads_)
                if (!t.done && !t.atBarrier) goto again;   // barrier released
            throw std::runtime_error("PTX: deadlock — threads blocked at bar.sync");
        }
    again:;
    }
}

StopReason PtxInterpreter::stepThread(int thread) {
    if (exited_ || thread < 0 || thread >= (int)threads_.size()) return StopReason::Exited;
    Thread& t = threads_[thread];
    stoppedThread_ = thread;
    if (t.done) return StopReason::Step;
    if (t.atBarrier) releaseBarrierIfReady();
    if (!t.atBarrier) execOne(t, thread);
    if (ctaFinished() && cta_ + 1 >= gridTotal_) { exited_ = true; return StopReason::Exited; }
    return StopReason::Step;
}

// ── Operands / registers / memory ─────────────────────────────────────────────

RegVal& PtxInterpreter::reg(Thread& t, const std::string& name) { return t.regs[name]; }

bool PtxInterpreter::readRegister(int thread, const std::string& name, uint64_t& out) const {
    if (thread < 0 || thread >= (int)threads_.size()) return false;
    const Thread& t = threads_[thread];
    auto p = t.preds.find(name);
    if (p != t.preds.end()) { out = p->second ? 1 : 0; return true; }
    auto r = t.regs.find(name);
    out = (r != t.regs.end()) ? r->second.u : 0;
    return true;
}

bool PtxInterpreter::writeRegister(int thread, const std::string& name, uint64_t v) {
    if (thread < 0 || thread >= (int)threads_.size()) return false;
    Thread& t = threads_[thread];
    if (!name.empty() && name.find("%p") == 0 &&
        (t.preds.count(name) || t.regs.count(name) == 0)) {
        t.preds[name] = v != 0;
        return true;
    }
    t.regs[name].u = v;
    return true;
}

bool PtxInterpreter::readGlobal(uint64_t addr, void* dst, size_t n) {
    if (!addr) return false;
#if defined(__linux__) || defined(__APPLE__)
    // Probe the page range before touching it (a bad debugger poke must not
    // crash the process, exactly like gdbserver's EFAULT handling).
    long pg = sysconf(_SC_PAGESIZE);
    uint64_t first = addr / pg, last = (addr + n - 1) / pg;
#if defined(__APPLE__)
    std::vector<char> vec((size_t)(last - first + 1));
#else
    std::vector<unsigned char> vec((size_t)(last - first + 1));
#endif
    if (mincore((void*)(first * pg), (size_t)((last - first + 1) * pg), vec.data()) != 0)
        return false;
#endif
    std::memcpy(dst, (const void*)addr, n);
    return true;
}

bool PtxInterpreter::writeGlobal(uint64_t addr, const void* src, size_t n) {
    if (!addr) return false;
#if defined(__linux__) || defined(__APPLE__)
    long pg = sysconf(_SC_PAGESIZE);
    uint64_t first = addr / pg, last = (addr + n - 1) / pg;
#if defined(__APPLE__)
    std::vector<char> vec((size_t)(last - first + 1));
#else
    std::vector<unsigned char> vec((size_t)(last - first + 1));
#endif
    if (mincore((void*)(first * pg), (size_t)((last - first + 1) * pg), vec.data()) != 0)
        return false;
#endif
    std::memcpy((void*)addr, src, n);
    return true;
}

// Value of a non-memory operand: register, special register, immediate,
// or a shared-variable name (its arena offset).
uint64_t PtxInterpreter::evalOperand(Thread& t, int tid, const std::string& s,
                                     int /*sizeBytes*/) const {
    if (s.empty()) return 0;
    if (s[0] == '%') {
        // Thread index within the CTA: decompose the linear tid into
        // %tid.{x,y,z} using the block extents (x fastest).
        const int bx = blockDim_[0], by = blockDim_[1];
        if (s == "%tid.x") return (uint64_t)(tid % bx);
        if (s == "%tid.y") return (uint64_t)((tid / bx) % by);
        if (s == "%tid.z") return (uint64_t)(tid / (bx * by));
        if (s == "%ntid.x") return (uint64_t)blockDim_[0];
        if (s == "%ntid.y") return (uint64_t)blockDim_[1];
        if (s == "%ntid.z") return (uint64_t)blockDim_[2];
        if (s == "%ctaid.x") return (uint64_t)ctaIdx_[0];
        if (s == "%ctaid.y") return (uint64_t)ctaIdx_[1];
        if (s == "%ctaid.z") return (uint64_t)ctaIdx_[2];
        if (s == "%nctaid.x") return (uint64_t)gridDim_[0];
        if (s == "%nctaid.y") return (uint64_t)gridDim_[1];
        if (s == "%nctaid.z") return (uint64_t)gridDim_[2];
        if (s == "%laneid") return (uint64_t)(tid & 31);
        if (s == "%warpid") return (uint64_t)(tid >> 5);
        auto p = t.preds.find(s);
        if (p != t.preds.end()) return p->second ? 1 : 0;
        auto r = t.regs.find(s);
        return r != t.regs.end() ? r->second.u : 0;
    }
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'f' || s[1] == 'F'))
        return std::stoul(s.substr(2), nullptr, 16);                 // f32 bit pattern
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'd' || s[1] == 'D'))
        return std::stoull(s.substr(2), nullptr, 16);                // f64 bit pattern
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'))
        return std::stoull(s.substr(2), nullptr, 16);
    if (s[0] == '-' || std::isdigit((unsigned char)s[0]))
        return (uint64_t)std::stoll(s);
    auto sh = kernel_.sharedVars.find(s);                            // mov %r, sharedName
    if (sh != kernel_.sharedVars.end()) return (uint64_t)sh->second;
    throw std::runtime_error("PTX: unknown operand '" + s + "'");
}

// "[%rd6+8]" / "[name]" → (base string, byte offset).
static void parseMemRef(const std::string& m, std::string& base, int64_t& off) {
    std::string inner = m;
    if (!inner.empty() && inner.front() == '[') inner = inner.substr(1, inner.size() - 2);
    size_t plus = inner.find_last_of("+-");
    off = 0;
    base = trim(inner);
    if (plus != std::string::npos && plus > 0) {
        // '+8' → +8; nvcc writes negative offsets as '+-8' (find_last_of lands
        // on the '-'); a trailing '+' left on the base after the cut is dropped.
        off = std::stoll(inner.substr(inner[plus] == '+' ? plus + 1 : plus));
        base = trim(inner.substr(0, plus));
        if (!base.empty() && base.back() == '+') base.pop_back();
        base = trim(base);
    }
}

uint64_t PtxInterpreter::loadFrom(Thread& t, int tid, const std::string& memRef,
                                  const std::string& space, int size) {
    std::string base; int64_t off;
    parseMemRef(memRef, base, off);
    uint64_t val = 0;
    if (space == "param") {
        int po = -1;
        for (const auto& p : kernel_.params)
            if (p.name == base) { po = p.offset; break; }
        if (po < 0) throw std::runtime_error("PTX: unknown param '" + base + "'");
        std::memcpy(&val, paramBlock_.data() + po + off, (size_t)size);
        return val;
    }
    uint64_t addr = evalOperand(t, tid, base, 8) + (uint64_t)off;
    if (space == "shared") {
        if (addr + size > shared_.size()) throw std::runtime_error("PTX: shared OOB load");
        std::memcpy(&val, shared_.data() + addr, (size_t)size);
        return val;
    }
    if (!readGlobal(addr, &val, (size_t)size))
        throw std::runtime_error("PTX: global load fault @" + std::to_string(addr));
    return val;
}

void PtxInterpreter::storeTo(Thread& t, int tid, const std::string& memRef,
                             const std::string& space, int size, uint64_t val) {
    std::string base; int64_t off;
    parseMemRef(memRef, base, off);
    uint64_t addr = evalOperand(t, tid, base, 8) + (uint64_t)off;
    if (space == "shared") {
        if (addr + size > shared_.size()) throw std::runtime_error("PTX: shared OOB store");
        std::memcpy(shared_.data() + addr, &val, (size_t)size);
        return;
    }
    if (space == "param") throw std::runtime_error("PTX: st.param outside call frames unsupported");
    if (!writeGlobal(addr, &val, (size_t)size))
        throw std::runtime_error("PTX: global store fault @" + std::to_string(addr));
}

// ── The instruction set ───────────────────────────────────────────────────────

bool PtxInterpreter::execOne(Thread& t, int tid) {
    const PtxInstr& I = kernel_.code[t.pc];

    // Predication.
    if (!I.pred.empty()) {
        bool p = false;
        auto it = t.preds.find(I.pred);
        if (it != t.preds.end()) p = it->second;
        if (p == I.predNeg) { ++t.pc; if (t.pc >= (int)kernel_.code.size()) t.done = true; return true; }
    }

    auto parts = splitDots(I.op);
    const std::string& mnem = parts[0];
    std::string ty = lastType(parts);
    int size = typeSize(ty);
    bool isF = isFloatType(ty);
    bool isS = isSignedType(ty);
    auto has = [&](const char* p) {
        for (const auto& s : parts) if (s == p) return true;
        return false;
    };
    auto A = [&](int i) -> const std::string& { return I.args[i]; };
    auto val = [&](int i) { return evalOperand(t, tid, A(i), size ? size : 8); };
    auto setReg = [&](const std::string& name, uint64_t v) {
        if (ty == "pred" || name.find("%p") == 0) t.preds[name] = v != 0;
        else t.regs[name].u = (size == 4) ? zeroExtend(v, 4) : v;
    };
    auto fval = [&](int i) {
        uint64_t v = val(i);
        return size == 8 ? asF64(v) : (double)asF32(v);
    };
    auto setF = [&](const std::string& name, double d) {
        t.regs[name].u = (size == 8) ? fromF64(d) : fromF32((float)d);
    };
    auto ival = [&](int i) -> int64_t {
        return isS ? signExtend(val(i), size) : (int64_t)zeroExtend(val(i), size);
    };

    bool branched = false;

    if (mnem == "ret" || mnem == "exit") {
        t.done = true;
        return true;
    } else if (mnem == "bar" || mnem == "barrier") {
        t.atBarrier = true;
        ++t.pc;
        releaseBarrierIfReady();
        return true;
    } else if (mnem == "bra") {
        const std::string& target = I.args.back();
        auto it = kernel_.labels.find(target);
        if (it == kernel_.labels.end())
            throw std::runtime_error("PTX: unknown label '" + target + "'");
        t.pc = it->second;
        branched = true;
    } else if (mnem == "mov" || mnem == "cvta") {
        // cvta(.to).global on a flat CPU address space is the identity.
        setReg(A(0), val(1));
    } else if (mnem == "ld") {
        std::string space = has("shared") ? "shared" : has("param") ? "param" : "global";
        uint64_t v = loadFrom(t, tid, A(1), space, size);
        if (isS && size < 8) v = (uint64_t)signExtend(v, size);
        setReg(A(0), v);
    } else if (mnem == "st") {
        std::string space = has("shared") ? "shared" : has("param") ? "param" : "global";
        storeTo(t, tid, A(0), space, size, val(1));
    } else if (mnem == "add" || mnem == "sub" || mnem == "mul" || mnem == "div" ||
               mnem == "rem" || mnem == "min" || mnem == "max") {
        if (isF) {
            double a = fval(1), b = fval(2), r = 0;
            if (mnem == "add") r = a + b;
            else if (mnem == "sub") r = a - b;
            else if (mnem == "mul") r = a * b;
            else if (mnem == "div") r = a / b;
            else if (mnem == "min") r = std::fmin(a, b);
            else if (mnem == "max") r = std::fmax(a, b);
            else throw std::runtime_error("PTX: rem on float");
            setF(A(0), r);
        } else if (mnem == "mul" && has("wide")) {
            int64_t a = ival(1), b = ival(2);
            t.regs[A(0)].u = (uint64_t)(a * b);       // result is 2× width
        } else {
            int64_t a = ival(1), b = ival(2), r = 0;
            if (mnem == "add") r = a + b;
            else if (mnem == "sub") r = a - b;
            else if (mnem == "mul") {
                if (has("hi") && size > 4) throw std::runtime_error("PTX: mul.hi.64 unsupported");
                r = has("hi") ? ((a * b) >> (size * 8)) : a * b;
            }
            else if (mnem == "div") { if (!b) throw std::runtime_error("PTX: div by zero"); r = a / b; }
            else if (mnem == "rem") { if (!b) throw std::runtime_error("PTX: rem by zero"); r = a % b; }
            else if (mnem == "min") r = std::min(a, b);
            else if (mnem == "max") r = std::max(a, b);
            setReg(A(0), (uint64_t)r);
        }
    } else if (mnem == "mad" || mnem == "fma") {
        if (isF) {
            if (size == 4)
                setF(A(0), (double)std::fmaf((float)fval(1), (float)fval(2), (float)fval(3)));
            else
                setF(A(0), std::fma(fval(1), fval(2), fval(3)));
        } else if (has("wide")) {
            // d (2× width) = a*b + c(2× width)
            int64_t r = ival(1) * ival(2) + (int64_t)evalOperand(t, tid, A(3), 8);
            t.regs[A(0)].u = (uint64_t)r;
        } else {
            if (has("hi") && size > 4)
                throw std::runtime_error("PTX: mad.hi.64 unsupported");
            int64_t r = has("hi") ? ((ival(1) * ival(2)) >> (size * 8))
                                  : ival(1) * ival(2);
            setReg(A(0), (uint64_t)(r + ival(3)));
        }
    } else if (mnem == "neg") {
        if (isF) setF(A(0), -fval(1));
        else setReg(A(0), (uint64_t)(-ival(1)));
    } else if (mnem == "abs") {
        if (isF) setF(A(0), std::fabs(fval(1)));
        else setReg(A(0), (uint64_t)std::llabs(ival(1)));
    } else if (mnem == "sqrt")  { setF(A(0), std::sqrt(fval(1)));
    } else if (mnem == "rsqrt") { setF(A(0), 1.0 / std::sqrt(fval(1)));
    } else if (mnem == "rcp")   { setF(A(0), 1.0 / fval(1));
    } else if (mnem == "ex2")   { setF(A(0), std::exp2(fval(1)));
    } else if (mnem == "lg2")   { setF(A(0), std::log2(fval(1)));
    } else if (mnem == "sin")   { setF(A(0), std::sin(fval(1)));
    } else if (mnem == "cos")   { setF(A(0), std::cos(fval(1)));
    } else if (mnem == "and" || mnem == "or" || mnem == "xor") {
        uint64_t a = val(1), b = val(2);
        uint64_t r = (mnem == "and") ? (a & b) : (mnem == "or") ? (a | b) : (a ^ b);
        if (ty == "pred") t.preds[A(0)] = r != 0;
        else setReg(A(0), r);
    } else if (mnem == "not") {
        if (ty == "pred") t.preds[A(0)] = !(val(1) != 0);
        else setReg(A(0), ~val(1));
    } else if (mnem == "shl") {
        setReg(A(0), zeroExtend(val(1), size) << (val(2) & 63));
    } else if (mnem == "shr") {
        uint64_t sh = val(2) & 63;
        if (isS) setReg(A(0), (uint64_t)(signExtend(val(1), size) >> sh));
        else setReg(A(0), zeroExtend(val(1), size) >> sh);
    } else if (mnem == "setp" || mnem == "set") {
        const std::string& cmp = parts[1];
        bool r;
        if (isF) {
            double a = fval(1), b = fval(2);
            r = cmp == "eq" ? a == b : cmp == "ne" ? a != b :
                cmp == "lt" || cmp == "ltu" ? a < b : cmp == "le" || cmp == "leu" ? a <= b :
                cmp == "gt" || cmp == "gtu" ? a > b : cmp == "ge" || cmp == "geu" ? a >= b :
                cmp == "neu" ? !(a == b) : cmp == "equ" ? a == b || std::isnan(a) || std::isnan(b)
                             : throw std::runtime_error("PTX: setp cmp " + cmp);
        } else if (isS) {
            int64_t a = ival(1), b = ival(2);
            r = cmp == "eq" ? a == b : cmp == "ne" ? a != b : cmp == "lt" ? a < b :
                cmp == "le" ? a <= b : cmp == "gt" ? a > b : cmp == "ge" ? a >= b
                            : throw std::runtime_error("PTX: setp cmp " + cmp);
        } else {
            uint64_t a = zeroExtend(val(1), size), b = zeroExtend(val(2), size);
            r = cmp == "eq" ? a == b : cmp == "ne" ? a != b :
                cmp == "lt" || cmp == "lo" ? a < b : cmp == "le" || cmp == "ls" ? a <= b :
                cmp == "gt" || cmp == "hi" ? a > b : cmp == "ge" || cmp == "hs" ? a >= b
                            : throw std::runtime_error("PTX: setp cmp " + cmp);
        }
        if (mnem == "setp") t.preds[A(0)] = r;
        else setReg(A(0), r ? (isF ? fromF32(1.0f) : ~0ull) : 0);   // set: 0 / -1 (or 1.0f)
    } else if (mnem == "selp") {
        bool p = t.preds.count(A(3)) ? t.preds[A(3)] : (val(3) != 0);
        setReg(A(0), p ? val(1) : val(2));
    } else if (mnem == "cvt") {
        // cvt[.rnd].DST.SRC — parts: cvt, [rn/rni/rzi/…], dst, src (last two are types).
        std::string dstT = parts[parts.size() - 2], srcT = parts[parts.size() - 1];
        int ss = typeSize(srcT);
        uint64_t raw = evalOperand(t, tid, A(1), ss);
        uint64_t out;
        if (isFloatType(srcT) && isFloatType(dstT)) {
            double d = ss == 8 ? asF64(raw) : (double)asF32(raw);
            out = typeSize(dstT) == 8 ? fromF64(d) : fromF32((float)d);
        } else if (isFloatType(srcT)) {                              // float → int
            double d = ss == 8 ? asF64(raw) : (double)asF32(raw);
            double r = has("rni") ? std::nearbyint(d) : std::trunc(d);   // rzi default
            out = isSignedType(dstT) ? (uint64_t)(int64_t)r : (uint64_t)r;
            out = zeroExtend(out, typeSize(dstT));
        } else if (isFloatType(dstT)) {                              // int → float
            int64_t v = isSignedType(srcT) ? signExtend(raw, ss) : (int64_t)zeroExtend(raw, ss);
            double d = isSignedType(srcT) ? (double)v : (double)(uint64_t)v;
            out = typeSize(dstT) == 8 ? fromF64(d) : fromF32((float)d);
        } else {                                                     // int → int
            int64_t v = isSignedType(srcT) ? signExtend(raw, ss) : (int64_t)zeroExtend(raw, ss);
            out = zeroExtend((uint64_t)v, typeSize(dstT));
        }
        t.regs[A(0)].u = out;
    } else {
        throw std::runtime_error("PTX: unsupported instruction '" + I.op + "' (" + I.text + ")");
    }

    if (!branched) ++t.pc;
    if (t.pc >= (int)kernel_.code.size()) t.done = true;
    return true;
}

}  // namespace debug
}  // namespace vgre
