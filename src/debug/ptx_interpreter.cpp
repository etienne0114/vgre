// PTX single-step interpreter — see include/vgre/debug/ptx_interpreter.h.

#include "vgre/debug/ptx_interpreter.h"

#include "vgre/common/atomic_rmw.h"
#include "vgre/core/texture_manager.h"   // shared sampler for vgretex*/vgresurf* ops
#include "vgre/xla/half.h"   // f16<->f32 codec (header-only, dependency-free)
#include "vgre/compiler/cpu_cuda_intrinsics.h"   // __mul64hi/__umul64hi (64-bit high multiply)

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
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

// Portable population count / count-leading-zeros over the low `bits` of v (no
// compiler builtins, so this stays correct on every toolchain the engine builds
// with). clz of 0 is the full width (matches PTX clz.b{32,64}).
int popcountBits(uint64_t v, int bits) {
    if (bits < 64) v &= (uint64_t(1) << bits) - 1;
    int c = 0; while (v) { v &= v - 1; ++c; } return c;
}
int clzBits(uint64_t v, int bits) {
    if (bits < 64) v &= (uint64_t(1) << bits) - 1;
    if (v == 0) return bits;
    int n = 0; uint64_t m = uint64_t(1) << (bits - 1);
    while (!(v & m)) { ++n; m >>= 1; } return n;
}
// Reverse the low `bits` bits of v (PTX brev.b{32,64}).
uint64_t brevBits(uint64_t v, int bits) {
    uint64_t r = 0;
    for (int i = 0; i < bits; ++i) { r = (r << 1) | (v & 1); v >>= 1; }
    return r;
}

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
    std::string err;
    if (!tryParse(ptx, entry, err)) throw std::runtime_error(err);
}

bool PtxInterpreter::tryParse(const std::string& ptx, const std::string& entry,
                              std::string& err) {
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
    if (ep == std::string::npos) {
        err = "PTX kernel not found: " + entry;
        return false;
    }
    kernel_.name = entry;

    // Parameters: "(.param .u64 name, .param .f32 name, …)".
    size_t po = ptx.find('(', ep);
    size_t body = ptx.find('{', ep);
    if (body == std::string::npos) { err = "PTX: missing kernel body"; return false; }
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
            if (name.empty() || sz <= 0) { err = "PTX: bad param: " + p; return false; }
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
            } else if (kind == ".extern") {
                // ".extern .shared .align A .b8 name[];" — dynamic `extern
                // __shared__`, sized by the launch. All such arrays alias the
                // region immediately after the static shared area (base offset =
                // the static shared byte count).
                std::string tok, name; bool isShared = false;
                while (ss >> tok) {
                    if (tok == ".shared") { isShared = true; continue; }
                    if (tok == ".align") { ss >> tok; continue; }
                    if (!tok.empty() && tok[0] != '.') name = tok;
                }
                if (isShared && !name.empty()) {
                    size_t br = name.find('[');
                    if (br != std::string::npos) name = name.substr(0, br);
                    kernel_.sharedVars[name] = kernel_.sharedBytes;   // dynamic base
                }
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
            } else if (kind == ".local") {
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
                kernel_.localBytes = (kernel_.localBytes + 15) / 16 * 16;
                kernel_.localVars[name] = kernel_.localBytes;
                kernel_.localBytes += bytes;
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
    if (kernel_.code.empty()) { err = "PTX: empty kernel body"; return false; }
    return true;
}

// ── Launch / scheduling ───────────────────────────────────────────────────────

void PtxInterpreter::launch(int gridX, int blockX, void* const* args, int numArgs) {
    // 1D convenience: {gridX,1,1} × {blockX,1,1}.
    launch(Dim3{gridX, 1, 1}, Dim3{blockX, 1, 1}, args, numArgs);
}

void PtxInterpreter::setupLaunch(const Dim3& grid, const Dim3& block, void* const* args, int numArgs) {
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
}

void PtxInterpreter::launch(const Dim3& grid, const Dim3& block, void* const* args, int numArgs) {
    setupLaunch(grid, block, args, numArgs);
    startCta(0);
}

// ── Non-throwing entry points (all exceptions handled in this TU) ────────────
bool PtxInterpreter::canParse(const std::string& ptx, const std::string& entry) noexcept {
    // Parse into a throwaway probe. tryParse reports malformed PTX by return
    // value (no exception), so nothing is ever thrown here — which is what makes
    // this safe on macOS: a thrown-and-caught std::runtime_error aborts the
    // process there when two libc++abi runtimes are loaded (catch(...) included).
    try {
        PtxInterpreter probe;
        std::string err;
        return probe.tryParse(ptx, entry, err);
    } catch (...) {
        // Belt-and-suspenders for a truly exceptional failure (e.g. bad_alloc);
        // the normal malformed-PTX path returns false above without throwing.
        return false;
    }
}

bool PtxInterpreter::runKernel(const std::string& ptx, const std::string& entry,
                               const Dim3& grid, const Dim3& block,
                               void* const* args, int numArgs,
                               size_t dynSharedBytes) noexcept {
    try {
        PtxInterpreter in(ptx, entry);
        in.setDynamicSharedBytes(dynSharedBytes);
        in.launch(grid, block, args, numArgs);
        return in.resume() == StopReason::Exited;
    } catch (...) {
        return false;
    }
}

bool PtxInterpreter::runKernelRange(const std::string& ptx, const std::string& entry,
                                    const Dim3& grid, const Dim3& block,
                                    void* const* args, int numArgs,
                                    int ctaBegin, int ctaEnd,
                                    size_t dynSharedBytes) noexcept {
    try {
        PtxInterpreter in(ptx, entry);
        in.setDynamicSharedBytes(dynSharedBytes);
        return in.runCtaRange(grid, block, args, numArgs, ctaBegin, ctaEnd);
    } catch (...) {
        return false;
    }
}

bool PtxInterpreter::runCtaRange(const Dim3& grid, const Dim3& block, void* const* args,
                                 int numArgs, int ctaBegin, int ctaEnd) {
    setupLaunch(grid, block, args, numArgs);

    if (ctaBegin < 0) ctaBegin = 0;
    if (ctaEnd > gridTotal_) ctaEnd = gridTotal_;

    // Run each CTA in the range to completion, honouring bar.sync / shfl.sync
    // between its threads — the same cooperative scheduler resume() uses, minus
    // breakpoints.
    for (int cta = ctaBegin; cta < ctaEnd; ++cta) {
        startCta(cta);
        while (true) {
            releaseBarrierIfReady();
            releaseBarrierRedIfReady();
            bool progressed = false;
            for (int tid = 0; tid < (int)threads_.size(); ++tid) {
                Thread& t = threads_[tid];
                while (runnable(t)) {
                    if (!execOne(t, tid)) break;
                    progressed = true;
                }
            }
            if (ctaFinished()) break;
            if (!progressed) {
                releaseBarrierIfReady();
                releaseBarrierRedIfReady();
                bool anyRunnable = false;
                for (const auto& t : threads_)
                    if (runnable(t)) { anyRunnable = true; break; }
                if (!anyRunnable)
                    throw std::runtime_error("PTX: deadlock — threads blocked at bar.sync / shfl.sync");
            }
        }
    }
    exited_ = true;
    return true;
}

void PtxInterpreter::startCta(int cta) {
    cta_ = cta;
    // Decompose the linear CTA index into %ctaid.{x,y,z} (x fastest).
    ctaIdx_[0] = cta % gridDim_[0];
    ctaIdx_[1] = (cta / gridDim_[0]) % gridDim_[1];
    ctaIdx_[2] = cta / (gridDim_[0] * gridDim_[1]);
    // Static `.shared` plus the launch's dynamic `extern __shared__` bytes, which
    // alias the region immediately after the static shared area.
    shared_.assign((size_t)std::max(kernel_.sharedBytes + dynSharedBytes_, 1), 0);
    threads_.assign((size_t)blockTotal_, Thread{});
    if (kernel_.localBytes > 0)
        for (auto& t : threads_) t.local.assign((size_t)kernel_.localBytes, 0);
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

void PtxInterpreter::releaseBarrierRedIfReady() {
    // CTA-wide: ready only when every non-exited thread has reached the bar.red.
    for (const auto& t : threads_)
        if (!t.done && !t.atBarrierRed) return;

    // Read each parked thread's source predicate (the last operand, optionally
    // negated with '!'), reducing over the whole block.
    auto srcPred = [&](Thread& t) -> bool {
        const PtxInstr& I = kernel_.code[t.pc];
        std::string s = I.args.back();
        bool neg = !s.empty() && s[0] == '!';
        if (neg) s = s.substr(1);
        auto it = t.preds.find(s);
        bool v = (it != t.preds.end()) ? it->second : false;
        return neg ? !v : v;
    };
    uint32_t count = 0; bool anyTrue = false, allTrue = true;
    for (auto& t : threads_) {
        if (!t.atBarrierRed) continue;                 // an exited thread
        bool p = srcPred(t);
        if (p) { ++count; anyTrue = true; } else allTrue = false;
    }

    for (auto& t : threads_) {
        if (!t.atBarrierRed) continue;
        const PtxInstr& I = kernel_.code[t.pc];
        std::vector<std::string> parts = splitDots(I.op);   // bar.red.<op>.<type>
        const std::string op = parts.size() > 2 ? parts[2] : "popc";
        if (op == "popc") t.regs[I.args[0]].u = zeroExtend(count, 4);  // count → u32 dest
        else if (op == "and") t.preds[I.args[0]] = allTrue;           // all → pred dest
        else                  t.preds[I.args[0]] = anyTrue;           // or  → pred dest
        t.atBarrierRed = false;
        ++t.pc;
        if (t.pc >= (int)kernel_.code.size()) t.done = true;
    }
}

void PtxInterpreter::releaseShflIfReady(int anyTid) {
    // Warps are 32 consecutive threads by flattened thread id.
    const int total = (int)threads_.size();
    const int warpBase = anyTid - (anyTid % 32);
    const int warpEnd = std::min(warpBase + 32, total);

    // Ready only when every active (non-exited) lane has reached the shuffle.
    for (int L = warpBase; L < warpEnd; ++L)
        if (!threads_[L].done && !threads_[L].atShfl) return;

    // Snapshot the offered values first (writes below must not see updated ones).
    uint64_t offered[32];
    for (int L = warpBase; L < warpEnd; ++L) offered[L - warpBase] = threads_[L].shflVal;

    for (int L = warpBase; L < warpEnd; ++L) {
        Thread& t = threads_[L];
        if (!t.atShfl) continue;                       // an exited lane
        const PtxInstr& I = kernel_.code[t.pc];
        std::vector<std::string> parts = splitDots(I.op);   // shfl.sync.<mode>.b32
        const std::string mode = parts.size() > 2 ? parts[2] : "idx";
        int width = (int)evalOperand(t, L, I.args[3], 4);    // c = subwarp width (our encoding)
        if (width <= 0 || width > 32) width = 32;
        const int laneArg = (int)evalOperand(t, L, I.args[2], 4);  // b = srcLane/delta/mask
        const int lane = L - warpBase;
        const int subBase = warpBase + (lane / width) * width;
        const int laneInSub = lane % width;

        int srcSub = laneInSub;
        bool own = false;
        if (mode == "idx")       srcSub = laneArg % width;
        else if (mode == "up")   { srcSub = laneInSub - laneArg; if (srcSub < 0) own = true; }
        else if (mode == "down") { srcSub = laneInSub + laneArg; if (srcSub >= width) own = true; }
        else if (mode == "bfly") { srcSub = laneInSub ^ laneArg; if (srcSub >= width) own = true; }

        const int src = subBase + srcSub;
        uint64_t result;
        if (own || src < warpBase || src >= warpEnd || threads_[src].done)
            result = t.shflVal;                        // inactive source → keep own value (CUDA)
        else
            result = offered[src - warpBase];

        t.regs[I.args[0]].u = zeroExtend(result, 4);   // write d (b32)
        t.atShfl = false;
        ++t.pc;
        if (t.pc >= (int)kernel_.code.size()) t.done = true;
    }
}

void PtxInterpreter::releaseVoteIfReady(int anyTid) {
    // Warps are 32 consecutive threads by flattened thread id (as for shuffle).
    const int total = (int)threads_.size();
    const int warpBase = anyTid - (anyTid % 32);
    const int warpEnd = std::min(warpBase + 32, total);

    // Ready only when every active (non-exited) lane has reached the vote.
    for (int L = warpBase; L < warpEnd; ++L)
        if (!threads_[L].done && !threads_[L].atVote) return;

    // Warp match (match.{any,all}.sync.b32 d[|p], a, membermask): warp-synchronous
    // like redux/vote. Each lane compares its 32-bit key `a` against every parked
    // lane in its membermask. `any` → the mask of lanes whose key equals this lane's;
    // `all` → membermask if every participating lane's key agrees (else 0), and p set.
    {
        int firstParked = -1;
        for (int L = warpBase; L < warpEnd; ++L) if (threads_[L].atVote) { firstParked = L; break; }
        if (firstParked >= 0 && splitDots(kernel_.code[threads_[firstParked].pc].op)[0] == "match") {
            uint32_t value[32] = {0}; bool parked[32] = {false};
            for (int L = warpBase; L < warpEnd; ++L) {
                Thread& t = threads_[L];
                if (!t.atVote) continue;
                parked[L - warpBase] = true;
                value[L - warpBase] = (uint32_t)evalOperand(t, L, kernel_.code[t.pc].args[1], 4);
            }
            for (int L = warpBase; L < warpEnd; ++L) {
                Thread& t = threads_[L];
                if (!t.atVote) continue;
                const PtxInstr& I = kernel_.code[t.pc];
                std::vector<std::string> parts = splitDots(I.op);   // match.<any|all>.sync.b32
                const bool all = parts.size() > 1 && parts[1] == "all";
                const uint32_t memMask = (uint32_t)evalOperand(t, L, I.args[2], 4);
                const int lane = L - warpBase;
                uint32_t part = 0;                                  // parked lanes in membermask
                for (int M = 0; M < warpEnd - warpBase; ++M)
                    if (parked[M] && (memMask & (1u << M))) part |= (1u << M);
                if (all) {
                    bool allSame = true;
                    for (int M = 0; M < 32; ++M)
                        if ((part & (1u << M)) && value[M] != value[lane]) { allSame = false; break; }
                    const std::string& dp = I.args[0];              // "d|p"
                    const size_t bar = dp.find('|');
                    const std::string d = bar == std::string::npos ? dp : dp.substr(0, bar);
                    t.regs[d].u = zeroExtend(allSame ? part : 0u, 4);
                    if (bar != std::string::npos) t.preds[dp.substr(bar + 1)] = allSame;
                } else {
                    uint32_t same = 0;
                    for (int M = 0; M < 32; ++M)
                        if ((part & (1u << M)) && value[M] == value[lane]) same |= (1u << M);
                    t.regs[I.args[0]].u = zeroExtend(same, 4);
                }
                t.atVote = false;
                ++t.pc;
                if (t.pc >= (int)kernel_.code.size()) t.done = true;
            }
            return;
        }
    }

    // Warp reduce (redux.sync.<op>.<type> d, a, membermask): the parked lanes are
    // warp-synchronous, so if the first one is at a redux they all are — each
    // participating lane gets the fold of `a` over the lanes its membermask selects.
    {
        int firstParked = -1;
        for (int L = warpBase; L < warpEnd; ++L) if (threads_[L].atVote) { firstParked = L; break; }
        if (firstParked >= 0 && splitDots(kernel_.code[threads_[firstParked].pc].op)[0] == "redux") {
            // Pass 1: compute every parked lane's fold while ALL lanes are still
            // parked (so clearing atVote below can't drop a lane from a later fold).
            std::vector<uint32_t> result(warpEnd - warpBase, 0);
            for (int L = warpBase; L < warpEnd; ++L) {
                Thread& t = threads_[L];
                if (!t.atVote) continue;
                const PtxInstr& I = kernel_.code[t.pc];
                std::vector<std::string> parts = splitDots(I.op);   // redux.sync.<op>.<type>
                const std::string op = parts.size() > 2 ? parts[2] : "add";
                const bool sgn = parts.size() > 3 && parts[3] == "s32";
                const uint32_t memMask = (uint32_t)evalOperand(t, L, I.args[2], 4);
                bool first = true; int64_t acc = 0;
                for (int M = warpBase; M < warpEnd; ++M) {
                    if (!threads_[M].atVote) continue;
                    if (!(memMask & (1u << (M - warpBase)))) continue;
                    const uint32_t raw = (uint32_t)evalOperand(threads_[M], M, kernel_.code[threads_[M].pc].args[1], 4);
                    const int64_t v = sgn ? (int64_t)(int32_t)raw : (int64_t)raw;
                    if (first) { acc = v; first = false; continue; }
                    if (op == "add")      acc = v + acc;
                    else if (op == "min") acc = std::min(acc, v);
                    else if (op == "max") acc = std::max(acc, v);
                    else if (op == "and") acc &= v;
                    else if (op == "or")  acc |= v;
                    else if (op == "xor") acc ^= v;
                }
                result[L - warpBase] = (uint32_t)acc;
            }
            // Pass 2: write results, release the lanes, advance.
            for (int L = warpBase; L < warpEnd; ++L) {
                Thread& t = threads_[L];
                if (!t.atVote) continue;
                t.regs[kernel_.code[t.pc].args[0]].u = zeroExtend(result[L - warpBase], 4);   // 32-bit result
                t.atVote = false;
                ++t.pc;
                if (t.pc >= (int)kernel_.code.size()) t.done = true;
            }
            return;
        }
    }

    // Build the raw ballot: bit `lane` set iff that lane is voting and its
    // predicate is true. activeMask tracks which lanes actually voted (for `all`).
    uint32_t predBallot = 0, activeMask = 0;
    for (int L = warpBase; L < warpEnd; ++L) {
        Thread& t = threads_[L];
        if (!t.atVote) continue;                       // an exited lane
        const int lane = L - warpBase;
        activeMask |= (1u << lane);
        const PtxInstr& I = kernel_.code[t.pc];
        if (evalOperand(t, L, I.args[1], 4) != 0)      // b = predicate
            predBallot |= (1u << lane);
    }

    for (int L = warpBase; L < warpEnd; ++L) {
        Thread& t = threads_[L];
        if (!t.atVote) continue;
        const PtxInstr& I = kernel_.code[t.pc];
        std::vector<std::string> parts = splitDots(I.op);   // vote.sync.<op>.<type>
        const std::string op = parts.size() > 2 ? parts[2] : "ballot";
        // c = membership mask (which lanes participate); result counts only those.
        const uint32_t memMask = (uint32_t)evalOperand(t, L, I.args[2], 4);
        const uint32_t masked  = predBallot & memMask;

        if (op == "ballot") {
            t.regs[I.args[0]].u = zeroExtend(masked, 4);            // b32 result
        } else if (op == "all") {
            // true iff every participating (masked & voting) lane's predicate is set
            t.preds[I.args[0]] = (masked == (memMask & activeMask));
        } else {                                                   // "any" (and "uni")
            t.preds[I.args[0]] = (masked != 0);
        }
        t.atVote = false;
        ++t.pc;
        if (t.pc >= (int)kernel_.code.size()) t.done = true;
    }
}

void PtxInterpreter::releaseWarpSyncIfReady(int anyTid) {
    // 32-lane-scoped barrier: release once every active lane of the warp arrived.
    // The pc was already advanced by the bar.warp.sync handler (as for bar.sync),
    // so this only clears the park flag.
    const int total = (int)threads_.size();
    const int warpBase = anyTid - (anyTid % 32);
    const int warpEnd = std::min(warpBase + 32, total);
    for (int L = warpBase; L < warpEnd; ++L)
        if (!threads_[L].done && !threads_[L].atWarpSync) return;
    for (int L = warpBase; L < warpEnd; ++L) threads_[L].atWarpSync = false;
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
        releaseBarrierRedIfReady();
        bool progressed = false;
        for (int tid = 0; tid < (int)threads_.size(); ++tid) {
            Thread& t = threads_[tid];
            while (runnable(t)) {
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
            releaseBarrierRedIfReady();
            for (const auto& t : threads_)
                if (runnable(t)) goto again;   // barrier/shuffle released
            throw std::runtime_error("PTX: deadlock — threads blocked at bar.sync / shfl.sync");
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
    if (t.atBarrierRed) releaseBarrierRedIfReady();
    if (runnable(t)) execOne(t, thread);
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
    auto lo = kernel_.localVars.find(s);                             // mov %r, localName
    if (lo != kernel_.localVars.end()) return (uint64_t)lo->second;
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
    if (space == "local") {
        if (addr + size > t.local.size()) throw std::runtime_error("PTX: local OOB load");
        std::memcpy(&val, t.local.data() + addr, (size_t)size);
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
    if (space == "local") {
        if (addr + size > t.local.size()) throw std::runtime_error("PTX: local OOB store");
        std::memcpy(t.local.data() + addr, &val, (size_t)size);
        return;
    }
    if (space == "param") throw std::runtime_error("PTX: st.param outside call frames unsupported");
    if (!writeGlobal(addr, &val, (size_t)size))
        throw std::runtime_error("PTX: global store fault @" + std::to_string(addr));
}

namespace {
// atom.global.add over the interpreter's raw-bits convention, returning the OLD
// value's raw bits — correct when a grid's CTAs run concurrently on separate
// interpreter instances (see InterpreterBackend::launch). `isF` selects float vs
// integer; for floats `fAdd` is the addend, for ints `intBits`. Dispatches to the
// shared lock-free RMW primitives in <vgre/common/atomic_rmw.h>.
uint64_t atomicAddGlobal(void* p, int size, bool isF, uint64_t intBits, double fAdd) {
    if (isF) {
        if (size == 8) {
            double o = vgre::common::atomicAddF64(p, fAdd);
            uint64_t r; std::memcpy(&r, &o, 8); return r;
        }
        float o = vgre::common::atomicAddF32(p, static_cast<float>(fAdd));
        uint32_t r; std::memcpy(&r, &o, 4); return static_cast<uint64_t>(r);
    }
    if (size == 8) return vgre::common::atomicAddU64(p, intBits);
    return static_cast<uint64_t>(vgre::common::atomicAddU32(p, static_cast<uint32_t>(intBits)));
}
}  // namespace

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
        if (has("warp")) {          // bar.warp.sync — 32-lane barrier (__syncwarp)
            t.atWarpSync = true;
            ++t.pc;
            releaseWarpSyncIfReady(tid);
        } else if (has("red")) {    // bar.red — CTA-wide barrier + predicate reduction
            // Park WITHOUT advancing pc: the reduction reads this instruction's
            // operands and writes its result when the whole block has arrived.
            t.atBarrierRed = true;
            releaseBarrierRedIfReady();
        } else {                    // bar.sync — CTA-wide barrier (__syncthreads)
            t.atBarrier = true;
            ++t.pc;
            releaseBarrierIfReady();
        }
        return true;
    } else if (mnem == "activemask") {
        // activemask.b32 d — bitmask of the warp's currently non-exited lanes.
        const int total = (int)threads_.size();
        const int warpBase = tid - (tid % 32);
        const int warpEnd = std::min(warpBase + 32, total);
        uint32_t mask = 0;
        for (int L = warpBase; L < warpEnd; ++L)
            if (!threads_[L].done) mask |= (1u << (L - warpBase));
        setReg(A(0), mask);
    } else if (mnem == "membar" || mnem == "fence") {
        // Memory fence (membar.{cta,gl,sys} / fence.*) — the __threadfence family.
        // Within one CTA the cooperative scheduler already runs a thread's memory
        // ops in program order, but the backend executes different CTAs on parallel
        // host threads (ThreadPool), so a device/system-scope fence orders global
        // memory that a concurrently-running block observes. Issue a real CPU
        // barrier: device/system scope → a full seq_cst fence (mfence on x86);
        // block scope → acquire-release (intra-CTA ordering only). This makes a
        // producer's writes-before-fence precede its writes-after-fence as seen by
        // another block, exactly as membar specifies.
        const bool blockScope = has("cta");
        std::atomic_thread_fence(blockScope ? std::memory_order_acq_rel
                                            : std::memory_order_seq_cst);
    } else if (mnem == "shfl") {
        // Warp shuffle: park at this instruction offering our value; the last
        // lane of the warp to arrive performs the exchange for everyone. pc is
        // NOT advanced here — releaseShflIfReady writes the result and advances.
        t.shflVal = evalOperand(t, tid, A(1), 4) & 0xffffffffu;   // 'a' (b32 value)
        t.atShfl = true;
        releaseShflIfReady(tid);
        return true;
    } else if (mnem == "vote") {
        // Warp vote (vote.sync.{ballot,any,all}): park offering our predicate; the
        // last active lane of the warp performs the reduction for everyone. pc is
        // NOT advanced here — releaseVoteIfReady writes the results and advances.
        t.atVote = true;
        releaseVoteIfReady(tid);
        return true;
    } else if (mnem == "redux") {
        // Warp reduce (redux.sync.<op>): park like a vote; releaseVoteIfReady detects
        // the redux and folds `a` across the warp's participating lanes for everyone.
        t.atVote = true;
        releaseVoteIfReady(tid);
        return true;
    } else if (mnem == "match") {
        // Warp match (match.{any,all}.sync.b32): park like a vote; releaseVoteIfReady
        // detects the match and gives each lane the mask of same-valued warp lanes.
        t.atVote = true;
        releaseVoteIfReady(tid);
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
        std::string space = has("shared") ? "shared" : has("local") ? "local" : has("param") ? "param" : "global";
        uint64_t v = loadFrom(t, tid, A(1), space, size);
        if (isS && size < 8) v = (uint64_t)signExtend(v, size);
        setReg(A(0), v);
    } else if (mnem == "st") {
        std::string space = has("shared") ? "shared" : has("local") ? "local" : has("param") ? "param" : "global";
        storeTo(t, tid, A(0), space, size, val(1));
    } else if (mnem == "atom") {
        // atom.global.<op>.<ty> dst, [addr], val  (cas: …, [addr], cmp, val) — a
        // real atomic RMW so a grid's CTAs stay correct when the backend runs them
        // on parallel instances. Returns the OLD value, matching CUDA.
        if (!has("global")) throw std::runtime_error("PTX: only atom.global.* is supported");
        std::string base; int64_t off;
        parseMemRef(A(1), base, off);
        uint64_t addr = evalOperand(t, tid, base, 8) + (uint64_t)off;
        uint64_t probe = 0;
        if (!readGlobal(addr, &probe, (size_t)size))
            throw std::runtime_error("PTX: atom global fault @" + std::to_string(addr));
        void* pp = reinterpret_cast<void*>(addr);
        const bool w8 = (size == 8);
        uint64_t old = 0;
        if (has("add")) {
            old = atomicAddGlobal(pp, size, isF, val(2), fval(2));
        } else if (has("exch")) {
            old = w8 ? vgre::common::atomicExchU64(pp, val(2))
                     : vgre::common::atomicExchU32(pp, (uint32_t)val(2));
        } else if (has("cas")) {
            old = w8 ? vgre::common::atomicCasU64(pp, val(2), val(3))
                     : vgre::common::atomicCasU32(pp, (uint32_t)val(2), (uint32_t)val(3));
        } else if (has("and")) {
            old = w8 ? vgre::common::atomicAndU64(pp, val(2))
                     : vgre::common::atomicAndU32(pp, (uint32_t)val(2));
        } else if (has("or")) {
            old = w8 ? vgre::common::atomicOrU64(pp, val(2))
                     : vgre::common::atomicOrU32(pp, (uint32_t)val(2));
        } else if (has("xor")) {
            old = w8 ? vgre::common::atomicXorU64(pp, val(2))
                     : vgre::common::atomicXorU32(pp, (uint32_t)val(2));
        } else if (has("min")) {
            old = w8 ? (isS ? (uint64_t)vgre::common::atomicMinS64(pp, (int64_t)val(2))
                            : vgre::common::atomicMinU64(pp, val(2)))
                     : (isS ? (uint64_t)(uint32_t)vgre::common::atomicMinS32(pp, (int32_t)val(2))
                            : vgre::common::atomicMinU32(pp, (uint32_t)val(2)));
        } else if (has("max")) {
            old = w8 ? (isS ? (uint64_t)vgre::common::atomicMaxS64(pp, (int64_t)val(2))
                            : vgre::common::atomicMaxU64(pp, val(2)))
                     : (isS ? (uint64_t)(uint32_t)vgre::common::atomicMaxS32(pp, (int32_t)val(2))
                            : vgre::common::atomicMaxU32(pp, (uint32_t)val(2)));
        } else {
            throw std::runtime_error("PTX: unsupported atom op '" + I.op + "'");
        }
        setReg(A(0), old);
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
        } else if (!isS && (mnem == "div" || mnem == "rem" || mnem == "min" || mnem == "max")) {
            // Unsigned div/rem/min/max: these are the integer ops whose result
            // differs by signedness (add/sub/mul.lo are identical in two's
            // complement). Operate on zero-extended magnitudes.
            uint64_t a = zeroExtend(val(1), size), b = zeroExtend(val(2), size), r = 0;
            if (mnem == "div") { if (!b) throw std::runtime_error("PTX: div by zero"); r = a / b; }
            else if (mnem == "rem") { if (!b) throw std::runtime_error("PTX: rem by zero"); r = a % b; }
            else if (mnem == "min") r = a < b ? a : b;
            else r = a > b ? a : b;
            setReg(A(0), r);
        } else {
            int64_t a = ival(1), b = ival(2), r = 0;
            if (mnem == "add") r = a + b;
            else if (mnem == "sub") r = a - b;
            else if (mnem == "mul") {
                if (has("hi"))
                    r = (size > 4) ? (isS ? vgre_cuda::__mul64hi(a, b)                             // high 64 of a 64×64 product
                                          : (int64_t)vgre_cuda::__umul64hi((uint64_t)a, (uint64_t)b))
                                   : ((a * b) >> (size * 8));                            // ≤32-bit: product fits in int64
                else r = a * b;
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
            int64_t r;
            if (has("hi"))
                r = (size > 4) ? (isS ? vgre_cuda::__mul64hi(ival(1), ival(2))
                                      : (int64_t)vgre_cuda::__umul64hi((uint64_t)ival(1), (uint64_t)ival(2)))
                               : ((ival(1) * ival(2)) >> (size * 8));
            else r = ival(1) * ival(2);
            setReg(A(0), (uint64_t)(r + ival(3)));
        }
    } else if (mnem == "neg") {
        if (isF) setF(A(0), -fval(1));
        else setReg(A(0), (uint64_t)(-ival(1)));
    } else if (mnem == "abs") {
        if (isF) setF(A(0), std::fabs(fval(1)));
        else setReg(A(0), (uint64_t)std::llabs(ival(1)));
    } else if (mnem == "popc") {
        // popc.b{32,64}: population count of the source → 32-bit result (__popc/ll).
        setReg(A(0), (uint64_t)popcountBits(val(1), size * 8));
    } else if (mnem == "clz") {
        // clz.b{32,64}: leading-zero count of the source → 32-bit result (__clz/ll).
        setReg(A(0), (uint64_t)clzBits(val(1), size * 8));
    } else if (mnem == "brev") {
        // brev.b{32,64}: bit reversal of the source (width-preserving).
        setReg(A(0), brevBits(val(1), size * 8));
    } else if (mnem == "sqrt")  { setF(A(0), std::sqrt(fval(1)));
    } else if (mnem == "rsqrt") { setF(A(0), 1.0 / std::sqrt(fval(1)));
    } else if (mnem == "rcp")   { setF(A(0), 1.0 / fval(1));
    } else if (mnem == "ex2")   { setF(A(0), std::exp2(fval(1)));
    } else if (mnem == "lg2")   { setF(A(0), std::log2(fval(1)));
    } else if (mnem == "sin")   { setF(A(0), std::sin(fval(1)));
    } else if (mnem == "cos")   { setF(A(0), std::cos(fval(1)));
    } else if (mnem == "tanh")  { setF(A(0), std::tanh(fval(1)));
    // Interpreter-tier math (no PTX approx op exists): inverse trig, cbrt, erf,
    // and the two-argument atan2. Emitted by the front-end as <op>.approx.<ty>.
    } else if (mnem == "atan")  { setF(A(0), std::atan(fval(1)));
    } else if (mnem == "asin")  { setF(A(0), std::asin(fval(1)));
    } else if (mnem == "acos")  { setF(A(0), std::acos(fval(1)));
    } else if (mnem == "cbrt")  { setF(A(0), std::cbrt(fval(1)));
    } else if (mnem == "erf")   { setF(A(0), std::erf(fval(1)));
    } else if (mnem == "atan2") { setF(A(0), std::atan2(fval(1), fval(2)));
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
            // Read source (f16 needs the codec; f32/f64 are raw bits).
            double d = (srcT == "f16") ? (double)vgre::xla::f16_to_f32((uint16_t)raw)
                     : (ss == 8)       ? asF64(raw)
                     :                   (double)asF32(raw);
            if (has("rmi")) d = std::floor(d);            // floorf/floor
            else if (has("rpi")) d = std::ceil(d);        // ceilf/ceil
            else if (has("rni")) d = std::nearbyint(d);   // round-to-nearest-even
            else if (has("rzi")) d = std::trunc(d);       // round-toward-zero
            out = (dstT == "f16")        ? (uint64_t)vgre::xla::f32_to_f16((float)d)
                : (typeSize(dstT) == 8)  ? fromF64(d)
                :                          fromF32((float)d);
        } else if (isFloatType(srcT)) {                              // float → int
            double d = ss == 8 ? asF64(raw) : (double)asF32(raw);
            // Integer rounding mode: rni=nearest-even, rmi=floor, rpi=ceil,
            // rzi=toward-zero (default).
            double r = has("rni") ? std::nearbyint(d)
                     : has("rmi") ? std::floor(d)
                     : has("rpi") ? std::ceil(d)
                     :              std::trunc(d);
            // PTX float→int conversions SATURATE to the destination range and map
            // NaN to 0 — unlike a C cast, which is undefined out of range. Clamp
            // before the narrowing cast so an out-of-range magnitude (or a division
            // by zero producing ±inf) yields INT_MAX/INT_MIN, not wrapped garbage.
            const int dbits = typeSize(dstT) * 8;
            if (std::isnan(r)) {
                out = 0;
            } else if (isSignedType(dstT)) {
                const int64_t hi = dbits >= 64 ? INT64_MAX : (((int64_t)1 << (dbits - 1)) - 1);
                const int64_t lo = dbits >= 64 ? INT64_MIN : -((int64_t)1 << (dbits - 1));
                int64_t sv = r >= (double)hi ? hi : r <= (double)lo ? lo : (int64_t)r;
                out = (uint64_t)sv;
            } else {
                const uint64_t hi = dbits >= 64 ? UINT64_MAX : (((uint64_t)1 << dbits) - 1);
                out = r <= 0.0 ? 0 : r >= (double)hi ? hi : (uint64_t)r;
            }
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
    } else if (mnem == "vgretex1d" || mnem == "vgretex1dfetch" || mnem == "vgretex2d" ||
               mnem == "vgretex3d" || mnem == "vgresurf2dread") {
        // In-house texture/surface fetch (see codegen emitTex): dest, handle(u64),
        // coords. Dispatches to the shared TextureManager → an f32 sample.
        auto& TM = ::vgre::core::TextureManager::instance();
        const uint64_t handle = evalOperand(t, tid, A(1), 8);
        float res = 0.0f;
        if (mnem == "vgretex1d")           res = TM.tex1D(handle, asF32(evalOperand(t, tid, A(2), 4)));
        else if (mnem == "vgretex1dfetch") res = TM.tex1Dfetch(handle, (int)evalOperand(t, tid, A(2), 4));
        else if (mnem == "vgretex2d")      res = TM.tex2D(handle, asF32(evalOperand(t, tid, A(2), 4)), asF32(evalOperand(t, tid, A(3), 4)));
        else if (mnem == "vgretex3d")      res = TM.tex3D(handle, asF32(evalOperand(t, tid, A(2), 4)), asF32(evalOperand(t, tid, A(3), 4)), asF32(evalOperand(t, tid, A(4), 4)));
        else { float v = 0.0f; TM.surf2Dread(handle, v, (int)evalOperand(t, tid, A(2), 4), (int)evalOperand(t, tid, A(3), 4)); res = v; }
        setReg(A(0), fromF32(res));
    } else if (mnem == "vgretex1dlayered" || mnem == "vgretex2dlayered" ||
               mnem == "vgretexcubemap" || mnem == "vgretexcubemaplayered" ||
               mnem == "vgretex2dlayeredlod") {
        // Layered (array) fetch: dest, handle(u64), coords…, layer index.
        // Cubemap fetch: dest, handle(u64), direction x,y,z.
        auto& TM = ::vgre::core::TextureManager::instance();
        const uint64_t handle = evalOperand(t, tid, A(1), 8);
        float res = 0.0f;
        if (mnem == "vgretex2dlayeredlod")
            res = TM.tex2DLayeredLod(handle, asF32(evalOperand(t, tid, A(2), 4)),
                                     asF32(evalOperand(t, tid, A(3), 4)),
                                     (int)evalOperand(t, tid, A(4), 4),
                                     asF32(evalOperand(t, tid, A(5), 4)));
        else if (mnem == "vgretex1dlayered")
            res = TM.tex1DLayered(handle, asF32(evalOperand(t, tid, A(2), 4)),
                                  (int)evalOperand(t, tid, A(3), 4));
        else if (mnem == "vgretex2dlayered")
            res = TM.tex2DLayered(handle, asF32(evalOperand(t, tid, A(2), 4)),
                                  asF32(evalOperand(t, tid, A(3), 4)),
                                  (int)evalOperand(t, tid, A(4), 4));
        else if (mnem == "vgretexcubemap")
            res = TM.texCubemap(handle, asF32(evalOperand(t, tid, A(2), 4)),
                                asF32(evalOperand(t, tid, A(3), 4)),
                                asF32(evalOperand(t, tid, A(4), 4)));
        else  // vgretexcubemaplayered
            res = TM.texCubemapLayered(handle, asF32(evalOperand(t, tid, A(2), 4)),
                                       asF32(evalOperand(t, tid, A(3), 4)),
                                       asF32(evalOperand(t, tid, A(4), 4)),
                                       (int)evalOperand(t, tid, A(5), 4));
        setReg(A(0), fromF32(res));
    } else if (mnem == "vgretex1dchan" || mnem == "vgretex2dchan" ||
               mnem == "vgretex3dchan" || mnem == "vgretex2dlod") {
        // Per-channel vector fetch (float2/3/4 textures) + explicit-LOD 2D fetch.
        // dest, handle(u64), coords…, and a trailing channel index (chan ops) or LOD
        // (lod op). Each returns one f32 lane; the front-end packs channels into a vector.
        auto& TM = ::vgre::core::TextureManager::instance();
        const uint64_t handle = evalOperand(t, tid, A(1), 8);
        float res = 0.0f;
        if (mnem == "vgretex1dchan")
            res = TM.tex1DChan(handle, asF32(evalOperand(t, tid, A(2), 4)),
                               (unsigned)evalOperand(t, tid, A(3), 4));
        else if (mnem == "vgretex2dchan")
            res = TM.tex2DChan(handle, asF32(evalOperand(t, tid, A(2), 4)),
                               asF32(evalOperand(t, tid, A(3), 4)),
                               (unsigned)evalOperand(t, tid, A(4), 4));
        else if (mnem == "vgretex3dchan")
            res = TM.tex3DChan(handle, asF32(evalOperand(t, tid, A(2), 4)),
                               asF32(evalOperand(t, tid, A(3), 4)),
                               asF32(evalOperand(t, tid, A(4), 4)),
                               (unsigned)evalOperand(t, tid, A(5), 4));
        else  // vgretex2dlod
            res = TM.tex2DLod(handle, asF32(evalOperand(t, tid, A(2), 4)),
                              asF32(evalOperand(t, tid, A(3), 4)),
                              asF32(evalOperand(t, tid, A(4), 4)));
        setReg(A(0), fromF32(res));
    } else if (mnem == "vgresurf2dwrite") {          // surf2Dwrite(val, surf, x, y) — no dest
        const float v = asF32(evalOperand(t, tid, A(0), 4));
        const uint64_t handle = evalOperand(t, tid, A(1), 8);
        ::vgre::core::TextureManager::instance().surf2Dwrite(
            handle, v, (int)evalOperand(t, tid, A(2), 4), (int)evalOperand(t, tid, A(3), 4));
    } else {
        throw std::runtime_error("PTX: unsupported instruction '" + I.op + "' (" + I.text + ")");
    }

    if (!branched) ++t.pc;
    if (t.pc >= (int)kernel_.code.size()) t.done = true;
    return true;
}

}  // namespace debug
}  // namespace vgre
