#include "vgre/compiler/ptx_translator.h"
#include "vgre/common/logger.h"
#include <regex>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cctype>

namespace vgre {
namespace compiler {

namespace {

// Trim whitespace
static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return {};
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Strip %r0, %f1, etc. → "r0", "f1"; map to C operands by index.
// Parse "output : input : clobber" constraint string.
struct AsmConstraints {
    std::vector<std::string> outputs;   // e.g. "=r"
    std::vector<std::string> inputs;    // e.g. "r"
    std::vector<std::string> clobbers;
};

static AsmConstraints parseConstraints(const std::string& cs) {
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

static const TranslateMap& getMap() {
    static const TranslateMap kMap = {
        // ── Integer arithmetic ─────────────────────────────────────────────
        {"add.s32",  [](auto& o){ return o[0]+" = "+o[1]+" + "+o[2]+";"; }},
        {"add.u32",  [](auto& o){ return o[0]+" = "+o[1]+" + "+o[2]+";"; }},
        {"add.s64",  [](auto& o){ return o[0]+" = "+o[1]+" + "+o[2]+";"; }},
        {"sub.s32",  [](auto& o){ return o[0]+" = "+o[1]+" - "+o[2]+";"; }},
        {"mul.lo.s32",[](auto& o){ return o[0]+" = (int)((int)"+o[1]+"*(int)"+o[2]+");"; }},
        {"mul.lo.u32",[](auto& o){ return o[0]+" = (unsigned)((unsigned)"+o[1]+"*(unsigned)"+o[2]+");"; }},
        {"mul.hi.u32",[](auto& o){
            return o[0]+" = (unsigned)(((unsigned long long)"+o[1]+"*(unsigned long long)"+o[2]+")>>32);";
        }},
        {"mad.lo.s32",[](auto& o){ return o[0]+" = "+o[1]+"*"+o[2]+"+"+o[3]+";"; }},
        {"mad.lo.u32",[](auto& o){ return o[0]+" = "+o[1]+"*"+o[2]+"+"+o[3]+";"; }},
        {"div.s32",  [](auto& o){ return o[0]+" = "+o[1]+" / "+o[2]+";"; }},
        {"rem.s32",  [](auto& o){ return o[0]+" = "+o[1]+" % "+o[2]+";"; }},
        {"neg.s32",  [](auto& o){ return o[0]+" = -("+o[1]+");"; }},
        {"abs.s32",  [](auto& o){ return o[0]+" = __builtin_abs("+o[1]+");"; }},
        {"min.s32",  [](auto& o){ return o[0]+" = ("+o[1]+"<"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        {"max.s32",  [](auto& o){ return o[0]+" = ("+o[1]+">"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        // ── Floating-point ─────────────────────────────────────────────────
        {"add.f32",  [](auto& o){ return o[0]+" = "+o[1]+" + "+o[2]+";"; }},
        {"add.rn.f32",[](auto& o){ return o[0]+" = "+o[1]+" + "+o[2]+";"; }},
        {"sub.f32",  [](auto& o){ return o[0]+" = "+o[1]+" - "+o[2]+";"; }},
        {"mul.f32",  [](auto& o){ return o[0]+" = "+o[1]+" * "+o[2]+";"; }},
        {"mul.rn.f32",[](auto& o){ return o[0]+" = "+o[1]+" * "+o[2]+";"; }},
        {"div.approx.f32",[](auto& o){ return o[0]+" = "+o[1]+" / "+o[2]+";"; }},
        {"div.rn.f32",[](auto& o){ return o[0]+" = "+o[1]+" / "+o[2]+";"; }},
        {"sqrt.approx.f32",[](auto& o){ return o[0]+" = __builtin_sqrtf("+o[1]+");"; }},
        {"rsqrt.approx.f32",[](auto& o){ return o[0]+" = 1.0f/__builtin_sqrtf("+o[1]+");"; }},
        {"fma.rn.f32",[](auto& o){ return o[0]+" = "+o[1]+"*"+o[2]+"+"+o[3]+";"; }},
        {"neg.f32",  [](auto& o){ return o[0]+" = -("+o[1]+");"; }},
        {"abs.f32",  [](auto& o){ return o[0]+" = __builtin_fabsf("+o[1]+");"; }},
        {"min.f32",  [](auto& o){ return o[0]+" = ("+o[1]+"<"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        {"max.f32",  [](auto& o){ return o[0]+" = ("+o[1]+">"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        // ── Bitwise ─────────────────────────────────────────────────────────
        {"and.b32",  [](auto& o){ return o[0]+" = "+o[1]+" & "+o[2]+";"; }},
        {"or.b32",   [](auto& o){ return o[0]+" = "+o[1]+" | "+o[2]+";"; }},
        {"xor.b32",  [](auto& o){ return o[0]+" = "+o[1]+" ^ "+o[2]+";"; }},
        {"not.b32",  [](auto& o){ return o[0]+" = ~"+o[1]+";"; }},
        {"shl.b32",  [](auto& o){ return o[0]+" = "+o[1]+" << "+o[2]+";"; }},
        {"shr.s32",  [](auto& o){ return o[0]+" = (int)"+o[1]+" >> "+o[2]+";"; }},
        {"shr.u32",  [](auto& o){ return o[0]+" = (unsigned)"+o[1]+" >> "+o[2]+";"; }},
        // ── Memory (simplistic: treat ptr operand as C pointer) ────────────
        {"ld.global.f32",[](auto& o){ return o[0]+" = *(float*)("+o[1]+");"; }},
        {"ld.global.u32",[](auto& o){ return o[0]+" = *(unsigned*)("+o[1]+");"; }},
        {"st.global.f32",[](auto& o){ return "*(float*)("+o[0]+") = "+o[1]+";"; }},
        {"st.global.u32",[](auto& o){ return "*(unsigned*)("+o[0]+") = "+o[1]+";"; }},
        // ── Move / conversion ───────────────────────────────────────────────
        {"mov.u32",  [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        {"mov.s32",  [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        {"mov.f32",  [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        {"cvt.rn.f32.s32",[](auto& o){ return o[0]+" = (float)("+o[1]+");"; }},
        {"cvt.rn.f32.u32",[](auto& o){ return o[0]+" = (float)("+o[1]+");"; }},
        {"cvt.rz.s32.f32",[](auto& o){ return o[0]+" = (int)("+o[1]+");"; }},
        // ── Bit counting ─────────────────────────────────────────────────────
        {"popc.b32", [](auto& o){ return o[0]+" = __builtin_popcount("+o[1]+");"; }},
        {"clz.b32",  [](auto& o){ return o[0]+" = __builtin_clz("+o[1]+");"; }},
        // ── FP64 arithmetic ────────────────────────────────────────────────
        {"add.f64",   [](auto& o){ return o[0]+" = "+o[1]+" + "+o[2]+";"; }},
        {"add.rn.f64",[](auto& o){ return o[0]+" = "+o[1]+" + "+o[2]+";"; }},
        {"sub.f64",   [](auto& o){ return o[0]+" = "+o[1]+" - "+o[2]+";"; }},
        {"mul.f64",   [](auto& o){ return o[0]+" = "+o[1]+" * "+o[2]+";"; }},
        {"mul.rn.f64",[](auto& o){ return o[0]+" = "+o[1]+" * "+o[2]+";"; }},
        {"div.rn.f64",[](auto& o){ return o[0]+" = "+o[1]+" / "+o[2]+";"; }},
        {"fma.rn.f64",[](auto& o){ return o[0]+" = "+o[1]+"*"+o[2]+"+"+o[3]+";"; }},
        {"sqrt.rn.f64",[](auto& o){ return o[0]+" = __builtin_sqrt("+o[1]+");"; }},
        {"neg.f64",   [](auto& o){ return o[0]+" = -("+o[1]+");"; }},
        {"abs.f64",   [](auto& o){ return o[0]+" = __builtin_fabs("+o[1]+");"; }},
        {"min.f64",   [](auto& o){ return o[0]+" = ("+o[1]+"<"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        {"max.f64",   [](auto& o){ return o[0]+" = ("+o[1]+">"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        // ── 64-bit integer arithmetic ──────────────────────────────────────
        {"add.u64",   [](auto& o){ return o[0]+" = (unsigned long long)("+o[1]+"+("+o[2]+"));"; }},
        {"add.s64",   [](auto& o){ return o[0]+" = (long long)("+o[1]+"+("+o[2]+"));"; }},
        {"sub.u64",   [](auto& o){ return o[0]+" = (unsigned long long)("+o[1]+"-("+o[2]+"));"; }},
        {"mul.lo.u64",[](auto& o){ return o[0]+" = (unsigned long long)("+o[1]+"*("+o[2]+"));"; }},
        {"mul.lo.s64",[](auto& o){ return o[0]+" = (long long)("+o[1]+"*("+o[2]+"));"; }},
        {"mul.hi.u64",[](auto& o){
            return o[0]+" = (unsigned long long)(__uint128_t("+o[1]+")*__uint128_t("+o[2]+")>>64);";
        }},
        {"mad.lo.u64",[](auto& o){ return o[0]+" = (unsigned long long)("+o[1]+"*("+o[2]+")+"+o[3]+");"; }},
        {"neg.s64",   [](auto& o){ return o[0]+" = -(long long)("+o[1]+");"; }},
        {"min.u64",   [](auto& o){ return o[0]+" = ("+o[1]+"<(unsigned long long)"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        {"max.u64",   [](auto& o){ return o[0]+" = ("+o[1]+">(unsigned long long)"+o[2]+"?"+o[1]+":"+o[2]+");"; }},
        {"shr.u64",   [](auto& o){ return o[0]+" = (unsigned long long)"+o[1]+" >> "+o[2]+";"; }},
        {"shl.b64",   [](auto& o){ return o[0]+" = (unsigned long long)"+o[1]+" << "+o[2]+";"; }},
        {"and.b64",   [](auto& o){ return o[0]+" = (unsigned long long)"+o[1]+" & "+o[2]+";"; }},
        {"or.b64",    [](auto& o){ return o[0]+" = (unsigned long long)"+o[1]+" | "+o[2]+";"; }},
        {"xor.b64",   [](auto& o){ return o[0]+" = (unsigned long long)"+o[1]+" ^ "+o[2]+";"; }},
        // ── Memory — shared memory ─────────────────────────────────────────
        {"ld.shared.f32",[](auto& o){ return o[0]+" = *(float*)("+o[1]+");"; }},
        {"ld.shared.u32",[](auto& o){ return o[0]+" = *(unsigned*)("+o[1]+");"; }},
        {"ld.shared.s32",[](auto& o){ return o[0]+" = *(int*)("+o[1]+");"; }},
        {"ld.shared.f64",[](auto& o){ return o[0]+" = *(double*)("+o[1]+");"; }},
        {"ld.shared.u64",[](auto& o){ return o[0]+" = *(unsigned long long*)("+o[1]+");"; }},
        {"ld.shared.v2.f32",[](auto& o){
            return "{float* _vp=(float*)("+o[2]+"); "+o[0]+"=_vp[0]; "+o[1]+"=_vp[1];}";
        }},
        {"st.shared.f32",[](auto& o){ return "*(float*)("+o[0]+") = "+o[1]+";"; }},
        {"st.shared.u32",[](auto& o){ return "*(unsigned*)("+o[0]+") = "+o[1]+";"; }},
        {"st.shared.s32",[](auto& o){ return "*(int*)("+o[0]+") = "+o[1]+";"; }},
        {"st.shared.f64",[](auto& o){ return "*(double*)("+o[0]+") = "+o[1]+";"; }},
        {"st.shared.u64",[](auto& o){ return "*(unsigned long long*)("+o[0]+") = "+o[1]+";"; }},
        // ── Memory — global vectorized ─────────────────────────────────────
        {"ld.global.f64",[](auto& o){ return o[0]+" = *(double*)("+o[1]+");"; }},
        {"ld.global.s32",[](auto& o){ return o[0]+" = *(int*)("+o[1]+");"; }},
        {"ld.global.s64",[](auto& o){ return o[0]+" = *(long long*)("+o[1]+");"; }},
        {"ld.global.u64",[](auto& o){ return o[0]+" = *(unsigned long long*)("+o[1]+");"; }},
        {"ld.global.v2.f32",[](auto& o){
            return "{float* _vp=(float*)("+o[2]+"); "+o[0]+"=_vp[0]; "+o[1]+"=_vp[1];}";
        }},
        {"ld.global.v4.f32",[](auto& o){
            return "{float* _vp=(float*)("+o[4]+"); "+o[0]+"=_vp[0]; "+o[1]+"=_vp[1]; "+
                   o[2]+"=_vp[2]; "+o[3]+"=_vp[3];}";
        }},
        {"ld.global.nc.f32",[](auto& o){ return o[0]+" = *(const float*)("+o[1]+");"; }},
        {"ld.global.nc.f64",[](auto& o){ return o[0]+" = *(const double*)("+o[1]+");"; }},
        {"ld.global.cs.f32",[](auto& o){ return o[0]+" = *(const float*)("+o[1]+");"; }},
        {"st.global.f64",  [](auto& o){ return "*(double*)("+o[0]+") = "+o[1]+";"; }},
        {"st.global.s32",  [](auto& o){ return "*(int*)("+o[0]+") = "+o[1]+";"; }},
        {"st.global.u64",  [](auto& o){ return "*(unsigned long long*)("+o[0]+") = "+o[1]+";"; }},
        {"st.global.v2.f32",[](auto& o){
            return "{float* _vp=(float*)("+o[0]+"); _vp[0]="+o[1]+"; _vp[1]="+o[2]+";}";
        }},
        {"st.global.v4.f32",[](auto& o){
            return "{float* _vp=(float*)("+o[0]+"); _vp[0]="+o[1]+"; _vp[1]="+o[2]+
                   "; _vp[2]="+o[3]+"; _vp[3]="+o[4]+";}";
        }},
        // ── Local memory ───────────────────────────────────────────────────
        {"ld.local.f32", [](auto& o){ return o[0]+" = *(float*)("+o[1]+");"; }},
        {"ld.local.u32", [](auto& o){ return o[0]+" = *(unsigned*)("+o[1]+");"; }},
        {"st.local.f32", [](auto& o){ return "*(float*)("+o[0]+") = "+o[1]+";"; }},
        {"st.local.u32", [](auto& o){ return "*(unsigned*)("+o[0]+") = "+o[1]+";"; }},
        // ── Atomic operations ──────────────────────────────────────────────
        {"atom.global.add.f32",[](auto& o){
            return o[0]+" = __atomic_fetch_add((float*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.global.add.s32",[](auto& o){
            return o[0]+" = __atomic_fetch_add((int*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.global.add.u32",[](auto& o){
            return o[0]+" = __atomic_fetch_add((unsigned*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.global.add.u64",[](auto& o){
            return o[0]+" = __atomic_fetch_add((unsigned long long*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"atom.global.max.s32",[](auto& o){
            return "{int _vv="+o[2]+"; int _cur=*(int*)("+o[1]+"); while(_vv>_cur && !__atomic_compare_exchange_n((int*)("+o[1]+"), &_cur, _vv, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)){} "+o[0]+"=_cur;}";
        }},
        {"atom.global.min.s32",[](auto& o){
            return "{int _vv="+o[2]+"; int _cur=*(int*)("+o[1]+"); while(_vv<_cur && !__atomic_compare_exchange_n((int*)("+o[1]+"), &_cur, _vv, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)){} "+o[0]+"=_cur;}";
        }},
        {"atom.global.cas.b32",[](auto& o){
            return "{unsigned _exp=(unsigned)"+o[2]+"; __atomic_compare_exchange_n((unsigned*)("+o[1]+"), &_exp, "+o[3]+", 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); "+o[0]+"=_exp;}";
        }},
        {"atom.global.exch.b32",[](auto& o){
            return o[0]+" = __atomic_exchange_n((unsigned*)("+o[1]+"), "+o[2]+", __ATOMIC_SEQ_CST);";
        }},
        {"red.global.add.f32",[](auto& o){
            return "__atomic_fetch_add((float*)("+o[0]+"), "+o[1]+", __ATOMIC_SEQ_CST);";
        }},
        {"red.global.add.s32",[](auto& o){
            return "__atomic_fetch_add((int*)("+o[0]+"), "+o[1]+", __ATOMIC_SEQ_CST);";
        }},
        // ── Warp voting ────────────────────────────────────────────────────
        {"vote.sync.all.pred",[](auto& o){ return o[0]+" = __all_sync("+o[2]+", "+o[1]+");"; }},
        {"vote.sync.any.pred",[](auto& o){ return o[0]+" = __any_sync("+o[2]+", "+o[1]+");"; }},
        {"vote.sync.ballot.b32",[](auto& o){ return o[0]+" = __ballot_sync("+o[2]+", "+o[1]+");"; }},
        {"activemask.b32",[](auto& o){ return o[0]+" = 0xFFFFFFFFu;"; }},
        // ── Warp shuffle ───────────────────────────────────────────────────
        {"shfl.sync.idx.b32",[](auto& o){
            // shfl.sync.idx.b32 dst, src, laneSel, memberMask, mask (5 operands in PTX)
            return o[0]+" = __shfl_sync((unsigned)"+o[4]+", "+o[1]+", (int)"+o[2]+", "+o[3]+");";
        }},
        {"shfl.sync.down.b32",[](auto& o){
            return o[0]+" = __shfl_down_sync((unsigned)"+o[4]+", "+o[1]+", "+o[2]+", "+o[3]+");";
        }},
        {"shfl.sync.up.b32",[](auto& o){
            return o[0]+" = __shfl_up_sync((unsigned)"+o[4]+", "+o[1]+", "+o[2]+", "+o[3]+");";
        }},
        {"shfl.sync.bfly.b32",[](auto& o){
            return o[0]+" = __shfl_xor_sync((unsigned)"+o[4]+", "+o[1]+", "+o[2]+", "+o[3]+");";
        }},
        // ── Convert (more variants) ────────────────────────────────────────
        {"cvt.rn.f32.f16",[](auto& o){ return o[0]+" = (float)("+o[1]+");"; }},
        {"cvt.rn.f16.f32",[](auto& o){ return o[0]+" = (__half)("+o[1]+");"; }},
        {"cvt.rn.f64.f32",[](auto& o){ return o[0]+" = (double)("+o[1]+");"; }},
        {"cvt.rn.f32.f64",[](auto& o){ return o[0]+" = (float)("+o[1]+");"; }},
        {"cvt.rn.f64.s32",[](auto& o){ return o[0]+" = (double)("+o[1]+");"; }},
        {"cvt.rn.f64.u32",[](auto& o){ return o[0]+" = (double)(unsigned)("+o[1]+");"; }},
        {"cvt.rz.s32.f64",[](auto& o){ return o[0]+" = (int)("+o[1]+");"; }},
        {"cvt.rn.u32.f32",[](auto& o){ return o[0]+" = (unsigned)("+o[1]+");"; }},
        {"cvt.s32.f32",   [](auto& o){ return o[0]+" = (int)("+o[1]+");"; }},
        {"cvt.u32.s32",   [](auto& o){ return o[0]+" = (unsigned)("+o[1]+");"; }},
        {"cvt.s32.u32",   [](auto& o){ return o[0]+" = (int)("+o[1]+");"; }},
        {"cvt.u64.u32",   [](auto& o){ return o[0]+" = (unsigned long long)(unsigned)("+o[1]+");"; }},
        {"cvt.u64.s32",   [](auto& o){ return o[0]+" = (unsigned long long)(int)("+o[1]+");"; }},
        {"cvt.s64.s32",   [](auto& o){ return o[0]+" = (long long)("+o[1]+");"; }},
        {"cvt.sat.u8.f32",[](auto& o){
            return o[0]+" = (unsigned char)(("+o[1]+"<0.f?0.f:("+o[1]+">255.f?255.f:"+o[1]+"))+0.5f);";
        }},
        // ── Move — extra variants ──────────────────────────────────────────
        {"mov.f64",   [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        {"mov.u64",   [](auto& o){ return o[0]+" = (unsigned long long)("+o[1]+");"; }},
        {"mov.s64",   [](auto& o){ return o[0]+" = (long long)("+o[1]+");"; }},
        {"mov.b32",   [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        {"mov.b64",   [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        // ── Predicate / select ─────────────────────────────────────────────
        {"setp.lt.f32",[](auto& o){ return o[0]+" = ((float)("+o[1]+")<(float)("+o[2]+"));"; }},
        {"setp.gt.f32",[](auto& o){ return o[0]+" = ((float)("+o[1]+")>(float)("+o[2]+"));"; }},
        {"setp.le.f32",[](auto& o){ return o[0]+" = ((float)("+o[1]+")<=(float)("+o[2]+"));"; }},
        {"setp.ge.f32",[](auto& o){ return o[0]+" = ((float)("+o[1]+")>=(float)("+o[2]+"));"; }},
        {"setp.eq.f32",[](auto& o){ return o[0]+" = ((float)("+o[1]+")==(float)("+o[2]+"));"; }},
        {"setp.ne.f32",[](auto& o){ return o[0]+" = ((float)("+o[1]+")!=(float)("+o[2]+"));"; }},
        {"setp.lt.s32",[](auto& o){ return o[0]+" = ((int)("+o[1]+")<(int)("+o[2]+"));"; }},
        {"setp.gt.s32",[](auto& o){ return o[0]+" = ((int)("+o[1]+")>(int)("+o[2]+"));"; }},
        {"setp.le.s32",[](auto& o){ return o[0]+" = ((int)("+o[1]+")<=(int)("+o[2]+"));"; }},
        {"setp.ge.s32",[](auto& o){ return o[0]+" = ((int)("+o[1]+")>=(int)("+o[2]+"));"; }},
        {"setp.eq.s32",[](auto& o){ return o[0]+" = ((int)("+o[1]+")==(int)("+o[2]+"));"; }},
        {"setp.ne.s32",[](auto& o){ return o[0]+" = ((int)("+o[1]+")!=(int)("+o[2]+"));"; }},
        {"setp.eq.u32",[](auto& o){ return o[0]+" = ((unsigned)("+o[1]+")==(unsigned)("+o[2]+"));"; }},
        {"setp.ne.u32",[](auto& o){ return o[0]+" = ((unsigned)("+o[1]+")!=(unsigned)("+o[2]+"));"; }},
        {"setp.lt.u32",[](auto& o){ return o[0]+" = ((unsigned)("+o[1]+")<(unsigned)("+o[2]+"));"; }},
        {"setp.ge.u32",[](auto& o){ return o[0]+" = ((unsigned)("+o[1]+")>=(unsigned)("+o[2]+"));"; }},
        {"selp.f32",  [](auto& o){ return o[0]+" = ("+o[3]+") ? ("+o[1]+") : ("+o[2]+");"; }},
        {"selp.s32",  [](auto& o){ return o[0]+" = ("+o[3]+") ? (int)("+o[1]+") : (int)("+o[2]+");"; }},
        {"selp.u32",  [](auto& o){ return o[0]+" = ("+o[3]+") ? (unsigned)("+o[1]+") : (unsigned)("+o[2]+");"; }},
        {"selp.b32",  [](auto& o){ return o[0]+" = ("+o[3]+") ? "+o[1]+" : "+o[2]+";"; }},
        // ── Control flow ───────────────────────────────────────────────────
        {"bar.sync",  [](auto&){ return "/* bar.sync — handled by __syncthreads */"; }},
        {"bar.arrive",[](auto&){ return "/* bar.arrive */"; }},
        {"membar.gl", [](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"membar.sys",[](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"membar.cta",[](auto&){ return "__atomic_thread_fence(__ATOMIC_SEQ_CST);"; }},
        {"ret",       [](auto&){ return "return;"; }},
        // ── Type punning ───────────────────────────────────────────────────
        {"mov.b32",   [](auto& o){ return o[0]+" = "+o[1]+";"; }},
        // ── Special register reads ─────────────────────────────────────────
        {"mov.u32.tid.x",  [](auto& o){ return o[0]+" = threadIdx.x;"; }},
        {"mov.u32.tid.y",  [](auto& o){ return o[0]+" = threadIdx.y;"; }},
        {"mov.u32.ntid.x", [](auto& o){ return o[0]+" = blockDim.x;"; }},
        {"mov.u32.ctaid.x",[](auto& o){ return o[0]+" = blockIdx.x;"; }},
        {"mov.u32.nctaid.x",[](auto& o){ return o[0]+" = gridDim.x;"; }},
        // ── FP32 intrinsics ────────────────────────────────────────────────
        {"ex2.approx.f32",[](auto& o){ return o[0]+" = __builtin_exp2f("+o[1]+");"; }},
        {"lg2.approx.f32",[](auto& o){ return o[0]+" = __builtin_log2f("+o[1]+");"; }},
        {"sin.approx.f32",[](auto& o){ return o[0]+" = __builtin_sinf("+o[1]+");"; }},
        {"cos.approx.f32",[](auto& o){ return o[0]+" = __builtin_cosf("+o[1]+");"; }},
        {"rcp.approx.f32",[](auto& o){ return o[0]+" = 1.0f/"+o[1]+";"; }},
        {"rcp.rn.f32",    [](auto& o){ return o[0]+" = 1.0f/"+o[1]+";"; }},
        // ── Type-punning bitcast ───────────────────────────────────────────
        {"bitcast.b32.f32",[](auto& o){
            return "{unsigned _bt; __builtin_memcpy(&_bt, &("+o[1]+"), 4); "+o[0]+"=_bt;}";
        }},
        {"bitcast.f32.b32",[](auto& o){
            return "{float _bt; __builtin_memcpy(&_bt, &("+o[1]+"), 4); "+o[0]+"=_bt;}";
        }},
    };
    return kMap;
}

// Split comma-separated operands respecting brackets
static std::vector<std::string> splitOperands(const std::string& s) {
    std::vector<std::string> r;
    std::string cur;
    int depth = 0;
    for (char c : s) {
        if (c == '[' || c == '(') ++depth;
        else if (c == ']' || c == ')') --depth;
        if (c == ',' && depth == 0) { r.push_back(trim(cur)); cur.clear(); }
        else cur += c;
    }
    if (!cur.empty()) r.push_back(trim(cur));
    return r;
}

} // namespace

std::string PTXTranslator::translateInstruction(
    const std::string& instr, const std::string& operands)
{
    const auto& m = getMap();
    auto it = m.find(instr);
    if (it == m.end())
        return "/* PTX: " + instr + " " + operands + " */";
    auto ops = splitOperands(operands);
    try { return it->second(ops); }
    catch (...) { return "/* PTX operand error: " + instr + " */"; }
}

std::string PTXTranslator::translateBlock(
    const std::string& ptxBody,
    const std::string& /*constraints*/,
    const std::string& /*clobbers*/)
{
    std::ostringstream out;
    std::istringstream lines(ptxBody);
    std::string line;
    while (std::getline(lines, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '/' || t[0] == '@') {
            out << "  " << "/* " << t << " */\n";
            continue;
        }
        // Remove trailing semicolon
        if (!t.empty() && t.back() == ';') t.pop_back();

        // Split: first token is opcode, rest is operands
        size_t sp = t.find(' ');
        std::string opcode  = (sp == std::string::npos) ? t : t.substr(0, sp);
        std::string operStr = (sp == std::string::npos) ? "" : trim(t.substr(sp + 1));
        std::transform(opcode.begin(), opcode.end(), opcode.begin(), ::tolower);

        out << "  " << translateInstruction(opcode, operStr) << "\n";
    }
    return out.str();
}

std::string PTXTranslator::translate(const std::string& source) {
    // Pattern: asm [volatile] ("ptx_body" : constraints...)
    // We match both single-string and multi-line forms.
    // Match: asm [volatile] ("body" : constraints)
    static const std::regex kAsmRe(
        "\\b(?:__asm__|asm)\\s*(?:volatile\\s*)?\\(\\s*\"([^\"\\\\]*(?:\\\\.[^\"\\\\]*)*)\"\\s*(:[^)]*)?\\s*\\)",
        std::regex::ECMAScript);

    std::string result = source;
    std::string out;
    out.reserve(source.size());

    auto it  = std::sregex_iterator(result.begin(), result.end(), kAsmRe);
    auto end = std::sregex_iterator();
    size_t pos = 0;

    for (; it != end; ++it) {
        const auto& m = *it;
        out += result.substr(pos, m.position() - pos);

        std::string ptxBody = m[1].str();
        std::string rest    = m.size() > 2 ? m[2].str() : "";

        // Replace escaped newlines (\n\t etc.) with real newlines
        std::string body;
        for (size_t i = 0; i < ptxBody.size(); ++i) {
            if (ptxBody[i] == '\\' && i+1 < ptxBody.size()) {
                switch (ptxBody[i+1]) {
                    case 'n': body += '\n'; ++i; break;
                    case 't': body += '\t'; ++i; break;
                    default:  body += ptxBody[i]; break;
                }
            } else body += ptxBody[i];
        }

        out += "/* PTX begin */\n";
        out += "{\n";
        out += translateBlock(body, rest, "");
        out += "}\n";
        out += "/* PTX end */";

        pos = m.position() + m.length();
    }
    out += result.substr(pos);

    if (pos > 0)
        VGRE_LOG_DEBUG("PTXTranslator",
            "Translated inline PTX assembly (" + std::to_string(pos) + " chars processed)");
    return out;
}

} // namespace compiler
} // namespace vgre
