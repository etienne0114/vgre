// VGRE XLA backend — serialization + executable C ABI (impl).
// See include/vgre/xla/{hlo_serialize,xla_c_api}.h.

#include "vgre/xla/xla_c_api.h"
#include "vgre/xla/hlo_serialize.h"

#include <cstring>
#include <map>
#include <mutex>

namespace vgre {
namespace xla {

namespace {
void putU8(std::string& b, uint8_t v) { b.push_back((char)v); }
void putU32(std::string& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((char)((v >> (8 * i)) & 0xFF)); }
void putU64(std::string& b, uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back((char)((v >> (8 * i)) & 0xFF)); }
void putI64(std::string& b, int64_t v) { putU64(b, (uint64_t)v); }
void putF32(std::string& b, float f) { uint32_t u; std::memcpy(&u, &f, 4); putU32(b, u); }
void putStr(std::string& b, const std::string& s) { putU32(b, (uint32_t)s.size()); b += s; }

struct Reader {
    const char* p; const char* e; bool ok = true;
    Reader(const std::string& s) : p(s.data()), e(s.data() + s.size()) {}
    uint8_t u8() { if (p + 1 > e) { ok = false; return 0; } return (uint8_t)*p++; }
    uint32_t u32() { if (p + 4 > e) { ok = false; return 0; } uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= (uint32_t)(uint8_t)p[i] << (8 * i); p += 4; return v; }
    uint64_t u64() { if (p + 8 > e) { ok = false; return 0; } uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= (uint64_t)(uint8_t)p[i] << (8 * i); p += 8; return v; }
    int64_t i64() { return (int64_t)u64(); }
    float f32() { uint32_t u = u32(); float f; std::memcpy(&f, &u, 4); return f; }
    std::string str() { uint32_t n = u32(); if (!ok || p + n > e) { ok = false; return {}; } std::string s(p, p + n); p += n; return s; }
};

void putShape(std::string& b, const Shape& s) {
    putU32(b, (uint32_t)s.dims.size());
    for (int64_t d : s.dims) putI64(b, d);
}
Shape getShape(Reader& r) {
    Shape s; uint32_t n = r.u32();
    for (uint32_t i = 0; i < n && r.ok; ++i) s.dims.push_back(r.i64());
    return s;
}

constexpr char kMagic[6] = {'V','X','L','A','1','\0'};
} // namespace

std::string serialize(const HloModule& m) {
    std::string b(kMagic, 6);
    putU32(b, (uint32_t)m.size());
    for (size_t i = 0; i < m.size(); ++i) {
        const HloInstruction& I = m.instr((int)i);
        putU8(b, (uint8_t)I.op);
        putShape(b, I.shape);
        putU32(b, (uint32_t)I.operands.size());
        for (int o : I.operands) putU32(b, (uint32_t)o);
        putI64(b, I.param_index);
        // literal (only meaningful for Constant)
        putShape(b, I.literal.shape);
        putU32(b, (uint32_t)I.literal.data.size());
        for (float f : I.literal.data) putF32(b, f);
        putI64(b, I.iota_dim);
        putU32(b, (uint32_t)I.dimensions.size());
        for (int64_t d : I.dimensions) putI64(b, d);
        putStr(b, I.compare_dir);
        putStr(b, I.reduce_kind);
        putF32(b, I.init_value);
    }
    putU32(b, (uint32_t)m.root());
    return b;
}

bool deserialize(const std::string& blob, HloModule& out) {
    if (blob.size() < 6 || std::memcmp(blob.data(), kMagic, 6) != 0) return false;
    std::string body(blob.begin() + 6, blob.end());
    Reader r(body);
    uint32_t n = r.u32();
    for (uint32_t i = 0; i < n && r.ok; ++i) {
        HloInstruction I;
        I.op = (HloOp)r.u8();
        I.shape = getShape(r);
        uint32_t no = r.u32();
        for (uint32_t k = 0; k < no && r.ok; ++k) I.operands.push_back((int)r.u32());
        I.param_index = (int)r.i64();
        I.literal.shape = getShape(r);
        uint32_t nd = r.u32();
        for (uint32_t k = 0; k < nd && r.ok; ++k) I.literal.data.push_back(r.f32());
        I.iota_dim = r.i64();
        uint32_t ndim = r.u32();
        for (uint32_t k = 0; k < ndim && r.ok; ++k) I.dimensions.push_back(r.i64());
        I.compare_dir = r.str();
        I.reduce_kind = r.str();
        I.init_value = r.f32();
        out.add(std::move(I));
    }
    int root = (int)r.u32();
    if (!r.ok) return false;
    out.setRoot(root);
    return true;
}

// ── executable registry + C ABI ─────────────────────────────────────────────
namespace {
std::mutex& exeMu() { static std::mutex m; return m; }
std::map<uint64_t, HloModule>& exes() { static std::map<uint64_t, HloModule> e; return e; }
uint64_t& nextHandle() { static uint64_t h = 1; return h; }
} // namespace

} // namespace xla
} // namespace vgre

using namespace vgre::xla;

extern "C" uint64_t vgre_xla_compile(const void* blob, size_t len) {
    if (!blob || !len) return 0;
    HloModule m;
    if (!deserialize(std::string((const char*)blob, len), m)) return 0;
    std::lock_guard<std::mutex> lk(exeMu());
    uint64_t h = nextHandle()++;
    exes()[h] = std::move(m);
    return h;
}

extern "C" int64_t vgre_xla_output_numel(uint64_t exe) {
    std::lock_guard<std::mutex> lk(exeMu());
    auto it = exes().find(exe);
    if (it == exes().end()) return -1;
    return it->second.instr(it->second.root()).shape.numel();
}

extern "C" int64_t vgre_xla_execute(uint64_t exe, const float* const* in_data,
                                    const int64_t* in_numel, int n_in,
                                    float* out_data, int64_t out_capacity) {
    HloModule mod;
    {
        std::lock_guard<std::mutex> lk(exeMu());
        auto it = exes().find(exe);
        if (it == exes().end()) return -1;
        mod = it->second; // copy out; evaluate() is const but keep the lock short
    }
    std::vector<Literal> params(n_in);
    // Fill parameters by their param_index from the instruction list.
    for (size_t i = 0; i < mod.size(); ++i) {
        const HloInstruction& I = mod.instr((int)i);
        if (I.op != HloOp::Parameter) continue;
        int pi = I.param_index;
        if (pi < 0 || pi >= n_in) return -1;
        Literal lit;
        lit.shape = I.shape;
        int64_t nm = in_numel[pi];
        if (nm != I.shape.numel()) return -1;
        lit.data.assign(in_data[pi], in_data[pi] + nm);
        params[pi] = std::move(lit);
    }
    Literal res;
    try {
        res = mod.evaluate(params);
    } catch (...) {
        return -1;
    }
    if ((int64_t)res.data.size() > out_capacity) return -1;
    std::memcpy(out_data, res.data.data(), res.data.size() * sizeof(float));
    return (int64_t)res.data.size();
}

extern "C" void vgre_xla_free(uint64_t exe) {
    std::lock_guard<std::mutex> lk(exeMu());
    exes().erase(exe);
}
