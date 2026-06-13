// VGRE XLA backend — serialization + executable C ABI (impl).
// See include/vgre/xla/{hlo_serialize,xla_c_api}.h.

#include "vgre/xla/xla_c_api.h"
#include "vgre/xla/hlo_serialize.h"

#include <cstring>
#include <map>
#include <memory>
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
void putVec(std::string& b, const std::vector<int64_t>& v) {
    putU32(b, (uint32_t)v.size());
    for (int64_t x : v) putI64(b, x);
}
std::vector<int64_t> getVec(Reader& r) {
    std::vector<int64_t> v; uint32_t n = r.u32();
    for (uint32_t i = 0; i < n && r.ok; ++i) v.push_back(r.i64());
    return v;
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
        // extended-op attributes
        putVec(b, I.lhs_batch); putVec(b, I.rhs_batch);
        putVec(b, I.lhs_contract); putVec(b, I.rhs_contract);
        putVec(b, I.slice_starts); putVec(b, I.slice_limits); putVec(b, I.slice_strides);
        putVec(b, I.pad_low); putVec(b, I.pad_high); putVec(b, I.pad_interior);
        putF32(b, I.pad_value);
        putI64(b, I.concat_dim);
        putI64(b, I.conv_in_batch); putI64(b, I.conv_in_feat);
        putI64(b, I.conv_k_out); putI64(b, I.conv_k_in);
        putI64(b, I.conv_out_batch); putI64(b, I.conv_out_feat);
        putVec(b, I.conv_in_spatial); putVec(b, I.conv_k_spatial); putVec(b, I.conv_out_spatial);
        putVec(b, I.conv_strides); putVec(b, I.conv_pad_low); putVec(b, I.conv_pad_high);
        putVec(b, I.conv_lhs_dil); putVec(b, I.conv_rhs_dil);
        putI64(b, I.conv_groups);
        putVec(b, I.gather_offset_dims); putVec(b, I.gather_collapsed);
        putVec(b, I.gather_start_map); putVec(b, I.gather_slice_sizes);
        putI64(b, I.gather_index_vector_dim);
        putVec(b, I.rw_window_dims); putVec(b, I.rw_window_strides);
        putVec(b, I.rw_pad_low); putVec(b, I.rw_pad_high);
        putVec(b, I.rw_base_dil); putVec(b, I.rw_window_dil);
        putI64(b, I.gte_index);
        putVec(b, I.dyn_slice_sizes);
        putI64(b, I.sort_dim);
        putVec(b, I.scatter_update_window_dims); putVec(b, I.scatter_inserted_window_dims);
        putVec(b, I.scatter_dims_to_operand); putI64(b, I.scatter_index_vector_dim);
        putU32(b, (uint32_t)I.subs.size());            // control-flow sub-computations
        for (const auto& s : I.subs) putStr(b, s ? serialize(*s) : std::string());
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
        I.lhs_batch = getVec(r); I.rhs_batch = getVec(r);
        I.lhs_contract = getVec(r); I.rhs_contract = getVec(r);
        I.slice_starts = getVec(r); I.slice_limits = getVec(r); I.slice_strides = getVec(r);
        I.pad_low = getVec(r); I.pad_high = getVec(r); I.pad_interior = getVec(r);
        I.pad_value = r.f32();
        I.concat_dim = r.i64();
        I.conv_in_batch = r.i64(); I.conv_in_feat = r.i64();
        I.conv_k_out = r.i64(); I.conv_k_in = r.i64();
        I.conv_out_batch = r.i64(); I.conv_out_feat = r.i64();
        I.conv_in_spatial = getVec(r); I.conv_k_spatial = getVec(r); I.conv_out_spatial = getVec(r);
        I.conv_strides = getVec(r); I.conv_pad_low = getVec(r); I.conv_pad_high = getVec(r);
        I.conv_lhs_dil = getVec(r); I.conv_rhs_dil = getVec(r);
        I.conv_groups = r.i64();
        I.gather_offset_dims = getVec(r); I.gather_collapsed = getVec(r);
        I.gather_start_map = getVec(r); I.gather_slice_sizes = getVec(r);
        I.gather_index_vector_dim = r.i64();
        I.rw_window_dims = getVec(r); I.rw_window_strides = getVec(r);
        I.rw_pad_low = getVec(r); I.rw_pad_high = getVec(r);
        I.rw_base_dil = getVec(r); I.rw_window_dil = getVec(r);
        I.gte_index = r.i64();
        I.dyn_slice_sizes = getVec(r);
        I.sort_dim = r.i64();
        I.scatter_update_window_dims = getVec(r); I.scatter_inserted_window_dims = getVec(r);
        I.scatter_dims_to_operand = getVec(r); I.scatter_index_vector_dim = r.i64();
        uint32_t nsub = r.u32();
        for (uint32_t s = 0; s < nsub && r.ok; ++s) {
            std::string sb = r.str();
            auto sm = std::make_shared<HloModule>();
            if (!sb.empty() && deserialize(sb, *sm)) I.subs.push_back(sm);
            else I.subs.push_back(nullptr);
        }
        out.add(std::move(I));
    }
    int root = (int)r.u32();
    if (!r.ok) return false;
    out.setRoot(root);
    return true;
}

// ── executable registry + C ABI ─────────────────────────────────────────────
// Executables are stored as shared_ptr<const HloModule>: compile installs an
// immutable module; execute grabs the shared_ptr under a brief lock and then
// evaluates WITHOUT holding the lock and WITHOUT copying the module. Because
// HloModule::evaluate is const and keeps all state in locals, many threads can
// execute the same (or different) executables concurrently.
namespace {
std::mutex& exeMu() { static std::mutex m; return m; }
std::map<uint64_t, std::shared_ptr<const HloModule>>& exes() {
    static std::map<uint64_t, std::shared_ptr<const HloModule>> e; return e;
}
uint64_t& nextHandle() { static uint64_t h = 1; return h; }

// Thread-local last error, surfaced to callers via vgre_xla_last_error().
std::string& lastError() { thread_local std::string e; return e; }
void setError(const std::string& m) { lastError() = m; }

std::shared_ptr<const HloModule> lookup(uint64_t exe) {
    std::lock_guard<std::mutex> lk(exeMu());
    auto it = exes().find(exe);
    return it == exes().end() ? nullptr : it->second;
}

// Bind C input buffers to the module's Parameter literals (by param_index).
bool bindParams(const HloModule& mod, const float* const* in_data, const int64_t* in_numel,
                int n_in, std::vector<Literal>& params) {
    params.assign(n_in, Literal{});
    for (size_t i = 0; i < mod.size(); ++i) {
        const HloInstruction& I = mod.instr((int)i);
        if (I.op != HloOp::Parameter) continue;
        int pi = I.param_index;
        if (pi < 0 || pi >= n_in) { setError("parameter index out of range"); return false; }
        if (in_numel[pi] != I.shape.numel()) {
            setError("input " + std::to_string(pi) + " size mismatch (got " +
                     std::to_string(in_numel[pi]) + ", expected " + std::to_string(I.shape.numel()) + ")");
            return false;
        }
        Literal lit; lit.shape = I.shape;
        lit.data.assign(in_data[pi], in_data[pi] + in_numel[pi]);
        params[pi] = std::move(lit);
    }
    return true;
}

uint64_t installModule(std::shared_ptr<const HloModule> m) {
    std::lock_guard<std::mutex> lk(exeMu());
    uint64_t h = nextHandle()++;
    exes()[h] = std::move(m);
    return h;
}
} // namespace

} // namespace xla
} // namespace vgre

using namespace vgre::xla;

extern "C" const char* vgre_xla_last_error(void) { return lastError().c_str(); }

extern "C" uint64_t vgre_xla_compile(const void* blob, size_t len) {
    if (!blob || !len) { setError("null/empty blob"); return 0; }
    auto m = std::make_shared<HloModule>();
    if (!deserialize(std::string((const char*)blob, len), *m)) { setError("malformed HLO blob"); return 0; }
    std::string verr;
    if (!m->validate(verr)) { setError("invalid module: " + verr); return 0; }
    return installModule(std::move(m));
}

extern "C" int64_t vgre_xla_output_numel(uint64_t exe) {
    auto mod = lookup(exe);
    if (!mod) { setError("invalid executable handle"); return -1; }
    return mod->instr(mod->root()).shape.numel();
}

extern "C" int64_t vgre_xla_execute(uint64_t exe, const float* const* in_data,
                                    const int64_t* in_numel, int n_in,
                                    float* out_data, int64_t out_capacity) {
    auto mod = lookup(exe);
    if (!mod) { setError("invalid executable handle"); return -1; }
    std::vector<Literal> params;
    if (!bindParams(*mod, in_data, in_numel, n_in, params)) return -1;
    Literal res;
    try {
        res = mod->evaluate(params);
    } catch (const std::exception& e) {
        setError(std::string("evaluation failed: ") + e.what());
        return -1;
    } catch (...) {
        setError("evaluation failed: unknown error");
        return -1;
    }
    if ((int64_t)res.data.size() > out_capacity) { setError("output capacity too small"); return -1; }
    std::memcpy(out_data, res.data.data(), res.data.size() * sizeof(float));
    return (int64_t)res.data.size();
}

// Multi-output execute: when the root is a Tuple (a function returning several
// arrays / a pytree), write every output flattened+concatenated into out_data
// and return the total element count. The caller splits by known per-output sizes.
extern "C" int64_t vgre_xla_execute_multi(uint64_t exe, const float* const* in_data,
                                          const int64_t* in_numel, int n_in,
                                          float* out_data, int64_t out_capacity) {
    auto mod = lookup(exe);
    if (!mod) { setError("invalid executable handle"); return -1; }
    std::vector<Literal> params;
    if (!bindParams(*mod, in_data, in_numel, n_in, params)) return -1;
    std::vector<Literal> res;
    try {
        res = mod->evaluateMulti(params);
    } catch (const std::exception& e) {
        setError(std::string("evaluation failed: ") + e.what());
        return -1;
    } catch (...) {
        setError("evaluation failed: unknown error");
        return -1;
    }
    int64_t total = 0;
    for (const auto& r : res) total += (int64_t)r.data.size();
    if (total > out_capacity) return -1;
    int64_t off = 0;
    for (const auto& r : res) {
        std::memcpy(out_data + off, r.data.data(), r.data.size() * sizeof(float));
        off += (int64_t)r.data.size();
    }
    return total;
}

extern "C" void vgre_xla_free(uint64_t exe) {
    std::lock_guard<std::mutex> lk(exeMu());
    exes().erase(exe);
}

// ── builder C ABI ────────────────────────────────────────────────────────────
namespace vgre { namespace xla { namespace {
std::mutex& bldMu() { static std::mutex m; return m; }
std::map<uint64_t, HloModule>& blds() { static std::map<uint64_t, HloModule> b; return b; }

// Run `fn(module)` under the builder lock; return its result or -1 if no builder.
template <class Fn>
int withBuilder(uint64_t b, Fn fn) {
    std::lock_guard<std::mutex> lk(bldMu());
    auto it = blds().find(b);
    if (it == blds().end()) return -1;
    return fn(it->second);
}
}}} // namespace vgre::xla::(anon)

extern "C" uint64_t vgre_xla_builder_new(void) {
    std::lock_guard<std::mutex> lk(bldMu());
    uint64_t h = nextHandle()++;
    blds()[h];  // default-construct
    return h;
}

extern "C" int vgre_xla_b_parameter(uint64_t b, int index, const int64_t* dims, int ndim) {
    return withBuilder(b, [&](HloModule& m) {
        return m.parameter(index, Shape{std::vector<int64_t>(dims, dims + ndim)});
    });
}

extern "C" int vgre_xla_b_constant(uint64_t b, const int64_t* dims, int ndim,
                                   const float* data, int n) {
    return withBuilder(b, [&](HloModule& m) {
        Shape s{std::vector<int64_t>(dims, dims + ndim)};
        return m.constant(Literal::make(s, std::vector<float>(data, data + n)));
    });
}

extern "C" int vgre_xla_b_binary(uint64_t b, int op, int lhs, int rhs) {
    return withBuilder(b, [&](HloModule& m) { return m.binary((HloOp)op, lhs, rhs); });
}

extern "C" int vgre_xla_b_unary(uint64_t b, int op, int x) {
    return withBuilder(b, [&](HloModule& m) { return m.unary((HloOp)op, x); });
}

extern "C" int vgre_xla_b_broadcast(uint64_t b, int x, const int64_t* out_dims, int n_out,
                                    const int64_t* bcast_dims, int n_bd) {
    return withBuilder(b, [&](HloModule& m) {
        return m.broadcast(x, Shape{std::vector<int64_t>(out_dims, out_dims + n_out)},
                           std::vector<int64_t>(bcast_dims, bcast_dims + n_bd));
    });
}

extern "C" int vgre_xla_b_reshape(uint64_t b, int x, const int64_t* out_dims, int n_out) {
    return withBuilder(b, [&](HloModule& m) {
        return m.reshape(x, Shape{std::vector<int64_t>(out_dims, out_dims + n_out)});
    });
}

extern "C" int vgre_xla_b_transpose(uint64_t b, int x, const int64_t* perm, int n_perm) {
    return withBuilder(b, [&](HloModule& m) {
        return m.transpose(x, std::vector<int64_t>(perm, perm + n_perm));
    });
}

extern "C" int vgre_xla_b_dot(uint64_t b, int lhs, int rhs) {
    return withBuilder(b, [&](HloModule& m) { return m.dot(lhs, rhs); });
}

extern "C" int vgre_xla_b_reduce(uint64_t b, int x, const int64_t* dims, int n_dims,
                                 const char* kind, float init) {
    return withBuilder(b, [&](HloModule& m) {
        return m.reduce(x, std::vector<int64_t>(dims, dims + n_dims), kind ? kind : "sum", init);
    });
}

extern "C" int vgre_xla_b_compare(uint64_t b, int lhs, int rhs, const char* dir) {
    return withBuilder(b, [&](HloModule& m) { return m.compare(lhs, rhs, dir ? dir : "GT"); });
}

extern "C" int vgre_xla_b_select(uint64_t b, int pred, int on_true, int on_false) {
    return withBuilder(b, [&](HloModule& m) { return m.select(pred, on_true, on_false); });
}

extern "C" void vgre_xla_b_set_root(uint64_t b, int id) {
    withBuilder(b, [&](HloModule& m) { m.setRoot(id); return 0; });
}

namespace { std::vector<int64_t> vec(const int64_t* p, int n) { return std::vector<int64_t>(p, p + n); } }

extern "C" int vgre_xla_b_dot_general(uint64_t b, int lhs, int rhs,
                                      const int64_t* lb, int nlb, const int64_t* rb, int nrb,
                                      const int64_t* lc, int nlc, const int64_t* rc, int nrc,
                                      const int64_t* out_dims, int n_out) {
    return withBuilder(b, [&](HloModule& m) {
        return m.dotGeneral(lhs, rhs, vec(lb, nlb), vec(rb, nrb), vec(lc, nlc), vec(rc, nrc),
                            Shape{vec(out_dims, n_out)});
    });
}

extern "C" int vgre_xla_b_concatenate(uint64_t b, const int* xs, int n_xs, int64_t dim,
                                      const int64_t* out_dims, int n_out) {
    return withBuilder(b, [&](HloModule& m) {
        return m.concatenate(std::vector<int>(xs, xs + n_xs), dim, Shape{vec(out_dims, n_out)});
    });
}

extern "C" int vgre_xla_b_slice(uint64_t b, int x, const int64_t* starts, const int64_t* limits,
                                const int64_t* strides, int n, const int64_t* out_dims, int n_out) {
    return withBuilder(b, [&](HloModule& m) {
        return m.slice(x, vec(starts, n), vec(limits, n), vec(strides, n), Shape{vec(out_dims, n_out)});
    });
}

extern "C" int vgre_xla_b_pad(uint64_t b, int x, int pad_val, const int64_t* low, const int64_t* high,
                              const int64_t* interior, int n, const int64_t* out_dims, int n_out) {
    return withBuilder(b, [&](HloModule& m) {
        return m.pad(x, pad_val, vec(low, n), vec(high, n), vec(interior, n), Shape{vec(out_dims, n_out)});
    });
}

extern "C" int vgre_xla_b_convolution(uint64_t b, int lhs, int rhs, const int64_t* out_dims, int n_out,
                                      int in_batch, int in_feat, int k_out, int k_in,
                                      int out_batch, int out_feat,
                                      const int64_t* in_sp, const int64_t* k_sp, const int64_t* out_sp, int n_sp,
                                      const int64_t* strides, const int64_t* pad_lo, const int64_t* pad_hi,
                                      const int64_t* rhs_dil, int groups) {
    return withBuilder(b, [&](HloModule& m) {
        int id = m.convolution(lhs, rhs, Shape{vec(out_dims, n_out)});
        HloInstruction& I = m.last();
        I.conv_in_batch = in_batch; I.conv_in_feat = in_feat;
        I.conv_k_out = k_out; I.conv_k_in = k_in;
        I.conv_out_batch = out_batch; I.conv_out_feat = out_feat;
        I.conv_in_spatial = vec(in_sp, n_sp); I.conv_k_spatial = vec(k_sp, n_sp);
        I.conv_out_spatial = vec(out_sp, n_sp);
        I.conv_strides = vec(strides, n_sp);
        I.conv_pad_low = vec(pad_lo, n_sp); I.conv_pad_high = vec(pad_hi, n_sp);
        I.conv_rhs_dil = vec(rhs_dil, n_sp);
        I.conv_lhs_dil.assign(n_sp, 1);
        I.conv_groups = groups < 1 ? 1 : groups;
        return id;
    });
}

extern "C" int vgre_xla_b_gather(uint64_t b, int operand, int indices, const int64_t* out_dims, int n_out,
                                 const int64_t* offset_dims, int n_off, const int64_t* collapsed, int n_col,
                                 const int64_t* start_map, int n_sm, const int64_t* slice_sizes, int n_ss,
                                 int index_vector_dim) {
    return withBuilder(b, [&](HloModule& m) {
        int id = m.gather(operand, indices, Shape{vec(out_dims, n_out)});
        HloInstruction& I = m.last();
        I.gather_offset_dims = vec(offset_dims, n_off);
        I.gather_collapsed = vec(collapsed, n_col);
        I.gather_start_map = vec(start_map, n_sm);
        I.gather_slice_sizes = vec(slice_sizes, n_ss);
        I.gather_index_vector_dim = index_vector_dim;
        return id;
    });
}

extern "C" int vgre_xla_b_reduce_window(uint64_t b, int x, int init, const char* kind,
                                        const int64_t* out_dims, int n_out,
                                        const int64_t* win_dims, const int64_t* win_strides,
                                        const int64_t* pad_lo, const int64_t* pad_hi,
                                        const int64_t* base_dil, const int64_t* win_dil, int n) {
    return withBuilder(b, [&](HloModule& m) {
        int id = m.reduceWindow(x, init, kind ? kind : "max", Shape{vec(out_dims, n_out)});
        HloInstruction& I = m.last();
        I.rw_window_dims = vec(win_dims, n); I.rw_window_strides = vec(win_strides, n);
        I.rw_pad_low = vec(pad_lo, n); I.rw_pad_high = vec(pad_hi, n);
        I.rw_base_dil = vec(base_dil, n); I.rw_window_dil = vec(win_dil, n);
        return id;
    });
}

// ── control flow builders ────────────────────────────────────────────────────
namespace {
// Move a builder's module out of the registry as a shared sub-computation.
std::shared_ptr<HloModule> takeBuilderModule(uint64_t h) {
    std::lock_guard<std::mutex> lk(bldMu());
    auto it = blds().find(h);
    if (it == blds().end()) return nullptr;
    auto m = std::make_shared<HloModule>(std::move(it->second));
    blds().erase(it);
    return m;
}
std::vector<int> ivec(const int* p, int n) { return std::vector<int>(p, p + n); }
}

extern "C" int vgre_xla_b_while(uint64_t b, const int* inits, int n_init, uint64_t cond, uint64_t body,
                                const int64_t* primary_dims, int n_primary) {
    auto condM = takeBuilderModule(cond);
    auto bodyM = takeBuilderModule(body);
    if (!condM || !bodyM) return -1;
    return withBuilder(b, [&](HloModule& m) {
        return m.whileOp(ivec(inits, n_init), condM, bodyM, Shape{vec(primary_dims, n_primary)});
    });
}

extern "C" int vgre_xla_b_tuple(uint64_t b, const int* elems, int n) {
    return withBuilder(b, [&](HloModule& m) { return m.tuple(ivec(elems, n)); });
}

extern "C" int vgre_xla_b_get_tuple_element(uint64_t b, int src, int index,
                                            const int64_t* out_dims, int n_out) {
    return withBuilder(b, [&](HloModule& m) {
        return m.getTupleElement(src, index, Shape{vec(out_dims, n_out)});
    });
}

extern "C" int vgre_xla_b_dynamic_slice(uint64_t b, int operand, const int* starts, int n_starts,
                                        const int64_t* sizes, int n_sizes,
                                        const int64_t* out_dims, int n_out) {
    return withBuilder(b, [&](HloModule& m) {
        return m.dynamicSlice(operand, ivec(starts, n_starts), vec(sizes, n_sizes),
                              Shape{vec(out_dims, n_out)});
    });
}

extern "C" int vgre_xla_b_dynamic_update_slice(uint64_t b, int operand, int update,
                                               const int* starts, int n_starts,
                                               const int64_t* out_dims, int n_out) {
    return withBuilder(b, [&](HloModule& m) {
        return m.dynamicUpdateSlice(operand, update, ivec(starts, n_starts), Shape{vec(out_dims, n_out)});
    });
}

extern "C" int vgre_xla_b_reverse(uint64_t b, int x, const int64_t* dims, int n) {
    return withBuilder(b, [&](HloModule& m) { return m.reverse(x, vec(dims, n)); });
}

extern "C" int vgre_xla_b_iota(uint64_t b, const int64_t* out_dims, int n_out, int64_t dim) {
    return withBuilder(b, [&](HloModule& m) { return m.iota(Shape{vec(out_dims, n_out)}, dim); });
}

extern "C" int vgre_xla_b_sort(uint64_t b, const int* operands, int n_ops, int64_t dim, uint64_t cmp) {
    auto cmpM = takeBuilderModule(cmp);
    if (!cmpM) return -1;
    return withBuilder(b, [&](HloModule& m) {
        return m.sort(ivec(operands, n_ops), dim, cmpM);
    });
}

extern "C" int vgre_xla_b_reduce_general(uint64_t b, const int* operands, int n_ops,
                                         const int64_t* dims, int n_dims, uint64_t body,
                                         const int64_t* primary_dims, int n_primary) {
    auto bodyM = takeBuilderModule(body);
    if (!bodyM) return -1;
    return withBuilder(b, [&](HloModule& m) {
        return m.reduceGeneral(ivec(operands, n_ops), vec(dims, n_dims), bodyM,
                               Shape{vec(primary_dims, n_primary)});
    });
}

extern "C" int vgre_xla_b_scatter(uint64_t b, int operand, int indices, int updates, uint64_t combiner,
                                  const int64_t* update_window_dims, int n_uwd,
                                  const int64_t* inserted_window_dims, int n_iwd,
                                  const int64_t* dims_to_operand, int n_dto, int64_t index_vector_dim) {
    auto cm = takeBuilderModule(combiner);
    if (!cm) return -1;
    return withBuilder(b, [&](HloModule& m) {
        int id = m.scatter(operand, indices, updates, cm);
        HloInstruction& I = m.last();
        I.scatter_update_window_dims = vec(update_window_dims, n_uwd);
        I.scatter_inserted_window_dims = vec(inserted_window_dims, n_iwd);
        I.scatter_dims_to_operand = vec(dims_to_operand, n_dto);
        I.scatter_index_vector_dim = index_vector_dim;
        return id;
    });
}

extern "C" uint64_t vgre_xla_b_compile(uint64_t b) {
    auto m = std::make_shared<HloModule>();
    {
        std::lock_guard<std::mutex> lk(bldMu());
        auto it = blds().find(b);
        if (it == blds().end()) { setError("invalid builder handle"); return 0; }
        *m = std::move(it->second);
        blds().erase(it);
    }
    std::string verr;
    if (!m->validate(verr)) { setError("invalid module: " + verr); return 0; }
    return installModule(std::move(m));
}

extern "C" void vgre_xla_b_free(uint64_t b) {
    std::lock_guard<std::mutex> lk(bldMu());
    blds().erase(b);
}
