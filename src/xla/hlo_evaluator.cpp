// VGRE XLA backend — HLO interpreter (impl).  See include/vgre/xla/hlo.h.

#include "vgre/xla/hlo.h"
#include "vgre/xla/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <thread>

namespace vgre {
namespace xla {

namespace {

std::vector<int64_t> rowMajorStrides(const Shape& s) {
    std::vector<int64_t> st(s.dims.size(), 1);
    for (int i = static_cast<int>(s.dims.size()) - 2; i >= 0; --i) st[i] = st[i + 1] * s.dims[i + 1];
    return st;
}
// linear index → multi-index coords
void decode(int64_t lin, const std::vector<int64_t>& strides, std::vector<int64_t>& coords) {
    for (size_t i = 0; i < strides.size(); ++i) { coords[i] = lin / strides[i]; lin %= strides[i]; }
}
int64_t encode(const std::vector<int64_t>& coords, const std::vector<int64_t>& strides) {
    int64_t lin = 0;
    for (size_t i = 0; i < strides.size(); ++i) lin += coords[i] * strides[i];
    return lin;
}

float applyBinary(HloOp op, float a, float b) {
    switch (op) {
        case HloOp::Add: return a + b;
        case HloOp::Subtract: return a - b;
        case HloOp::Multiply: return a * b;
        case HloOp::Divide: return a / b;
        case HloOp::Maximum: return a > b ? a : b;
        case HloOp::Minimum: return a < b ? a : b;
        case HloOp::Power: return std::pow(a, b);
        case HloOp::And: return (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f;
        case HloOp::Or: return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f;
        case HloOp::Xor: return ((a != 0.0f) != (b != 0.0f)) ? 1.0f : 0.0f;
        default: throw std::runtime_error("not a binary op");
    }
}
float applyUnary(HloOp op, float a) {
    switch (op) {
        case HloOp::Negate: return -a;
        case HloOp::Exp: return std::exp(a);
        case HloOp::Log: return std::log(a);
        case HloOp::Tanh: return std::tanh(a);
        case HloOp::Abs: return std::fabs(a);
        case HloOp::Rsqrt: return 1.0f / std::sqrt(a);
        case HloOp::Sqrt: return std::sqrt(a);
        case HloOp::Sign: return (a > 0.0f) - (a < 0.0f);
        case HloOp::Floor: return std::floor(a);
        case HloOp::Ceil: return std::ceil(a);
        case HloOp::Logistic: return 1.0f / (1.0f + std::exp(-a));
        case HloOp::Not: return (a != 0.0f) ? 0.0f : 1.0f;
        case HloOp::IsFinite: return std::isfinite(a) ? 1.0f : 0.0f;
        case HloOp::RoundNearestEven: return std::nearbyint(a);   // ties to even (banker's)
        case HloOp::RoundNearestAfz: return std::round(a);        // ties away from zero
        case HloOp::Erf: return std::erf(a);
        case HloOp::Erfc: return std::erfc(a);
        case HloOp::Cos: return std::cos(a);
        case HloOp::Sin: return std::sin(a);
        default: throw std::runtime_error("not a unary op");
    }
}
// Parallelize a 0..n loop over the shared work-stealing pool when the work is
// large enough. The pool is sized by VGRE_XLA_THREADS (default min(hw,8); set 1
// for deterministic serial execution) so a multi-tenant host can bound cores per
// executable. `body(i)` must be independent across i (the interpreter writes
// disjoint output elements) and must not throw.
template <class Fn>
void parallelFor(int64_t n, int64_t work_estimate, Fn body) {
    if (n <= 1 || work_estimate < 8192) {
        for (int64_t i = 0; i < n; ++i) body(i);
        return;
    }
    ThreadPool& pool = ThreadPool::global();
    if (pool.concurrency() < 2) {           // VGRE_XLA_THREADS=1 → serial
        for (int64_t i = 0; i < n; ++i) body(i);
        return;
    }
    // a few chunks per worker keeps load balanced via stealing
    int64_t grain = std::max<int64_t>(1, n / (static_cast<int64_t>(pool.concurrency()) * 4));
    std::function<void(int64_t)> fn(body);
    pool.parallelFor(n, grain, fn);
}

bool cmp(const std::string& dir, float a, float b) {
    if (dir == "GT") return a > b;
    if (dir == "GE") return a >= b;
    if (dir == "LT") return a < b;
    if (dir == "LE") return a <= b;
    if (dir == "EQ") return a == b;
    if (dir == "NE") return a != b;
    throw std::runtime_error("bad compare dir: " + dir);
}

} // namespace

std::vector<Literal> HloModule::evaluateMulti(const std::vector<Literal>& params) const {
    std::vector<Literal> vals(instrs_.size());              // primary (element-0) value
    std::vector<std::vector<Literal>> multi(instrs_.size()); // full tuple for Tuple/While

    // Liveness for buffer reuse: every op produces a *fresh* output from reads of
    // its operands (no aliasing), so an operand's buffer is dead — and can be
    // freed — after the last instruction that reads it. This bounds peak memory
    // by the live set rather than the sum of all intermediates, which is what
    // lets deep/large models fit in limited RAM. last_use[v] = max index that
    // references v as an operand (the root is pinned).
    std::vector<int64_t> last_use(instrs_.size(), -1);
    for (size_t j = 0; j < instrs_.size(); ++j)
        for (int op : instrs_[j].operands)
            if (op >= 0 && op < (int)instrs_.size()) last_use[op] = (int64_t)j;
    if (root_ >= 0 && root_ < (int)instrs_.size()) last_use[root_] = (int64_t)instrs_.size();

    for (size_t idx = 0; idx < instrs_.size(); ++idx) {
        const HloInstruction& I = instrs_[idx];
        Literal out;
        out.shape = I.shape;
        out.data.assign(I.shape.numel(), 0.0f);

        switch (I.op) {
            case HloOp::Parameter: {
                if (I.param_index < 0 || I.param_index >= (int)params.size())
                    throw std::runtime_error("missing parameter");
                out = params[I.param_index];
                break;
            }
            case HloOp::Constant: out = I.literal; break;
            case HloOp::Iota: {
                auto st = rowMajorStrides(I.shape);
                std::vector<int64_t> c(I.shape.dims.size());
                for (int64_t i = 0; i < out.shape.numel(); ++i) {
                    decode(i, st, c);
                    out.data[i] = static_cast<float>(c[I.iota_dim]);
                }
                break;
            }
            case HloOp::Add: case HloOp::Subtract: case HloOp::Multiply: case HloOp::Divide:
            case HloOp::Maximum: case HloOp::Minimum: case HloOp::Power:
            case HloOp::And: case HloOp::Or: case HloOp::Xor: {
                const Literal& a = vals[I.operands[0]];
                const Literal& b = vals[I.operands[1]];
                if (!(a.shape == b.shape)) throw std::runtime_error("binary op shape mismatch");
                for (size_t i = 0; i < a.data.size(); ++i) out.data[i] = applyBinary(I.op, a.data[i], b.data[i]);
                break;
            }
            case HloOp::Negate: case HloOp::Exp: case HloOp::Log: case HloOp::Tanh:
            case HloOp::Abs: case HloOp::Rsqrt: case HloOp::Sqrt: case HloOp::Sign:
            case HloOp::Floor: case HloOp::Ceil: case HloOp::Logistic: case HloOp::Not:
            case HloOp::IsFinite: case HloOp::RoundNearestEven: case HloOp::RoundNearestAfz:
            case HloOp::Erf: case HloOp::Erfc: case HloOp::Cos: case HloOp::Sin: {
                const Literal& a = vals[I.operands[0]];
                for (size_t i = 0; i < a.data.size(); ++i) out.data[i] = applyUnary(I.op, a.data[i]);
                break;
            }
            case HloOp::Compare: {
                const Literal& a = vals[I.operands[0]];
                const Literal& b = vals[I.operands[1]];
                for (size_t i = 0; i < a.data.size(); ++i) out.data[i] = cmp(I.compare_dir, a.data[i], b.data[i]) ? 1.0f : 0.0f;
                break;
            }
            case HloOp::Select: {
                const Literal& p = vals[I.operands[0]];
                const Literal& t = vals[I.operands[1]];
                const Literal& f = vals[I.operands[2]];
                // StableHLO allows a scalar predicate (and scalar branches) that
                // broadcast across the result; index those as element 0.
                bool ps = p.data.size() == 1, ts = t.data.size() == 1, fs = f.data.size() == 1;
                for (size_t i = 0; i < out.data.size(); ++i)
                    out.data[i] = ((ps ? p.data[0] : p.data[i]) != 0.0f)
                                      ? (ts ? t.data[0] : t.data[i])
                                      : (fs ? f.data[0] : f.data[i]);
                break;
            }
            case HloOp::Broadcast: {
                const Literal& a = vals[I.operands[0]];
                auto outSt = rowMajorStrides(I.shape);
                auto inSt = rowMajorStrides(a.shape);
                std::vector<int64_t> oc(I.shape.dims.size()), ic(a.shape.dims.size());
                for (int64_t i = 0; i < out.shape.numel(); ++i) {
                    decode(i, outSt, oc);
                    // Map each input dim to its output dim; size-1 input dims are
                    // stretched (degenerate broadcast), so clamp their index to 0.
                    for (size_t k = 0; k < I.dimensions.size(); ++k)
                        ic[k] = (a.shape.dims[k] == 1) ? 0 : oc[I.dimensions[k]];
                    out.data[i] = a.data[encode(ic, inSt)];
                }
                break;
            }
            case HloOp::Reshape: {
                const Literal& a = vals[I.operands[0]];
                out.data = a.data; // row-major contiguous reinterpret
                break;
            }
            case HloOp::Transpose: {
                const Literal& a = vals[I.operands[0]];
                auto outSt = rowMajorStrides(I.shape);
                auto inSt = rowMajorStrides(a.shape);
                std::vector<int64_t> oc(I.shape.dims.size()), ic(a.shape.dims.size());
                for (int64_t i = 0; i < out.shape.numel(); ++i) {
                    decode(i, outSt, oc);
                    for (size_t d = 0; d < I.dimensions.size(); ++d) ic[I.dimensions[d]] = oc[d];
                    out.data[i] = a.data[encode(ic, inSt)];
                }
                break;
            }
            case HloOp::Dot: {
                const Literal& a = vals[I.operands[0]]; // [M,K]
                const Literal& b = vals[I.operands[1]]; // [K,N]
                int64_t M = a.shape.dims[0], K = a.shape.dims[1], N = b.shape.dims[1];
                if (b.shape.dims[0] != K) throw std::runtime_error("dot contracting-dim mismatch");
                parallelFor(M, M * N * K, [&](int64_t i) {
                    for (int64_t j = 0; j < N; ++j) {
                        float s = 0;
                        for (int64_t k = 0; k < K; ++k) s += a.data[i * K + k] * b.data[k * N + j];
                        out.data[i * N + j] = s;
                    }
                });
                break;
            }
            case HloOp::Reduce: {
                const Literal& a = vals[I.operands[0]];
                auto inSt = rowMajorStrides(a.shape);
                auto outSt = rowMajorStrides(I.shape);
                // which input dims are reduced
                std::vector<bool> isRed(a.shape.dims.size(), false);
                for (int64_t d : I.dimensions) isRed[d] = true;
                std::vector<int> keep;
                for (int d = 0; d < (int)a.shape.dims.size(); ++d) if (!isRed[d]) keep.push_back(d);

                for (auto& v : out.data) v = I.init_value; // init, then accumulate
                for (int64_t i = 0; i < a.shape.numel(); ++i) {
                    std::vector<int64_t> ic(a.shape.dims.size());
                    decode(i, inSt, ic);
                    std::vector<int64_t> oc(keep.size());
                    for (size_t k = 0; k < keep.size(); ++k) oc[k] = ic[keep[k]];
                    int64_t o = encode(oc, outSt);
                    float x = a.data[i], &acc = out.data[o];
                    if (I.reduce_kind == "sum") acc += x;
                    else if (I.reduce_kind == "prod") acc *= x;
                    else if (I.reduce_kind == "max") acc = acc > x ? acc : x;
                    else if (I.reduce_kind == "min") acc = acc < x ? acc : x;
                    else throw std::runtime_error("bad reduce kind: " + I.reduce_kind);
                }
                break;
            }
            case HloOp::DotGeneral: {
                const Literal& a = vals[I.operands[0]];
                const Literal& b = vals[I.operands[1]];
                auto aSt = rowMajorStrides(a.shape), bSt = rowMajorStrides(b.shape);
                auto oSt = rowMajorStrides(I.shape);
                // free dims = dims that are neither batch nor contracting, in order
                auto freeDims = [](const Shape& s, const std::vector<int64_t>& batch,
                                   const std::vector<int64_t>& con) {
                    std::vector<int64_t> f;
                    for (int64_t d = 0; d < s.rank(); ++d) {
                        bool used = false;
                        for (int64_t x : batch) used |= (x == d);
                        for (int64_t x : con) used |= (x == d);
                        if (!used) f.push_back(d);
                    }
                    return f;
                };
                auto lf = freeDims(a.shape, I.lhs_batch, I.lhs_contract);
                auto rf = freeDims(b.shape, I.rhs_batch, I.rhs_contract);
                int64_t nb = (int64_t)I.lhs_batch.size();
                // contracting extent
                int64_t C = 1;
                for (int64_t d : I.lhs_contract) C *= a.shape.dims[d];
                std::vector<int64_t> cExt(I.lhs_contract.size());
                for (size_t k = 0; k < I.lhs_contract.size(); ++k) cExt[k] = a.shape.dims[I.lhs_contract[k]];
                parallelFor(I.shape.numel(), I.shape.numel() * C, [&](int64_t o) {
                    std::vector<int64_t> oc(I.shape.rank()), ai(a.shape.rank()), bi(b.shape.rank());
                    decode(o, oSt, oc);
                    // output layout: [batch..., lhs_free..., rhs_free...]
                    for (int64_t k = 0; k < nb; ++k) {
                        ai[I.lhs_batch[k]] = oc[k];
                        bi[I.rhs_batch[k]] = oc[k];
                    }
                    for (size_t k = 0; k < lf.size(); ++k) ai[lf[k]] = oc[nb + (int64_t)k];
                    for (size_t k = 0; k < rf.size(); ++k) bi[rf[k]] = oc[nb + (int64_t)lf.size() + (int64_t)k];
                    float s = 0.0f;
                    std::vector<int64_t> cc(I.lhs_contract.size(), 0);
                    for (int64_t t = 0; t < C; ++t) {
                        int64_t rem = t;
                        for (int k = (int)cc.size() - 1; k >= 0; --k) { cc[k] = rem % cExt[k]; rem /= cExt[k]; }
                        for (size_t k = 0; k < I.lhs_contract.size(); ++k) {
                            ai[I.lhs_contract[k]] = cc[k];
                            bi[I.rhs_contract[k]] = cc[k];
                        }
                        s += a.data[encode(ai, aSt)] * b.data[encode(bi, bSt)];
                    }
                    out.data[o] = s;
                });
                break;
            }
            case HloOp::Concatenate: {
                auto oSt = rowMajorStrides(I.shape);
                std::vector<int64_t> oc(I.shape.rank());
                int64_t offset = 0;
                for (int opi = 0; opi < (int)I.operands.size(); ++opi) {
                    const Literal& a = vals[I.operands[opi]];
                    auto aSt = rowMajorStrides(a.shape);
                    std::vector<int64_t> ac(a.shape.rank());
                    for (int64_t i = 0; i < a.shape.numel(); ++i) {
                        decode(i, aSt, ac);
                        oc = ac;
                        oc[I.concat_dim] = ac[I.concat_dim] + offset;
                        out.data[encode(oc, oSt)] = a.data[i];
                    }
                    offset += a.shape.dims[I.concat_dim];
                }
                break;
            }
            case HloOp::Slice: {
                const Literal& a = vals[I.operands[0]];
                auto aSt = rowMajorStrides(a.shape);
                auto oSt = rowMajorStrides(I.shape);
                std::vector<int64_t> oc(I.shape.rank()), ac(a.shape.rank());
                for (int64_t o = 0; o < I.shape.numel(); ++o) {
                    decode(o, oSt, oc);
                    for (int64_t d = 0; d < a.shape.rank(); ++d)
                        ac[d] = I.slice_starts[d] + oc[d] * I.slice_strides[d];
                    out.data[o] = a.data[encode(ac, aSt)];
                }
                break;
            }
            case HloOp::Pad: {
                const Literal& a = vals[I.operands[0]];
                auto aSt = rowMajorStrides(a.shape);
                auto oSt = rowMajorStrides(I.shape);
                float fill = (I.operands.size() > 1) ? vals[I.operands[1]].data[0] : I.pad_value;
                for (auto& v : out.data) v = fill;
                std::vector<int64_t> ac(a.shape.rank()), oc(I.shape.rank());
                for (int64_t i = 0; i < a.shape.numel(); ++i) {
                    decode(i, aSt, ac);
                    for (int64_t d = 0; d < a.shape.rank(); ++d)
                        oc[d] = I.pad_low[d] + ac[d] * (I.pad_interior[d] + 1);
                    out.data[encode(oc, oSt)] = a.data[i];
                }
                break;
            }
            case HloOp::Convolution: {
                const Literal& in = vals[I.operands[0]];
                const Literal& w = vals[I.operands[1]];
                auto inSt = rowMajorStrides(in.shape), wSt = rowMajorStrides(w.shape);
                auto oSt = rowMajorStrides(I.shape);
                int ns = (int)I.conv_in_spatial.size();
                int64_t Cin = in.shape.dims[I.conv_in_feat];
                int64_t Cout = I.shape.dims[I.conv_out_feat];
                int64_t cinPerG = Cin / I.conv_groups, coutPerG = Cout / I.conv_groups;
                parallelFor(I.shape.numel(), I.shape.numel() * cinPerG * 9, [&](int64_t o) {
                    std::vector<int64_t> oc(I.shape.rank()), ic(in.shape.rank()), kc(w.shape.rank());
                    decode(o, oSt, oc);
                    int64_t n = oc[I.conv_out_batch], co = oc[I.conv_out_feat];
                    int64_t g = co / coutPerG;
                    float acc = 0.0f;
                    // iterate kernel input-feature + spatial window
                    std::vector<int64_t> kpos(ns);
                    int64_t kWin = 1;
                    for (int s = 0; s < ns; ++s) kWin *= w.shape.dims[I.conv_k_spatial[s]];
                    for (int64_t ci = 0; ci < cinPerG; ++ci) {
                        int64_t inC = g * cinPerG + ci;
                        for (int64_t kw = 0; kw < kWin; ++kw) {
                            int64_t rem = kw; bool inb = true;
                            for (int s = ns - 1; s >= 0; --s) {
                                int64_t ks = w.shape.dims[I.conv_k_spatial[s]];
                                kpos[s] = rem % ks; rem /= ks;
                            }
                            for (int s = 0; s < ns && inb; ++s) {
                                int64_t outp = oc[I.conv_out_spatial[s]];
                                int64_t pos = outp * I.conv_strides[s] - I.conv_pad_low[s]
                                              + kpos[s] * I.conv_rhs_dil[s];
                                if (pos < 0 || pos >= in.shape.dims[I.conv_in_spatial[s]]) { inb = false; break; }
                                ic[I.conv_in_spatial[s]] = pos;
                            }
                            if (!inb) continue;
                            ic[I.conv_in_batch] = n; ic[I.conv_in_feat] = inC;
                            kc[I.conv_k_out] = co; kc[I.conv_k_in] = ci;
                            for (int s = 0; s < ns; ++s) kc[I.conv_k_spatial[s]] = kpos[s];
                            acc += in.data[encode(ic, inSt)] * w.data[encode(kc, wSt)];
                        }
                    }
                    out.data[o] = acc;
                });
                break;
            }
            case HloOp::Gather: {
                // Embedding-style gather: operand[..], indices select along start_map dims.
                const Literal& op0 = vals[I.operands[0]];
                const Literal& idx = vals[I.operands[1]];
                auto opSt = rowMajorStrides(op0.shape);
                auto oSt = rowMajorStrides(I.shape);
                auto idxSt = rowMajorStrides(idx.shape);
                int64_t ivd = I.gather_index_vector_dim;
                int64_t idxVecLen = (ivd < idx.shape.rank()) ? idx.shape.dims[ivd] : 1;
                // batch dims of output = those not in gather_offset_dims
                std::vector<int64_t> batchOut;
                {
                    std::vector<bool> isOff(I.shape.rank(), false);
                    for (int64_t d : I.gather_offset_dims) isOff[d] = true;
                    for (int64_t d = 0; d < I.shape.rank(); ++d) if (!isOff[d]) batchOut.push_back(d);
                }
                std::vector<int64_t> oc(I.shape.rank()), opc(op0.shape.rank()), idc(idx.shape.rank());
                for (int64_t o = 0; o < I.shape.numel(); ++o) {
                    decode(o, oSt, oc);
                    // build index coords: batch part maps to idx (minus the vector dim)
                    int64_t bi = 0;
                    for (int64_t d = 0; d < idx.shape.rank(); ++d) {
                        if (d == ivd) { idc[d] = 0; continue; }
                        idc[d] = oc[batchOut[bi++]];
                    }
                    // start point in operand from start_index_map
                    for (int64_t d = 0; d < op0.shape.rank(); ++d) opc[d] = 0;
                    for (int64_t k = 0; k < idxVecLen; ++k) {
                        if (ivd < idx.shape.rank()) idc[ivd] = k;
                        int64_t start = (int64_t)std::llround(idx.data[encode(idc, idxSt)]);
                        int64_t dim = I.gather_start_map[k];
                        int64_t maxStart = op0.shape.dims[dim] - I.gather_slice_sizes[dim];
                        if (start < 0) start = 0;
                        if (start > maxStart) start = maxStart;
                        opc[dim] += start;
                    }
                    // offset part: gather_offset_dims map to operand dims not collapsed
                    std::vector<bool> isColl(op0.shape.rank(), false);
                    for (int64_t d : I.gather_collapsed) isColl[d] = true;
                    int64_t offi = 0;
                    for (int64_t d = 0; d < op0.shape.rank(); ++d) {
                        if (isColl[d]) continue;
                        opc[d] += oc[I.gather_offset_dims[offi++]];
                    }
                    out.data[o] = op0.data[encode(opc, opSt)];
                }
                break;
            }
            case HloOp::ReduceWindow: {
                const Literal& in = vals[I.operands[0]];
                float init = vals[I.operands[1]].data[0];
                int rank = (int)in.shape.rank();
                auto inSt = rowMajorStrides(in.shape);
                auto oSt = rowMajorStrides(I.shape);
                int64_t wnum = 1;
                for (int64_t w : I.rw_window_dims) wnum *= w;
                for (int64_t o = 0; o < I.shape.numel(); ++o) {
                    std::vector<int64_t> oc(rank), wc(rank), ic(rank);
                    decode(o, oSt, oc);
                    float acc = init;
                    for (int64_t w = 0; w < wnum; ++w) {
                        int64_t rem = w; bool inb = true;
                        for (int d = rank - 1; d >= 0; --d) { wc[d] = rem % I.rw_window_dims[d]; rem /= I.rw_window_dims[d]; }
                        for (int d = 0; d < rank; ++d) {
                            int64_t pos = oc[d] * I.rw_window_strides[d] - I.rw_pad_low[d]
                                          + wc[d] * I.rw_window_dil[d];
                            if (pos < 0 || pos >= in.shape.dims[d]) { inb = false; break; }
                            ic[d] = pos;
                        }
                        if (!inb) continue;  // outside padding contributes the identity
                        float x = in.data[encode(ic, inSt)];
                        if (I.reduce_kind == "sum") acc += x;
                        else if (I.reduce_kind == "max") acc = acc > x ? acc : x;
                        else if (I.reduce_kind == "min") acc = acc < x ? acc : x;
                        else if (I.reduce_kind == "prod") acc *= x;
                        else throw std::runtime_error("bad reduce_window kind");
                    }
                    out.data[o] = acc;
                }
                break;
            }
            case HloOp::Tuple: {
                std::vector<Literal> elems;
                for (int op : I.operands) elems.push_back(vals[op]);
                out = elems.empty() ? Literal{} : elems[0];
                multi[idx] = std::move(elems);
                break;
            }
            case HloOp::GetTupleElement: {
                const auto& src = multi[I.operands[0]];
                if (I.gte_index < 0 || I.gte_index >= (int64_t)src.size())
                    throw std::runtime_error("GetTupleElement index out of range");
                out = src[I.gte_index];
                break;
            }
            case HloOp::While: {
                if (I.subs.size() < 2 || !I.subs[0] || !I.subs[1])
                    throw std::runtime_error("While missing cond/body");
                std::vector<Literal> carried;
                for (int op : I.operands) carried.push_back(vals[op]);
                int64_t guard = 0;
                for (;;) {
                    Literal c = I.subs[0]->evaluate(carried);
                    if (c.data.empty() || c.data[0] == 0.0f) break;
                    carried = I.subs[1]->evaluateMulti(carried);
                    if (++guard > 100000000) throw std::runtime_error("While exceeded iteration guard");
                }
                out = carried.empty() ? Literal{} : carried[0];
                multi[idx] = std::move(carried);
                break;
            }
            case HloOp::DynamicSlice: {
                const Literal& a = vals[I.operands[0]];
                auto aSt = rowMajorStrides(a.shape);
                auto oSt = rowMajorStrides(I.shape);
                int rank = (int)a.shape.rank();
                std::vector<int64_t> start(rank);
                for (int d = 0; d < rank; ++d) {
                    int64_t s = (int64_t)std::llround(vals[I.operands[1 + d]].data[0]);
                    int64_t maxS = a.shape.dims[d] - I.dyn_slice_sizes[d];
                    start[d] = s < 0 ? 0 : (s > maxS ? maxS : s);  // XLA clamps start
                }
                std::vector<int64_t> oc(rank), ac(rank);
                for (int64_t o = 0; o < I.shape.numel(); ++o) {
                    decode(o, oSt, oc);
                    for (int d = 0; d < rank; ++d) ac[d] = start[d] + oc[d];
                    out.data[o] = a.data[encode(ac, aSt)];
                }
                break;
            }
            case HloOp::Reverse: {
                const Literal& a = vals[I.operands[0]];
                auto st = rowMajorStrides(a.shape);
                std::vector<bool> rev(a.shape.rank(), false);
                for (int64_t d : I.dimensions) rev[d] = true;
                std::vector<int64_t> oc(a.shape.rank()), ic(a.shape.rank());
                for (int64_t o = 0; o < I.shape.numel(); ++o) {
                    decode(o, st, oc);
                    for (int64_t d = 0; d < a.shape.rank(); ++d)
                        ic[d] = rev[d] ? a.shape.dims[d] - 1 - oc[d] : oc[d];
                    out.data[o] = a.data[encode(ic, st)];
                }
                break;
            }
            case HloOp::Sort: {
                int n = (int)I.operands.size();
                std::vector<const Literal*> ins;
                for (int op : I.operands) ins.push_back(&vals[op]);
                const Shape& shp = ins[0]->shape;
                int rank = (int)shp.rank();
                int64_t sd = I.sort_dim;
                int64_t L = shp.dims[sd];
                auto st = rowMajorStrides(shp);
                std::vector<Literal> outs(n);
                for (int k = 0; k < n; ++k) outs[k] = *ins[k];
                // iterate over every slice along `sd` (all other coords fixed)
                int64_t outer = shp.numel() / (L == 0 ? 1 : L);
                std::vector<int64_t> base(rank);
                for (int64_t s = 0; s < outer; ++s) {
                    // decode s into coords with sd skipped
                    int64_t rem = s;
                    for (int d = rank - 1; d >= 0; --d) {
                        if (d == sd) { base[d] = 0; continue; }
                        base[d] = rem % shp.dims[d]; rem /= shp.dims[d];
                    }
                    std::vector<int64_t> pos(L);
                    for (int64_t i = 0; i < L; ++i) {
                        auto c = base; c[sd] = i; pos[i] = encode(c, st);
                    }
                    std::vector<int64_t> perm(L);
                    for (int64_t i = 0; i < L; ++i) perm[i] = i;
                    auto precedes = [&](int64_t ia, int64_t ib) {
                        // comparator params are grouped per operand: (a_k, b_k) for each k
                        std::vector<Literal> p;
                        for (int k = 0; k < n; ++k) {
                            p.push_back(Literal::scalar(ins[k]->data[pos[ia]]));
                            p.push_back(Literal::scalar(ins[k]->data[pos[ib]]));
                        }
                        return I.subs[0]->evaluate(p).data[0] != 0.0f;
                    };
                    std::stable_sort(perm.begin(), perm.end(), precedes);
                    for (int k = 0; k < n; ++k)
                        for (int64_t i = 0; i < L; ++i)
                            outs[k].data[pos[i]] = ins[k]->data[pos[perm[i]]];
                }
                out = outs.empty() ? Literal{} : outs[0];
                multi[idx] = std::move(outs);
                break;
            }
            case HloOp::ReduceGeneral: {
                int n = (int)I.operands.size() / 2;       // n inputs + n inits
                std::vector<const Literal*> ins, inits;
                for (int k = 0; k < n; ++k) ins.push_back(&vals[I.operands[k]]);
                for (int k = 0; k < n; ++k) inits.push_back(&vals[I.operands[n + k]]);
                const Shape& ishape = ins[0]->shape;
                auto inSt = rowMajorStrides(ishape);
                auto outSt = rowMajorStrides(I.shape);
                std::vector<bool> isRed(ishape.rank(), false);
                for (int64_t d : I.dimensions) isRed[d] = true;
                std::vector<int> keep;
                for (int d = 0; d < (int)ishape.rank(); ++d) if (!isRed[d]) keep.push_back(d);
                // n accumulators, each I.shape-sized, seeded with the init scalars
                std::vector<Literal> acc(n);
                for (int k = 0; k < n; ++k) {
                    acc[k].shape = I.shape;
                    acc[k].data.assign(I.shape.numel(), inits[k]->data[0]);
                }
                std::vector<int64_t> ic(ishape.rank()), oc(keep.size());
                for (int64_t i = 0; i < ishape.numel(); ++i) {
                    decode(i, inSt, ic);
                    for (size_t k = 0; k < keep.size(); ++k) oc[k] = ic[keep[k]];
                    int64_t o = encode(oc, outSt);
                    std::vector<Literal> params;
                    for (int k = 0; k < n; ++k) params.push_back(Literal::scalar(acc[k].data[o]));
                    for (int k = 0; k < n; ++k) params.push_back(Literal::scalar(ins[k]->data[i]));
                    std::vector<Literal> res = I.subs[0]->evaluateMulti(params);
                    for (int k = 0; k < n; ++k) acc[k].data[o] = res[k].data[0];
                }
                out = acc.empty() ? Literal{} : acc[0];
                multi[idx] = std::move(acc);
                break;
            }
            case HloOp::Scatter: {
                const Literal& operand = vals[I.operands[0]];
                const Literal& sidx = vals[I.operands[1]];
                const Literal& updates = vals[I.operands[2]];
                out = operand;  // copy then combine updates in
                int ro = (int)operand.shape.rank(), ru = (int)updates.shape.rank(),
                    ri = (int)sidx.shape.rank();
                auto oSt = rowMajorStrides(operand.shape), uSt = rowMajorStrides(updates.shape),
                     iSt = rowMajorStrides(sidx.shape);
                std::vector<bool> isWin(ru, false);
                for (int64_t d : I.scatter_update_window_dims) isWin[d] = true;
                std::vector<int64_t> updScatter;          // update dims that index scatter rows
                for (int d = 0; d < ru; ++d) if (!isWin[d]) updScatter.push_back(d);
                std::vector<bool> isIns(ro, false);
                for (int64_t d : I.scatter_inserted_window_dims) isIns[d] = true;
                int ncomp = (int)I.scatter_dims_to_operand.size();
                std::vector<int64_t> uc(ru), ic(ri, 0), oc(ro);
                for (int64_t u = 0; u < updates.shape.numel(); ++u) {
                    decode(u, uSt, uc);
                    int b = 0;
                    for (int d = 0; d < ri; ++d) {
                        if (d == I.scatter_index_vector_dim) { ic[d] = 0; continue; }
                        ic[d] = uc[updScatter[b++]];
                    }
                    std::vector<int64_t> start(ro, 0);
                    for (int k = 0; k < ncomp; ++k) {
                        if (I.scatter_index_vector_dim < ri) ic[I.scatter_index_vector_dim] = k;
                        start[I.scatter_dims_to_operand[k]] = (int64_t)std::llround(sidx.data[encode(ic, iSt)]);
                    }
                    int j = 0;
                    for (int d = 0; d < ro; ++d) {
                        oc[d] = start[d] + (isIns[d] ? 0 : uc[I.scatter_update_window_dims[j++]]);
                        if (oc[d] < 0) oc[d] = 0;
                        if (oc[d] >= operand.shape.dims[d]) oc[d] = operand.shape.dims[d] - 1;
                    }
                    int64_t oi = encode(oc, oSt);
                    out.data[oi] = I.subs[0]->evaluate(
                        {Literal::scalar(out.data[oi]), Literal::scalar(updates.data[u])}).data[0];
                }
                break;
            }
            case HloOp::DynamicUpdateSlice: {
                const Literal& a = vals[I.operands[0]];
                const Literal& u = vals[I.operands[1]];
                out = a;  // copy then overwrite the window
                auto aSt = rowMajorStrides(a.shape);
                auto uSt = rowMajorStrides(u.shape);
                int rank = (int)a.shape.rank();
                std::vector<int64_t> start(rank);
                for (int d = 0; d < rank; ++d) {
                    int64_t s = (int64_t)std::llround(vals[I.operands[2 + d]].data[0]);
                    int64_t maxS = a.shape.dims[d] - u.shape.dims[d];
                    start[d] = s < 0 ? 0 : (s > maxS ? maxS : s);
                }
                std::vector<int64_t> uc(rank), ac(rank);
                for (int64_t i = 0; i < u.shape.numel(); ++i) {
                    decode(i, uSt, uc);
                    for (int d = 0; d < rank; ++d) ac[d] = start[d] + uc[d];
                    out.data[encode(ac, aSt)] = u.data[i];
                }
                break;
            }
        }
        vals[idx] = std::move(out);

        // Release operands whose last use was this instruction (frees the buffer
        // and its tuple slot). Safe because all ops copy their inputs.
        for (int op : I.operands) {
            if (op >= 0 && op < (int)instrs_.size() && last_use[op] == (int64_t)idx) {
                std::vector<float>().swap(vals[op].data);
                if (!multi[op].empty()) std::vector<Literal>().swap(multi[op]);
            }
        }
    }

    if (root_ < 0 || root_ >= (int)vals.size()) throw std::runtime_error("HloModule has no root");
    if (!multi[root_].empty()) return multi[root_];
    return {vals[root_]};
}

Literal HloModule::evaluate(const std::vector<Literal>& params) const {
    auto r = evaluateMulti(params);
    if (r.empty()) throw std::runtime_error("HloModule produced no result");
    return std::move(r[0]);
}

bool HloModule::validate(std::string& err) const {
    if (instrs_.empty()) { err = "empty module"; return false; }
    for (size_t i = 0; i < instrs_.size(); ++i) {
        const HloInstruction& I = instrs_[i];
        for (int op : I.operands) {
            if (op < 0 || op >= (int)i) {     // operands must be earlier (acyclic, ordered)
                err = "instruction " + std::to_string(i) + " references invalid operand " +
                      std::to_string(op);
                return false;
            }
        }
        if (I.op == HloOp::Parameter && I.param_index < 0) {
            err = "instruction " + std::to_string(i) + " has negative parameter index";
            return false;
        }
        for (const auto& s : I.subs) {
            std::string se;
            if (s && !s->validate(se)) { err = "sub-computation: " + se; return false; }
        }
    }
    if (root_ < 0 || root_ >= (int)instrs_.size()) { err = "root index out of range"; return false; }
    return true;
}

// ── module + builder ────────────────────────────────────────────────────────
int HloModule::add(HloInstruction inst) {
    instrs_.push_back(std::move(inst));
    int idx = static_cast<int>(instrs_.size()) - 1;
    root_ = idx; // last added is the default root
    return idx;
}
int HloModule::parameter(int index, Shape shape) {
    HloInstruction i; i.op = HloOp::Parameter; i.shape = std::move(shape); i.param_index = index;
    return add(std::move(i));
}
int HloModule::constant(Literal lit) {
    HloInstruction i; i.op = HloOp::Constant; i.shape = lit.shape; i.literal = std::move(lit);
    return add(std::move(i));
}
int HloModule::binary(HloOp op, int lhs, int rhs) {
    HloInstruction i; i.op = op; i.shape = instrs_[lhs].shape; i.operands = {lhs, rhs};
    return add(std::move(i));
}
int HloModule::unary(HloOp op, int x) {
    HloInstruction i; i.op = op; i.shape = instrs_[x].shape; i.operands = {x};
    return add(std::move(i));
}
int HloModule::broadcast(int x, Shape out, std::vector<int64_t> bdims) {
    HloInstruction i; i.op = HloOp::Broadcast; i.shape = std::move(out); i.operands = {x}; i.dimensions = std::move(bdims);
    return add(std::move(i));
}
int HloModule::reshape(int x, Shape out) {
    HloInstruction i; i.op = HloOp::Reshape; i.shape = std::move(out); i.operands = {x};
    return add(std::move(i));
}
int HloModule::transpose(int x, std::vector<int64_t> perm) {
    HloInstruction i; i.op = HloOp::Transpose; i.operands = {x}; i.dimensions = perm;
    Shape s; for (int64_t p : perm) s.dims.push_back(instrs_[x].shape.dims[p]);
    i.shape = s;
    return add(std::move(i));
}
int HloModule::dot(int lhs, int rhs) {
    HloInstruction i; i.op = HloOp::Dot; i.operands = {lhs, rhs};
    i.shape = Shape{{instrs_[lhs].shape.dims[0], instrs_[rhs].shape.dims[1]}};
    return add(std::move(i));
}
int HloModule::reduce(int x, std::vector<int64_t> dims, const std::string& kind, float init) {
    HloInstruction i; i.op = HloOp::Reduce; i.operands = {x}; i.dimensions = dims;
    i.reduce_kind = kind; i.init_value = init;
    std::vector<bool> red(instrs_[x].shape.dims.size(), false);
    for (int64_t d : dims) red[d] = true;
    Shape s;
    for (int d = 0; d < (int)instrs_[x].shape.dims.size(); ++d) if (!red[d]) s.dims.push_back(instrs_[x].shape.dims[d]);
    i.shape = s;
    return add(std::move(i));
}
int HloModule::compare(int lhs, int rhs, const std::string& dir) {
    HloInstruction i; i.op = HloOp::Compare; i.shape = instrs_[lhs].shape; i.operands = {lhs, rhs}; i.compare_dir = dir;
    return add(std::move(i));
}
int HloModule::select(int pred, int on_true, int on_false) {
    HloInstruction i; i.op = HloOp::Select; i.shape = instrs_[on_true].shape; i.operands = {pred, on_true, on_false};
    return add(std::move(i));
}
int HloModule::dotGeneral(int lhs, int rhs, std::vector<int64_t> lb, std::vector<int64_t> rb,
                          std::vector<int64_t> lc, std::vector<int64_t> rc, Shape out) {
    HloInstruction i; i.op = HloOp::DotGeneral; i.operands = {lhs, rhs}; i.shape = std::move(out);
    i.lhs_batch = std::move(lb); i.rhs_batch = std::move(rb);
    i.lhs_contract = std::move(lc); i.rhs_contract = std::move(rc);
    return add(std::move(i));
}
int HloModule::concatenate(const std::vector<int>& xs, int64_t dim, Shape out) {
    HloInstruction i; i.op = HloOp::Concatenate; i.operands = xs; i.concat_dim = dim; i.shape = std::move(out);
    return add(std::move(i));
}
int HloModule::slice(int x, std::vector<int64_t> starts, std::vector<int64_t> limits,
                     std::vector<int64_t> strides, Shape out) {
    HloInstruction i; i.op = HloOp::Slice; i.operands = {x}; i.shape = std::move(out);
    i.slice_starts = std::move(starts); i.slice_limits = std::move(limits); i.slice_strides = std::move(strides);
    return add(std::move(i));
}
int HloModule::pad(int x, int pad_val, std::vector<int64_t> low, std::vector<int64_t> high,
                   std::vector<int64_t> interior, Shape out) {
    HloInstruction i; i.op = HloOp::Pad; i.operands = {x, pad_val}; i.shape = std::move(out);
    i.pad_low = std::move(low); i.pad_high = std::move(high); i.pad_interior = std::move(interior);
    return add(std::move(i));
}
int HloModule::convolution(int lhs, int rhs, Shape out) {
    HloInstruction i; i.op = HloOp::Convolution; i.operands = {lhs, rhs}; i.shape = std::move(out);
    return add(std::move(i));
}
int HloModule::gather(int operand, int indices, Shape out) {
    HloInstruction i; i.op = HloOp::Gather; i.operands = {operand, indices}; i.shape = std::move(out);
    return add(std::move(i));
}
int HloModule::reduceWindow(int x, int init, const std::string& kind, Shape out) {
    HloInstruction i; i.op = HloOp::ReduceWindow; i.operands = {x, init}; i.shape = std::move(out);
    i.reduce_kind = kind;
    return add(std::move(i));
}
int HloModule::whileOp(const std::vector<int>& inits, std::shared_ptr<HloModule> cond,
                       std::shared_ptr<HloModule> body, Shape primary) {
    HloInstruction i; i.op = HloOp::While; i.operands = inits; i.shape = std::move(primary);
    i.subs = {std::move(cond), std::move(body)};
    return add(std::move(i));
}
int HloModule::tuple(const std::vector<int>& elems) {
    HloInstruction i; i.op = HloOp::Tuple; i.operands = elems;
    i.shape = elems.empty() ? Shape{} : instrs_[elems[0]].shape;
    return add(std::move(i));
}
int HloModule::getTupleElement(int src, int index, Shape out) {
    HloInstruction i; i.op = HloOp::GetTupleElement; i.operands = {src}; i.gte_index = index;
    i.shape = std::move(out);
    return add(std::move(i));
}
int HloModule::dynamicSlice(int operand, const std::vector<int>& starts,
                            std::vector<int64_t> sizes, Shape out) {
    HloInstruction i; i.op = HloOp::DynamicSlice; i.operands = {operand};
    for (int s : starts) i.operands.push_back(s);
    i.dyn_slice_sizes = std::move(sizes); i.shape = std::move(out);
    return add(std::move(i));
}
int HloModule::dynamicUpdateSlice(int operand, int update, const std::vector<int>& starts, Shape out) {
    HloInstruction i; i.op = HloOp::DynamicUpdateSlice; i.operands = {operand, update};
    for (int s : starts) i.operands.push_back(s);
    i.shape = std::move(out);
    return add(std::move(i));
}
int HloModule::reverse(int x, std::vector<int64_t> dims) {
    HloInstruction i; i.op = HloOp::Reverse; i.operands = {x}; i.shape = instrs_[x].shape;
    i.dimensions = std::move(dims);
    return add(std::move(i));
}
int HloModule::iota(Shape out, int64_t dim) {
    HloInstruction i; i.op = HloOp::Iota; i.shape = std::move(out); i.iota_dim = dim;
    return add(std::move(i));
}
int HloModule::sort(const std::vector<int>& operands, int64_t dim, std::shared_ptr<HloModule> cmp) {
    HloInstruction i; i.op = HloOp::Sort; i.operands = operands; i.sort_dim = dim;
    i.subs = {std::move(cmp)};
    i.shape = operands.empty() ? Shape{} : instrs_[operands[0]].shape;
    return add(std::move(i));
}
int HloModule::reduceGeneral(const std::vector<int>& operands, std::vector<int64_t> dims,
                             std::shared_ptr<HloModule> body, Shape primary) {
    HloInstruction i; i.op = HloOp::ReduceGeneral; i.operands = operands;
    i.dimensions = std::move(dims); i.subs = {std::move(body)}; i.shape = std::move(primary);
    return add(std::move(i));
}
int HloModule::scatter(int operand, int indices, int updates, std::shared_ptr<HloModule> combiner) {
    HloInstruction i; i.op = HloOp::Scatter; i.operands = {operand, indices, updates};
    i.subs = {std::move(combiner)}; i.shape = instrs_[operand].shape;
    return add(std::move(i));
}

} // namespace xla
} // namespace vgre
