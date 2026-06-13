// VGRE XLA backend — HLO interpreter (impl).  See include/vgre/xla/hlo.h.

#include "vgre/xla/hlo.h"

#include <cmath>
#include <stdexcept>

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
        default: throw std::runtime_error("not a unary op");
    }
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

Literal HloModule::evaluate(const std::vector<Literal>& params) const {
    std::vector<Literal> vals(instrs_.size());

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
            case HloOp::Maximum: case HloOp::Minimum: case HloOp::Power: {
                const Literal& a = vals[I.operands[0]];
                const Literal& b = vals[I.operands[1]];
                if (!(a.shape == b.shape)) throw std::runtime_error("binary op shape mismatch");
                for (size_t i = 0; i < a.data.size(); ++i) out.data[i] = applyBinary(I.op, a.data[i], b.data[i]);
                break;
            }
            case HloOp::Negate: case HloOp::Exp: case HloOp::Log: case HloOp::Tanh:
            case HloOp::Abs: case HloOp::Rsqrt: case HloOp::Sqrt: case HloOp::Sign:
            case HloOp::Floor: case HloOp::Ceil: case HloOp::Logistic: {
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
                for (int64_t i = 0; i < M; ++i)
                    for (int64_t j = 0; j < N; ++j) {
                        float s = 0;
                        for (int64_t k = 0; k < K; ++k) s += a.data[i * K + k] * b.data[k * N + j];
                        out.data[i * N + j] = s;
                    }
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
                std::vector<int64_t> oc(I.shape.rank()), ai(a.shape.rank()), bi(b.shape.rank());
                for (int64_t o = 0; o < I.shape.numel(); ++o) {
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
                }
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
                std::vector<int64_t> oc(I.shape.rank()), ic(in.shape.rank()), kc(w.shape.rank());
                for (int64_t o = 0; o < I.shape.numel(); ++o) {
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
                }
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
        }
        vals[idx] = std::move(out);
    }

    if (root_ < 0 || root_ >= (int)vals.size()) throw std::runtime_error("HloModule has no root");
    return vals[root_];
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

} // namespace xla
} // namespace vgre
