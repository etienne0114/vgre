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
            case HloOp::Abs: case HloOp::Rsqrt: {
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
                for (size_t i = 0; i < p.data.size(); ++i) out.data[i] = (p.data[i] != 0.0f) ? t.data[i] : f.data[i];
                break;
            }
            case HloOp::Broadcast: {
                const Literal& a = vals[I.operands[0]];
                auto outSt = rowMajorStrides(I.shape);
                auto inSt = rowMajorStrides(a.shape);
                std::vector<int64_t> oc(I.shape.dims.size()), ic(a.shape.dims.size());
                for (int64_t i = 0; i < out.shape.numel(); ++i) {
                    decode(i, outSt, oc);
                    for (size_t k = 0; k < I.dimensions.size(); ++k) ic[k] = oc[I.dimensions[k]];
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

} // namespace xla
} // namespace vgre
