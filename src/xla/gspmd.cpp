// GSPMD-style automatic partitioner — see include/vgre/xla/gspmd.h.

#include "vgre/xla/gspmd.h"

#include <stdexcept>

#include "vgre/xla/blas_gemm.h"
#include "vgre/xla/parallel.h"  // Communicator (all-reduce / all-gather)

namespace vgre {
namespace xla {

namespace {

// Even split boundaries of `total` into `ranks` parts (earlier parts get +1).
std::vector<std::pair<int64_t, int64_t>> splitRange(int64_t total, int ranks) {
    std::vector<std::pair<int64_t, int64_t>> r;
    int64_t base = total / ranks, rem = total % ranks, b = 0;
    for (int i = 0; i < ranks; ++i) { int64_t sz = base + (i < rem ? 1 : 0); r.emplace_back(b, b + sz); b += sz; }
    return r;
}

// Split a 2-D Literal along `dim` (0 = rows, 1 = cols) into `ranks` shards.
std::vector<Literal> split2D(const Literal& t, int dim, int ranks) {
    const int64_t R = t.shape.dims[0], C = t.shape.dims[1];
    std::vector<Literal> out;
    if (dim == 0) {
        for (auto [a, b] : splitRange(R, ranks)) {
            Literal s; s.shape.dims = {b - a, C};
            s.data.assign(t.data.begin() + (size_t)(a * C), t.data.begin() + (size_t)(b * C));
            out.push_back(std::move(s));
        }
    } else {
        for (auto [a, b] : splitRange(C, ranks)) {
            int64_t cw = b - a;
            Literal s; s.shape.dims = {R, cw}; s.data.resize((size_t)(R * cw));
            for (int64_t i = 0; i < R; ++i)
                for (int64_t j = 0; j < cw; ++j) s.data[(size_t)(i * cw + j)] = t.data[(size_t)(i * C + a + j)];
            out.push_back(std::move(s));
        }
    }
    return out;
}

// Evaluate a single elementwise instruction on rank-local operands by replaying
// it in a one-op sub-module (reuses the engine's op semantics). Output shape ==
// operand shape, which holds for unary/binary elementwise ops.
Literal evalLocal(const HloInstruction& I, const std::vector<Literal>& opv) {
    HloModule m;
    std::vector<int> ps;
    for (size_t k = 0; k < opv.size(); ++k) ps.push_back(m.parameter((int)k, opv[k].shape));
    HloInstruction J = I;
    J.operands = ps;
    J.shape = opv[0].shape;
    int idx = m.add(J);
    m.setRoot(idx);
    return m.evaluate(opv);
}

Literal localDot(const Literal& A, const Literal& B) {  // [M,K]·[K,N]
    int64_t M = A.shape.dims[0], K = A.shape.dims[1], N = B.shape.dims[1];
    Literal y; y.shape.dims = {M, N}; y.data.resize((size_t)(M * N));
    gemm_f32(false, false, M, N, K, A.data.data(), B.data.data(), y.data.data(), 1);
    return y;
}

bool isElementwise(HloOp op) {
    switch (op) {
        case HloOp::Add: case HloOp::Subtract: case HloOp::Multiply: case HloOp::Divide:
        case HloOp::Maximum: case HloOp::Minimum: case HloOp::Power:
        case HloOp::Negate: case HloOp::Exp: case HloOp::Log: case HloOp::Tanh: case HloOp::Abs:
        case HloOp::Rsqrt: case HloOp::Sqrt: case HloOp::Sign: case HloOp::Floor: case HloOp::Ceil:
        case HloOp::Logistic: case HloOp::Erf: case HloOp::Erfc: case HloOp::Cos: case HloOp::Sin:
            return true;
        default: return false;
    }
}

}  // namespace

Literal gspmdExecute(const HloModule& m, const std::vector<Literal>& params,
                     const std::map<int, Sharding>& paramSharding, int ranks) {
    const int n = (int)m.size();
    std::vector<std::vector<Literal>> vals((size_t)n, std::vector<Literal>((size_t)ranks));
    std::vector<Sharding> shard((size_t)n, Sharding::replicate());

    auto gatherToReplicated = [&](int idx) {  // make a sharded value replicated on all ranks
        if (shard[idx].replicated()) return;
        Literal full;
        if (shard[idx].dim == 1) {
            full = Communicator::allGatherConcatLastAxis(vals[idx]);
        } else {  // dim 0: concat rows
            int64_t C = vals[idx][0].shape.dims[1], Rtot = 0;
            for (auto& s : vals[idx]) Rtot += s.shape.dims[0];
            full.shape.dims = {Rtot, C}; full.data.reserve((size_t)(Rtot * C));
            for (auto& s : vals[idx]) full.data.insert(full.data.end(), s.data.begin(), s.data.end());
        }
        for (int r = 0; r < ranks; ++r) vals[idx][(size_t)r] = full;
        shard[idx] = Sharding::replicate();
    };

    for (int idx = 0; idx < n; ++idx) {
        const HloInstruction& I = m.instr(idx);

        if (I.op == HloOp::Parameter) {
            auto it = paramSharding.find(I.param_index);
            Sharding s = (it != paramSharding.end()) ? it->second : Sharding::replicate();
            const Literal& p = params[(size_t)I.param_index];
            if (s.replicated()) for (int r = 0; r < ranks; ++r) vals[idx][(size_t)r] = p;
            else { auto sh = split2D(p, s.dim, ranks); for (int r = 0; r < ranks; ++r) vals[idx][(size_t)r] = sh[(size_t)r]; }
            shard[idx] = s;

        } else if (I.op == HloOp::Constant) {
            for (int r = 0; r < ranks; ++r) vals[idx][(size_t)r] = I.literal;
            shard[idx] = Sharding::replicate();

        } else if (I.op == HloOp::Dot) {
            int L = I.operands[0], R = I.operands[1];
            Sharding ls = shard[L], rs = shard[R];
            if (ls.replicated() && rs.dim == 1) {            // column-parallel
                for (int r = 0; r < ranks; ++r) vals[idx][(size_t)r] = localDot(vals[L][(size_t)r], vals[R][(size_t)r]);
                shard[idx] = Sharding::on(1);
            } else if (ls.dim == 1 && rs.dim == 0) {         // row-parallel → all-reduce
                std::vector<Literal> partials((size_t)ranks);
                for (int r = 0; r < ranks; ++r) partials[(size_t)r] = localDot(vals[L][(size_t)r], vals[R][(size_t)r]);
                Literal reduced = Communicator::allReduceSum(partials);
                for (int r = 0; r < ranks; ++r) vals[idx][(size_t)r] = reduced;
                shard[idx] = Sharding::replicate();
            } else {                                         // fall back: replicate operands, plain dot
                gatherToReplicated(L); gatherToReplicated(R);
                Literal y = localDot(vals[L][0], vals[R][0]);
                for (int r = 0; r < ranks; ++r) vals[idx][(size_t)r] = y;
                shard[idx] = Sharding::replicate();
            }

        } else if (isElementwise(I.op)) {
            // Output sharding = the (consistent) operand sharding; replicated
            // operands are sliced to match a sharded output.
            Sharding out = Sharding::replicate();
            for (int op : I.operands) if (!shard[op].replicated()) { out = shard[op]; break; }
            for (int op : I.operands)
                if (!shard[op].replicated() && shard[op].dim != out.dim)
                    throw std::runtime_error("gspmd: incompatible operand shardings");
            for (int r = 0; r < ranks; ++r) {
                std::vector<Literal> opv;
                for (int op : I.operands) {
                    if (shard[op].replicated() && !out.replicated())
                        opv.push_back(split2D(vals[op][(size_t)r], out.dim, ranks)[(size_t)r]);
                    else
                        opv.push_back(vals[op][(size_t)r]);
                }
                vals[idx][(size_t)r] = evalLocal(I, opv);
            }
            shard[idx] = out;

        } else {
            // Unsupported op under sharding: gather operands to replicated, run locally.
            std::vector<Literal> opv;
            for (int op : I.operands) { gatherToReplicated(op); opv.push_back(vals[op][0]); }
            Literal y = evalLocal(I, opv);
            for (int r = 0; r < ranks; ++r) vals[idx][(size_t)r] = y;
            shard[idx] = Sharding::replicate();
        }
    }

    int root = m.root();
    gatherToReplicated(root);
    return vals[(size_t)root][0];
}

}  // namespace xla
}  // namespace vgre
