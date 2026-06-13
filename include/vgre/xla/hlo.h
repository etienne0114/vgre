// VGRE  execution engine.  docs/missingFeatures.md §4.2.
//
// The compute core of an XLA/PJRT backend: an explicit HLO computation graph and
// an interpreter that evaluates it. JAX / TensorFlow / PyTorch-XLA all lower to
// HLO (StableHLO), so this is the engine they ultimately dispatch into. This
// header defines the IR + evaluator; the PJRT C ABI shim and StableHLO→HLO
// translation are layered on top (subsequent steps).
//
// Tensors are row-major dense float32. Semantics follow XLA HLO exactly: ops are
// explicit (no implicit broadcasting — use the Broadcast op), matching what the
// frameworks emit.
#ifndef VGRE_XLA_HLO_H
#define VGRE_XLA_HLO_H

#include <cstdint>
#include <string>
#include <vector>

namespace vgre {
namespace xla {

struct Shape {
    std::vector<int64_t> dims;
    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : dims) n *= d;
        return dims.empty() ? 1 : n;
    }
    int64_t rank() const { return static_cast<int64_t>(dims.size()); }
    bool operator==(const Shape& o) const { return dims == o.dims; }
};

// Dense row-major float32 tensor.
struct Literal {
    Shape shape;
    std::vector<float> data;
    static Literal scalar(float v) { return Literal{Shape{{}}, {v}}; }
    static Literal r1(std::vector<float> v) { Shape s{{(int64_t)v.size()}}; return Literal{s, std::move(v)}; }
    static Literal make(Shape s, std::vector<float> d) { return Literal{std::move(s), std::move(d)}; }
};

enum class HloOp {
    Parameter, Constant, Iota,
    Add, Subtract, Multiply, Divide, Maximum, Minimum, Power,
    Negate, Exp, Log, Tanh, Abs, Rsqrt,
    Compare, Select,
    Broadcast, Reshape, Transpose,
    Dot, Reduce,
};

struct HloInstruction {
    HloOp op;
    Shape shape;                       // output shape
    std::vector<int> operands;         // indices into HloModule instruction list

    // op-specific attributes
    int         param_index = 0;       // Parameter
    Literal     literal;               // Constant
    int64_t     iota_dim = 0;          // Iota
    std::vector<int64_t> dimensions;   // Broadcast(broadcast_dims) / Reduce(dims) / Transpose(perm)
    std::string compare_dir;           // Compare: "GT","GE","LT","LE","EQ","NE"
    std::string reduce_kind;           // Reduce: "sum","max","min","prod"
    float       init_value = 0.0f;     // Reduce init
};

class HloModule {
public:
    // Append an instruction; returns its index. Operands must already be added.
    int add(HloInstruction inst);
    void setRoot(int idx) { root_ = idx; }
    int root() const { return root_; }
    size_t size() const { return instrs_.size(); }
    const HloInstruction& instr(int i) const { return instrs_[i]; }

    // Evaluate the module given parameter literals (indexed by Parameter.param_index).
    // Returns the root instruction's literal.
    Literal evaluate(const std::vector<Literal>& params) const;

    // ── builder conveniences (return the new instruction index) ──
    int parameter(int index, Shape shape);
    int constant(Literal lit);
    int binary(HloOp op, int lhs, int rhs);                 // shape from lhs
    int unary(HloOp op, int x);
    int broadcast(int x, Shape out, std::vector<int64_t> bdims);
    int reshape(int x, Shape out);
    int transpose(int x, std::vector<int64_t> perm);
    int dot(int lhs, int rhs);                              // 2D matmul [M,K]·[K,N]
    int reduce(int x, std::vector<int64_t> dims, const std::string& kind, float init);
    int compare(int lhs, int rhs, const std::string& dir);
    int select(int pred, int on_true, int on_false);

private:
    std::vector<HloInstruction> instrs_;
    int root_ = -1;
};

} // namespace xla
} // namespace vgre

#endif // VGRE_XLA_HLO_H
