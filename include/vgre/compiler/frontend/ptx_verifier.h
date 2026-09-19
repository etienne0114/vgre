// A structural verifier for the PTX emitted by VGRE's from-scratch code generator
// (Track Z). It is a self-check: the codegen should only ever produce valid PTX,
// so a verification failure means a codegen bug — the front-end reports it as an
// internal error rather than handing malformed PTX to a backend.
//
// It checks the classes of defect that codegen bugs actually produce:
//   * a branch to a label that is never defined,
//   * a virtual register used outside its declared `.reg …<N>` range,
//   * a malformed instruction operand (an empty operand between commas — the shape
//     the old scalar-__shared__ `++` bug emitted as `add.s32 , , 1`).
// It is deliberately lightweight (it does not type-check operands); the interpreter
// and the compiled tier remain the semantic authority.
#ifndef VGRE_COMPILER_FRONTEND_PTX_VERIFIER_H
#define VGRE_COMPILER_FRONTEND_PTX_VERIFIER_H

#include <string>

namespace vgre {
namespace compiler {
namespace frontend {

struct PtxVerifyResult {
    bool ok = true;
    std::string error;   // human-readable reason (with the offending PTX line) on failure
};

// Verify the structural well-formedness of `ptx`. Returns ok, or the first defect.
PtxVerifyResult verifyPtx(const std::string& ptx);

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_FRONTEND_PTX_VERIFIER_H
