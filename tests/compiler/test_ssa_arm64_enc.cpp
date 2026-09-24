// Validates the Tier-2 AArch64 native emitter's instruction ENCODINGS against the exact
// bytes llvm-mc assembles for each mnemonic (baked into arm64EncSelfTest). The AArch64
// encoder is pure, architecture-independent byte generation, so this runs and passes on
// ANY host — including the x86-64 dev/CI machine — giving local confidence in the ARM
// codegen's encoding layer. (ARM EXECUTION is validated separately by the macos-arm64
// CI job running the SsaIr differential test with VGRE_SSA_ARM_NATIVE=1.)
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/frontend/ssa_ir.h"

#include <cstdio>
#include <string>

int main() {
    std::string err;
    bool ok = vgre::compiler::frontend::arm64EncSelfTest(err);
    if (!ok) {
        std::printf("FAILED: AArch64 encoder self-test: %s\n", err.c_str());
        return 1;
    }
    std::printf("PASS: AArch64 emitter encodings match llvm-mc (encoding layer verified on this host)\n");
    return 0;
}
