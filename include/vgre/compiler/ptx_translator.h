#ifndef VGRE_COMPILER_PTX_TRANSLATOR_H
#define VGRE_COMPILER_PTX_TRANSLATOR_H

#include <string>

namespace vgre {
namespace compiler {

// Scans kernel source for inline PTX assembly blocks (asm("...") or
// __asm("...")) and replaces each block with equivalent C++ code.
// Unsupported instructions are replaced with a no-op comment so that
// compilation succeeds even for kernels with complex PTX.
class PTXTranslator {
public:
    // Replace inline PTX in-place. Returns the transformed source.
    static std::string translate(const std::string& source);

private:
    static std::string translateBlock(const std::string& ptxBody,
                                      const std::string& constraints,
                                      const std::string& clobbers);
    static std::string translateInstruction(const std::string& instr,
                                             const std::string& operands);
};

} // namespace compiler
} // namespace vgre

#endif // VGRE_COMPILER_PTX_TRANSLATOR_H
