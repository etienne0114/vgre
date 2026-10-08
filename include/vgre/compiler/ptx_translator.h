#ifndef VGRE_COMPILER_PTX_TRANSLATOR_H
#define VGRE_COMPILER_PTX_TRANSLATOR_H

#include <string>

namespace vgre {
namespace compiler {

// Scans kernel source for inline PTX assembly blocks (asm("...") or
// __asm("...")) and replaces each block with equivalent C++ code.
// Operands are bound from GCC-style output/input constraints. Unsupported
// instructions and predicated PTX fail translation instead of becoming no-ops.
class PTXTranslator {
public:
    // Replace inline PTX in-place. Returns the transformed source.
    static std::string translate(const std::string& source);

    // Status-returning alternative for callers crossing shared-library
    // boundaries. On failure, `translated` is empty and `error` explains why.
    static bool tryTranslate(const std::string& source,
                             std::string& translated,
                             std::string& error);

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
