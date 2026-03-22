#ifndef VGRE_COMPILER_CLANG_KERNEL_PARSER_H
#define VGRE_COMPILER_CLANG_KERNEL_PARSER_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include "vgre/compiler/kernel_parser.h"
#include <unordered_map>
#include <mutex>

namespace vgre {
namespace compiler {

/**
 * @brief Authoritative kernel parser using Clang's JSON AST.
 *
 * This implementation replaces the regex-based KernelParser with a robust
 * Clang-driven analysis capable of handling complex C++ templates and structs.
 */
class ClangKernelParser : public KernelParser {
public:
    ClangKernelParser();
    virtual ~ClangKernelParser();

    /**
     * @brief Parse kernel metadata using Clang.
     * @param name The kernel name to find.
     * @param source The full CUDA source code.
     * @param outIR The resulting KernelIR object.
     * @return VGREResult::SUCCESS on success.
     */
    VGREResult parse(const std::string& name,
                     const std::string& source,
                     KernelIR& outIR) override;

private:
    std::string runClangAstDump(const std::string& source);
    
    std::unordered_map<std::string, KernelIR> cache_;
    std::mutex cacheMutex_;
};

} // namespace compiler
} // namespace vgre

#endif // VGRE_COMPILER_CLANG_KERNEL_PARSER_H
