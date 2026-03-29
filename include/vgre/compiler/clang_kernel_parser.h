#ifndef VGRE_COMPILER_CLANG_KERNEL_PARSER_H
#define VGRE_COMPILER_CLANG_KERNEL_PARSER_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include "vgre/compiler/kernel_parser.h"
#include <unordered_map>
#include <mutex>
#include <vector>

// Forward declarations
namespace llvm {
namespace json {
class Object;
class Array;
}
}

namespace vgre {
namespace compiler {

/**
 * @brief Memory access pattern information
 */
struct MemoryAccessPattern {
    enum class Type {
        SEQUENTIAL,     // Sequential access (coalesced)
        STRIDED,        // Strided access (stride > 1)
        RANDOM,         // Random access
        BROADCAST,      // Same address accessed by multiple threads
        UNKNOWN
    };
    
    Type type = Type::UNKNOWN;
    size_t stride = 0;          // Stride for strided access
    bool isCoalesced = false;   // Whether access is coalesced
    size_t accessSize = 0;      // Size of each access in bytes
    uint64_t estimatedAccesses = 0; // Estimated number of accesses
};

/**
 * @brief FLOP (Floating Point Operation) analysis
 */
struct FLOPAnalysis {
    uint64_t addOps = 0;        // Addition/subtraction operations
    uint64_t mulOps = 0;        // Multiplication operations
    uint64_t divOps = 0;        // Division operations
    uint64_t fmaOps = 0;        // Fused multiply-add operations
    uint64_t sqrtOps = 0;       // Square root operations
    uint64_t transcendentalOps = 0; // sin, cos, exp, log, etc.
    uint64_t totalFLOPs = 0;    // Total FLOPs
    
    void calculateTotal() {
        // FMA counts as 2 FLOPs (multiply + add)
        // Transcendental operations have realistic costs:
        // - sin/cos/tan: ~10 FLOPs (Taylor series approximation)
        // - exp/log/pow: ~15 FLOPs (complex approximations)
        // - sqrt: ~5 FLOPs (Newton-Raphson iteration)
        // Using conservative estimate of 10 FLOPs per transcendental
        totalFLOPs = addOps + mulOps + divOps + (fmaOps * 2) + 
                     (sqrtOps * 5) + (transcendentalOps * 10);
    }
};

/**
 * @brief Instruction-level profiling information
 */
struct InstructionProfile {
    uint64_t loadInstructions = 0;      // Memory load operations
    uint64_t storeInstructions = 0;     // Memory store operations
    uint64_t branchInstructions = 0;    // Branch/jump operations
    uint64_t compareInstructions = 0;   // Comparison operations
    uint64_t castInstructions = 0;      // Type cast operations
    uint64_t callInstructions = 0;      // Function call operations
    uint64_t totalInstructions = 0;     // Total instruction count
    
    void calculateTotal() {
        totalInstructions = loadInstructions + storeInstructions + 
                           branchInstructions + compareInstructions +
                           castInstructions + callInstructions;
    }
};

/**
 * @brief Enhanced kernel IR with detailed analysis
 */
struct EnhancedKernelIR : public KernelIR {
    FLOPAnalysis flopAnalysis;
    InstructionProfile instructionProfile;
    std::vector<MemoryAccessPattern> memoryPatterns;
    std::vector<std::string> deviceFunctions;  // Called device functions
    bool hasRecursion = false;
    bool hasTemplates = false;
    std::string templateSignature;
    uint64_t estimatedMemoryAccesses = 0;
    double arithmeticIntensity = 0.0;  // FLOPs per byte
};

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
    
    /**
     * @brief Parse kernel with enhanced analysis
     * @param name The kernel name to find.
     * @param source The full CUDA source code.
     * @param outIR The resulting EnhancedKernelIR object.
     * @return VGREResult::SUCCESS on success.
     */
    VGREResult parseEnhanced(const std::string& name,
                            const std::string& source,
                            EnhancedKernelIR& outIR);

private:
    std::string runClangAstDump(const std::string& source);
    
    // AST analysis functions
    void analyzeFLOPs(const llvm::json::Object* obj, FLOPAnalysis& flops);
    void analyzeInstructions(const llvm::json::Object* obj, InstructionProfile& profile);
    void analyzeMemoryAccess(const llvm::json::Object* obj, 
                            std::vector<MemoryAccessPattern>& patterns);
    void findDeviceFunctions(const llvm::json::Object* obj,
                            std::vector<std::string>& functions);
    bool detectRecursion(const llvm::json::Object* obj, 
                        const std::string& functionName);
    bool detectTemplates(const llvm::json::Object* obj,
                        std::string& signature);
    
    // Helper functions
    bool isFLOPOperation(const std::string& opcode);
    bool isMemoryOperation(const std::string& kind);
    MemoryAccessPattern::Type classifyAccessPattern(const llvm::json::Object* expr);
    
    std::unordered_map<std::string, KernelIR> cache_;
    std::unordered_map<std::string, EnhancedKernelIR> enhancedCache_;
    std::unordered_map<std::string, std::string> astCache_; // Cache raw AST output
    std::recursive_mutex cacheMutex_;
};

} // namespace compiler
} // namespace vgre

#endif // VGRE_COMPILER_CLANG_KERNEL_PARSER_H
