#ifndef VGRE_COMPILER_KERNEL_PARSER_H
#define VGRE_COMPILER_KERNEL_PARSER_H

#include "vgre/common/types.h"
#include "vgre/common/error_codes.h"

#include <string>
#include <vector>
#include <unordered_map>

namespace vgre {
namespace compiler {

// ── Token types for kernel source ──────────────────────────────────────────
enum class TokenType : uint8_t {
    KEYWORD,
    IDENTIFIER,
    TYPE,
    OPERATOR,
    LITERAL_INT,
    LITERAL_FLOAT,
    LPAREN,
    RPAREN,
    LBRACE,
    RBRACE,
    LBRACKET,
    RBRACKET,
    SEMICOLON,
    COMMA,
    BUILTIN_VAR,    // threadIdx, blockIdx, blockDim, gridDim
    END_OF_SOURCE
};

struct Token {
    TokenType   type;
    std::string value;
    int         line = 0;
    int         col  = 0;
};

// ── Parsed parameter descriptor ────────────────────────────────────────────
struct ParsedParam {
    std::string typeName;     // e.g. "float*", "int"
    std::string name;
    ArgType     argType;
    bool        isPointer = false;
};

// ── Kernel Parser ──────────────────────────────────────────────────────────
class KernelParser {
public:
    KernelParser();
    virtual ~KernelParser();

    // Parse a CUDA-like kernel source to KernelIR
    virtual VGREResult parse(const std::string& name,
                             const std::string& source,
                             KernelIR& outIR);

    // Extract parameter list from a function signature AST subrange
    VGREResult parseParameters(const std::vector<Token>& tokens,
                               size_t startIndex, size_t endIndex,
                               std::vector<ParsedParam>& outParams);

    // Identify CUDA built-in variables in the body
    std::vector<std::string> findBuiltinVars(const std::string& body);

    // Parse struct definitions from kernel source and compute sizes
    size_t computeStructSize(const std::string& typeName,
                             const std::string& fullSource) const;

    // Extract inline struct definitions from kernel source
    static bool extractStructBody(const std::string& typeName,
                                  const std::string& source,
                                  std::string& outBody);

protected:
    // Tokenizer
    std::vector<Token> tokenize(const std::string& source);

    // Extract function signature indices, body, and qualifiers
    VGREResult extractFunction(const std::vector<Token>& tokens,
                               std::string& outName,
                               size_t& outParamStart,
                               size_t& outParamEnd,
                               std::string& outBody,
                               bool& outIsGlobal);

    // Map C type string to ArgType
    ArgType mapType(const std::string& typeName, bool isPointer, bool& recognized);

    // Built-in variable set
    // Built-in variable map (lazily initialized)
    static const std::unordered_map<std::string, std::string>& getBuiltinVars();

    // Cache for token-based fallback results (keyed by kernel source SHA)
    std::unordered_map<std::string, KernelIR> fallbackCache_;
};

// ── PTX register count parser ────────────────────────────────────────────────
// Scans a PTX module for `.reg` declarations within a named `.entry` kernel
// and returns the total number of registers allocated.  Returns 0 if the
// kernel cannot be found or the PTX is empty, allowing the caller to fall
// back to a conservative default.
int parsePTXRegisterCount(const std::string& ptx, const std::string& kernelName);

} // namespace compiler
} // namespace vgre

#endif // VGRE_COMPILER_KERNEL_PARSER_H
