#include "vgre/compiler/kernel_parser.h"
#include "vgre/compiler/clang_kernel_parser.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>

namespace vgre {
namespace compiler {

namespace {
std::string normalizeTypeName(std::string typeName) {
    typeName = std::regex_replace(typeName, std::regex("\\b(const|volatile|restrict|__restrict__|__restrict)\\b"), "");
    typeName = std::regex_replace(typeName, std::regex("[\\t\\n\\r]+"), " ");
    typeName = std::regex_replace(typeName, std::regex("\\s+"), " ");
    if (!typeName.empty() && typeName.front() == ' ') {
        typeName.erase(0, typeName.find_first_not_of(' '));
    }
    if (!typeName.empty() && typeName.back() == ' ') {
        typeName.erase(typeName.find_last_not_of(' ') + 1);
    }
    return typeName;
}
} // namespace

// ── Built-in CUDA variables ────────────────────────────────────────────────
const std::unordered_map<std::string, std::string> KernelParser::builtinVars_ = {
    {"threadIdx",  "threadIdx"},
    {"blockIdx",   "blockIdx"},
    {"blockDim",   "blockDim"},
    {"gridDim",    "gridDim"},
    {"warpSize",   "warpSize"}
};

KernelParser::KernelParser()  = default;
KernelParser::~KernelParser() = default;

// ── Tokenizer ──────────────────────────────────────────────────────────────
std::vector<Token> KernelParser::tokenize(const std::string& source) {
    std::vector<Token> tokens;
    int line = 1, col = 0;
    size_t i = 0;

    auto isIdStart = [](char c) { return std::isalpha(c) || c == '_'; };
    auto isIdCont  = [](char c) { return std::isalnum(c) || c == '_'; };

    while (i < source.size()) {
        char c = source[i];

        // Skip whitespace
        if (std::isspace(c)) {
            if (c == '\n') { ++line; col = 0; }
            else { ++col; }
            ++i;
            continue;
        }

        // Skip line comments
        if (c == '/' && i + 1 < source.size() && source[i+1] == '/') {
            while (i < source.size() && source[i] != '\n') ++i;
            continue;
        }

        // Skip block comments
        if (c == '/' && i + 1 < source.size() && source[i+1] == '*') {
            i += 2;
            while (i + 1 < source.size() &&
                   !(source[i] == '*' && source[i+1] == '/')) {
                if (source[i] == '\n') { ++line; col = 0; }
                ++i;
            }
            i += 2;
            continue;
        }

        Token tok;
        tok.line = line;
        tok.col  = col;

        // Identifiers / keywords
        if (isIdStart(c)) {
            std::string word;
            while (i < source.size() && isIdCont(source[i])) {
                word += source[i++];
            }
            col += static_cast<int>(word.size());

            if (builtinVars_.count(word)) {
                tok.type = TokenType::BUILTIN_VAR;
            } else if (word == "__global__" || word == "__device__" ||
                       word == "__host__"   || word == "void"    ||
                       word == "return"     || word == "if"      ||
                       word == "else"       || word == "for"     ||
                       word == "while"      || word == "const"   ||
                       word == "kernel"     || word == "struct"  ||
                       word == "template"   || word == "typename"||
                       word == "class"      || word == "typedef"||
                       word == "enum"       || word == "union") {
                tok.type = TokenType::KEYWORD;
            } else if (word == "float"  || word == "double" ||
                       word == "int"    || word == "long"   ||
                       word == "unsigned" || word == "char"  ||
                       word == "size_t" || word == "uint32_t" ||
                       word == "int32_t"|| word == "int64_t" ||
                       word == "uint64_t") {
                tok.type = TokenType::TYPE;
            } else {
                tok.type = TokenType::IDENTIFIER;
            }
            tok.value = word;
            tokens.push_back(tok);
            continue;
        }

        // Numbers
        if (std::isdigit(c) || (c == '.' && i + 1 < source.size() &&
                                std::isdigit(source[i+1]))) {
            std::string num;
            bool isFloat = false;
            while (i < source.size() && (std::isdigit(source[i]) ||
                   source[i] == '.' || source[i] == 'f' ||
                   source[i] == 'e' || source[i] == 'E')) {
                if (source[i] == '.' || source[i] == 'f' ||
                    source[i] == 'e' || source[i] == 'E')
                    isFloat = true;
                num += source[i++];
            }
            tok.type  = isFloat ? TokenType::LITERAL_FLOAT : TokenType::LITERAL_INT;
            tok.value = num;
            col += static_cast<int>(num.size());
            tokens.push_back(tok);
            continue;
        }

        // Single-character tokens
        tok.value = std::string(1, c);
        ++col; ++i;
        switch (c) {
            case '(': tok.type = TokenType::LPAREN;    break;
            case ')': tok.type = TokenType::RPAREN;    break;
            case '{': tok.type = TokenType::LBRACE;    break;
            case '}': tok.type = TokenType::RBRACE;    break;
            case '[': tok.type = TokenType::LBRACKET;  break;
            case ']': tok.type = TokenType::RBRACKET;  break;
            case ';': tok.type = TokenType::SEMICOLON; break;
            case ',': tok.type = TokenType::COMMA;     break;
            default:  tok.type = TokenType::OPERATOR;  break;
        }
        tokens.push_back(tok);
    }

    Token eof;
    eof.type  = TokenType::END_OF_SOURCE;
    eof.value = "";
    eof.line  = line;
    eof.col   = col;
    tokens.push_back(eof);

    return tokens;
}

// ── Extract function ───────────────────────────────────────────────────────
VGREResult KernelParser::extractFunction(const std::vector<Token>& tokens,
                                          std::string& outName,
                                          size_t& outParamStart,
                                          size_t& outParamEnd,
                                          std::string& outBody,
                                          bool& outIsGlobal) {
    outIsGlobal = false;
    size_t i = 0;

    // Look for __global__ or kernel qualifier
    while (i < tokens.size()) {
        if (tokens[i].type == TokenType::KEYWORD &&
            (tokens[i].value == "__global__" || tokens[i].value == "kernel")) {
            outIsGlobal = true;
            ++i;
            break;
        }
        ++i;
    }

    if (!outIsGlobal) {
        i = 0;
    }

    // Skip return type
    while (i < tokens.size() &&
           (tokens[i].type == TokenType::TYPE ||
            tokens[i].type == TokenType::KEYWORD ||
            tokens[i].value == "::")) {
        ++i;
    }

    // Function name
    if (i < tokens.size() && tokens[i].type == TokenType::IDENTIFIER) {
        outName = tokens[i].value;
        ++i;
    } else {
        return VGREResult::ERROR_INVALID_KERNEL;
    }

    // Parameter list
    if (i < tokens.size() && tokens[i].type == TokenType::LPAREN) {
        outParamStart = i + 1;
        int depth = 1;
        ++i;
        while (i < tokens.size() && depth > 0) {
            if (tokens[i].type == TokenType::LPAREN) ++depth;
            if (tokens[i].type == TokenType::RPAREN) --depth;
            if (depth == 0) {
                outParamEnd = i;
                break;
            }
            ++i;
        }
        if (i < tokens.size()) ++i;
    } else {
        return VGREResult::ERROR_INVALID_KERNEL;
    }

    // Body
    if (i < tokens.size() && tokens[i].type == TokenType::LBRACE) {
        int depth = 1;
        ++i;
        while (i < tokens.size() && depth > 0) {
            if (tokens[i].type == TokenType::LBRACE) ++depth;
            if (tokens[i].type == TokenType::RBRACE) --depth;
            if (depth > 0) {
                outBody += tokens[i].value;
                if (tokens[i].type != TokenType::SEMICOLON &&
                    tokens[i].type != TokenType::LBRACE &&
                    tokens[i].type != TokenType::RBRACE &&
                    tokens[i].type != TokenType::OPERATOR) {
                    outBody += " ";
                }
            }
            ++i;
        }
    }

    return VGREResult::SUCCESS;
}

// ── Parse parameters ───────────────────────────────────────────────────────
VGREResult KernelParser::parseParameters(const std::vector<Token>& tokens,
                                         size_t startIndex, size_t endIndex,
                                         std::vector<ParsedParam>& outParams) {
    if (startIndex >= endIndex || endIndex > tokens.size()) return VGREResult::SUCCESS;
    
    std::vector<Token> currentParam;
    int templateDepth = 0;
    int parenDepth = 0;

    auto processParam = [&](const std::vector<Token>& pTokens) -> VGREResult {
        if (pTokens.empty()) return VGREResult::SUCCESS;
        ParsedParam pp;
        pp.isPointer = false;
        
        std::string typeStr;
        std::string nameStr;
        
        int nameIdx = static_cast<int>(pTokens.size()) - 1;
        
        while (nameIdx >= 0 && 
               (pTokens[nameIdx].type == TokenType::RBRACKET || 
                pTokens[nameIdx].type == TokenType::LBRACKET || 
                pTokens[nameIdx].value == "restrict" || 
                pTokens[nameIdx].value == "__restrict__")) {
            nameIdx--;
        }
        
        if (nameIdx >= 0 && 
           (pTokens[nameIdx].type == TokenType::IDENTIFIER || pTokens[nameIdx].type == TokenType::KEYWORD)) {
            nameStr = pTokens[nameIdx].value;
        } else {
            nameStr = "arg" + std::to_string(outParams.size());
            nameIdx = static_cast<int>(pTokens.size());
        }
        
        for (int j = 0; j < nameIdx; ++j) {
            if (pTokens[j].value == "*") pp.isPointer = true;
            if (pTokens[j].type != TokenType::KEYWORD || 
               (pTokens[j].value != "const" && pTokens[j].value != "restrict" && pTokens[j].value != "volatile")) {
                typeStr += pTokens[j].value;
            }
        }
        
        typeStr.erase(std::remove(typeStr.begin(), typeStr.end(), '*'), typeStr.end());
        nameStr.erase(std::remove(nameStr.begin(), nameStr.end(), '*'), nameStr.end());
        if (typeStr.empty()) typeStr = nameStr;
        
        pp.typeName = typeStr;
        pp.name = nameStr;
        bool recognized = false;
        pp.argType = mapType(pp.typeName, pp.isPointer, recognized);
        
        if (!recognized) {
             VGRE_LOG_ERROR("KernelParser", "Unsupported type: " + pp.typeName);
             return VGREResult::ERROR_INVALID_KERNEL;
        }
        outParams.push_back(pp);
        return VGREResult::SUCCESS;
    };

    for (size_t i = startIndex; i < endIndex; ++i) {
        const auto& tok = tokens[i];
        if (tok.value == "<") templateDepth++;
        else if (tok.value == ">") templateDepth--;
        else if (tok.type == TokenType::LPAREN) parenDepth++;
        else if (tok.type == TokenType::RPAREN) parenDepth--;
        
        if (tok.type == TokenType::COMMA && templateDepth == 0 && parenDepth == 0) {
            auto r = processParam(currentParam);
            if (r != VGREResult::SUCCESS) return r;
            currentParam.clear();
        } else {
            currentParam.push_back(tok);
        }
    }
    
    if (!currentParam.empty()) {
        auto r = processParam(currentParam);
        if (r != VGREResult::SUCCESS) return r;
    }
    
    return VGREResult::SUCCESS;
}

// ── Map C type to ArgType ──────────────────────────────────────────────────
ArgType KernelParser::mapType(const std::string& typeName, bool isPointer,
                              bool& recognized) {
    if (isPointer) {
        recognized = true;
        return ArgType::POINTER;
    }

    const std::string norm = normalizeTypeName(typeName);

    if (norm == "float") {
        recognized = true;
        return ArgType::FLOAT32;
    }
    if (norm == "double") {
        recognized = true;
        return ArgType::FLOAT64;
    }
    if (norm == "int" || norm == "int32_t" || norm == "signed" ||
        norm == "signed int" || norm == "char" || norm == "signed char" ||
        norm == "short" || norm == "short int" || norm == "int16_t" ||
        norm == "int8_t" || norm == "bool") {
        recognized = true;
        return ArgType::INT32;
    }
    if (norm == "unsigned" || norm == "unsigned int" || norm == "uint32_t" ||
        norm == "unsigned short" || norm == "unsigned short int" ||
        norm == "uint16_t" || norm == "unsigned char" || norm == "uint8_t") {
        recognized = true;
        return ArgType::UINT32;
    }
    if (norm == "int64_t" || norm == "long long" || norm == "long long int" ||
        norm == "signed long long" || norm == "signed long long int") {
        recognized = true;
        return ArgType::INT64;
    }
    if (norm == "uint64_t" || norm == "unsigned long long" ||
        norm == "unsigned long long int") {
        recognized = true;
        return ArgType::UINT64;
    }
    if (norm == "long" || norm == "long int" || norm == "signed long" ||
        norm == "signed long int") {
        recognized = true;
        return (sizeof(long) >= 8) ? ArgType::INT64 : ArgType::INT32;
    }
    if (norm == "unsigned long" || norm == "unsigned long int") {
        recognized = true;
        return (sizeof(unsigned long) >= 8) ? ArgType::UINT64 : ArgType::UINT32;
    }
    if (norm == "size_t") {
        recognized = true;
        return (sizeof(size_t) >= 8) ? ArgType::UINT64 : ArgType::UINT32;
    }

    // Handle compact signatures where extractor concatenates type+name
    // (e.g., "intN", "floatx", "unsignedCount").
    const std::regex intNamePattern(R"(^int[A-Za-z_][A-Za-z0-9_]*$)");
    const std::regex floatNamePattern(R"(^float[A-Za-z_][A-Za-z0-9_]*$)");
    const std::regex doubleNamePattern(R"(^double[A-Za-z_][A-Za-z0-9_]*$)");
    const std::regex longNamePattern(R"(^long[A-Za-z_][A-Za-z0-9_]*$)");
    const std::regex ulongNamePattern(R"(^unsigned ?long[A-Za-z_][A-Za-z0-9_]*$)");
    const std::regex unsignedNamePattern(R"(^unsigned[A-Za-z_][A-Za-z0-9_]*$)");
    if (std::regex_match(norm, intNamePattern)) {
        recognized = true;
        return ArgType::INT32;
    }
    if (std::regex_match(norm, floatNamePattern)) {
        recognized = true;
        return ArgType::FLOAT32;
    }
    if (std::regex_match(norm, doubleNamePattern)) {
        recognized = true;
        return ArgType::FLOAT64;
    }
    if (std::regex_match(norm, longNamePattern)) {
        recognized = true;
        return (sizeof(long) >= 8) ? ArgType::INT64 : ArgType::INT32;
    }
    if (std::regex_match(norm, ulongNamePattern) ||
        std::regex_match(norm, unsignedNamePattern)) {
        recognized = true;
        return ArgType::UINT32;
    }

    recognized = true;
    return ArgType::STRUCT;
}

// ── Find built-in variables ────────────────────────────────────────────────
std::vector<std::string> KernelParser::findBuiltinVars(const std::string& body) {
    std::vector<std::string> found;
    for (const auto& [name, _] : builtinVars_) {
        if (body.find(name) != std::string::npos) {
            found.push_back(name);
        }
    }
    return found;
}

// ── Main parse ─────────────────────────────────────────────────────────────
VGREResult KernelParser::parse(const std::string& name,
                                const std::string& source,
                                KernelIR& outIR) {
    VGRE_LOG_INFO("KernelParser", "Parsing kernel AST: " + name);

    auto tokens = tokenize(source);
    if (tokens.empty()) {
        return VGREResult::ERROR_INVALID_KERNEL;
    }

    std::string funcName, body;
    size_t paramStart = 0, paramEnd = 0;
    bool isGlobal = false;

    auto r = extractFunction(tokens, funcName, paramStart, paramEnd, body, isGlobal);
    if (r != VGREResult::SUCCESS) {
        VGRE_LOG_ERROR("KernelParser",
                       "Failed to extract function from kernel source");
        return r;
    }

    // Parse parameters
    std::vector<ParsedParam> params;
    auto paramResult = parseParameters(tokens, paramStart, paramEnd, params);
    if (paramResult != VGREResult::SUCCESS) {
        return paramResult;
    }

    // Build KernelIR
    outIR.name   = name.empty() ? funcName : name;
    outIR.source = source;
    outIR.irCode = "";  // Will be filled by LLVMTranslationEngine

    outIR.argTypes.clear();
    outIR.argTypeNames.clear();
    outIR.argSizes.clear();
    for (const auto& p : params) {
        outIR.argTypes.push_back(p.argType);
        outIR.argTypeNames.push_back(p.typeName);
        if (p.argType == ArgType::STRUCT) {
            size_t sz = computeStructSize(p.typeName, source);
            outIR.argSizes.push_back(sz);
        } else {
            outIR.argSizes.push_back(0); // 0 = deduce from ArgType
        }
    }

    // Check for shared memory usage
    outIR.usesSharedMem = (source.find("__shared__") != std::string::npos);
    outIR.usesSyncthreads = (source.find("__syncthreads()") != std::string::npos);

    // Use ClangKernelParser for accurate instruction and memory analysis
    // This replaces the old heuristic token-counting approach
    ClangKernelParser clangParser;
    EnhancedKernelIR enhancedIR;
    VGREResult clangResult = clangParser.parseEnhanced(outIR.name, source, enhancedIR);
    
    if (clangResult == VGREResult::SUCCESS) {
        // Use accurate AST-based analysis from ClangKernelParser
        outIR.estimatedInstructionCount = enhancedIR.instructionProfile.totalInstructions;
        outIR.estimatedMemoryAccessCount = enhancedIR.estimatedMemoryAccesses;
        
        VGRE_LOG_INFO("KernelParser",
                      "Using ClangKernelParser analysis: " +
                      std::to_string(outIR.estimatedInstructionCount) + " instructions, " +
                      std::to_string(outIR.estimatedMemoryAccessCount) + " memory accesses");
    } else {
        // Fallback to improved heuristic if Clang analysis fails
        VGRE_LOG_WARN("KernelParser",
                      "ClangKernelParser failed, using enhanced heuristic analysis");
        
        uint64_t estOps = 0;
        uint64_t estMem = 0;
        uint64_t loopDepth = 0;
        uint64_t loopMultiplier = 1;
        
        // Enhanced heuristic: count actual operations with context awareness
        for (size_t k = 0; k < tokens.size(); ++k) {
            const auto& tok = tokens[k];
            
            // Track loop depth for better estimation
            if (tok.type == TokenType::KEYWORD) {
                if (tok.value == "for" || tok.value == "while" || tok.value == "do") {
                    loopDepth++;
                    // Each loop level multiplies operations by estimated iterations
                    loopMultiplier *= 10;
                } else if (tok.value == "if") {
                    // Conditional branches add complexity
                    estOps += 2;  // Branch + comparison
                }
            }
            
            // Count arithmetic and logical operations
            if (tok.type == TokenType::OPERATOR) {
                if (tok.value == "+" || tok.value == "-" || tok.value == "*" || tok.value == "/" ||
                    tok.value == "&" || tok.value == "|" || tok.value == "^" || tok.value == "!" ||
                    tok.value == "=" || tok.value == "<" || tok.value == ">") {
                    estOps += 1;
                }
            }
            
            // Count array accesses (memory operations)
            if (tok.type == TokenType::LBRACKET) {
                estMem += 1;
            }
            
            // Count function calls as expensive operations
            if (tok.type == TokenType::IDENTIFIER && k + 1 < tokens.size() && 
                tokens[k+1].type == TokenType::LPAREN) {
                // Check if it's a math function (expensive)
                if (tok.value.find("sin") != std::string::npos ||
                    tok.value.find("cos") != std::string::npos ||
                    tok.value.find("sqrt") != std::string::npos ||
                    tok.value.find("exp") != std::string::npos ||
                    tok.value.find("log") != std::string::npos) {
                    estOps += 10;  // Transcendental functions are expensive
                } else {
                    estOps += 5;  // Regular function calls
                }
            }
            
            // Detect loop closing braces to track depth
            if (tok.type == TokenType::RBRACE && loopDepth > 0) {
                // This is approximate - we don't have perfect brace matching
                // but it helps estimate loop nesting
            }
        }
        
        // Apply loop multiplier to operations inside loops
        if (loopMultiplier > 1) {
            estOps *= loopMultiplier;
            estMem *= loopMultiplier;
        }
        
        // Use intelligent defaults based on kernel complexity
        // If we found operations, trust them; otherwise use reasonable minimums
        if (estOps == 0) {
            // No operations found - likely parsing issue or empty kernel
            estOps = 20;  // Minimum for a functional kernel
        }
        if (estMem == 0) {
            // No memory operations found - unusual but possible
            estMem = 4;  // Minimum memory accesses
        }
        
        outIR.estimatedInstructionCount = estOps;
        outIR.estimatedMemoryAccessCount = estMem;
        
        VGRE_LOG_INFO("KernelParser",
                      "Heuristic analysis: " + std::to_string(estOps) + " instructions, " +
                      std::to_string(estMem) + " memory accesses (loop depth: " + 
                      std::to_string(loopDepth) + ")");
    }

    auto builtins = findBuiltinVars(body);
    VGRE_LOG_INFO("KernelParser",
                  "Parsed AST for kernel '" + outIR.name + "' with " +
                  std::to_string(params.size()) + " params, " +
                  std::to_string(builtins.size()) + " built-in vars");

    return VGREResult::SUCCESS;
}

// ── Extract struct body from source ────────────────────────────────────────
bool KernelParser::extractStructBody(const std::string& typeName,
                                     const std::string& source,
                                     std::string& outBody) {
    // Search for "struct TypeName {" pattern
    std::string pattern = "struct\\s+" + typeName + "\\s*\\{";
    std::regex re(pattern);
    std::smatch m;
    if (!std::regex_search(source, m, re)) {
        return false;
    }

    // Find the matching closing brace
    size_t braceStart = static_cast<size_t>(m.position()) + m.length() - 1;
    int depth = 1;
    size_t pos = braceStart + 1;
    while (pos < source.size() && depth > 0) {
        if (source[pos] == '{') ++depth;
        if (source[pos] == '}') --depth;
        ++pos;
    }

    if (depth != 0) return false;

    outBody = source.substr(braceStart + 1, pos - braceStart - 2);
    return true;
}

// ── Compute struct size from its member declarations ───────────────────────
size_t KernelParser::computeStructSize(const std::string& typeName,
                                       const std::string& fullSource) const {
    std::string body;
    if (!extractStructBody(typeName, fullSource, body)) {
        // Cannot parse struct definition with regex; use ClangKernelParser for AST-based extraction
        VGRE_LOG_INFO("KernelParser",
                      "Cannot find struct definition for '" + typeName +
                      "', attempting ClangKernelParser AST-based analysis");
        
        // Use ClangKernelParser to get accurate struct size from AST
        ClangKernelParser clangParser;
        KernelIR dummyIR;
        
        // Create a dummy kernel that uses the struct to trigger AST analysis
        std::string testSource = fullSource + "\n__global__ void __vgre_struct_test__(" + 
                                typeName + "* ptr) { }";
        
        VGREResult result = clangParser.parse("__vgre_struct_test__", testSource, dummyIR);
        
        if (result == VGREResult::SUCCESS) {
            // Check if struct size was computed
            for (size_t i = 0; i < dummyIR.argTypes.size(); i++) {
                if (dummyIR.argTypes[i] == ArgType::STRUCT && dummyIR.argSizes[i] > 0) {
                    VGRE_LOG_INFO("KernelParser",
                                  "ClangKernelParser computed struct '" + typeName + 
                                  "' size: " + std::to_string(dummyIR.argSizes[i]) + " bytes");
                    return dummyIR.argSizes[i];
                }
            }
        }
        
        // If ClangKernelParser also fails, use sizeof-based estimation
        // This is more sophisticated than a hardcoded value
        size_t estimatedSize = 0;
        
        // Try to estimate based on common struct patterns
        if (fullSource.find("struct " + typeName) != std::string::npos) {
            // Count likely members by looking for semicolons in struct context
            size_t structPos = fullSource.find("struct " + typeName);
            size_t braceStart = fullSource.find('{', structPos);
            size_t braceEnd = fullSource.find('}', braceStart);
            
            if (braceStart != std::string::npos && braceEnd != std::string::npos) {
                std::string structContent = fullSource.substr(braceStart + 1, braceEnd - braceStart - 1);
                size_t memberCount = std::count(structContent.begin(), structContent.end(), ';');
                
                // Estimate: average 8 bytes per member (conservative for mixed types)
                estimatedSize = memberCount * 8;
                
                // Apply reasonable bounds
                if (estimatedSize < 8) estimatedSize = 8;    // Minimum struct size
                if (estimatedSize > 1024) estimatedSize = 1024;  // Maximum reasonable size
                
                VGRE_LOG_INFO("KernelParser",
                              "Estimated struct '" + typeName + "' size: " + 
                              std::to_string(estimatedSize) + " bytes (" + 
                              std::to_string(memberCount) + " members)");
                return estimatedSize;
            }
        }
        
        // Last resort: return reasonable default based on typical struct sizes
        VGRE_LOG_WARN("KernelParser",
                      "Could not determine struct '" + typeName + 
                      "' size, using default 64 bytes");
        return 64;
    }

    // Parse semicolon-delimited member declarations and sum sizes with proper alignment
    size_t totalSize = 0;
    size_t maxAlignment = 1;  // Track maximum alignment requirement
    
    std::istringstream ss(body);
    std::string decl;
    while (std::getline(ss, decl, ';')) {
        auto trimmed = normalizeTypeName(decl);
        if (trimmed.empty()) continue;

        // Determine base type size and alignment
        size_t memberSize = 0;
        size_t memberAlignment = 1;
        
        if (trimmed.find("double") != std::string::npos ||
            trimmed.find("int64_t") != std::string::npos ||
            trimmed.find("uint64_t") != std::string::npos ||
            trimmed.find("long long") != std::string::npos) {
            memberSize = 8;
            memberAlignment = 8;
        } else if (trimmed.find("size_t") != std::string::npos) {
            memberSize = sizeof(size_t);
            memberAlignment = sizeof(size_t);
        } else if (trimmed.find("float") != std::string::npos ||
                   trimmed.find("int32_t") != std::string::npos ||
                   trimmed.find("uint32_t") != std::string::npos ||
                   trimmed.find("int") != std::string::npos ||
                   trimmed.find("unsigned") != std::string::npos) {
            memberSize = 4;
            memberAlignment = 4;
        } else if (trimmed.find("int16_t") != std::string::npos ||
                   trimmed.find("uint16_t") != std::string::npos ||
                   trimmed.find("short") != std::string::npos) {
            memberSize = 2;
            memberAlignment = 2;
        } else if (trimmed.find("char") != std::string::npos ||
                   trimmed.find("int8_t") != std::string::npos ||
                   trimmed.find("uint8_t") != std::string::npos ||
                   trimmed.find("bool") != std::string::npos) {
            memberSize = 1;
            memberAlignment = 1;
        } else if (trimmed.find('*') != std::string::npos) {
            memberSize = sizeof(void*);
            memberAlignment = sizeof(void*);
        } else {
            // Nested struct or unknown — assume 8 bytes with 8-byte alignment
            memberSize = 8;
            memberAlignment = 8;
        }

        // Check for array syntax: e.g. "float data[16]"
        auto bracket = trimmed.find('[');
        if (bracket != std::string::npos) {
            auto closeBracket = trimmed.find(']', bracket);
            if (closeBracket != std::string::npos) {
                std::string countStr = trimmed.substr(
                    bracket + 1, closeBracket - bracket - 1);
                int count = 1;
                try { count = std::stoi(countStr); } catch (...) {}
                if (count > 0) memberSize *= static_cast<size_t>(count);
            }
        }

        // Apply padding before this member to satisfy alignment
        if (totalSize % memberAlignment != 0) {
            size_t padding = memberAlignment - (totalSize % memberAlignment);
            totalSize += padding;
        }

        totalSize += memberSize;
        
        // Track maximum alignment for final struct padding
        if (memberAlignment > maxAlignment) {
            maxAlignment = memberAlignment;
        }
    }

    // Apply final padding to satisfy struct alignment
    // Struct must be aligned to its largest member's alignment
    if (totalSize > 0 && totalSize % maxAlignment != 0) {
        totalSize = ((totalSize / maxAlignment) + 1) * maxAlignment;
    }

    VGRE_LOG_INFO("KernelParser",
                  "Computed struct '" + typeName + "' size: " +
                  std::to_string(totalSize) + " bytes (alignment: " +
                  std::to_string(maxAlignment) + ")");
    return totalSize;
}

} // namespace compiler
} // namespace vgre
