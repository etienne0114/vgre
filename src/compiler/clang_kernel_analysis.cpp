#include "vgre/compiler/clang_kernel_parser.h"
#include "vgre/compiler/kernel_cache.h"
#include "vgre/common/logger.h"
#include "vgre/common/system_utils.h"

#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FormatVariadic.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#ifndef _WIN32
#include <unistd.h>
#include <pwd.h>
#include <sys/wait.h>
#endif


#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>

namespace vgre {
namespace compiler {

void ClangKernelParser::analyzeInstructions(const llvm::json::Object* obj, InstructionProfile& profile) {
    if (!obj) return;
    
    std::string kind = obj->getString("kind").value_or("").str();
    
    // Detect stores: ArraySubscriptExpr on LHS of assignment
    if (kind == "BinaryOperator") {
        std::string opcode = obj->getString("opcode").value_or("").str();
        
        // Assignment operators indicate a store
        if (opcode == "=" || opcode == "+=" || opcode == "-=" || 
            opcode == "*=" || opcode == "/=") {
            const auto* inner = obj->getArray("inner");
            if (inner && inner->size() >= 1) {
                const auto* lhs = (*inner)[0].getAsObject();
                if (lhs) {
                    std::string lhsKind = lhs->getString("kind").value_or("").str();
                    if (lhsKind == "ArraySubscriptExpr" || lhsKind == "MemberExpr") {
                        profile.storeInstructions++;
                    }
                }
            }
        }
        
        // Comparison operations
        if (opcode == "<" || opcode == ">" || opcode == "<=" || opcode == ">=" ||
            opcode == "==" || opcode == "!=") {
            profile.compareInstructions++;
        }
    }
    
    // Detect loads: ArraySubscriptExpr NOT on LHS of assignment
    // We need to track if we're in an assignment context
    if (kind == "ArraySubscriptExpr" || kind == "MemberExpr") {
        // This is a memory access - determine if it's a load or store
        // by checking if parent is an assignment (handled above)
        // For now, count as load (stores are counted in assignment handling)
        profile.loadInstructions++;
    }
    
    // Branch operations
    if (kind == "IfStmt" || kind == "SwitchStmt" || kind == "ConditionalOperator" ||
        kind == "ForStmt" || kind == "WhileStmt" || kind == "DoStmt") {
        profile.branchInstructions++;
    }
    
    // Cast operations
    if (kind == "CStyleCastExpr" || kind == "CXXStaticCastExpr" || 
        kind == "CXXReinterpretCastExpr" || kind == "ImplicitCastExpr") {
        profile.castInstructions++;
    }
    
    // Function calls
    if (kind == "CallExpr") {
        profile.callInstructions++;
    }
    
    // Recurse into children
    const auto* inner = obj->getArray("inner");
    if (inner) {
        for (const auto& node : *inner) {
            if (const auto* child = node.getAsObject()) {
                analyzeInstructions(child, profile);
            }
        }
    }
}

void ClangKernelParser::analyzeFLOPs(const llvm::json::Object* obj, FLOPAnalysis& flops) {
    if (!obj) return;
    
    std::string kind = obj->getString("kind").value_or("").str();
    
    if (kind == "BinaryOperator") {
        std::string opcode = obj->getString("opcode").value_or("").str();
        const auto* typeObj = obj->getObject("type");
        if (typeObj) {
            std::string qualType = typeObj->getString("qualType").value_or("").str();
            bool isFloat = (qualType.find("float") != std::string::npos || 
                           qualType.find("double") != std::string::npos);
            
            if (isFloat) {
                if (opcode == "+" || opcode == "-" || opcode == "+=" || opcode == "-=") {
                    flops.addOps++;
                } else if (opcode == "*" || opcode == "*=") {
                    flops.mulOps++;
                } else if (opcode == "/" || opcode == "/=") {
                    flops.divOps++;
                }
            }
        }
    } else if (kind == "CallExpr") {
        // Try multiple ways to extract function name
        std::string calleeName;
        
        const auto* inner = obj->getArray("inner");
        if (inner && !inner->empty()) {
            const auto* calleeObj = (*inner)[0].getAsObject();
            if (calleeObj) {
                std::string calleeKind = calleeObj->getString("kind").value_or("").str();
                
                if (calleeKind == "DeclRefExpr") {
                    // Try referencedDecl first
                    const auto* refDeclObj = calleeObj->getObject("referencedDecl");
                    if (refDeclObj) {
                        calleeName = refDeclObj->getString("name").value_or("").str();
                    }
                    
                    // If empty, try name field
                    if (calleeName.empty()) {
                        calleeName = calleeObj->getString("name").value_or("").str();
                    }
                } else if (calleeKind == "ImplicitCastExpr") {
                    // Function call might be wrapped in implicit cast
                    const auto* castInner = calleeObj->getArray("inner");
                    if (castInner && !castInner->empty()) {
                        const auto* innerRef = (*castInner)[0].getAsObject();
                        if (innerRef && innerRef->getString("kind").value_or("") == "DeclRefExpr") {
                            const auto* refDeclObj = innerRef->getObject("referencedDecl");
                            if (refDeclObj) {
                                calleeName = refDeclObj->getString("name").value_or("").str();
                            }
                            if (calleeName.empty()) {
                                calleeName = innerRef->getString("name").value_or("").str();
                            }
                        }
                    }
                }
            }
        }
        
        // Classify the function call
        if (!calleeName.empty()) {
            if (calleeName.find("sqrt") != std::string::npos) {
                flops.sqrtOps++;
            } else if (calleeName.find("fma") != std::string::npos) {
                flops.fmaOps++;
            } else if (calleeName.find("sin") != std::string::npos ||
                      calleeName.find("cos") != std::string::npos ||
                      calleeName.find("tan") != std::string::npos ||
                      calleeName.find("exp") != std::string::npos ||
                      calleeName.find("log") != std::string::npos ||
                      calleeName.find("pow") != std::string::npos) {
                flops.transcendentalOps++;
            }
        }
    } else if (kind == "ForStmt" || kind == "WhileStmt") {
        // Analyze loop body
        FLOPAnalysis loopFlops;
        uint64_t iterationEstimate = 10; // Reasonable default for unknown bounds
        bool foundBound = false;
        
        const auto* body = obj->getArray("inner");
        if (body) {
            // Recursively search for IntegerLiteral (loop bound) in the entire loop structure
            std::function<void(const llvm::json::Object*)> extractBound;
            extractBound = [&](const llvm::json::Object* node) {
                if (!node || foundBound) return;
                
                std::string nodeKind = node->getString("kind").value_or("").str();
                
                // Look for integer literals in loop conditions
                if (nodeKind == "IntegerLiteral") {
                    std::string value = node->getString("value").value_or("").str();
                    if (!value.empty()) {
                        try {
                            uint64_t bound = std::stoull(value);
                            // Accept reasonable loop bounds (1 to 100,000)
                            if (bound > 1 && bound < 100000) {
                                iterationEstimate = bound;
                                foundBound = true;
                                return;
                            }
                        } catch (...) {}
                    }
                }
                
                // Also check for DeclRefExpr that might reference constants
                if (!foundBound && nodeKind == "DeclRefExpr") {
                    const auto* refDecl = node->getObject("referencedDecl");
                    if (refDecl) {
                        std::string refKind = refDecl->getString("kind").value_or("").str();
                        // If it's a const variable, try to find its value
                        if (refKind == "VarDecl") {
                            const auto* refInner = refDecl->getArray("inner");
                            if (refInner) {
                                for (const auto& child : *refInner) {
                                    if (const auto* childObj = child.getAsObject()) {
                                        extractBound(childObj);
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Recurse into children
                const auto* innerNodes = node->getArray("inner");
                if (innerNodes) {
                    for (const auto& child : *innerNodes) {
                        if (const auto* childObj = child.getAsObject()) {
                            extractBound(childObj);
                        }
                    }
                }
            };
            
            // First pass: extract loop bound
            for (const auto& node : *body) {
                if (const auto* child = node.getAsObject()) {
                    extractBound(child);
                }
            }
            
            // Second pass: analyze loop body FLOPs
            // The body is typically the last element in ForStmt
            for (size_t i = 0; i < body->size(); i++) {
                if (const auto* child = (*body)[i].getAsObject()) {
                    std::string childKind = child->getString("kind").value_or("").str();
                    // Analyze CompoundStmt (the actual loop body)
                    if (childKind == "CompoundStmt") {
                        analyzeFLOPs(child, loopFlops);
                    }
                }
            }
        }
        
        // For while loops without clear bounds, use higher estimate
        // While loops typically run more iterations than for loops
        if (!foundBound && kind == "WhileStmt") {
            iterationEstimate = 100; // Higher estimate for while loops
        }
        
        // Multiply loop body FLOPs by estimated iteration count
        flops.addOps += loopFlops.addOps * iterationEstimate;
        flops.mulOps += loopFlops.mulOps * iterationEstimate;
        flops.divOps += loopFlops.divOps * iterationEstimate;
        flops.fmaOps += loopFlops.fmaOps * iterationEstimate;
        flops.sqrtOps += loopFlops.sqrtOps * iterationEstimate;
        flops.transcendentalOps += loopFlops.transcendentalOps * iterationEstimate;
        
        // Don't recurse into inner after processing loop
        return;
    }
    
    const auto* inner = obj->getArray("inner");
    if (inner) {
        for (const auto& node : *inner) {
            if (const auto* child = node.getAsObject()) {
                analyzeFLOPs(child, flops);
            }
        }
    }
}

MemoryAccessPattern::Type ClangKernelParser::classifyAccessPattern(const llvm::json::Object* expr) {
    if (!expr) return MemoryAccessPattern::Type::UNKNOWN;
    
    std::string kind = expr->getString("kind").value_or("").str();
    
    if (kind == "ArraySubscriptExpr") {
        const auto* inner = expr->getArray("inner");
        if (inner && inner->size() >= 2) {
            const auto* indexExpr = (*inner)[1].getAsObject();
            if (indexExpr) {
                std::string indexKind = indexExpr->getString("kind").value_or("").str();
                
                // Check for direct threadIdx access (coalesced)
                if (indexKind == "DeclRefExpr") {
                    // Try to get the referenced declaration name
                    const auto* refDecl = indexExpr->getObject("referencedDecl");
                    std::string refName;
                    if (refDecl) {
                        refName = refDecl->getString("name").value_or("").str();
                    }
                    if (refName.empty()) {
                        refName = indexExpr->getString("name").value_or("").str();
                    }
                    
                    if (refName.find("threadIdx") != std::string::npos || 
                        refName.find("idx") != std::string::npos) {
                        return MemoryAccessPattern::Type::SEQUENTIAL;
                    }
                }
                // Check for strided access (threadIdx * stride or threadIdx + blockIdx * blockDim)
                else if (indexKind == "BinaryOperator") {
                    std::string opcode = indexExpr->getString("opcode").value_or("").str();
                    
                    // Multiplication indicates strided access
                    if (opcode == "*") {
                        return MemoryAccessPattern::Type::STRIDED;
                    }
                    // Addition might be sequential (threadIdx + blockIdx * blockDim)
                    else if (opcode == "+") {
                        // Check if it's the standard global index pattern
                        const auto* addInner = indexExpr->getArray("inner");
                        if (addInner && addInner->size() >= 2) {
                            bool hasThreadIdx = false;
                            bool hasBlockOffset = false;
                            
                            for (const auto& operand : *addInner) {
                                if (const auto* opObj = operand.getAsObject()) {
                                    std::string opKind = opObj->getString("kind").value_or("").str();
                                    
                                    // Check for threadIdx
                                    if (opKind == "DeclRefExpr") {
                                        const auto* refDecl = opObj->getObject("referencedDecl");
                                        std::string refName;
                                        if (refDecl) {
                                            refName = refDecl->getString("name").value_or("").str();
                                        }
                                        if (refName.find("threadIdx") != std::string::npos) {
                                            hasThreadIdx = true;
                                        }
                                    }
                                    // Check for blockIdx * blockDim pattern
                                    else if (opKind == "BinaryOperator") {
                                        std::string innerOp = opObj->getString("opcode").value_or("").str();
                                        if (innerOp == "*") {
                                            hasBlockOffset = true;
                                        }
                                    }
                                }
                            }
                            
                            // Standard global index pattern: threadIdx + blockIdx * blockDim
                            if (hasThreadIdx && hasBlockOffset) {
                                return MemoryAccessPattern::Type::SEQUENTIAL;
                            }
                        }
                        
                        // Otherwise, it's likely strided
                        return MemoryAccessPattern::Type::STRIDED;
                    }
                }
                // ImplicitCastExpr might wrap the actual index
                else if (indexKind == "ImplicitCastExpr") {
                    const auto* castInner = indexExpr->getArray("inner");
                    if (castInner && !castInner->empty()) {
                        const auto* actualIndex = (*castInner)[0].getAsObject();
                        if (actualIndex) {
                            // Recursively classify the actual index expression
                            // Create a temporary ArraySubscriptExpr-like structure
                            return classifyAccessPattern(actualIndex);
                        }
                    }
                }
            }
        }
    }
    
    return MemoryAccessPattern::Type::UNKNOWN;
}

void ClangKernelParser::analyzeMemoryAccess(const llvm::json::Object* obj,
                                           std::vector<MemoryAccessPattern>& patterns) {
    if (!obj) return;
    
    std::string kind = obj->getString("kind").value_or("").str();
    
    if (kind == "ArraySubscriptExpr") {
        MemoryAccessPattern pattern;
        pattern.type = classifyAccessPattern(obj);
        pattern.isCoalesced = (pattern.type == MemoryAccessPattern::Type::SEQUENTIAL);
        
        // Extract actual access size from type information
        pattern.accessSize = 4; // Default: float
        const auto* typeObj = obj->getObject("type");
        if (typeObj) {
            std::string qualType = typeObj->getString("qualType").value_or("").str();
            
            // Determine size based on type
            if (qualType.find("double") != std::string::npos) {
                pattern.accessSize = 8;
            } else if (qualType.find("float") != std::string::npos) {
                pattern.accessSize = 4;
            } else if (qualType.find("int") != std::string::npos || 
                      qualType.find("unsigned") != std::string::npos) {
                pattern.accessSize = 4;
            } else if (qualType.find("short") != std::string::npos) {
                pattern.accessSize = 2;
            } else if (qualType.find("char") != std::string::npos || 
                      qualType.find("byte") != std::string::npos) {
                pattern.accessSize = 1;
            } else if (qualType.find("long long") != std::string::npos) {
                pattern.accessSize = 8;
            }
            
            // Handle vector types (float2, float4, etc.)
            if (qualType.find("float2") != std::string::npos) {
                pattern.accessSize = 8;
            } else if (qualType.find("float4") != std::string::npos) {
                pattern.accessSize = 16;
            } else if (qualType.find("double2") != std::string::npos) {
                pattern.accessSize = 16;
            } else if (qualType.find("int2") != std::string::npos) {
                pattern.accessSize = 8;
            } else if (qualType.find("int4") != std::string::npos) {
                pattern.accessSize = 16;
            }
        }
        
        pattern.estimatedAccesses = 1;
        patterns.push_back(pattern);
    }
    
    const auto* inner = obj->getArray("inner");
    if (inner) {
        for (const auto& node : *inner) {
            if (const auto* child = node.getAsObject()) {
                analyzeMemoryAccess(child, patterns);
            }
        }
    }
}

void ClangKernelParser::findDeviceFunctions(const llvm::json::Object* obj,
                                           std::vector<std::string>& functions) {
    if (!obj) return;
    
    std::string kind = obj->getString("kind").value_or("").str();
    
    if (kind == "CallExpr") {
        const auto* inner = obj->getArray("inner");
        if (inner && !inner->empty()) {
            const auto* calleeObj = (*inner)[0].getAsObject();
            if (calleeObj && calleeObj->getString("kind").value_or("") == "DeclRefExpr") {
                std::string funcName = calleeObj->getString("referencedDecl").value_or("").str();
                if (!funcName.empty() && funcName.find("__device__") != std::string::npos) {
                    functions.push_back(funcName);
                }
            }
        }
    }
    
    const auto* inner = obj->getArray("inner");
    if (inner) {
        for (const auto& node : *inner) {
            if (const auto* child = node.getAsObject()) {
                findDeviceFunctions(child, functions);
            }
        }
    }
}

bool ClangKernelParser::detectRecursion(const llvm::json::Object* obj,
                                       const std::string& functionName) {
    if (!obj) return false;
    
    std::string kind = obj->getString("kind").value_or("").str();
    
    if (kind == "CallExpr") {
        const auto* inner = obj->getArray("inner");
        if (inner && !inner->empty()) {
            const auto* calleeObj = (*inner)[0].getAsObject();
            if (calleeObj) {
                std::string calleeName = calleeObj->getString("referencedDecl").value_or("").str();
                if (calleeName.find(functionName) != std::string::npos) {
                    return true;
                }
            }
        }
    }
    
    const auto* inner = obj->getArray("inner");
    if (inner) {
        for (const auto& node : *inner) {
            if (const auto* child = node.getAsObject()) {
                if (detectRecursion(child, functionName)) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

bool ClangKernelParser::detectTemplates(const llvm::json::Object* obj,
                                       std::string& signature) {
    if (!obj) return false;
    
    std::string kind = obj->getString("kind").value_or("").str();
    
    if (kind == "FunctionTemplateDecl" || kind == "ClassTemplateDecl") {
        signature = obj->getString("name").value_or("").str();
        const auto* templateParams = obj->getArray("templateParams");
        if (templateParams) {
            signature += "<";
            bool first = true;
            for (const auto& param : *templateParams) {
                if (!first) signature += ", ";
                const auto* paramObj = param.getAsObject();
                if (paramObj) {
                    signature += paramObj->getString("name").value_or("T").str();
                }
                first = false;
            }
            signature += ">";
        }
        return true;
    }
    
    return false;
}


} // namespace compiler
} // namespace vgre
