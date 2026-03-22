#include "vgre/compiler/clang_kernel_parser.h"
#include "vgre/common/logger.h"
#include "vgre/common/system_utils.h"

#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>

namespace vgre {
namespace compiler {

ClangKernelParser::ClangKernelParser() = default;
ClangKernelParser::~ClangKernelParser() = default;

std::string ClangKernelParser::runClangAstDump(const std::string& source) {
    auto tmpDir = std::filesystem::temp_directory_path();
    std::string tempPath = (tmpDir / "vgre_kernel_tmp.cu").string();
    
    std::ofstream ofs(tempPath);
    if (!ofs.is_open()) {
        VGRE_LOG_ERROR("ClangKernelParser", "Failed to create temp file: " + tempPath);
        return "";
    }
    ofs << source;
    ofs.close();

    // Run clang++ to get JSON AST
    std::string includePath = vgre::common::findIncludeDir();
    std::string cmd = "clang++ -Xclang -ast-dump=json -fsyntax-only -xc++ -w -I\"" + includePath + "\" \"" + tempPath + "\" 2>&1";
    VGRE_LOG_DEBUG("ClangKernelParser", "Running: " + cmd);
    
    std::string result;
    char buffer[4096];
#if defined(_WIN32)
#define VGRE_POPEN _popen
#define VGRE_PCLOSE _pclose
#else
#define VGRE_POPEN popen
#define VGRE_PCLOSE pclose
#endif
    FILE* pipe = VGRE_POPEN(cmd.c_str(), "r");
    if (pipe) {
        while (fgets(buffer, sizeof(buffer), pipe)) {
            result += buffer;
        }
        int status = VGRE_PCLOSE(pipe);
        VGRE_LOG_DEBUG("ClangKernelParser", "Command finished with status: " + std::to_string(status));
    } else {
        VGRE_LOG_ERROR("ClangKernelParser", "popen failed for: " + cmd);
    }

    if (result.empty()) {
        VGRE_LOG_WARN("ClangKernelParser", "Command returned NO output");
    }

    std::filesystem::remove(tempPath);
    return result;
}

VGREResult ClangKernelParser::parse(const std::string& name,
                                  const std::string& source,
                                  KernelIR& outIR) {
    // 1. Check in-memory cache first to avoid slow Clang process
    std::string cacheKey = name + ":::" + source;
    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        auto it = cache_.find(cacheKey);
        if (it != cache_.end()) {
            VGRE_LOG_DEBUG("ClangKernelParser", "Parser Cache HIT for kernel: " + name);
            outIR = it->second;
            return VGREResult::SUCCESS;
        }
    }

    VGRE_LOG_INFO("ClangKernelParser", "Authoritative Parsing: " + name);

    // Inject the environment header to ensure __global__ and __device__ are correctly defined as annotations
    std::string sourceWithHeader = "#include \"vgre/compiler/cpu_cuda_env.h\"\n" + source;
    std::string jsonAst = runClangAstDump(sourceWithHeader);
    if (jsonAst.empty()) {
        VGRE_LOG_ERROR("ClangKernelParser", "Failed to get AST from Clang");
        return VGREResult::ERROR_COMPILATION;
    }

    auto expectedAst = llvm::json::parse(jsonAst);
    if (!expectedAst) {
        VGRE_LOG_ERROR("ClangKernelParser", "Failed to parse Clang JSON AST");
        return VGREResult::ERROR_INVALID_KERNEL;
    }

    const auto* root = expectedAst->getAsObject();
    if (!root) return VGREResult::ERROR_INVALID_KERNEL;

    const auto* inner = root->getArray("inner");
    if (!inner) return VGREResult::ERROR_INVALID_KERNEL;

    bool found = false;
    std::function<const llvm::json::Object*(const llvm::json::Array*)> findKernel;
    findKernel = [&](const llvm::json::Array* innerArray) -> const llvm::json::Object* {
        if (!innerArray) return nullptr;
        
        for (const auto& node : *innerArray) {
            const auto* obj = node.getAsObject();
            if (!obj) continue;
            
            std::string kind = obj->getString("kind").value_or("").str();
            
            if (kind == "FunctionDecl") {
                std::string funcName = obj->getString("name").value_or("").str();
                if (name.empty() || funcName == name) {
                    // Check for SectionAttr "vgre_global"
                    const auto* funcInner = obj->getArray("inner");
                    if (funcInner) {
                        for (const auto& child : *funcInner) {
                            const auto* childObj = child.getAsObject();
                            if (childObj && childObj->getString("kind").value_or("") == "SectionAttr") {
                                if (childObj->getString("section_name").value_or("") == "vgre_global") {
                                    return obj;
                                }
                            }
                        }
                    }
                }
            } else if (kind == "LinkageSpecDecl" || kind == "NamespaceDecl") {
                // Recurse into linkage or namespace
                const auto* nestedObj = findKernel(obj->getArray("inner"));
                if (nestedObj) return nestedObj;
            }
        }
        return nullptr;
    };

    const auto* kernelObj = findKernel(inner);
    if (kernelObj) {
        found = true;
        // Extract parameters from kernelObj
        std::string funcName = kernelObj->getString("name").value_or("").str();
        outIR.name = funcName;
        outIR.source = source;
        outIR.argTypes.clear();
        outIR.argTypeNames.clear();
        outIR.argSizes.clear();

        const auto* funcInner = kernelObj->getArray("inner");
        if (funcInner) {
            for (const auto& child : *funcInner) {
                const auto* pObj = child.getAsObject();
                if (pObj && pObj->getString("kind").value_or("") == "ParmVarDecl") {
                    const auto* typeObj = pObj->getObject("type");
                    if (typeObj) {
                        std::string qualType = typeObj->getString("qualType").value_or("").str();
                        outIR.argTypeNames.push_back(qualType);
                        
                        bool recognized = false;
                        bool isPointer = (qualType.find('*') != std::string::npos);
                        ArgType at = mapType(qualType, isPointer, recognized);
                        outIR.argTypes.push_back(at);
                        
                        if (at == ArgType::STRUCT) {
                            outIR.argSizes.push_back(computeStructSize(qualType, source));
                        } else {
                            outIR.argSizes.push_back(0);
                        }
                    }
                }
            }
        }
        outIR.usesSharedMem = (source.find("__shared__") != std::string::npos);
        outIR.usesSyncthreads = (source.find("__syncthreads()") != std::string::npos);
    }

    if (!found) {
        VGRE_LOG_ERROR("ClangKernelParser", "Kernel function not found or not __global__: " + name);
        return VGREResult::ERROR_INVALID_KERNEL;
    }

    {
        std::lock_guard<std::mutex> lock(cacheMutex_);
        cache_[cacheKey] = outIR;
    }

    return VGREResult::SUCCESS;
}

} // namespace compiler
} // namespace vgre
