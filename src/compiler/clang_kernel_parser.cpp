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
#endif

namespace vgre {
namespace compiler {

// ── Process-level caches shared across all ClangKernelParser instances ──────
// Prevents repeated llvm::json::parse() (~12s per 378MB JSON) when different
// test instances parse the same kernel within the same binary run.
namespace {
static std::recursive_mutex& getProcessCacheMutex() {
    static std::recursive_mutex* m = new std::recursive_mutex();
    return *m;
}

static std::unordered_map<std::string, EnhancedKernelIR>& getEnhancedIRCache() {
    static std::unordered_map<std::string, EnhancedKernelIR>* cache = new std::unordered_map<std::string, EnhancedKernelIR>();
    return *cache;
}

// Return the vgre disk cache directory
static std::string getVgreCacheDir() {
    return vgre::common::getCacheRoot() + "/cache";
}

// Strip template arguments from a function/class name for flexible matching.
// "myKernel<float, 4>" → "myKernel"
static std::string stripTemplateArgs(const std::string& s) {
    auto pos = s.find('<');
    return pos == std::string::npos ? s : s.substr(0, pos);
}

// Normalize whitespace within template argument lists so that
// "myKernel<float ,4>" and "myKernel<float, 4>" compare equal.
// Removes spaces immediately adjacent to '<', '>', and ','.
static std::string normalizeTemplateName(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == ' ' || c == '\t') {
            if (!out.empty()) {
                char prev = out.back();
                if (prev == '<' || prev == '>' || prev == ',') continue;
            }
            if (i + 1 < s.size()) {
                char next = s[i + 1];
                if (next == '<' || next == '>' || next == ',') continue;
            }
        }
        out += c;
    }
    return out;
}

// Match a candidate name against a target, ignoring template arguments on
// either side and normalizing whitespace within template argument lists.
static bool kernelNameMatches(const std::string& candidate, const std::string& target) {
    if (target.empty()) return true;       // empty target → match anything
    if (candidate == target) return true;
    std::string normCand = normalizeTemplateName(candidate);
    std::string normTgt  = normalizeTemplateName(target);
    if (normCand == normTgt) return true;
    return stripTemplateArgs(normCand) == stripTemplateArgs(normTgt) ||
           stripTemplateArgs(normCand) == normTgt ||
           normCand == stripTemplateArgs(normTgt);
}

// Compute disk path for a cached EnhancedKernelIR
static std::string enhancedCachePath(const std::string& hash) {
    std::string dir = getVgreCacheDir() + "/" + hash.substr(0, 2);
    try { std::filesystem::create_directories(dir); } catch (...) {}
    return dir + "/" + hash + ".enh.json";
}

// Serialize EnhancedKernelIR to disk
static void saveEnhancedIR(const std::string& hash, const EnhancedKernelIR& ir) {
    std::string path = enhancedCachePath(hash);
    llvm::json::Object obj;
    // Basic KernelIR fields
    obj["name"]   = ir.name;
    obj["source"] = ir.source;
    obj["irCode"] = ir.irCode;
    llvm::json::Array types;
    for (auto t : ir.argTypes) types.push_back(static_cast<int64_t>(t));
    obj["argTypes"] = std::move(types);
    llvm::json::Array anames;
    for (const auto& n : ir.argTypeNames) anames.push_back(n);
    obj["argTypeNames"] = std::move(anames);
    llvm::json::Array sizes;
    for (auto s : ir.argSizes) sizes.push_back(static_cast<int64_t>(s));
    obj["argSizes"] = std::move(sizes);
    obj["usesSharedMem"]              = ir.usesSharedMem;
    obj["usesSyncthreads"]            = ir.usesSyncthreads;
    obj["sharedMemSize"]              = static_cast<int64_t>(ir.sharedMemSize);
    obj["estimatedInstructionCount"]  = static_cast<int64_t>(ir.estimatedInstructionCount);
    obj["estimatedMemoryAccessCount"] = static_cast<int64_t>(ir.estimatedMemoryAccessCount);
    obj["staticFlopCount"]            = static_cast<int64_t>(ir.staticFlopCount);
    // Enhanced fields
    llvm::json::Object flop;
    flop["addOps"]             = static_cast<int64_t>(ir.flopAnalysis.addOps);
    flop["mulOps"]             = static_cast<int64_t>(ir.flopAnalysis.mulOps);
    flop["divOps"]             = static_cast<int64_t>(ir.flopAnalysis.divOps);
    flop["fmaOps"]             = static_cast<int64_t>(ir.flopAnalysis.fmaOps);
    flop["sqrtOps"]            = static_cast<int64_t>(ir.flopAnalysis.sqrtOps);
    flop["transcendentalOps"]  = static_cast<int64_t>(ir.flopAnalysis.transcendentalOps);
    flop["totalFLOPs"]         = static_cast<int64_t>(ir.flopAnalysis.totalFLOPs);
    obj["flopAnalysis"] = std::move(flop);
    llvm::json::Object prof;
    prof["loadInstructions"]    = static_cast<int64_t>(ir.instructionProfile.loadInstructions);
    prof["storeInstructions"]   = static_cast<int64_t>(ir.instructionProfile.storeInstructions);
    prof["branchInstructions"]  = static_cast<int64_t>(ir.instructionProfile.branchInstructions);
    prof["compareInstructions"] = static_cast<int64_t>(ir.instructionProfile.compareInstructions);
    prof["castInstructions"]    = static_cast<int64_t>(ir.instructionProfile.castInstructions);
    prof["callInstructions"]    = static_cast<int64_t>(ir.instructionProfile.callInstructions);
    prof["totalInstructions"]   = static_cast<int64_t>(ir.instructionProfile.totalInstructions);
    obj["instructionProfile"] = std::move(prof);
    llvm::json::Array pats;
    for (const auto& p : ir.memoryPatterns) {
        llvm::json::Object po;
        po["type"]             = static_cast<int64_t>(static_cast<int>(p.type));
        po["stride"]           = static_cast<int64_t>(p.stride);
        po["isCoalesced"]      = p.isCoalesced;
        po["accessSize"]       = static_cast<int64_t>(p.accessSize);
        po["estimatedAccesses"]= static_cast<int64_t>(p.estimatedAccesses);
        pats.push_back(std::move(po));
    }
    obj["memoryPatterns"] = std::move(pats);
    llvm::json::Array dfuncs;
    for (const auto& f : ir.deviceFunctions) dfuncs.push_back(f);
    obj["deviceFunctions"]        = std::move(dfuncs);
    obj["hasRecursion"]           = ir.hasRecursion;
    obj["hasTemplates"]           = ir.hasTemplates;
    obj["templateSignature"]      = ir.templateSignature;
    obj["estimatedMemoryAccesses"]= static_cast<int64_t>(ir.estimatedMemoryAccesses);
    obj["arithmeticIntensity"]    = ir.arithmeticIntensity;
    std::ofstream f(path);
    if (f.is_open()) {
        f << llvm::formatv("{0:2}", llvm::json::Value(std::move(obj))).str();
    }
}

// Deserialize EnhancedKernelIR from disk; returns true on success
static bool loadEnhancedIR(const std::string& hash, EnhancedKernelIR& outIR) {
    std::string path = enhancedCachePath(hash);
    std::ifstream f(path);
    if (!f.is_open()) return false;
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    auto expected = llvm::json::parse(json);
    if (!expected) return false;
    const auto* obj = expected->getAsObject();
    if (!obj) return false;
    // Basic KernelIR
    outIR.name   = obj->getString("name").value_or("").str();
    outIR.source = obj->getString("source").value_or("").str();
    outIR.irCode = obj->getString("irCode").value_or("").str();
    if (const auto* arr = obj->getArray("argTypes"))
        for (const auto& v : *arr)
            outIR.argTypes.push_back(static_cast<ArgType>(v.getAsInteger().value_or(0)));
    if (const auto* arr = obj->getArray("argTypeNames"))
        for (const auto& v : *arr)
            outIR.argTypeNames.push_back(v.getAsString().value_or("").str());
    if (const auto* arr = obj->getArray("argSizes"))
        for (const auto& v : *arr)
            outIR.argSizes.push_back(static_cast<size_t>(v.getAsInteger().value_or(0)));
    outIR.usesSharedMem              = obj->getBoolean("usesSharedMem").value_or(false);
    outIR.usesSyncthreads            = obj->getBoolean("usesSyncthreads").value_or(false);
    outIR.sharedMemSize              = static_cast<size_t>(obj->getInteger("sharedMemSize").value_or(0));
    outIR.estimatedInstructionCount  = static_cast<uint64_t>(obj->getInteger("estimatedInstructionCount").value_or(0));
    outIR.estimatedMemoryAccessCount = static_cast<uint64_t>(obj->getInteger("estimatedMemoryAccessCount").value_or(0));
    outIR.staticFlopCount            = static_cast<uint64_t>(obj->getInteger("staticFlopCount").value_or(0));
    // Enhanced fields
    if (const auto* fl = obj->getObject("flopAnalysis")) {
        outIR.flopAnalysis.addOps            = static_cast<uint64_t>(fl->getInteger("addOps").value_or(0));
        outIR.flopAnalysis.mulOps            = static_cast<uint64_t>(fl->getInteger("mulOps").value_or(0));
        outIR.flopAnalysis.divOps            = static_cast<uint64_t>(fl->getInteger("divOps").value_or(0));
        outIR.flopAnalysis.fmaOps            = static_cast<uint64_t>(fl->getInteger("fmaOps").value_or(0));
        outIR.flopAnalysis.sqrtOps           = static_cast<uint64_t>(fl->getInteger("sqrtOps").value_or(0));
        outIR.flopAnalysis.transcendentalOps = static_cast<uint64_t>(fl->getInteger("transcendentalOps").value_or(0));
        outIR.flopAnalysis.totalFLOPs        = static_cast<uint64_t>(fl->getInteger("totalFLOPs").value_or(0));
    }
    if (const auto* pr = obj->getObject("instructionProfile")) {
        outIR.instructionProfile.loadInstructions    = static_cast<uint64_t>(pr->getInteger("loadInstructions").value_or(0));
        outIR.instructionProfile.storeInstructions   = static_cast<uint64_t>(pr->getInteger("storeInstructions").value_or(0));
        outIR.instructionProfile.branchInstructions  = static_cast<uint64_t>(pr->getInteger("branchInstructions").value_or(0));
        outIR.instructionProfile.compareInstructions = static_cast<uint64_t>(pr->getInteger("compareInstructions").value_or(0));
        outIR.instructionProfile.castInstructions    = static_cast<uint64_t>(pr->getInteger("castInstructions").value_or(0));
        outIR.instructionProfile.callInstructions    = static_cast<uint64_t>(pr->getInteger("callInstructions").value_or(0));
        outIR.instructionProfile.totalInstructions   = static_cast<uint64_t>(pr->getInteger("totalInstructions").value_or(0));
    }
    if (const auto* pats = obj->getArray("memoryPatterns")) {
        for (const auto& pv : *pats) {
            if (const auto* po = pv.getAsObject()) {
                MemoryAccessPattern p;
                p.type             = static_cast<MemoryAccessPattern::Type>(po->getInteger("type").value_or(4));
                p.stride           = static_cast<size_t>(po->getInteger("stride").value_or(0));
                p.isCoalesced      = po->getBoolean("isCoalesced").value_or(false);
                p.accessSize       = static_cast<size_t>(po->getInteger("accessSize").value_or(4));
                p.estimatedAccesses= static_cast<uint64_t>(po->getInteger("estimatedAccesses").value_or(0));
                outIR.memoryPatterns.push_back(p);
            }
        }
    }
    if (const auto* df = obj->getArray("deviceFunctions"))
        for (const auto& v : *df)
            outIR.deviceFunctions.push_back(v.getAsString().value_or("").str());
    outIR.hasRecursion            = obj->getBoolean("hasRecursion").value_or(false);
    outIR.hasTemplates            = obj->getBoolean("hasTemplates").value_or(false);
    outIR.templateSignature       = obj->getString("templateSignature").value_or("").str();
    outIR.estimatedMemoryAccesses = static_cast<uint64_t>(obj->getInteger("estimatedMemoryAccesses").value_or(0));
    outIR.arithmeticIntensity     = obj->getNumber("arithmeticIntensity").value_or(0.0);
    return true;
}
} // anonymous namespace

ClangKernelParser::ClangKernelParser() {
    // Initialize persistent cache on first use
    KernelCache::instance().initialize();
}

ClangKernelParser::~ClangKernelParser() = default;

static uint64_t countInstructionsRecursively(const llvm::json::Object* obj) {
    if (!obj) return 0;
    uint64_t count = 0;
    
    std::string kind = obj->getString("kind").value_or("").str();
    // Core Compute & Memory Ops
    if (kind == "BinaryOperator" || kind == "UnaryOperator" || 
        kind == "ArraySubscriptExpr" || kind == "MemberExpr" ||
        kind == "CallExpr" || kind == "ImplicitCastExpr") {
        count = 1;
    }
    // Control Flow and Structure
    if (kind == "ForStmt" || kind == "WhileStmt" || kind == "DoStmt") {
        // Kernels typically run many iterations; 32 is a reasonable warp-aligned minimum for estimation
        count = 32; 
    } else if (kind == "IfStmt" || kind == "SwitchStmt" || kind == "ConditionalOperator") {
        // Branches have a cost of comparison + jump
        count = 4;
    }
    
    const auto* inner = obj->getArray("inner");
    if (inner) {
        for (const auto& node : *inner) {
            if (const auto* child = node.getAsObject()) {
                count += countInstructionsRecursively(child);
            }
        }
    }
    return count;
}

std::string ClangKernelParser::runClangAstDump(const std::string& source) {
    // Generate hash for cache key
    std::string sourceHash = std::to_string(std::hash<std::string>{}(source));
    
    VGRE_LOG_INFO("ClangKernelParser", "Source hash: " + sourceHash.substr(0, 16) + "... (length: " + std::to_string(source.length()) + " bytes)");
    
    // Check persistent disk cache first (FAST PATH)
    std::string cachedAst;
    if (KernelCache::instance().getAST(sourceHash, cachedAst)) {
        VGRE_LOG_INFO("ClangKernelParser", "✓ Persistent cache HIT - skipping Clang invocation");
        return cachedAst;
    }
    
    // Check in-memory AST cache (MEDIUM PATH)
    {
        std::lock_guard<std::recursive_mutex> lock(cacheMutex_);
        auto it = astCache_.find(sourceHash);
        if (it != astCache_.end()) {
            VGRE_LOG_INFO("ClangKernelParser", "✓ Memory cache HIT");
            return it->second;
        }
    }
    
    // SLOW PATH: Run Clang (only happens on first parse of new kernel)
    VGRE_LOG_WARN("ClangKernelParser", "✗ Cache MISS - running Clang (this will be slow ~25s)");
    
    auto tmpDir = std::filesystem::temp_directory_path();
    std::string tempPath = (tmpDir / "vgre_kernel_tmp.cu").string();
    
    std::ofstream ofs(tempPath);
    if (!ofs.is_open()) {
        VGRE_LOG_ERROR("ClangKernelParser", "Failed to create temp file: " + tempPath);
        return "";
    }
    ofs << source;
    ofs.close();

    // Run clang++ to get JSON AST with optimizations
    std::string includePath = vgre::common::findIncludeDir();
    std::string clangPath = vgre::common::findCompilerPath();
    
    // Add -fno-delayed-template-parsing for faster parsing
    std::string cmd = "\"" + clangPath + "\" -Xclang -ast-dump=json -fsyntax-only -xc++ -w "
                     "-fno-delayed-template-parsing -I\"" + includePath + "\" \"" + 
                     tempPath + "\" 2>&1";
    VGRE_LOG_DEBUG("ClangKernelParser", "Running: " + cmd);
    
    std::string result;
    char buffer[8192]; // Larger buffer for faster I/O
#if defined(_WIN32)
#define VGRE_POPEN _popen
#define VGRE_PCLOSE _pclose
#else
#define VGRE_POPEN popen
#define VGRE_PCLOSE pclose
#endif
    FILE* pipe = VGRE_POPEN(cmd.c_str(), "r");
    int status = -1;
    if (pipe) {
        result.reserve(65536); // Pre-allocate for better performance
        while (fgets(buffer, sizeof(buffer), pipe)) {
            result += buffer;
        }
        status = VGRE_PCLOSE(pipe);
        VGRE_LOG_DEBUG("ClangKernelParser", "Command finished with status: " + std::to_string(status));
    } else {
        VGRE_LOG_ERROR("ClangKernelParser", "popen failed for: " + cmd);
        return "";
    }

    if (status != 0) {
        VGRE_LOG_ERROR("ClangKernelParser", "Clang failed with status " + std::to_string(status) + ":\n" + result);
        std::filesystem::remove(tempPath);
        return "";
    }

    if (result.empty()) {
        VGRE_LOG_WARN("ClangKernelParser", "Command returned NO output");
    }

    std::filesystem::remove(tempPath);
    
    // Cache the AST result in BOTH caches
    {
        std::lock_guard<std::recursive_mutex> lock(cacheMutex_);
        astCache_[sourceHash] = result;
    }
    
    // Store in persistent cache for future runs
    KernelCache::instance().putAST(sourceHash, result);
    VGRE_LOG_INFO("ClangKernelParser", "✓ AST cached - subsequent parses will be instant");
    
    return result;
}

VGREResult ClangKernelParser::parse(const std::string& name,
                                  const std::string& source,
                                  KernelIR& outIR) {
    // 1. Check in-memory cache first to avoid slow Clang process
    std::string cacheKey = name + ":::" + source;
    {
        std::lock_guard<std::recursive_mutex> lock(cacheMutex_);
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
    std::string sourceHash = std::to_string(std::hash<std::string>{}(sourceWithHeader));

    // Check KernelIR disk cache first (Ultra-Fast Path)
    if (KernelCache::instance().getKernelIR(sourceHash, name, outIR)) {
        VGRE_LOG_INFO("ClangKernelParser", "✓ KernelIR cache HIT for " + name + " (hash: " + sourceHash.substr(0,8) + ")");
        // Update memory cache
        std::lock_guard<std::recursive_mutex> lock(cacheMutex_);
        cache_[cacheKey] = outIR;
        return VGREResult::SUCCESS;
    }

    std::string jsonAst = runClangAstDump(sourceWithHeader);

    if (jsonAst.empty()) {
        VGRE_LOG_ERROR("ClangKernelParser", "Failed to get AST from Clang");
        return VGREResult::ERR_COMPILATION;
    }

    auto expectedAst = llvm::json::parse(jsonAst);
    if (!expectedAst) {
        VGRE_LOG_ERROR("ClangKernelParser", "Failed to parse Clang JSON AST");
        return VGREResult::ERR_INVALID_KERNEL;
    }

    const auto* root = expectedAst->getAsObject();
    if (!root) return VGREResult::ERR_INVALID_KERNEL;

    const auto* inner = root->getArray("inner");
    if (!inner) return VGREResult::ERR_INVALID_KERNEL;

    bool found = false;
    // Helper: check whether a FunctionDecl node has the SectionAttr "vgre_global"
    auto hasSectionAttr = [](const llvm::json::Object* funcObj) -> bool {
        const auto* inner = funcObj->getArray("inner");
        if (!inner) return false;
        for (const auto& child : *inner) {
            const auto* childObj = child.getAsObject();
            if (childObj && childObj->getString("kind").value_or("") == "SectionAttr" &&
                childObj->getString("section_name").value_or("") == "vgre_global") {
                return true;
            }
        }
        return false;
    };

    std::function<const llvm::json::Object*(const llvm::json::Array*, const std::string&)> findKernel;
    findKernel = [&](const llvm::json::Array* innerArray, const std::string& targetName) -> const llvm::json::Object* {
        if (!innerArray) return nullptr;

        for (const auto& node : *innerArray) {
            const auto* obj = node.getAsObject();
            if (!obj) continue;

            std::string kind = obj->getString("kind").value_or("").str();

            if (kind == "FunctionDecl") {
                std::string funcName = obj->getString("name").value_or("").str();
                if (kernelNameMatches(funcName, targetName) && hasSectionAttr(obj)) {
                    return obj;
                }
            } else if (kind == "FunctionTemplateDecl") {
                // Primary template: contains TemplateTypeParmDecl + FunctionDecl children.
                std::string tplName = obj->getString("name").value_or("").str();
                if (kernelNameMatches(tplName, targetName)) {
                    const auto* tplInner = obj->getArray("inner");
                    if (tplInner) {
                        for (const auto& tplChild : *tplInner) {
                            const auto* tplChildObj = tplChild.getAsObject();
                            if (!tplChildObj) continue;
                            std::string ck = tplChildObj->getString("kind").value_or("").str();
                            if (ck == "FunctionDecl") {
                                if (hasSectionAttr(tplChildObj) || !targetName.empty()) {
                                    return tplChildObj;
                                }
                            }
                        }
                    }
                }
            } else if (kind == "FunctionTemplateSpecializationDecl" || kind == "FunctionDecl") {
                // Explicit template specializations emitted as FunctionDecl with a
                // templateSpecializationArgs field, or a dedicated specialization node.
                std::string funcName = obj->getString("name").value_or("").str();
                if (kernelNameMatches(funcName, targetName) && hasSectionAttr(obj)) {
                    return obj;
                }
            } else if (kind == "ClassTemplateDecl" || kind == "ClassTemplateSpecializationDecl" ||
                       kind == "CXXRecordDecl") {
                // Classes/structs may contain __global__ static member functions.
                const auto* nested = findKernel(obj->getArray("inner"), targetName);
                if (nested) return nested;
            } else if (kind == "LinkageSpecDecl" || kind == "NamespaceDecl") {
                // Recurse into extern "C" blocks and namespaces.
                const auto* nested = findKernel(obj->getArray("inner"), targetName);
                if (nested) return nested;
            }
        }
        return nullptr;
    };

    const auto* kernelObj = findKernel(inner, name);
    if (!kernelObj && !name.empty()) {
        // Fallback: If requested name fails, try finding ANY global kernel
        VGRE_LOG_WARN("ClangKernelParser", "Specific name '" + name + "' not found, falling back to first global kernel found.");
        kernelObj = findKernel(inner, ""); // Pass empty name to find any
    }

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
        outIR.usesSharedMem = (source.find("__shared__") != std::string::npos || 
                               source.find("sharedMem") != std::string::npos);
        outIR.usesSyncthreads = (source.find("__syncthreads()") != std::string::npos);

        // Authoritative Instruction Estimation (AST-based)
        outIR.estimatedInstructionCount = countInstructionsRecursively(kernelObj);
        // Apply intelligent minimum based on kernel complexity
        // Simple kernels have at least 5 instructions (prologue + epilogue)
        // But if we found any operations, trust the count
        if (outIR.estimatedInstructionCount == 0) {
            // Empty kernel body - use minimal count
            outIR.estimatedInstructionCount = 5;
        }
    }

    if (!found) {
        VGRE_LOG_ERROR("ClangKernelParser", "Kernel function not found or not __global__: " + name);
        return VGREResult::ERR_INVALID_KERNEL;
    }

    // Store to persistent cache for "Very Firster" speed in next sessions
    KernelCache::instance().putKernelIR(sourceHash, name, outIR);

    {
        std::lock_guard<std::recursive_mutex> lock(cacheMutex_);
        cache_[cacheKey] = outIR;
    }

    return VGREResult::SUCCESS;
}



// ── Enhanced Analysis Functions ────────────────────────────────────────────

bool ClangKernelParser::isFLOPOperation(const std::string& opcode) {
    static const std::unordered_set<std::string> flopOps = {
        "+", "-", "*", "/", "+=", "-=", "*=", "/=",
        "fadd", "fsub", "fmul", "fdiv"
    };
    return flopOps.find(opcode) != flopOps.end();
}

bool ClangKernelParser::isMemoryOperation(const std::string& kind) {
    return (kind == "ArraySubscriptExpr" || kind == "MemberExpr" || 
            kind == "DeclRefExpr" || kind == "UnaryOperator");
}

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

VGREResult ClangKernelParser::parseEnhanced(const std::string& name,
                                           const std::string& source,
                                           EnhancedKernelIR& outIR) {
    // Build cache keys up-front (needed by all cache tiers)
    std::string cacheKey = name + ":::" + source;
    std::string enhHash = std::to_string(std::hash<std::string>{}(cacheKey));
    std::string sourceWithHeader = "#include \"vgre/compiler/cpu_cuda_env.h\"\n" + source;

    // ── Fast paths: check all cache tiers BEFORE calling parse() so that
    // the outIR is still empty when loadEnhancedIR() pushes into its vectors.

    // 1. Process-level static cache (instant, within-binary sharing)
    {
        std::lock_guard<std::recursive_mutex> lock(getProcessCacheMutex());
        auto it = getEnhancedIRCache().find(cacheKey);
        if (it != getEnhancedIRCache().end()) {
            VGRE_LOG_DEBUG("ClangKernelParser", "EnhancedIR process-cache HIT for " + name);
            outIR = it->second;
            return VGREResult::SUCCESS;
        }
    }

    // 2. Instance-level cache (intra-instance fast path)
    {
        std::lock_guard<std::recursive_mutex> lock(cacheMutex_);
        auto it = enhancedCache_.find(cacheKey);
        if (it != enhancedCache_.end()) {
            outIR = it->second;
            std::lock_guard<std::recursive_mutex> slock(getProcessCacheMutex());
            getEnhancedIRCache()[cacheKey] = outIR;
            return VGREResult::SUCCESS;
        }
    }

    // 3. Disk cache (skips llvm::json::parse entirely on warm runs)
    //    outIR must be default-constructed here so push_back starts from empty.
    outIR = EnhancedKernelIR{};
    if (loadEnhancedIR(enhHash, outIR)) {
        VGRE_LOG_INFO("ClangKernelParser", "✓ EnhancedIR disk cache HIT for " + name);
        {
            std::lock_guard<std::recursive_mutex> lock(cacheMutex_);
            enhancedCache_[cacheKey] = outIR;
        }
        {
            std::lock_guard<std::recursive_mutex> lock(getProcessCacheMutex());
            getEnhancedIRCache()[cacheKey] = outIR;
        }
        return VGREResult::SUCCESS;
    }

    // ── Slow path: run full Clang analysis for the first time ────────────────
    // Get basic IR first (may itself hit disk cache, so is fast after first run)
    KernelIR basicIR;
    VGREResult r = parse(name, source, basicIR);
    if (r != VGREResult::SUCCESS) {
        return r;
    }

    // Copy basic IR fields into outIR
    outIR = EnhancedKernelIR{};
    outIR.name = basicIR.name;
    outIR.source = basicIR.source;
    outIR.argTypes = basicIR.argTypes;
    outIR.argTypeNames = basicIR.argTypeNames;
    outIR.argSizes = basicIR.argSizes;
    outIR.usesSharedMem = basicIR.usesSharedMem;
    outIR.usesSyncthreads = basicIR.usesSyncthreads;
    outIR.estimatedInstructionCount = basicIR.estimatedInstructionCount;

    std::string jsonAst = runClangAstDump(sourceWithHeader);
    if (jsonAst.empty()) {
        return VGREResult::ERR_COMPILATION;
    }
    
    auto expectedAst = llvm::json::parse(jsonAst);
    if (!expectedAst) {
        return VGREResult::ERR_INVALID_KERNEL;
    }
    
    const auto* root = expectedAst->getAsObject();
    if (!root) return VGREResult::ERR_INVALID_KERNEL;
    
    // Find kernel function in AST
    const auto* inner = root->getArray("inner");
    if (!inner) return VGREResult::ERR_INVALID_KERNEL;
    
    std::function<const llvm::json::Object*(const llvm::json::Array*)> findKernel;
    findKernel = [&](const llvm::json::Array* innerArray) -> const llvm::json::Object* {
        if (!innerArray) return nullptr;
        for (const auto& node : *innerArray) {
            const auto* obj = node.getAsObject();
            if (!obj) continue;
            std::string kind = obj->getString("kind").value_or("").str();
            if (kind == "FunctionDecl") {
                std::string funcName = obj->getString("name").value_or("").str();
                if (kernelNameMatches(funcName, name)) {
                    return obj;
                }
            } else if (kind == "FunctionTemplateDecl") {
                // Primary template: name matches → return the nested FunctionDecl
                std::string tplName = obj->getString("name").value_or("").str();
                if (kernelNameMatches(tplName, name)) {
                    const auto* tplInner = obj->getArray("inner");
                    if (tplInner) {
                        for (const auto& tplChild : *tplInner) {
                            const auto* tplChildObj = tplChild.getAsObject();
                            if (tplChildObj &&
                                tplChildObj->getString("kind").value_or("") == "FunctionDecl") {
                                return tplChildObj;
                            }
                        }
                    }
                }
            } else if (kind == "FunctionTemplateSpecializationDecl") {
                std::string funcName = obj->getString("name").value_or("").str();
                if (kernelNameMatches(funcName, name)) {
                    return obj;
                }
            } else if (kind == "ClassTemplateDecl" ||
                       kind == "ClassTemplateSpecializationDecl" ||
                       kind == "CXXRecordDecl") {
                // Search for __global__ static member functions inside classes.
                const auto* nested = findKernel(obj->getArray("inner"));
                if (nested) return nested;
            } else if (kind == "LinkageSpecDecl" || kind == "NamespaceDecl") {
                const auto* nested = findKernel(obj->getArray("inner"));
                if (nested) return nested;
            }
        }
        return nullptr;
    };
    
    const auto* kernelObj = findKernel(inner);
    if (!kernelObj) {
        return VGREResult::ERR_INVALID_KERNEL;
    }
    
    // Perform enhanced analysis
    analyzeFLOPs(kernelObj, outIR.flopAnalysis);
    outIR.flopAnalysis.calculateTotal();
    
    analyzeInstructions(kernelObj, outIR.instructionProfile);
    outIR.instructionProfile.calculateTotal();
    
    analyzeMemoryAccess(kernelObj, outIR.memoryPatterns);
    
    findDeviceFunctions(kernelObj, outIR.deviceFunctions);
    
    outIR.hasRecursion = detectRecursion(kernelObj, name);
    
    outIR.hasTemplates = detectTemplates(kernelObj, outIR.templateSignature);
    
    // Calculate arithmetic intensity with actual memory access sizes
    outIR.estimatedMemoryAccesses = outIR.memoryPatterns.size();
    if (outIR.estimatedMemoryAccesses > 0) {
        // Calculate total bytes transferred based on actual access sizes
        uint64_t totalBytes = 0;
        for (const auto& pattern : outIR.memoryPatterns) {
            totalBytes += pattern.accessSize * pattern.estimatedAccesses;
        }
        
        // Fallback if no bytes calculated
        if (totalBytes == 0) {
            totalBytes = outIR.estimatedMemoryAccesses * 4; // Default to 4 bytes
        }
        
        outIR.arithmeticIntensity = static_cast<double>(outIR.flopAnalysis.totalFLOPs) / totalBytes;
    }
    
    // Cache the result in all layers: instance, process-static, and disk
    {
        std::lock_guard<std::recursive_mutex> lock(cacheMutex_);
        enhancedCache_[cacheKey] = outIR;
    }
    {
        std::lock_guard<std::recursive_mutex> lock(getProcessCacheMutex());
        getEnhancedIRCache()[cacheKey] = outIR;
    }
    saveEnhancedIR(enhHash, outIR);

    // Log detailed FLOP breakdown for transparency
    std::string flopBreakdown = "FLOP breakdown: ";
    if (outIR.flopAnalysis.addOps > 0) 
        flopBreakdown += "Add=" + std::to_string(outIR.flopAnalysis.addOps) + " ";
    if (outIR.flopAnalysis.mulOps > 0) 
        flopBreakdown += "Mul=" + std::to_string(outIR.flopAnalysis.mulOps) + " ";
    if (outIR.flopAnalysis.divOps > 0) 
        flopBreakdown += "Div=" + std::to_string(outIR.flopAnalysis.divOps) + " ";
    if (outIR.flopAnalysis.fmaOps > 0) 
        flopBreakdown += "FMA=" + std::to_string(outIR.flopAnalysis.fmaOps) + "(*2) ";
    if (outIR.flopAnalysis.sqrtOps > 0) 
        flopBreakdown += "Sqrt=" + std::to_string(outIR.flopAnalysis.sqrtOps) + "(*5) ";
    if (outIR.flopAnalysis.transcendentalOps > 0) 
        flopBreakdown += "Transcendental=" + std::to_string(outIR.flopAnalysis.transcendentalOps) + "(*10) ";
    
    VGRE_LOG_DEBUG("ClangKernelParser", flopBreakdown);
    
    VGRE_LOG_INFO("ClangKernelParser", 
                  "Enhanced analysis complete: " + name + 
                  " (FLOPs: " + std::to_string(outIR.flopAnalysis.totalFLOPs) + 
                  ", Memory: " + std::to_string(outIR.estimatedMemoryAccesses) + 
                  ", Instructions: " + std::to_string(outIR.instructionProfile.totalInstructions) +
                  ", AI: " + std::to_string(outIR.arithmeticIntensity) + ")");
    
    return VGREResult::SUCCESS;
}

} // namespace compiler
} // namespace vgre
