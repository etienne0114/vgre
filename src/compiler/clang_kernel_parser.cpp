#include "vgre/compiler/clang_kernel_parser.h"
#include "vgre/compiler/kernel_cache.h"
#include "vgre/common/logger.h"
#include "vgre/common/system_utils.h"

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4244 4267 4100 4127 4624)
#endif
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FormatVariadic.h>
#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "vgre/common/os_backend.h"
#if !defined(_WIN32)
#include <pwd.h>       // getpwuid — home dir for temp files
#include <sys/wait.h>  // waitpid — clang subprocess
#endif

namespace vgre {
namespace compiler {

// ── Minimal CUDA stub for AST-only analysis ─────────────────────────────────
// Used instead of #include "cpu_cuda_env.h" when running clang++ purely for
// AST structure analysis (argument types, FLOP counts, memory patterns, etc.).
// The full cpu_cuda_env.h pulls in <cmath>, <cstdint>, CUB, cooperative groups
// and other heavy headers, inflating the JSON AST dump to 300-400 MB and
// making llvm::json::parse() consume 600 MB+ of RAM per call.  This inline
// stub provides only the declarations needed to parse typical CUDA kernel
// signatures without errors, keeping the JSON to ~50 KB.
static constexpr const char kAstAnalysisStub[] = R"VGRE_STUB(
// Block re-inclusion of the full vgre CUDA emulation headers.
// This stub provides only what is needed to parse kernel AST without OOM.
#define VGRE_COMPILER_CPU_CUDA_ENV_H
#define VGRE_COMPILER_CPU_CUDA_FP16_H
#define VGRE_COMPILER_CPU_CUDA_WARP_H
#define VGRE_WMMA_EMULATION_H
#define VGRE_COMPILER_CUDA_DEVICE_LIBS_COOPERATIVE_GROUPS_H
#define VGRE_COMPILER_CUDA_DEVICE_LIBS_CUB_FALLBACK_H
#define VGRE_CURAND_KERNEL_H

typedef signed char        int8_t;
typedef short              int16_t;
typedef int                int32_t;
typedef long long          int64_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned long      size_t;

// CUDA kernel annotations — must produce SectionAttr "vgre_global" so that
// hasSectionAttr() in the parser can identify __global__ functions.
#define __global__ __attribute__((section("vgre_global")))
#define __device__ __attribute__((section("vgre_device")))
#define __host__
#define __shared__
#define __restrict__
#define __forceinline__ inline
#define __noinline__

// Minimal CUDA built-in types
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
    unsigned int total() const { return x * y * z; }
};
extern dim3 threadIdx, blockIdx, blockDim, gridDim;

void __syncthreads();
void __syncwarp(unsigned mask = 0xffffffff);

// Scalar math (forward declarations — no heavy <cmath> needed)
float sinf(float); float cosf(float); float tanf(float);
float asinf(float); float acosf(float); float atanf(float);
float atan2f(float, float);
float expf(float); float exp2f(float); float logf(float); float log2f(float);
float powf(float, float); float sqrtf(float); float rsqrtf(float);
float fabsf(float); float floorf(float); float ceilf(float);
float roundf(float); float truncf(float); float fmodf(float, float);
float fmaf(float, float, float);
float __fdividef(float, float);
float __fadd_rn(float, float);
float __fmul_rn(float, float);
double sin(double); double cos(double); double tan(double);
double exp(double); double log(double); double sqrt(double);
double pow(double, double); double fabs(double);
double floor(double); double ceil(double);

// Atomic operations
template<typename T> T atomicAdd(T*, T);
template<typename T> T atomicSub(T*, T);
template<typename T> T atomicExch(T*, T);
template<typename T> T atomicCAS(T*, T, T);
template<typename T> T atomicMin(T*, T);
template<typename T> T atomicMax(T*, T);
template<typename T> T atomicOr(T*, T);
template<typename T> T atomicAnd(T*, T);

// Warp primitives
template<typename T> T __shfl_sync(unsigned mask, T var, int srcLane, int width = 32);
template<typename T> T __shfl_up_sync(unsigned mask, T var, unsigned delta, int width = 32);
template<typename T> T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width = 32);
template<typename T> T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = 32);
unsigned __ballot_sync(unsigned mask, int pred);
int __all_sync(unsigned mask, int pred);
int __any_sync(unsigned mask, int pred);

// Minimal half-precision stubs (avoid heavy cuda_fp16.h)
struct __half { unsigned short __x; };
struct __nv_bfloat16 { unsigned short __x; };

// Cooperative groups stub — provides just enough for AST parsing.
// Actual implementation lives in cuda_device_libs/cooperative_groups.h
// which is still used by the JIT compilation path.
namespace cooperative_groups {
    class thread_block {
    public:
        void sync() const {}
        unsigned size() const { return 0; }
        unsigned thread_rank() const { return 0; }
    };
    class grid_group {
    public:
        void sync() const {}
        unsigned size() const { return 0; }
        unsigned thread_rank() const { return 0; }
        bool is_valid() const { return true; }
    };
    class multi_grid_group {
    public:
        void sync() const {}
        unsigned size() const { return 0; }
        unsigned thread_rank() const { return 0; }
        unsigned num_grids() const { return 1; }
        unsigned grid_rank() const { return 0; }
    };
    // coalesced_group: active-thread subset; sync() + shfl operations
    class coalesced_group {
    public:
        void sync() const {}
        unsigned size() const { return 32; }
        unsigned thread_rank() const { return 0; }
        template<typename T> T shfl(T val, int) const { return val; }
        template<typename T> T shfl_up(T val, unsigned) const { return val; }
        template<typename T> T shfl_down(T val, unsigned) const { return val; }
        template<typename T> T shfl_xor(T val, int) const { return val; }
    };
    template<int Sz> class thread_block_tile {
    public:
        void sync() const {}
        unsigned size() const { return Sz; }
        unsigned thread_rank() const { return 0; }
        template<typename T> T shfl(T val, int) const { return val; }
        template<typename T> T shfl_xor(T val, int) const { return val; }
        template<typename T> T shfl_down(T val, unsigned) const { return val; }
        template<typename T> T shfl_up(T val, unsigned) const { return val; }
    };
    inline thread_block this_thread_block() { return thread_block{}; }
    inline grid_group this_grid() { return grid_group{}; }
    inline multi_grid_group this_multi_grid() { return multi_grid_group{}; }
    inline coalesced_group coalesced_threads() { return coalesced_group{}; }
    template<int Sz>
    inline thread_block_tile<Sz> tiled_partition(const thread_block&) {
        return thread_block_tile<Sz>{};
    }
    // partition_copy stub: forwards to simple sequential copy (AST parse only)
    template<typename Group, typename InIt, typename OutIt1, typename OutIt2, typename Pred>
    inline OutIt1 partition_copy(const Group&, InIt first, InIt last,
                                  OutIt1 out_t, OutIt2 out_f, Pred pred) {
        for (; first != last; ++first) {
            if (pred(*first)) *out_t++ = *first; else *out_f++ = *first;
        }
        return out_t;
    }
    // inclusive_scan / exclusive_scan stubs for AST parsing
    template<typename Group, typename T, typename BinaryOp>
    inline T inclusive_scan(const Group&, T val, BinaryOp) { return val; }
    template<typename Group, typename T>
    inline T inclusive_scan(const Group&, T val) { return val; }
    template<typename Group, typename T, typename BinaryOp>
    inline T exclusive_scan(const Group&, T val, BinaryOp, T init = T{}) { (void)val; return init; }
    template<typename Group, typename T>
    inline T exclusive_scan(const Group&, T, T init = T{}) { return init; }
} // namespace cooperative_groups
)VGRE_STUB";

// ── Process-level caches shared across all ClangKernelParser instances ──────
// Prevents repeated llvm::json::parse() when different test instances parse
// the same kernel within the same binary run.
namespace {
static std::recursive_mutex& getProcessCacheMutex() {
    static std::recursive_mutex m;
    return m;
}

static std::unordered_map<std::string, EnhancedKernelIR>& getEnhancedIRCache() {
    static std::unordered_map<std::string, EnhancedKernelIR> cache;
    return cache;
}

// Return the vgre disk cache directory
static std::string getVgreCacheDir() {
    return vgre::common::getCacheRoot() + "/cache";
}

// Strip template arguments from a function/class name for flexible matching.
// Handles nested templates: "vector<pair<float,int>>" → "vector",
// "myKernel<float, 4>" → "myKernel".
// Finds the first '<' at depth 0 and returns everything before it.
static std::string stripTemplateArgs(const std::string& s) {
    // Walk character by character; the first unmatched '<' starts the template args.
    int depth = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '<') {
            if (depth == 0) return s.substr(0, i); // strip here
            ++depth;
        } else if (s[i] == '>') {
            if (depth > 0) --depth;
        }
    }
    return s; // no template arguments found
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
    obj["usesWarpShuffle"]            = ir.usesWarpShuffle;
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
    outIR.usesWarpShuffle            = obj->getBoolean("usesWarpShuffle").value_or(false);
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

    // Run clang++ via fork+execvp (POSIX) or popen (Windows) to get JSON AST.
    // Using execvp avoids shell injection: paths and flags are never interpreted
    // by a shell, so a kernel source file with backticks or $() in its path
    // cannot execute arbitrary commands.
    std::string includePath = vgre::common::findIncludeDir();
    std::string clangPath   = vgre::common::findCompilerPath();

    std::string result;
    int status = -1;

#if defined(_WIN32)
    {
      std::string cmd = "\"" + clangPath + "\" -Xclang -ast-dump=json -fsyntax-only -xc++ -w"
                        " -fno-delayed-template-parsing -I\"" + includePath + "\" \""
                        + tempPath + "\" 2>&1";
      char buffer[8192];
      FILE* pipe = _popen(cmd.c_str(), "r");
      if (pipe) {
          result.reserve(65536);
          while (fgets(buffer, sizeof(buffer), pipe)) result += buffer;
          status = _pclose(pipe);
      } else {
          VGRE_LOG_ERROR("ClangKernelParser", "popen failed");
          return "";
      }
    }
#else
    {
      // Pipe: parent reads child's stdout+stderr.
      int pipefds[2];
      if (::pipe(pipefds) != 0) {
          VGRE_LOG_ERROR("ClangKernelParser", "pipe() failed");
          return "";
      }
      pid_t pid = fork();
      if (pid == 0) {
          // Child: redirect stdout+stderr into write end of pipe.
          close(pipefds[0]);
          dup2(pipefds[1], STDOUT_FILENO);
          dup2(pipefds[1], STDERR_FILENO);
          close(pipefds[1]);
          const char* argv[] = {
            clangPath.c_str(),
            "-Xclang", "-ast-dump=json",
            "-fsyntax-only", "-xc++", "-w",
            "-fno-delayed-template-parsing",
            "-I", includePath.c_str(),
            tempPath.c_str(),
            nullptr
          };
          execvp(clangPath.c_str(), const_cast<char* const*>(argv));
          std::_Exit(127);
      } else if (pid > 0) {
          close(pipefds[1]);
          result.reserve(65536);
          char buffer[8192];
          ssize_t n;
          while ((n = read(pipefds[0], buffer, sizeof(buffer))) > 0)
              result.append(buffer, static_cast<size_t>(n));
          close(pipefds[0]);
          int wstatus = 0;
          waitpid(pid, &wstatus, 0);
          status = WIFEXITED(wstatus) ? WEXITSTATUS(wstatus) : 1;
      } else {
          VGRE_LOG_ERROR("ClangKernelParser", "fork() failed");
          return "";
      }
    }
#endif

    if (status != 0) {
        VGRE_LOG_ERROR("ClangKernelParser",
            "Clang AST dump failed (exit " + std::to_string(status) + "):\n" + result);
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

// ── PTX source detection and conversion ──────────────────────────────────────
// Detects raw PTX source (starts with .version / .target) and converts it to
// a minimal CUDA C++ kernel that the ClangKernelParser pipeline can compile.
// For each .param entry the type is mapped to a C++ equivalent; the kernel
// body is transliterated to valid C++ (PTX `ret` → `return`; register
// declarations are dropped; unrecognised instructions are commented out).

static bool isPTXSource(const std::string &src) {
    return src.find(".version") != std::string::npos &&
           src.find(".target")  != std::string::npos;
}

// Map a PTX scalar-type token (.u64, .s32, .f32, …) to a C++ type string.
// If the parameter carries a `.ptr` qualifier, return "void*".
static std::string ptxTypeToCpp(const std::string &ptxType, bool isPtr) {
    if (isPtr) return "void*";
    if (ptxType == ".u64" || ptxType == ".b64") return "unsigned long long";
    if (ptxType == ".s64")                       return "long long";
    if (ptxType == ".u32" || ptxType == ".b32") return "unsigned int";
    if (ptxType == ".s32")                       return "int";
    if (ptxType == ".u16" || ptxType == ".b16") return "unsigned short";
    if (ptxType == ".s16")                       return "short";
    if (ptxType == ".u8"  || ptxType == ".b8")  return "unsigned char";
    if (ptxType == ".s8")                        return "char";
    if (ptxType == ".f32")                       return "float";
    if (ptxType == ".f64")                       return "double";
    if (ptxType == ".pred")                      return "int";
    return "unsigned int";  // safe fallback
}

// Convert a full PTX source file to equivalent CUDA C++ for the named kernel.
// Returns empty string if the named .entry directive is not found.
static std::string convertPTXToCUDA(const std::string &ptxSrc,
                                     const std::string &kernelName,
                                     KernelIR &outIR) {
    // Locate the .entry directive for the requested kernel.
    std::string entryPat = ".entry " + kernelName;
    size_t entryPos = ptxSrc.find(entryPat);
    if (entryPos == std::string::npos) return "";

    size_t paramStart = ptxSrc.find('(', entryPos + entryPat.size());
    size_t paramEnd   = (paramStart != std::string::npos)
                            ? ptxSrc.find(')', paramStart + 1)
                            : std::string::npos;

    std::string paramListStr;
    if (paramStart != std::string::npos && paramEnd != std::string::npos)
        paramListStr = ptxSrc.substr(paramStart + 1, paramEnd - paramStart - 1);

    // Parse each ".param .TYPE [.ptr] name" entry.
    std::vector<std::string> cppParams;
    outIR.argTypes.clear();
    outIR.argTypeNames.clear();

    // Leaky static (never destroyed): runs on the JIT worker, which may execute
    // during process teardown while static locals are being destroyed.
    static const std::regex *kParamRe = new std::regex(
        R"(\.param\s+(\.\w+)(?:\s+\.align\s+\d+)?\s*(\.ptr\s+[.\w]+)?\s+(\w+))");
    auto pit = std::sregex_iterator(paramListStr.begin(), paramListStr.end(), *kParamRe);
    for (auto pend = std::sregex_iterator(); pit != pend; ++pit) {
        std::string scalarType = (*pit)[1].str();
        bool isPtr             = !(*pit)[2].str().empty();
        std::string pname      = (*pit)[3].str();
        std::string cppType    = ptxTypeToCpp(scalarType, isPtr);
        ArgType at = ArgType::UINT32;
        size_t  sz = 4;
        if (isPtr) {
            at = ArgType::POINTER; sz = sizeof(void*);
        } else if (scalarType == ".f32") {
            at = ArgType::FLOAT32; sz = 4;
        } else if (scalarType == ".f64") {
            at = ArgType::FLOAT64; sz = 8;
        } else if (scalarType == ".s32") {
            at = ArgType::INT32;   sz = 4;
        } else if (scalarType == ".s64") {
            at = ArgType::INT64;   sz = 8;
        } else if (scalarType == ".u64" || scalarType == ".b64") {
            at = ArgType::UINT64;  sz = 8;
        } else if (scalarType == ".u32" || scalarType == ".b32") {
            at = ArgType::UINT32;  sz = 4;
        } else if (scalarType == ".u16" || scalarType == ".b16" ||
                   scalarType == ".s16") {
            at = ArgType::UINT32;  sz = 2;
        } else if (scalarType == ".u8"  || scalarType == ".b8" ||
                   scalarType == ".s8") {
            at = ArgType::UINT32;  sz = 1;
        }
        cppParams.push_back(cppType + " " + pname);
        outIR.argTypeNames.push_back(cppType);
        outIR.argTypes.push_back(at);
        outIR.argSizes.push_back(sz);
    }

    // Locate the kernel body between the outer braces.
    size_t bodyOpen = (paramEnd != std::string::npos)
                          ? ptxSrc.find('{', paramEnd + 1)
                          : std::string::npos;
    if (bodyOpen == std::string::npos) return "";

    // Walk braces to find the matching close.
    size_t bodyClose = std::string::npos;
    int depth = 1;
    for (size_t i = bodyOpen + 1; i < ptxSrc.size(); ++i) {
        if (ptxSrc[i] == '{') ++depth;
        else if (ptxSrc[i] == '}') { if (--depth == 0) { bodyClose = i; break; } }
    }
    if (bodyClose == std::string::npos) return "";

    std::string ptxBody = ptxSrc.substr(bodyOpen + 1, bodyClose - bodyOpen - 1);

    // Translate the PTX body to C++ line by line.
    // .reg declarations are dropped; `ret` becomes `return;`; unrecognised
    // instructions are emitted as comments so the wrapper body is valid C++.
    std::ostringstream cppBody;
    std::istringstream bodyStream(ptxBody);
    std::string line;
    while (std::getline(bodyStream, line)) {
        // Strip leading/trailing whitespace.
        size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos) continue;
        std::string t = line.substr(s);
        if (t.empty() || t[0] == '/' || t[0] == ';') continue;
        if (t[0] == '.' && t.substr(0, 4) == ".reg") continue;  // drop register decls
        if (t[0] == '.' && t.substr(0, 5) == ".loc")  continue;  // drop debug info
        if (t == "ret;" || t == "ret") { cppBody << "    return;\n"; continue; }
        // Everything else: emit as a comment so compilation doesn't fail.
        cppBody << "    /* PTX: " << t << " */\n";
    }

    // Build the CUDA C++ kernel source.
    std::ostringstream cuda;
    cuda << "__global__ void " << kernelName << "(";
    for (size_t i = 0; i < cppParams.size(); ++i) {
        if (i) cuda << ", ";
        cuda << cppParams[i];
    }
    cuda << ") {\n" << cppBody.str() << "}\n";

    outIR.name   = kernelName;
    outIR.source = cuda.str();
    return cuda.str();
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

    // ── PTX fast path ─────────────────────────────────────────────────────────
    // Full PTX source (.version + .target) cannot be parsed as C++ by Clang.
    // Convert it directly to a CUDA C++ stub using PTX parameter extraction.
    if (isPTXSource(source)) {
        VGRE_LOG_INFO("ClangKernelParser",
                      "PTX source detected — converting to CUDA C++ for: " + name);
        KernelIR ptxIR;
        std::string cudaSrc = convertPTXToCUDA(source, name, ptxIR);
        if (cudaSrc.empty()) {
            VGRE_LOG_ERROR("ClangKernelParser",
                           "PTX kernel '" + name + "' not found in PTX source");
            return VGREResult::ERR_INVALID_KERNEL;
        }
        // Reparse the generated CUDA C++ through the normal ClangKernelParser
        // pipeline so that argTypes/argSizes are verified by Clang.
        VGREResult r2 = parse(name, cudaSrc, outIR);
        if (r2 != VGREResult::SUCCESS) {
            // Fallback: use the ptxIR we built directly (parameters already extracted).
            outIR = ptxIR;
        }
        // Store in memory cache keyed by the ORIGINAL PTX source.
        std::lock_guard<std::recursive_mutex> lock(cacheMutex_);
        cache_[name + ":::" + source] = outIR;
        return VGREResult::SUCCESS;
    }

    // Use the minimal analysis stub (NOT the full cpu_cuda_env.h) so that the
    // clang JSON AST stays small (~50 KB vs ~300 MB) — avoids OOM during parsing.
    std::string sourceWithHeader = std::string(kAstAnalysisStub) + source;
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
        // Kernel not found in this AST — the AST cache may have a hash
        // collision (different kernel source hashing to the same key).
        // Evict the stale AST entry and re-run Clang to get the correct AST.
        VGRE_LOG_WARN("ClangKernelParser",
                      "Kernel '" + name + "' not found in cached AST (hash: " +
                      sourceHash.substr(0, 8) + ") — evicting and re-parsing.");
        KernelCache::instance().evictAST(sourceHash);
        {
            std::lock_guard<std::recursive_mutex> lock(cacheMutex_);
            astCache_.erase(sourceHash);
        }
        // Re-run Clang (cache miss guaranteed after eviction above).
        jsonAst = runClangAstDump(sourceWithHeader);
        if (jsonAst.empty()) {
            VGRE_LOG_ERROR("ClangKernelParser", "Re-parse failed for '" + name + "'");
            return VGREResult::ERR_COMPILATION;
        }
        auto expectedAst2 = llvm::json::parse(jsonAst);
        if (!expectedAst2) return VGREResult::ERR_INVALID_KERNEL;
        const auto* root2 = expectedAst2->getAsObject();
        if (!root2) return VGREResult::ERR_INVALID_KERNEL;
        inner = root2->getArray("inner");
        if (!inner) return VGREResult::ERR_INVALID_KERNEL;
        kernelObj = findKernel(inner, name);
        if (!kernelObj) {
            VGRE_LOG_ERROR("ClangKernelParser",
                           "Kernel '" + name + "' not found even after fresh Clang parse");
            return VGREResult::ERR_INVALID_KERNEL;
        }
        // inner now points into expectedAst2 which is stack-allocated — safe
        // because we do not return or yield between here and the use below.
        // Store root2/expectedAst2 so they outlive the kernelObj pointer.
        // (expectedAst is still in scope and will be used if kernelObj came from it)
        expectedAst = std::move(expectedAst2);
        root = expectedAst->getAsObject();
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
        outIR.usesSyncthreads  = (source.find("__syncthreads") != std::string::npos);
        outIR.usesWarpShuffle  = (source.find("__shfl_sync") != std::string::npos
                               || source.find("__shfl_down_sync") != std::string::npos
                               || source.find("__shfl_up_sync") != std::string::npos
                               || source.find("__shfl_xor_sync") != std::string::npos
                               || source.find("__ballot_sync") != std::string::npos
                               || source.find("__shfl(") != std::string::npos
                               || source.find("__shfl_down(") != std::string::npos);

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


VGREResult ClangKernelParser::parseEnhanced(const std::string& name,
                                           const std::string& source,
                                           EnhancedKernelIR& outIR) {
    // Build cache keys up-front (needed by all cache tiers)
    std::string cacheKey = name + ":::" + source;
    std::string enhHash = std::to_string(std::hash<std::string>{}(cacheKey));
    std::string sourceWithHeader = std::string(kAstAnalysisStub) + source;

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
    outIR.argSizes        = basicIR.argSizes;
    outIR.usesSharedMem   = basicIR.usesSharedMem;
    outIR.usesSyncthreads = basicIR.usesSyncthreads;
    outIR.usesWarpShuffle = basicIR.usesWarpShuffle;
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
