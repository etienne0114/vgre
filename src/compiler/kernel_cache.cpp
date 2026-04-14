#include "vgre/compiler/kernel_cache.h"
#include "vgre/common/logger.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#else
#include <unistd.h>
#include <pwd.h>
#endif

namespace vgre {
namespace compiler {

KernelCache &KernelCache::instance() {
  static KernelCache* instance = new KernelCache();
  return *instance;
}

VGREResult KernelCache::initialize(const std::string& cacheDir) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    
    if (!cacheDir.empty()) {
        cacheDir_ = cacheDir;
    } else {
        // Use default cache directory: ~/.vgre/cache
#ifdef _WIN32
        char path[MAX_PATH];
        if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_PROFILE, NULL, 0, path))) {
            cacheDir_ = std::string(path) + "\\.vgre\\cache";
        } else {
            cacheDir_ = "C:\\vgre_cache";
        }
#else
        const char* home = getenv("HOME");
        if (!home) {
            struct passwd* pw = getpwuid(getuid());
            if (pw) {
                home = pw->pw_dir;
            }
        }
        if (home) {
            cacheDir_ = std::string(home) + "/.vgre/cache";
        } else {
            cacheDir_ = "/tmp/vgre_cache";
        }
#endif
    }
    
    // Create cache directory if it doesn't exist
    try {
        std::filesystem::create_directories(cacheDir_);
        VGRE_LOG_INFO("KernelCache", "Initialized cache directory: " + cacheDir_);
    } catch (const std::exception& e) {
        VGRE_LOG_ERROR("KernelCache", "Failed to create cache directory: " + std::string(e.what()));
        return VGREResult::ERR_IO;
    }
    
    return VGREResult::SUCCESS;
}

std::string KernelCache::getCacheFilePath(const std::string& sourceHash, const std::string& ext) const {
    // Use first 2 chars of hash for subdirectory (reduces files per directory)
    std::string subdir = sourceHash.substr(0, 2);
    std::string cachePath = cacheDir_ + "/" + subdir;
    
    // Create subdirectory if needed
    try {
        std::filesystem::create_directories(cachePath);
    } catch (...) {
        // Ignore errors, will fail on file write if needed
    }
    
    return cachePath + "/" + sourceHash + ext;
}

bool KernelCache::getAST(const std::string& sourceHash, std::string& outAst) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    
    VGRE_LOG_DEBUG("KernelCache", "Looking up cache for hash: " + sourceHash.substr(0, 16) + "...");
    
    // Check memory cache first (fastest)
    auto it = memoryCache_.find(sourceHash);
    if (it != memoryCache_.end()) {
        outAst = it->second;
        stats_.hits++;
        stats_.memoryHits++;
        VGRE_LOG_INFO("KernelCache", "✓ Memory cache HIT for hash: " + sourceHash.substr(0, 8) + "...");
        return true;
    }
    
    // Check disk cache
    std::string cachePath = getCacheFilePath(sourceHash);
    VGRE_LOG_DEBUG("KernelCache", "Checking disk cache at: " + cachePath);
    
    std::ifstream file(cachePath, std::ios::binary);
    if (!file.is_open()) {
        stats_.misses++;
        VGRE_LOG_INFO("KernelCache", "✗ Cache MISS for hash: " + sourceHash.substr(0, 8) + "... (file not found)");
        return false;
    }
    
    // Read cached AST
    std::stringstream buffer;
    buffer << file.rdbuf();
    outAst = buffer.str();
    file.close();
    
    if (outAst.empty()) {
        stats_.misses++;
        VGRE_LOG_WARN("KernelCache", "✗ Cache file empty for hash: " + sourceHash.substr(0, 8) + "...");
        return false;
    }
    
    // Add to memory cache for future hits
    if (memoryCache_.size() < MAX_MEMORY_CACHE_SIZE) {
        memoryCache_[sourceHash] = outAst;
    }
    
    stats_.hits++;
    stats_.diskReads++;
    VGRE_LOG_INFO("KernelCache", "✓ Disk cache HIT for hash: " + sourceHash.substr(0, 8) + "... (size: " + std::to_string(outAst.size()) + " bytes)");
    return true;
}

void KernelCache::putAST(const std::string& sourceHash, const std::string& ast) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    
    VGRE_LOG_DEBUG("KernelCache", "Storing AST for hash: " + sourceHash.substr(0, 16) + "... (size: " + std::to_string(ast.size()) + " bytes)");
    
    // Add to memory cache
    if (memoryCache_.size() < MAX_MEMORY_CACHE_SIZE) {
        memoryCache_[sourceHash] = ast;
    } else {
        // Evict oldest entry (simple FIFO)
        auto it = memoryCache_.begin();
        memoryCache_.erase(it);
        memoryCache_[sourceHash] = ast;
    }
    
    // Write to disk cache
    std::string cachePath = getCacheFilePath(sourceHash);
    VGRE_LOG_DEBUG("KernelCache", "Writing cache file: " + cachePath);
    
    std::ofstream file(cachePath, std::ios::binary);
    if (!file.is_open()) {
        VGRE_LOG_ERROR("KernelCache", "Failed to write cache file: " + cachePath);
        return;
    }
    
    file << ast;
    file.close();
    
    stats_.diskWrites++;
    stats_.cacheSize++;
    VGRE_LOG_INFO("KernelCache", "✓ Cached AST for hash: " + sourceHash.substr(0, 8) + "... at " + cachePath);
}

void KernelCache::clear() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    
    memoryCache_.clear();
    
    // Clear disk cache
    try {
        std::filesystem::remove_all(cacheDir_);
        std::filesystem::create_directories(cacheDir_);
        VGRE_LOG_INFO("KernelCache", "Cache cleared");
    } catch (const std::exception& e) {
        VGRE_LOG_ERROR("KernelCache", "Failed to clear cache: " + std::string(e.what()));
    }
    
    stats_ = Stats();
}

KernelCache::Stats KernelCache::getStats() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return stats_;
}

bool KernelCache::getKernelIR(const std::string& sourceHash, const std::string& name, KernelIR& outIr) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    
    // IR cache is specific to kernel name as well as source hash
    std::string irHash = sourceHash + "_" + name;
    std::string cachePath = getCacheFilePath(irHash, ".ir.json");
    
    std::ifstream file(cachePath);
    if (!file.is_open()) return false;
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string jsonStr = buffer.str();
    
    auto expected = llvm::json::parse(jsonStr);
    if (!expected) return false;
    
    const auto* obj = expected->getAsObject();
    if (!obj) {
        VGRE_LOG_ERROR("KernelCache", "Cached IR is not a JSON object");
        return false;
    }
    
    outIr.name = obj->getString("name").value_or("").str();
    outIr.source = obj->getString("source").value_or("").str();
    outIr.irCode = obj->getString("irCode").value_or("").str();
    VGRE_LOG_DEBUG("KernelCache", "Loaded basic IR metadata for " + outIr.name);
    
    // Deserialize argTypes
    if (const auto* types = obj->getArray("argTypes")) {
        for (const auto& v : *types) {
            outIr.argTypes.push_back(static_cast<ArgType>(v.getAsInteger().value_or(0)));
        }
    }
    VGRE_LOG_DEBUG("KernelCache", "Deserialized argTypes (" + std::to_string(outIr.argTypes.size()) + ")");
    
    if (const auto* names = obj->getArray("argTypeNames")) {
        for (const auto& v : *names) {
            outIr.argTypeNames.push_back(v.getAsString().value_or("").str());
        }
    }
    VGRE_LOG_DEBUG("KernelCache", "Deserialized argTypeNames");
    
    if (const auto* sizes = obj->getArray("argSizes")) {
        for (const auto& v : *sizes) {
            outIr.argSizes.push_back(static_cast<size_t>(v.getAsInteger().value_or(0)));
        }
    }
    VGRE_LOG_DEBUG("KernelCache", "Deserialized argSizes");
    
    outIr.usesSharedMem = obj->getBoolean("usesSharedMem").value_or(false);
    outIr.usesSyncthreads = obj->getBoolean("usesSyncthreads").value_or(false);
    outIr.sharedMemSize = static_cast<size_t>(obj->getInteger("sharedMemSize").value_or(0));
    outIr.estimatedInstructionCount = static_cast<uint64_t>(obj->getInteger("estimatedInstructionCount").value_or(0));
    outIr.estimatedMemoryAccessCount = static_cast<uint64_t>(obj->getInteger("estimatedMemoryAccessCount").value_or(0));
    outIr.staticFlopCount = static_cast<uint64_t>(obj->getInteger("staticFlopCount").value_or(0));
    VGRE_LOG_DEBUG("KernelCache", "Deserialized metrics");

    VGRE_LOG_INFO("KernelCache", "✓ IR cache HIT for " + name + " (hash: " + sourceHash.substr(0,8) + ")");
    return true;
}

void KernelCache::putKernelIR(const std::string& sourceHash, const std::string& name, const KernelIR& ir) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    
    llvm::json::Object obj;
    obj["name"] = ir.name;
    obj["source"] = ir.source;
    obj["irCode"] = ir.irCode;
    
    llvm::json::Array types;
    for (auto t : ir.argTypes) types.push_back(static_cast<int64_t>(t));
    obj["argTypes"] = std::move(types);
    
    llvm::json::Array names;
    for (const auto& n : ir.argTypeNames) names.push_back(n);
    obj["argTypeNames"] = std::move(names);
    
    llvm::json::Array sizes;
    for (auto s : ir.argSizes) sizes.push_back(static_cast<int64_t>(s));
    obj["argSizes"] = std::move(sizes);
    
    obj["usesSharedMem"] = ir.usesSharedMem;
    obj["usesSyncthreads"] = ir.usesSyncthreads;
    obj["sharedMemSize"] = static_cast<int64_t>(ir.sharedMemSize);
    obj["estimatedInstructionCount"] = static_cast<int64_t>(ir.estimatedInstructionCount);
    obj["estimatedMemoryAccessCount"] = static_cast<int64_t>(ir.estimatedMemoryAccessCount);
    obj["staticFlopCount"] = static_cast<int64_t>(ir.staticFlopCount);
    
    std::string irHash = sourceHash + "_" + name;
    std::string cachePath = getCacheFilePath(irHash, ".ir.json");
    
    std::ofstream file(cachePath);
    if (file.is_open()) {
        file << llvm::formatv("{0:2}", llvm::json::Value(std::move(obj))).str();
        file.close();
        VGRE_LOG_INFO("KernelCache", "✓ Cached IR for " + name + " (hash: " + sourceHash.substr(0,8) + ")");
    }
}

} // namespace compiler
} // namespace vgre
