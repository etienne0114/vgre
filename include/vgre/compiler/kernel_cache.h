#ifndef VGRE_KERNEL_CACHE_H
#define VGRE_KERNEL_CACHE_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include <list>
#include <string>
#include <mutex>
#include <unordered_map>

namespace vgre {
namespace compiler {

/**
 * Persistent disk-based cache for kernel AST and compilation results
 * Dramatically improves performance by avoiding repeated Clang invocations
 */
class KernelCache {
public:
    static KernelCache& instance();
    
    /**
     * Initialize cache with specified directory
     * @param cacheDir Directory to store cache files (default: ~/.vgre/cache)
     * @return SUCCESS if cache initialized
     */
    VGREResult initialize(const std::string& cacheDir = "");
    
    /**
     * Get cached AST JSON for a kernel source
     * @param sourceHash Hash of the kernel source
     * @param outAst Output AST JSON string
     * @return true if cache hit, false if cache miss
     */
    bool getAST(const std::string& sourceHash, std::string& outAst);
    
    /**
     * Store AST JSON in cache
     * @param sourceHash Hash of the kernel source
     * @param ast AST JSON string to cache
     */
    void putAST(const std::string& sourceHash, const std::string& ast);

    /**
     * Remove a cached AST entry (both memory and disk) so the next call to
     * getAST returns a miss and Clang is re-invoked with the current source.
     * Used to recover from hash-collision entries that contain the wrong AST.
     */
    void evictAST(const std::string& sourceHash);

    /**
     * Get cached Kernel IR metadata
     * @param sourceHash Hash of the kernel source
     * @param name Name of the kernel
     * @param outIr Output KernelIR struct
     * @return true if cache hit, false if cache miss
     */
    bool getKernelIR(const std::string& sourceHash, const std::string& name, KernelIR& outIr);

    /**
     * Store Kernel IR metadata in cache
     * @param sourceHash Hash of the kernel source
     * @param name Name of the kernel
     * @param ir KernelIR struct to cache
     */
    void putKernelIR(const std::string& sourceHash, const std::string& name, const KernelIR& ir);
    
    /**
     * Clear all cached data
     */
    void clear();
    
    /**
     * Get cache statistics
     */
    struct Stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t diskReads = 0;
        uint64_t diskWrites = 0;
        uint64_t memoryHits = 0;
        size_t cacheSize = 0;
    };
    
    Stats getStats() const;
    
private:
    KernelCache() = default;
    ~KernelCache() = default;
    KernelCache(const KernelCache&) = delete;
    KernelCache& operator=(const KernelCache&) = delete;
    
    std::string getCacheFilePath(const std::string& sourceHash, const std::string& ext = ".ast.json") const;
    
    std::string cacheDir_;
    mutable std::recursive_mutex mutex_;
    
    // In-memory LRU cache: lruOrder_ tracks access order (MRU at front).
    // memoryCache_ maps hash → (value, iterator into lruOrder_) for O(1) access.
    using LruList = std::list<std::string>;                      // keys in MRU order
    using LruMap  = std::unordered_map<std::string,
                        std::pair<std::string, LruList::iterator>>;
    LruList lruOrder_;
    LruMap  memoryCache_;

    using LruIRList = std::list<std::string>;
    using LruIRMap = std::unordered_map<std::string, std::pair<KernelIR, LruIRList::iterator>>;
    LruIRList lruIROrder_;
    LruIRMap memoryIRCache_;
    
    static constexpr size_t MAX_MEMORY_CACHE_SIZE = 100;
    
    // Statistics
    mutable Stats stats_;
};

} // namespace compiler
} // namespace vgre

#endif // VGRE_KERNEL_CACHE_H
