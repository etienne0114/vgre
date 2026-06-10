#ifndef VGRE_CORE_KV_CACHE_H
#define VGRE_CORE_KV_CACHE_H

// Track 18 — Paged KV-cache + paged attention (vLLM-style).
//
// The KV cache is stored in fixed-size physical BLOCKS (block_size tokens each),
// allocated from a pool by a free list. A sequence does not need contiguous
// memory: its tokens are scattered across whatever physical blocks were free,
// and a per-sequence BLOCK TABLE maps logical block index → physical block. This
// is exactly PagedAttention: it eliminates KV fragmentation so many sequences of
// different lengths share one pool, and a sequence grows one block at a time.

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace core {

using SeqId = uint64_t;

class KVCacheManager {
public:
    // Pool of numBlocks blocks; each block holds blockSize tokens, each token
    // carrying numHeads × headDim floats for K and the same for V.
    KVCacheManager(int numBlocks, int blockSize, int numHeads, int headDim);

    int blockSize() const { return blockSize_; }
    int numHeads()  const { return numHeads_; }
    int headDim()   const { return headDim_; }
    int freeBlocks() const { return static_cast<int>(freeList_.size()); }

    // Allocate / release a physical block from the pool. allocateBlock returns
    // -1 when the pool is exhausted.
    int  allocateBlock();
    void freeBlock(int physBlock);

    // Append one token to a sequence (K/V for ALL heads, each headDim floats,
    // laid out head-major). Grows the sequence's block table, allocating a new
    // block at a logical block boundary. Returns false if the pool is exhausted.
    bool appendToken(SeqId seq, const float* kAllHeads, const float* vAllHeads);

    int seqLen(SeqId seq) const;
    const std::vector<int>& blockTable(SeqId seq) const;

    // Read pointers into the pool for (physical block, slot, head).
    const float* keyPtr(int physBlock, int slot, int head) const;
    const float* valPtr(int physBlock, int slot, int head) const;

    // Return a finished sequence's blocks to the pool.
    void freeSequence(SeqId seq);

private:
    struct Seq { std::vector<int> blockTable; int length = 0; };

    int blockSize_, numHeads_, headDim_;
    int tokenStride_;   // floats per token (numHeads*headDim)
    int blockStride_;   // floats per block (blockSize*tokenStride)
    std::vector<float> kPool_;
    std::vector<float> vPool_;
    std::vector<int>   freeList_;
    std::unordered_map<SeqId, Seq> seqs_;
};

// Paged attention for one query row of one head, attending over `seq`'s cached
// K/V via its block table (online softmax). `q` and `out` are headDim floats.
// `causalUpTo` < 0 attends to all cached tokens; otherwise only tokens with
// index <= causalUpTo (the query's own position) are attended.
void pagedAttention(const float* q, int head, SeqId seq,
                    const KVCacheManager& kv, float scale, float* out,
                    int causalUpTo = -1);

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_KV_CACHE_H
