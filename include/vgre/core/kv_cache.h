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

// KV element storage type. Quantized modes shrink the paged pools with a per-head
// symmetric absmax scale (see vgre/core/kv_quant.h) — the same codec the generation
// path uses — for the ~3.8× (int8) / ~7× (int4) KV-memory reduction at long context.
// int4 needs an even headDim (2 codes/byte, head-aligned); an odd headDim downgrades
// the request to int8.
enum class KVDType { F32, I8, I4 };

class KVCacheManager {
public:
    // Pool of numBlocks blocks; each block holds blockSize tokens, each token
    // carrying numHeads × headDim K and V elements (fp32, or quantized per `dtype`).
    KVCacheManager(int numBlocks, int blockSize, int numHeads, int headDim,
                   KVDType dtype = KVDType::F32);

    int blockSize() const { return blockSize_; }
    int numHeads()  const { return numHeads_; }
    int headDim()   const { return headDim_; }
    int freeBlocks() const { return static_cast<int>(freeList_.size()); }
    KVDType dtype() const { return dtype_; }
    // Stored bytes per token for K and V together (one head-scale per head in the
    // quantized modes). Lets callers verify the memory reduction vs fp32.
    size_t bytesPerToken() const;

    // Allocate / release a physical block from the pool. allocateBlock returns
    // -1 when the pool is exhausted.
    int  allocateBlock();
    void freeBlock(int physBlock);

    // Append one token to a sequence (K/V for ALL heads, each headDim floats,
    // laid out head-major). Grows the sequence's block table, allocating a new
    // block at a logical block boundary. Returns false if the pool is exhausted.
    // In a quantized mode the K/V are quantized per head on the way in.
    bool appendToken(SeqId seq, const float* kAllHeads, const float* vAllHeads);

    int seqLen(SeqId seq) const;
    const std::vector<int>& blockTable(SeqId seq) const;

    // Read pointers into the pool for (physical block, slot, head). Valid only in
    // fp32 mode; nullptr in a quantized mode (use readKey / readVal instead).
    const float* keyPtr(int physBlock, int slot, int head) const;
    const float* valPtr(int physBlock, int slot, int head) const;

    // Dequantize (or copy, in fp32 mode) one head's K/V slice into `out` (headDim
    // floats). Works in every dtype — the read path pagedAttention uses.
    void readKey(int physBlock, int slot, int head, float* out) const;
    void readVal(int physBlock, int slot, int head, float* out) const;

    // Return a finished sequence's blocks to the pool.
    void freeSequence(SeqId seq);

private:
    struct Seq { std::vector<int> blockTable; int length = 0; };

    // Quantize (or copy) one token's K and V (all heads) into physical (block, slot).
    void writeToken(int physBlock, int slot, const float* kAllHeads, const float* vAllHeads);

    int blockSize_, numHeads_, headDim_;
    KVDType dtype_;
    int tokenStride_;   // elements per token (numHeads*headDim)
    int blockStride_;   // elements per block (blockSize*tokenStride)
    int packDim_;       // int4 bytes per head (headDim/2)
    // Exactly one representation is allocated (chosen by dtype_).
    std::vector<float>   kPool_,  vPool_;    // F32
    std::vector<int8_t>  kPool8_, vPool8_;   // I8 codes
    std::vector<uint8_t> kPool4_, vPool4_;   // I4 packed codes (2/byte)
    std::vector<float>   kScale_, vScale_;   // per-(block,slot,head) scale (I8/I4)
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

// ── Continuous-batching request scheduler (Track 22) ─────────────────────────
// vLLM-style in-flight batching over the paged KV-cache: requests of different
// lengths are admitted into a running batch as capacity frees up, advanced one
// token per scheduler step, and retired (their KV blocks reclaimed) the moment
// they finish — without waiting for the whole batch. New requests submitted
// mid-flight join at the next step.
struct Request {
    SeqId id = 0;
    int promptLen = 0;       // tokens prefilled on admission
    int maxNewTokens = 0;    // tokens to generate
    int generated = 0;       // generated so far
    enum State { WAITING, RUNNING, FINISHED } state = WAITING;
};

class ContinuousBatchScheduler {
public:
    ContinuousBatchScheduler(KVCacheManager& kv, int maxBatch);

    // Submit a request (queued WAITING). Token *content* is abstracted: the
    // scheduler models lifecycle + KV growth, appending zeroed K/V per token.
    void addRequest(SeqId id, int promptLen, int maxNewTokens);

    // Run one scheduling step:
    //   1. retire finished running requests (reclaim their KV blocks),
    //   2. admit waiting requests while batch slots AND KV blocks allow (prefill),
    //   3. advance each running request by one decode token (append to its KV).
    // Returns the number of requests still running after the step.
    int step();

    bool allDone() const { return waiting_.empty() && running_.empty(); }
    int runningCount()  const { return static_cast<int>(running_.size()); }
    int waitingCount()  const { return static_cast<int>(waiting_.size()); }
    int finishedCount() const { return finished_; }

private:
    int blocksFor(int tokens) const;        // ceil(tokens / blockSize)
    bool appendOneToken(SeqId id);          // append a zeroed K/V token to a seq

    KVCacheManager& kv_;
    int maxBatch_;
    std::vector<Request> waiting_;
    std::vector<Request> running_;
    int finished_ = 0;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_KV_CACHE_H
