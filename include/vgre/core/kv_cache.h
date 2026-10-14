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

#include <cstddef>   // size_t (not transitively guaranteed on older libstdc++)
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    // Invalid dimensions and capacities that overflow the supported indexing
    // range throw std::invalid_argument or std::length_error.
    KVCacheManager(int numBlocks, int blockSize, int numHeads, int headDim,
                   KVDType dtype = KVDType::F32);

    int blockSize() const { return blockSize_; }
    int totalBlocks() const { return static_cast<int>(blockState_.size()); }
    int numHeads()  const { return numHeads_; }
    int headDim()   const { return headDim_; }
    int freeBlocks() const { return static_cast<int>(freeList_.size()); }
    KVDType dtype() const { return dtype_; }
    // Stored bytes per token for K and V together (one head-scale per head in the
    // quantized modes). Lets callers verify the memory reduction vs fp32.
    size_t bytesPerToken() const;

    // Allocate / release a physical block from the pool. allocateBlock returns
    // -1 when the pool is exhausted. Invalid, duplicate, and sequence-owned
    // releases are ignored; sequence-owned blocks are released by freeSequence.
    int  allocateBlock();
    void freeBlock(int physBlock);

    // Append one token to a sequence (K/V for ALL heads, each headDim floats,
    // laid out head-major). Grows the sequence's block table, allocating a new
    // block at a logical block boundary. Returns false if the pool is exhausted.
    // In a quantized mode the K/V are quantized per head on the way in.
    bool appendToken(SeqId seq, const float* kAllHeads, const float* vAllHeads);

    // Reserve enough physical blocks for `tokenCapacity` tokens without
    // changing the logical sequence length. Existing tokens and reservations
    // are retained. Returns false without reserving anything when the pool
    // cannot satisfy the full additional capacity.
    bool reserveSequenceCapacity(SeqId seq, int tokenCapacity);

    int seqLen(SeqId seq) const;
    bool hasSequence(SeqId seq) const;
    const std::vector<int>& blockTable(SeqId seq) const;

    // Read pointers into the pool for (physical block, slot, head). Valid only in
    // fp32 mode; nullptr in a quantized mode or for invalid indices (use
    // readKey / readVal instead).
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
    void releaseBlock(int physBlock, bool sequenceRelease);

    int blockSize_, numHeads_, headDim_;
    KVDType dtype_;
    size_t tokenStride_;   // elements per token (numHeads*headDim)
    size_t blockStride_;   // elements per block (blockSize*tokenStride)
    int packDim_;       // int4 bytes per head (headDim/2)
    // Exactly one representation is allocated (chosen by dtype_).
    std::vector<float>   kPool_,  vPool_;    // F32
    std::vector<int8_t>  kPool8_, vPool8_;   // I8 codes
    std::vector<uint8_t> kPool4_, vPool4_;   // I4 packed codes (2/byte)
    std::vector<float>   kScale_, vScale_;   // per-(block,slot,head) scale (I8/I4)
    std::vector<int>   freeList_;
    std::vector<uint8_t> blockState_;         // free/raw/sequence; O(1) release validation
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
// In-flight batching over the paged KV-cache: requests of different lengths
// are admitted as capacity frees up, run through the caller's model executor,
// and retired as soon as they finish. New requests submitted mid-flight join
// at the next step. The scheduler manages lifecycle and cache capacity; it
// does not invent token contents or perform model inference itself.
struct Request {
    SeqId id = 0;
    int promptLen = 0;       // tokens prefilled on admission
    int maxNewTokens = 0;    // tokens to generate
    int generated = 0;       // generated so far
    enum State { WAITING, RUNNING, FINISHED } state = WAITING;
};

enum class ScheduledTokenPhase { PREFILL, DECODE };
enum class SchedulerError {
    NONE,
    MISSING_EXECUTOR,
    EXECUTOR_FAILURE,
    INVALID_KV_OUTPUT,
    CACHE_APPEND_FAILURE,
    CACHE_RESERVATION_FAILURE,
    REENTRANT_STEP
};

class ContinuousBatchScheduler {
public:
    // The caller runs the model for one position and fills K/V for every head
    // (head-major, numHeads*headDim floats each). Request IDs identify the
    // caller's token/model state. `position` is the absolute sequence position.
    // Executor failures are caught at the shared-library boundary and reported
    // by step()/lastError(); callbacks should be safe to retry at the same
    // request/position.
    using TokenExecutor = std::function<void(const Request&, ScheduledTokenPhase,
                                             int position,
                                             std::vector<float>& keys,
                                             std::vector<float>& values)>;

    ContinuousBatchScheduler(KVCacheManager& kv, int maxBatch);

    // A scheduler must have a real executor before stepping requests that
    // contain prompt or generated tokens. Passing an empty function clears it.
    void setTokenExecutor(TokenExecutor executor);

    // Submit a request (queued WAITING). The caller's executor resolves prompt
    // tokens and model state from the request ID. Negative lengths and
    // duplicate active IDs throw std::invalid_argument.
    void addRequest(SeqId id, int promptLen, int maxNewTokens);

    // Cancel a waiting or running request at a scheduler safe point (between
    // step() calls). Waiting removal updates the capacity index in O(log B)
    // and the request-ID index in O(log W); running removal releases its
    // reserved KV blocks and compacts the active vector in O(R). Returns false
    // for an unknown/terminal ID or while step() is executing. The scheduler is
    // not thread-safe; callers must serialize add/cancel/step operations.
    bool cancelRequest(SeqId id);

    // Run one scheduling step:
    //   1. retire finished running requests (reclaim their KV blocks),
    //   2. admit the oldest waiting request that fits the available KV blocks
    //      while batch slots remain, then execute prompt prefill,
    //   3. execute one decode position for every running request.
    // Returns the number of requests still running after the step, or -1 when
    // token execution failed. lastError()/lastErrorMessage() describe that
    // failure; failed tokens do not append K/V. Callback exceptions are caught
    // here so C++ runtime exception types do not cross a shared-library ABI.
    int step();

    SchedulerError lastError() const { return lastError_; }
    const std::string& lastErrorMessage() const { return lastErrorMessage_; }

    bool allDone() const { return waitingCount_ == 0 && running_.empty(); }
    int runningCount()  const { return static_cast<int>(running_.size()); }
    int waitingCount()  const { return static_cast<int>(waitingCount_); }
    int finishedCount() const { return finished_; }
    int cancelledCount() const { return cancelled_; }

private:
    struct QueuedRequest {
        Request request;
        uint64_t order;
    };
    using WaitingQueue = std::list<QueuedRequest>;
    struct WaitingLocation {
        size_t bucket;
        WaitingQueue::iterator position;
        bool zeroTokenRequest;
    };

    int blocksFor(int tokens) const;        // ceil(tokens / blockSize)
    size_t olderWaitingBucket(size_t lhs, size_t rhs) const;
    void updateWaitingBucket(size_t bucket);
    size_t oldestEligibleWaitingBucket(int availableBlocks,
                                       uint64_t beforeOrder) const;
    bool executeAndAppend(const Request& request, ScheduledTokenPhase phase,
                          int position);

    KVCacheManager& kv_;
    int maxBatch_;
    TokenExecutor tokenExecutor_;
    std::vector<float> keyScratch_;
    std::vector<float> valueScratch_;
    SchedulerError lastError_ = SchedulerError::NONE;
    std::string lastErrorMessage_;
    // Requests are grouped by their full-cache block requirement. The segment
    // tree stores the oldest order in each subtree, so a prefix query finds the
    // oldest request that fits in O(log B), where B is the cache block count.
    std::vector<std::unique_ptr<WaitingQueue>> waitingByBlocks_;
    std::map<SeqId, WaitingLocation> waitingLocations_;
    std::vector<size_t> waitingTree_;
    size_t waitingTreeBase_ = 0;
    size_t waitingCount_ = 0;
    size_t zeroTokenWaitingCount_ = 0;
    uint64_t nextWaitingOrder_ = 0;
    std::vector<Request> running_;
    std::unordered_set<SeqId> activeRequestIds_;
    bool inStep_ = false;
    int finished_ = 0;
    int cancelled_ = 0;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_KV_CACHE_H
