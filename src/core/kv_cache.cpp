#include "vgre/core/kv_cache.h"

#include <cmath>
#include <cstring>

namespace vgre {
namespace core {

KVCacheManager::KVCacheManager(int numBlocks, int blockSize, int numHeads, int headDim)
    : blockSize_(blockSize), numHeads_(numHeads), headDim_(headDim),
      tokenStride_(numHeads * headDim),
      blockStride_(blockSize * numHeads * headDim) {
    kPool_.assign(static_cast<size_t>(numBlocks) * blockStride_, 0.0f);
    vPool_.assign(static_cast<size_t>(numBlocks) * blockStride_, 0.0f);
    freeList_.reserve(numBlocks);
    for (int i = numBlocks - 1; i >= 0; --i) freeList_.push_back(i);  // pop from back = ascending
}

int KVCacheManager::allocateBlock() {
    if (freeList_.empty()) return -1;
    int b = freeList_.back();
    freeList_.pop_back();
    return b;
}

void KVCacheManager::freeBlock(int physBlock) {
    if (physBlock >= 0) freeList_.push_back(physBlock);
}

const float* KVCacheManager::keyPtr(int physBlock, int slot, int head) const {
    return &kPool_[(static_cast<size_t>(physBlock) * blockSize_ + slot) * tokenStride_ +
                   static_cast<size_t>(head) * headDim_];
}
const float* KVCacheManager::valPtr(int physBlock, int slot, int head) const {
    return &vPool_[(static_cast<size_t>(physBlock) * blockSize_ + slot) * tokenStride_ +
                   static_cast<size_t>(head) * headDim_];
}

bool KVCacheManager::appendToken(SeqId seq, const float* kAllHeads, const float* vAllHeads) {
    Seq& s = seqs_[seq];
    const int slot = s.length % blockSize_;
    if (slot == 0) {                       // crossing into a new logical block
        int b = allocateBlock();
        if (b < 0) return false;           // pool exhausted
        s.blockTable.push_back(b);
    }
    const int physBlock = s.blockTable.back();
    float* kdst = &kPool_[(static_cast<size_t>(physBlock) * blockSize_ + slot) * tokenStride_];
    float* vdst = &vPool_[(static_cast<size_t>(physBlock) * blockSize_ + slot) * tokenStride_];
    std::memcpy(kdst, kAllHeads, sizeof(float) * tokenStride_);
    std::memcpy(vdst, vAllHeads, sizeof(float) * tokenStride_);
    ++s.length;
    return true;
}

int KVCacheManager::seqLen(SeqId seq) const {
    auto it = seqs_.find(seq);
    return it == seqs_.end() ? 0 : it->second.length;
}

const std::vector<int>& KVCacheManager::blockTable(SeqId seq) const {
    static const std::vector<int> kEmpty;
    auto it = seqs_.find(seq);
    return it == seqs_.end() ? kEmpty : it->second.blockTable;
}

void KVCacheManager::freeSequence(SeqId seq) {
    auto it = seqs_.find(seq);
    if (it == seqs_.end()) return;
    for (int b : it->second.blockTable) freeBlock(b);
    seqs_.erase(it);
}

void pagedAttention(const float* q, int head, SeqId seq,
                    const KVCacheManager& kv, float scale, float* out,
                    int causalUpTo) {
    const int d   = kv.headDim();
    const int bs  = kv.blockSize();
    const int len = kv.seqLen(seq);
    const auto& table = kv.blockTable(seq);

    for (int i = 0; i < d; ++i) out[i] = 0.0f;
    float runMax = -3.402823466e+38f, runSum = 0.0f;

    const int last = (causalUpTo < 0 || causalUpTo >= len) ? len - 1 : causalUpTo;
    for (int t = 0; t <= last; ++t) {
        const int logical = t / bs, slot = t % bs;
        if (logical >= static_cast<int>(table.size())) break;
        const int physBlock = table[logical];
        const float* k = kv.keyPtr(physBlock, slot, head);

        float dot = 0.0f;
        for (int i = 0; i < d; ++i) dot += q[i] * k[i];
        const float s = dot * scale;

        // Online softmax: rescale accumulator once for the new running max, add.
        const float newMax = (runMax > s) ? runMax : s;
        const float resc    = std::exp(runMax - newMax);
        const float e       = std::exp(s - newMax);
        for (int i = 0; i < d; ++i) out[i] *= resc;
        runSum = runSum * resc + e;
        const float* v = kv.valPtr(physBlock, slot, head);
        for (int i = 0; i < d; ++i) out[i] += e * v[i];
        runMax = newMax;
    }
    if (runSum > 0.0f)
        for (int i = 0; i < d; ++i) out[i] /= runSum;
}

// ── Continuous-batching scheduler ────────────────────────────────────────────

ContinuousBatchScheduler::ContinuousBatchScheduler(KVCacheManager& kv, int maxBatch)
    : kv_(kv), maxBatch_(maxBatch < 1 ? 1 : maxBatch) {}

void ContinuousBatchScheduler::addRequest(SeqId id, int promptLen, int maxNewTokens) {
    Request r;
    r.id = id; r.promptLen = promptLen; r.maxNewTokens = maxNewTokens;
    r.state = Request::WAITING;
    waiting_.push_back(r);
}

int ContinuousBatchScheduler::blocksFor(int tokens) const {
    const int bs = kv_.blockSize();
    return (tokens + bs - 1) / bs;
}

bool ContinuousBatchScheduler::appendOneToken(SeqId id) {
    // Token content is abstracted: append a zeroed K/V token (real serving would
    // append the layer's K/V). Returns false if the pool is exhausted.
    static thread_local std::vector<float> zero;
    const int stride = kv_.numHeads() * kv_.headDim();
    if (static_cast<int>(zero.size()) < stride) zero.assign(stride, 0.0f);
    return kv_.appendToken(id, zero.data(), zero.data());
}

int ContinuousBatchScheduler::step() {
    // 1. Retire finished running requests; reclaim their KV blocks.
    {
        std::vector<Request> stillRunning;
        stillRunning.reserve(running_.size());
        for (auto& r : running_) {
            if (r.generated >= r.maxNewTokens) {
                r.state = Request::FINISHED;
                kv_.freeSequence(r.id);   // return blocks to the pool
                ++finished_;
            } else {
                stillRunning.push_back(r);
            }
        }
        running_.swap(stillRunning);
    }

    // 2. Admit waiting requests while batch slots AND KV blocks allow. Prefill
    //    the prompt (block reservation needs ceil((promptLen+maxNew)/blockSize)
    //    in the worst case, but we admit on the prompt cost and let decode grow).
    {
        std::vector<Request> stillWaiting;
        for (auto& r : waiting_) {
            if (static_cast<int>(running_.size()) < maxBatch_ &&
                kv_.freeBlocks() >= blocksFor(r.promptLen > 0 ? r.promptLen : 1)) {
                // Prefill the prompt tokens into the KV cache.
                bool ok = true;
                for (int t = 0; t < r.promptLen && ok; ++t) ok = appendOneToken(r.id);
                if (ok) { r.state = Request::RUNNING; running_.push_back(r); }
                else    { kv_.freeSequence(r.id); stillWaiting.push_back(r); } // roll back, retry later
            } else {
                stillWaiting.push_back(r);
            }
        }
        waiting_.swap(stillWaiting);
    }

    // 3. Advance each running request by one decode token.
    for (auto& r : running_) {
        if (appendOneToken(r.id)) ++r.generated;
        // If the pool is momentarily exhausted the request simply does not
        // advance this step; it will progress once a finished request frees space.
    }
    return static_cast<int>(running_.size());
}

} // namespace core
} // namespace vgre
