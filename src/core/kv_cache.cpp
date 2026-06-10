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

} // namespace core
} // namespace vgre
