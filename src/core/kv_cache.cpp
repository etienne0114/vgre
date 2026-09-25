#include "vgre/core/kv_cache.h"
#include "vgre/core/kv_quant.h"

#include <cmath>
#include <cstring>

namespace vgre {
namespace core {

KVCacheManager::KVCacheManager(int numBlocks, int blockSize, int numHeads, int headDim,
                               KVDType dtype)
    : blockSize_(blockSize), numHeads_(numHeads), headDim_(headDim),
      dtype_(dtype),
      tokenStride_(numHeads * headDim),
      blockStride_(blockSize * numHeads * headDim),
      packDim_(headDim / 2) {
    // int4 needs an even headDim (2 codes/byte, head-aligned); otherwise use int8.
    if (dtype_ == KVDType::I4 && (headDim & 1)) dtype_ = KVDType::I8;
    const size_t nBlk = static_cast<size_t>(numBlocks);
    const size_t nSlotHead = nBlk * blockSize_ * numHeads_;   // one scale per (block,slot,head)
    if (dtype_ == KVDType::I8) {
        kPool8_.assign(nBlk * blockStride_, 0);
        vPool8_.assign(nBlk * blockStride_, 0);
        kScale_.assign(nSlotHead, 0.0f);
        vScale_.assign(nSlotHead, 0.0f);
    } else if (dtype_ == KVDType::I4) {
        kPool4_.assign(nBlk * blockSize_ * numHeads_ * packDim_, 0);
        vPool4_.assign(nBlk * blockSize_ * numHeads_ * packDim_, 0);
        kScale_.assign(nSlotHead, 0.0f);
        vScale_.assign(nSlotHead, 0.0f);
    } else {
        kPool_.assign(nBlk * blockStride_, 0.0f);
        vPool_.assign(nBlk * blockStride_, 0.0f);
    }
    freeList_.reserve(numBlocks);
    for (int i = numBlocks - 1; i >= 0; --i) freeList_.push_back(i);  // pop from back = ascending
}

size_t KVCacheManager::bytesPerToken() const {
    const size_t scaleBytes = static_cast<size_t>(numHeads_) * sizeof(float);   // per head, K and V each
    switch (dtype_) {
        case KVDType::I8: return 2 * (static_cast<size_t>(tokenStride_) * sizeof(int8_t)  + scaleBytes);
        case KVDType::I4: return 2 * (static_cast<size_t>(numHeads_) * packDim_          + scaleBytes);
        default:          return 2 * (static_cast<size_t>(tokenStride_) * sizeof(float));
    }
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
    if (dtype_ != KVDType::F32) return nullptr;   // quantized: no fp32 to point at — use readKey
    return &kPool_[(static_cast<size_t>(physBlock) * blockSize_ + slot) * tokenStride_ +
                   static_cast<size_t>(head) * headDim_];
}
const float* KVCacheManager::valPtr(int physBlock, int slot, int head) const {
    if (dtype_ != KVDType::F32) return nullptr;
    return &vPool_[(static_cast<size_t>(physBlock) * blockSize_ + slot) * tokenStride_ +
                   static_cast<size_t>(head) * headDim_];
}

// Dequantize (or copy) one head's slice into `out` (headDim floats). `q8`/`q4`/`f32`
// select the pool; `scale` the per-head scale pool; identical for K and V.
namespace {
void readSlice(KVDType dt, const std::vector<float>& f32,
               const std::vector<int8_t>& q8, const std::vector<uint8_t>& q4,
               const std::vector<float>& scale, size_t slotHead, size_t elemBase,
               size_t packBase, int headDim, float* out) {
    if (dt == KVDType::F32) {
        std::memcpy(out, &f32[elemBase], sizeof(float) * headDim);
    } else if (dt == KVDType::I8) {
        const float sc = scale[slotHead];
        for (int i = 0; i < headDim; ++i) out[i] = kvDequantInt8(&q8[elemBase], i, sc);
    } else {
        const float sc = scale[slotHead];
        const uint8_t* row = &q4[packBase];
        for (int i = 0; i < headDim; ++i) out[i] = kvDequantInt4(row, i, sc);
    }
}
}  // namespace

void KVCacheManager::readKey(int physBlock, int slot, int head, float* out) const {
    const size_t st = static_cast<size_t>(physBlock) * blockSize_ + slot;
    readSlice(dtype_, kPool_, kPool8_, kPool4_, kScale_,
              st * numHeads_ + head, st * tokenStride_ + static_cast<size_t>(head) * headDim_,
              (st * numHeads_ + head) * packDim_, headDim_, out);
}
void KVCacheManager::readVal(int physBlock, int slot, int head, float* out) const {
    const size_t st = static_cast<size_t>(physBlock) * blockSize_ + slot;
    readSlice(dtype_, vPool_, vPool8_, vPool4_, vScale_,
              st * numHeads_ + head, st * tokenStride_ + static_cast<size_t>(head) * headDim_,
              (st * numHeads_ + head) * packDim_, headDim_, out);
}

void KVCacheManager::writeToken(int physBlock, int slot, const float* kAllHeads, const float* vAllHeads) {
    const size_t st = static_cast<size_t>(physBlock) * blockSize_ + slot;
    if (dtype_ == KVDType::F32) {
        std::memcpy(&kPool_[st * tokenStride_], kAllHeads, sizeof(float) * tokenStride_);
        std::memcpy(&vPool_[st * tokenStride_], vAllHeads, sizeof(float) * tokenStride_);
        return;
    }
    // Quantize each head slice with its own absmax scale (K and V independently).
    for (int h = 0; h < numHeads_; ++h) {
        const size_t sh = st * numHeads_ + h;
        const float* ksrc = kAllHeads + static_cast<size_t>(h) * headDim_;
        const float* vsrc = vAllHeads + static_cast<size_t>(h) * headDim_;
        if (dtype_ == KVDType::I8) {
            kvQuantInt8(ksrc, &kPool8_[st * tokenStride_ + static_cast<size_t>(h) * headDim_], kScale_[sh], headDim_);
            kvQuantInt8(vsrc, &vPool8_[st * tokenStride_ + static_cast<size_t>(h) * headDim_], vScale_[sh], headDim_);
        } else {  // I4 — each head owns its own packDim_ bytes, so off=0 is head-aligned
            kvQuantInt4(ksrc, &kPool4_[sh * packDim_], kScale_[sh], 0, headDim_);
            kvQuantInt4(vsrc, &vPool4_[sh * packDim_], vScale_[sh], 0, headDim_);
        }
    }
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
    writeToken(physBlock, slot, kAllHeads, vAllHeads);
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

    const bool f32 = kv.dtype() == KVDType::F32;
    std::vector<float> kbuf, vbuf;   // dequant scratch (quantized modes only)
    if (!f32) { kbuf.resize(d); vbuf.resize(d); }

    const int last = (causalUpTo < 0 || causalUpTo >= len) ? len - 1 : causalUpTo;
    for (int t = 0; t <= last; ++t) {
        const int logical = t / bs, slot = t % bs;
        if (logical >= static_cast<int>(table.size())) break;
        const int physBlock = table[logical];
        const float* k;
        if (f32) k = kv.keyPtr(physBlock, slot, head);
        else     { kv.readKey(physBlock, slot, head, kbuf.data()); k = kbuf.data(); }

        float dot = 0.0f;
        for (int i = 0; i < d; ++i) dot += q[i] * k[i];
        const float s = dot * scale;

        // Online softmax: rescale accumulator once for the new running max, add.
        const float newMax = (runMax > s) ? runMax : s;
        const float resc    = std::exp(runMax - newMax);
        const float e       = std::exp(s - newMax);
        for (int i = 0; i < d; ++i) out[i] *= resc;
        runSum = runSum * resc + e;
        const float* v;
        if (f32) v = kv.valPtr(physBlock, slot, head);
        else     { kv.readVal(physBlock, slot, head, vbuf.data()); v = vbuf.data(); }
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
