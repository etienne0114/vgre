#include "vgre/core/kv_cache.h"
#include "vgre/core/kv_quant.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace vgre {
namespace core {

namespace {

constexpr uint8_t kBlockFree = 0;
constexpr uint8_t kBlockRaw = 1;
constexpr uint8_t kBlockSequence = 2;

int requirePositiveDimension(int value, const char* name) {
    if (value <= 0)
        throw std::invalid_argument(std::string("KV cache ") + name + " must be positive");
    return value;
}

size_t checkedProduct(size_t a, size_t b, const char* what) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
        throw std::length_error(std::string("KV cache size overflow while computing ") + what);
    return a * b;
}

size_t checkedSum(size_t a, size_t b, const char* what) {
    if (a > std::numeric_limits<size_t>::max() - b)
        throw std::length_error(std::string("KV cache size overflow while computing ") + what);
    return a + b;
}

}  // namespace

KVCacheManager::KVCacheManager(int numBlocks, int blockSize, int numHeads, int headDim,
                               KVDType dtype)
    : blockSize_(requirePositiveDimension(blockSize, "block size")),
      numHeads_(requirePositiveDimension(numHeads, "head count")),
      headDim_(requirePositiveDimension(headDim, "head dimension")),
      dtype_(dtype),
      tokenStride_(checkedProduct(static_cast<size_t>(numHeads_),
                                  static_cast<size_t>(headDim_), "token stride")),
      blockStride_(checkedProduct(static_cast<size_t>(blockSize_), tokenStride_,
                                  "block stride")),
      packDim_(headDim / 2) {
    if (numBlocks < 0)
        throw std::invalid_argument("KV cache block count cannot be negative");
    if (dtype_ != KVDType::F32 && dtype_ != KVDType::I8 && dtype_ != KVDType::I4)
        throw std::invalid_argument("KV cache data type is invalid");

    // int4 needs an even headDim (2 codes/byte, head-aligned); otherwise use int8.
    if (dtype_ == KVDType::I4 && (headDim & 1)) dtype_ = KVDType::I8;
    const size_t nBlk = static_cast<size_t>(numBlocks);
    const size_t nTokens = checkedProduct(nBlk, static_cast<size_t>(blockSize_),
                                          "token capacity");
    const size_t nSlotHead = checkedProduct(nTokens, static_cast<size_t>(numHeads_),
                                             "per-head scales");
    const size_t nElements = checkedProduct(nBlk, blockStride_, "K/V element count");
    if (dtype_ == KVDType::I8) {
        kPool8_.assign(nElements, 0);
        vPool8_.assign(nElements, 0);
        kScale_.assign(nSlotHead, 0.0f);
        vScale_.assign(nSlotHead, 0.0f);
    } else if (dtype_ == KVDType::I4) {
        const size_t nPacked = checkedProduct(nSlotHead, static_cast<size_t>(packDim_),
                                              "packed int4 element count");
        kPool4_.assign(nPacked, 0);
        vPool4_.assign(nPacked, 0);
        kScale_.assign(nSlotHead, 0.0f);
        vScale_.assign(nSlotHead, 0.0f);
    } else {
        kPool_.assign(nElements, 0.0f);
        vPool_.assign(nElements, 0.0f);
    }
    freeList_.reserve(nBlk);
    blockState_.assign(nBlk, kBlockFree);
    for (int i = numBlocks - 1; i >= 0; --i) freeList_.push_back(i);  // pop from back = ascending
}

size_t KVCacheManager::bytesPerToken() const {
    const size_t scaleBytes = checkedProduct(static_cast<size_t>(numHeads_), sizeof(float),
                                             "per-head scale bytes");
    switch (dtype_) {
        case KVDType::I8:
            return checkedProduct(2, checkedSum(checkedProduct(tokenStride_, sizeof(int8_t),
                                                               "int8 bytes per token"),
                                                scaleBytes, "int8 bytes per token"),
                                  "int8 bytes per token");
        case KVDType::I4:
            return checkedProduct(2, checkedSum(checkedProduct(static_cast<size_t>(numHeads_),
                                                               static_cast<size_t>(packDim_),
                                                               "int4 bytes per token"),
                                                scaleBytes, "int4 bytes per token"),
                                  "int4 bytes per token");
        default:
            return checkedProduct(2, checkedProduct(tokenStride_, sizeof(float),
                                                     "fp32 bytes per token"),
                                  "fp32 bytes per token");
    }
}

int KVCacheManager::allocateBlock() {
    if (freeList_.empty()) return -1;
    int b = freeList_.back();
    freeList_.pop_back();
    blockState_[static_cast<size_t>(b)] = kBlockRaw;
    return b;
}

bool KVCacheManager::reserveSequenceCapacity(SeqId seq, int tokenCapacity) {
    if (tokenCapacity < 0) return false;
    const size_t requiredBlocks = static_cast<size_t>(tokenCapacity / blockSize_) +
        (tokenCapacity % blockSize_ != 0 ? 1u : 0u);
    auto it = seqs_.find(seq);
    if (it == seqs_.end()) {
        if (requiredBlocks > freeList_.size()) return false;
        Seq fresh;
        fresh.blockTable.reserve(requiredBlocks);
        it = seqs_.emplace(seq, std::move(fresh)).first;
    }
    Seq& s = it->second;
    if (tokenCapacity < s.length) return false;
    if (requiredBlocks <= s.blockTable.size()) return true;
    const size_t additional = requiredBlocks - s.blockTable.size();
    if (additional > freeList_.size()) return false;

    // Complete any allocation that can throw before removing blocks from the
    // free list. Pushing ints is then non-throwing because capacity is reserved.
    s.blockTable.reserve(requiredBlocks);
    for (size_t i = 0; i < additional; ++i) {
        const int block = freeList_.back();
        freeList_.pop_back();
        s.blockTable.push_back(block);
        blockState_[static_cast<size_t>(block)] = kBlockSequence;
    }
    return true;
}

void KVCacheManager::freeBlock(int physBlock) {
    releaseBlock(physBlock, false);
}

void KVCacheManager::releaseBlock(int physBlock, bool sequenceRelease) {
    if (physBlock < 0 || static_cast<size_t>(physBlock) >= blockState_.size()) return;
    const size_t block = static_cast<size_t>(physBlock);
    const uint8_t expected = sequenceRelease ? kBlockSequence : kBlockRaw;
    if (blockState_[block] != expected) return;
    blockState_[block] = kBlockFree;
    freeList_.push_back(physBlock);
}

const float* KVCacheManager::keyPtr(int physBlock, int slot, int head) const {
    if (dtype_ != KVDType::F32 || physBlock < 0 ||
        static_cast<size_t>(physBlock) >= blockState_.size() ||
        blockState_[static_cast<size_t>(physBlock)] == kBlockFree ||
        slot < 0 || slot >= blockSize_ || head < 0 || head >= numHeads_)
        return nullptr;   // quantized or invalid: use readKey / check indices
    return &kPool_[(static_cast<size_t>(physBlock) * blockSize_ + slot) * tokenStride_ +
                   static_cast<size_t>(head) * headDim_];
}
const float* KVCacheManager::valPtr(int physBlock, int slot, int head) const {
    if (dtype_ != KVDType::F32 || physBlock < 0 ||
        static_cast<size_t>(physBlock) >= blockState_.size() ||
        blockState_[static_cast<size_t>(physBlock)] == kBlockFree ||
        slot < 0 || slot >= blockSize_ || head < 0 || head >= numHeads_)
        return nullptr;
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
    if (!out || physBlock < 0 || static_cast<size_t>(physBlock) >= blockState_.size() ||
        blockState_[static_cast<size_t>(physBlock)] == kBlockFree ||
        slot < 0 || slot >= blockSize_ || head < 0 || head >= numHeads_) return;
    const size_t st = static_cast<size_t>(physBlock) * blockSize_ + slot;
    readSlice(dtype_, kPool_, kPool8_, kPool4_, kScale_,
              st * numHeads_ + head, st * tokenStride_ + static_cast<size_t>(head) * headDim_,
              (st * numHeads_ + head) * packDim_, headDim_, out);
}
void KVCacheManager::readVal(int physBlock, int slot, int head, float* out) const {
    if (!out || physBlock < 0 || static_cast<size_t>(physBlock) >= blockState_.size() ||
        blockState_[static_cast<size_t>(physBlock)] == kBlockFree ||
        slot < 0 || slot >= blockSize_ || head < 0 || head >= numHeads_) return;
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
    if (!kAllHeads || !vAllHeads) return false;
    Seq& s = seqs_[seq];
    if (s.length == std::numeric_limits<int>::max()) return false;
    const int slot = s.length % blockSize_;
    const size_t logicalBlock = static_cast<size_t>(s.length / blockSize_);
    if (logicalBlock >= s.blockTable.size()) {  // crossing into an unreserved block
        // Reserve table storage before taking a physical block so allocation
        // failure cannot leak a block from the pool.
        s.blockTable.reserve(s.blockTable.size() + 1);
        int b = allocateBlock();
        if (b < 0) return false;           // pool exhausted
        s.blockTable.push_back(b);
        blockState_[static_cast<size_t>(b)] = kBlockSequence;
    }
    const int physBlock = s.blockTable[logicalBlock];
    writeToken(physBlock, slot, kAllHeads, vAllHeads);
    ++s.length;
    return true;
}

int KVCacheManager::seqLen(SeqId seq) const {
    auto it = seqs_.find(seq);
    return it == seqs_.end() ? 0 : it->second.length;
}

bool KVCacheManager::hasSequence(SeqId seq) const {
    return seqs_.find(seq) != seqs_.end();
}

const std::vector<int>& KVCacheManager::blockTable(SeqId seq) const {
    static const std::vector<int> kEmpty;
    auto it = seqs_.find(seq);
    return it == seqs_.end() ? kEmpty : it->second.blockTable;
}

void KVCacheManager::freeSequence(SeqId seq) {
    auto it = seqs_.find(seq);
    if (it == seqs_.end()) return;
    for (int b : it->second.blockTable) releaseBlock(b, true);
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

void ContinuousBatchScheduler::setTokenExecutor(TokenExecutor executor) {
    tokenExecutor_ = std::move(executor);
}

void ContinuousBatchScheduler::addRequest(SeqId id, int promptLen, int maxNewTokens) {
    if (promptLen < 0 || maxNewTokens < 0)
        throw std::invalid_argument("request token counts cannot be negative");
    if (maxNewTokens > std::numeric_limits<int>::max() - promptLen)
        throw std::length_error("request prompt and generation length exceed sequence indexing range");
    if (blocksFor(promptLen + maxNewTokens) > kv_.totalBlocks())
        throw std::length_error("request's maximum KV capacity exceeds the cache pool");
    if (kv_.hasSequence(id))
        throw std::invalid_argument("request ID already owns a KV-cache sequence");
    if (!activeRequestIds_.insert(id).second)
        throw std::invalid_argument("request ID is already waiting or running");
    Request r;
    r.id = id; r.promptLen = promptLen; r.maxNewTokens = maxNewTokens;
    r.state = Request::WAITING;
    try {
        waiting_.push_back(r);
    } catch (...) {
        activeRequestIds_.erase(id);
        throw;
    }
}

int ContinuousBatchScheduler::blocksFor(int tokens) const {
    const int bs = kv_.blockSize();
    return tokens / bs + (tokens % bs != 0 ? 1 : 0);
}

void ContinuousBatchScheduler::executeAndAppend(const Request& request,
                                                 ScheduledTokenPhase phase,
                                                 int position) {
    if (!tokenExecutor_)
        throw std::logic_error("continuous batching requires a model token executor");

    const size_t stride = static_cast<size_t>(kv_.numHeads()) *
                          static_cast<size_t>(kv_.headDim());
    const float invalid = std::numeric_limits<float>::quiet_NaN();
    keyScratch_.assign(stride, invalid);
    valueScratch_.assign(stride, invalid);
    tokenExecutor_(request, phase, position, keyScratch_, valueScratch_);
    if (keyScratch_.size() != stride || valueScratch_.size() != stride)
        throw std::runtime_error("model token executor returned an invalid K/V shape");
    for (size_t i = 0; i < stride; ++i) {
        if (!std::isfinite(keyScratch_[i]) || !std::isfinite(valueScratch_[i]))
            throw std::runtime_error("model token executor left K/V unset or non-finite");
    }
    if (!kv_.appendToken(request.id, keyScratch_.data(), valueScratch_.data()))
        throw std::runtime_error("reserved KV capacity could not append a computed token");
}

int ContinuousBatchScheduler::step() {
    if (!tokenExecutor_) {
        for (const Request& r : running_)
            if (r.generated < r.maxNewTokens)
                throw std::logic_error("continuous batching requires a model token executor");
        for (const Request& r : waiting_)
            if (r.promptLen > 0 || r.maxNewTokens > 0)
                throw std::logic_error("continuous batching requires a model token executor");
    }

    // 1. Retire finished running requests; reclaim their KV blocks.
    size_t live = 0;
    const size_t runningAtStart = running_.size();
    for (size_t i = 0; i < runningAtStart; ++i) {
        Request& r = running_[i];
        if (r.generated >= r.maxNewTokens) {
            r.state = Request::FINISHED;
            kv_.freeSequence(r.id);   // return blocks to the pool
            activeRequestIds_.erase(r.id);
            ++finished_;
        } else {
            if (live != i) running_[live] = std::move(r);
            ++live;
        }
    }
    running_.resize(live);

    // 2. Admit only when the whole requested sequence fits. Reserving the
    //    configured maximum up front prevents a live request from holding a
    //    partial cache forever while waiting for blocks needed to finish.
    size_t stillWaiting = 0;
    const size_t waitingAtStart = waiting_.size();
    for (size_t i = 0; i < waitingAtStart; ++i) {
        Request& r = waiting_[i];
        try {
            const int tokenCapacity = r.promptLen + r.maxNewTokens;
            const bool hasBatchSlot = static_cast<int>(running_.size()) < maxBatch_;
            if (hasBatchSlot && running_.size() == running_.capacity()) {
                // Grow scheduler storage before taking cache blocks.
                const size_t capacity = running_.capacity();
                const size_t limit = static_cast<size_t>(maxBatch_);
                const size_t next = capacity == 0 ? 1
                    : (capacity > limit / 2 ? limit : capacity * 2);
                running_.reserve(next);
            }
            const int requiredBlocks = blocksFor(tokenCapacity);
            if (hasBatchSlot && kv_.freeBlocks() >= requiredBlocks &&
                kv_.reserveSequenceCapacity(r.id, tokenCapacity)) {
                for (int t = 0; t < r.promptLen; ++t)
                    executeAndAppend(r, ScheduledTokenPhase::PREFILL, t);
                r.state = Request::RUNNING;
                running_.push_back(r);
            } else {
                if (stillWaiting != i) waiting_[stillWaiting] = std::move(r);
                ++stillWaiting;
            }
        } catch (...) {
            kv_.freeSequence(r.id);
            // Earlier non-admitted requests may have been compacted in place.
            // Restore this request and every not-yet-processed one so the
            // scheduler remains consistent if the model executor throws.
            size_t restoredWaiting = stillWaiting;
            for (size_t j = i; j < waitingAtStart; ++j) {
                if (restoredWaiting != j)
                    waiting_[restoredWaiting] = std::move(waiting_[j]);
                ++restoredWaiting;
            }
            waiting_.resize(restoredWaiting);
            throw;
        }
    }
    waiting_.resize(stillWaiting);

    // 3. Advance each running request by one decode token.
    for (auto& r : running_) {
        if (r.generated < r.maxNewTokens) {
            const int position = r.promptLen + r.generated;
            executeAndAppend(r, ScheduledTokenPhase::DECODE, position);
            ++r.generated;
        }
    }
    return static_cast<int>(running_.size());
}

} // namespace core
} // namespace vgre
