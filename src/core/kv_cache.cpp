#include "vgre/core/kv_cache.h"
#include "vgre/core/kv_quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
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
    : kv_(kv), maxBatch_(maxBatch < 1 ? 1 : maxBatch) {
    const size_t bucketCount = static_cast<size_t>(kv_.totalBlocks()) + 1;
    waitingByBlocks_.resize(bucketCount);

    size_t leafCount = 1;
    while (leafCount < bucketCount) {
        if (leafCount > std::numeric_limits<size_t>::max() / 2)
            throw std::length_error("scheduler waiting index size overflow");
        leafCount *= 2;
    }
    if (leafCount > waitingTree_.max_size() / 2)
        throw std::length_error("scheduler waiting index exceeds vector capacity");
    waitingTreeBase_ = leafCount;
    // A bucket index outside [0, bucketCount) means the subtree has no waiter.
    waitingTree_.assign(leafCount * 2, bucketCount);
}

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
    if (waitingCount_ >= static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("scheduler waiting request count exceeds its public count range");
    if (kv_.hasSequence(id))
        throw std::invalid_argument("request ID already owns a KV-cache sequence");
    if (!activeRequestIds_.insert(id).second)
        throw std::invalid_argument("request ID is already waiting or running");
    Request r;
    r.id = id; r.promptLen = promptLen; r.maxNewTokens = maxNewTokens;
    r.state = Request::WAITING;
    const size_t requiredBlocks = static_cast<size_t>(blocksFor(promptLen + maxNewTokens));
    try {
        if (nextWaitingOrder_ == std::numeric_limits<uint64_t>::max())
            throw std::length_error("scheduler request ordering counter exhausted");
        std::unique_ptr<std::deque<QueuedRequest>>& queue = waitingByBlocks_[requiredBlocks];
        if (!queue)
            queue.reset(new std::deque<QueuedRequest>());
        queue->push_back(QueuedRequest{r, nextWaitingOrder_});
    } catch (...) {
        if (waitingByBlocks_[requiredBlocks] && waitingByBlocks_[requiredBlocks]->empty())
            waitingByBlocks_[requiredBlocks].reset();
        activeRequestIds_.erase(id);
        throw;
    }
    ++nextWaitingOrder_;
    ++waitingCount_;
    if (promptLen == 0 && maxNewTokens == 0)
        ++zeroTokenWaitingCount_;
    updateWaitingBucket(requiredBlocks);
}

int ContinuousBatchScheduler::blocksFor(int tokens) const {
    const int bs = kv_.blockSize();
    return tokens / bs + (tokens % bs != 0 ? 1 : 0);
}

size_t ContinuousBatchScheduler::olderWaitingBucket(size_t lhs, size_t rhs) const {
    const size_t noBucket = waitingByBlocks_.size();
    if (lhs >= noBucket) return rhs;
    if (rhs >= noBucket) return lhs;
    return waitingByBlocks_[lhs]->front().order <= waitingByBlocks_[rhs]->front().order
        ? lhs : rhs;
}

void ContinuousBatchScheduler::updateWaitingBucket(size_t bucket) {
    size_t node = waitingTreeBase_ + bucket;
    if (waitingByBlocks_[bucket] && waitingByBlocks_[bucket]->empty())
        waitingByBlocks_[bucket].reset();
    waitingTree_[node] = waitingByBlocks_[bucket] ? bucket : waitingByBlocks_.size();
    while (node > 1) {
        node /= 2;
        waitingTree_[node] = olderWaitingBucket(waitingTree_[node * 2],
                                                waitingTree_[node * 2 + 1]);
    }
}

size_t ContinuousBatchScheduler::oldestEligibleWaitingBucket(int availableBlocks,
                                                              uint64_t beforeOrder) const {
    const size_t noBucket = waitingByBlocks_.size();
    if (waitingCount_ == 0 || availableBlocks < 0)
        return noBucket;
    const size_t lastEligible = std::min(static_cast<size_t>(availableBlocks),
                                         noBucket - 1);
    size_t left = waitingTreeBase_;
    size_t right = waitingTreeBase_ + lastEligible + 1;
    size_t oldestLeft = noBucket;
    size_t oldestRight = noBucket;
    while (left < right) {
        if (left & 1u)
            oldestLeft = olderWaitingBucket(oldestLeft, waitingTree_[left++]);
        if (right & 1u)
            oldestRight = olderWaitingBucket(waitingTree_[--right], oldestRight);
        left /= 2;
        right /= 2;
    }
    const size_t oldest = olderWaitingBucket(oldestLeft, oldestRight);
    if (oldest < noBucket &&
        waitingByBlocks_[oldest]->front().order >= beforeOrder)
        return noBucket;
    return oldest;
}

bool ContinuousBatchScheduler::executeAndAppend(const Request& request,
                                                 ScheduledTokenPhase phase,
                                                 int position) {
    if (!tokenExecutor_) {
        lastError_ = SchedulerError::MISSING_EXECUTOR;
        lastErrorMessage_ = "continuous batching requires a model token executor";
        return false;
    }

    const size_t stride = static_cast<size_t>(kv_.numHeads()) *
                          static_cast<size_t>(kv_.headDim());
    const float invalid = std::numeric_limits<float>::quiet_NaN();
    keyScratch_.assign(stride, invalid);
    valueScratch_.assign(stride, invalid);
    try {
        tokenExecutor_(request, phase, position, keyScratch_, valueScratch_);
    } catch (const std::exception& error) {
        lastError_ = SchedulerError::EXECUTOR_FAILURE;
        lastErrorMessage_ = error.what();
        return false;
    } catch (...) {
        lastError_ = SchedulerError::EXECUTOR_FAILURE;
        lastErrorMessage_ = "model token executor threw a non-standard exception";
        return false;
    }
    if (keyScratch_.size() != stride || valueScratch_.size() != stride) {
        lastError_ = SchedulerError::INVALID_KV_OUTPUT;
        lastErrorMessage_ = "model token executor returned an invalid K/V shape";
        return false;
    }
    for (size_t i = 0; i < stride; ++i) {
        if (!std::isfinite(keyScratch_[i]) || !std::isfinite(valueScratch_[i])) {
            lastError_ = SchedulerError::INVALID_KV_OUTPUT;
            lastErrorMessage_ = "model token executor left K/V unset or non-finite";
            return false;
        }
    }
    if (!kv_.appendToken(request.id, keyScratch_.data(), valueScratch_.data())) {
        lastError_ = SchedulerError::CACHE_APPEND_FAILURE;
        lastErrorMessage_ = "reserved KV capacity could not append a computed token";
        return false;
    }
    return true;
}

int ContinuousBatchScheduler::step() {
    lastError_ = SchedulerError::NONE;
    lastErrorMessage_.clear();
    const uint64_t admissionOrderLimit = nextWaitingOrder_;
    if (!tokenExecutor_) {
        for (const Request& r : running_)
            if (r.generated < r.maxNewTokens) {
                lastError_ = SchedulerError::MISSING_EXECUTOR;
                lastErrorMessage_ = "continuous batching requires a model token executor";
                return -1;
            }
        if (waitingCount_ > zeroTokenWaitingCount_) {
            lastError_ = SchedulerError::MISSING_EXECUTOR;
            lastErrorMessage_ = "continuous batching requires a model token executor";
            return -1;
        }
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

    // 2. The prefix-min tree selects the oldest request whose full configured
    //    capacity fits. Skipped requests stay in their size buckets, so an
    //    oversized head item cannot block smaller work and the waiting queue
    //    is never scanned or compacted. Each admission/query updates O(log B)
    //    index nodes for B physical cache blocks.
    while (static_cast<int>(running_.size()) < maxBatch_) {
        const size_t bucket = oldestEligibleWaitingBucket(kv_.freeBlocks(),
                                                          admissionOrderLimit);
        if (bucket == waitingByBlocks_.size())
            break;

        // Work from a local snapshot: the executor may enqueue another request,
        // which can reallocate unrelated storage but cannot invalidate this one.
        Request request = waitingByBlocks_[bucket]->front().request;
        const int tokenCapacity = request.promptLen + request.maxNewTokens;
        if (running_.size() == running_.capacity()) {
            // Grow scheduler storage before taking cache blocks or invoking model code.
            const size_t capacity = running_.capacity();
            const size_t limit = static_cast<size_t>(maxBatch_);
            const size_t next = capacity == 0 ? 1
                : (capacity > limit / 2 ? limit : capacity * 2);
            running_.reserve(next);
        }

        if (kv_.hasSequence(request.id)) {
            lastError_ = SchedulerError::CACHE_RESERVATION_FAILURE;
            lastErrorMessage_ = "waiting request ID acquired a KV sequence outside this scheduler";
            return -1;
        }
        bool reserved = false;
        try {
            reserved = kv_.reserveSequenceCapacity(request.id, tokenCapacity);
        } catch (...) {
            // reserveSequenceCapacity can install an empty sequence record
            // before a later vector allocation throws.
            kv_.freeSequence(request.id);
            throw;
        }
        if (!reserved) {
            lastError_ = SchedulerError::CACHE_RESERVATION_FAILURE;
            lastErrorMessage_ = "eligible request could not reserve its KV capacity";
            return -1;
        }

        try {
            for (int t = 0; t < request.promptLen; ++t) {
                if (!executeAndAppend(request, ScheduledTokenPhase::PREFILL, t)) {
                    kv_.freeSequence(request.id);
                    return -1;
                }
            }
        } catch (...) {
            // Allocation failures during scratch-buffer growth must not leak
            // the reservation; the request remains at the head of its bucket.
            kv_.freeSequence(request.id);
            throw;
        }

        std::deque<QueuedRequest>& queue = *waitingByBlocks_[bucket];
        const bool zeroTokenRequest = request.promptLen == 0 && request.maxNewTokens == 0;
        queue.pop_front();
        --waitingCount_;
        if (zeroTokenRequest)
            --zeroTokenWaitingCount_;
        updateWaitingBucket(bucket);
        request.state = Request::RUNNING;
        running_.push_back(std::move(request));
    }

    // 3. Advance each running request by one decode token.
    for (auto& r : running_) {
        if (r.generated < r.maxNewTokens) {
            const int position = r.promptLen + r.generated;
            if (!executeAndAppend(r, ScheduledTokenPhase::DECODE, position))
                return -1;
            ++r.generated;
        }
    }
    return static_cast<int>(running_.size());
}

} // namespace core
} // namespace vgre
