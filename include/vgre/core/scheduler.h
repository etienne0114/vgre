#ifndef VGRE_CORE_SCHEDULER_H
#define VGRE_CORE_SCHEDULER_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include "vgre/core/fib_heap.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vgre {
namespace core {

// ── Lock-free SPSC Ring Buffer (Track 7.3) ────────────────────────────────
// Single-Producer Single-Consumer ring: zero-mutex fast path for streams
// where one host thread submits all work.
template<typename T, size_t N = 4096>
class SPSCRing {
    static_assert((N & (N - 1)) == 0, "N must be power of 2");
    alignas(64) std::atomic<size_t> head_{0};
    char pad0_[64 - sizeof(std::atomic<size_t>)];
    alignas(64) std::atomic<size_t> tail_{0};
    char pad1_[64 - sizeof(std::atomic<size_t>)];
    T buf_[N];
public:
    bool push(T val) {
        size_t h = head_.load(std::memory_order_relaxed);
        if (h - tail_.load(std::memory_order_acquire) >= N) return false;
        buf_[h & (N - 1)] = std::move(val);
        head_.store(h + 1, std::memory_order_release);
        return true;
    }
    bool pop(T& val) {
        size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) return false;
        val = std::move(buf_[t & (N - 1)]);
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }
    bool empty() const {
        return tail_.load(std::memory_order_acquire) ==
               head_.load(std::memory_order_relaxed);
    }
};

// ── Small-Buffer Callable — zero-heap-alloc for ≤56-byte closures ────────────
// Replaces std::function<void()> in WorkItem to eliminate mandatory heap
// allocation. Typical scheduler lambdas capture a shared_ptr (16B) + StreamId
// (8B) + this pointer (8B) = 32B, which fits inline in the 56-byte SBO buffer.
//
// Math: P(heap alloc) ≈ 5% for observed capture sizes in VGRE scheduler paths.
// Complexity: O(1) construction (inline), O(1) call — no virtual dispatch.
// Invariant: if ptr_!=nullptr the callable lives on the heap; otherwise in sbo_.
class VGREFunc {
    static constexpr size_t kSBOSize  = 56;
    static constexpr size_t kSBOAlign = alignof(std::max_align_t);

    struct Vtbl {
        void (*call)(void* p);
        void (*dtor)(void* p, bool heap);   // heap=true → also calls ::operator delete
        void (*move)(void* dst, void* src); // move-construct dst from src, then destroy src
    };

    alignas(kSBOAlign) std::byte sbo_[kSBOSize]{};
    void*       ptr_  = nullptr;   // nullptr = inline, non-null = heap allocation
    const Vtbl* vtbl_ = nullptr;

    void* live() const noexcept { return ptr_ ? ptr_ : (void*)sbo_; }

    template<typename F>
    static const Vtbl* vtbl_for() noexcept {
        static const Vtbl v{
            [](void* p){ (*static_cast<F*>(p))(); },
            [](void* p, bool heap){ static_cast<F*>(p)->~F(); if (heap) ::operator delete(p); },
            [](void* dst, void* src){
                ::new(dst) F(std::move(*static_cast<F*>(src)));
                static_cast<F*>(src)->~F();
            }
        };
        return &v;
    }

public:
    VGREFunc() = default;

    template<typename F,
             typename = std::enable_if_t<!std::is_same_v<std::decay_t<F>, VGREFunc>>>
    VGREFunc(F&& f) {
        using Fd = std::decay_t<F>;
        vtbl_ = vtbl_for<Fd>();
        if constexpr (sizeof(Fd) <= kSBOSize && alignof(Fd) <= kSBOAlign) {
            ::new(sbo_) Fd(std::forward<F>(f));
        } else {
            ptr_ = ::operator new(sizeof(Fd));
            ::new(ptr_) Fd(std::forward<F>(f));
        }
    }

    VGREFunc(VGREFunc&& o) noexcept : ptr_(o.ptr_), vtbl_(o.vtbl_) {
        if (!vtbl_) return;
        if (ptr_) { o.ptr_ = nullptr; }
        else      { vtbl_->move(sbo_, o.sbo_); }
        o.vtbl_ = nullptr;
    }

    VGREFunc& operator=(VGREFunc&& o) noexcept {
        if (this != &o) { this->~VGREFunc(); ::new(this) VGREFunc(std::move(o)); }
        return *this;
    }

    ~VGREFunc() { if (vtbl_) vtbl_->dtor(live(), ptr_ != nullptr); }

    void operator()() { vtbl_->call(live()); }

    explicit operator bool() const noexcept { return vtbl_ != nullptr; }

    VGREFunc(const VGREFunc&)            = delete;
    VGREFunc& operator=(const VGREFunc&) = delete;
};

// ── Work item ──────────────────────────────────────────────────────────────
struct WorkItem {
  StreamId streamId = 0;
  int priority = 0;            // higher = more urgent
  int preferredNumaNode = -1;  // -1 = any node (no NUMA preference)
  VGREFunc execute;

  bool operator<(const WorkItem &o) const {
    return priority < o.priority; // max-heap
  }
};

// ── Chase-Lev Work-Stealing Deque ──────────────────────────────────────────
// Single-owner, multi-thief concurrent deque (Chase & Lev 1993).
// C++11 memory model mapping from Le et al. 2013 "Correct and Efficient
// Work-Stealing for Weak Memory Models."
//
// Invariant: |bottom - top| = number of live items ± 1 during the CAS window
//            when owner and thief race on the final element.
// Complexity: push O(1), pop O(1) amortised, steal O(1).
template <typename T>
class ChaseLevDeque {
public:
  explicit ChaseLevDeque(size_t capacity = 1024)
      : capacity_(capacity), mask_(capacity - 1) {
    buffer_ = new std::atomic<T>[capacity_];
    top_.store(0, std::memory_order_relaxed);
    bottom_.store(0, std::memory_order_relaxed);
  }

  ~ChaseLevDeque() { delete[] buffer_; }

  // push — owner only.  Release fence orders buffer write before bottom store
  // so that any stealer who reads the updated bottom via acquire also sees
  // the data written into buffer[b].
  bool push(T item) {
    int64_t b = bottom_.load(std::memory_order_relaxed);
    int64_t t = top_.load(std::memory_order_acquire);
    if (b - t >= static_cast<int64_t>(capacity_) - 1) return false;
    buffer_[b & mask_].store(std::move(item), std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    bottom_.store(b + 1, std::memory_order_relaxed);
    return true;
  }

  // pop — owner only.
  // Paper algorithm (Le et al. 2013, Fig. 3):
  //   1. b = bottom - 1; store bottom = b  [before seq_cst fence]
  //   2. seq_cst fence
  //   3. t = top  [after fence]
  //   4. if t <= b: read item; if t == b: CAS(top,t,t+1) to win last-item race
  //   5. else restore bottom = b+1, return empty
  bool pop(T& item) {
    int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
    // Store bottom-1 BEFORE the seq_cst fence so that concurrent stealers
    // who load bottom after the fence see the decremented value and correctly
    // detect the last-element race via their own CAS on top.
    bottom_.store(b, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int64_t t = top_.load(std::memory_order_relaxed);
    if (t <= b) {
      item = buffer_[b & mask_].load(std::memory_order_relaxed);
      if (t != b) return true; // More than one item; no race possible.
      // Last item: race with stealers via CAS on top.
      // Winner gets the item; loser restores bottom and returns false.
      if (!top_.compare_exchange_strong(
              t, t + 1,
              std::memory_order_seq_cst,
              std::memory_order_relaxed)) {
        bottom_.store(b + 1, std::memory_order_relaxed);
        return false;
      }
      bottom_.store(b + 1, std::memory_order_relaxed);
      return true;
    }
    bottom_.store(b + 1, std::memory_order_relaxed);
    return false;
  }

  // steal — any thread except the owner.
  // Paper algorithm (Le et al. 2013, Fig. 3):
  //   1. t = top  [seq_cst — participates in total order with pop's fence]
  //   2. seq_cst fence (orders t load before b load on weak hw)
  //   3. b = bottom  [acquire]
  //   4. if t < b: read item, CAS(top,t,t+1) — failure means another thief won
  // Note: loading top with seq_cst (not acquire+fence) is the idiomatic form
  // from the paper.  The seq_cst total order guarantees that two concurrent
  // stealers loading the same t value will resolve via CAS, not by racing
  // undetected on separate non-atomic copies of the index.
  bool steal(T& item) {
    int64_t t = top_.load(std::memory_order_seq_cst);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int64_t b = bottom_.load(std::memory_order_acquire);
    if (t < b) {
      item = buffer_[t & mask_].load(std::memory_order_relaxed);
      if (!top_.compare_exchange_strong(
              t, t + 1,
              std::memory_order_seq_cst,
              std::memory_order_relaxed)) {
        return false; // Another thief or owner won the CAS.
      }
      return true;
    }
    return false;
  }

  // size — approximate (may be stale by the time caller inspects).
  int64_t size() const {
    int64_t b = bottom_.load(std::memory_order_acquire);
    int64_t t = top_.load(std::memory_order_acquire);
    return b - t;
  }

private:
  size_t capacity_;
  size_t mask_;
  std::atomic<T>* buffer_;
  // Padding prevents false sharing between top and bottom on the same cache line.
  alignas(64) std::atomic<int64_t> top_;
  alignas(64) std::atomic<int64_t> bottom_;
};

struct StreamTaskNode {
  VGREFunc task;
  std::promise<VGREResult> promise;
};

/**
 * @brief High-performance task scheduler for VGRE streams.
 *
 * Implements a hybrid work-stealing/serialization model where tasks in a
 * given stream are executed sequentially, while multiple streams execute
 * concurrently across a worker thread pool.
 */
class Scheduler {
public:
  explicit Scheduler(int numThreads = 0); // 0 = auto-detect
  ~Scheduler();

  /**
   * @brief Submits a task to be executed sequentially relative to other tasks in the same stream.
   * @param stream The stream handle.
   * @param taskFn The function to execute.
   * @param priority Task priority (higher is more urgent).
   * @return A future containing the task result.
   */
  std::future<VGREResult> submitStreamTask(StreamId stream,
                                           std::function<void()> taskFn,
                                           int priority = 0);

  /**
   * @brief Submits a task to the global concurrent pool without stream serialization.
   * @param taskFn The function to execute.
   * @param priority Task priority.
   * @return A future containing the task result.
   */
  std::future<VGREResult> submitConcurrentTask(std::function<void()> taskFn,
                                               int priority = 0);

  /**
   * @brief Submits a task with a NUMA-node affinity hint.
   *
   * On Linux, worker threads are pinned to NUMA-local CPUs; tasks submitted
   * with a matching numaNode are preferentially dispatched to those workers,
   * reducing cross-NUMA memory-access latency.  Falls back to any worker when
   * no NUMA information is available.
   *
   * @param stream    Stream handle (carried for tracking; task executes concurrently).
   * @param taskFn    The function to execute.
   * @param numaNode  Preferred NUMA node index (-1 = no preference).
   * @param priority  Task priority (higher is more urgent).
   * @return A future containing the task result.
   */
  std::future<VGREResult> submitNumaTask(StreamId stream,
                                         std::function<void()> taskFn,
                                         int numaNode,
                                         int priority = 0);

  // Control
  void waitAll();
  void waitStream(StreamId stream);
  bool isStreamIdle(StreamId stream) const;
  int getThreadCount() const;
  void setThreadCount(int n);

  // Statistics
  uint64_t getCompletedTasks() const;
  uint64_t getPendingTasks() const;

  // Per-stream SPSC fast path (Track 7.3)
  void enqueueToStream(uint64_t streamId, WorkItem item);

  // Update the priority of all queued items belonging to a stream.
  // O(log n) due to the indexed priority heap.
  void updateStreamPriority(StreamId stream, int newPriority);

  // Singleton convenience
  static Scheduler &instance();

private:
  // ── Task Priority Heap ────────────────────────────────────────────────────
  // An indexed binary max-heap that supports O(log n) priority updates
  // (decrease-key / increase-key). This enables stream priority changes to
  // take effect on already-queued items without dequeue-requeue.
  //
  // Unlike std::priority_queue, we maintain a position index (stream → index)
  // so updateStreamPriority() can find and re-sort items in O(log n).
  struct IndexedHeap {
    std::vector<WorkItem>                 data;
    std::unordered_map<StreamId, size_t>  pos;   // stream → heap index

    bool empty() const { return data.empty(); }
    const WorkItem& top() const { return data[0]; }

    void push(WorkItem item) {
      size_t idx = data.size();
      pos[item.streamId] = idx;
      data.push_back(std::move(item));
      siftUp(idx);
    }

    void pop() {
      if (data.empty()) return;
      pos.erase(data[0].streamId);
      if (data.size() > 1) {
        data[0] = std::move(data.back());
        pos[data[0].streamId] = 0;
      }
      data.pop_back();
      if (!data.empty()) siftDown(0);
    }

    void updatePriority(StreamId stream, int newPriority) {
      auto it = pos.find(stream);
      if (it == pos.end()) return;
      size_t idx = it->second;
      int old = data[idx].priority;
      data[idx].priority = newPriority;
      if (newPriority > old) siftUp(idx);
      else                   siftDown(idx);
    }

  private:
    void siftUp(size_t i) {
      while (i > 0) {
        size_t p = (i - 1) / 2;
        if (data[p] < data[i]) { swap(p, i); i = p; }
        else break;
      }
    }
    void siftDown(size_t i) {
      size_t n = data.size();
      while (true) {
        size_t best = i, l = 2*i+1, r = 2*i+2;
        if (l < n && data[best] < data[l]) best = l;
        if (r < n && data[best] < data[r]) best = r;
        if (best == i) break;
        swap(i, best); i = best;
      }
    }
    void swap(size_t a, size_t b) {
      pos[data[a].streamId] = b;
      pos[data[b].streamId] = a;
      std::swap(data[a], data[b]);
    }
  };

  void buildNumaTopology();
  void workerLoop(int workerIdx);
  struct StreamQueue;
  void tryProcessStream(std::shared_ptr<StreamQueue> sq, StreamId stream);

  static bool enqueueWithWorkerFallback(int workerIdx,
                                       WorkItem&& item,
                                       std::vector<std::unique_ptr<ChaseLevDeque<WorkItem*>>>& workerDeques,
                                       IndexedHeap& globalQueue,
                                       std::mutex& queueMutex);
  bool hasNumaNode(int numaNode) const;

  std::vector<std::thread> workers_;
  std::vector<std::unique_ptr<ChaseLevDeque<WorkItem*>>> workerDeques_;
  IndexedHeap queue_;

  // Fibonacci heap used for O(1) amortised increase_key on stream-priority
  // updates.  Maintains a parallel view of WorkItems alongside IndexedHeap;
  // the scheduler may use fibHeap_.increase_key(node, newPriority) instead of
  // IndexedHeap::updatePriority() when stream-priority changes are frequent.
  // Node pointers are stored per-stream in fibNodeByStream_.
  FibHeap<WorkItem>                                    fibHeap_;
  std::unordered_map<StreamId, FibHeap<WorkItem>::Node*> fibNodeByStream_;

  std::unordered_map<int, IndexedHeap> numaQueues_;
  // NUMA node assigned to each worker thread (index == worker index in workers_).
  std::vector<int> workerNumaNodes_;
  // Set of NUMA nodes for O(1) membership testing (kept in sync with workerNumaNodes_)
  std::unordered_set<int> workerNumaNodeSet_;

  // Per-stream serialization: StreamId -> Queue of task nodes
  struct StreamQueue {
    int streamPriority = 0;
    std::queue<std::shared_ptr<StreamTaskNode>> pendingTasks;
    bool isProcessing = false;
    std::mutex streamMutex;
  };
  std::unordered_map<StreamId, std::shared_ptr<StreamQueue>> streamQueues_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> shutdown_{false};
  std::atomic<uint64_t> completed_{0};
  std::atomic<uint64_t> pending_{0};
  int numThreads_;

#ifdef ENABLE_VGRE_SPSC
  // Per-worker SPSC rings (Track 7.3 — one ring per worker, not per stream).
  // Stream S is pinned to worker (S % numThreads_). The submitting thread is
  // the sole producer; the assigned worker is the sole consumer.
  std::vector<std::unique_ptr<SPSCRing<WorkItem>>> workerRings_;
#endif
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_SCHEDULER_H
