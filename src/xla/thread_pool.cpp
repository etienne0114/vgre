// Work-stealing fork-join thread pool — impl. See include/vgre/xla/thread_pool.h.

#include "vgre/xla/thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace vgre {
namespace xla {

namespace {
// Which pool worker the current thread is (-1 = an external/caller thread).
// Used to push a job's tasks onto the submitter's own deque for locality.
thread_local int g_worker_index = -1;
}  // namespace

ThreadPool::ThreadPool(unsigned workers) {
    workers_.reserve(workers);
    for (unsigned i = 0; i < workers; ++i) workers_.push_back(std::make_unique<Worker>());
    threads_.reserve(workers);
    for (unsigned i = 0; i < workers; ++i)
        threads_.emplace_back([this, i] { workerLoop(static_cast<int>(i)); });
}

ThreadPool::~ThreadPool() {
    stop_.store(true, std::memory_order_release);
    wake_cv_.notify_all();
    for (auto& t : threads_)
        if (t.joinable()) t.join();
}

void ThreadPool::push(int worker, std::function<void()> task) {
    {
        Worker& w = *workers_[worker];
        std::lock_guard<std::mutex> lk(w.m);
        w.dq.push_back(std::move(task));
    }
    pending_.fetch_add(1, std::memory_order_release);
    wake_cv_.notify_one();
}

bool ThreadPool::tryRunOne(int self) {
    std::function<void()> task;
    {
        Worker& w = *workers_[self];
        std::lock_guard<std::mutex> lk(w.m);
        if (!w.dq.empty()) {
            task = std::move(w.dq.back());  // own: LIFO (cache-warm)
            w.dq.pop_back();
        }
    }
    if (!task) {
        const int n = static_cast<int>(workers_.size());
        for (int k = 1; k <= n; ++k) {           // steal from others: FIFO
            int v = (self + k) % n;
            Worker& w = *workers_[v];
            std::lock_guard<std::mutex> lk(w.m);
            if (!w.dq.empty()) {
                task = std::move(w.dq.front());
                w.dq.pop_front();
                break;
            }
        }
    }
    if (task) {
        pending_.fetch_sub(1, std::memory_order_acq_rel);
        task();
        return true;
    }
    return false;
}

void ThreadPool::workerLoop(int index) {
    g_worker_index = index;
    while (!stop_.load(std::memory_order_acquire)) {
        if (!tryRunOne(index)) {
            std::unique_lock<std::mutex> lk(wake_m_);
            wake_cv_.wait_for(lk, std::chrono::milliseconds(2), [this] {
                return stop_.load(std::memory_order_acquire) ||
                       pending_.load(std::memory_order_acquire) > 0;
            });
        }
    }
}

void ThreadPool::parallelFor(int64_t count, int64_t grain,
                             const std::function<void(int64_t)>& body) {
    if (count <= 0) return;
    if (grain < 1) grain = 1;
    int64_t chunks = (count + grain - 1) / grain;
    if (workers_.empty() || chunks <= 1) {        // serial fallback
        for (int64_t i = 0; i < count; ++i) body(i);
        return;
    }
    // The submitter pushes onto its own deque (worker 0 for external threads),
    // spreading chunks round-robin; a shared counter tracks completion. `body`
    // and `done` outlive every task because we block below until done == 0.
    auto done = std::make_shared<std::atomic<int64_t>>(chunks);
    int self = g_worker_index < 0 ? 0 : g_worker_index;
    const int nw = static_cast<int>(workers_.size());
    for (int64_t c = 0; c < chunks; ++c) {
        int64_t lo = c * grain, hi = std::min(count, lo + grain);
        push(static_cast<int>((self + c) % nw), [&body, lo, hi, done] {
            for (int64_t i = lo; i < hi; ++i) body(i);
            done->fetch_sub(1, std::memory_order_acq_rel);
        });
    }
    // Participate until *this* job is finished (may also run others' tasks).
    while (done->load(std::memory_order_acquire) > 0) {
        if (!tryRunOne(self)) std::this_thread::yield();
    }
}

ThreadPool& ThreadPool::global() {
    // Worker count from VGRE_XLA_THREADS (total threads incl. caller). The pool
    // holds total-1 workers; total==1 → 0 workers → everything runs serially.
    static ThreadPool pool([] {
        unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        unsigned total = std::min(hw, 8u);
        if (const char* e = std::getenv("VGRE_XLA_THREADS")) {
            long v = std::strtol(e, nullptr, 10);
            if (v >= 1) total = std::min<unsigned>(static_cast<unsigned>(v), hw);
        }
        return total > 0 ? total - 1 : 0;
    }());
    return pool;
}

}  // namespace xla
}  // namespace vgre
