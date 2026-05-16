#include "vgre/core/scheduler.h"
#include "vgre/common/logger.h"
#include <future>
#include <memory>

namespace vgre {
namespace core {

thread_local int t_workerIdx = -1;

// ── Helper functions shared across split implementation files ─────────────
bool Scheduler::enqueueWithWorkerFallback(int workerIdx,
                                        WorkItem&& item,
                                        std::vector<std::unique_ptr<ChaseLevDeque<WorkItem*>>>& workerDeques,
                                        std::priority_queue<WorkItem>& globalQueue,
                                        std::mutex& queueMutex) {
  if (workerIdx >= 0 && workerIdx < static_cast<int>(workerDeques.size())) {
    WorkItem* localItem = new WorkItem(std::move(item));
    if (workerDeques[workerIdx]->push(localItem)) {
      return true;
    }
    WorkItem movedBack = std::move(*localItem);
    delete localItem;
    std::lock_guard<std::mutex> lock(queueMutex);
    globalQueue.push(std::move(movedBack));
    return true;
  }
  std::lock_guard<std::mutex> lock(queueMutex);
  globalQueue.push(std::move(item));
  return true;
}

bool Scheduler::hasNumaNode(int numaNode) const {
  if (numaNode < 0) {
    return false;
  }
  // Use unordered_set for O(1) lookup instead of O(n) linear search
  return workerNumaNodeSet_.find(numaNode) != workerNumaNodeSet_.end();
}

// ── Constructor ────────────────────────────────────────────────────────────
Scheduler::Scheduler(int numThreads) : numThreads_(numThreads) {
  if (numThreads_ <= 0) {
    numThreads_ = static_cast<int>(std::thread::hardware_concurrency());
    if (numThreads_ <= 0)
      numThreads_ = 4;
  }

  workerNumaNodes_.resize(numThreads_, -1);
  workerNumaNodeSet_.clear();
  workerDeques_.resize(numThreads_);
  for (int i = 0; i < numThreads_; ++i) {
    workerDeques_[i] = std::make_unique<ChaseLevDeque<WorkItem*>>();
  }

  for (int i = 0; i < numThreads_; ++i) {
    workers_.emplace_back([this, i]() {
      this->workerLoop(i);
    });
  }

  buildNumaTopology();

  VGRE_LOG_INFO("Scheduler", "Started thread pool with " +
                                 std::to_string(numThreads_) + " workers");
}

// ── Destructor ─────────────────────────────────────────────────────────────
Scheduler::~Scheduler() {
  shutdown_ = true;
  cv_.notify_all();
  for (auto &w : workers_) {
    if (!w.joinable()) continue;
    auto p = std::make_shared<std::promise<void>>();
    auto f = p->get_future();
    std::thread joiner([p, innerW = std::move(w)]() mutable {
        innerW.join();
        p->set_value();
    });
    joiner.detach();
    f.wait_for(std::chrono::seconds(5)); // non-blocking after timeout
  }
  VGRE_LOG_DEBUG("Scheduler", "Shut down — completed " +
                                  std::to_string(completed_.load()) + " tasks");
}

} // namespace core
} // namespace vgre
