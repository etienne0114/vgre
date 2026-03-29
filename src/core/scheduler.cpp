#include "vgre/core/scheduler.h"
#include "vgre/common/logger.h"

#include <sstream>
#include <cstring>

#if defined(__linux__)
#include <dirent.h>
#include <fstream>
#include <pthread.h>
#include <sched.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace vgre {
namespace core {

// ── Constructor ────────────────────────────────────────────────────────────
Scheduler::Scheduler(int numThreads) : numThreads_(numThreads) {
  if (numThreads_ <= 0) {
    numThreads_ = static_cast<int>(std::thread::hardware_concurrency());
    if (numThreads_ <= 0)
      numThreads_ = 4;
  }

  workerNumaNodes_.resize(numThreads_, -1);
  for (int i = 0; i < numThreads_; ++i) {
    workers_.emplace_back([this, i]() {
      this->workerLoop(i);
    });
  }

  // Discover NUMA topology and pin worker threads.
  // Called after workers_.emplace_back() so native_handle() is valid.
  buildNumaTopology();

  VGRE_LOG_INFO("Scheduler", "Started thread pool with " +
                                 std::to_string(numThreads_) + " workers");
}

// ── Destructor ─────────────────────────────────────────────────────────────
Scheduler::~Scheduler() {
  shutdown_ = true;
  cv_.notify_all();
  for (auto &w : workers_) {
    if (w.joinable())
      w.join();
  }
  VGRE_LOG_DEBUG("Scheduler", "Shut down — completed " +
                                  std::to_string(completed_.load()) + " tasks");
}

// ── NUMA Topology Discovery ────────────────────────────────────────────────
void Scheduler::buildNumaTopology() {
#if defined(__linux__)
  // Struct is local to avoid cpu_set_t leaking into the header.
  struct NodeInfo {
    int nodeId;
    cpu_set_t cpuSet;
  };

  // Helper: parse Linux cpulist format "0-3,8-11" into a cpu_set_t.
  auto parseCpuList = [](const std::string &cpulist, cpu_set_t &cs) {
    CPU_ZERO(&cs);
    std::istringstream ss(cpulist);
    std::string token;
    while (std::getline(ss, token, ',')) {
      if (token.empty()) continue;
      auto dash = token.find('-');
      if (dash == std::string::npos) {
        int cpu = std::stoi(token);
        if (cpu >= 0 && cpu < CPU_SETSIZE) CPU_SET(cpu, &cs);
      } else {
        int first = std::stoi(token.substr(0, dash));
        int last  = std::stoi(token.substr(dash + 1));
        for (int c = first; c <= last && c < CPU_SETSIZE; ++c) CPU_SET(c, &cs);
      }
    }
  };

  // Enumerate /sys/devices/system/node/nodeN directories.
  std::vector<NodeInfo> nodes;
  DIR *dir = opendir("/sys/devices/system/node");
  if (dir) {
    struct dirent *entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
      if (std::strncmp(entry->d_name, "node", 4) != 0) continue;
      char *numEnd = nullptr;
      long nodeId = std::strtol(entry->d_name + 4, &numEnd, 10);
      if (numEnd == entry->d_name + 4) continue;  // not a numeric suffix

      std::string cpulistPath = std::string("/sys/devices/system/node/") +
                                entry->d_name + "/cpulist";
      std::ifstream f(cpulistPath);
      if (!f.is_open()) continue;

      std::string cpulist;
      std::getline(f, cpulist);
      // Strip trailing whitespace
      while (!cpulist.empty() && (cpulist.back() == '\n' || cpulist.back() == '\r'))
        cpulist.pop_back();
      if (cpulist.empty()) continue;

      NodeInfo ni;
      ni.nodeId = static_cast<int>(nodeId);
      parseCpuList(cpulist, ni.cpuSet);
      nodes.push_back(ni);
    }
    closedir(dir);
    std::sort(nodes.begin(), nodes.end(),
              [](const NodeInfo &a, const NodeInfo &b) { return a.nodeId < b.nodeId; });
  }

  if (nodes.empty()) {
    VGRE_LOG_DEBUG("Scheduler", "NUMA: no topology found — workers use any CPU");
    return;
  }

  VGRE_LOG_INFO("Scheduler",
                "NUMA topology: " + std::to_string(nodes.size()) + " node(s) detected");

  // Assign workers to NUMA nodes round-robin and pin via pthread affinity.
  int numNodes = static_cast<int>(nodes.size());
  for (int i = 0; i < numThreads_ && i < static_cast<int>(workers_.size()); ++i) {
    int nodeIdx  = i % numNodes;
    int nodeId   = nodes[nodeIdx].nodeId;
    workerNumaNodes_[i] = nodeId;

    int rc = pthread_setaffinity_np(workers_[i].native_handle(),
                                    sizeof(cpu_set_t),
                                    &nodes[nodeIdx].cpuSet);
    if (rc != 0) {
      VGRE_LOG_WARN("Scheduler",
                    "pthread_setaffinity_np failed for worker " + std::to_string(i) +
                    " (node " + std::to_string(nodeId) + "): errno=" + std::to_string(rc));
    } else {
      VGRE_LOG_DEBUG("Scheduler",
                     "Worker " + std::to_string(i) + " pinned to NUMA node " +
                     std::to_string(nodeId));
    }
  }
#else
  // Non-Linux: no NUMA pinning; workerNumaNodes_ stays -1 for all workers.
  VGRE_LOG_DEBUG("Scheduler", "NUMA thread pinning not supported on this platform");
#endif
}

// ── Worker loop ────────────────────────────────────────────────────────────
void Scheduler::workerLoop(int workerIdx) {
  // My NUMA node, or -1 if topology is unavailable.
  int myNode = (workerIdx >= 0 && workerIdx < static_cast<int>(workerNumaNodes_.size()))
               ? workerNumaNodes_[workerIdx]
               : -1;

  while (true) {
    WorkItem item;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this, myNode] {
        if (shutdown_) return true;
        if (!queue_.empty()) return true;
        if (myNode >= 0) {
          auto it = numaQueues_.find(myNode);
          if (it != numaQueues_.end() && !it->second.empty()) return true;
        }
        return false;
      });

      // Check all queues empty for shutdown
      bool allEmpty = queue_.empty();
      if (allEmpty && myNode >= 0) {
        auto it = numaQueues_.find(myNode);
        if (it != numaQueues_.end() && !it->second.empty()) allEmpty = false;
      }
      if (shutdown_ && allEmpty)
        return;

      // NUMA-local queue takes priority over the global work-stealing queue.
      if (myNode >= 0) {
        auto it = numaQueues_.find(myNode);
        if (it != numaQueues_.end() && !it->second.empty()) {
          item = it->second.top();
          it->second.pop();
        } else if (!queue_.empty()) {
          item = queue_.top();
          queue_.pop();
        } else {
          continue;  // Spurious wakeup
        }
      } else {
        if (queue_.empty()) continue;
        item = queue_.top();
        queue_.pop();
      }
    }

    // Execute the generic task for this work-item (outside mutex!)
    try {
      if (item.execute) {
        item.execute();
      }
    } catch (...) {
      // Prevent worker death on unexpected task exceptions.
      VGRE_LOG_ERROR("Scheduler", "Unhandled exception in scheduled task");
    }

    if (pending_.load() > 0) {
      pending_--;
    }
    completed_++;
    cv_.notify_all(); // Wake waitStream/waitAll
  }
}

// ── Submit Generic Stream Task ─────────────────────────────────────────────
std::future<VGREResult>
Scheduler::submitStreamTask(StreamId stream, std::function<void()> taskFn,
                            int priority) {
  if (!taskFn) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERROR_INVALID_VALUE);
    return p.get_future();
  }
  if (shutdown_.load()) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERROR_NOT_INITIALIZED);
    return p.get_future();
  }

  auto node = std::make_shared<StreamTaskNode>();
  node->task = std::move(taskFn);
  auto future = node->promise.get_future();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_.load()) {
      node->promise.set_value(VGREResult::ERROR_NOT_INITIALIZED);
      return future;
    }
    auto &sq = streamQueues_[stream];
    if (!sq) {
      sq = std::make_shared<StreamQueue>();
      sq->streamPriority = priority;
    } else {
      sq->streamPriority = priority;
    }

    sq->pendingTasks.push(node);
    if (!sq->isProcessing) {
      sq->isProcessing = true;
      tryProcessStream(stream);
    }
  }

  return future;
}

void Scheduler::tryProcessStream(StreamId stream) {
  // Note: Called with global mutex_ already locked.
  auto it = streamQueues_.find(stream);
  if (it == streamQueues_.end())
    return;

  auto &sq = *it->second;
  if (sq.pendingTasks.empty()) {
    sq.isProcessing = false;
    cv_.notify_all();
    return;
  }

  auto node = sq.pendingTasks.front();

  WorkItem item;
  item.streamId = stream;
  item.priority = sq.streamPriority;
  item.execute = [node, stream, this]() {
    // Execute the actual kernel/task work
    try {
      if (node->task) {
        node->task();
      }
      try {
        node->promise.set_value(VGREResult::SUCCESS);
      } catch (...) {
      }
    } catch (...) {
      try {
        node->promise.set_value(VGREResult::ERROR_LAUNCH_FAILURE);
      } catch (...) {
      }
    }

    // Chain next task: lock mutex, pop completed, schedule next
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      auto it = this->streamQueues_.find(stream);
      if (it == this->streamQueues_.end() || !it->second) {
        return;
      }
      auto &sq2 = *it->second;
      if (sq2.pendingTasks.empty()) {
        sq2.isProcessing = false;
        this->cv_.notify_all();
        return;
      }
      sq2.pendingTasks.pop();
      sq2.isProcessing = false;
      this->tryProcessStream(stream);
    }
  };

  queue_.push(std::move(item));
  pending_++;
  cv_.notify_all();
}

std::future<VGREResult>
Scheduler::submitConcurrentTask(std::function<void()> taskFn, int priority) {
  if (!taskFn) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERROR_INVALID_VALUE);
    return p.get_future();
  }
  if (shutdown_.load()) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERROR_NOT_INITIALIZED);
    return p.get_future();
  }

  auto node = std::make_shared<StreamTaskNode>();
  node->task = std::move(taskFn);
  auto future = node->promise.get_future();

  WorkItem item;
  item.streamId = 0;      // No stream association
  item.priority = priority;
  item.execute = [node]() {
    try {
      if (node->task) {
        node->task();
      }
      node->promise.set_value(VGREResult::SUCCESS);
    } catch (...) {
      node->promise.set_value(VGREResult::ERROR_LAUNCH_FAILURE);
    }
  };

  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(item));
    pending_++;
  }
  cv_.notify_all();

  return future;
}

// ── NUMA Task Submission ───────────────────────────────────────────────────
std::future<VGREResult>
Scheduler::submitNumaTask(StreamId stream, std::function<void()> taskFn,
                          int numaNode, int priority) {
  if (!taskFn) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERROR_INVALID_VALUE);
    return p.get_future();
  }
  if (shutdown_.load()) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERROR_NOT_INITIALIZED);
    return p.get_future();
  }

  auto node = std::make_shared<StreamTaskNode>();
  node->task = std::move(taskFn);
  auto future = node->promise.get_future();

  WorkItem item;
  item.streamId          = stream;
  item.priority          = priority;
  item.preferredNumaNode = numaNode;
  item.execute = [node]() {
    try {
      if (node->task) node->task();
      node->promise.set_value(VGREResult::SUCCESS);
    } catch (...) {
      node->promise.set_value(VGREResult::ERROR_LAUNCH_FAILURE);
    }
  };

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (numaNode >= 0) {
      numaQueues_[numaNode].push(std::move(item));
    } else {
      queue_.push(std::move(item));
    }
    pending_++;
  }
  cv_.notify_all();

  return future;
}

// ── Control ────────────────────────────────────────────────────────────────
void Scheduler::waitStream(StreamId stream) {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this, stream]() {
    auto it = streamQueues_.find(stream);
    if (it == streamQueues_.end())
      return true; // Stream never used
    auto &sq = *it->second;
    return !sq.isProcessing && sq.pendingTasks.empty();
  });
}

bool Scheduler::isStreamIdle(StreamId stream) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = streamQueues_.find(stream);
  if (it == streamQueues_.end())
    return true;
  auto &sq = *it->second;
  return !sq.isProcessing && sq.pendingTasks.empty();
}

// ── Control ────────────────────────────────────────────────────────────────
void Scheduler::waitAll() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this]() { return pending_.load() == 0; });
}

int Scheduler::getThreadCount() const { return numThreads_; }

void Scheduler::setThreadCount(int n) {
  if (n <= 0) {
    n = static_cast<int>(std::thread::hardware_concurrency());
    if (n <= 0)
      n = 4;
  }
  if (n == numThreads_)
    return;

  VGRE_LOG_INFO("Scheduler", "Resizing thread pool from " +
                                 std::to_string(numThreads_) + " to " +
                                 std::to_string(n) + " workers");

  // 1. Drain all pending work
  waitAll();

  // 2. Signal shutdown and join existing workers
  shutdown_ = true;
  cv_.notify_all();
  for (auto &w : workers_) {
    if (w.joinable())
      w.join();
  }
  workers_.clear();

  // 3. Spawn new pool
  shutdown_ = false;
  numThreads_ = n;
  workerNumaNodes_.assign(numThreads_, -1);
  for (int i = 0; i < numThreads_; ++i) {
    workers_.emplace_back([this, i]() {
      this->workerLoop(i);
    });
  }
  buildNumaTopology();

  VGRE_LOG_INFO("Scheduler", "Thread pool resized to " +
                                 std::to_string(numThreads_) + " workers");
}

// ── Statistics ─────────────────────────────────────────────────────────────
uint64_t Scheduler::getCompletedTasks() const { return completed_.load(); }
uint64_t Scheduler::getPendingTasks() const { return pending_.load(); }

// ── Singleton ──────────────────────────────────────────────────────────────
Scheduler &Scheduler::instance() {
  static Scheduler sched;
  return sched;
}

} // namespace core
} // namespace vgre
