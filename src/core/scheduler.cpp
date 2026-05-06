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
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <pthread.h>
#include <mach/thread_policy.h>
#include <mach/thread_act.h>
#endif

namespace vgre {
namespace core {

thread_local int t_workerIdx = -1;
namespace {

bool enqueueWithWorkerFallback(int workerIdx,
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

bool hasNumaNode(const std::vector<int>& workerNumaNodes, int numaNode) {
  if (numaNode < 0) {
    return false;
  }
  for (int node : workerNumaNodes) {
    if (node == numaNode) {
      return true;
    }
  }
  return false;
}

} // namespace

// ── Constructor ────────────────────────────────────────────────────────────
Scheduler::Scheduler(int numThreads) : numThreads_(numThreads) {
  if (numThreads_ <= 0) {
    numThreads_ = static_cast<int>(std::thread::hardware_concurrency());
    if (numThreads_ <= 0)
      numThreads_ = 4;
  }

  workerNumaNodes_.resize(numThreads_, -1);
  workerDeques_.resize(numThreads_);
  for (int i = 0; i < numThreads_; ++i) {
    workerDeques_[i] = std::make_unique<ChaseLevDeque<WorkItem*>>();
  }

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
#elif defined(_WIN32)
  // Windows: enumerate NUMA nodes via GetNumaHighestNodeNumber and
  // GetNumaNodeProcessorMaskEx, then set thread affinity with SetThreadAffinityMask.
  ULONG highestNode = 0;
  if (!GetNumaHighestNodeNumber(&highestNode) || highestNode == 0) {
    VGRE_LOG_DEBUG("Scheduler", "Windows NUMA: single node or detection failed");
    return;
  }

  struct WinNodeInfo {
    ULONG nodeId;
    ULONGLONG mask; // processor mask for this node
  };
  std::vector<WinNodeInfo> nodes;
  for (ULONG n = 0; n <= highestNode; ++n) {
    ULONGLONG mask = 0;
    if (GetNumaNodeProcessorMask(static_cast<UCHAR>(n), &mask) && mask != 0)
      nodes.push_back({n, mask});
  }
  if (nodes.empty()) {
    VGRE_LOG_DEBUG("Scheduler", "Windows NUMA: no usable nodes found");
    return;
  }

  VGRE_LOG_INFO("Scheduler",
                "NUMA topology (Windows): " + std::to_string(nodes.size()) + " node(s)");

  int numNodes = static_cast<int>(nodes.size());
  for (int i = 0; i < numThreads_ && i < static_cast<int>(workers_.size()); ++i) {
    int ni = i % numNodes;
    workerNumaNodes_[i] = static_cast<int>(nodes[ni].nodeId);
    HANDLE h = static_cast<HANDLE>(workers_[i].native_handle());
    if (SetThreadAffinityMask(h, static_cast<DWORD_PTR>(nodes[ni].mask)) == 0) {
      VGRE_LOG_WARN("Scheduler",
                    "SetThreadAffinityMask failed for worker " + std::to_string(i));
    } else {
      VGRE_LOG_DEBUG("Scheduler", "Worker " + std::to_string(i) +
                                      " pinned to NUMA node " +
                                      std::to_string(nodes[ni].nodeId));
    }
  }

#elif defined(__APPLE__)
  // macOS: no NUMA topology API; use physical vs efficiency cores via sysctl.
  // Detect P-core and E-core counts (Apple Silicon) and round-robin threads
  // across them.  Falls back gracefully to a single logical group on Intel.
  int physCpus = 0, perfCores = 0, effCores = 0;
  size_t sz = sizeof(int);
  sysctlbyname("hw.physicalcpu_max", &physCpus, &sz, nullptr, 0);
  sz = sizeof(int);
  sysctlbyname("hw.perflevel0.physicalcpu", &perfCores, &sz, nullptr, 0);
  sz = sizeof(int);
  sysctlbyname("hw.perflevel1.physicalcpu", &effCores, &sz, nullptr, 0);

  if (physCpus <= 0) {
    VGRE_LOG_DEBUG("Scheduler", "macOS NUMA: sysctl hw.physicalcpu_max unavailable");
    return;
  }

  VGRE_LOG_INFO("Scheduler",
                "macOS CPU topology: physicalcpu=" + std::to_string(physCpus) +
                    " perf=" + std::to_string(perfCores) +
                    " eff=" + std::to_string(effCores));

  // macOS does not expose fine-grained affinity pinning via public API.
  // Assign notional NUMA-node IDs: node 0 = P-cores, node 1 = E-cores.
  // Actual affinity tagging uses thread_policy_set with THREAD_AFFINITY_POLICY.
  int numGroups = (perfCores > 0 && effCores > 0) ? 2 : 1;
  for (int i = 0; i < numThreads_ && i < static_cast<int>(workers_.size()); ++i) {
    int groupIdx = i % numGroups;
    workerNumaNodes_[i] = groupIdx;

    thread_affinity_policy_data_t policy = {groupIdx + 1}; // tag ≥ 1
    thread_port_t mach_thread = pthread_mach_thread_np(workers_[i].native_handle());
    kern_return_t kr = thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                                         (thread_policy_t)&policy,
                                         THREAD_AFFINITY_POLICY_COUNT);
    if (kr != KERN_SUCCESS) {
      VGRE_LOG_DEBUG("Scheduler",
                     "macOS thread_policy_set failed for worker " + std::to_string(i) +
                         " (kr=" + std::to_string(kr) + ") — affinity hints only");
    } else {
      VGRE_LOG_DEBUG("Scheduler", "Worker " + std::to_string(i) +
                                      " assigned affinity group " +
                                      std::to_string(groupIdx));
    }
  }

#else
  VGRE_LOG_DEBUG("Scheduler", "NUMA thread pinning not supported on this platform");
#endif
}

// ── Worker loop ────────────────────────────────────────────────────────────
void Scheduler::workerLoop(int workerIdx) {
  t_workerIdx = workerIdx;
  int myNode = (workerIdx >= 0 && workerIdx < static_cast<int>(workerNumaNodes_.size()))
               ? workerNumaNodes_[workerIdx]
               : -1;

  while (true) {
    // item holds the next work unit by value — no heap allocation needed for the
    // global-queue path.  Items from the Chase-Lev deques are heap-allocated by
    // the submitters; we move from them and then delete the pointer immediately.
    WorkItem item;
    bool gotItem = false;

    // 1. Pop from own deque (lock-free, O(1))
    {
      WorkItem* p = nullptr;
      if (workerDeques_[workerIdx]->pop(p) && p) {
        item = std::move(*p);
        delete p;
        gotItem = true;
      }
    }

    // 2. Work-stealing from other deques (lock-free)
    if (!gotItem) {
      for (int i = 1; i < numThreads_; ++i) {
        WorkItem* p = nullptr;
        int targetIdx = (workerIdx + i) % numThreads_;
        if (workerDeques_[targetIdx]->steal(p) && p) {
          item = std::move(*p);
          delete p;
          gotItem = true;
          break;
        }
      }
    }

    // 3. Fallback: wait on global / NUMA-local priority queue
    if (!gotItem) {
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

      bool allEmpty = queue_.empty();
      if (allEmpty && myNode >= 0) {
        auto it = numaQueues_.find(myNode);
        if (it != numaQueues_.end() && !it->second.empty()) allEmpty = false;
      }
      if (shutdown_ && allEmpty) return;

      // Move item off the priority queue — no heap allocation.
      // priority_queue::top() returns const ref; const_cast + move is safe here
      // because we call pop() immediately after, so the element is logically gone.
      if (myNode >= 0) {
        auto it = numaQueues_.find(myNode);
        if (it != numaQueues_.end() && !it->second.empty()) {
          item = std::move(const_cast<WorkItem&>(it->second.top()));
          it->second.pop();
          gotItem = true;
        } else if (!queue_.empty()) {
          item = std::move(const_cast<WorkItem&>(queue_.top()));
          queue_.pop();
          gotItem = true;
        }
      } else if (!queue_.empty()) {
        item = std::move(const_cast<WorkItem&>(queue_.top()));
        queue_.pop();
        gotItem = true;
      }
    }

    if (!gotItem) continue;

    // Execute the generic task for this work-item (outside mutex!)
    try {
      if (item.execute) {
        // v0.1.3 Extraordinary Sophistication: Temporary Affinity Yielding
        // If this worker is pinned to a specific core, and it launches an 
        // OpenMP task, all OMP threads will be forced onto that same core.
        // We yield affinity here so the OMP runtime can see and use all cores.
        bool affinityShifted = false;
#if defined(__linux__)
        cpu_set_t oldAffinity;
        if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &oldAffinity) == 0) {
            cpu_set_t allCores;
            CPU_ZERO(&allCores);
            for (int k = 0; k < CPU_SETSIZE; k++) CPU_SET(k, &allCores);
            if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &allCores) == 0) {
                affinityShifted = true;
            }
        }
#endif

        item.execute();

        // Restore pinning for next scheduler work selection
        if (affinityShifted) {
#if defined(__linux__)
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &oldAffinity);
#endif
        }
      }
    } catch (...) {
      // Prevent worker death on unexpected task exceptions.
      VGRE_LOG_ERROR("Scheduler", "Unhandled exception in scheduled task");
    }

    pending_.fetch_sub(1, std::memory_order_acq_rel);
    completed_.fetch_add(1, std::memory_order_relaxed);
    cv_.notify_all(); // Wake waitStream/waitAll
  }
}

// ── Submit Generic Stream Task ─────────────────────────────────────────────
std::future<VGREResult>
Scheduler::submitStreamTask(StreamId stream, std::function<void()> taskFn,
                            int priority) {
  if (!taskFn) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERR_INVALID_VALUE);
    return p.get_future();
  }
  if (shutdown_.load()) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERR_NOT_INITIALIZED);
    return p.get_future();
  }

  auto node = std::make_shared<StreamTaskNode>();
  node->task = std::move(taskFn);
  auto future = node->promise.get_future();

  std::shared_ptr<StreamQueue> sq;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_.load()) {
      node->promise.set_value(VGREResult::ERR_NOT_INITIALIZED);
      return future;
    }
    sq = streamQueues_[stream];
    if (!sq) {
      sq = std::make_shared<StreamQueue>();
      streamQueues_[stream] = sq;
    }
  }

  {
    std::lock_guard<std::mutex> sqLock(sq->streamMutex);
    sq->streamPriority = priority;
    sq->pendingTasks.push(node);
    pending_++; // Increment immediately to prevent waitAll() returning early
    if (!sq->isProcessing) {
      sq->isProcessing = true;
      VGRE_LOG_DEBUG("Scheduler", "Started processing stream " + std::to_string(stream));
      tryProcessStream(sq, stream);
    }
  }

  VGRE_LOG_DEBUG("Scheduler", "Task submitted to stream " + std::to_string(stream));
  return future;
}

void Scheduler::tryProcessStream(std::shared_ptr<StreamQueue> sq, StreamId stream) {
  // Note: Called with sq->streamMutex already locked.
  if (sq->pendingTasks.empty()) {
    sq->isProcessing = false;
    cv_.notify_all();
    return;
  }

  auto node = sq->pendingTasks.front();
  VGRE_LOG_DEBUG("Scheduler", "Dequeueing task from stream " + std::to_string(stream));

  WorkItem item;
  item.streamId = stream;
  item.priority = sq->streamPriority;
  item.execute = [node, stream, sq, this]() {
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
        node->promise.set_value(VGREResult::ERR_LAUNCH_FAILURE);
      } catch (...) {
      }
    }

    // Chain next task: lock sq mutex, pop completed, schedule next
    {
      std::lock_guard<std::mutex> sqLock(sq->streamMutex);
      if (sq->pendingTasks.empty()) {
        sq->isProcessing = false;
        this->cv_.notify_all();
        return;
      }
      sq->pendingTasks.pop();
      sq->isProcessing = false;
      if (!sq->pendingTasks.empty()) {
        sq->isProcessing = true;
        this->tryProcessStream(sq, stream);
      } else {
        this->cv_.notify_all();
      }
    }
  };

  enqueueWithWorkerFallback(t_workerIdx, std::move(item), workerDeques_, queue_, mutex_);
  cv_.notify_all();
}

std::future<VGREResult>
Scheduler::submitConcurrentTask(std::function<void()> taskFn, int priority) {
  if (!taskFn) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERR_INVALID_VALUE);
    return p.get_future();
  }
  if (shutdown_.load()) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERR_NOT_INITIALIZED);
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
      node->promise.set_value(VGREResult::ERR_LAUNCH_FAILURE);
    }
  };

  enqueueWithWorkerFallback(t_workerIdx, std::move(item), workerDeques_, queue_, mutex_);
  pending_++;
  cv_.notify_all();

  return future;
}

// ── NUMA Task Submission ───────────────────────────────────────────────────
std::future<VGREResult>
Scheduler::submitNumaTask(StreamId stream, std::function<void()> taskFn,
                          int numaNode, int priority) {
  if (!taskFn) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERR_INVALID_VALUE);
    return p.get_future();
  }
  if (shutdown_.load()) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERR_NOT_INITIALIZED);
    return p.get_future();
  }

  auto node = std::make_shared<StreamTaskNode>();
  node->task = std::move(taskFn);
  auto future = node->promise.get_future();

  bool numaNodeValid = hasNumaNode(workerNumaNodes_, numaNode);
  if (numaNode >= 0 && !numaNodeValid) {
    VGRE_LOG_WARN("Scheduler",
                  "Invalid NUMA node " + std::to_string(numaNode) +
                  " for submitNumaTask; falling back to global queue");
    numaNode = -1;
  }

  WorkItem item;
  item.streamId          = stream;
  item.priority          = priority;
  item.preferredNumaNode = numaNode;
  item.execute = [node]() {
    try {
      if (node->task) node->task();
      node->promise.set_value(VGREResult::SUCCESS);
    } catch (...) {
      node->promise.set_value(VGREResult::ERR_LAUNCH_FAILURE);
    }
  };

  if (t_workerIdx != -1 && t_workerIdx < numThreads_) {
    enqueueWithWorkerFallback(t_workerIdx, std::move(item), workerDeques_, queue_, mutex_);
  } else {
    std::lock_guard<std::mutex> lock(mutex_);
    if (numaNode >= 0) {
      numaQueues_[numaNode].push(std::move(item));
    } else {
      queue_.push(std::move(item));
    }
  }
  pending_++;
  cv_.notify_all();

  return future;
}

// ── Control ────────────────────────────────────────────────────────────────
void Scheduler::waitStream(StreamId stream) {
  VGRE_LOG_DEBUG("Scheduler", "Waiting for stream " + std::to_string(stream));
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this, stream]() {
    auto it = streamQueues_.find(stream);
    if (it == streamQueues_.end())
      return true; // Stream never used
    auto &sq = *it->second;
    bool done = !sq.isProcessing && sq.pendingTasks.empty();
    if (!done) {
        VGRE_LOG_DEBUG("Scheduler", "Stream " + std::to_string(stream) + " still busy: isProcessing=" + 
                       (sq.isProcessing ? "true" : "false") + " pending=" + std::to_string(sq.pendingTasks.size()));
    }
    return done;
  });
  VGRE_LOG_DEBUG("Scheduler", "Stream " + std::to_string(stream) + " synchronization complete");
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
  workerDeques_.clear();
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

  VGRE_LOG_INFO("Scheduler", "Thread pool resized to " +
                                 std::to_string(numThreads_) + " workers");
}

// ── Statistics ─────────────────────────────────────────────────────────────
uint64_t Scheduler::getCompletedTasks() const { return completed_.load(); }
uint64_t Scheduler::getPendingTasks() const { return pending_.load(); }

// ── Singleton ──────────────────────────────────────────────────────────────
Scheduler &Scheduler::instance() {
  static Scheduler inst;
  return inst;
}

} // namespace core
} // namespace vgre
