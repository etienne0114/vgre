#include "vgre/core/scheduler.h"
#include "vgre/common/logger.h"

#include <sstream>
#include <cstring>

#include "vgre/common/os_backend.h"
#if defined(__linux__)
#include <dirent.h>   // readdir — NUMA /sys/devices/system/node scan
#include <fstream>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/thread_policy.h>
#include <mach/thread_act.h>
#include <pthread.h>            // pthread_mach_thread_np
// macOS exposes no Linux-style NUMA topology and removed hard CPU pinning
// (pthread_setaffinity_np). Instead we use Mach THREAD_AFFINITY_POLICY tags to
// GROUP workers, split by Apple-Silicon performance (perflevel0) vs efficiency
// (perflevel1) cores so same-group workers share an L2 affinity hint. This is a
// scheduler *hint*, not a hard pin — the documented macOS boundary.
#endif

namespace vgre {
namespace core {

// ── NUMA Topology Discovery ────────────────────────────────────────────────
void Scheduler::buildNumaTopology() {
#if defined(__linux__)
  struct NodeInfo {
    int nodeId;
    cpu_set_t cpuSet;
  };

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

  std::vector<NodeInfo> nodes;
  DIR *dir = opendir("/sys/devices/system/node");
  if (dir) {
    struct dirent *entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
      if (std::strncmp(entry->d_name, "node", 4) != 0) continue;
      char *numEnd = nullptr;
      long nodeId = std::strtol(entry->d_name + 4, &numEnd, 10);
      if (numEnd == entry->d_name + 4) continue;

      std::string cpulistPath = std::string("/sys/devices/system/node/") +
                                entry->d_name + "/cpulist";
      std::ifstream f(cpulistPath);
      if (!f.is_open()) continue;

      std::string cpulist;
      std::getline(f, cpulist);
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

  int numNodes = static_cast<int>(nodes.size());
  for (int i = 0; i < numThreads_ && i < static_cast<int>(workers_.size()); ++i) {
    int nodeIdx  = i % numNodes;
    int nodeId   = nodes[nodeIdx].nodeId;
    // Atomic release store: pairs with the acquire load in workerLoop() so the
    // benign NUMA-hint read there is race-detector-clean (workers were already
    // spawned — affinity pinning below needs their handles).
    __atomic_store_n(&workerNumaNodes_[i], nodeId, __ATOMIC_RELEASE);
    workerNumaNodeSet_.insert(nodeId);

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
  ULONG highestNode = 0;
  if (!GetNumaHighestNodeNumber(&highestNode) || highestNode == 0) {
    VGRE_LOG_DEBUG("Scheduler", "Windows NUMA: single node or detection failed");
    return;
  }

  struct WinNodeInfo {
    ULONG nodeId;
    ULONGLONG mask;
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
    __atomic_store_n(&workerNumaNodes_[i], static_cast<int>(nodes[ni].nodeId), __ATOMIC_RELEASE);
    workerNumaNodeSet_.insert(static_cast<int>(nodes[ni].nodeId));
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

  int numGroups = (perfCores > 0 && effCores > 0) ? 2 : 1;
  for (int i = 0; i < numThreads_ && i < static_cast<int>(workers_.size()); ++i) {
    int groupIdx = i % numGroups;
    __atomic_store_n(&workerNumaNodes_[i], groupIdx, __ATOMIC_RELEASE);
    workerNumaNodeSet_.insert(groupIdx);

    thread_affinity_policy_data_t policy = {groupIdx + 1};
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

} // namespace core
} // namespace vgre
