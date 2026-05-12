/**
 * VGRE C API — Telemetry, Profiling, Bandwidth, Cluster Info Implementation
 */

#include "vgre/api/vgre_c_api.h"
#include <atomic>
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/runtime/gpu_cache.h"
#include "vgre/advanced/ipc_manager.h"
#include "vgre/advanced/resource_ledger.h"
#include "vgre/advanced/runtime_profiler.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/vgre_workload_engine.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/virtual_gpu_device.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <vector>

namespace {

static int to_status_tel(vgre::VGREResult r) {
  switch (r) {
  case vgre::VGREResult::SUCCESS:             return VGRE_SUCCESS;
  case vgre::VGREResult::ERR_OUT_OF_MEMORY:   return VGRE_ERROR_OUT_OF_MEMORY;
  case vgre::VGREResult::ERR_INVALID_VALUE:   return VGRE_ERROR_INVALID_VALUE;
  case vgre::VGREResult::ERR_INVALID_KERNEL:  return VGRE_ERROR_INVALID_KERNEL;
  case vgre::VGREResult::ERR_LAUNCH_FAILURE:  return VGRE_ERROR_LAUNCH_FAILURE;
  case vgre::VGREResult::ERR_IO:              return VGRE_ERROR_IO;
  case vgre::VGREResult::ERR_NOT_INITIALIZED: return VGRE_ERROR_NOT_INIT;
  case vgre::VGREResult::ERR_AUTH_FAILED:     return VGRE_ERROR_AUTH_FAILED;
  case vgre::VGREResult::ERR_CRYPTO:          return VGRE_ERROR_CRYPTO;
  default:                                    return VGRE_ERROR_GENERIC;
  }
}

static int require_initialized_tel() {
  if (!vgre::core::RuntimeEngine::instance().isInitialized())
    return VGRE_ERROR_NOT_INIT;
  return VGRE_SUCCESS;
}

static uint64_t telemetry_now_ms() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

static std::string json_escape(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
    case '\\': out += "\\\\"; break;
    case '"':  out += "\\\""; break;
    case '\n': out += "\\n";  break;
    case '\r': out += "\\r";  break;
    case '\t': out += "\\t";  break;
    default:   out += c;      break;
    }
  }
  return out;
}

} // anonymous namespace

/* ── Telemetry ──────────────────────────────────────────────────────────────
 */

int vgre_get_telemetry(vgre_telemetry_t *telemetry) {
  if (!telemetry)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized_tel(); s != VGRE_SUCCESS)
    return s;
  std::memset(telemetry, 0, sizeof(*telemetry));

  auto &ae = vgre::advanced::AdaptiveExecutionEngine::instance();
  auto &mm = vgre::core::RuntimeEngine::instance().getMemoryManager();
  auto &profiler = vgre::advanced::RuntimeProfiler::instance();

  telemetry->timestamp = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  telemetry->version_major = 0;
  telemetry->version_minor = 1;
  telemetry->version_patch = 1;

  // GFLOPS
  ae.updateInstantaneousMetrics();
  telemetry->gflops = ae.getInstantaneousGFLOPS();
  telemetry->max_gflops = ae.getMaxGFLOPS();
  telemetry->compute_utilization =
      (telemetry->max_gflops > 0.1)
          ? (telemetry->gflops / telemetry->max_gflops) * 100.0
          : 0.0;
  if (telemetry->compute_utilization < 0.0)
    telemetry->compute_utilization = 0.0;
  if (telemetry->compute_utilization > 100.0)
    telemetry->compute_utilization = 100.0;

  // Memory
  telemetry->memory_bandwidth_gbps = ae.getInstantaneousBandwidth();
  telemetry->max_memory_bandwidth_gbps = ae.getMaxMemoryBandwidth();
  telemetry->memory_bus_utilization =
      (telemetry->max_memory_bandwidth_gbps > 0.1)
          ? (telemetry->memory_bandwidth_gbps /
             telemetry->max_memory_bandwidth_gbps) *
                100.0
          : 0.0;
  if (telemetry->memory_bus_utilization < 0.0)
    telemetry->memory_bus_utilization = 0.0;
  if (telemetry->memory_bus_utilization > 100.0)
    telemetry->memory_bus_utilization = 100.0;

  telemetry->memory_used_bytes = mm.getUsedMemory();
  telemetry->memory_total_bytes = mm.getTotalMemory();

  // UVM Stats
  telemetry->total_pages = mm.getTotalMemory() / 4096;
  mm.getPageResidency(telemetry->uvm_map);

  // Normalize resident pages for the 1024-cell UI grid
  int residentCells = mm.getResidentPageCount();
  if (residentCells < 0)
    residentCells = 0;
  if (residentCells > 1024)
    residentCells = 1024;
  telemetry->resident_pages =
      (telemetry->total_pages * static_cast<uint64_t>(residentCells)) / 1024;
  if (telemetry->resident_pages > telemetry->total_pages) {
    telemetry->resident_pages = telemetry->total_pages;
  }
  telemetry->evicted_pages = telemetry->total_pages - telemetry->resident_pages;
  telemetry->page_faults_per_sec = static_cast<double>(mm.getPageFaultRate());

  // Device Stats
  auto &sched = vgre::core::Scheduler::instance();
  auto pendingTasks = sched.getPendingTasks();
  telemetry->active_kernels =
      pendingTasks > 0 ? static_cast<int64_t>(pendingTasks)
                       : static_cast<int64_t>(ae.getActiveKernelCount());
  auto threadCount = sched.getThreadCount();
  telemetry->active_threads =
      pendingTasks > 0
          ? static_cast<int64_t>(
                std::min<uint64_t>(pendingTasks,
                                   static_cast<uint64_t>(threadCount)))
          : 0;

  // Real device properties
  auto &dev = vgre::core::RuntimeEngine::instance().getDevice();
  auto props = dev.getProperties();
  telemetry->device_clock_mhz = static_cast<double>(props.clockRate) / 1000.0;

  // Real ECC reporting: DISABLED for Intel Integrated Graphics
  bool is_intel = std::string(props.name).find("Intel") != std::string::npos;
  telemetry->ecc_enabled = is_intel ? 0 : (props.major >= 7 ? 1 : 0);

  telemetry->avg_kernel_latency_ms = ae.getAvgLatencyMs();

  // If runtime profiler is enabled, prefer measured averages.
  if (profiler.isEnabled()) {
    auto stats = profiler.getAllStats();
    if (!stats.empty()) {
      double totalInv = 0.0;
      double avgMs = 0.0;
      for (const auto &s : stats) {
        totalInv += static_cast<double>(s.invocations);
        avgMs += s.avgTimeMs * static_cast<double>(s.invocations);
      }
      if (totalInv > 0.0) {
        // We use instantaneous rates for the main gauges (Source of Truth),
        // but the profiler still provides the authoritative average latency.
        telemetry->avg_kernel_latency_ms = avgMs / totalInv;
        telemetry->compute_utilization =
            (telemetry->max_gflops > 0.0)
                ? (telemetry->gflops / telemetry->max_gflops) * 100.0
                : 0.0;
        if (telemetry->compute_utilization < 0.0)
          telemetry->compute_utilization = 0.0;
        if (telemetry->compute_utilization > 100.0)
          telemetry->compute_utilization = 100.0;

        telemetry->memory_bus_utilization =
            (telemetry->max_memory_bandwidth_gbps > 0.0)
                ? (telemetry->memory_bandwidth_gbps /
                   telemetry->max_memory_bandwidth_gbps) *
                      100.0
                : 0.0;
        if (telemetry->memory_bus_utilization < 0.0)
          telemetry->memory_bus_utilization = 0.0;
        if (telemetry->memory_bus_utilization > 100.0)
          telemetry->memory_bus_utilization = 100.0;
      }
    }
  }

  // Smooth jitter for dashboard consumption (EMA).
  // Keeps UI stable without hiding trend direction.
  {
    static std::mutex s_telemetry_mutex;
    static bool s_has_prev = false;
    static double s_gflops = 0.0;
    static double s_bw = 0.0;
    static double s_latency = 0.0;
    static uint64_t s_last_ts = 0;

    std::lock_guard<std::mutex> lock(s_telemetry_mutex);
    const uint64_t ts = telemetry->timestamp;
    const double alpha = 0.35;

    if (!s_has_prev) {
      s_gflops = telemetry->gflops;
      s_bw = telemetry->memory_bandwidth_gbps;
      s_latency = telemetry->avg_kernel_latency_ms;
      s_last_ts = ts;
      s_has_prev = true;
    } else if (ts >= s_last_ts) {
      s_gflops = s_gflops * (1.0 - alpha) + telemetry->gflops * alpha;
      s_bw = s_bw * (1.0 - alpha) + telemetry->memory_bandwidth_gbps * alpha;
      s_latency =
          s_latency * (1.0 - alpha) + telemetry->avg_kernel_latency_ms * alpha;
      s_last_ts = ts;
    }

    telemetry->gflops = s_gflops;
    telemetry->memory_bandwidth_gbps = s_bw;
    telemetry->avg_kernel_latency_ms = s_latency;

    telemetry->compute_utilization =
        (telemetry->max_gflops > 0.0)
            ? (telemetry->gflops / telemetry->max_gflops) * 100.0
            : 0.0;
    if (telemetry->compute_utilization < 0.0)
      telemetry->compute_utilization = 0.0;
    if (telemetry->compute_utilization > 100.0)
      telemetry->compute_utilization = 100.0;

    telemetry->memory_bus_utilization =
        (telemetry->max_memory_bandwidth_gbps > 0.0)
            ? (telemetry->memory_bandwidth_gbps /
               telemetry->max_memory_bandwidth_gbps) *
                  100.0
            : 0.0;
    if (telemetry->memory_bus_utilization < 0.0)
      telemetry->memory_bus_utilization = 0.0;
    if (telemetry->memory_bus_utilization > 100.0)
      telemetry->memory_bus_utilization = 100.0;
  }
  telemetry->device_temperature =
      static_cast<double>(ae.getDeviceTemperature());
  telemetry->background_compute_active = static_cast<int64_t>(
      vgre::advanced::WorkloadEngine::instance().isEnabled() ? 1 : 0);

  // Add IPC aggregation if we are the master (Dashboard)
  auto &ipc = vgre::advanced::IPCManager::instance();
  if (ipc.isEnabled()) {
    // 1. Update our slot in shared memory with current local stats
    ipc.updateLocalTelemetry(*telemetry);
    // 2. Aggregate global stats (local + other processes + remote cluster)
    ipc.getGlobalTelemetry(*telemetry);
  }

  // Add TCP Cluster aggregation and SHM synchronization
  auto &tcp = vgre::advanced::TCPClusterManager::instance();
  if (tcp.isEnabled()) {
    if (tcp.isMaster()) {
      tcp.aggregateRemoteTelemetry(*telemetry);

      // Phase 12: Sync cluster topology to SHM for Dashboard visibility
      std::vector<vgre::advanced::TCPClusterManager::ClusterNodeInfo> connections;
      tcp.getConnectedNodes(connections);
      std::vector<vgre_cluster_node_t> to_sync;
      for (const auto &conn : connections) {
          vgre_cluster_node_t node{};
          std::strncpy(node.address, conn.ip_address.c_str(), sizeof(node.address) - 1);
          node.port = conn.port;
          node.cpu_cores = conn.cpu_cores;
          node.memory_bytes = conn.cpu_memory;
          node.latency_ms = conn.last_telemetry.avg_kernel_latency_ms;
          node.available = conn.active ? 1 : 0;
          std::strncpy(node.igpu_name, conn.igpu_name, sizeof(node.igpu_name) - 1);
          to_sync.push_back(node);
      }
      ipc.updateClusterNodes(to_sync);
    } else {
      tcp.broadcastLocalTelemetry(*telemetry);
    }
  }

  return VGRE_SUCCESS;
}

int vgre_get_memory_info_json(char **out_json) {
  if (!out_json)
    return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized_tel(); s != VGRE_SUCCESS)
    return s;

  auto &mm = vgre::core::MemoryManager::instance();

  std::stringstream ss;
  ss << "{\"allocations\":[";

  bool first = true;
  for (const auto& [handle, alloc] : mm.getAllocations()) {
    if (!first) ss << ",";
    ss << "{\"ptr\":\"" << handle << "\","
       << "\"size\":" << alloc.size << ","
       << "\"managed\":" << (alloc.isManaged ? "true" : "false") << ","
       << "\"resident\":" << (alloc.isResidentOnHost ? "true" : "false") << ","
       << "\"device\":" << alloc.deviceId << "}";
    first = false;
  }

  ss << "],\"pools\":[";
  first = true;
  for (const auto& [id, pool] : mm.getPools()) {
    if (!first) ss << ",";
    uint64_t activeCount = (pool.allocCount >= pool.freeCount) ? (pool.allocCount - pool.freeCount) : 0;
    ss << "{\"id\":" << id << ","
       << "\"blockSize\":" << pool.blockSize << ","
       << "\"total\":" << pool.totalAllocated << ","
       << "\"peak\":" << pool.peakAllocated << ","
       << "\"active\":" << activeCount << ","
       << "\"free\":" << pool.freeCount << "}";
    first = false;
  }
  ss << "]}";

  std::string s = ss.str();
  *out_json = (char *)std::malloc(s.size() + 1);
  if (!*out_json) return VGRE_ERROR_OUT_OF_MEMORY;
  std::strcpy(*out_json, s.c_str());
  return VGRE_SUCCESS;
}

int vgre_get_logs(char ***buffer, int *count) {
  if (!buffer || !count)
    return VGRE_ERROR_INVALID_VALUE;

  auto logs = vgre::Logger::instance().getRecentLogs();
  if (logs.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return VGRE_ERROR_OUT_OF_MEMORY;
  }
  *count = static_cast<int>(logs.size());

  if (*count == 0) {
    *buffer = nullptr;
    return VGRE_SUCCESS;
  }

  char **lines = (char **)malloc(sizeof(char *) * (*count));
  if (!lines) {
    *buffer = nullptr;
    *count = 0;
    return VGRE_ERROR_OUT_OF_MEMORY;
  }
  for (int i = 0; i < *count; ++i) {
    const auto &line = logs[static_cast<size_t>(i)];
    size_t len = line.size();
    lines[i] = static_cast<char *>(malloc(len + 1));
    if (lines[i]) {
      std::memcpy(lines[i], line.c_str(), len + 1);
    }
    if (!lines[i]) {
      for (int j = 0; j < i; ++j) {
        free(lines[j]);
      }
      free(lines);
      *buffer = nullptr;
      *count = 0;
      return VGRE_ERROR_OUT_OF_MEMORY;
    }
  }

  *buffer = lines;
  return VGRE_SUCCESS;
}

void vgre_free_logs(char **buffer, int count) {
  if (!buffer)
    return;
  for (int i = 0; i < count; ++i) {
    free(buffer[i]);
  }
  free(buffer);
}

int vgre_get_profiler_json(char **out_json, int top_n) {
  if (!out_json)
    return VGRE_ERROR_INVALID_VALUE;

  auto &profiler = vgre::advanced::RuntimeProfiler::instance();
  if (!profiler.isEnabled()) {
    *out_json = nullptr;
    return VGRE_SUCCESS;
  }

  auto stats = profiler.getAllStats();
  if (top_n > 0 && static_cast<size_t>(top_n) < stats.size()) {
    stats.resize(static_cast<size_t>(top_n));
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4);
  oss << "{\n";
  oss << "  \"timestamp_ms\": " << telemetry_now_ms() << ",\n";
  oss << "  \"total_kernels\": " << stats.size() << ",\n";
  oss << "  \"top_kernels\": [\n";

  for (size_t i = 0; i < stats.size(); ++i) {
    const auto &s = stats[i];
    oss << "    {\n";
    oss << "      \"name\": \"" << json_escape(s.kernelName) << "\",\n";
    oss << "      \"invocations\": " << s.invocations << ",\n";
    oss << "      \"total_time_ms\": " << s.totalTimeMs << ",\n";
    oss << "      \"avg_time_ms\": " << s.avgTimeMs << ",\n";
    oss << "      \"min_time_ms\": " << s.minTimeMs << ",\n";
    oss << "      \"max_time_ms\": " << s.maxTimeMs << ",\n";
      oss << "      \"avg_throughput_gbps\": " << s.avgThroughputGBps << ",\n";
      oss << "      \"avg_gflops\": " << s.avgGflops << ",\n";
      oss << "      \"source_code\": \"" << json_escape(s.sourceCode) << "\",\n";
      oss << "      \"ir_code\": \"" << json_escape(s.irCode) << "\"\n";
      oss << "    }" << (i + 1 < stats.size() ? "," : "") << "\n";
  }
  oss << "  ]\n";
  oss << "}\n";

  const std::string json = oss.str();
  char *buf = static_cast<char *>(malloc(json.size() + 1));
  if (!buf)
    return VGRE_ERROR_OUT_OF_MEMORY;
  std::memcpy(buf, json.c_str(), json.size() + 1);
  *out_json = buf;
  return VGRE_SUCCESS;
}

int vgre_get_kernel_history_json(const char *kernel_name, char **out_json) {
  if (!kernel_name || !out_json)
    return VGRE_ERROR_INVALID_VALUE;

  auto &profiler = vgre::advanced::RuntimeProfiler::instance();
  auto events = profiler.getEventsByKernel(kernel_name);

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4);
  oss << "[\n";
  for (size_t i = 0; i < events.size(); ++i) {
    const auto &ev = events[i];
    oss << "  {\n";
    oss << "    \"timestamp_ms\": " << ev.timestamp_ms << ",\n";
    oss << "    \"duration_ms\": " << ev.durationMs << ",\n";
    oss << "    \"throughput_gbps\": " << ev.throughputGBps << ",\n";
    oss << "    \"gflops\": " << ev.gflops << ",\n";
    oss << "    \"threads_used\": " << ev.threadsUsed << "\n";
    oss << "  }" << (i + 1 < events.size() ? "," : "") << "\n";
  }
  oss << "]\n";

  const std::string json = oss.str();
  char *buf = static_cast<char *>(malloc(json.size() + 1));
  if (!buf) return VGRE_ERROR_OUT_OF_MEMORY;
  std::memcpy(buf, json.c_str(), json.size() + 1);
  *out_json = buf;
  return VGRE_SUCCESS;
}

void vgre_free_string(char *str) {
  if (str)
    free(str);
}

int vgre_set_profiler_enabled(int enabled) {
  if (int s = require_initialized_tel(); s != VGRE_SUCCESS)
    return s;
  vgre::advanced::RuntimeProfiler::instance().setEnabled(enabled != 0);
  return VGRE_SUCCESS;
}

int vgre_set_background_compute(int enabled) {
  if (int s = require_initialized_tel(); s != VGRE_SUCCESS)
    return s;
  vgre::advanced::WorkloadEngine::instance().setEnabled(enabled != 0);
  return VGRE_SUCCESS;
}

int vgre_set_service_mode(int is_master) {
  if (int s = require_initialized_tel(); s != VGRE_SUCCESS)
    return s;
  if (!vgre::advanced::IPCManager::instance().initialize(is_master != 0)) {
    return VGRE_ERROR_IO;
  }
  return VGRE_SUCCESS;
}

int vgre_set_block_threads(int enabled) {
  // Note: BlockWorkerPool::initialize() has already run at this point and will
  // never re-read VGRE_BLOCK_THREADS, so calling setenv() here is both useless
  // and unsafe (setenv reallocates environ without locking against concurrent
  // getenv calls on other threads). This function is a no-op for now.
  // Future: implement a thread-safe runtime flag if needed.
  VGRE_LOG_INFO("VGRE", std::string("VGRE_BLOCK_THREADS set to ") + (enabled ? "1" : "0"));
  return VGRE_SUCCESS;
}

// ── Phase 5: Global Compute Network ──────────────────────────────────────

int vgre_cluster_set_security(int enabled) {
  return to_status_tel(vgre::advanced::TCPClusterManager::instance().enableSecurity(enabled != 0));
}

int vgre_cluster_get_security_info(vgre_security_info_t *info) {
  if (!info) return VGRE_ERROR_INVALID_VALUE;

  auto sinfo = vgre::advanced::TCPClusterManager::instance().getSecurityInfo();
  std::strncpy(info->cipher_name, sinfo.cipher_name, sizeof(info->cipher_name) - 1);
  info->cipher_name[sizeof(info->cipher_name) - 1] = '\0';
  std::strncpy(info->key_fingerprint, sinfo.key_fingerprint, sizeof(info->key_fingerprint) - 1);
  info->key_fingerprint[sizeof(info->key_fingerprint) - 1] = '\0';
  info->session_seconds = sinfo.session_seconds;
  info->is_encrypted = sinfo.is_encrypted ? 1 : 0;
  info->packets_sent = sinfo.packets_sent;
  info->packets_received = sinfo.packets_received;
  info->bytes_sent = sinfo.bytes_sent;
  info->bytes_received = sinfo.bytes_received;

  return VGRE_SUCCESS;
}

int vgre_cluster_wait(uint64_t kernel_id, int timeout_ms) {
    if (int s = require_initialized_tel(); s != VGRE_SUCCESS)
      return s;
    auto r = vgre::advanced::TCPClusterManager::instance().waitForRemoteResult(kernel_id, timeout_ms);
    return to_status_tel(r);
}

int vgre_credits_get_balance(const char *address, vgre_credit_info_t *info) {
  if (!address || !info) return VGRE_ERROR_INVALID_VALUE;

  vgre::advanced::NodeBalance bal;
  vgre::VGREResult r = vgre::advanced::ResourceLedger::instance().getBalance(address, bal);
  if (r != vgre::VGREResult::SUCCESS) return to_status_tel(r);

  std::strncpy(info->address, bal.address.c_str(), sizeof(info->address) - 1);
  info->address[sizeof(info->address) - 1] = '\0'; // Ensure null termination
  info->total_credits = bal.total_credits;
  info->total_debits = bal.total_debits;
  info->balance = bal.balance;
  info->last_activity = bal.last_activity;
  info->transaction_count = bal.transaction_count;

  return VGRE_SUCCESS;
}

int vgre_credits_get_all(vgre_credit_info_t *nodes, int *count) {
  if (!count) return VGRE_ERROR_INVALID_VALUE;

  auto balances = vgre::advanced::ResourceLedger::instance().getAllBalances();
  int total = static_cast<int>(balances.size());

  if (!nodes) {
    *count = total;
    return VGRE_SUCCESS;
  }

  int to_fill = std::min(*count, total);
  for (int i = 0; i < to_fill; ++i) {
    std::strncpy(nodes[i].address, balances[i].address.c_str(), sizeof(nodes[i].address) - 1);
    nodes[i].address[sizeof(nodes[i].address) - 1] = '\0'; // Ensure null termination
    nodes[i].total_credits = balances[i].total_credits;
    nodes[i].total_debits = balances[i].total_debits;
    nodes[i].balance = balances[i].balance;
    nodes[i].last_activity = balances[i].last_activity;
    nodes[i].transaction_count = balances[i].transaction_count;
  }

  *count = to_fill;
  return VGRE_SUCCESS;
}

int vgre_credits_reset(void) {
  vgre::advanced::ResourceLedger::instance().reset();
  return VGRE_SUCCESS;
}

// ── JIT Telemetry Reporting ───────────────────────────────────────────────

extern "C" void vgre_jit_report_flops(uint64_t flops) {
  vgre::advanced::AdaptiveExecutionEngine::instance().recordRealFlops(flops);
  // Phase 5: Report compute time to master for billing if we are a worker
  auto &cluster = vgre::advanced::TCPClusterManager::instance();
  if (cluster.isEnabled() && cluster.isWorker()) {
     // For now, we don't have a direct GFLOP to second mapping here,
     // but we trigger the wall-clock reporting in the handler.
  }
}

extern "C" void vgre_jit_report_memory(uint64_t bytes) {
  vgre::advanced::AdaptiveExecutionEngine::instance().recordRealMemoryAccess(bytes);
}

// ── Memory Pool C-API ─────────────────────────────────────────────────────

int vgre_pool_create(uint64_t *out_pool, size_t block_size) {
  if (!out_pool) return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized_tel(); s != VGRE_SUCCESS) return s;
  auto r = vgre::core::MemoryManager::instance().createPool(*out_pool, block_size);
  return to_status_tel(r);
}

int vgre_pool_alloc(uint64_t pool, size_t size, void **out_ptr) {
  if (!out_ptr || size == 0) return VGRE_ERROR_INVALID_VALUE;
  if (int s = require_initialized_tel(); s != VGRE_SUCCESS) return s;
  vgre::MemoryHandle handle;
  auto r = vgre::core::MemoryManager::instance().allocateFromPool(pool, size, handle);
  if (r != vgre::VGREResult::SUCCESS) return to_status_tel(r);
  *out_ptr = handle;
  return VGRE_SUCCESS;
}

int vgre_pool_free(uint64_t pool, void *ptr) {
  if (!ptr) return VGRE_SUCCESS;
  if (int s = require_initialized_tel(); s != VGRE_SUCCESS) return s;
  auto r = vgre::core::MemoryManager::instance().freeToPool(pool, ptr);
  return to_status_tel(r);
}

int vgre_pool_destroy(uint64_t pool) {
  if (int s = require_initialized_tel(); s != VGRE_SUCCESS) return s;
  auto r = vgre::core::MemoryManager::instance().destroyPool(pool);
  return to_status_tel(r);
}

int vgre_get_cluster_nodes(vgre_cluster_node_t *nodes, int *count) {
  if (!count) return VGRE_ERROR_INVALID_VALUE;

  auto &tcp = vgre::advanced::TCPClusterManager::instance();
  auto &ipc = vgre::advanced::IPCManager::instance();

  std::vector<vgre::advanced::TCPClusterManager::ClusterNodeInfo> connections;
  bool isMaster = tcp.isEnabled() && tcp.isMaster();

  if (isMaster) {
      tcp.getConnectedNodes(connections);
  }

  // If we are not the master OR we have no local connections, try reading from SHM
  if (connections.empty() && ipc.isEnabled()) {
      std::vector<vgre_cluster_node_t> shm_nodes;
      ipc.getClusterNodes(shm_nodes);

      int max_count = *count;
      int actual_count = static_cast<int>(shm_nodes.size());
      *count = actual_count;

      if (!nodes || max_count <= 0) return VGRE_SUCCESS;

      int copy_count = std::min(max_count, actual_count);
      for (int i = 0; i < copy_count; ++i) {
          nodes[i] = shm_nodes[i];
      }
      return VGRE_SUCCESS;
  }

  // Direct Master Mode: Populate the SHM to ensure consistency
  // Note: TCPClusterManager now proactively calls updateClusterNodes, but we
  // sync here as well to ensure any C-API caller gets the absolute latest.
  int max_count = *count;
  int actual_count = static_cast<int>(connections.size());
  *count = actual_count;

  if (!nodes || max_count <= 0) return VGRE_SUCCESS;

  int copy_count = std::min(max_count, actual_count);
  std::vector<vgre_cluster_node_t> to_sync;

  for (int i = 0; i < copy_count; ++i) {
    const auto &conn = connections[i];
    std::strncpy(nodes[i].address, conn.ip_address.c_str(), sizeof(nodes[i].address) - 1);
    nodes[i].port = conn.port;
    nodes[i].cpu_cores = conn.cpu_cores;
    nodes[i].memory_bytes = conn.cpu_memory;
    nodes[i].latency_ms = conn.last_telemetry.avg_kernel_latency_ms;

    if (conn.security_established) {
        nodes[i].available = 1;
    } else if (conn.is_authenticating) {
        nodes[i].available = 2;
    } else if (conn.active) {
        nodes[i].available = 1; // Plaintext active
    } else {
        nodes[i].available = 0;
    }
    std::strncpy(nodes[i].igpu_name, conn.igpu_name, sizeof(nodes[i].igpu_name) - 1);

    to_sync.push_back(nodes[i]);
  }

  if (isMaster && ipc.isEnabled()) {
      ipc.updateClusterNodes(to_sync);
  }

  return VGRE_SUCCESS;
}

// ── GPU Cache Statistics ───────────────────────────────────────────────────────

int vgre_get_cache_stats(vgre_cache_stats_t *stats) {
    if (!stats) return VGRE_ERROR_INVALID_VALUE;

    auto s = vgre::runtime::GPUCacheL2::instance().stats();
    stats->l2_hits       = s.hits;
    stats->l2_misses     = s.misses;
    stats->l2_evictions  = s.evictions;
    stats->l2_hit_rate   = s.hitRate();
    stats->l1_config_kb  = vgre::runtime::gpuL1CacheSizeKB();
    stats->l2_config_mb  = []() -> uint64_t {
        const char* v = vgre_get_config("VGRE_L2_CACHE_MB");
        if (v) {
            uint64_t mb = static_cast<uint64_t>(std::strtoul(v, nullptr, 10));
            if (mb >= 2 && mb <= 40) return mb;
        }
        return vgre::runtime::kL2DefaultMB;
    }();
    return VGRE_SUCCESS;
}

int vgre_reset_cache_stats(void) {
    vgre::runtime::GPUCacheL2::instance().flush();
    return VGRE_SUCCESS;
}
