#include "vgre/advanced/runtime_profiler.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/virtual_gpu_device.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstdio>
#include <cstring>

// Platform socket includes for exportOTLPToHTTP
#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "Ws2_32.lib")
#  define CLOSE_SOCK(s) ::closesocket(s)
#  define SOCK_T        SOCKET
#  define INVALID_SOCK  INVALID_SOCKET
#else
#  include <sys/socket.h>
#  include <netdb.h>
#  include <unistd.h>
#  define CLOSE_SOCK(s) ::close(s)
#  define SOCK_T        int
#  define INVALID_SOCK  (-1)
#endif

namespace vgre {
namespace advanced {

namespace {
static std::string escapeJsonString(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    switch (c) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out += c;
      break;
    }
  }
  return out;
}

} // namespace

RuntimeProfiler::RuntimeProfiler() {
    VGRE_LOG_DEBUG("RuntimeProfiler", "Initialized (disabled by default)");
}

RuntimeProfiler::~RuntimeProfiler() = default;

// ── Enable/disable ─────────────────────────────────────────────────────────
void RuntimeProfiler::setEnabled(bool enabled) {
    enabled_.store(enabled, std::memory_order_relaxed);
    VGRE_LOG_INFO("RuntimeProfiler",
                  std::string("Profiling ") +
                  (enabled ? "enabled" : "disabled"));
}

bool RuntimeProfiler::isEnabled() const {
    return enabled_.load(std::memory_order_relaxed);
}

// ── Record event ───────────────────────────────────────────────────────────
void RuntimeProfiler::recordEvent(const ProfileEvent& event) {
    if (!enabled_.load(std::memory_order_relaxed)) return;

    std::lock_guard<std::recursive_mutex> lock(mutex_);
    ProfileEvent ev = event;
    if (ev.timestamp_ms == 0) {
        ev.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    events_.push_back(ev);
    updateStats(ev.kernelName, ev);
}

// ── Timer ──────────────────────────────────────────────────────────────────
void RuntimeProfiler::startTimer(const std::string& kernelName) {
    if (!enabled_.load(std::memory_order_relaxed)) return;
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    timers_[kernelName] = std::chrono::steady_clock::now();
}

double RuntimeProfiler::stopTimer(const std::string& kernelName) {
    if (!enabled_.load(std::memory_order_relaxed)) return 0.0;

    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = timers_.find(kernelName);
    if (it == timers_.end()) return 0.0;

    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(
                    end - it->second).count();
    timers_.erase(it);

    return ms;
}

// ── Kernel Source Registration ─────────────────────────────────────────────
void RuntimeProfiler::setKernelSource(const std::string& name,
                                       const std::string& source,
                                       const std::string& ir) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    stats_[name].kernelName = name;
    stats_[name].sourceCode = source;
    stats_[name].irCode     = ir;
}

// ── Update aggregate stats ─────────────────────────────────────────────────
void RuntimeProfiler::updateStats(const std::string& name,
                                   const ProfileEvent& event) {
    auto& s = stats_[name];
    s.kernelName = name;
    s.invocations++;
    s.totalTimeMs += event.durationMs;
    s.avgTimeMs    = s.totalTimeMs / s.invocations;
    s.minTimeMs    = std::min(s.minTimeMs, event.durationMs);
    s.maxTimeMs    = std::max(s.maxTimeMs, event.durationMs);

    if (event.throughputGBps > 0) {
        s.avgThroughputGBps = (s.avgThroughputGBps *
                               (s.invocations - 1) +
                               event.throughputGBps) / s.invocations;
    }
    if (event.gflops > 0) {
        s.avgGflops = (s.avgGflops * (s.invocations - 1) +
                       event.gflops) / s.invocations;
    }

    // Phase 10: instruction sampler aggregation
    uint64_t instTotal = event.instructions.total();
    if (instTotal > 0) {
        s.totalInstructions += instTotal;
        s.avgInstructionsPerInvocation = s.totalInstructions / s.invocations;
        auto& mix = s.instructionMix;
        mix.loadCount    += event.instructions.loadCount;
        mix.storeCount   += event.instructions.storeCount;
        mix.aluCount     += event.instructions.aluCount;
        mix.barrierCount += event.instructions.barrierCount;
        mix.branchCount  += event.instructions.branchCount;
        mix.otherCount   += event.instructions.otherCount;
    }
}

// ── Get stats ──────────────────────────────────────────────────────────────
KernelStats RuntimeProfiler::getKernelStats(
    const std::string& kernelName) const {

    std::lock_guard<std::recursive_mutex> lock(mutex_);
    auto it = stats_.find(kernelName);
    if (it == stats_.end()) {
        KernelStats empty;
        empty.kernelName = kernelName;
        empty.minTimeMs = 0.0;
        return empty;
    }
    return it->second;
}

std::vector<KernelStats> RuntimeProfiler::getAllStats() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::vector<KernelStats> result;
    result.reserve(stats_.size());
    for (const auto& [_, s] : stats_) {
        result.push_back(s);
    }
    // Sort by total time descending (hottest kernels first)
    std::sort(result.begin(), result.end(),
              [](const KernelStats& a, const KernelStats& b) {
                  return a.totalTimeMs > b.totalTimeMs;
              });
    return result;
}
 
std::vector<ProfileEvent> RuntimeProfiler::getEventsByKernel(
    const std::string& kernelName) const {
 
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    std::vector<ProfileEvent> result;
    for (const auto& ev : events_) {
        if (ev.kernelName == kernelName) {
            result.push_back(ev);
        }
    }
    // Return last 100 events to avoid bloat
    if (result.size() > 100) {
        result.erase(result.begin(), result.end() - 100);
    }
    return result;
}

// ── JSON export ────────────────────────────────────────────────────────────
// ── Chrome Trace Export (chrome://tracing) ─────────────────────────────────
std::string RuntimeProfiler::toChromeTraceJSON() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    
    std::ostringstream oss;
    oss << "[\n";
    
    bool first = true;
    for (const auto& ev : events_) {
        if (!first) oss << ",\n";
        first = false;
        
        // Duration event (X)
        oss << "  { \"name\": \"" << escapeJsonString(ev.kernelName) << "\", "
            << "\"cat\": \"kernel\", \"ph\": \"X\", "
            << "\"ts\": " << (ev.timestamp_ms * 1000) << ", "
            << "\"dur\": " << (ev.durationMs * 1000) << ", "
            << "\"pid\": 1, \"tid\": " << ev.threadsUsed << ", "
            << "\"args\": { "
            << "\"gflops\": " << ev.gflops << ", "
            << "\"throughput_gbps\": " << ev.throughputGBps << ", "
            << "\"instructions\": " << ev.instructions.total() << ", "
            << "\"grid\": [" << ev.gridDim.x << "," << ev.gridDim.y << "," << ev.gridDim.z << "], "
            << "\"block\": [" << ev.blockDim.x << "," << ev.blockDim.y << "," << ev.blockDim.z << "] "
            << "} }";
    }
    
    oss << "\n]\n";
    return oss.str();
}

std::string RuntimeProfiler::toJSON() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\n";
    oss << "  \"profiler\": \"VGRE Runtime Profiler\",\n";
    oss << "  \"total_events\": " << events_.size() << ",\n";
    oss << "  \"kernels\": [\n";

    bool first = true;
    for (const auto& [name, s] : stats_) {
        if (!first) oss << ",\n";
        first = false;
        oss << "    {\n";
        oss << "      \"name\": \"" << escapeJsonString(s.kernelName) << "\",\n";
        oss << "      \"invocations\": " << s.invocations << ",\n";
        oss << "      \"total_time_ms\": " << s.totalTimeMs << ",\n";
        oss << "      \"avg_time_ms\": " << s.avgTimeMs << ",\n";
        oss << "      \"min_time_ms\": " << s.minTimeMs << ",\n";
        oss << "      \"max_time_ms\": " << s.maxTimeMs << ",\n";
        oss << "      \"avg_throughput_gbps\": " << s.avgThroughputGBps << ",\n";
        oss << "      \"avg_gflops\": " << s.avgGflops << ",\n";
        oss << "      \"total_instructions\": " << s.totalInstructions << ",\n";
        oss << "      \"avg_instructions\": " << s.avgInstructionsPerInvocation << ",\n";
        const auto& mix = s.instructionMix;
        oss << "      \"instruction_mix\": {\n";
        oss << "        \"load\": " << mix.loadCount << ",\n";
        oss << "        \"store\": " << mix.storeCount << ",\n";
        oss << "        \"alu\": " << mix.aluCount << ",\n";
        oss << "        \"barrier\": " << mix.barrierCount << ",\n";
        oss << "        \"branch\": " << mix.branchCount << ",\n";
        oss << "        \"other\": " << mix.otherCount << "\n";
        oss << "      },\n";
        oss << "      \"source_code\": \"" << escapeJsonString(s.sourceCode) << "\",\n";
        oss << "      \"ir_code\": \"" << escapeJsonString(s.irCode) << "\"\n";
        oss << "    }";
    }

    oss << "\n  ]\n";
    oss << "}\n";

    return oss.str();
}

VGREResult RuntimeProfiler::exportToFile(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return VGREResult::ERR_IO;
    }
    file << toJSON();
    file.close();
    VGRE_LOG_INFO("RuntimeProfiler",
                  "Exported profile to: " + filepath);
    return VGREResult::SUCCESS;
}

// ── OTLP JSON Export ───────────────────────────────────────────────────────
std::string RuntimeProfiler::toOTLPJSON() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    // Stable per-session trace ID: derived from wall-clock time at first call.
    static uint64_t s_traceHi = 0, s_traceLo = 0;
    if (s_traceHi == 0) {
        s_traceHi = static_cast<uint64_t>(
            std::chrono::system_clock::now().time_since_epoch().count());
#if defined(_WIN32)
        s_traceLo = static_cast<uint64_t>(::GetCurrentProcessId()) << 32;
#elif defined(__linux__) || defined(__APPLE__)
        s_traceLo = static_cast<uint64_t>(::getpid()) << 32;
#endif
    }
    char traceId[33];
    std::snprintf(traceId, sizeof(traceId), "%016llx%016llx",
                  static_cast<unsigned long long>(s_traceHi),
                  static_cast<unsigned long long>(s_traceLo));

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);

    // Collect hw.gpu.* metrics for resource attributes from runtime state.
    uint64_t memLimit = 4ULL * 1024 * 1024 * 1024;
    if (vgre::core::RuntimeEngine::instance().isInitialized()) {
      memLimit =
          vgre::core::RuntimeEngine::instance().getDevice().getProperties().totalGlobalMem;
    }
    uint64_t memUsage = static_cast<uint64_t>(
        vgre::core::MemoryManager::instance().getUsedMemory());
    double memUtil =
        (memLimit > 0) ? std::min(1.0, static_cast<double>(memUsage) / memLimit) : 0.0;

    auto &ae = vgre::advanced::AdaptiveExecutionEngine::instance();
    ae.updateInstantaneousMetrics();
    double peakGflops = ae.getMaxGFLOPS();
    double instGflops = ae.getInstantaneousGFLOPS();
    double gpuUtil = (peakGflops > 0.0)
                         ? std::clamp(instGflops / peakGflops, 0.0, 1.0)
                         : memUtil;
    float tempC = ae.getDeviceTemperature();
    const char *status =
        (tempC > 0.0f && tempC >= 85.0f) ? "degraded" : "ok";

    oss << "{\n"
        << "  \"resourceSpans\": [\n"
        << "    {\n"
        << "      \"resource\": {\n"
        << "        \"attributes\": [\n"
        << "          { \"key\": \"service.name\","
           " \"value\": { \"stringValue\": \"vgre\" } },\n"
        << "          { \"key\": \"service.version\","
           " \"value\": { \"stringValue\": \"1.1.0\" } },\n"
        << "          { \"key\": \"telemetry.sdk.name\","
           " \"value\": { \"stringValue\": \"vgre-runtime-profiler\" } },\n"
        // OTel Hardware Semantic Conventions (hw.gpu.*)
        << "          { \"key\": \"hw.id\","
           " \"value\": { \"stringValue\": \"vgre-0\" } },\n"
        << "          { \"key\": \"hw.type\","
           " \"value\": { \"stringValue\": \"gpu\" } },\n"
        << "          { \"key\": \"hw.model.name\","
           " \"value\": { \"stringValue\": \"VGRE Virtual GPU\" } },\n"
        << "          { \"key\": \"driver.version\","
           " \"value\": { \"stringValue\": \"1.1.0\" } },\n"
        << "          { \"key\": \"hw.gpu.memory.limit\","
           " \"value\": { \"intValue\": \"" << memLimit << "\" } },\n"
        << "          { \"key\": \"hw.gpu.memory.usage\","
           " \"value\": { \"intValue\": \"" << memUsage << "\" } },\n"
        << "          { \"key\": \"hw.gpu.memory.utilization\","
           " \"value\": { \"doubleValue\": " << memUtil << " } },\n"
        << "          { \"key\": \"hw.gpu.utilization\","
           " \"value\": { \"doubleValue\": " << gpuUtil << " } },\n"
        << "          { \"key\": \"hw.gpu.errors\","
           " \"value\": { \"intValue\": \"0\" } },\n"
        << "          { \"key\": \"hw.status\","
           " \"value\": { \"stringValue\": \"" << status << "\" } }\n"
        << "        ]\n"
        << "      },\n"
        << "      \"scopeSpans\": [\n"
        << "        {\n"
        << "          \"scope\": { \"name\": \"vgre.runtime_profiler\","
           " \"version\": \"0.1.2\" },\n"
        << "          \"spans\": [\n";

    bool firstSpan = true;
    for (size_t idx = 0; idx < events_.size(); ++idx) {
        const auto& ev = events_[idx];

        // Span ID: 8 bytes (16 hex chars) derived from timestamp + index.
        char spanId[17];
        uint64_t spanVal = (ev.timestamp_ms * 1000ULL) ^ (idx * 0x9E3779B97F4A7C15ULL);
        std::snprintf(spanId, sizeof(spanId), "%016llx",
                      static_cast<unsigned long long>(spanVal));

        // OTLP timestamps: nanoseconds since Unix epoch, encoded as strings.
        uint64_t startNs = ev.timestamp_ms * 1'000'000ULL;
        uint64_t endNs   = startNs + static_cast<uint64_t>(ev.durationMs * 1'000'000.0);

        if (!firstSpan) oss << ",\n";
        firstSpan = false;

        oss << "            {\n"
            << "              \"traceId\": \"" << traceId << "\",\n"
            << "              \"spanId\": \"" << spanId << "\",\n"
            << "              \"name\": \"" << escapeJsonString(ev.kernelName) << "\",\n"
            << "              \"startTimeUnixNano\": \"" << startNs << "\",\n"
            << "              \"endTimeUnixNano\": \"" << endNs << "\",\n"
            << "              \"attributes\": [\n"
            << "                { \"key\": \"vgre.gflops\","
               " \"value\": { \"doubleValue\": " << ev.gflops << " } },\n"
            << "                { \"key\": \"vgre.throughput_gbps\","
               " \"value\": { \"doubleValue\": " << ev.throughputGBps << " } },\n"
            << "                { \"key\": \"vgre.grid_dim.x\","
               " \"value\": { \"intValue\": \"" << ev.gridDim.x << "\" } },\n"
            << "                { \"key\": \"vgre.grid_dim.y\","
               " \"value\": { \"intValue\": \"" << ev.gridDim.y << "\" } },\n"
            << "                { \"key\": \"vgre.grid_dim.z\","
               " \"value\": { \"intValue\": \"" << ev.gridDim.z << "\" } },\n"
            << "                { \"key\": \"vgre.threads_used\","
               " \"value\": { \"intValue\": \"" << ev.threadsUsed << "\" } }\n"
            << "              ],\n"
            << "              \"status\": { \"code\": \"STATUS_CODE_OK\" }\n"
            << "            }";
    }

    oss << "\n"
        << "          ]\n"   // spans
        << "        }\n"     // scopeSpans[0]
        << "      ]\n"       // scopeSpans
        << "    }\n"         // resourceSpans[0]
        << "  ]\n"           // resourceSpans
        << "}\n";

    return oss.str();
}

VGREResult RuntimeProfiler::exportOTLPToHTTP(const std::string& endpoint) const {
    // Parse URL: http://host:port/path
    std::string host, path;
    int port = 4318;  // OTLP HTTP default

    const std::string prefix = "http://";
    size_t start = endpoint.rfind(prefix);
    std::string rest = (start != std::string::npos)
                       ? endpoint.substr(start + prefix.size())
                       : endpoint;

    size_t slash = rest.find('/');
    std::string hostPort = (slash != std::string::npos) ? rest.substr(0, slash) : rest;
    path = (slash != std::string::npos) ? rest.substr(slash) : "/v1/traces";

    size_t colon = hostPort.rfind(':');
    if (colon != std::string::npos) {
        host = hostPort.substr(0, colon);
        try { port = std::stoi(hostPort.substr(colon + 1)); } catch (...) {}
    } else {
        host = hostPort;
    }
    if (host.empty()) host = "localhost";

    // Resolve and connect
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    std::string portStr = std::to_string(port);
    if (::getaddrinfo(host.c_str(), portStr.c_str(), &hints, &res) != 0 || !res) {
        VGRE_LOG_ERROR("RuntimeProfiler",
            "exportOTLPToHTTP: DNS resolution failed for " + host);
        return VGREResult::ERR_IO;
    }

    SOCK_T sock = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sock == INVALID_SOCK) {
        ::freeaddrinfo(res);
        VGRE_LOG_ERROR("RuntimeProfiler", "exportOTLPToHTTP: socket() failed");
        return VGREResult::ERR_IO;
    }

    if (::connect(sock, res->ai_addr, static_cast<int>(res->ai_addrlen)) != 0) {
        ::freeaddrinfo(res);
        CLOSE_SOCK(sock);
        VGRE_LOG_ERROR("RuntimeProfiler",
            "exportOTLPToHTTP: connect() failed to " + host + ":" + portStr);
        return VGREResult::ERR_IO;
    }
    ::freeaddrinfo(res);

    // Build the OTLP payload and HTTP request.
    std::string body = toOTLPJSON();
    std::ostringstream req;
    req << "POST " << path << " HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: close\r\n"
        << "\r\n"
        << body;

    std::string reqStr = req.str();
    int sent = static_cast<int>(::send(sock, reqStr.data(),
                                        static_cast<int>(reqStr.size()), 0));
    if (sent <= 0) {
        CLOSE_SOCK(sock);
        VGRE_LOG_ERROR("RuntimeProfiler", "exportOTLPToHTTP: send() failed");
        return VGREResult::ERR_IO;
    }

    // Read response status line to check HTTP status code.
    char respBuf[512] = {};
    ::recv(sock, respBuf, sizeof(respBuf) - 1, 0);
    CLOSE_SOCK(sock);

    // Parse "HTTP/1.1 200 OK" — status is the second token.
    VGREResult result = VGREResult::ERR_UNKNOWN;
    const char* sp = std::strchr(respBuf, ' ');
    if (sp) {
        int statusCode = std::atoi(sp + 1);
        if (statusCode >= 200 && statusCode < 300) {
            VGRE_LOG_INFO("RuntimeProfiler",
                "OTLP export OK (HTTP " + std::to_string(statusCode) + ") → " +
                endpoint + "  [" + std::to_string(events_.size()) + " spans]");
            result = VGREResult::SUCCESS;
        } else {
            VGRE_LOG_WARN("RuntimeProfiler",
                "OTLP export returned HTTP " + std::to_string(statusCode));
        }
    }
    return result;
}

// ── Clear ──────────────────────────────────────────────────────────────────
void RuntimeProfiler::clear() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    events_.clear();
    stats_.clear();
    timers_.clear();
    instructionMixes_.clear();
}

// ── Singleton ──────────────────────────────────────────────────────────────
RuntimeProfiler& RuntimeProfiler::instance() {
    static RuntimeProfiler inst;
    return inst;
}

} // namespace advanced
} // namespace vgre
