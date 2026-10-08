#ifndef VGRE_ADVANCED_RUNTIME_PROFILER_H
#define VGRE_ADVANCED_RUNTIME_PROFILER_H

#include "vgre/common/types.h"
#include "vgre/common/error_codes.h"
#include "vgre/core/skip_list.h"   // timestamp-ordered event timeline

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <functional>
#include <utility>

namespace vgre {
namespace advanced {

// ── Instruction mix sample ───────────────────────────────────────────────
struct InstructionSample {
    uint64_t loadCount   = 0;
    uint64_t storeCount  = 0;
    uint64_t aluCount    = 0;
    uint64_t barrierCount= 0;
    uint64_t branchCount = 0;
    uint64_t otherCount  = 0;

    uint64_t total() const {
        return loadCount + storeCount + aluCount + barrierCount + branchCount + otherCount;
    }
};

enum class ProfileApi : uint8_t {
    Unknown = 0,
    CudaDriver = 1,
    CudaRuntime = 2
};

// Tags launch records with the public API path that submitted them, allowing
// profiling observers to report the real entry point used by the caller.
class ScopedProfileApi {
public:
    explicit ScopedProfileApi(ProfileApi api);
    ~ScopedProfileApi();
    ScopedProfileApi(const ScopedProfileApi&) = delete;
    ScopedProfileApi& operator=(const ScopedProfileApi&) = delete;
private:
    ProfileApi previous_;
};

ProfileApi currentProfileApi();

// ── Profiling event ────────────────────────────────────────────────────────
struct ProfileEvent {
    std::string kernelName;
    double      durationMs      = 0.0;
    size_t      memoryBytes     = 0;
    size_t      flops           = 0;
    double      throughputGBps  = 0.0;
    double      gflops          = 0.0;
    dim3        gridDim;
    dim3        blockDim;
    size_t      staticSharedMemoryBytes = 0;
    size_t      dynamicSharedMemoryBytes = 0;
    ProfileApi  api             = ProfileApi::Unknown;
    // True only for a successfully executed kernel launch. NVTX ranges,
    // library operations, graph host work, and other timeline events remain
    // profile events but are not CUDA kernel activity/counter samples.
    bool        isKernelLaunch  = false;
    int         threadsUsed     = 0;
    InstructionSample instructions;  // software estimate from static kernel analysis
    std::chrono::steady_clock::time_point timestamp;
    uint64_t    timestamp_ms    = 0;
};

// ── Kernel aggregate stats ─────────────────────────────────────────────────
struct KernelStats {
    std::string kernelName;
    std::string sourceCode;      // Original CUDA-like source
    std::string irCode;          // Compiled LLVM-IR
    int         invocations      = 0;
    double      totalTimeMs      = 0.0;
    double      avgTimeMs        = 0.0;
    double      minTimeMs        = 1e12;
    double      maxTimeMs        = 0.0;
    double      avgThroughputGBps= 0.0;
    double      avgGflops        = 0.0;
    uint64_t    totalInstructions= 0;
    uint64_t    avgInstructionsPerInvocation = 0;
    InstructionSample instructionMix; // aggregated across all invocations
};

// ── Runtime Profiler ───────────────────────────────────────────────────────
class RuntimeProfiler {
public:
    using EventListener = std::function<void(const ProfileEvent&)>;
    using EventListenerId = uint64_t;
    RuntimeProfiler();
    ~RuntimeProfiler();

    // Enable/disable profiling
    void setEnabled(bool enabled);
    bool isEnabled() const;

    // Record a profiling event
    void recordEvent(const ProfileEvent& event);

    // Register an observer for each event recorded while profiling is enabled.
    // Listeners run after the event is committed and outside the profiler lock.
    EventListenerId addEventListener(EventListener listener);
    bool removeEventListener(EventListenerId listenerId);

    // Start/stop a timer for a kernel
    void startTimer(const std::string& kernelName);
    double stopTimer(const std::string& kernelName); // returns ms

    // Register source code for a kernel
    void setKernelSource(const std::string& name, 
                         const std::string& source,
                         const std::string& ir);

    // Get statistics
    KernelStats getKernelStats(const std::string& kernelName) const;
    std::vector<KernelStats> getAllStats() const;
    std::vector<ProfileEvent> getEventsByKernel(const std::string& kernelName) const;
    std::vector<ProfileEvent> getAllEvents() const;
    uint64_t getEventGeneration() const;
    std::pair<std::vector<ProfileEvent>, uint64_t> getEventSnapshot() const;

    // Time-window query: all events whose timestamp_ms is in [startMs, endMs],
    // returned in ascending time order. Backed by a timestamp-ordered skip-list
    // timeline, so this is O(log n + k) rather than an O(n) scan + sort — useful
    // for trace windows and overlap/occupancy-over-time analysis.
    std::vector<ProfileEvent> getEventsInWindow(uint64_t startMs, uint64_t endMs) const;

    // Export
    std::string toJSON() const;
    std::string toChromeTraceJSON() const; // Phase 10: standard tracing
    VGREResult  exportToFile(const std::string& filepath) const;

    /**
     * @brief Generates an OpenTelemetry OTLP/JSON trace payload.
     *
     * Produces a JSON document conforming to the OTLP trace protobuf-JSON
     * mapping (opentelemetry-proto/trace/v1/trace.proto).  Each recorded
     * ProfileEvent becomes one span with VGRE-specific attributes
     * (vgre.gflops, vgre.throughput_gbps, vgre.grid_dim.*).
     *
     * The payload can be POSTed directly to any OpenTelemetry collector
     * (Jaeger, Tempo, OTLP HTTP receiver) at `/v1/traces`.
     */
    std::string toOTLPJSON() const;

    /**
     * @brief POSTs the OTLP trace payload to a collector via raw TCP.
     *
     * Uses a plain HTTP/1.1 POST — no libcurl dependency.
     * @param endpoint  URL in the form "http://host:port/path"
     *                  (e.g. "http://localhost:4318/v1/traces").
     * @return SUCCESS on HTTP 200/204, ERROR_IO on connection failure,
     *         ERROR_UNKNOWN on non-2xx response.
     */
    VGREResult  exportOTLPToHTTP(const std::string& endpoint) const;

    // ── Software instruction accounting ────────────────────────────────────
    // Record an actual instrumented sample when available. Static estimates are
    // kept distinct from hardware counters and are not exposed as CUPTI metrics.
    void recordInstructionSample(const std::string& kernelName,
                                  const InstructionSample& sample);

    // Estimate instructions for a kernel launch based on grid/block size
    // and a per-thread instruction count derived from static analysis.
    void estimateInstructions(const std::string& kernelName,
                              const dim3& gridDim, const dim3& blockDim,
                              uint64_t instructionsPerThread);

    // Get the aggregated instruction mix for a kernel.
    InstructionSample getInstructionMix(const std::string& kernelName) const;

    // Reset
    void clear();

    // Singleton
    static RuntimeProfiler& instance();

private:
    void updateStats(const std::string& name, const ProfileEvent& event);

    std::atomic<bool>                             enabled_{false};
    std::vector<ProfileEvent>                     events_;
    std::unordered_map<std::string, KernelStats>  stats_;
    std::unordered_map<std::string,
        std::chrono::steady_clock::time_point>    timers_;
    // Software instruction mix; entries may include source-analysis estimates.
    std::unordered_map<std::string, InstructionSample> instructionMixes_;
    // Timestamp-ordered timeline: key = (timestamp_ms << 20) | seq, value = index
    // into events_. Maintains events in time order for O(log n + k) window queries.
    SkipList<uint64_t, std::size_t, 20> timeline_;
    uint64_t                            timelineSeq_ = 0;
    uint64_t                            eventGeneration_ = 0;
    mutable std::recursive_mutex                  mutex_;
    mutable std::mutex                            listenerMutex_;
    std::unordered_map<EventListenerId, EventListener> listeners_;
    EventListenerId                               nextListenerId_ = 1;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_RUNTIME_PROFILER_H
