// NVTX v3 shim implementation for VGRE.
// All NVTX ranges/markers are forwarded to RuntimeProfiler so they appear in
// Chrome trace and OTLP exports alongside CUDA kernel spans.

#include "vgre/api/nvtx.h"
#include "vgre/advanced/runtime_profiler.h"
#include "vgre/common/logger.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <stack>
#include <string>
#include <unordered_map>

// ── Internal state ────────────────────────────────────────────────────────────

namespace {

// Monotonic range ID counter.
static std::atomic<uint64_t> g_rangeIdCounter{1};

// Per-thread push/pop stack (maps nesting depth → ProfileEvent start time).
static thread_local std::stack<std::pair<std::string, std::chrono::steady_clock::time_point>> t_rangeStack;

// Domain registry: domain handle → name string.
static std::mutex g_domainMu;
static std::unordered_map<nvtxDomainHandle_t, std::string> g_domains;

// Active range registry: rangeId → (name, start_time).
static std::mutex g_rangeMu;
static std::unordered_map<uint64_t,
    std::pair<std::string, std::chrono::steady_clock::time_point>> g_activeRanges;

// Extract a human-readable label from an event attribute struct.
static std::string extractLabel(const nvtxEventAttributes_t* attr) {
    if (!attr) return "(nvtx)";
    if (attr->messageType == NVTX_MESSAGE_TYPE_ASCII && attr->message.ascii)
        return std::string(attr->message.ascii);
    if (attr->messageType == NVTX_MESSAGE_TYPE_REGISTERED && attr->message.registered)
        return std::string(static_cast<const char*>(attr->message.registered));
    return "(nvtx)";
}

// Push a range start into RuntimeProfiler as a zero-duration marker for now;
// the full duration is recorded when the range ends.
static void recordRangeStart(const std::string& label) {
    vgre::advanced::ProfileEvent ev;
    ev.kernelName  = "[NVTX] " + label;
    ev.durationMs  = 0.0;
    ev.timestamp   = std::chrono::steady_clock::now();
    ev.timestamp_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            ev.timestamp.time_since_epoch()).count());
    vgre::advanced::RuntimeProfiler::instance().recordEvent(ev);
}

static void recordRangeEnd(const std::string& label, double durationMs) {
    vgre::advanced::ProfileEvent ev;
    ev.kernelName  = "[NVTX] " + label;
    ev.durationMs  = durationMs;
    ev.timestamp   = std::chrono::steady_clock::now();
    ev.timestamp_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            ev.timestamp.time_since_epoch()).count());
    vgre::advanced::RuntimeProfiler::instance().recordEvent(ev);
}

} // namespace

// ── Marker APIs ───────────────────────────────────────────────────────────────

extern "C" {

void nvtxMarkA(const char* message) {
    if (!message) return;
    recordRangeStart(message);
}

void nvtxMarkEx(const nvtxEventAttributes_t* eventAttrib) {
    recordRangeStart(extractLabel(eventAttrib));
}

// ── Range APIs (start/end by ID) ──────────────────────────────────────────────

nvtxRangeId_t nvtxRangeStartA(const char* message) {
    std::string label = message ? message : "(nvtx)";
    uint64_t id = g_rangeIdCounter.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(g_rangeMu);
        g_activeRanges[id] = {label, std::chrono::steady_clock::now()};
    }
    recordRangeStart(label);
    return static_cast<nvtxRangeId_t>(id);
}

nvtxRangeId_t nvtxRangeStartEx(const nvtxEventAttributes_t* eventAttrib) {
    return nvtxRangeStartA(extractLabel(eventAttrib).c_str());
}

void nvtxRangeEnd(nvtxRangeId_t id) {
    std::string label;
    double durationMs = 0.0;
    {
        std::lock_guard<std::mutex> lk(g_rangeMu);
        auto it = g_activeRanges.find(static_cast<uint64_t>(id));
        if (it != g_activeRanges.end()) {
            label      = it->second.first;
            auto end   = std::chrono::steady_clock::now();
            durationMs = std::chrono::duration<double, std::milli>(
                             end - it->second.second).count();
            g_activeRanges.erase(it);
        }
    }
    if (!label.empty())
        recordRangeEnd(label, durationMs);
}

// ── Push/Pop (nested) range APIs ─────────────────────────────────────────────

int nvtxRangePushA(const char* message) {
    std::string label = message ? message : "(nvtx)";
    t_rangeStack.push({label, std::chrono::steady_clock::now()});
    recordRangeStart(label);
    return static_cast<int>(t_rangeStack.size()) - 1;
}

int nvtxRangePushEx(const nvtxEventAttributes_t* eventAttrib) {
    return nvtxRangePushA(extractLabel(eventAttrib).c_str());
}

int nvtxRangePop(void) {
    if (t_rangeStack.empty()) return -1;
    auto [label, start] = t_rangeStack.top();
    t_rangeStack.pop();
    double durationMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    recordRangeEnd(label, durationMs);
    return static_cast<int>(t_rangeStack.size());
}

// ── Domain APIs ───────────────────────────────────────────────────────────────

nvtxDomainHandle_t nvtxDomainCreateA(const char* name) {
    // Allocate a small heap object to use as a unique opaque handle.
    auto* handle = new char[1];
    std::lock_guard<std::mutex> lk(g_domainMu);
    g_domains[handle] = name ? name : "";
    VGRE_LOG_DEBUG("NVTX", "Domain created: " + std::string(name ? name : ""));
    return static_cast<nvtxDomainHandle_t>(handle);
}

nvtxDomainHandle_t nvtxDomainCreateW(const wchar_t* name) {
    // Convert wide string to narrow for logging; domain is otherwise identical.
    return nvtxDomainCreateA("(wide-name domain)");
}

nvtxDomainHandle_t nvtxDomainCreate(const char* name) {
    return nvtxDomainCreateA(name);
}

void nvtxDomainDestroy(nvtxDomainHandle_t domain) {
    if (!domain) return;
    std::lock_guard<std::mutex> lk(g_domainMu);
    g_domains.erase(domain);
    delete[] static_cast<char*>(domain);
}

void nvtxDomainMarkEx(nvtxDomainHandle_t /*domain*/,
                      const nvtxEventAttributes_t* eventAttrib) {
    nvtxMarkEx(eventAttrib);
}

nvtxRangeId_t nvtxDomainRangeStartEx(nvtxDomainHandle_t /*domain*/,
                                      const nvtxEventAttributes_t* eventAttrib) {
    return nvtxRangeStartEx(eventAttrib);
}

void nvtxDomainRangeEnd(nvtxDomainHandle_t /*domain*/, nvtxRangeId_t id) {
    nvtxRangeEnd(id);
}

int nvtxDomainRangePushEx(nvtxDomainHandle_t /*domain*/,
                           const nvtxEventAttributes_t* eventAttrib) {
    return nvtxRangePushEx(eventAttrib);
}

int nvtxDomainRangePop(nvtxDomainHandle_t /*domain*/) {
    return nvtxRangePop();
}

// ── String registration ───────────────────────────────────────────────────────

nvtxStringHandle_t nvtxDomainRegisterStringA(nvtxDomainHandle_t /*domain*/,
                                              const char* string) {
    if (!string) return nullptr;
    // Store the string in a static map to guarantee lifetime.
    static std::mutex sMu;
    static std::unordered_map<std::string, std::string> sStrings;
    std::lock_guard<std::mutex> lk(sMu);
    auto it = sStrings.emplace(string, string).first;
    return static_cast<nvtxStringHandle_t>(const_cast<char*>(it->second.c_str()));
}

nvtxStringHandle_t nvtxDomainRegisterStringW(nvtxDomainHandle_t domain,
                                              const wchar_t* /*string*/) {
    return nvtxDomainRegisterStringA(domain, "(wide-string)");
}

// ── Resource naming (logged for tracing correlation) ──────────────────────────

void nvtxNameCategoryA(uint32_t category, const char* name) {
    VGRE_LOG_DEBUG("NVTX", "Category " + std::to_string(category) +
                   " named: " + (name ? name : ""));
}

void nvtxNameOsThreadA(uint32_t tid, const char* name) {
    VGRE_LOG_DEBUG("NVTX", "Thread " + std::to_string(tid) +
                   " named: " + (name ? name : ""));
}

void nvtxNameCudaDeviceA(int device, const char* name) {
    VGRE_LOG_INFO("NVTX", "CUDA device " + std::to_string(device) +
                  " named: " + (name ? name : ""));
}

void nvtxNameCudaStreamA(cudaStream_t_nvtx stream, const char* name) {
    VGRE_LOG_DEBUG("NVTX", "CUDA stream " +
                   std::to_string(reinterpret_cast<uintptr_t>(stream)) +
                   " named: " + (name ? name : ""));
}

void nvtxNameCudaEventA(void* event, const char* name) {
    VGRE_LOG_DEBUG("NVTX", "CUDA event " +
                   std::to_string(reinterpret_cast<uintptr_t>(event)) +
                   " named: " + (name ? name : ""));
}

void nvtxNameCudaDeviceEx(nvtxDomainHandle_t /*domain*/, int device, const char* name) {
    nvtxNameCudaDeviceA(device, name);
}

void nvtxNameCudaStreamEx(nvtxDomainHandle_t /*domain*/, cudaStream_t_nvtx stream, const char* name) {
    nvtxNameCudaStreamA(stream, name);
}

void nvtxNameCudaEventEx(nvtxDomainHandle_t /*domain*/, void* event, const char* name) {
    nvtxNameCudaEventA(event, name);
}

void nvtxNameCudaThreadEx(nvtxDomainHandle_t /*domain*/, uint32_t tid, const char* name) {
    nvtxNameOsThreadA(tid, name);
}

} // extern "C"
