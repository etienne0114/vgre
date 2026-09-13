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

// Use function-local static initialization to prevent blocking during library load
// This ensures the counter and maps are only initialized when first used
std::atomic<uint64_t>& getRangeIdCounter() {
    static std::atomic<uint64_t> counter{1};
    return counter;
}

// Per-thread push/pop stack (maps nesting depth → ProfileEvent start time).
static thread_local std::stack<std::pair<std::string, std::chrono::steady_clock::time_point>> t_rangeStack;

// Domain registry: domain handle → name string.
std::mutex& getDomainMutex() {
    static std::mutex mu;
    return mu;
}

std::unordered_map<nvtxDomainHandle_t, std::string>& getDomains() {
    static std::unordered_map<nvtxDomainHandle_t, std::string> domains;
    return domains;
}

// Active range registry: rangeId → (name, start_time).
std::mutex& getRangeMutex() {
    static std::mutex mu;
    return mu;
}

std::unordered_map<uint64_t, std::pair<std::string, std::chrono::steady_clock::time_point>>& getActiveRanges() {
    static std::unordered_map<uint64_t, std::pair<std::string, std::chrono::steady_clock::time_point>> activeRanges;
    return activeRanges;
}

// Extract a human-readable label from an event attribute struct.
static std::string extractLabel(const nvtxEventAttributes_t* attr) {
    if (!attr) return "(nvtx)";
    if (attr->messageType == NVTX_MESSAGE_TYPE_ASCII && attr->message.ascii)
        return std::string(attr->message.ascii);
    if (attr->messageType == NVTX_MESSAGE_TYPE_REGISTERED && attr->message.registered)
        return std::string(static_cast<const char*>(attr->message.registered));
    return "(nvtx)";
}

// Push a range start into RuntimeProfiler as a zero-duration marker; the
// full duration is recorded by recordRangeEnd() when the range ends.
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
    uint64_t id = getRangeIdCounter().fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(getRangeMutex());
        getActiveRanges()[id] = {label, std::chrono::steady_clock::now()};
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
        std::lock_guard<std::mutex> lk(getRangeMutex());
        auto it = getActiveRanges().find(static_cast<uint64_t>(id));
        if (it != getActiveRanges().end()) {
            label      = it->second.first;
            auto end   = std::chrono::steady_clock::now();
            durationMs = std::chrono::duration<double, std::milli>(
                             end - it->second.second).count();
            getActiveRanges().erase(it);
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
    std::lock_guard<std::mutex> lk(getDomainMutex());
    getDomains()[handle] = name ? name : "";
    VGRE_LOG_DEBUG("NVTX", "Domain created: " + std::string(name ? name : ""));
    return static_cast<nvtxDomainHandle_t>(handle);
}

nvtxDomainHandle_t nvtxDomainCreateW(const wchar_t* name) {
    // Convert wide string to narrow for logging; domain is otherwise identical.
    std::string narrowName;
    if (name) {
        // Simple conversion from wide to narrow string
        size_t len = wcslen(name);
        narrowName.resize(len);
        for (size_t i = 0; i < len; ++i) {
            narrowName[i] = static_cast<char>(name[i]);
        }
    } else {
        narrowName = "(unnamed wide domain)";
    }
    return nvtxDomainCreateA(narrowName.c_str());
}

nvtxDomainHandle_t nvtxDomainCreate(const char* name) {
    return nvtxDomainCreateA(name);
}

void nvtxDomainDestroy(nvtxDomainHandle_t domain) {
    if (!domain) return;
    std::lock_guard<std::mutex> lk(getDomainMutex());
    getDomains().erase(domain);
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
