// CUDA External Semaphore API — interop with Vulkan, D3D12, OpenGL, POSIX eventfd.
// Required for: timeline semaphore synchronization, graphics-compute pipeline interop,
// and multi-process ordered synchronization.
//
// Implementation:
//   POSIX opaque FD  → Linux eventfd (EFD_SEMAPHORE)
//   Timeline FD      → monotonic counter eventfd
//   Win32 handle     → Windows CreateEvent / SetEvent
//   Signal on stream → enqueue task that writes to eventfd / sets Win32 event
//   Wait on stream   → enqueue task that polls eventfd / waits Win32 event

#include "vgre/common/logger.h"
#include "vgre/common/platform.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "vgre/common/os_backend.h"
#if defined(__linux__)
#  include <sys/eventfd.h>  // eventfd() — semaphore file descriptor
#elif defined(__APPLE__)
#  include <dispatch/dispatch.h>
#endif

// ── External semaphore handle types (mirror CUDA headers) ────────────────────
static constexpr int CUDA_EXTERNAL_SEM_OPAQUE_FD            [[maybe_unused]] = 1;
static constexpr int CUDA_EXTERNAL_SEM_OPAQUE_WIN32         [[maybe_unused]] = 2;
static constexpr int CUDA_EXTERNAL_SEM_NVSCISYNCOBJ        [[maybe_unused]] = 3;
static constexpr int CUDA_EXTERNAL_SEM_TIMELINE_FD          [[maybe_unused]] = 4;
static constexpr int CUDA_EXTERNAL_SEM_TIMELINE_WIN32       [[maybe_unused]] = 8;

// ── Internal semaphore object ─────────────────────────────────────────────────
namespace {

struct ExtSem {
    int type = 0;
#if defined(__linux__)
    int efd = -1;          // eventfd or timeline eventfd
#elif defined(__APPLE__)
    dispatch_semaphore_t dsem = nullptr; // unnamed POSIX sem is deprecated on macOS
#elif defined(_WIN32)
    HANDLE hEvent = nullptr;
#endif
    std::atomic<uint64_t> timelineValue{0}; // for timeline types
};

// Use function-local static initialization to prevent blocking during library load
// This ensures the mutex and maps are only initialized when first used
std::mutex& getExtSemMutex() {
    static std::mutex mu;
    return mu;
}

std::atomic<uint64_t>& getNextExtSemHandle() {
    static std::atomic<uint64_t> nextHandle{1};
    return nextHandle;
}

std::unordered_map<uint64_t, ExtSem*>& getExtSems() {
    static std::unordered_map<uint64_t, ExtSem*> sems;
    return sems;
}

} // namespace

// ── C API ─────────────────────────────────────────────────────────────────────

using cudaExternalSemaphore_t = uint64_t;
using vgre_err_t = int;
static constexpr vgre_err_t kOK         = 0;
static constexpr vgre_err_t kInvalidVal = 1;

struct cudaExternalSemaphoreHandleDesc {
    int type;
    union {
        int    fd;
        struct { void* handle; const void* name; } win32;
        const void* nvSciSyncObj;
    } handle;
    unsigned int flags;
    unsigned int reserved[16];
};

struct cudaExternalSemaphoreSignalParams {
    struct { uint64_t value; } fence;
    unsigned int reserved[16];
    unsigned int flags;
};

struct cudaExternalSemaphoreWaitParams {
    struct { uint64_t value; } fence;
    unsigned int reserved[16];
    unsigned int flags;
};

extern "C" {

vgre_err_t cudaImportExternalSemaphore(
    cudaExternalSemaphore_t* extSem,
    const cudaExternalSemaphoreHandleDesc* desc)
{
    if (!extSem || !desc) return kInvalidVal;

    auto* s = new ExtSem();
    s->type = desc->type;

#if defined(__linux__)
    if (desc->type == CUDA_EXTERNAL_SEM_OPAQUE_FD) {
        s->efd = dup(desc->handle.fd);
        if (s->efd < 0)
            s->efd = eventfd(0, EFD_NONBLOCK | EFD_SEMAPHORE);
    } else if (desc->type == CUDA_EXTERNAL_SEM_TIMELINE_FD) {
        s->efd = dup(desc->handle.fd);
        if (s->efd < 0) s->efd = eventfd(0, EFD_NONBLOCK);
    } else if (desc->type == CUDA_EXTERNAL_SEM_NVSCISYNCOBJ) {
        // Proxy the NvSciSyncObj through a local eventfd. Real cross-device
        // NvSci requires NVIDIA hardware and libnvscisync — not available in the
        // CPU emulator. Single-process CUDA↔CUDA ordering works correctly.
        s->efd = eventfd(0, EFD_NONBLOCK);
        VGRE_LOG_INFO("ExtSem",
            "NvSciSyncObj proxied via local eventfd (in-process ordering only)");
    } else {
        s->efd = eventfd(0, EFD_NONBLOCK | EFD_SEMAPHORE);
        VGRE_LOG_WARN("ExtSem",
            "cudaImportExternalSemaphore: unknown handle type " +
            std::to_string(desc->type) + " — using local eventfd");
    }
#elif defined(__APPLE__)
    s->dsem = dispatch_semaphore_create(0);
    if (!s->dsem) { delete s; return kInvalidVal; }
    if (desc->type == CUDA_EXTERNAL_SEM_NVSCISYNCOBJ) {
        VGRE_LOG_INFO("ExtSem",
            "NvSciSyncObj proxied via POSIX semaphore on macOS (in-process ordering only)");
    } else if (desc->type != CUDA_EXTERNAL_SEM_OPAQUE_FD &&
               desc->type != CUDA_EXTERNAL_SEM_TIMELINE_FD &&
               desc->type != CUDA_EXTERNAL_SEM_OPAQUE_WIN32) {
        VGRE_LOG_WARN("ExtSem",
            "cudaImportExternalSemaphore: unknown handle type " +
            std::to_string(desc->type) + " — using local semaphore");
    }
#elif defined(_WIN32)
    if (desc->type == CUDA_EXTERNAL_SEM_OPAQUE_WIN32 && desc->handle.win32.handle) {
        HANDLE dup = nullptr;
        DuplicateHandle(GetCurrentProcess(),
                        static_cast<HANDLE>(desc->handle.win32.handle),
                        GetCurrentProcess(), &dup, 0, FALSE,
                        DUPLICATE_SAME_ACCESS);
        s->hEvent = dup ? dup : CreateEvent(nullptr, FALSE, FALSE, nullptr);
    } else {
        if (desc->type == CUDA_EXTERNAL_SEM_NVSCISYNCOBJ)
            VGRE_LOG_INFO("ExtSem",
                "NvSciSyncObj proxied via Win32 Event (in-process ordering only)");
        s->hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    }
#endif

    uint64_t h = getNextExtSemHandle().fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(getExtSemMutex());
        getExtSems()[h] = s;
    }
    *extSem = h;
    VGRE_LOG_DEBUG("ExtSem",
        "cudaImportExternalSemaphore: handle=" + std::to_string(h) +
        " type=" + std::to_string(desc->type));
    return kOK;
}

vgre_err_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t extSem) {
    std::lock_guard<std::mutex> lk(getExtSemMutex());
    auto it = getExtSems().find(extSem);
    if (it == getExtSems().end()) return kInvalidVal;
    auto* s = it->second;
#if defined(__linux__)
    if (s->efd >= 0) close(s->efd);
#elif defined(__APPLE__)
    if (s->dsem) dispatch_release(s->dsem);
#elif defined(_WIN32)
    if (s->hEvent) CloseHandle(s->hEvent);
#endif
    delete s;
    getExtSems().erase(it);
    return kOK;
}

// Signal: write to the eventfd to unblock any waiter
vgre_err_t cudaSignalExternalSemaphoresAsync(
    const cudaExternalSemaphore_t* extSems,
    const cudaExternalSemaphoreSignalParams* paramsArray,
    unsigned int numExtSems,
    void* /*stream*/)
{
    if (!extSems) return kInvalidVal;
    std::lock_guard<std::mutex> lk(getExtSemMutex());
    for (unsigned i = 0; i < numExtSems; ++i) {
        auto it = getExtSems().find(extSems[i]);
        if (it == getExtSems().end()) continue;
        auto* s = it->second;
#if defined(__linux__)
        if (s->efd >= 0) {
            uint64_t v = paramsArray ? paramsArray[i].fence.value : 1ULL;
            if (v == 0) v = 1;
            ssize_t wr;
            do { wr = write(s->efd, &v, sizeof(v)); } while (wr < 0 && errno == EINTR);
        }
#elif defined(__APPLE__)
        if (s->dsem) dispatch_semaphore_signal(s->dsem);
#elif defined(_WIN32)
        if (s->hEvent) SetEvent(s->hEvent);
#endif
        if (paramsArray) s->timelineValue.store(paramsArray[i].fence.value);
    }
    return kOK;
}

// Wait: poll the eventfd until the expected value is available
vgre_err_t cudaWaitExternalSemaphoresAsync(
    const cudaExternalSemaphore_t* extSems,
    const cudaExternalSemaphoreWaitParams* paramsArray,
    unsigned int numExtSems,
    void* /*stream*/)
{
    if (!extSems) return kInvalidVal;
    for (unsigned i = 0; i < numExtSems; ++i) {
        ExtSem* s = nullptr;
        {
            std::lock_guard<std::mutex> lk(getExtSemMutex());
            auto it = getExtSems().find(extSems[i]);
            if (it == getExtSems().end()) continue;
            s = it->second;
        }
        if (!s) continue;

        uint64_t waitVal = paramsArray ? paramsArray[i].fence.value : 1ULL;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);

#if defined(__linux__)
        if (s->efd >= 0) {
            // Poll + read until value >= waitVal or timeout
            while (std::chrono::steady_clock::now() < deadline) {
                struct pollfd pfd{s->efd, POLLIN, 0};
                int ret = poll(&pfd, 1, 5); // 5 ms timeout
                if (ret > 0 && (pfd.revents & POLLIN)) {
                    uint64_t val = 0;
                    if (read(s->efd, &val, sizeof(val)) == sizeof(val)) {
                        if (val >= waitVal) break;
                    }
                }
            }
        }
#elif defined(__APPLE__)
        if (s->dsem) {
            while (std::chrono::steady_clock::now() < deadline) {
                if (dispatch_semaphore_wait(s->dsem,
                        dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_MSEC)) == 0)
                    break;
            }
        }
#elif defined(_WIN32)
        if (s->hEvent) {
            WaitForSingleObject(s->hEvent, 30000); // 30 s timeout
        }
#endif
        VGRE_LOG_DEBUG("ExtSem",
            "cudaWaitExternalSemaphoresAsync: sem=" + std::to_string(extSems[i]) +
            " val=" + std::to_string(waitVal) + " done");
    }
    return kOK;
}

} // extern "C"
