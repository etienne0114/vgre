// CUDA IPC Memory API — routes cudaIpcGetMemHandle / cudaIpcOpenMemHandle to
// VGRE's existing POSIX shared-memory subsystem (ShmManager).
//
// Handle layout (64 bytes, opaque to callers):
//   [0..7]   magic    = 0x5647524549504300ULL ("VGREIPC\0")
//   [8..15]  devPtr   = host virtual address of the VGRE allocation
//   [16..23] size     = allocation size in bytes
//   [24..55] shmName  = POSIX shm name (null-terminated, max 32 chars)
//   [56..63] pad
//
// Multiple processes open the same shm segment by name; the pointer returned
// to the opener is the base of the mapped region, adjusted to the sub-offset
// within the original allocation.

#include "vgre/core/shm_manager.h"
#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/error_codes.h"
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <memory>

// ── Handle layout ─────────────────────────────────────────────────────────────
static constexpr uint64_t kIPCMagic = 0x5647524549504300ULL;
static constexpr int      kHandleSize = 64;
static constexpr int      kShmNameOff = 24;
static constexpr int      kShmNameMax = 32;

struct IPCHandle {
    uint64_t magic;
    uint64_t devPtr;
    uint64_t size;
    char     shmName[kShmNameMax];
    uint8_t  pad[8];
};
static_assert(sizeof(IPCHandle) == kHandleSize, "IPC handle size mismatch");

// cudaIpcMemHandle_t is 64 bytes in CUDA headers; replicate that.
struct cudaIpcMemHandle_st { char reserved[kHandleSize]; };
typedef cudaIpcMemHandle_st cudaIpcMemHandle_t;

// ── Open-handle registry ──────────────────────────────────────────────────────
namespace {
struct OpenedRegion {
    std::shared_ptr<vgre::core::ShmManager> shm;
    void* mappedPtr = nullptr;   // base of mapped region
    size_t size     = 0;
};

static std::mutex            g_ipcMutex;
static std::atomic<uint64_t> g_ipcCounter{1};
static std::unordered_map<void*, OpenedRegion> g_opened; // key = returned devPtr
} // namespace

// ── C API ─────────────────────────────────────────────────────────────────────
extern "C" {

// cudaIpcGetMemHandle: export a VGRE allocation for use by other processes.
int cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr) {
    if (!handle || !devPtr) return 1; // cudaErrorInvalidValue

    auto& mm = vgre::core::MemoryManager::instance();
    size_t size = mm.getAllocationSize(devPtr);
    if (size == 0) {
        VGRE_LOG_ERROR("cudaIPC", "cudaIpcGetMemHandle: pointer not a VGRE allocation");
        return 1;
    }
    void* rawPtr = mm.getPointer(devPtr);
    if (!rawPtr) rawPtr = devPtr;

    // Create or reuse a POSIX SHM segment named after the address.
    // Name pattern: /vgre_ipc_<pid>_<counter>
    uint64_t cnt = g_ipcCounter.fetch_add(1, std::memory_order_relaxed);
    char shmName[kShmNameMax];
    std::snprintf(shmName, sizeof(shmName), "/vgre_ipc_%llu",
                  (unsigned long long)cnt);

    auto shm = std::make_shared<vgre::core::ShmManager>();
    if (shm->open(std::string(shmName), size, /*create=*/true) != vgre::VGREResult::SUCCESS) {
        VGRE_LOG_ERROR("cudaIPC", "Failed to create SHM segment: " + std::string(shmName));
        return 2; // cudaErrorMemoryAllocation
    }

    // Copy current device memory into the SHM segment so remote processes
    // see the initial values.
    std::memcpy(shm->getBasePtr(), rawPtr, size);

    // Store the opened region so we can sync back on close.
    {
        std::lock_guard<std::mutex> lk(g_ipcMutex);
        g_opened[devPtr] = {shm, shm->getBasePtr(), size};
    }

    // Write handle
    IPCHandle h{};
    h.magic  = kIPCMagic;
    h.devPtr = reinterpret_cast<uint64_t>(devPtr);
    h.size   = static_cast<uint64_t>(size);
    std::strncpy(h.shmName, shmName, kShmNameMax - 1);
    std::memcpy(handle->reserved, &h, sizeof(IPCHandle));

    VGRE_LOG_INFO("cudaIPC", "IPC handle created: " + std::string(shmName) +
                  " size=" + std::to_string(size));
    return 0; // cudaSuccess
}

// cudaIpcOpenMemHandle: import a handle exported by another process.
// flags: cudaIpcMemLazyEnablePeerAccess (ignored — CPU always has access).
int cudaIpcOpenMemHandle(void** devPtr, cudaIpcMemHandle_t handle, unsigned int /*flags*/) {
    if (!devPtr) return 1;

    IPCHandle h{};
    std::memcpy(&h, handle.reserved, sizeof(IPCHandle));
    if (h.magic != kIPCMagic) {
        VGRE_LOG_ERROR("cudaIPC", "Invalid IPC handle magic");
        return 1;
    }
    if (h.size == 0 || h.shmName[0] == '\0') return 1;

    auto shm = std::make_shared<vgre::core::ShmManager>();
    if (shm->open(std::string(h.shmName), static_cast<size_t>(h.size), /*create=*/false)
        != vgre::VGREResult::SUCCESS)
    {
        VGRE_LOG_ERROR("cudaIPC", "Failed to open SHM segment: " + std::string(h.shmName));
        return 2;
    }

    void* ptr = shm->getBasePtr();
    {
        std::lock_guard<std::mutex> lk(g_ipcMutex);
        g_opened[ptr] = {shm, ptr, static_cast<size_t>(h.size)};
    }
    *devPtr = ptr;

    VGRE_LOG_INFO("cudaIPC", "IPC handle opened: " + std::string(h.shmName) +
                  " → " + std::to_string(reinterpret_cast<uintptr_t>(ptr)));
    return 0;
}

// cudaIpcCloseMemHandle: unmap an imported handle.
int cudaIpcCloseMemHandle(void* devPtr) {
    std::lock_guard<std::mutex> lk(g_ipcMutex);
    auto it = g_opened.find(devPtr);
    if (it == g_opened.end()) return 1;
    // ShmManager destructor closes the mapping when shared_ptr refcount drops.
    g_opened.erase(it);
    VGRE_LOG_DEBUG("cudaIPC", "IPC handle closed");
    return 0;
}

// ── CUDA IPC Event Handles ────────────────────────────────────────────────────
// VGRE events are host-side std::atomic<int> flags.  We serialize the event's
// shared-memory name into the 64-byte handle so another process can open it.
//
// Handle layout (64 bytes):
//   [0..7]  magic   = 0x5647524549504345ULL ("VGREIPC\0")  (different from mem magic)
//   [8..15] eventId = atomic counter, unique per event export
//   [16..47] shmName = POSIX shm name (/vgre_ev_<id>)
//   [48..63] pad

static constexpr uint64_t kEventIPCMagic = 0x5647524549504345ULL;

struct EventIPCHandle { uint64_t magic; uint64_t eventId; char shmName[32]; uint8_t pad[16]; };
static_assert(sizeof(EventIPCHandle) == 64, "event IPC handle must be 64 bytes");

struct EventIPCHandle_st { char reserved[64]; };

static std::atomic<uint64_t> g_eventIpcCounter{1};
// Map exported eventId → ShmManager holding a 4-byte flag (1=signalled, 0=not)
static std::unordered_map<uint64_t, std::shared_ptr<vgre::core::ShmManager>> g_eventShm;

int cudaIpcGetEventHandle(EventIPCHandle_st* handle, void* event) {
    if (!handle || !event) return 1;

    uint64_t eid = g_eventIpcCounter.fetch_add(1, std::memory_order_relaxed);
    char shmName[32];
    std::snprintf(shmName, sizeof(shmName), "/vgre_ev_%llu", (unsigned long long)eid);

    // Allocate a 4-byte SHM segment that mirrors the event's signalled state.
    auto shm = std::make_shared<vgre::core::ShmManager>();
    if (shm->open(std::string(shmName), 4, true) != vgre::VGREResult::SUCCESS) {
        VGRE_LOG_ERROR("cudaIPC", "cudaIpcGetEventHandle: failed to create event SHM");
        return 2;
    }
    // Copy current event signalled state (VGRE events are host atomics at *event).
    int32_t state = *reinterpret_cast<std::atomic<int32_t>*>(event);
    std::memcpy(shm->getBasePtr(), &state, 4);

    {
        std::lock_guard<std::mutex> lk(g_ipcMutex);
        g_eventShm[eid] = shm;
    }

    EventIPCHandle h{};
    h.magic   = kEventIPCMagic;
    h.eventId = eid;
    std::strncpy(h.shmName, shmName, sizeof(h.shmName) - 1);
    std::memcpy(handle->reserved, &h, 64);
    return 0;
}

int cudaIpcOpenEventHandle(void** event, EventIPCHandle_st handle) {
    if (!event) return 1;
    EventIPCHandle h{};
    std::memcpy(&h, handle.reserved, 64);
    if (h.magic != kEventIPCMagic) return 1;

    auto shm = std::make_shared<vgre::core::ShmManager>();
    if (shm->open(std::string(h.shmName), 4, false) != vgre::VGREResult::SUCCESS) {
        VGRE_LOG_ERROR("cudaIPC", "cudaIpcOpenEventHandle: cannot open event SHM");
        return 2;
    }
    // Return the base pointer of the SHM as the event handle; callers can
    // read/write the 4-byte flag directly (VGRE event semantic).
    *event = shm->getBasePtr();
    {
        std::lock_guard<std::mutex> lk(g_ipcMutex);
        g_eventShm[h.eventId] = shm; // keep alive
    }
    return 0;
}

} // extern "C"
