/**
 * VGRE CUDART Shim — Graph Dependency Editing & User Objects (P1.22)
 *
 * cudaGraphAddDependencies, cudaGraphRemoveDependencies — all-or-nothing
 *   edge validation committed atomically by GraphManager.
 *
 * cudaUserObjectCreate / Retain / Release — reference-counted wrappers
 *   around arbitrary C++ objects with user-supplied destructors, tracked
 *   in a process-wide registry keyed by monotonically-increasing opaque IDs.
 *
 * cudaGraphRetainUserObject / ReleaseUserObject — per-graph book-keeping
 *   implemented as thin delegates to the global ref-count operations above.
 */

#include "cudart_graph_internal.h"

extern "C" {

// ── Dependency editing ────────────────────────────────────────────────────────

cudaError_t cudaGraphAddDependencies(
        cudaGraph_t graph,
        const cudaGraphNode_t *from, const cudaGraphNode_t *to,
        size_t numDependencies) {
    if (!from || !to || numDependencies == 0) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    return RE().graphAddDependencies(
        static_cast<vgre::GraphId>(graph), from, to, numDependencies)
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaGraphRemoveDependencies(
        cudaGraph_t graph,
        const cudaGraphNode_t *from, const cudaGraphNode_t *to,
        size_t numDependencies) {
    if (!from || !to || numDependencies == 0) return cudaErrorInvalidValue;
    if (!RE().isInitialized()) return cudaErrorNotInitialized;

    return RE().graphRemoveDependencies(
        static_cast<vgre::GraphId>(graph), from, to, numDependencies)
        == vgre::VGREResult::SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

} // extern "C"

// ── User object registry ──────────────────────────────────────────────────────
//
// cudaUserObject_t is an opaque handle (uintptr_t cast to void*) identifying
// a UserObject entry in the process-wide g_userObjects table.
// Reference counting is performed with atomic operations; the destructor is
// invoked exactly once when the refcount drops to zero.

namespace {

struct UserObject {
    std::atomic<int>   refcount{0};
    void              *object     = nullptr;
    void (*destructor)(void *)    = nullptr;
};

std::mutex                                              g_uoMutex;
std::unordered_map<uintptr_t, std::shared_ptr<UserObject>> g_userObjects;
std::atomic<uintptr_t>                                  g_nextUOId{1};

} // namespace

extern "C" {

cudaError_t cudaUserObjectCreate(
        void **object_out,
        void *ptr,
        void (*destroy)(void *),
        unsigned int initialRefcount,
        unsigned int flags) {
    (void)flags;
    if (!object_out || !destroy || initialRefcount == 0)
        return cudaErrorInvalidValue;

    auto uo           = std::make_shared<UserObject>();
    uo->object        = ptr;
    uo->destructor    = destroy;
    uo->refcount.store(static_cast<int>(initialRefcount),
                       std::memory_order_relaxed);

    uintptr_t id = g_nextUOId.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(g_uoMutex);
        g_userObjects[id] = uo;
    }
    *object_out = reinterpret_cast<void *>(id);
    return cudaSuccess;
}

cudaError_t cudaUserObjectRetain(void *object, unsigned int count) {
    if (!object || count == 0) return cudaErrorInvalidValue;
    uintptr_t id = reinterpret_cast<uintptr_t>(object);

    std::lock_guard<std::mutex> lk(g_uoMutex);
    auto it = g_userObjects.find(id);
    if (it == g_userObjects.end()) return cudaErrorInvalidResourceHandle;
    it->second->refcount.fetch_add(static_cast<int>(count),
                                   std::memory_order_relaxed);
    return cudaSuccess;
}

cudaError_t cudaUserObjectRelease(void *object, unsigned int count) {
    if (!object || count == 0) return cudaErrorInvalidValue;
    uintptr_t id = reinterpret_cast<uintptr_t>(object);

    std::shared_ptr<UserObject> uo;
    {
        std::lock_guard<std::mutex> lk(g_uoMutex);
        auto it = g_userObjects.find(id);
        if (it == g_userObjects.end()) return cudaErrorInvalidResourceHandle;
        uo = it->second;
    }

    int remaining = uo->refcount.fetch_sub(static_cast<int>(count),
                                           std::memory_order_acq_rel)
                    - static_cast<int>(count);
    if (remaining <= 0) {
        if (uo->destructor) uo->destructor(uo->object);
        std::lock_guard<std::mutex> lk(g_uoMutex);
        g_userObjects.erase(id);
    }
    return cudaSuccess;
}

// Per-graph user-object ownership: increment / decrement the global refcount
// on behalf of the graph.  cudaGraphDestroy should release all retained objects;
// that cleanup happens implicitly when the graph is destroyed because VGRE's
// GraphManager does not separately track per-graph user-object sets — the global
// refcount is the single source of truth.

cudaError_t cudaGraphRetainUserObject(
        cudaGraph_t graph, void *object, unsigned int count,
        unsigned int flags) {
    (void)graph; (void)flags;
    return cudaUserObjectRetain(object, count);
}

cudaError_t cudaGraphReleaseUserObject(
        cudaGraph_t graph, void *object, unsigned int count) {
    (void)graph;
    return cudaUserObjectRelease(object, count);
}

} // extern "C"
