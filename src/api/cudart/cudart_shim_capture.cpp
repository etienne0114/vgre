/**
 * VGRE CUDART Shim — Stream Capture Introspection
 *
 * P1.19: cudaStreamIsCapturing, cudaStreamGetCaptureInfo,
 *         cudaStreamGetCaptureInfo_v2, cudaThreadExchangeStreamCaptureMode,
 *         cudaStreamUpdateCaptureDependencies, cudaStreamCopyAttributes
 *
 * Capture state is maintained by RuntimeEngine::captureState_ and
 * captureSeedDeps_.  These APIs expose that state to the caller without
 * mutating it (except ThreadExchangeStreamCaptureMode which swaps a
 * per-thread capture mode flag).
 */

#include "vgre/api/cuda_interceptor.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/common/logger.h"

#include <atomic>
#include <mutex>
#include <unordered_map>

using namespace vgre::api;

// ── Per-thread capture mode (cudaStreamCaptureMode) ──────────────────────────

// cudaStreamCaptureModeGlobal   = 0
// cudaStreamCaptureModeThreadLocal = 1  (most restrictive)
// cudaStreamCaptureModeRelaxed  = 2  (least restrictive)

namespace {

thread_local int t_captureMode = 0;  // cudaStreamCaptureModeGlobal

} // namespace

// ── cudaStreamIsCapturing ─────────────────────────────────────────────────────

extern "C" cudaError_t cudaStreamIsCapturing(
        cudaStream_t stream, unsigned int *pCaptureStatus) {
    if (!pCaptureStatus) return cudaErrorInvalidValue;
    auto &re = vgre::core::RuntimeEngine::instance();
    if (!re.isInitialized()) return cudaErrorNotInitialized;

    // cudaStreamCaptureStatusNone    = 0
    // cudaStreamCaptureStatusActive  = 1
    // cudaStreamCaptureStatusInvalidated = 2
    bool capturing = re.isStreamCapturing(
        static_cast<vgre::StreamId>(stream));
    *pCaptureStatus = capturing ? 1u : 0u;
    return cudaSuccess;
}

// ── cudaStreamGetCaptureInfo ──────────────────────────────────────────────────

extern "C" cudaError_t cudaStreamGetCaptureInfo(
        cudaStream_t stream,
        unsigned int *pCaptureStatus,
        unsigned long long *pId) {
    if (!pCaptureStatus) return cudaErrorInvalidValue;
    auto &re = vgre::core::RuntimeEngine::instance();
    if (!re.isInitialized()) return cudaErrorNotInitialized;

    vgre::GraphId captureGraphId = 0;
    bool capturing = re.getStreamCaptureInfo(
        static_cast<vgre::StreamId>(stream), captureGraphId);

    *pCaptureStatus = capturing ? 1u : 0u;
    if (pId) *pId = static_cast<unsigned long long>(captureGraphId);
    return cudaSuccess;
}

// ── cudaStreamGetCaptureInfo_v2 ───────────────────────────────────────────────
// v2 additionally returns the current capture dependency set and the
// capture mode that was in effect when capture began.

extern "C" cudaError_t cudaStreamGetCaptureInfo_v2(
        cudaStream_t stream,
        unsigned int *pCaptureStatus,
        unsigned long long *pId,
        cudaGraph_t *pGraph,
        const unsigned long long **pDependencies,
        size_t *pNumDependencies) {
    if (!pCaptureStatus) return cudaErrorInvalidValue;
    auto &re = vgre::core::RuntimeEngine::instance();
    if (!re.isInitialized()) return cudaErrorNotInitialized;

    vgre::GraphId captureGraphId = 0;
    std::vector<uint64_t> captureDeps;
    bool capturing = re.getStreamCaptureInfoV2(
        static_cast<vgre::StreamId>(stream),
        captureGraphId, captureDeps);

    *pCaptureStatus = capturing ? 1u : 0u;
    if (pId)    *pId    = static_cast<unsigned long long>(captureGraphId);
    if (pGraph) *pGraph = static_cast<cudaGraph_t>(captureGraphId);

    // Return the frontier dependency IDs. We store them in thread-local
    // storage that remains valid until the next call from this thread.
    thread_local std::vector<unsigned long long> t_depBuffer;
    t_depBuffer.clear();
    for (auto d : captureDeps)
        t_depBuffer.push_back(static_cast<unsigned long long>(d));

    if (pDependencies)    *pDependencies    = t_depBuffer.data();
    if (pNumDependencies) *pNumDependencies = t_depBuffer.size();
    return cudaSuccess;
}

// ── cudaThreadExchangeStreamCaptureMode ───────────────────────────────────────

extern "C" cudaError_t cudaThreadExchangeStreamCaptureMode(
        unsigned int *mode) {
    if (!mode) return cudaErrorInvalidValue;
    // Swap the caller's value with the thread-local capture mode.
    int prev = t_captureMode;
    int next = static_cast<int>(*mode);
    if (next < 0 || next > 2) return cudaErrorInvalidValue;
    t_captureMode = next;
    *mode = static_cast<unsigned int>(prev);
    return cudaSuccess;
}

// ── cudaStreamUpdateCaptureDependencies ───────────────────────────────────────

extern "C" cudaError_t cudaStreamUpdateCaptureDependencies(
        cudaStream_t stream,
        uint64_t *dependencies, size_t numDependencies,
        unsigned int flags) {
    if (!dependencies && numDependencies > 0) return cudaErrorInvalidValue;
    auto &re = vgre::core::RuntimeEngine::instance();
    if (!re.isInitialized()) return cudaErrorNotInitialized;

    // flags: 0 = add deps to frontier; 1 = replace frontier with deps.
    std::vector<uint64_t> deps(dependencies, dependencies + numDependencies);
    bool replace = (flags == 1u);
    auto r = re.streamUpdateCaptureDependencies(
        static_cast<vgre::StreamId>(stream), deps, replace);
    return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
}

// ── cudaStreamCopyAttributes ──────────────────────────────────────────────────
// Copy stream attributes (priority, sync policy, access-window policy) from
// source to destination stream.  VGRE tracks these in cudart_shim_stream.cpp's
// g_streamMeta map; we expose a shim that copies the metadata.

extern "C" cudaError_t cudaStreamCopyAttributes(
        cudaStream_t dst, cudaStream_t src) {
    auto &re = vgre::core::RuntimeEngine::instance();
    if (!re.isInitialized()) return cudaErrorNotInitialized;
    // Delegate to RuntimeEngine stream attribute copy.
    auto r = re.streamCopyAttributes(
        static_cast<vgre::StreamId>(dst),
        static_cast<vgre::StreamId>(src));
    return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
}
