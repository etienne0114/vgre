/**
 * VGRE CUDART Shim — Pitched N-D Memset
 *
 * Implements cudaMemset2D, cudaMemset3D and their async variants.
 * All operations write to CPU-side "device" memory managed by MemoryManager,
 * so pitched memset is a row-by-row (and slice-by-slice) std::memset.
 *
 * cudaMemset2D(devPtr, pitch, value, width, height)
 *   - Writes `value` to `width` bytes per row, with rows spaced `pitch` bytes apart.
 *
 * cudaMemset3D(pitchedDevPtr, value, extent)
 *   - Writes `value` to extent.width bytes per row across extent.height rows per
 *     slice, for extent.depth slices.  Row stride = pitchedDevPtr.pitch;
 *     slice stride = pitchedDevPtr.pitch * pitchedDevPtr.ysize.
 *
 * Async variants enqueue the work on the specified stream via the Scheduler.
 */

#include "vgre/api/cuda_interceptor.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"
#include "vgre/common/logger.h"

#include <cstring>
#include <functional>

using namespace vgre::api;

// ── 2D pitched memset ─────────────────────────────────────────────────────────

extern "C" cudaError_t cudaMemset2D(void *devPtr, size_t pitch,
                                     int value, size_t width, size_t height) {
    if (!devPtr)
        return cudaErrorInvalidValue;
    if (width == 0 || height == 0)
        return cudaSuccess;
    if (pitch < width)
        return cudaErrorInvalidValue;

    auto *base = static_cast<uint8_t *>(devPtr);
    const unsigned char fill = static_cast<unsigned char>(value);
    for (size_t row = 0; row < height; ++row)
        std::memset(base + row * pitch, fill, width);

    VGRE_LOG_DEBUG("CUDART", "cudaMemset2D: " + std::to_string(width) +
                   "x" + std::to_string(height) + " pitch=" + std::to_string(pitch));
    return cudaSuccess;
}

extern "C" cudaError_t cudaMemset2DAsync(void *devPtr, size_t pitch,
                                          int value, size_t width, size_t height,
                                          cudaStream_t stream) {
    if (!devPtr)
        return cudaErrorInvalidValue;
    if (width == 0 || height == 0)
        return cudaSuccess;
    if (pitch < width)
        return cudaErrorInvalidValue;

    // Capture parameters by value for the async lambda.
    auto task = [devPtr, pitch, value, width, height]() {
        auto *base = static_cast<uint8_t *>(devPtr);
        const unsigned char fill = static_cast<unsigned char>(value);
        for (size_t row = 0; row < height; ++row)
            std::memset(base + row * pitch, fill, width);
    };

    auto &sched = vgre::core::Scheduler::instance();
    sched.submitStreamTask(static_cast<vgre::StreamId>(stream),
                           std::move(task));
    return cudaSuccess;
}

// ── 3D pitched memset ─────────────────────────────────────────────────────────

// Mirror the CUDA-compatible cudaPitchedPtr and cudaExtent here so this file
// is self-contained.  The actual types live in cuda_interceptor.h under the
// vgre::api namespace but are also exposed as plain structs to the C ABI.

extern "C" cudaError_t cudaMemset3D(struct cudaPitchedPtr pitchedDevPtr,
                                     int value, struct cudaExtent extent) {
    if (!pitchedDevPtr.ptr)
        return cudaErrorInvalidValue;
    if (extent.width == 0 || extent.height == 0 || extent.depth == 0)
        return cudaSuccess;
    if (pitchedDevPtr.pitch < extent.width)
        return cudaErrorInvalidValue;

    // pitchedDevPtr.pitch  = bytes per row (including padding)
    // pitchedDevPtr.ysize  = logical height of one slice in rows
    // Slice stride = pitch * ysize
    const size_t pitch      = pitchedDevPtr.pitch;
    const size_t sliceRows  = pitchedDevPtr.ysize > 0
                              ? pitchedDevPtr.ysize : extent.height;
    const size_t sliceStride = pitch * sliceRows;
    auto *base = static_cast<uint8_t *>(pitchedDevPtr.ptr);
    const unsigned char fill = static_cast<unsigned char>(value);

    for (size_t d = 0; d < extent.depth; ++d) {
        uint8_t *sliceBase = base + d * sliceStride;
        for (size_t h = 0; h < extent.height; ++h)
            std::memset(sliceBase + h * pitch, fill, extent.width);
    }

    VGRE_LOG_DEBUG("CUDART",
                   "cudaMemset3D: " + std::to_string(extent.width) + "x" +
                   std::to_string(extent.height) + "x" + std::to_string(extent.depth) +
                   " pitch=" + std::to_string(pitch));
    return cudaSuccess;
}

extern "C" cudaError_t cudaMemset3DAsync(struct cudaPitchedPtr pitchedDevPtr,
                                          int value, struct cudaExtent extent,
                                          cudaStream_t stream) {
    if (!pitchedDevPtr.ptr)
        return cudaErrorInvalidValue;
    if (extent.width == 0 || extent.height == 0 || extent.depth == 0)
        return cudaSuccess;
    if (pitchedDevPtr.pitch < extent.width)
        return cudaErrorInvalidValue;

    const size_t pitch      = pitchedDevPtr.pitch;
    const size_t sliceRows  = pitchedDevPtr.ysize > 0
                              ? pitchedDevPtr.ysize : extent.height;
    const size_t sliceStride = pitch * sliceRows;

    auto task = [pitchedDevPtr, value, extent, pitch, sliceStride]() {
        auto *base = static_cast<uint8_t *>(pitchedDevPtr.ptr);
        const unsigned char fill = static_cast<unsigned char>(value);
        for (size_t d = 0; d < extent.depth; ++d) {
            uint8_t *sliceBase = base + d * sliceStride;
            for (size_t h = 0; h < extent.height; ++h)
                std::memset(sliceBase + h * pitch, fill, extent.width);
        }
    };

    auto &sched = vgre::core::Scheduler::instance();
    sched.submitStreamTask(static_cast<vgre::StreamId>(stream),
                           std::move(task));
    return cudaSuccess;
}
