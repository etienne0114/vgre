#ifndef VGRE_RUNTIME_VECTOR_ENGINE_H
#define VGRE_RUNTIME_VECTOR_ENGINE_H

#include "vgre/common/types.h"
#include "vgre/common/error_codes.h"

#include <cstddef>
#include <string>

namespace vgre {
namespace runtime {

// ── CPU SIMD feature flags ─────────────────────────────────────────────────
struct SIMDCapabilities {
    bool hasSSE2  = false;
    bool hasSSE4  = false;
    bool hasAVX   = false;
    bool hasAVX2  = false;
    bool hasAVX512= false;
    bool hasFMA   = false;
};

// ── Vector Engine ──────────────────────────────────────────────────────────
// Provides SIMD-accelerated math primitives used by the parallel executor
// and translation engine for inner-loop vectorization.
class VectorEngine {
public:
    VectorEngine();
    ~VectorEngine();

    // Detect CPU SIMD features at runtime
    const SIMDCapabilities& getCapabilities() const;
    std::string             getCapabilityString() const;

    // ── Vectorized float operations ────────────────────────────────────────
    void vectorAdd(const float* a, const float* b, float* c, size_t n);
    void vectorMul(const float* a, const float* b, float* c, size_t n);
    void vectorFMA(const float* a, const float* b, const float* c,
                   float* out, size_t n);  // out = a*b + c
    void vectorScale(const float* a, float scalar, float* out, size_t n);
    float vectorDot(const float* a, const float* b, size_t n);
    float vectorSum(const float* a, size_t n);

    // ── Vectorized double operations ───────────────────────────────────────
    void vectorAdd(const double* a, const double* b, double* c, size_t n);
    void vectorMul(const double* a, const double* b, double* c, size_t n);
    void vectorScale(const double* a, double scalar, double* out, size_t n);
    double vectorDot(const double* a, const double* b, size_t n);

    // ── Memory operations ──────────────────────────────────────────────────
    void vectorFill(float* dst, float value, size_t n);
    void vectorCopy(const float* src, float* dst, size_t n);

    // Singleton
    static VectorEngine& instance();

private:
    void detectCapabilities();

    SIMDCapabilities caps_;
};

} // namespace runtime
} // namespace vgre

#endif // VGRE_RUNTIME_VECTOR_ENGINE_H
