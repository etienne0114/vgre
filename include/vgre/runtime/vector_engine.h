#ifndef VGRE_RUNTIME_VECTOR_ENGINE_H
#define VGRE_RUNTIME_VECTOR_ENGINE_H



#include <cstddef>
#include <cstdint>
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
    bool hasVNNI  = false;
    bool hasAMX   = false;
};

// ── BFloat16 Support ───────────────────────────────────────────────────────
// Brain Floating Point (BF16) format: 1 sign bit, 8 exponent bits, 7 mantissa bits.
// This matches the top 16 bits of an IEEE 754 FP32 float.
typedef uint16_t vgre_bf16;

inline float bf16_to_fp32(vgre_bf16 h) {
    uint32_t f = static_cast<uint32_t>(h) << 16;
    return *reinterpret_cast<float*>(&f);
}

inline vgre_bf16 fp32_to_bf16(float f) {
    uint32_t i = *reinterpret_cast<uint32_t*>(&f);
    // Round-to-nearest-even tie-break
    uint32_t lsb = (i >> 16) & 1;
    uint32_t bias = 0x7FFF + lsb;
    return static_cast<vgre_bf16>((i + bias) >> 16);
}

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
    float vectorDot(const vgre_bf16* a, const vgre_bf16* b, size_t n);
    float vectorSum(const float* a, size_t n);
    void  vectorDiv(const float* a, const float* b, float* c, size_t n);
    void  vectorSqrt(const float* a, float* c, size_t n);
    double benchmarkFMA(size_t n, int iterations);
    double benchmarkBF16(size_t n, int iterations);

    // ── Vectorized double operations ───────────────────────────────────────
    void vectorAdd(const double* a, const double* b, double* c, size_t n);
    void vectorMul(const double* a, const double* b, double* c, size_t n);
    void   vectorScale(const double* a, double scalar, double* out, size_t n);
    double vectorDot(const double* a, const double* b, size_t n);
    void   vectorDiv(const double* a, const double* b, double* c, size_t n);
    void   vectorSqrt(const double* a, double* c, size_t n);

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
