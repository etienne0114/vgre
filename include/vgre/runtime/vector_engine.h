#ifndef VGRE_RUNTIME_VECTOR_ENGINE_H
#define VGRE_RUNTIME_VECTOR_ENGINE_H



#include <cstddef>
#include <cstdint>
#include <string>
#include <cstring>

namespace vgre {
namespace runtime {

// ── CPU SIMD feature flags ─────────────────────────────────────────────────
struct SIMDCapabilities {
    bool hasSSE2    = false;
    bool hasSSE4    = false;
    bool hasAVX     = false;
    bool hasAVX2    = false;
    bool hasAVX512  = false;
    bool hasFMA     = false;
    bool hasVNNI    = false;    // AVX-512 VNNI (CPUID 7.0 ECX[11], Ice Lake+)
    bool hasAVXVNNI = false;   // AVX-VNNI without AVX-512 (CPUID 7.1 EAX[4], Alder Lake+)
    // AMX sub-features (Intel Sapphire Rapids / Granite Rapids)
    bool hasAMXTile = false;   // CPUID 7.0 EDX[24] — AMX-TILE base support
    bool hasAMXBF16 = false;   // CPUID 7.0 EDX[22] — AMX-BF16 dot-product
    // True only when CPUID bits set AND OS has granted XTILEDATA via arch_prctl
    bool amxEnabled = false;
    // Legacy alias kept for existing callers
    bool hasAMX     = false;   // set to (amxEnabled) after init
};

// ── BFloat16 Support ───────────────────────────────────────────────────────
// Brain Floating Point (BF16) format: 1 sign bit, 8 exponent bits, 7 mantissa bits.
// This matches the top 16 bits of an IEEE 754 FP32 float.
typedef uint16_t vgre_bf16;

inline float bf16_to_fp32(vgre_bf16 h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f = 0.0f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

inline vgre_bf16 fp32_to_bf16(float f) {
    uint32_t u = 0;
    memcpy(&u, &f, sizeof(u));
    // Round-to-nearest-even tie-break
    uint32_t lsb = (u >> 16) & 1;
    uint32_t bias = 0x7FFF + lsb;
    return static_cast<vgre_bf16>((u + bias) >> 16);
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
    // Inverse square root via Halley's 3rd-order iteration (QUEUE-23).
    // |error| < 2^-23 in ≤ 2 iterations for all normal IEEE-754 floats.
    // Special cases: a=0 → +Inf, a=+Inf → 0, a<0 or NaN → NaN.
    void  vectorRsqrt(const float* a, float* out, size_t n);
    double benchmarkFMA(size_t n, int iterations);
    double benchmarkBF16(size_t n, int iterations);

    // ── Vectorized double operations ───────────────────────────────────────
    void vectorAdd(const double* a, const double* b, double* c, size_t n);
    void vectorMul(const double* a, const double* b, double* c, size_t n);
    void   vectorScale(const double* a, double scalar, double* out, size_t n);
    double vectorDot(const double* a, const double* b, size_t n);
    void   vectorDiv(const double* a, const double* b, double* c, size_t n);
    void   vectorSqrt(const double* a, double* c, size_t n);

    // ── Element-wise unary / binary reduction ops ─────────────────────────
    // Element-wise min / max
    void vectorMin(const float* a, const float* b, float* out, size_t n);
    void vectorMax(const float* a, const float* b, float* out, size_t n);
    // Rectified Linear Unit: out[i] = max(a[i], 0)
    void vectorReLU(const float* a, float* out, size_t n);
    // Absolute value: out[i] = |a[i]|
    void vectorAbs(const float* a, float* out, size_t n);
    // Clamp: out[i] = min(max(a[i], lo), hi)
    void vectorClamp(const float* a, float lo, float hi, float* out, size_t n);

    // ── Memory operations ──────────────────────────────────────────────────
    void vectorFill(float* dst, float value, size_t n);
    void vectorCopy(const float* src, float* dst, size_t n);

    // ── Matrix multiply (AMX / VNNI accelerated, Phase 12-B) ──────────────
    // INT8 GEMM: C = A × B, row-major, signed INT8 inputs → INT32 output.
    // Uses AVX-VNNI on Alder Lake+, AMX-INT8 on Sapphire Rapids+,
    // falls back to AVX2 INT16 widening or scalar.
    // Requires ENABLE_VGRE_AMX at compile time; scalar fallback always present.
    void matMulInt8(const int8_t* A, const int8_t* B, int32_t* C,
                    int M, int N, int K);

    // BF16 GEMM: C = A × B, row-major, BF16 inputs → FP32 output.
    // Uses AMX-BF16 on Sapphire Rapids+, falls back to AVX2 FP32 or scalar.
    void matMulBF16(const vgre_bf16* A, const vgre_bf16* B, float* C,
                    int M, int N, int K);

    // Singleton
    static VectorEngine& instance();

private:
    void detectCapabilities();

    SIMDCapabilities caps_;
};

} // namespace runtime
} // namespace vgre

#endif // VGRE_RUNTIME_VECTOR_ENGINE_H
