// In-tree MXFP4 (OCP Microscaling FP4) codec + matmul — see vgre/xla/mxfp4.h.

#include "vgre/xla/mxfp4.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vgre {
namespace xla {
namespace mxfp4 {

namespace {

// E2M1 magnitudes indexed by the 3-bit code (exp:mant). Bit 3 of a full code is
// the sign. {0, .5, 1, 1.5, 2, 3, 4, 6}.
constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

inline float decode(uint8_t code) {
    const float m = kMag[code & 0x7];
    return (code & 0x8) ? -m : m;
}

// Round a magnitude to the nearest E2M1 code (0..7).
inline uint8_t encode_mag(float a) {
    uint8_t best = 0;
    float bestErr = std::fabs(a - kMag[0]);
    for (uint8_t i = 1; i < 8; ++i) {
        const float e = std::fabs(a - kMag[i]);
        if (e < bestErr) { bestErr = e; best = i; }
    }
    return best;
}

inline uint8_t encode(float v) {
    const uint8_t sign = (v < 0.0f) ? 0x8 : 0x0;
    return sign | encode_mag(std::fabs(v));
}

// Shared block scale as an E8M0 exponent byte: 2^(floor(log2(amax))−2), so the
// block max normalizes into the E2M1 range. Returns 0 for an all-zero block.
inline uint8_t block_scale_byte(float amax) {
    if (amax <= 0.0f) return 0;
    int p = (int)std::floor(std::log2(amax)) - 2;
    int biased = p + 127;
    if (biased < 0) biased = 0;
    if (biased > 255) biased = 255;
    return (uint8_t)biased;
}

inline float scale_from_byte(uint8_t b) { return std::ldexp(1.0f, (int)b - 127); }

inline void put_code(uint8_t* codes, int64_t idx, uint8_t code) {
    uint8_t& byte = codes[idx >> 1];
    if (idx & 1) byte = (uint8_t)((byte & 0x0F) | (code << 4));
    else         byte = (uint8_t)((byte & 0xF0) | (code & 0x0F));
}

inline uint8_t get_code(const uint8_t* codes, int64_t idx) {
    const uint8_t byte = codes[idx >> 1];
    return (idx & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
}

}  // namespace

void quantize(int64_t K, int64_t N, const float* W,
              uint8_t* codes, uint8_t* scales) {
    std::memset(codes, 0, (size_t)num_code_bytes(K, N));
    const int64_t nblk = (K + kBlock - 1) / kBlock;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t b = 0; b < nblk; ++b) {
            const int64_t k0 = b * kBlock, k1 = std::min(k0 + kBlock, K);
            float amax = 0.0f;
            for (int64_t k = k0; k < k1; ++k) amax = std::max(amax, std::fabs(W[k * N + n]));
            const uint8_t sb = block_scale_byte(amax);
            scales[b * N + n] = sb;
            const float inv = (amax > 0.0f) ? 1.0f / scale_from_byte(sb) : 0.0f;
            for (int64_t k = k0; k < k1; ++k)
                put_code(codes, k * N + n, encode(W[k * N + n] * inv));
        }
    }
}

void dequantize(int64_t K, int64_t N, const uint8_t* codes, const uint8_t* scales,
                float* W) {
    for (int64_t k = 0; k < K; ++k) {
        const int64_t b = k / kBlock;
        for (int64_t n = 0; n < N; ++n) {
            const float s = scale_from_byte(scales[b * N + n]);
            W[k * N + n] = decode(get_code(codes, k * N + n)) * s;
        }
    }
}

void gemm(int64_t M, int64_t N, int64_t K,
          const float* A, const uint8_t* codes, const uint8_t* scales, float* C) {
    const int64_t nblk = (K + kBlock - 1) / kBlock;
    // For each output column, accumulate block by block so the shared scale is
    // applied once per 32-element block (dequant-in-GEMM: the E2M1 code stays
    // 4-bit until multiplied by the activation).
    for (int64_t m = 0; m < M; ++m) {
        const float* a = A + m * K;
        float* c = C + m * N;
        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t b = 0; b < nblk; ++b) {
                const int64_t k0 = b * kBlock, k1 = std::min(k0 + kBlock, K);
                const float s = scale_from_byte(scales[b * N + n]);
                float blk = 0.0f;
                for (int64_t k = k0; k < k1; ++k)
                    blk += a[k] * decode(get_code(codes, k * N + n));
                acc += blk * s;                       // scale factors out of the block
            }
            c[n] = acc;
        }
    }
}

}  // namespace mxfp4
}  // namespace xla
}  // namespace vgre
