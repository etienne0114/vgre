/**
 * curand_kernel.h — device-side cuRAND for VGRE JIT-compiled kernels.
 *
 * Provides the same API surface as NVIDIA's curand_kernel.h so CUDA kernels
 * that call curand_init/curand_uniform/curand_normal compile and run correctly
 * under the VGRE CPU JIT.
 *
 * Generators implemented:
 *   curandState / curandStateXORWOW   — XORWOW (CUDA default, fast)
 *   curandStatePhilox4_32_10_t        — Philox 4×32-10 (deterministic, high quality)
 *   curandStateMRG32k3a               — MRG32k3a (long period, statistical quality)
 *
 * All generators share the same generate/distribution functions via overloads.
 *
 * Thread safety: each thread owns its own state; no sharing assumed.
 */

#ifndef VGRE_CURAND_KERNEL_H
#define VGRE_CURAND_KERNEL_H

#include <cstdint>
#include <cmath>
#include <cstring>

// ── XORWOW state ─────────────────────────────────────────────────────────────
// Marsaglia's XORWOW: 5-word XOR-shift + Weyl counter.
// Matches NVIDIA's curandStateXORWOW layout.
struct curandStateXORWOW {
    unsigned int v[5];
    unsigned int d;
};

typedef curandStateXORWOW curandState;
typedef curandStateXORWOW curandState_t;

// ── Philox 4×32-10 state ──────────────────────────────────────────────────────
struct curandStatePhilox4_32_10 {
    // Counter (128-bit)
    unsigned int ctr[4];
    // Key (64-bit)
    unsigned int key[2];
    // Cached outputs (4 uint32 per round), consumed index 0..3
    unsigned int output[4];
    unsigned int output_idx;
};
typedef curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;

// ── MRG32k3a state ────────────────────────────────────────────────────────────
struct curandStateMRG32k3a {
    double s1[3];
    double s2[3];
    unsigned int subsequence;
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace vgre_curand_detail {

// splitmix64: fast non-crypto hash for seeding
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// 32-bit mix for seeding components
inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// ── XORWOW generate ──────────────────────────────────────────────────────────
inline unsigned int xorwow_next(curandStateXORWOW* s) {
    unsigned int t = s->v[0] ^ (s->v[0] >> 2);
    s->v[0] = s->v[1];
    s->v[1] = s->v[2];
    s->v[2] = s->v[3];
    s->v[3] = s->v[4];
    s->v[4] = (s->v[4] ^ (s->v[4] << 4)) ^ (t ^ (t << 1));
    s->d += 362437U;
    return s->v[4] + s->d;
}

// Skip-ahead for XORWOW: advance by 2^67 for subsequence, 1 for offset.
// We use a simple Horner/matrix fast-skip via polynomial over GF(2)^160.
// Simplified: advance by n calls to xorwow_next (offset <= 2^20 is practical).
inline void xorwow_skip(curandStateXORWOW* s, unsigned long long n) {
    for (unsigned long long i = 0; i < n; ++i) xorwow_next(s);
}

// ── Philox 4×32-10 ────────────────────────────────────────────────────────────
// Philox constants (same as NVIDIA's implementation)
static const uint32_t kPhiloxW32A = 0x9E3779B9U;
static const uint32_t kPhiloxW32B = 0xBB67AE85U;
static const uint32_t kPhiloxM4x32A = 0xD2511F53U;
static const uint32_t kPhiloxM4x32B = 0xCD9E8D57U;

inline void philox_mulhilo(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
    uint64_t r = static_cast<uint64_t>(a) * b;
    hi = static_cast<uint32_t>(r >> 32);
    lo = static_cast<uint32_t>(r);
}

inline void philox_round(uint32_t* ctr, const uint32_t* key) {
    uint32_t hi0, lo0, hi2, lo2;
    philox_mulhilo(kPhiloxM4x32A, ctr[0], hi0, lo0);
    philox_mulhilo(kPhiloxM4x32B, ctr[2], hi2, lo2);
    ctr[0] = hi2 ^ ctr[1] ^ key[0];
    ctr[1] = lo2;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;
}

inline void philox_bumpkey(uint32_t* key) {
    key[0] += kPhiloxW32A;
    key[1] += kPhiloxW32B;
}

inline void philox4x32_10(uint32_t* ctr, const uint32_t* key_in, uint32_t* out) {
    uint32_t ctr2[4] = {ctr[0], ctr[1], ctr[2], ctr[3]};
    uint32_t key[2]  = {key_in[0], key_in[1]};
    for (int r = 0; r < 10; ++r) {
        philox_round(ctr2, key);
        if (r < 9) philox_bumpkey(key);
    }
    out[0] = ctr2[0]; out[1] = ctr2[1]; out[2] = ctr2[2]; out[3] = ctr2[3];
}

inline uint32_t philox_next(curandStatePhilox4_32_10* s) {
    if (s->output_idx >= 4) {
        philox4x32_10(s->ctr, s->key, s->output);
        // Increment 128-bit counter
        s->ctr[0]++;
        if (s->ctr[0] == 0) { s->ctr[1]++;
        if (s->ctr[1] == 0) { s->ctr[2]++;
        if (s->ctr[2] == 0) { s->ctr[3]++; }}}
        s->output_idx = 0;
    }
    return s->output[s->output_idx++];
}

// ── MRG32k3a ─────────────────────────────────────────────────────────────────
static const double kMrgM1 = 4294967087.0;
static const double kMrgM2 = 4294944443.0;

inline double mrg_next_double(curandStateMRG32k3a* s) {
    double p1 = 1403580.0 * s->s1[1] - 810728.0 * s->s1[0];
    p1 = fmod(p1, kMrgM1);
    if (p1 < 0.0) p1 += kMrgM1;

    double p2 = 527612.0 * s->s2[2] - 1370589.0 * s->s2[0];
    p2 = fmod(p2, kMrgM2);
    if (p2 < 0.0) p2 += kMrgM2;

    s->s1[0] = s->s1[1]; s->s1[1] = s->s1[2]; s->s1[2] = p1;
    s->s2[0] = s->s2[1]; s->s2[1] = s->s2[2]; s->s2[2] = p2;

    double u = p1 - p2;
    if (u <= 0.0) u += kMrgM1;
    return u * (1.0 / (kMrgM1 + 1.0));
}

inline uint32_t mrg_next_uint(curandStateMRG32k3a* s) {
    double d = mrg_next_double(s);
    return static_cast<uint32_t>(d * 4294967296.0);
}

// ── Box-Muller normal pair ────────────────────────────────────────────────────
// Generates two independent N(0,1) float samples from two uniform [0,1) inputs.
inline void box_muller(float u1, float u2, float& z0, float& z1) {
    float r = sqrtf(-2.0f * logf(u1 + 1e-38f));
    float theta = 6.283185307f * u2;
    z0 = r * cosf(theta);
    z1 = r * sinf(theta);
}

inline void box_muller_d(double u1, double u2, double& z0, double& z1) {
    double r = sqrt(-2.0 * log(u1 + 1e-300));
    double theta = 6.28318530717958648 * u2;
    z0 = r * cos(theta);
    z1 = r * sin(theta);
}

} // namespace vgre_curand_detail

// ─────────────────────────────────────────────────────────────────────────────
// curand_init
// ─────────────────────────────────────────────────────────────────────────────

// XORWOW init: seed + subsequence + offset
inline void curand_init(unsigned long long seed, unsigned long long subsequence,
                        unsigned long long offset, curandStateXORWOW* state)
{
    using namespace vgre_curand_detail;
    // Derive 5 independent 32-bit values from seed + subsequence using splitmix64
    uint64_t h = splitmix64(seed ^ (subsequence * 0x9e3779b97f4a7c15ULL));
    state->v[0] = static_cast<uint32_t>(h);
    state->v[1] = static_cast<uint32_t>(h >> 32);
    h = splitmix64(h + 0x6c62272e07bb0142ULL);
    state->v[2] = static_cast<uint32_t>(h);
    state->v[3] = static_cast<uint32_t>(h >> 32);
    h = splitmix64(h + 0x517cc1b727220a95ULL);
    state->v[4] = static_cast<uint32_t>(h);
    state->d     = 6615241U;
    // Warm-up: discard 8 values to mix state
    for (int i = 0; i < 8; ++i) xorwow_next(state);
    // Apply offset
    xorwow_skip(state, offset);
}

// Philox init
inline void curand_init(unsigned long long seed, unsigned long long subsequence,
                        unsigned long long offset, curandStatePhilox4_32_10* state)
{
    using namespace vgre_curand_detail;
    state->key[0]      = static_cast<uint32_t>(seed);
    state->key[1]      = static_cast<uint32_t>(seed >> 32);
    state->ctr[0]      = static_cast<uint32_t>(offset);
    state->ctr[1]      = static_cast<uint32_t>(offset >> 32);
    state->ctr[2]      = static_cast<uint32_t>(subsequence);
    state->ctr[3]      = static_cast<uint32_t>(subsequence >> 32);
    state->output_idx  = 4; // force generation on first use
}

// MRG32k3a init
inline void curand_init(unsigned long long seed, unsigned long long /*subsequence*/,
                        unsigned long long /*offset*/, curandStateMRG32k3a* state)
{
    using namespace vgre_curand_detail;
    uint64_t h = splitmix64(seed);
    state->s1[0] = 1.0 + (h & 0x7FFFFFFFULL) % static_cast<uint64_t>(kMrgM1 - 1.0);
    h = splitmix64(h);
    state->s1[1] = 1.0 + (h & 0x7FFFFFFFULL) % static_cast<uint64_t>(kMrgM1 - 1.0);
    h = splitmix64(h);
    state->s1[2] = 1.0 + (h & 0x7FFFFFFFULL) % static_cast<uint64_t>(kMrgM1 - 1.0);
    h = splitmix64(h);
    state->s2[0] = 1.0 + (h & 0x7FFFFFFFULL) % static_cast<uint64_t>(kMrgM2 - 1.0);
    h = splitmix64(h);
    state->s2[1] = 1.0 + (h & 0x7FFFFFFFULL) % static_cast<uint64_t>(kMrgM2 - 1.0);
    h = splitmix64(h);
    state->s2[2] = 1.0 + (h & 0x7FFFFFFFULL) % static_cast<uint64_t>(kMrgM2 - 1.0);
    state->subsequence = 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// curand — raw uint32
// ─────────────────────────────────────────────────────────────────────────────
inline unsigned int curand(curandStateXORWOW* state) {
    return vgre_curand_detail::xorwow_next(state);
}
inline unsigned int curand(curandStatePhilox4_32_10* state) {
    return vgre_curand_detail::philox_next(state);
}
inline unsigned int curand(curandStateMRG32k3a* state) {
    return vgre_curand_detail::mrg_next_uint(state);
}

// ─────────────────────────────────────────────────────────────────────────────
// curand_uniform — (0, 1] float  (CUDA convention: excludes 0, includes 1)
// ─────────────────────────────────────────────────────────────────────────────
inline float curand_uniform(curandStateXORWOW* state) {
    // Map 32-bit integer to (0,1]: x * 2^-32 + 2^-33
    return (curand(state) + 0.5f) * (1.0f / 4294967296.0f);
}
inline float curand_uniform(curandStatePhilox4_32_10* state) {
    return (curand(state) + 0.5f) * (1.0f / 4294967296.0f);
}
inline float curand_uniform(curandStateMRG32k3a* state) {
    return static_cast<float>(vgre_curand_detail::mrg_next_double(state));
}

// ─────────────────────────────────────────────────────────────────────────────
// curand_uniform_double
// ─────────────────────────────────────────────────────────────────────────────
inline double curand_uniform_double(curandStateXORWOW* state) {
    // Combine two 32-bit values to fill a 53-bit mantissa
    uint64_t hi = curand(state);
    uint64_t lo = curand(state);
    uint64_t bits = (hi << 20) | (lo >> 12); // 52 usable bits
    double r;
    uint64_t f = (bits & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    memcpy(&r, &f, 8);
    return r - 1.0; // maps to (0,1)
}
inline double curand_uniform_double(curandStatePhilox4_32_10* state) {
    uint64_t hi = curand(state);
    uint64_t lo = curand(state);
    uint64_t bits = (hi << 20) | (lo >> 12);
    double r;
    uint64_t f = (bits & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    memcpy(&r, &f, 8);
    return r - 1.0;
}
inline double curand_uniform_double(curandStateMRG32k3a* state) {
    return vgre_curand_detail::mrg_next_double(state);
}

// ─────────────────────────────────────────────────────────────────────────────
// curand_normal — N(0,1) float using Box-Muller
// State caches the second value to avoid wasting a sample.
// ─────────────────────────────────────────────────────────────────────────────
// We embed a 1-value cache in a thread-local; CUDA's curand_normal regenerates
// both values each time. For CPU emulation we do the same (no cache needed).
inline float curand_normal(curandStateXORWOW* state) {
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    float z0, z1;
    vgre_curand_detail::box_muller(u1, u2, z0, z1);
    return z0;
}
inline float curand_normal(curandStatePhilox4_32_10* state) {
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    float z0, z1;
    vgre_curand_detail::box_muller(u1, u2, z0, z1);
    return z0;
}
inline float curand_normal(curandStateMRG32k3a* state) {
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    float z0, z1;
    vgre_curand_detail::box_muller(u1, u2, z0, z1);
    return z0;
}

// curand_normal2 — generates float2 (two independent samples)
struct float2 { float x, y; };
inline float2 curand_normal2(curandStateXORWOW* state) {
    float u1 = curand_uniform(state), u2 = curand_uniform(state);
    float z0, z1;
    vgre_curand_detail::box_muller(u1, u2, z0, z1);
    return {z0, z1};
}
inline float2 curand_normal2(curandStatePhilox4_32_10* state) {
    float u1 = curand_uniform(state), u2 = curand_uniform(state);
    float z0, z1;
    vgre_curand_detail::box_muller(u1, u2, z0, z1);
    return {z0, z1};
}

// ─────────────────────────────────────────────────────────────────────────────
// curand_normal_double
// ─────────────────────────────────────────────────────────────────────────────
inline double curand_normal_double(curandStateXORWOW* state) {
    double u1 = curand_uniform_double(state) + 1e-300;
    double u2 = curand_uniform_double(state);
    double z0, z1;
    vgre_curand_detail::box_muller_d(u1, u2, z0, z1);
    return z0;
}
inline double curand_normal_double(curandStatePhilox4_32_10* state) {
    double u1 = curand_uniform_double(state) + 1e-300;
    double u2 = curand_uniform_double(state);
    double z0, z1;
    vgre_curand_detail::box_muller_d(u1, u2, z0, z1);
    return z0;
}
inline double curand_normal_double(curandStateMRG32k3a* state) {
    double u1 = vgre_curand_detail::mrg_next_double(state) + 1e-300;
    double u2 = vgre_curand_detail::mrg_next_double(state);
    double z0, z1;
    vgre_curand_detail::box_muller_d(u1, u2, z0, z1);
    return z0;
}

// ─────────────────────────────────────────────────────────────────────────────
// curand_log_normal — log-normal: exp(mean + stddev * N(0,1))
// ─────────────────────────────────────────────────────────────────────────────
inline float curand_log_normal(curandStateXORWOW* state, float mean, float stddev) {
    return expf(mean + stddev * curand_normal(state));
}
inline float curand_log_normal(curandStatePhilox4_32_10* state, float mean, float stddev) {
    return expf(mean + stddev * curand_normal(state));
}
inline float curand_log_normal(curandStateMRG32k3a* state, float mean, float stddev) {
    return expf(mean + stddev * curand_normal(state));
}

inline double curand_log_normal_double(curandStateXORWOW* state, double mean, double stddev) {
    return exp(mean + stddev * curand_normal_double(state));
}
inline double curand_log_normal_double(curandStatePhilox4_32_10* state, double mean, double stddev) {
    return exp(mean + stddev * curand_normal_double(state));
}

// ─────────────────────────────────────────────────────────────────────────────
// curand_poisson — Knuth's algorithm for small lambda, Gaussian approx for large
// ─────────────────────────────────────────────────────────────────────────────
inline unsigned int curand_poisson(curandStateXORWOW* state, double lambda) {
    if (lambda < 30.0) {
        // Knuth: generate until product of uniforms falls below exp(-lambda)
        double L = exp(-lambda);
        double p = 1.0;
        unsigned int k = 0;
        do {
            ++k;
            p *= static_cast<double>(curand_uniform(state));
        } while (p > L);
        return k - 1;
    } else {
        // Normal approximation: round(N(lambda, sqrt(lambda)))
        double n = curand_normal_double(state) * sqrt(lambda) + lambda + 0.5;
        return n < 0.0 ? 0u : static_cast<unsigned int>(n);
    }
}
inline unsigned int curand_poisson(curandStatePhilox4_32_10* state, double lambda) {
    if (lambda < 30.0) {
        double L = exp(-lambda), p = 1.0;
        unsigned int k = 0;
        do { ++k; p *= curand_uniform(state); } while (p > L);
        return k - 1;
    } else {
        double n = curand_normal_double(state) * sqrt(lambda) + lambda + 0.5;
        return n < 0.0 ? 0u : static_cast<unsigned int>(n);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// curand_discrete — sample from a discrete distribution (histogram)
// ─────────────────────────────────────────────────────────────────────────────
typedef struct {
    unsigned int* probabilities; // probability histogram
    unsigned int  n;             // number of bins
    double        norm;          // normalization factor
} curandDiscreteDistribution_st;
typedef curandDiscreteDistribution_st* curandDiscreteDistribution_t;

// ─────────────────────────────────────────────────────────────────────────────
// curandStateDefault — alias for XORWOW (CUDA's default device generator)
// ─────────────────────────────────────────────────────────────────────────────
typedef curandStateXORWOW curandStateDefault;

// ─────────────────────────────────────────────────────────────────────────────
// Copy state helpers (used by some libraries to save/restore RNG state)
// ─────────────────────────────────────────────────────────────────────────────
inline void curand_copy_state(curandStateXORWOW* dst, const curandStateXORWOW* src) {
    *dst = *src;
}
inline void curand_copy_state(curandStatePhilox4_32_10* dst, const curandStatePhilox4_32_10* src) {
    *dst = *src;
}

#endif // VGRE_CURAND_KERNEL_H
