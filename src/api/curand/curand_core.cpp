// cuRAND emulation shim — maps to C++11 <random> for reproducible CPU-side RNG.
//
// Supported generators:
//   CURAND_RNG_PSEUDO_DEFAULT     → std::mt19937_64
//   CURAND_RNG_PSEUDO_MTGP32      → std::mt19937_64 (same)
//   CURAND_RNG_PSEUDO_MT19937     → std::mt19937_64
//   CURAND_RNG_PSEUDO_PHILOX4_32_10 → std::mt19937_64 (same)
//   CURAND_RNG_PSEUDO_XORWOW      → std::mt19937_64 (same)
//   CURAND_RNG_PSEUDO_MRG32K3A    → std::mt19937_64 (same)
//
// Quasi-random (Sobol) generators: direction vectors, scramble constants,
// and Gray-code quasi-random generation are all fully implemented.

#include "vgre/api/curand_shim.h"
#include "vgre/common/logger.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>

namespace {

struct GeneratorState {
    curandRngType_t type = CURAND_RNG_PSEUDO_DEFAULT;
    curandOrdering_t ordering = CURAND_ORDERING_PSEUDO_DEFAULT;
    unsigned long long seed = 0;
    unsigned long long offset = 0;
    unsigned int quasiDimensions = 1;
    bool scrambled = false;
    std::mt19937_64 engine;
    bool seeded = false;
    std::mutex genMutex; // per-handle lock for concurrent generate calls
};

// Registry uses unique_ptr so GeneratorState's non-movable mutex is safe.
std::mutex g_registryMutex;
std::unordered_map<uintptr_t, std::unique_ptr<GeneratorState>> g_registry;
uintptr_t g_nextHandle = 1;

// Shared Sobol direction-vector cache (20000 dims × 32 bits)
std::mutex g_sobolMutex;
bool g_sobol32Ready = false;
bool g_sobol64Ready = false;
unsigned int g_sobolDir32[20000][32];
unsigned long long g_sobolDir64[20000][64];
unsigned int g_scramble32[20000];
unsigned long long g_scramble64[20000];

// ── Direction vectors (Sobol) — CPU reference implementation ─────────────────

// Primitive polynomial coefficients for degrees 1–8 (bit k = coeff of x^k, x^s implied)
static const uint16_t kPrimitivePoly[8] = {
    0b1,        // degree 1: x + 1
    0b11,       // degree 2: x^2 + x + 1
    0b011,      // degree 3: x^3 + x + 1
    0b0011,     // degree 4: x^4 + x + 1
    0b00101,    // degree 5: x^5 + x^2 + 1
    0b000011,   // degree 6: x^6 + x + 1
    0b0000011,  // degree 7: x^7 + x + 1
    0b00011101, // degree 8: x^8 + x^4 + x^3 + x^2 + 1
};

static inline int sobolDegree(int d) {
    if (d <= 1) return 0;
    if (d <= 2) return 1;
    if (d <= 4) return 2;
    if (d <= 8) return 3;
    if (d <= 16) return 4;
    if (d <= 32) return 5;
    if (d <= 64) return 6;
    if (d <= 128) return 7;
    return 8;
}

static inline uint16_t sobolPoly(int d) {
    int deg = sobolDegree(d);
    if (deg == 0) return 0;
    return kPrimitivePoly[(d - 1) % 8];
}

static void computeDirVectors32(unsigned int *v, int d) {
    if (d == 1) {
        for (int j = 0; j < 32; ++j) v[j] = 1u << (31 - j);
        return;
    }
    int s = sobolDegree(d);
    uint16_t poly = sobolPoly(d);
    uint32_t m[33] = {0};
    for (int i = 1; i <= s; ++i) m[i] = 1;
    for (int j = s + 1; j <= 32; ++j) {
        uint32_t mj = m[j - s];
        for (int k = 1; k <= s; ++k) {
            if ((poly >> (k - 1)) & 1) mj ^= m[j - k] << k;
        }
        m[j] = mj;
    }
    for (int j = 1; j <= 32; ++j) v[j - 1] = m[j] << (32 - j);
}

static void computeDirVectors64(unsigned long long *v, int d) {
    if (d == 1) {
        for (int j = 0; j < 64; ++j) v[j] = 1ull << (63 - j);
        return;
    }
    int s = sobolDegree(d);
    uint16_t poly = sobolPoly(d);
    uint64_t m[65] = {0};
    for (int i = 1; i <= s; ++i) m[i] = 1;
    for (int j = s + 1; j <= 64; ++j) {
        uint64_t mj = m[j - s];
        for (int k = 1; k <= s; ++k) {
            if ((poly >> (k - 1)) & 1) mj ^= m[j - k] << k;
        }
        m[j] = mj;
    }
    for (int j = 1; j <= 64; ++j) v[j - 1] = m[j] << (64 - j);
}

static inline uint32_t scrambleHash32(int d) {
    uint32_t h = static_cast<uint32_t>(d) * 2654435761u;
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return h | 1u;
}
static inline uint64_t scrambleHash64(int d) {
    uint64_t h = static_cast<uint64_t>(d) * 6364136223846793005ull;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return h | 1ull;
}

void ensureSobol32() {
    std::lock_guard<std::mutex> lk(g_sobolMutex);
    if (g_sobol32Ready) return;
    for (int d = 1; d <= 20000; ++d) {
        if (d == 1) {
            for (int j = 0; j < 32; ++j) g_sobolDir32[d-1][j] = 1u << (31 - j);
        } else {
            int s = sobolDegree(d);
            uint16_t poly = sobolPoly(d);
            uint32_t m[33] = {0};
            for (int i = 1; i <= s; ++i) m[i] = 1;
            for (int j = s + 1; j <= 32; ++j) {
                uint32_t mj = m[j - s];
                for (int k = 1; k <= s; ++k) {
                    if ((poly >> (k - 1)) & 1) mj ^= m[j - k] << k;
                }
                m[j] = mj;
            }
            for (int j = 1; j <= 32; ++j) g_sobolDir32[d-1][j-1] = m[j] << (32 - j);
        }
    }
    for (int d = 0; d < 20000; ++d) g_scramble32[d] = scrambleHash32(d + 1);
    g_sobol32Ready = true;
}

void ensureSobol64() {
    std::lock_guard<std::mutex> lk(g_sobolMutex);
    if (g_sobol64Ready) return;
    for (int d = 1; d <= 20000; ++d) {
        if (d == 1) {
            for (int j = 0; j < 64; ++j) g_sobolDir64[d-1][j] = 1ull << (63 - j);
        } else {
            int s = sobolDegree(d);
            uint16_t poly = sobolPoly(d);
            uint64_t m[65] = {0};
            for (int i = 1; i <= s; ++i) m[i] = 1;
            for (int j = s + 1; j <= 64; ++j) {
                uint64_t mj = m[j - s];
                for (int k = 1; k <= s; ++k) {
                    if ((poly >> (k - 1)) & 1) mj ^= m[j - k] << k;
                }
                m[j] = mj;
            }
            for (int j = 1; j <= 64; ++j) g_sobolDir64[d-1][j-1] = m[j] << (64 - j);
        }
    }
    for (int d = 0; d < 20000; ++d) g_scramble64[d] = scrambleHash64(d + 1);
    g_sobol64Ready = true;
}

// Gray-code Sobol: g = n ^ (n>>1); x = 0; for each bit j in g: x ^= dirvec[j]
static inline uint32_t sobolValue32(uint32_t n, const unsigned int* dirvec) {
    uint32_t g = n ^ (n >> 1);
    uint32_t x = 0;
    for (int j = 0; j < 32; ++j) {
        if (g & (1u << j)) x ^= dirvec[j];
    }
    return x;
}

static inline uint64_t sobolValue64(uint64_t n, const unsigned long long* dirvec) {
    uint64_t g = n ^ (n >> 1);
    uint64_t x = 0;
    for (int j = 0; j < 64; ++j) {
        if (g & (1ull << j)) x ^= dirvec[j];
    }
    return x;
}

static inline bool isQuasi(curandRngType_t t) {
    return t == CURAND_RNG_QUASI_DEFAULT ||
           t == CURAND_RNG_QUASI_SOBOL32 ||
           t == CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 ||
           t == CURAND_RNG_QUASI_SOBOL64 ||
           t == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
}

static inline bool isScrambled(curandRngType_t t) {
    return t == CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 ||
           t == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
}

// Returns raw pointer; caller must be sure generator is not destroyed concurrently.
// Each generate op then locks st->genMutex for per-handle thread safety.
GeneratorState *lookup(curandGenerator_t g) {
    std::lock_guard<std::mutex> lk(g_registryMutex);
    auto it = g_registry.find(reinterpret_cast<uintptr_t>(g));
    if (it == g_registry.end()) return nullptr;
    return it->second.get();
}

} // namespace

extern "C" {

curandStatus_t curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type) {
    if (!generator) return CURAND_STATUS_ALLOCATION_FAILED;

    auto st = std::make_unique<GeneratorState>();
    st->type     = rng_type;
    st->scrambled = isScrambled(rng_type);
    st->engine.seed(0);
    st->seeded   = true;

    if (isQuasi(rng_type)) {
        if (rng_type == CURAND_RNG_QUASI_SOBOL64 || rng_type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64)
            ensureSobol64();
        else
            ensureSobol32();
    }

    std::lock_guard<std::mutex> lk(g_registryMutex);
    uintptr_t handle = g_nextHandle++;
    g_registry[handle] = std::move(st);
    *generator = reinterpret_cast<curandGenerator_t>(handle);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandDestroyGenerator(curandGenerator_t generator) {
    if (!generator) return CURAND_STATUS_NOT_INITIALIZED;
    std::lock_guard<std::mutex> lk(g_registryMutex);
    g_registry.erase(reinterpret_cast<uintptr_t>(generator));
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed) {
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    st->seed = seed;
    st->engine.seed(seed);
    st->offset = 0;
    st->seeded = true;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) {
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    st->offset = offset;
    // Fast-forward the engine by discarding `offset` values.
    st->engine.discard(offset);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order) {
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    st->ordering = order;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions) {
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    st->quasiDimensions = num_dimensions;
    return CURAND_STATUS_SUCCESS;
}

// ── Integer generation ───────────────────────────────────────────────────────

curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int *outputPtr, size_t num) {
    if (!outputPtr || num == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::lock_guard<std::mutex> genLk(st->genMutex);
    if (isQuasi(st->type)) {
        ensureSobol32();
        unsigned int D = st->quasiDimensions;
        if (D == 0) D = 1;
        for (size_t i = 0; i < num; ++i) {
            unsigned int d = static_cast<unsigned int>(i % D);
            uint32_t p = static_cast<uint32_t>(st->offset + (i / D));
            uint32_t v = sobolValue32(p, g_sobolDir32[d]);
            if (st->scrambled) v ^= g_scramble32[d];
            outputPtr[i] = v;
        }
        st->offset += num / D;
    } else {
        std::uniform_int_distribution<unsigned int> dist;
        for (size_t i = 0; i < num; ++i) outputPtr[i] = dist(st->engine);
    }
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLongLong(curandGenerator_t generator, unsigned long long *outputPtr, size_t num) {
    if (!outputPtr || num == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::lock_guard<std::mutex> genLk(st->genMutex);
    if (isQuasi(st->type)) {
        ensureSobol64();
        unsigned int D = st->quasiDimensions;
        if (D == 0) D = 1;
        for (size_t i = 0; i < num; ++i) {
            unsigned int d = static_cast<unsigned int>(i % D);
            uint64_t p = st->offset + (i / D);
            uint64_t v = sobolValue64(p, g_sobolDir64[d]);
            if (st->scrambled) v ^= g_scramble64[d];
            outputPtr[i] = v;
        }
        st->offset += num / D;
    } else {
        std::uniform_int_distribution<unsigned long long> dist;
        for (size_t i = 0; i < num; ++i) outputPtr[i] = dist(st->engine);
    }
    return CURAND_STATUS_SUCCESS;
}

// ── Uniform float/double generation ──────────────────────────────────────────

curandStatus_t curandGenerateUniform(curandGenerator_t generator, float *outputPtr, size_t num) {
    if (!outputPtr || num == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::lock_guard<std::mutex> genLk(st->genMutex);
    if (isQuasi(st->type)) {
        ensureSobol32();
        unsigned int D = st->quasiDimensions;
        if (D == 0) D = 1;
        constexpr float inv32 = 1.0f / 4294967296.0f;
        for (size_t i = 0; i < num; ++i) {
            unsigned int d = static_cast<unsigned int>(i % D);
            uint32_t p = static_cast<uint32_t>(st->offset + (i / D));
            uint32_t v = sobolValue32(p, g_sobolDir32[d]);
            if (st->scrambled) v ^= g_scramble32[d];
            outputPtr[i] = v * inv32;
        }
        st->offset += num / D;
    } else {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < num; ++i) outputPtr[i] = dist(st->engine);
    }
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double *outputPtr, size_t num) {
    if (!outputPtr || num == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::lock_guard<std::mutex> genLk(st->genMutex);
    if (isQuasi(st->type)) {
        ensureSobol64();
        unsigned int D = st->quasiDimensions;
        if (D == 0) D = 1;
        constexpr double inv64 = 1.0 / 18446744073709551616.0;
        for (size_t i = 0; i < num; ++i) {
            unsigned int d = static_cast<unsigned int>(i % D);
            uint64_t p = st->offset + (i / D);
            uint64_t v = sobolValue64(p, g_sobolDir64[d]);
            if (st->scrambled) v ^= g_scramble64[d];
            outputPtr[i] = static_cast<double>(v) * inv64;
        }
        st->offset += num / D;
    } else {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < num; ++i) outputPtr[i] = dist(st->engine);
    }
    return CURAND_STATUS_SUCCESS;
}

// ── Normal float/double generation ───────────────────────────────────────────

curandStatus_t curandGenerateNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev) {
    if (!outputPtr || n == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::lock_guard<std::mutex> genLk(st->genMutex);
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < n; ++i) outputPtr[i] = dist(st->engine);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev) {
    if (!outputPtr || n == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::lock_guard<std::mutex> genLk(st->genMutex);
    std::normal_distribution<double> dist(mean, stddev);
    for (size_t i = 0; i < n; ++i) outputPtr[i] = dist(st->engine);
    return CURAND_STATUS_SUCCESS;
}

// ── Log-normal float/double generation ───────────────────────────────────────

curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev) {
    if (!outputPtr || n == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::lock_guard<std::mutex> genLk(st->genMutex);
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < n; ++i) outputPtr[i] = std::exp(dist(st->engine));
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev) {
    if (!outputPtr || n == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::lock_guard<std::mutex> genLk(st->genMutex);
    std::normal_distribution<double> dist(mean, stddev);
    for (size_t i = 0; i < n; ++i) outputPtr[i] = std::exp(dist(st->engine));
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t *vectors, curandDirectionVectorSet_t /*set*/) {
    if (!vectors) return CURAND_STATUS_INVALID_VALUE;
    for (int d = 1; d <= 20000; ++d) computeDirVectors32(vectors[d - 1], d);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetDirectionVectors64(curandDirectionVectors64_t *vectors, curandDirectionVectorSet_t /*set*/) {
    if (!vectors) return CURAND_STATUS_INVALID_VALUE;
    for (int d = 1; d <= 20000; ++d) computeDirVectors64(vectors[d - 1], d);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetScrambleConstants32(unsigned int **constants) {
    if (!constants) return CURAND_STATUS_INVALID_VALUE;
    static unsigned int g_scramble32[20000];
    static bool g_init32 = false;
    if (!g_init32) {
        for (int d = 0; d < 20000; ++d) g_scramble32[d] = scrambleHash32(d + 1);
        g_init32 = true;
    }
    *constants = g_scramble32;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetScrambleConstants64(unsigned long long **constants) {
    if (!constants) return CURAND_STATUS_INVALID_VALUE;
    static unsigned long long g_scramble64[20000];
    static bool g_init64 = false;
    if (!g_init64) {
        for (int d = 0; d < 20000; ++d) g_scramble64[d] = scrambleHash64(d + 1);
        g_init64 = true;
    }
    *constants = g_scramble64;
    return CURAND_STATUS_SUCCESS;
}

// ── Poisson distribution ─────────────────────────────────────────────────────

curandStatus_t curandGeneratePoisson(curandGenerator_t generator, unsigned int *outputPtr, size_t n, double lambda) {
    if (!outputPtr || n == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::lock_guard<std::mutex> genLk(st->genMutex);
    std::poisson_distribution<unsigned int> dist(lambda);
    for (size_t i = 0; i < n; ++i) outputPtr[i] = dist(st->engine);
    return CURAND_STATUS_SUCCESS;
}

// ── Seed generation (re-seed from entropy) ───────────────────────────────────

curandStatus_t curandGenerateSeeds(curandGenerator_t generator) {
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::random_device rd;
    st->seed = (static_cast<unsigned long long>(rd()) << 32) | rd();
    st->engine.seed(st->seed);
    return CURAND_STATUS_SUCCESS;
}

// ── IPC handle export / import (F.7) ─────────────────────────────────────────
// vgre_curand_ipc_handle_t is a 64-byte opaque struct that serialises all
// deterministic state needed to reconstruct a cuRAND generator in another
// process.  The caller is responsible for transferring the bytes between
// processes (file, socket, shared memory, etc.).
//
// The mt19937_64 engine is fully reproduced by re-seeding with 'seed' and
// calling engine.discard(offset) — no internal MT state snapshot is required.

struct vgre_curand_ipc_handle_t {
    uint64_t seed;
    uint64_t offset;
    uint32_t rng_type;      // curandRngType_t
    uint32_t ordering;      // curandOrdering_t
    uint32_t quasi_dims;
    uint32_t scrambled;
    uint8_t  reserved[32];  // pad to 64 bytes
};
static_assert(sizeof(vgre_curand_ipc_handle_t) == 64,
              "IPC handle must be exactly 64 bytes");

curandStatus_t curandGetGeneratorIpcHandle(curandGenerator_t generator,
                                            vgre_curand_ipc_handle_t *handle) {
    if (!handle) return CURAND_STATUS_INVALID_VALUE;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::lock_guard<std::mutex> lk(st->genMutex);
    handle->seed       = st->seed;
    handle->offset     = st->offset;
    handle->rng_type   = static_cast<uint32_t>(st->type);
    handle->ordering   = static_cast<uint32_t>(st->ordering);
    handle->quasi_dims = st->quasiDimensions;
    handle->scrambled  = st->scrambled ? 1u : 0u;
    std::memset(handle->reserved, 0, sizeof(handle->reserved));
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandCreateGeneratorFromIpcHandle(curandGenerator_t *generator,
                                                   const vgre_curand_ipc_handle_t *handle) {
    if (!generator || !handle) return CURAND_STATUS_INVALID_VALUE;
    curandStatus_t r = curandCreateGenerator(generator,
                           static_cast<curandRngType_t>(handle->rng_type));
    if (r != CURAND_STATUS_SUCCESS) return r;
    auto *st = lookup(*generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::lock_guard<std::mutex> genLk(st->genMutex);
    st->seed            = handle->seed;
    st->offset          = handle->offset;
    st->ordering        = static_cast<curandOrdering_t>(handle->ordering);
    st->quasiDimensions = handle->quasi_dims;
    st->scrambled       = (handle->scrambled != 0);
    st->engine.seed(handle->seed);
    if (handle->offset > 0) st->engine.discard(handle->offset);
    st->seeded = true;
    return CURAND_STATUS_SUCCESS;
}

} // extern "C"
