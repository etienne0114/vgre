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
// Quasi-random (Sobol) is not fully implemented; seeding is accepted but
// generation falls back to std::mt19937_64 for simplicity.

#include "vgre/api/curand_shim.h"
#include "vgre/common/logger.h"

#include <cmath>
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

    // C++11 engine + distributions
    std::mt19937_64 engine;
    bool seeded = false;
};

// Global registry: generator handle → state
std::mutex g_registryMutex;
std::unordered_map<uintptr_t, GeneratorState> g_registry;
uintptr_t g_nextHandle = 1;

GeneratorState *lookup(curandGenerator_t g) {
    std::lock_guard<std::mutex> lk(g_registryMutex);
    auto it = g_registry.find(reinterpret_cast<uintptr_t>(g));
    if (it == g_registry.end()) return nullptr;
    return &it->second;
}

} // namespace

extern "C" {

curandStatus_t curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type) {
    if (!generator) return CURAND_STATUS_ALLOCATION_FAILED;

    std::lock_guard<std::mutex> lk(g_registryMutex);
    uintptr_t handle = g_nextHandle++;
    GeneratorState &st = g_registry[handle];
    st.type = rng_type;
    st.engine.seed(0);
    st.seeded = true;
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
    std::uniform_int_distribution<unsigned int> dist;
    for (size_t i = 0; i < num; ++i) outputPtr[i] = dist(st->engine);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLongLong(curandGenerator_t generator, unsigned long long *outputPtr, size_t num) {
    if (!outputPtr || num == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::uniform_int_distribution<unsigned long long> dist;
    for (size_t i = 0; i < num; ++i) outputPtr[i] = dist(st->engine);
    return CURAND_STATUS_SUCCESS;
}

// ── Uniform float/double generation ──────────────────────────────────────────

curandStatus_t curandGenerateUniform(curandGenerator_t generator, float *outputPtr, size_t num) {
    if (!outputPtr || num == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < num; ++i) outputPtr[i] = dist(st->engine);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double *outputPtr, size_t num) {
    if (!outputPtr || num == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < num; ++i) outputPtr[i] = dist(st->engine);
    return CURAND_STATUS_SUCCESS;
}

// ── Normal float/double generation ───────────────────────────────────────────

curandStatus_t curandGenerateNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev) {
    if (!outputPtr || n == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < n; ++i) outputPtr[i] = dist(st->engine);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev) {
    if (!outputPtr || n == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::normal_distribution<double> dist(mean, stddev);
    for (size_t i = 0; i < n; ++i) outputPtr[i] = dist(st->engine);
    return CURAND_STATUS_SUCCESS;
}

// ── Log-normal float/double generation ───────────────────────────────────────

curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev) {
    if (!outputPtr || n == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < n; ++i) outputPtr[i] = std::exp(dist(st->engine));
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev) {
    if (!outputPtr || n == 0) return CURAND_STATUS_SUCCESS;
    auto *st = lookup(generator);
    if (!st) return CURAND_STATUS_NOT_INITIALIZED;
    std::normal_distribution<double> dist(mean, stddev);
    for (size_t i = 0; i < n; ++i) outputPtr[i] = std::exp(dist(st->engine));
    return CURAND_STATUS_SUCCESS;
}

// ── Direction vectors (Sobol) — not implemented, return error ─────────────────

curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t *vectors, curandDirectionVectorSet_t set) {
    (void)vectors; (void)set;
    return CURAND_STATUS_DOUBLE_PRECISION_REQUIRED; // placeholder error
}

curandStatus_t curandGetDirectionVectors64(curandDirectionVectors64_t *vectors, curandDirectionVectorSet_t set) {
    (void)vectors; (void)set;
    return CURAND_STATUS_DOUBLE_PRECISION_REQUIRED;
}

curandStatus_t curandGetScrambleConstants32(unsigned int **constants) {
    (void)constants;
    return CURAND_STATUS_DOUBLE_PRECISION_REQUIRED;
}

curandStatus_t curandGetScrambleConstants64(unsigned long long **constants) {
    (void)constants;
    return CURAND_STATUS_DOUBLE_PRECISION_REQUIRED;
}

} // extern "C"
