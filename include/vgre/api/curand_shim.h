#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ── Status codes ─────────────────────────────────────────────────────────────
typedef enum {
    CURAND_STATUS_SUCCESS = 0,
    CURAND_STATUS_VERSION_MISMATCH = 100,
    CURAND_STATUS_NOT_INITIALIZED = 101,
    CURAND_STATUS_ALLOCATION_FAILED = 102,
    CURAND_STATUS_TYPE_ERROR = 103,
    CURAND_STATUS_OUT_OF_RANGE = 104,
    CURAND_STATUS_INVALID_VALUE = 107,
    CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
    CURAND_STATUS_LAUNCH_FAILURE = 201,
    CURAND_STATUS_PREEXISTING_FAILURE = 202,
    CURAND_STATUS_INITIALIZATION_FAILED = 203,
    CURAND_STATUS_ARCH_MISMATCH = 204,
    CURAND_STATUS_INTERNAL_ERROR = 999
} curandStatus_t;

// ── Generator types ────────────────────────────────────────────────────────
typedef enum {
    CURAND_RNG_TEST = 0,
    CURAND_RNG_PSEUDO_DEFAULT = 1,
    CURAND_RNG_PSEUDO_XORWOW = 2,
    CURAND_RNG_PSEUDO_MRG32K3A = 3,
    CURAND_RNG_PSEUDO_MTGP32 = 4,
    CURAND_RNG_PSEUDO_MT19937 = 5,
    CURAND_RNG_PSEUDO_PHILOX4_32_10 = 6,
    CURAND_RNG_QUASI_DEFAULT = 7,
    CURAND_RNG_QUASI_SOBOL32 = 8,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 9,
    CURAND_RNG_QUASI_SOBOL64 = 10,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 11
} curandRngType_t;

// ── Ordering ─────────────────────────────────────────────────────────────────
typedef enum {
    CURAND_ORDERING_PSEUDO_BEST = 0,
    CURAND_ORDERING_PSEUDO_DEFAULT = 1,
    CURAND_ORDERING_PSEUDO_SEEDED = 2,
    CURAND_ORDERING_PSEUDO_LEGACY = 3,
    CURAND_ORDERING_QUASI_DEFAULT = 4
} curandOrdering_t;

// ── Direction vector sets (Sobol) ────────────────────────────────────────────
typedef enum {
    CURAND_DIRECTION_VECTORS_32_JOE_KUO6 = 0,
    CURAND_DIRECTION_VECTORS_32_JOE_KUO6_SCRAMBLED = 1,
    CURAND_DIRECTION_VECTORS_64_JOE_KUO6 = 2,
    CURAND_DIRECTION_VECTORS_64_JOE_KUO6_SCRAMBLED = 3
} curandDirectionVectorSet_t;

typedef unsigned int curandDirectionVectors32_t[32];
typedef unsigned long long curandDirectionVectors64_t[64];

// ── Opaque handle ────────────────────────────────────────────────────────────
struct curandGenerator_st;
typedef struct curandGenerator_st *curandGenerator_t;

// ── Lifecycle ────────────────────────────────────────────────────────────────
curandStatus_t curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type);
curandStatus_t curandDestroyGenerator(curandGenerator_t generator);

// ── Seeding / offset ───────────────────────────────────────────────────────
curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed);
curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset);
curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order);
curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator, unsigned int num_dimensions);

// ── Generation ─────────────────────────────────────────────────────────────────
curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int *outputPtr, size_t num);
curandStatus_t curandGenerateLongLong(curandGenerator_t generator, unsigned long long *outputPtr, size_t num);
curandStatus_t curandGenerateUniform(curandGenerator_t generator, float *outputPtr, size_t num);
curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator, double *outputPtr, size_t num);
curandStatus_t curandGenerateNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev);
curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev);
curandStatus_t curandGenerateLogNormal(curandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev);
curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev);

// ── Direction vectors (Sobol) ──────────────────────────────────────────────
curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t *vectors, curandDirectionVectorSet_t set);
curandStatus_t curandGetDirectionVectors64(curandDirectionVectors64_t *vectors, curandDirectionVectorSet_t set);
curandStatus_t curandGetScrambleConstants32(unsigned int **constants);
curandStatus_t curandGetScrambleConstants64(unsigned long long **constants);

#ifdef __cplusplus
} // extern "C"
#endif
