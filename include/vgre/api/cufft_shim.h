#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ── Result codes ─────────────────────────────────────────────────────────────
typedef enum {
    CUFFT_SUCCESS = 0,
    CUFFT_INVALID_PLAN = 1,
    CUFFT_ALLOC_FAILED = 2,
    CUFFT_INVALID_TYPE = 3,
    CUFFT_INVALID_VALUE = 4,
    CUFFT_INTERNAL_ERROR = 5,
    CUFFT_EXEC_FAILED = 6,
    CUFFT_SETUP_FAILED = 7,
    CUFFT_INVALID_SIZE = 8,
    CUFFT_UNALIGNED_DATA = 9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10,
    CUFFT_INVALID_DEVICE = 11,
    CUFFT_PARSE_ERROR = 12,
    CUFFT_NO_WORKSPACE = 13,
    CUFFT_NOT_IMPLEMENTED = 14,
    CUFFT_NOT_SUPPORTED = 15
} cufftResult_t;

// ── Opaque plan handle ───────────────────────────────────────────────────────
typedef uint64_t cufftHandle;

// ── Transform types ──────────────────────────────────────────────────────────
typedef enum {
    CUFFT_R2C = 0x2a,
    CUFFT_C2R = 0x2c,
    CUFFT_C2C = 0x29,
    CUFFT_D2Z = 0x6a,
    CUFFT_Z2D = 0x6c,
    CUFFT_Z2Z = 0x69
} cufftType_t;

// ── Direction ────────────────────────────────────────────────────────────────
typedef enum {
    CUFFT_FORWARD = -1,
    CUFFT_INVERSE = 1
} cufftDirection_t;

// ── Plan creation ────────────────────────────────────────────────────────────
cufftResult_t cufftPlan1d(cufftHandle *plan, int nx, cufftType_t type, int batch);
cufftResult_t cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType_t type);
cufftResult_t cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType_t type);
cufftResult_t cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed,
                            int istride, int idist, int *onembed, int ostride,
                            int odist, cufftType_t type, int batch);
cufftResult_t cufftDestroy(cufftHandle plan);

// ── Execution (complex → complex) ────────────────────────────────────────────
cufftResult_t cufftExecC2C(cufftHandle plan, void *idata, void *odata, int direction);
cufftResult_t cufftExecZ2Z(cufftHandle plan, void *idata, void *odata, int direction);

// ── Execution (real ↔ complex) ─────────────────────────────────────────────
cufftResult_t cufftExecR2C(cufftHandle plan, void *idata, void *odata);
cufftResult_t cufftExecC2R(cufftHandle plan, void *idata, void *odata);
cufftResult_t cufftExecD2Z(cufftHandle plan, void *idata, void *odata);
cufftResult_t cufftExecZ2D(cufftHandle plan, void *idata, void *odata);

// ── Advanced ─────────────────────────────────────────────────────────────────
cufftResult_t cufftSetStream(cufftHandle plan, void *stream);
cufftResult_t cufftSetWorkArea(cufftHandle plan, void *workArea);

#ifdef __cplusplus
} // extern "C"
#endif
