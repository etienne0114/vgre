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
    CUFFT_Z2Z = 0x69,
    // VGRE extension: half-precision transforms. CUDA exposes FP16 plans
    // through cuFFT Xt; these compact types keep the shim usable without
    // pulling in the CUDA headers.
    CUFFT_R16C   = 0x72,
    CUFFT_C16R   = 0x73,
    CUFFT_C16C   = 0x74,
    CUFFT_C16BFC = 0x75, // bfloat16 complex-to-complex
    // VGRE extension: real-to-real DCT via 2N-point FFT embedding (RFC-none).
    // DCT-II: X[k] = Σ x[n]·cos(π(2n+1)k/(2N)), n=0..N-1, O(N log N)
    // DCT-III: x[n] = X[0]/2 + Σ X[k]·cos(π(2n+1)k/(2N)), k=1..N-1
    // DCT-III(DCT-II(x)) = (N/2)·x  (orthogonality up to scale)
    CUFFT_DCT2 = 0x80,
    CUFFT_DCT3 = 0x81
} cufftType_t;

typedef struct {
    uint16_t x;
    uint16_t y;
} cufftHalfComplex;

typedef struct {
    uint16_t x;  // bfloat16 real part
    uint16_t y;  // bfloat16 imaginary part
} cufftBF16Complex;

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
cufftResult_t cufftExecR16C(cufftHandle plan, void *idata, void *odata);
cufftResult_t cufftExecC16R(cufftHandle plan, void *idata, void *odata);
cufftResult_t cufftExecC16C(cufftHandle plan, void *idata, void *odata, int direction);
cufftResult_t cufftExecC16BFC(cufftHandle plan, void *idata, void *odata, int direction);

// ── DCT-II / DCT-III (real → real, float only) ────────────────────────────────
cufftResult_t cufftExecDCT2(cufftHandle plan, float *idata, float *odata);
cufftResult_t cufftExecDCT3(cufftHandle plan, float *idata, float *odata);

// ── Stream / workspace association ───────────────────────────────────────────
cufftResult_t cufftSetStream(cufftHandle plan, void *stream);
cufftResult_t cufftGetStream(cufftHandle plan, void **stream);
cufftResult_t cufftSetWorkArea(cufftHandle plan, void *workArea);
cufftResult_t cufftGetVersion(int *version);
cufftResult_t cufftEstimate1d(int nx, cufftType_t type, int batch, size_t *workSize);
cufftResult_t cufftEstimate2d(int nx, int ny, cufftType_t type, size_t *workSize);
cufftResult_t cufftEstimate3d(int nx, int ny, int nz, cufftType_t type, size_t *workSize);
cufftResult_t cufftGetSize1d(cufftHandle plan, int nx, cufftType_t type, int batch, size_t *workSize);
cufftResult_t cufftGetSize2d(cufftHandle plan, int nx, int ny, cufftType_t type, size_t *workSize);
cufftResult_t cufftGetSize3d(cufftHandle plan, int nx, int ny, int nz, cufftType_t type, size_t *workSize);
cufftResult_t cufftGetSizeMany(cufftHandle plan, int rank, int *n, int *inembed,
                               int istride, int idist, int *onembed, int ostride,
                               int odist, cufftType_t type, int batch, size_t *workSize);
cufftResult_t cufftMakePlanMany(cufftHandle plan, int rank, int *n, int *inembed,
                                int istride, int idist, int *onembed, int ostride,
                                int odist, cufftType_t type, int batch, size_t *workSize);

typedef struct vgre_cufft_ipc_handle_t {
    int32_t rank;
    int32_t nx, ny, nz;
    int32_t batch;
    int32_t type;
    uint8_t reserved[40];
} vgre_cufft_ipc_handle_t;
cufftResult_t cufftGetPlanIpcHandle(cufftHandle plan, vgre_cufft_ipc_handle_t *handle);
cufftResult_t cufftCreatePlanFromIpcHandle(cufftHandle *plan,
                                           const vgre_cufft_ipc_handle_t *handle);

// ── cuFFT Xt multi-GPU / callback API ────────────────────────────────────────

typedef enum {
    CUFFT_XT_FORMAT_INPUT              = 0x00,
    CUFFT_XT_FORMAT_OUTPUT             = 0x01,
    CUFFT_XT_FORMAT_INPLACE            = 0x02,
    CUFFT_XT_FORMAT_INPLACE_SHUFFLED   = 0x03,
    CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED  = 0x04,
    CUFFT_XT_FORMAT_UNDEFINED          = 0x80
} cufftXtSubFormat_t;

typedef enum {
    CUFFT_COPY_HOST_TO_DEVICE   = 1,
    CUFFT_COPY_DEVICE_TO_HOST   = 2,
    CUFFT_COPY_DEVICE_TO_DEVICE = 3
} cufftXtCopyType_t;

typedef enum {
    CUFFT_CB_LD_COMPLEX        = 0x0,
    CUFFT_CB_LD_COMPLEX_DOUBLE = 0x1,
    CUFFT_CB_LD_REAL           = 0x2,
    CUFFT_CB_LD_REAL_DOUBLE    = 0x3,
    CUFFT_CB_ST_COMPLEX        = 0x4,
    CUFFT_CB_ST_COMPLEX_DOUBLE = 0x5,
    CUFFT_CB_ST_REAL           = 0x6,
    CUFFT_CB_ST_REAL_DOUBLE    = 0x7,
    CUFFT_CB_UNDEFINED         = 0x8
} cufftXtCallbackType_t;

typedef struct cudaLibXtDesc {
    int    version;
    int    nGPUs;
    int    GPUs[16];
    void*  data[16];
    size_t size[16];
    void*  subFormat;
} cudaLibXtDesc;

cufftResult_t cufftXtMalloc(cufftHandle plan, cudaLibXtDesc **descriptor,
                             cufftXtSubFormat_t format);
cufftResult_t cufftXtFree(cudaLibXtDesc *descriptor);
cufftResult_t cufftXtMemcpy(cufftHandle plan, void *dstPointer, void *srcPointer,
                             cufftXtCopyType_t type);
cufftResult_t cufftXtExecDescriptor(cufftHandle plan, cudaLibXtDesc *input,
                                    cudaLibXtDesc *output, int direction);
cufftResult_t cufftXtGetSizeMany(cufftHandle plan, size_t *workSize,
                                 cufftXtSubFormat_t inputFormat,
                                 cufftXtSubFormat_t outputFormat,
                                 cufftXtSubFormat_t executionFormat,
                                 int executionType,
                                 int numGPUs, int *whichGPUs);
cufftResult_t cufftXtMakePlanMany(cufftHandle plan, int rank, long long int *n,
                                  long long int *inembed, long long int istride,
                                  long long int idist, int inputType,
                                  long long int *onembed, long long int ostride,
                                  long long int odist, int outputType,
                                  long long int batch, size_t *workSize,
                                  int executionType);
cufftResult_t cufftXtSetCallback(cufftHandle plan, void **callbackRoutine,
                                  cufftXtCallbackType_t cbType, void **callerInfo);
cufftResult_t cufftXtClearCallback(cufftHandle plan, cufftXtCallbackType_t cbType);

#ifdef __cplusplus
} // extern "C"
#endif
