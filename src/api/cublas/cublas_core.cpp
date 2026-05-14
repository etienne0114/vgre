// cuBLAS core API functions

#include "cublas_internal.h"

extern "C" {


cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    *handle = new cublasContext();
    VGRE_LOG_DEBUG("cuBLAS", "cublasCreate → handle=" + std::to_string(reinterpret_cast<uintptr_t>(*handle)));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    delete static_cast<cublasContext*>(handle);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t /*handle*/, int* version) {
    if (version) *version = 12000;
    return CUBLAS_STATUS_SUCCESS;
}

// ── Stream / math mode (stored in handle, applied on synchronize) ────────────
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, void* stream) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    static_cast<cublasContext*>(handle)->stream = stream;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, void** stream) {
    if (!handle || !stream) return CUBLAS_STATUS_INVALID_VALUE;
    *stream = static_cast<cublasContext*>(handle)->stream;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, int mode) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    static_cast<cublasContext*>(handle)->mathMode = static_cast<cublasMath_t>(mode);
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasGetMathMode(cublasHandle_t handle, int* mode) {
    if (!handle || !mode) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = static_cast<int>(static_cast<cublasContext*>(handle)->mathMode);
    return CUBLAS_STATUS_SUCCESS;
}

// ── Pointer mode APIs (P2.2) ──────────────────────────────────────────────
cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, int mode) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    static_cast<cublasContext*>(handle)->pointerMode = static_cast<cublasPointerMode_t>(mode);
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, int* mode) {
    if (!handle || !mode) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = static_cast<int>(static_cast<cublasContext*>(handle)->pointerMode);
    return CUBLAS_STATUS_SUCCESS;
}

// ── Atomics mode APIs (P2.2) ────────────────────────────────────────────────
cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, int mode) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    static_cast<cublasContext*>(handle)->atomicsMode = static_cast<cublasAtomicsMode_t>(mode);
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, int* mode) {
    if (!handle || !mode) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = static_cast<int>(static_cast<cublasContext*>(handle)->atomicsMode);
    return CUBLAS_STATUS_SUCCESS;
}

// ── SYR ──────────────────────────────────────────────────────────────────────
cublasStatus_t cublasSsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
    const float* alpha, const float* x, int incx, float* A, int lda)
{
    if (!handle || !alpha || !x || !A) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_ssyr(CblasRowMajor, cublasToCblasUplo(uplo), n,
               *alpha, x, incx, A, lda);
#else
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        float temp = (*alpha) * x[j*incx];
        if (upper) {
            for (int i = 0; i <= j; ++i)
                A[i*lda+j] += x[i*incx] * temp;
        } else {
            for (int i = j; i < n; ++i)
                A[i*lda+j] += x[i*incx] * temp;
        }
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyr_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
    const double* alpha, const double* x, int incx, double* A, int lda)
{
    if (!handle || !alpha || !x || !A) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_dsyr(CblasRowMajor, cublasToCblasUplo(uplo), n,
               *alpha, x, incx, A, lda);
#else
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        double temp = (*alpha) * x[j*incx];
        if (upper) {
            for (int i = 0; i <= j; ++i)
                A[i*lda+j] += x[i*incx] * temp;
        } else {
            for (int i = j; i < n; ++i)
                A[i*lda+j] += x[i*incx] * temp;
        }
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}


// ── Legacy v1 aliases ─────────────────────────────────────────────────────────
cublasStatus_t cublasCreate(cublasHandle_t* h)  { return cublasCreate_v2(h); }
cublasStatus_t cublasDestroy(cublasHandle_t h)  { return cublasDestroy_v2(h); }

} // extern "C"
