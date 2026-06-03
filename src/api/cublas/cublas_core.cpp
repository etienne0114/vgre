// cuBLAS core API functions

#include "cublas_internal.h"
#include "vgre/common/openmp_helper.h"

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

// ── Workspace APIs (QUEUE-36) ─────────────────────────────────────────────────
// cublasSetWorkspace_v2: associates a user-provided scratch buffer with the handle.
// workspace==nullptr with workspaceSizeInBytes==0 is valid and clears the binding.
// workspace==nullptr with workspaceSizeInBytes>0 is rejected (INVALID_VALUE).
// Math: cuBLAS operations use workspace for split-K GEMM tiles and other
// temporary storage proportional to the problem size.
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle,
                                      void *workspace, size_t workspaceSizeInBytes) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (workspace == nullptr && workspaceSizeInBytes != 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    auto *ctx = static_cast<cublasContext*>(handle);
    ctx->workspace            = workspace;
    ctx->workspaceSizeInBytes = workspaceSizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

// cublasGetWorkspaceSize: returns the workspace buffer size currently bound to handle.
cublasStatus_t cublasGetWorkspaceSize(cublasHandle_t handle, size_t *workspaceSizeInBytes) {
    if (!handle)               return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!workspaceSizeInBytes) return CUBLAS_STATUS_INVALID_VALUE;
    *workspaceSizeInBytes = static_cast<cublasContext*>(handle)->workspaceSizeInBytes;
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
    #ifdef _OPENMP
    #pragma omp parallel for if (n > 64)
    #endif
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
    #ifdef _OPENMP
    #pragma omp parallel for if (n > 64)
    #endif
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

// Unversioned alias — some frameworks call cublasGetVersion without _v2 suffix
cublasStatus_t cublasGetVersion(cublasHandle_t h, int* version) {
    return cublasGetVersion_v2(h, version);
}

// ── Matrix / vector transfer helpers ─────────────────────────────────────────
// These are data movement utilities that copy between row-major host storage
// and column-major device storage. On CPU emulation both pointers are host
// memory so we perform a strided copy respecting lda/ldb layout differences.
cublasStatus_t cublasSetVector(int n, int elemSize, const void* x, int incx,
                                void* y, int incy) {
    if (!x || !y || elemSize <= 0 || n < 0) return CUBLAS_STATUS_INVALID_VALUE;
    const char* src = static_cast<const char*>(x);
    char*       dst = static_cast<char*>(y);
    for (int i = 0; i < n; ++i)
        memcpy(dst + (size_t)i * incy * elemSize,
                    src + (size_t)i * incx * elemSize,
                    elemSize);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVector(int n, int elemSize, const void* x, int incx,
                                void* y, int incy) {
    return cublasSetVector(n, elemSize, x, incx, y, incy);
}

// cublasSetMatrix: copy rows*cols matrix from host (lda columns per row)
//   to device column-major storage (ldb rows per column).
// CUDA convention: A is row-major on host (rows x cols, lda >= cols),
//   DevA is column-major (ldb >= rows).
cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize,
                                const void* A, int lda,
                                void* devA, int ldb) {
    if (!A || !devA || elemSize <= 0 || rows < 0 || cols < 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    const char* src = static_cast<const char*>(A);
    char*       dst = static_cast<char*>(devA);
    // copy column-by-column: dst col j starts at dst + j*ldb*elemSize
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            memcpy(dst + ((size_t)j*ldb + i)*elemSize,
                        src + ((size_t)i*lda + j)*elemSize,
                        elemSize);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize,
                                const void* devA, int lda,
                                void* B, int ldb) {
    if (!devA || !B || elemSize <= 0 || rows < 0 || cols < 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    const char* src = static_cast<const char*>(devA);
    char*       dst = static_cast<char*>(B);
    // devA is column-major (lda >= rows), B is row-major (ldb >= cols)
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            memcpy(dst + ((size_t)i*ldb + j)*elemSize,
                        src + ((size_t)j*lda + i)*elemSize,
                        elemSize);
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"
