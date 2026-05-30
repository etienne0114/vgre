// cuSOLVER Dn type-erasure API (CUDA 11.1+)
//
// cusolverDnXgetrf / cusolverDnXpotrf / cusolverDnXgesvd / cusolverDnXsygvd
// dispatch on cudaDataType (CUDA_R_32F -> S-prefix LAPACK, CUDA_R_64F -> D-prefix).
// cusolverDnParams is an opaque handle; we store no configuration in it.

#include "vgre/api/cusolver_shim.h"
#include "vgre/api/cusparse_shim.h"   // for cudaDataType_t / CUDA_R_32F / CUDA_R_64F
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

// All typed cuSolver functions are declared in cusolver_shim.h (included above).

// ── cusolverDnParams (opaque; no actual options needed for VGRE) ──────────────
struct cusolverDnParams_st {};
typedef cusolverDnParams_st* cusolverDnParams_t;

// ── Uplo / FillMode helper ────────────────────────────────────────────────────
// CUBLAS/cuSOLVER convention: CUBLAS_FILL_MODE_LOWER=0, UPPER=1
static char fillModeToChar(int fillMode) {
    return (fillMode == 0) ? 'L' : 'U';
}

// ── Compute type helper ───────────────────────────────────────────────────────
// We only distinguish float vs double; all other compute types promote to double.
static bool isFloat32(cudaDataType_t dt) {
    return (dt == CUDA_R_32F);
}

extern "C" {

// ── Params handle ────────────────────────────────────────────────────────────
cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* params) {
    if (!params) return CUSOLVER_STATUS_INVALID_VALUE;
    *params = new cusolverDnParams_st();
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t params) {
    delete params;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSetAdvOptions(cusolverDnParams_t, int, int) {
    return CUSOLVER_STATUS_SUCCESS;  // no-op; VGRE has one algorithm path
}

// ── cusolverDnXgetrf_bufferSize ───────────────────────────────────────────────
cusolverStatus_t cusolverDnXgetrf_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t /*params*/,
    int m, int n,
    cudaDataType_t dataTypeA, void* /*A*/, int lda,
    cudaDataType_t /*computeType*/,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost)
{
    if (!workspaceInBytesOnDevice || !workspaceInBytesOnHost) return CUSOLVER_STATUS_INVALID_VALUE;
    int lwork = 0;
    cusolverStatus_t st;
    if (isFloat32(dataTypeA)) {
        st = cusolverDnSgetrf_bufferSize(handle, m, n, nullptr, lda, &lwork);
        *workspaceInBytesOnDevice = (size_t)lwork * sizeof(float);
    } else {
        st = cusolverDnDgetrf_bufferSize(handle, m, n, nullptr, lda, &lwork);
        *workspaceInBytesOnDevice = (size_t)lwork * sizeof(double);
    }
    *workspaceInBytesOnHost = 0;
    return st;
}

// ── cusolverDnXgetrf ──────────────────────────────────────────────────────────
cusolverStatus_t cusolverDnXgetrf(
    cusolverDnHandle_t handle,
    cusolverDnParams_t /*params*/,
    int m, int n,
    cudaDataType_t dataTypeA, void* A, int lda,
    int64_t* devIpiv,
    cudaDataType_t /*computeType*/,
    void* bufferOnDevice, size_t /*bufferOnDeviceBytes*/,
    void* /*bufferOnHost*/,  size_t /*bufferOnHostBytes*/,
    int* devInfo)
{
    if (!A) return CUSOLVER_STATUS_INVALID_VALUE;
    // devIpiv is int64_t* in the X API; typed variants use int*.  Allocate a
    // temporary int array and copy back when done.
    int* ipiv32 = nullptr;
    int minmn = (m < n) ? m : n;
    if (devIpiv) {
        ipiv32 = new int[minmn]();
    }
    cusolverStatus_t st;
    if (isFloat32(dataTypeA)) {
        st = cusolverDnSgetrf(handle, m, n, (float*)A, lda,
                              (float*)bufferOnDevice, ipiv32, devInfo);
    } else {
        st = cusolverDnDgetrf(handle, m, n, (double*)A, lda,
                              (double*)bufferOnDevice, ipiv32, devInfo);
    }
    if (devIpiv && ipiv32) {
        for (int i = 0; i < minmn; ++i) devIpiv[i] = (int64_t)ipiv32[i];
    }
    delete[] ipiv32;
    return st;
}

// ── cusolverDnXpotrf_bufferSize ───────────────────────────────────────────────
cusolverStatus_t cusolverDnXpotrf_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t /*params*/,
    int fillMode, int n,
    cudaDataType_t dataTypeA, void* A, int lda,
    cudaDataType_t /*computeType*/,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost)
{
    if (!workspaceInBytesOnDevice || !workspaceInBytesOnHost) return CUSOLVER_STATUS_INVALID_VALUE;
    char uplo = fillModeToChar(fillMode);
    int lwork = 0;
    cusolverStatus_t st;
    if (isFloat32(dataTypeA)) {
        st = cusolverDnSpotrf_bufferSize(handle, uplo, n, (float*)A, lda, &lwork);
        *workspaceInBytesOnDevice = (size_t)lwork * sizeof(float);
    } else {
        st = cusolverDnDpotrf_bufferSize(handle, uplo, n, (double*)A, lda, &lwork);
        *workspaceInBytesOnDevice = (size_t)lwork * sizeof(double);
    }
    *workspaceInBytesOnHost = 0;
    return st;
}

// ── cusolverDnXpotrf ──────────────────────────────────────────────────────────
cusolverStatus_t cusolverDnXpotrf(
    cusolverDnHandle_t handle,
    cusolverDnParams_t /*params*/,
    int fillMode, int n,
    cudaDataType_t dataTypeA, void* A, int lda,
    cudaDataType_t /*computeType*/,
    void* bufferOnDevice, size_t bufferOnDeviceBytes,
    void* /*bufferOnHost*/,  size_t /*bufferOnHostBytes*/,
    int* devInfo)
{
    if (!A) return CUSOLVER_STATUS_INVALID_VALUE;
    char uplo = fillModeToChar(fillMode);
    if (isFloat32(dataTypeA)) {
        int lwork = (int)(bufferOnDeviceBytes / sizeof(float));
        return cusolverDnSpotrf(handle, uplo, n, (float*)A, lda,
                                (float*)bufferOnDevice, lwork, devInfo);
    } else {
        int lwork = (int)(bufferOnDeviceBytes / sizeof(double));
        return cusolverDnDpotrf(handle, uplo, n, (double*)A, lda,
                                (double*)bufferOnDevice, lwork, devInfo);
    }
}

// ── cusolverDnXgesvd_bufferSize ───────────────────────────────────────────────
cusolverStatus_t cusolverDnXgesvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t /*params*/,
    char jobu, char jobvt,
    int m, int n,
    cudaDataType_t dataTypeA, void* /*A*/, int /*lda*/,
    cudaDataType_t /*dataTypeS*/,
    void* /*U*/,  int /*ldu*/,
    void* /*Vt*/, int /*ldvt*/,
    cudaDataType_t /*computeType*/,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost)
{
    if (!workspaceInBytesOnDevice || !workspaceInBytesOnHost) return CUSOLVER_STATUS_INVALID_VALUE;
    (void)jobu; (void)jobvt;
    int lwork = 0;
    cusolverStatus_t st;
    if (isFloat32(dataTypeA)) {
        st = cusolverDnSgesvd_bufferSize(handle, m, n, &lwork);
        *workspaceInBytesOnDevice = (size_t)lwork * sizeof(float);
    } else {
        st = cusolverDnDgesvd_bufferSize(handle, m, n, &lwork);
        *workspaceInBytesOnDevice = (size_t)lwork * sizeof(double);
    }
    *workspaceInBytesOnHost = 0;
    return st;
}

// ── cusolverDnXgesvd ──────────────────────────────────────────────────────────
cusolverStatus_t cusolverDnXgesvd(
    cusolverDnHandle_t handle,
    cusolverDnParams_t /*params*/,
    char jobu, char jobvt,
    int m, int n,
    cudaDataType_t dataTypeA, void* A, int lda,
    cudaDataType_t /*dataTypeS*/, void* S,
    void* U,  int ldu,
    void* Vt, int ldvt,
    cudaDataType_t /*computeType*/,
    void* bufferOnDevice, size_t bufferOnDeviceBytes,
    void* /*bufferOnHost*/,  size_t /*bufferOnHostBytes*/,
    int* devInfo)
{
    if (!A || !S) return CUSOLVER_STATUS_INVALID_VALUE;
    if (isFloat32(dataTypeA)) {
        int lwork = (int)(bufferOnDeviceBytes / sizeof(float));
        return cusolverDnSgesvd(handle, jobu, jobvt, m, n,
                                (float*)A, lda, (float*)S,
                                (float*)U, ldu, (float*)Vt, ldvt,
                                (float*)bufferOnDevice, lwork,
                                nullptr, devInfo);
    } else {
        int lwork = (int)(bufferOnDeviceBytes / sizeof(double));
        return cusolverDnDgesvd(handle, jobu, jobvt, m, n,
                                (double*)A, lda, (double*)S,
                                (double*)U, ldu, (double*)Vt, ldvt,
                                (double*)bufferOnDevice, lwork,
                                nullptr, devInfo);
    }
}

// ── cusolverDnXsygvd_bufferSize ───────────────────────────────────────────────
cusolverStatus_t cusolverDnXsygvd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t /*params*/,
    int /*itype*/, char jobz, char uplo,
    int n,
    cudaDataType_t dataTypeA, void* A, int lda,
    cudaDataType_t /*dataTypeB*/, void* /*B*/, int /*ldb*/,
    cudaDataType_t /*dataTypeW*/, void* /*W*/,
    cudaDataType_t /*computeType*/,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost)
{
    if (!workspaceInBytesOnDevice || !workspaceInBytesOnHost) return CUSOLVER_STATUS_INVALID_VALUE;
    int lwork = 0;
    cusolverStatus_t st;
    if (isFloat32(dataTypeA)) {
        st = cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, (float*)A, lda, nullptr, &lwork);
        *workspaceInBytesOnDevice = (size_t)lwork * sizeof(float);
    } else {
        st = cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, (double*)A, lda, nullptr, &lwork);
        *workspaceInBytesOnDevice = (size_t)lwork * sizeof(double);
    }
    *workspaceInBytesOnHost = 0;
    return st;
}

} // extern "C" — pause for C++ template helpers

template<typename T>
static void transform_generalized_congruence(int n, char uplo, T* A, int lda, const T* B, int ldb) {
    bool lower = (uplo == 'L' || uplo == 'l');
    std::vector<T> Y(static_cast<size_t>(n) * n, 0);

    if (lower) {
        // Solve L * Y = A (forward substitution on columns of A)
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                T sum = A[i + j * lda];
                for (int k = 0; k < i; ++k) {
                    sum -= B[i + k * ldb] * Y[k + j * n];
                }
                Y[i + j * n] = sum / B[i + i * ldb];
            }
        }
        // Solve L * A' = Y^T (forward substitution on rows of A')
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                T sum = Y[j + i * n]; // Y^T_{j,i}
                for (int k = 0; k < j; ++k) {
                    sum -= B[j + k * ldb] * A[i + k * lda];
                }
                A[i + j * lda] = sum / B[j + j * ldb];
            }
        }
    } else {
        // Solve U^T * Y = A
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                T sum = A[i + j * lda];
                for (int k = 0; k < i; ++k) {
                    sum -= B[k + i * ldb] * Y[k + j * n];
                }
                Y[i + j * n] = sum / B[i + i * ldb];
            }
        }
        // Solve U^T * A' = Y^T
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                T sum = Y[j + i * n];
                for (int k = 0; k < j; ++k) {
                    sum -= B[k + j * ldb] * A[i + k * lda];
                }
                A[i + j * lda] = sum / B[j + j * ldb];
            }
        }
    }
}

template<typename T>
static void back_project_eigenvectors(int n, char uplo, T* A, int lda, const T* B, int ldb) {
    bool lower = (uplo == 'L' || uplo == 'l');
    std::vector<T> X_col(n);

    if (lower) {
        // Solve L^T * X = Z (backward substitution on columns of Z in A)
        for (int j = 0; j < n; ++j) {
            for (int i = n - 1; i >= 0; --i) {
                T sum = A[i + j * lda];
                for (int k = i + 1; k < n; ++k) {
                    sum -= B[k + i * ldb] * X_col[k];
                }
                X_col[i] = sum / B[i + i * ldb];
            }
            for (int i = 0; i < n; ++i) {
                A[i + j * lda] = X_col[i];
            }
        }
    } else {
        // Solve U * X = Z
        for (int j = 0; j < n; ++j) {
            for (int i = n - 1; i >= 0; --i) {
                T sum = A[i + j * lda];
                for (int k = i + 1; k < n; ++k) {
                    sum -= B[i + k * ldb] * X_col[k];
                }
                X_col[i] = sum / B[i + i * ldb];
            }
            for (int i = 0; i < n; ++i) {
                A[i + j * lda] = X_col[i];
            }
        }
    }
}

extern "C" { // resume C linkage

// ── cusolverDnXsygvd ──────────────────────────────────────────────────────────
// Generalised eigenvalue problem A*x = lambda*B*x.
// Strategy: compute Cholesky of B, transform A, then syevd, and back-project.
cusolverStatus_t cusolverDnXsygvd(
    cusolverDnHandle_t handle,
    cusolverDnParams_t /*params*/,
    int itype, char jobz, char uplo,
    int n,
    cudaDataType_t dataTypeA, void* A, int lda,
    cudaDataType_t /*dataTypeB*/, void* B, int ldb,
    cudaDataType_t /*dataTypeW*/, void* W,
    cudaDataType_t /*computeType*/,
    void* bufferOnDevice, size_t bufferOnDeviceBytes,
    void* /*bufferOnHost*/,  size_t /*bufferOnHostBytes*/,
    int* devInfo)
{
    if (!A || !B || !W) return CUSOLVER_STATUS_INVALID_VALUE;
    (void)itype;

    // Step 1: Cholesky factorize B.
    cusolverStatus_t st;
    if (isFloat32(dataTypeA)) {
        int lwork = (int)(bufferOnDeviceBytes / sizeof(float));
        st = cusolverDnSpotrf(handle, uplo, n, (float*)B, ldb,
                              (float*)bufferOnDevice, lwork, devInfo);
        if (st != CUSOLVER_STATUS_SUCCESS) return st;
        if (*devInfo != 0) return CUSOLVER_STATUS_SUCCESS;  // not positive definite

        // Step 2: Transform A → L^{-1} A L^{-T} or U^{-T} A U^{-1} (itype=1).
        transform_generalized_congruence<float>(n, uplo, (float*)A, lda, (const float*)B, ldb);

        // Step 3: Solve standard symmetric eigenvalue problem on transformed A
        st = cusolverDnSsyevd(handle, jobz, uplo, n, (float*)A, lda,
                              (float*)W, (float*)bufferOnDevice, lwork, devInfo);
        if (st != CUSOLVER_STATUS_SUCCESS) return st;

        // Step 4: Back-project eigenvectors (if requested)
        if (jobz == 'V' || jobz == 'v') {
            back_project_eigenvectors<float>(n, uplo, (float*)A, lda, (const float*)B, ldb);
        }
    } else {
        int lwork = (int)(bufferOnDeviceBytes / sizeof(double));
        st = cusolverDnDpotrf(handle, uplo, n, (double*)B, ldb,
                              (double*)bufferOnDevice, lwork, devInfo);
        if (st != CUSOLVER_STATUS_SUCCESS) return st;
        if (*devInfo != 0) return CUSOLVER_STATUS_SUCCESS;

        transform_generalized_congruence<double>(n, uplo, (double*)A, lda, (const double*)B, ldb);

        st = cusolverDnDsyevd(handle, jobz, uplo, n, (double*)A, lda,
                              (double*)W, (double*)bufferOnDevice, lwork, devInfo);
        if (st != CUSOLVER_STATUS_SUCCESS) return st;

        if (jobz == 'V' || jobz == 'v') {
            back_project_eigenvectors<double>(n, uplo, (double*)A, lda, (const double*)B, ldb);
        }
    }
    return st;
}

// ── cusolverDnXsyevd_bufferSize / cusolverDnXsyevd ───────────────────────────
// Standard symmetric eigenvalue (not generalised).
cusolverStatus_t cusolverDnXsyevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverDnParams_t /*params*/,
    char jobz, char uplo, int n,
    cudaDataType_t dataTypeA, void* A, int lda,
    cudaDataType_t /*dataTypeW*/, void* /*W*/,
    cudaDataType_t /*computeType*/,
    size_t* workspaceInBytesOnDevice,
    size_t* workspaceInBytesOnHost)
{
    if (!workspaceInBytesOnDevice || !workspaceInBytesOnHost) return CUSOLVER_STATUS_INVALID_VALUE;
    int lwork = 0;
    cusolverStatus_t st;
    if (isFloat32(dataTypeA)) {
        st = cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, (float*)A, lda, nullptr, &lwork);
        *workspaceInBytesOnDevice = (size_t)lwork * sizeof(float);
    } else {
        st = cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, (double*)A, lda, nullptr, &lwork);
        *workspaceInBytesOnDevice = (size_t)lwork * sizeof(double);
    }
    *workspaceInBytesOnHost = 0;
    return st;
}

cusolverStatus_t cusolverDnXsyevd(
    cusolverDnHandle_t handle,
    cusolverDnParams_t /*params*/,
    char jobz, char uplo, int n,
    cudaDataType_t dataTypeA, void* A, int lda,
    cudaDataType_t /*dataTypeW*/, void* W,
    cudaDataType_t /*computeType*/,
    void* bufferOnDevice, size_t bufferOnDeviceBytes,
    void* /*bufferOnHost*/,  size_t /*bufferOnHostBytes*/,
    int* devInfo)
{
    if (!A || !W) return CUSOLVER_STATUS_INVALID_VALUE;
    if (isFloat32(dataTypeA)) {
        int lwork = (int)(bufferOnDeviceBytes / sizeof(float));
        return cusolverDnSsyevd(handle, jobz, uplo, n, (float*)A, lda,
                                (float*)W, (float*)bufferOnDevice, lwork, devInfo);
    } else {
        int lwork = (int)(bufferOnDeviceBytes / sizeof(double));
        return cusolverDnDsyevd(handle, jobz, uplo, n, (double*)A, lda,
                                (double*)W, (double*)bufferOnDevice, lwork, devInfo);
    }
}

} // extern "C"
