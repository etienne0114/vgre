// cuBLAS Complex (C/Z) Level-1 routines: COPY, SWAP, SCAL, AXPY, DOT, NRM2, AMAX, ASUM, ROT

#include "cublas_internal.h"
#include <cmath>

extern "C" {

// ═══════════════════════════════════════════════════════════════════════════════
// Level-1 Complex BLAS
// ═══════════════════════════════════════════════════════════════════════════════

// ── CCOPY / ZCOPY ─────────────────────────────────────────────────────────────
cublasStatus_t cublasCcopy_v2(cublasHandle_t handle, int n,
    const cuComplex* x, int incx, cuComplex* y, int incy)
{
    if (!handle || n < 0 || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < n; ++i)
        y[i * incy] = x[i * incx];
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZcopy_v2(cublasHandle_t handle, int n,
    const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy)
{
    if (!handle || n < 0 || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < n; ++i)
        y[i * incy] = x[i * incx];
    return CUBLAS_STATUS_SUCCESS;
}

// ── CSWAP / ZSWAP ────────────────────────────────────────────────────────────
cublasStatus_t cublasCswap_v2(cublasHandle_t handle, int n,
    cuComplex* x, int incx, cuComplex* y, int incy)
{
    if (!handle || n < 0 || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < n; ++i) {
        cuComplex tmp = x[i * incx];
        x[i * incx] = y[i * incy];
        y[i * incy] = tmp;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZswap_v2(cublasHandle_t handle, int n,
    cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy)
{
    if (!handle || n < 0 || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
    for (int i = 0; i < n; ++i) {
        cuDoubleComplex tmp = x[i * incx];
        x[i * incx] = y[i * incy];
        y[i * incy] = tmp;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── CSCAL / ZSCAL / CSSCAL / ZDSCAL ─────────────────────────────────────────
cublasStatus_t cublasCscal_v2(cublasHandle_t handle, int n,
    const cuComplex* alpha, cuComplex* x, int incx)
{
    if (!handle || n < 0 || !alpha || !x) return CUBLAS_STATUS_INVALID_VALUE;
    cuComplex a = *alpha;
    for (int i = 0; i < n; ++i)
        x[i * incx] = cuCmulf(a, x[i * incx]);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZscal_v2(cublasHandle_t handle, int n,
    const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx)
{
    if (!handle || n < 0 || !alpha || !x) return CUBLAS_STATUS_INVALID_VALUE;
    cuDoubleComplex a = *alpha;
    for (int i = 0; i < n; ++i)
        x[i * incx] = cuCmul(a, x[i * incx]);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCsscal_v2(cublasHandle_t handle, int n,
    const float* alpha, cuComplex* x, int incx)
{
    if (!handle || n < 0 || !alpha || !x) return CUBLAS_STATUS_INVALID_VALUE;
    float a = *alpha;
    for (int i = 0; i < n; ++i)
        x[i * incx] = cuCmulf_real(x[i * incx], a);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdscal_v2(cublasHandle_t handle, int n,
    const double* alpha, cuDoubleComplex* x, int incx)
{
    if (!handle || n < 0 || !alpha || !x) return CUBLAS_STATUS_INVALID_VALUE;
    double a = *alpha;
    for (int i = 0; i < n; ++i)
        x[i * incx] = cuCmul_real(x[i * incx], a);
    return CUBLAS_STATUS_SUCCESS;
}

// ── CAXPY / ZAXPY ────────────────────────────────────────────────────────────
cublasStatus_t cublasCaxpy_v2(cublasHandle_t handle, int n,
    const cuComplex* alpha, const cuComplex* x, int incx,
    cuComplex* y, int incy)
{
    if (!handle || n < 0 || !alpha || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
    cuComplex a = *alpha;
    for (int i = 0; i < n; ++i)
        y[i * incy] = cuCaddf(y[i * incy], cuCmulf(a, x[i * incx]));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZaxpy_v2(cublasHandle_t handle, int n,
    const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx,
    cuDoubleComplex* y, int incy)
{
    if (!handle || n < 0 || !alpha || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
    cuDoubleComplex a = *alpha;
    for (int i = 0; i < n; ++i)
        y[i * incy] = cuCadd(y[i * incy], cuCmul(a, x[i * incx]));
    return CUBLAS_STATUS_SUCCESS;
}

// ── CDOTU / ZDOTU (unconjugated dot product) ─────────────────────────────────
cublasStatus_t cublasCdotu_v2(cublasHandle_t handle, int n,
    const cuComplex* x, int incx, const cuComplex* y, int incy,
    cuComplex* result)
{
    if (!handle || n < 0 || !x || !y || !result) return CUBLAS_STATUS_INVALID_VALUE;
    cuComplex acc = make_cuComplex(0.f, 0.f);
    for (int i = 0; i < n; ++i)
        acc = cuCaddf(acc, cuCmulf(x[i * incx], y[i * incy]));
    *result = acc;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotu_v2(cublasHandle_t handle, int n,
    const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy,
    cuDoubleComplex* result)
{
    if (!handle || n < 0 || !x || !y || !result) return CUBLAS_STATUS_INVALID_VALUE;
    cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < n; ++i)
        acc = cuCadd(acc, cuCmul(x[i * incx], y[i * incy]));
    *result = acc;
    return CUBLAS_STATUS_SUCCESS;
}

// ── CDOTC / ZDOTC (conjugated dot product: conj(x) . y) ─────────────────────
cublasStatus_t cublasCdotc_v2(cublasHandle_t handle, int n,
    const cuComplex* x, int incx, const cuComplex* y, int incy,
    cuComplex* result)
{
    if (!handle || n < 0 || !x || !y || !result) return CUBLAS_STATUS_INVALID_VALUE;
    cuComplex acc = make_cuComplex(0.f, 0.f);
    for (int i = 0; i < n; ++i)
        acc = cuCaddf(acc, cuCmulf(cuConjf(x[i * incx]), y[i * incy]));
    *result = acc;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZdotc_v2(cublasHandle_t handle, int n,
    const cuDoubleComplex* x, int incx, const cuDoubleComplex* y, int incy,
    cuDoubleComplex* result)
{
    if (!handle || n < 0 || !x || !y || !result) return CUBLAS_STATUS_INVALID_VALUE;
    cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
    for (int i = 0; i < n; ++i)
        acc = cuCadd(acc, cuCmul(cuConj(x[i * incx]), y[i * incy]));
    *result = acc;
    return CUBLAS_STATUS_SUCCESS;
}

// ── SCNRM2 / DZNRM2 (Euclidean norm) ────────────────────────────────────────
cublasStatus_t cublasScnrm2_v2(cublasHandle_t handle, int n,
    const cuComplex* x, int incx, float* result)
{
    if (!handle || n < 0 || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        cuComplex v = x[i * incx];
        sum += v.x * v.x + v.y * v.y;
    }
    *result = std::sqrt(sum);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDznrm2_v2(cublasHandle_t handle, int n,
    const cuDoubleComplex* x, int incx, double* result)
{
    if (!handle || n < 0 || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        cuDoubleComplex v = x[i * incx];
        sum += v.x * v.x + v.y * v.y;
    }
    *result = std::sqrt(sum);
    return CUBLAS_STATUS_SUCCESS;
}

// ── ICAMAX / IZAMAX ──────────────────────────────────────────────────────────
cublasStatus_t cublasIcamax_v2(cublasHandle_t handle, int n,
    const cuComplex* x, int incx, int* result)
{
    if (!handle || n < 0 || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) { *result = 0; return CUBLAS_STATUS_SUCCESS; }
    int idx = 0;
    float maxVal = std::fabs(x[0].x) + std::fabs(x[0].y);
    for (int i = 1; i < n; ++i) {
        float v = std::fabs(x[i * incx].x) + std::fabs(x[i * incx].y);
        if (v > maxVal) { maxVal = v; idx = i; }
    }
    *result = idx + 1; // 1-based index (Fortran convention)
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIzamax_v2(cublasHandle_t handle, int n,
    const cuDoubleComplex* x, int incx, int* result)
{
    if (!handle || n < 0 || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) { *result = 0; return CUBLAS_STATUS_SUCCESS; }
    int idx = 0;
    double maxVal = std::fabs(x[0].x) + std::fabs(x[0].y);
    for (int i = 1; i < n; ++i) {
        double v = std::fabs(x[i * incx].x) + std::fabs(x[i * incx].y);
        if (v > maxVal) { maxVal = v; idx = i; }
    }
    *result = idx + 1;
    return CUBLAS_STATUS_SUCCESS;
}

// ── SCASUM / DZASUM (sum of absolute values) ────────────────────────────────
cublasStatus_t cublasScasum_v2(cublasHandle_t handle, int n,
    const cuComplex* x, int incx, float* result)
{
    if (!handle || n < 0 || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        cuComplex v = x[i * incx];
        sum += std::fabs(v.x) + std::fabs(v.y);
    }
    *result = sum;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDzasum_v2(cublasHandle_t handle, int n,
    const cuDoubleComplex* x, int incx, double* result)
{
    if (!handle || n < 0 || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        cuDoubleComplex v = x[i * incx];
        sum += std::fabs(v.x) + std::fabs(v.y);
    }
    *result = sum;
    return CUBLAS_STATUS_SUCCESS;
}

// ── CROT / ZROT (plane rotation) ─────────────────────────────────────────────
cublasStatus_t cublasCrot_v2(cublasHandle_t handle, int n,
    cuComplex* x, int incx, cuComplex* y, int incy,
    const float* c, const cuComplex* s)
{
    if (!handle || n < 0 || !x || !y || !c || !s) return CUBLAS_STATUS_INVALID_VALUE;
    float cs = *c;
    cuComplex sn = *s;
    for (int i = 0; i < n; ++i) {
        cuComplex xi = x[i * incx];
        cuComplex yi = y[i * incy];
        x[i * incx] = cuCaddf(cuCmulf_real(xi, cs), cuCmulf(sn, yi));
        y[i * incy] = cuCsubf(cuCmulf_real(yi, cs), cuCmulf(cuConjf(sn), xi));
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZrot_v2(cublasHandle_t handle, int n,
    cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy,
    const double* c, const cuDoubleComplex* s)
{
    if (!handle || n < 0 || !x || !y || !c || !s) return CUBLAS_STATUS_INVALID_VALUE;
    double cs = *c;
    cuDoubleComplex sn = *s;
    for (int i = 0; i < n; ++i) {
        cuDoubleComplex xi = x[i * incx];
        cuDoubleComplex yi = y[i * incy];
        x[i * incx] = cuCadd(cuCmul_real(xi, cs), cuCmul(sn, yi));
        y[i * incy] = cuCsub(cuCmul_real(yi, cs), cuCmul(cuConj(sn), xi));
    }
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"
