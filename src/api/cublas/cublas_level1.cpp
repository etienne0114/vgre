// cuBLAS level1 API functions

#include "cublas_internal.h"
#include "vgre/common/openmp_helper.h"

// ── Kahan compensated summation helpers (QUEUE-08) ───────────────────────────
// Algorithm: Kahan 1965 / Neumaier 1974.
// Loop invariant: (sum + c) approximates the exact partial sum.
// The correction step c = (t - sum) - y must NOT be contracted with FMA:
// FMA would compute (t - sum - y) exactly, hiding the rounding error that
// Kahan is designed to recover. Plain scalar +/- preserves the invariant.
//
// AVX2 vectorized Kahan: 8 (float) or 4 (double) independent Kahan lanes
// run in parallel over contiguous data. After the SIMD loop the 8/4 partial
// sums are horizontally reduced and the scalar tail finishes with Kahan.
// Used for the unit-stride fast path; non-unit stride falls back to scalar.

#ifdef __AVX2__
#include <immintrin.h>

// Horizontal sum of 8 float lanes — exact for ≤ 8 addends (no Kahan needed).
static inline float kahan_hsum8_f32(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// Horizontal sum of 4 double lanes.
static inline double kahan_hsum4_f64(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d s  = _mm_add_pd(lo, hi);
    s = _mm_hadd_pd(s, s);
    return _mm_cvtsd_f64(s);
}
#endif

extern "C" {

// ── SAXPY / DAXPY ────────────────────────────────────────────────────────────
cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n,
    const float* alpha, const float* x, int incx, float* y, int incy)
{
    if (!handle || !x || !y || !alpha) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_saxpy(n, *alpha, x, incx, y, incy);
#else
    #ifdef _OPENMP
    #pragma omp parallel for if (n > 1024)
    #endif
    for (int i = 0; i < n; ++i) y[i*incy] += (*alpha) * x[i*incx];
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n,
    const double* alpha, const double* x, int incx, double* y, int incy)
{
    if (!handle || !x || !y || !alpha) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_daxpy(n, *alpha, x, incx, y, incy);
#else
    #ifdef _OPENMP
    #pragma omp parallel for if (n > 1024)
    #endif
    for (int i = 0; i < n; ++i) y[i*incy] += (*alpha) * x[i*incx];
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SDOT / DDOT ──────────────────────────────────────────────────────────────
cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n,
    const float* x, int incx, const float* y, int incy, float* result)
{
    if (!handle || !x || !y || !result) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    *result = cblas_sdot(n, x, incx, y, incy);
#else
#ifdef __AVX2__
    if (incx == 1 && incy == 1) {
        __m256 vsum = _mm256_setzero_ps(), vc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 xi   = _mm256_loadu_ps(x + i);
            __m256 yi   = _mm256_loadu_ps(y + i);
            __m256 term = _mm256_mul_ps(xi, yi);
            __m256 yv   = _mm256_sub_ps(term, vc);
            __m256 vt   = _mm256_add_ps(vsum, yv);
            vc   = _mm256_sub_ps(_mm256_sub_ps(vt, vsum), yv);
            vsum = vt;
        }
        float sum = kahan_hsum8_f32(vsum), c = 0.f;
        for (; i < n; ++i) {
            float w = x[i] * y[i];
            float yv2 = w - c;
            float t  = sum + yv2;
            c   = (t - sum) - yv2;
            sum = t;
        }
        *result = sum;
        return CUBLAS_STATUS_SUCCESS;
    }
#endif
    float sum = 0.f, c = 0.f;
    for (int i = 0; i < n; ++i) {
        float w  = x[i*incx] * y[i*incy];
        float yv = w - c;
        float t  = sum + yv;
        c   = (t - sum) - yv;
        sum = t;
    }
    *result = sum;
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n,
    const double* x, int incx, const double* y, int incy, double* result)
{
    if (!handle || !x || !y || !result) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    *result = cblas_ddot(n, x, incx, y, incy);
#else
#ifdef __AVX2__
    if (incx == 1 && incy == 1) {
        __m256d vsum = _mm256_setzero_pd(), vc = _mm256_setzero_pd();
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d xi   = _mm256_loadu_pd(x + i);
            __m256d yi   = _mm256_loadu_pd(y + i);
            __m256d term = _mm256_mul_pd(xi, yi);
            __m256d yv   = _mm256_sub_pd(term, vc);
            __m256d vt   = _mm256_add_pd(vsum, yv);
            vc   = _mm256_sub_pd(_mm256_sub_pd(vt, vsum), yv);
            vsum = vt;
        }
        double sum = kahan_hsum4_f64(vsum), c = 0.0;
        for (; i < n; ++i) {
            double w  = x[i] * y[i];
            double yv = w - c;
            double t  = sum + yv;
            c   = (t - sum) - yv;
            sum = t;
        }
        *result = sum;
        return CUBLAS_STATUS_SUCCESS;
    }
#endif
    double sum = 0.0, c = 0.0;
    for (int i = 0; i < n; ++i) {
        double w  = x[i*incx] * y[i*incy];
        double yv = w - c;
        double t  = sum + yv;
        c   = (t - sum) - yv;
        sum = t;
    }
    *result = sum;
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SNRM2 / DNRM2 ────────────────────────────────────────────────────────────
cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n,
    const float* x, int incx, float* result)
{
    if (!handle || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    *result = cblas_snrm2(n, x, incx);
#else
#ifdef __AVX2__
    if (incx == 1) {
        __m256 vsum = _mm256_setzero_ps(), vc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 xi   = _mm256_loadu_ps(x + i);
            __m256 term = _mm256_mul_ps(xi, xi);
            __m256 yv   = _mm256_sub_ps(term, vc);
            __m256 vt   = _mm256_add_ps(vsum, yv);
            vc   = _mm256_sub_ps(_mm256_sub_ps(vt, vsum), yv);
            vsum = vt;
        }
        float sum = kahan_hsum8_f32(vsum), c = 0.f;
        for (; i < n; ++i) {
            float w  = x[i] * x[i];
            float yv = w - c;
            float t  = sum + yv;
            c   = (t - sum) - yv;
            sum = t;
        }
        *result = sqrtf(sum);
        return CUBLAS_STATUS_SUCCESS;
    }
#endif
    float sum = 0.f, c = 0.f;
    for (int i = 0; i < n; ++i) {
        float w  = x[i*incx] * x[i*incx];
        float yv = w - c;
        float t  = sum + yv;
        c   = (t - sum) - yv;
        sum = t;
    }
    *result = sqrtf(sum);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDnrm2_v2(cublasHandle_t handle, int n,
    const double* x, int incx, double* result)
{
    if (!handle || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    *result = cblas_dnrm2(n, x, incx);
#else
#ifdef __AVX2__
    if (incx == 1) {
        __m256d vsum = _mm256_setzero_pd(), vc = _mm256_setzero_pd();
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d xi   = _mm256_loadu_pd(x + i);
            __m256d term = _mm256_mul_pd(xi, xi);
            __m256d yv   = _mm256_sub_pd(term, vc);
            __m256d vt   = _mm256_add_pd(vsum, yv);
            vc   = _mm256_sub_pd(_mm256_sub_pd(vt, vsum), yv);
            vsum = vt;
        }
        double sum = kahan_hsum4_f64(vsum), c = 0.0;
        for (; i < n; ++i) {
            double w  = x[i] * x[i];
            double yv = w - c;
            double t  = sum + yv;
            c   = (t - sum) - yv;
            sum = t;
        }
        *result = sqrt(sum);
        return CUBLAS_STATUS_SUCCESS;
    }
#endif
    double sum = 0.0, c = 0.0;
    for (int i = 0; i < n; ++i) {
        double w  = x[i*incx] * x[i*incx];
        double yv = w - c;
        double t  = sum + yv;
        c   = (t - sum) - yv;
        sum = t;
    }
    *result = sqrt(sum);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SSCAL / DSCAL ────────────────────────────────────────────────────────────
cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n,
    const float* alpha, float* x, int incx)
{
    if (!handle || !x || !alpha) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_sscal(n, *alpha, x, incx);
#else
    #ifdef _OPENMP
    #pragma omp parallel for if (n > 1024)
    #endif
    for (int i = 0; i < n; ++i) x[i*incx] *= *alpha;
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDscal_v2(cublasHandle_t handle, int n,
    const double* alpha, double* x, int incx)
{
    if (!handle || !x || !alpha) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_dscal(n, *alpha, x, incx);
#else
    #ifdef _OPENMP
    #pragma omp parallel for if (n > 1024)
    #endif
    for (int i = 0; i < n; ++i) x[i*incx] *= *alpha;
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SCOPY / DCOPY ────────────────────────────────────────────────────────────
cublasStatus_t cublasScopy_v2(cublasHandle_t handle, int n,
    const float* x, int incx, float* y, int incy)
{
    if (!handle || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_scopy(n, x, incx, y, incy);
#else
    for (int i = 0; i < n; ++i) y[i*incy] = x[i*incx];
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDcopy_v2(cublasHandle_t handle, int n,
    const double* x, int incx, double* y, int incy)
{
    if (!handle || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_dcopy(n, x, incx, y, incy);
#else
    for (int i = 0; i < n; ++i) y[i*incy] = x[i*incx];
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SSWAP / DSWAP ────────────────────────────────────────────────────────────
cublasStatus_t cublasSswap_v2(cublasHandle_t handle, int n,
    float* x, int incx, float* y, int incy)
{
    if (!handle || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_sswap(n, x, incx, y, incy);
#else
    for (int i = 0; i < n; ++i) {
        float t = x[i*incx];
        x[i*incx] = y[i*incy];
        y[i*incy] = t;
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDswap_v2(cublasHandle_t handle, int n,
    double* x, int incx, double* y, int incy)
{
    if (!handle || !x || !y) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_dswap(n, x, incx, y, incy);
#else
    for (int i = 0; i < n; ++i) {
        double t = x[i*incx];
        x[i*incx] = y[i*incy];
        y[i*incy] = t;
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SASUM / DASUM ──────────────────────────────────────────────────────────────
cublasStatus_t cublasSasum_v2(cublasHandle_t handle, int n,
    const float* x, int incx, float* result)
{
    if (!handle || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    *result = cblas_sasum(n, x, incx);
#else
#ifdef __AVX2__
    if (incx == 1) {
        // Clear sign bit to compute |x[i]| without branching.
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        __m256 vsum = _mm256_setzero_ps(), vc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 term = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(x + i));
            __m256 yv   = _mm256_sub_ps(term, vc);
            __m256 vt   = _mm256_add_ps(vsum, yv);
            vc   = _mm256_sub_ps(_mm256_sub_ps(vt, vsum), yv);
            vsum = vt;
        }
        float sum = kahan_hsum8_f32(vsum), c = 0.f;
        for (; i < n; ++i) {
            float w  = std::abs(x[i]);
            float yv = w - c;
            float t  = sum + yv;
            c   = (t - sum) - yv;
            sum = t;
        }
        *result = sum;
        return CUBLAS_STATUS_SUCCESS;
    }
#endif
    float sum = 0.f, c = 0.f;
    for (int i = 0; i < n; ++i) {
        float w  = std::abs(x[i*incx]);
        float yv = w - c;
        float t  = sum + yv;
        c   = (t - sum) - yv;
        sum = t;
    }
    *result = sum;
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDasum_v2(cublasHandle_t handle, int n,
    const double* x, int incx, double* result)
{
    if (!handle || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    *result = cblas_dasum(n, x, incx);
#else
#ifdef __AVX2__
    if (incx == 1) {
        const __m256d sign_mask = _mm256_set1_pd(-0.0);
        __m256d vsum = _mm256_setzero_pd(), vc = _mm256_setzero_pd();
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            __m256d term = _mm256_andnot_pd(sign_mask, _mm256_loadu_pd(x + i));
            __m256d yv   = _mm256_sub_pd(term, vc);
            __m256d vt   = _mm256_add_pd(vsum, yv);
            vc   = _mm256_sub_pd(_mm256_sub_pd(vt, vsum), yv);
            vsum = vt;
        }
        double sum = kahan_hsum4_f64(vsum), c = 0.0;
        for (; i < n; ++i) {
            double w  = std::abs(x[i]);
            double yv = w - c;
            double t  = sum + yv;
            c   = (t - sum) - yv;
            sum = t;
        }
        *result = sum;
        return CUBLAS_STATUS_SUCCESS;
    }
#endif
    double sum = 0.0, c = 0.0;
    for (int i = 0; i < n; ++i) {
        double w  = std::abs(x[i*incx]);
        double yv = w - c;
        double t  = sum + yv;
        c   = (t - sum) - yv;
        sum = t;
    }
    *result = sum;
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── ISAMAX / IDAMAX ───────────────────────────────────────────────────────────
cublasStatus_t cublasIsamax_v2(cublasHandle_t handle, int n,
    const float* x, int incx, int* result)
{
    if (!handle || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    // cblas_isamax returns 0-based index; cuBLAS returns 1-based
    *result = cblas_isamax(n, x, incx) + 1;
#else
    if (n <= 0) { *result = 0; return CUBLAS_STATUS_SUCCESS; }
    int idx = 0;
    float maxv = std::abs(x[0]);
    for (int i = 1; i < n; ++i) {
        float v = std::abs(x[i*incx]);
        if (v > maxv) { maxv = v; idx = i; }
    }
    *result = idx + 1; // 1-based
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIdamax_v2(cublasHandle_t handle, int n,
    const double* x, int incx, int* result)
{
    if (!handle || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    *result = cblas_idamax(n, x, incx) + 1;
#else
    if (n <= 0) { *result = 0; return CUBLAS_STATUS_SUCCESS; }
    int idx = 0;
    double maxv = std::abs(x[0]);
    for (int i = 1; i < n; ++i) {
        double v = std::abs(x[i*incx]);
        if (v > maxv) { maxv = v; idx = i; }
    }
    *result = idx + 1;
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── ISAMIN / IDAMIN ───────────────────────────────────────────────────────────
cublasStatus_t cublasIsamin_v2(cublasHandle_t handle, int n,
    const float* x, int incx, int* result)
{
    if (!handle || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    *result = cblas_isamin(n, x, incx) + 1;
#else
    if (n <= 0) { *result = 0; return CUBLAS_STATUS_SUCCESS; }
    int idx = 0;
    float minv = std::abs(x[0]);
    for (int i = 1; i < n; ++i) {
        float v = std::abs(x[i*incx]);
        if (v < minv) { minv = v; idx = i; }
    }
    *result = idx + 1;
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIdamin_v2(cublasHandle_t handle, int n,
    const double* x, int incx, int* result)
{
    if (!handle || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    *result = cblas_idamin(n, x, incx) + 1;
#else
    if (n <= 0) { *result = 0; return CUBLAS_STATUS_SUCCESS; }
    int idx = 0;
    double minv = std::abs(x[0]);
    for (int i = 1; i < n; ++i) {
        double v = std::abs(x[i*incx]);
        if (v < minv) { minv = v; idx = i; }
    }
    *result = idx + 1;
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SROT / DROT ───────────────────────────────────────────────────────────────
cublasStatus_t cublasSrot_v2(cublasHandle_t handle, int n,
    float* x, int incx, float* y, int incy,
    const float* c, const float* s)
{
    if (!handle || !x || !y || !c || !s) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_srot(n, x, incx, y, incy, *c, *s);
#else
    for (int i = 0; i < n; ++i) {
        float xi = x[i*incx];
        float yi = y[i*incy];
        x[i*incx] = (*c)*xi + (*s)*yi;
        y[i*incy] = -(*s)*xi + (*c)*yi;
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDrot_v2(cublasHandle_t handle, int n,
    double* x, int incx, double* y, int incy,
    const double* c, const double* s)
{
    if (!handle || !x || !y || !c || !s) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_drot(n, x, incx, y, incy, *c, *s);
#else
    for (int i = 0; i < n; ++i) {
        double xi = x[i*incx];
        double yi = y[i*incy];
        x[i*incx] = (*c)*xi + (*s)*yi;
        y[i*incy] = -(*s)*xi + (*c)*yi;
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SROTG / DROTG ─────────────────────────────────────────────────────────────
cublasStatus_t cublasSrotg_v2(cublasHandle_t handle, float* a, float* b, float* c_, float* s_)
{
    if (!handle || !a || !b || !c_ || !s_) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_srotg(a, b, c_, s_);
#else
    float aa = *a, bb = *b;
    float r, c, s;
    if (bb == 0.0f) {
        c = (aa >= 0.0f) ? 1.0f : -1.0f;
        s = 0.0f;
        r = std::abs(aa);
    } else if (aa == 0.0f) {
        c = 0.0f;
        s = (bb >= 0.0f) ? 1.0f : -1.0f;
        r = std::abs(bb);
    } else {
        float scale = std::abs(aa) + std::abs(bb);
        float rsa = aa / scale;
        float rsb = bb / scale;
        r = scale * std::sqrt(rsa*rsa + rsb*rsb);
        r = (aa < 0.0f) ? -r : r;
        c = aa / r;
        s = bb / r;
    }
    *c_ = c; *s_ = s; *a = r; *b = s;
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDrotg_v2(cublasHandle_t handle, double* a, double* b, double* c_, double* s_)
{
    if (!handle || !a || !b || !c_ || !s_) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_drotg(a, b, c_, s_);
#else
    double aa = *a, bb = *b;
    double r, c, s;
    if (bb == 0.0) {
        c = (aa >= 0.0) ? 1.0 : -1.0;
        s = 0.0;
        r = std::abs(aa);
    } else if (aa == 0.0) {
        c = 0.0;
        s = (bb >= 0.0) ? 1.0 : -1.0;
        r = std::abs(bb);
    } else {
        double scale = std::abs(aa) + std::abs(bb);
        double rsa = aa / scale;
        double rsb = bb / scale;
        r = scale * std::sqrt(rsa*rsa + rsb*rsb);
        r = (aa < 0.0) ? -r : r;
        c = aa / r;
        s = bb / r;
    }
    *c_ = c; *s_ = s; *a = r; *b = s;
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SROTM / DROTM ─────────────────────────────────────────────────────────────
// Apply a modified Givens rotation to vectors x and y.
// param[0] = flag: -2=identity, -1=full H, 0=h12=h21=1, 1=h11=h22=1
cublasStatus_t cublasSrotm_v2(cublasHandle_t handle, int n,
    float* x, int incx, float* y, int incy, const float* param)
{
    if (!handle || !x || !y || !param) return CUBLAS_STATUS_INVALID_VALUE;
    float flag = param[0];
    if (flag == -2.0f) return CUBLAS_STATUS_SUCCESS;
#if HAVE_CBLAS
    cblas_srotm(n, x, incx, y, incy, param);
#else
    float h11, h12, h21, h22;
    if (flag == -1.0f) {
        h11 = param[1]; h12 = param[3];
        h21 = param[2]; h22 = param[4];
    } else if (flag == 0.0f) {
        h11 = 1.0f; h12 = param[3];
        h21 = param[2]; h22 = 1.0f;
    } else { // flag == 1.0f
        h11 = param[1]; h12 = 1.0f;
        h21 = -1.0f; h22 = param[4];
    }
    for (int i = 0; i < n; ++i) {
        float xi = x[i*incx];
        float yi = y[i*incy];
        x[i*incx] = h11*xi + h12*yi;
        y[i*incy] = h21*xi + h22*yi;
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDrotm_v2(cublasHandle_t handle, int n,
    double* x, int incx, double* y, int incy, const double* param)
{
    if (!handle || !x || !y || !param) return CUBLAS_STATUS_INVALID_VALUE;
    double flag = param[0];
    if (flag == -2.0) return CUBLAS_STATUS_SUCCESS;
#if HAVE_CBLAS
    cblas_drotm(n, x, incx, y, incy, param);
#else
    double h11, h12, h21, h22;
    if (flag == -1.0) {
        h11 = param[1]; h12 = param[3];
        h21 = param[2]; h22 = param[4];
    } else if (flag == 0.0) {
        h11 = 1.0; h12 = param[3];
        h21 = param[2]; h22 = 1.0;
    } else {
        h11 = param[1]; h12 = 1.0;
        h21 = -1.0; h22 = param[4];
    }
    for (int i = 0; i < n; ++i) {
        double xi = x[i*incx];
        double yi = y[i*incy];
        x[i*incx] = h11*xi + h12*yi;
        y[i*incy] = h21*xi + h22*yi;
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SROTMG / DROTMG ───────────────────────────────────────────────────────────
// Generate the parameters for a modified Givens rotation.
cublasStatus_t cublasSrotmg_v2(cublasHandle_t handle,
    float* d1, float* d2, float* x1, float* y1, float* param)
{
    if (!handle || !d1 || !d2 || !x1 || !y1 || !param) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_srotmg(d1, d2, x1, y1, param);
#else
    float flag = -2.0f, h11 = 0.0f, h12 = 0.0f, h21 = 0.0f, h22 = 0.0f;
    if (*d1 <= 0.0f) {
        flag = -1.0f;
        *d1 = *d2 = *x1 = 0.0f;
    } else {
        float p2 = *d2 * *y1;
        if (p2 == 0.0f) {
            param[0] = -2.0f; param[1] = param[2] = param[3] = param[4] = 0.0f;
            return CUBLAS_STATUS_SUCCESS;
        }
        float q2 = p2 * *y1;
        float q1 = (*d1 * *x1) * *x1;
        if (std::abs(q1) > std::abs(q2)) {
            float q = *y1 / *x1;
            float p = q * *d2 / *d1;
            float u = 1.0f - p;
            if (u > 0.0f) {
                flag = 0.0f;
                *d1 /= u; *d2 /= u; *x1 *= u;
                h11 = 1.0f; h22 = 1.0f;
                h21 = -p; h12 = q;
            }
        } else {
            float q = *x1 / *y1;
            float p = q * *d1 / *d2;
            float u = 1.0f - p;
            if (u > 0.0f) {
                flag = 1.0f;
                *d1 /= u; *d2 /= u; *x1 *= u;
                h11 = p; h22 = q;
                h21 = -1.0f; h12 = 1.0f;
            }
        }
    }
    param[0] = flag;
    if (flag == -2.0f) {
        param[1] = param[2] = param[3] = param[4] = 0.0f;
    } else if (flag == -1.0f) {
        param[1] = h11; param[2] = h21; param[3] = h12; param[4] = h22;
    } else if (flag == 0.0f) {
        param[1] = 1.0f; param[2] = h21; param[3] = h12; param[4] = 1.0f;
    } else {
        param[1] = h11; param[2] = 1.0f; param[3] = 1.0f; param[4] = h22;
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDrotmg_v2(cublasHandle_t handle,
    double* d1, double* d2, double* x1, double* y1, double* param)
{
    if (!handle || !d1 || !d2 || !x1 || !y1 || !param) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_drotmg(d1, d2, x1, y1, param);
#else
    double flag = -2.0, h11 = 0.0, h12 = 0.0, h21 = 0.0, h22 = 0.0;
    if (*d1 <= 0.0) {
        flag = -1.0;
        *d1 = *d2 = *x1 = 0.0;
    } else {
        double p2 = *d2 * *y1;
        if (p2 == 0.0) {
            param[0] = -2.0; param[1] = param[2] = param[3] = param[4] = 0.0;
            return CUBLAS_STATUS_SUCCESS;
        }
        double q2 = p2 * *y1;
        double q1 = (*d1 * *x1) * *x1;
        if (std::abs(q1) > std::abs(q2)) {
            double q = *y1 / *x1;
            double p = q * *d2 / *d1;
            double u = 1.0 - p;
            if (u > 0.0) {
                flag = 0.0;
                *d1 /= u; *d2 /= u; *x1 *= u;
                h11 = 1.0; h22 = 1.0;
                h21 = -p; h12 = q;
            }
        } else {
            double q = *x1 / *y1;
            double p = q * *d1 / *d2;
            double u = 1.0 - p;
            if (u > 0.0) {
                flag = 1.0;
                *d1 /= u; *d2 /= u; *x1 *= u;
                h11 = p; h22 = q;
                h21 = -1.0; h12 = 1.0;
            }
        }
    }
    param[0] = flag;
    if (flag == -2.0) {
        param[1] = param[2] = param[3] = param[4] = 0.0;
    } else if (flag == -1.0) {
        param[1] = h11; param[2] = h21; param[3] = h12; param[4] = h22;
    } else if (flag == 0.0) {
        param[1] = 1.0; param[2] = h21; param[3] = h12; param[4] = 1.0;
    } else {
        param[1] = h11; param[2] = 1.0; param[3] = 1.0; param[4] = h22;
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}


} // extern "C"
