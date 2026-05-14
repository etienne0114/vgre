// cuBLAS level1 API functions

#include "cublas_internal.h"

extern "C" {

// ── SAXPY / DAXPY ────────────────────────────────────────────────────────────
cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n,
    const float* alpha, const float* x, int incx, float* y, int incy)
{
    if (!handle || !x || !y || !alpha) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_saxpy(n, *alpha, x, incx, y, incy);
#else
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
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += x[i*incx] * y[i*incy];
    *result = s;
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
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i*incx] * y[i*incy];
    *result = s;
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
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += x[i*incx] * x[i*incx];
    *result = sqrtf(s);
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
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += std::abs(x[i*incx]);
    *result = s;
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
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::abs(x[i*incx]);
    *result = s;
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
