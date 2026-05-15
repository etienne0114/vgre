#include "cublas_internal.h"

extern "C" {

// ─── Packed / Banded Level-2 BLAS ─────────────────────────────────────────────
//
// These routines operate on packed (SP/DP) and banded (SB/DB) storage formats.
// Reference scalar loops are used; CBLAS is avoided because not all implementations
// expose packed/banded interfaces reliably.
//
// Packed symmetric matrix storage: element A[i,j] (j >= i for upper, j <= i for lower)
// is stored at index i*n - i*(i-1)/2 + j  (upper) or j*(2n-j-1)/2 + i  (lower).
//
// Banded storage: A[i,j] is stored at AB[ku + i - j][j] where ku = upper bandwidth.

// ── STBSV / DTBSV — triangular banded solve ───────────────────────────────────
cublasStatus_t cublasStbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
    const float *A, int lda, float *x, int incx)
{
    if (!handle || !A || !x || n <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool unit  = (diag == CUBLAS_DIAG_UNIT);
    bool notrans = (trans == CUBLAS_OP_N);
    if (notrans) {
        if (upper) {
            for (int i = n - 1; i >= 0; --i) {
                if (!unit) {
                    int diag_band = k;
                    if (i < lda) x[i*incx] /= A[diag_band * lda + i];
                }
                for (int j = std::max(0, i - k); j < i; ++j) {
                    int band = k + j - i;
                    if (band >= 0 && band < lda)
                        x[j*incx] -= A[band * lda + i] * x[i*incx];
                }
            }
        } else {
            for (int i = 0; i < n; ++i) {
                if (!unit) {
                    if (0 < lda) x[i*incx] /= A[0 * lda + i];
                }
                for (int j = i + 1; j < std::min(n, i + k + 1); ++j) {
                    int band = j - i;
                    if (band < lda)
                        x[j*incx] -= A[band * lda + i] * x[i*incx];
                }
            }
        }
    } else {
        // Transposed case — reverse traversal
        if (upper) {
            for (int i = 0; i < n; ++i) {
                if (!unit) {
                    int diag_band = k;
                    if (diag_band < lda) x[i*incx] /= A[diag_band * lda + i];
                }
                for (int j = i + 1; j < std::min(n, i + k + 1); ++j) {
                    int band = k + i - j;
                    if (band >= 0 && band < lda)
                        x[j*incx] -= A[band * lda + j] * x[i*incx];
                }
            }
        } else {
            for (int i = n - 1; i >= 0; --i) {
                if (!unit) {
                    if (0 < lda) x[i*incx] /= A[0 * lda + i];
                }
                for (int j = std::max(0, i - k); j < i; ++j) {
                    int band = i - j;
                    if (band < lda)
                        x[j*incx] -= A[band * lda + j] * x[i*incx];
                }
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtbsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
    const double *A, int lda, double *x, int incx)
{
    if (!handle || !A || !x || n <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool unit  = (diag == CUBLAS_DIAG_UNIT);
    bool notrans = (trans == CUBLAS_OP_N);
    if (notrans) {
        if (upper) {
            for (int i = n - 1; i >= 0; --i) {
                if (!unit && k < lda) x[i*incx] /= A[k * lda + i];
                for (int j = std::max(0, i - k); j < i; ++j) {
                    int b = k + j - i; if (b >= 0 && b < lda) x[j*incx] -= A[b*lda+i]*x[i*incx];
                }
            }
        } else {
            for (int i = 0; i < n; ++i) {
                if (!unit && 0 < lda) x[i*incx] /= A[0*lda+i];
                for (int j = i+1; j < std::min(n, i+k+1); ++j) {
                    int b = j - i; if (b < lda) x[j*incx] -= A[b*lda+i]*x[i*incx];
                }
            }
        }
    } else {
        if (upper) {
            for (int i = 0; i < n; ++i) {
                if (!unit && k < lda) x[i*incx] /= A[k*lda+i];
                for (int j = i+1; j < std::min(n, i+k+1); ++j) {
                    int b = k+i-j; if (b>=0&&b<lda) x[j*incx] -= A[b*lda+j]*x[i*incx];
                }
            }
        } else {
            for (int i = n-1; i >= 0; --i) {
                if (!unit && 0<lda) x[i*incx] /= A[0*lda+i];
                for (int j = std::max(0,i-k); j < i; ++j) {
                    int b = i-j; if (b<lda) x[j*incx] -= A[b*lda+j]*x[i*incx];
                }
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStbsv(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,
    cublasDiagType_t d, int n, int k, const float *A, int lda, float *x, int incx)
{ return cublasStbsv_v2(h,u,t,d,n,k,A,lda,x,incx); }
cublasStatus_t cublasDtbsv(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,
    cublasDiagType_t d, int n, int k, const double *A, int lda, double *x, int incx)
{ return cublasDtbsv_v2(h,u,t,d,n,k,A,lda,x,incx); }

// ── STPSV / DTPSV — triangular packed solve ────────────────────────────────────
// Packed upper: AP[i*(i+1)/2 + j] = A[j,i] for j<=i
// Packed lower: AP[i*(2n-i-1)/2 + j] = A[j,i] for j>=i

static inline int packIdxUpper(int i, int j) { return j*(j+1)/2 + i; } // j>=i upper col-maj
static inline int packIdxLower(int n, int i, int j) { return j*(2*n-j-1)/2 + i; } // j<=i lower

cublasStatus_t cublasStpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int n,
    const float *AP, float *x, int incx)
{
    if (!handle || !AP || !x || n <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper  = (uplo  == CUBLAS_FILL_MODE_UPPER);
    bool unit   = (diag  == CUBLAS_DIAG_UNIT);
    bool notrans = (trans == CUBLAS_OP_N);
    if (upper) {
        if (notrans) {
            for (int i = n-1; i >= 0; --i) {
                if (!unit) x[i*incx] /= AP[packIdxUpper(i,i)];
                for (int j = 0; j < i; ++j)
                    x[j*incx] -= AP[packIdxUpper(j,i)] * x[i*incx];
            }
        } else {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < i; ++j)
                    x[i*incx] -= AP[packIdxUpper(j,i)] * x[j*incx];
                if (!unit) x[i*incx] /= AP[packIdxUpper(i,i)];
            }
        }
    } else {
        if (notrans) {
            for (int i = 0; i < n; ++i) {
                if (!unit) x[i*incx] /= AP[packIdxLower(n,i,i)];
                for (int j = i+1; j < n; ++j)
                    x[j*incx] -= AP[packIdxLower(n,j,i)] * x[i*incx];
            }
        } else {
            for (int i = n-1; i >= 0; --i) {
                for (int j = i+1; j < n; ++j)
                    x[i*incx] -= AP[packIdxLower(n,j,i)] * x[j*incx];
                if (!unit) x[i*incx] /= AP[packIdxLower(n,i,i)];
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtpsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int n,
    const double *AP, double *x, int incx)
{
    if (!handle || !AP || !x || n <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo==CUBLAS_FILL_MODE_UPPER), unit=(diag==CUBLAS_DIAG_UNIT);
    bool notrans=(trans==CUBLAS_OP_N);
    if (upper) {
        if (notrans) {
            for (int i=n-1;i>=0;--i) {
                if (!unit) x[i*incx]/=AP[packIdxUpper(i,i)];
                for (int j=0;j<i;++j) x[j*incx]-=AP[packIdxUpper(j,i)]*x[i*incx];
            }
        } else {
            for (int i=0;i<n;++i) {
                for (int j=0;j<i;++j) x[i*incx]-=AP[packIdxUpper(j,i)]*x[j*incx];
                if (!unit) x[i*incx]/=AP[packIdxUpper(i,i)];
            }
        }
    } else {
        if (notrans) {
            for (int i=0;i<n;++i) {
                if (!unit) x[i*incx]/=AP[packIdxLower(n,i,i)];
                for (int j=i+1;j<n;++j) x[j*incx]-=AP[packIdxLower(n,j,i)]*x[i*incx];
            }
        } else {
            for (int i=n-1;i>=0;--i) {
                for (int j=i+1;j<n;++j) x[i*incx]-=AP[packIdxLower(n,j,i)]*x[j*incx];
                if (!unit) x[i*incx]/=AP[packIdxLower(n,i,i)];
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStpsv(cublasHandle_t h,cublasFillMode_t u,cublasOperation_t t,
    cublasDiagType_t d,int n,const float*AP,float*x,int incx)
{ return cublasStpsv_v2(h,u,t,d,n,AP,x,incx); }
cublasStatus_t cublasDtpsv(cublasHandle_t h,cublasFillMode_t u,cublasOperation_t t,
    cublasDiagType_t d,int n,const double*AP,double*x,int incx)
{ return cublasDtpsv_v2(h,u,t,d,n,AP,x,incx); }

// ── SSPMV / DSPMV — symmetric packed matrix-vector multiply ───────────────────

cublasStatus_t cublasSspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    int n, const float *alpha, const float *AP,
    const float *x, int incx, const float *beta, float *y, int incy)
{
    if (!handle||!AP||!x||!y||!alpha||!beta||n<=0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper=(uplo==CUBLAS_FILL_MODE_UPPER);
    for (int i=0;i<n;++i) y[i*incy] = (*beta)*y[i*incy];
    for (int i=0;i<n;++i) {
        float xi = (*alpha)*x[i*incx];
        for (int j=0;j<n;++j) {
            int a_idx = upper ? (j>=i ? packIdxUpper(i,j) : packIdxUpper(j,i))
                              : (j<=i ? packIdxLower(n,i,j) : packIdxLower(n,j,i));
            y[i*incy] += AP[a_idx] * (*alpha)*x[j*incx];
            (void)xi;
        }
    }
    // Recompute correctly: y = alpha*A*x + beta*y
    for (int i=0;i<n;++i) y[i*incy] = (*beta)*y[i*incy];
    for (int i=0;i<n;++i) {
        float acc=0.f;
        for (int j=0;j<n;++j) {
            int a_idx = upper ? (j>=i ? packIdxUpper(i,j) : packIdxUpper(j,i))
                              : (j<=i ? packIdxLower(n,i,j) : packIdxLower(n,j,i));
            acc += AP[a_idx]*x[j*incx];
        }
        y[i*incy] += (*alpha)*acc;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDspmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    int n, const double *alpha, const double *AP,
    const double *x, int incx, const double *beta, double *y, int incy)
{
    if (!handle||!AP||!x||!y||!alpha||!beta||n<=0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper=(uplo==CUBLAS_FILL_MODE_UPPER);
    for (int i=0;i<n;++i) {
        double acc=0.0;
        for (int j=0;j<n;++j) {
            int a_idx = upper ? (j>=i ? packIdxUpper(i,j) : packIdxUpper(j,i))
                              : (j<=i ? packIdxLower(n,i,j) : packIdxLower(n,j,i));
            acc += AP[a_idx]*x[j*incx];
        }
        y[i*incy] = (*alpha)*acc + (*beta)*y[i*incy];
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSspmv(cublasHandle_t h,cublasFillMode_t u,int n,
    const float*a,const float*AP,const float*x,int incx,const float*b,float*y,int incy)
{ return cublasSspmv_v2(h,u,n,a,AP,x,incx,b,y,incy); }
cublasStatus_t cublasDspmv(cublasHandle_t h,cublasFillMode_t u,int n,
    const double*a,const double*AP,const double*x,int incx,const double*b,double*y,int incy)
{ return cublasDspmv_v2(h,u,n,a,AP,x,incx,b,y,incy); }

// ── SSBMV / DSBMV — symmetric banded matrix-vector multiply ───────────────────
// Storage: A stored in AB such that AB[ku + i - j][j] = A[i,j], ku = k.

cublasStatus_t cublasSsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    int n, int k, const float *alpha, const float *A, int lda,
    const float *x, int incx, const float *beta, float *y, int incy)
{
    if (!handle||!A||!x||!y||!alpha||!beta||n<=0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper=(uplo==CUBLAS_FILL_MODE_UPPER);
    for (int i=0;i<n;++i) y[i*incy] = (*beta)*y[i*incy];
    for (int j=0;j<n;++j) {
        float xj = (*alpha)*x[j*incx];
        int ibegin = upper ? std::max(0, j-k) : j;
        int iend   = upper ? j : std::min(n-1, j+k);
        for (int i=ibegin;i<=iend;++i) {
            int band = upper ? (k + i - j) : (i - j);
            if (band>=0 && band<lda) {
                y[i*incy] += A[band*lda+j] * xj;
                if (i != j) y[j*incy] += A[band*lda+j] * (*alpha)*x[i*incx];
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    int n, int k, const double *alpha, const double *A, int lda,
    const double *x, int incx, const double *beta, double *y, int incy)
{
    if (!handle||!A||!x||!y||!alpha||!beta||n<=0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper=(uplo==CUBLAS_FILL_MODE_UPPER);
    for (int i=0;i<n;++i) y[i*incy] = (*beta)*y[i*incy];
    for (int j=0;j<n;++j) {
        double xj = (*alpha)*x[j*incx];
        int ibegin = upper ? std::max(0,j-k) : j;
        int iend   = upper ? j : std::min(n-1,j+k);
        for (int i=ibegin;i<=iend;++i) {
            int band = upper ? (k+i-j) : (i-j);
            if (band>=0&&band<lda) {
                y[i*incy] += A[band*lda+j]*xj;
                if (i!=j) y[j*incy] += A[band*lda+j]*(*alpha)*x[i*incx];
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSsbmv(cublasHandle_t h,cublasFillMode_t u,int n,int k,
    const float*a,const float*A,int lda,const float*x,int incx,const float*b,float*y,int incy)
{ return cublasSsbmv_v2(h,u,n,k,a,A,lda,x,incx,b,y,incy); }
cublasStatus_t cublasDsbmv(cublasHandle_t h,cublasFillMode_t u,int n,int k,
    const double*a,const double*A,int lda,const double*x,int incx,const double*b,double*y,int incy)
{ return cublasDsbmv_v2(h,u,n,k,a,A,lda,x,incx,b,y,incy); }

// ── SSPR / DSPR — symmetric packed rank-1 update ──────────────────────────────
// A += alpha * x * x^T  (A in packed storage)

cublasStatus_t cublasSspr_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    int n, const float *alpha, const float *x, int incx, float *AP)
{
    if (!handle||!AP||!x||!alpha||n<=0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper=(uplo==CUBLAS_FILL_MODE_UPPER);
    for (int i=0;i<n;++i)
        for (int j=(upper?i:0); j<=(upper?n-1:i); ++j) {
            int idx = upper ? packIdxUpper(i,j) : packIdxLower(n,j,i);
            AP[idx] += (*alpha)*x[i*incx]*x[j*incx];
        }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDspr_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    int n, const double *alpha, const double *x, int incx, double *AP)
{
    if (!handle||!AP||!x||!alpha||n<=0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper=(uplo==CUBLAS_FILL_MODE_UPPER);
    for (int i=0;i<n;++i)
        for (int j=(upper?i:0);j<=(upper?n-1:i);++j) {
            int idx = upper ? packIdxUpper(i,j) : packIdxLower(n,j,i);
            AP[idx] += (*alpha)*x[i*incx]*x[j*incx];
        }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSspr(cublasHandle_t h,cublasFillMode_t u,int n,
    const float*a,const float*x,int incx,float*AP)
{ return cublasSspr_v2(h,u,n,a,x,incx,AP); }
cublasStatus_t cublasDspr(cublasHandle_t h,cublasFillMode_t u,int n,
    const double*a,const double*x,int incx,double*AP)
{ return cublasDspr_v2(h,u,n,a,x,incx,AP); }

// ── SSPR2 / DSPR2 — symmetric packed rank-2 update ────────────────────────────
// A += alpha*x*y^T + alpha*y*x^T

cublasStatus_t cublasSspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    int n, const float *alpha, const float *x, int incx,
    const float *y, int incy, float *AP)
{
    if (!handle||!AP||!x||!y||!alpha||n<=0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper=(uplo==CUBLAS_FILL_MODE_UPPER);
    for (int i=0;i<n;++i)
        for (int j=(upper?i:0);j<=(upper?n-1:i);++j) {
            int idx = upper ? packIdxUpper(i,j) : packIdxLower(n,j,i);
            AP[idx] += (*alpha)*(x[i*incx]*y[j*incy] + y[i*incy]*x[j*incx]);
        }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDspr2_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    int n, const double *alpha, const double *x, int incx,
    const double *y, int incy, double *AP)
{
    if (!handle||!AP||!x||!y||!alpha||n<=0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper=(uplo==CUBLAS_FILL_MODE_UPPER);
    for (int i=0;i<n;++i)
        for (int j=(upper?i:0);j<=(upper?n-1:i);++j) {
            int idx = upper ? packIdxUpper(i,j) : packIdxLower(n,j,i);
            AP[idx] += (*alpha)*(x[i*incx]*y[j*incy]+y[i*incy]*x[j*incx]);
        }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSspr2(cublasHandle_t h,cublasFillMode_t u,int n,
    const float*a,const float*x,int incx,const float*y,int incy,float*AP)
{ return cublasSspr2_v2(h,u,n,a,x,incx,y,incy,AP); }
cublasStatus_t cublasDspr2(cublasHandle_t h,cublasFillMode_t u,int n,
    const double*a,const double*x,int incx,const double*y,int incy,double*AP)
{ return cublasDspr2_v2(h,u,n,a,x,incx,y,incy,AP); }

// ── STBMV / DTBMV — triangular banded matrix-vector multiply ──────────────────

cublasStatus_t cublasStbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
    const float *A, int lda, float *x, int incx)
{
    if (!handle||!A||!x||n<=0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper=(uplo==CUBLAS_FILL_MODE_UPPER), unit=(diag==CUBLAS_DIAG_UNIT);
    bool notrans=(trans==CUBLAS_OP_N);
    std::vector<float> tmp(n);
    for (int i=0;i<n;++i) tmp[i]=x[i*incx];
    for (int i=0;i<n;++i) {
        float acc=0.f;
        if (notrans) {
            if (!unit) { int b=upper?k:0; if(b<lda) acc+=A[b*lda+i]*tmp[i]; else acc+=tmp[i]; }
            else acc=tmp[i];
            int jbegin=upper?std::max(0,i-k):i+1;
            int jend  =upper?i-1:std::min(n-1,i+k);
            for (int j=jbegin;j<=jend;++j) {
                int band=upper?(k+j-i):(j-i); if(band>=0&&band<lda) acc+=A[band*lda+i]*tmp[j]; // off-diagonal
            }
            // Fix: compute correctly
            acc=0.f;
            if (!unit) { int b=upper?k:0; if(b<lda) acc+=A[b*lda+i]*tmp[i]; else acc+=tmp[i]; }
            else acc=tmp[i];
            for (int j=(upper?std::max(0,i-k):i+1); j<=(upper?i-1:std::min(n-1,i+k)); ++j) {
                int band=upper?(k+j-i):(j-i); if(band>=0&&band<lda) acc+=A[band*lda+i]*tmp[j];
            }
        } else {
            acc = unit ? tmp[i] : 0.f;
            if (!unit) { int b=upper?k:0; if(b<lda) acc=A[b*lda+i]*tmp[i]; }
            for (int j=(upper?i+1:std::max(0,i-k)); j<=(upper?std::min(n-1,i+k):i-1); ++j) {
                int band=upper?(k+i-j):(i-j); if(band>=0&&band<lda) acc+=A[band*lda+j]*tmp[j];
            }
        }
        x[i*incx]=acc;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtbmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int n, int k,
    const double *A, int lda, double *x, int incx)
{
    if (!handle||!A||!x||n<=0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper=(uplo==CUBLAS_FILL_MODE_UPPER), unit=(diag==CUBLAS_DIAG_UNIT);
    bool notrans=(trans==CUBLAS_OP_N);
    std::vector<double> tmp(n);
    for (int i=0;i<n;++i) tmp[i]=x[i*incx];
    for (int i=0;i<n;++i) {
        double acc=unit?tmp[i]:0.0;
        if (!unit) { int b=upper?k:0; if(b<lda) acc=A[b*lda+i]*tmp[i]; }
        if (notrans) {
            for (int j=(upper?std::max(0,i-k):i+1);j<=(upper?i-1:std::min(n-1,i+k));++j) {
                int band=upper?(k+j-i):(j-i); if(band>=0&&band<lda) acc+=A[band*lda+i]*tmp[j];
            }
        } else {
            for (int j=(upper?i+1:std::max(0,i-k));j<=(upper?std::min(n-1,i+k):i-1);++j) {
                int band=upper?(k+i-j):(i-j); if(band>=0&&band<lda) acc+=A[band*lda+j]*tmp[j];
            }
        }
        x[i*incx]=acc;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStbmv(cublasHandle_t h,cublasFillMode_t u,cublasOperation_t t,
    cublasDiagType_t d,int n,int k,const float*A,int lda,float*x,int incx)
{ return cublasStbmv_v2(h,u,t,d,n,k,A,lda,x,incx); }
cublasStatus_t cublasDtbmv(cublasHandle_t h,cublasFillMode_t u,cublasOperation_t t,
    cublasDiagType_t d,int n,int k,const double*A,int lda,double*x,int incx)
{ return cublasDtbmv_v2(h,u,t,d,n,k,A,lda,x,incx); }

// ── STPMV / DTPMV — triangular packed matrix-vector multiply ──────────────────

cublasStatus_t cublasStpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int n,
    const float *AP, float *x, int incx)
{
    if (!handle||!AP||!x||n<=0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper=(uplo==CUBLAS_FILL_MODE_UPPER), unit=(diag==CUBLAS_DIAG_UNIT);
    bool notrans=(trans==CUBLAS_OP_N);
    std::vector<float> tmp(n);
    for (int i=0;i<n;++i) tmp[i]=x[i*incx];
    if (upper) {
        if (notrans) {
            for (int i=0;i<n;++i) {
                float acc=unit?tmp[i]:AP[packIdxUpper(i,i)]*tmp[i];
                for (int j=i+1;j<n;++j) acc+=AP[packIdxUpper(i,j)]*tmp[j];
                x[i*incx]=acc;
            }
        } else {
            for (int i=n-1;i>=0;--i) {
                float acc=unit?tmp[i]:AP[packIdxUpper(i,i)]*tmp[i];
                for (int j=0;j<i;++j) acc+=AP[packIdxUpper(j,i)]*tmp[j];
                x[i*incx]=acc;
            }
        }
    } else {
        if (notrans) {
            for (int i=n-1;i>=0;--i) {
                float acc=unit?tmp[i]:AP[packIdxLower(n,i,i)]*tmp[i];
                for (int j=0;j<i;++j) acc+=AP[packIdxLower(n,i,j)]*tmp[j];
                x[i*incx]=acc;
            }
        } else {
            for (int i=0;i<n;++i) {
                float acc=unit?tmp[i]:AP[packIdxLower(n,i,i)]*tmp[i];
                for (int j=i+1;j<n;++j) acc+=AP[packIdxLower(n,j,i)]*tmp[j];
                x[i*incx]=acc;
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtpmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int n,
    const double *AP, double *x, int incx)
{
    if (!handle||!AP||!x||n<=0) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper=(uplo==CUBLAS_FILL_MODE_UPPER), unit=(diag==CUBLAS_DIAG_UNIT);
    bool notrans=(trans==CUBLAS_OP_N);
    std::vector<double> tmp(n);
    for (int i=0;i<n;++i) tmp[i]=x[i*incx];
    if (upper) {
        if (notrans) {
            for (int i=0;i<n;++i) {
                double acc=unit?tmp[i]:AP[packIdxUpper(i,i)]*tmp[i];
                for (int j=i+1;j<n;++j) acc+=AP[packIdxUpper(i,j)]*tmp[j];
                x[i*incx]=acc;
            }
        } else {
            for (int i=n-1;i>=0;--i) {
                double acc=unit?tmp[i]:AP[packIdxUpper(i,i)]*tmp[i];
                for (int j=0;j<i;++j) acc+=AP[packIdxUpper(j,i)]*tmp[j];
                x[i*incx]=acc;
            }
        }
    } else {
        if (notrans) {
            for (int i=n-1;i>=0;--i) {
                double acc=unit?tmp[i]:AP[packIdxLower(n,i,i)]*tmp[i];
                for (int j=0;j<i;++j) acc+=AP[packIdxLower(n,i,j)]*tmp[j];
                x[i*incx]=acc;
            }
        } else {
            for (int i=0;i<n;++i) {
                double acc=unit?tmp[i]:AP[packIdxLower(n,i,i)]*tmp[i];
                for (int j=i+1;j<n;++j) acc+=AP[packIdxLower(n,j,i)]*tmp[j];
                x[i*incx]=acc;
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasStpmv(cublasHandle_t h,cublasFillMode_t u,cublasOperation_t t,
    cublasDiagType_t d,int n,const float*AP,float*x,int incx)
{ return cublasStpmv_v2(h,u,t,d,n,AP,x,incx); }
cublasStatus_t cublasDtpmv(cublasHandle_t h,cublasFillMode_t u,cublasOperation_t t,
    cublasDiagType_t d,int n,const double*AP,double*x,int incx)
{ return cublasDtpmv_v2(h,u,t,d,n,AP,x,incx); }

} // extern "C"
