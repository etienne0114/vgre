// cuBLAS internal shared types, enums, structs, and reference helpers.
#pragma once

#include "vgre/common/logger.h"
#include "vgre/common/error_codes.h"
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

#if defined(VGRE_HAS_CBLAS)
#  include <cblas.h>
#  define HAVE_CBLAS 1
#else
#  define HAVE_CBLAS 0
#endif

// ── cuBLAS type aliases (avoid including cublas_v2.h) ─────────────────────────
typedef void*    cublasHandle_t;
typedef int      cublasStatus_t;
typedef int      cublasOperation_t;
typedef int      cublasFillMode_t;
typedef int      cublasDiagType_t;
typedef int      cublasSideMode_t;

static constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS         = 0;
static constexpr cublasStatus_t CUBLAS_STATUS_NOT_INITIALIZED = 1;
static constexpr cublasStatus_t CUBLAS_STATUS_ALLOC_FAILED    = 3;
static constexpr cublasStatus_t CUBLAS_STATUS_INVALID_VALUE   = 7;
static constexpr cublasStatus_t CUBLAS_STATUS_EXECUTION_FAILED= 13;
static constexpr cublasStatus_t CUBLAS_STATUS_NOT_SUPPORTED   = 15;
static constexpr cublasStatus_t CUBLAS_STATUS_INTERNAL_ERROR  = 14;

static constexpr cublasOperation_t CUBLAS_OP_N = 0;  // no transpose
static constexpr cublasOperation_t CUBLAS_OP_T = 1;  // transpose
static constexpr cublasOperation_t CUBLAS_OP_C = 2;  // conjugate transpose

// ── cuComplex types (minimal for Hermitian BLAS) ────────────────────────────
struct cuComplex { float x, y; };
struct cuDoubleComplex { double x, y; };

static inline cuComplex make_cuComplex(float x, float y) { return {x, y}; }
static inline cuDoubleComplex make_cuDoubleComplex(double x, double y) { return {x, y}; }

static inline cuComplex cuConjf(cuComplex a) { return {a.x, -a.y}; }
static inline cuDoubleComplex cuConj(cuDoubleComplex a) { return {a.x, -a.y}; }

static inline float cuCrealf(cuComplex a) { return a.x; }
static inline float cuCimagf(cuComplex a) { return a.y; }
static inline double cuCreal(cuDoubleComplex a) { return a.x; }
static inline double cuCimag(cuDoubleComplex a) { return a.y; }

static inline cuComplex cuCaddf(cuComplex a, cuComplex b) { return {a.x + b.x, a.y + b.y}; }
static inline cuComplex cuCmulf(cuComplex a, cuComplex b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
static inline cuDoubleComplex cuCadd(cuDoubleComplex a, cuDoubleComplex b) { return {a.x + b.x, a.y + b.y}; }
static inline cuDoubleComplex cuCmul(cuDoubleComplex a, cuDoubleComplex b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
static inline cuComplex cuCmulf_real(cuComplex a, float s) { return {a.x * s, a.y * s}; }
static inline cuDoubleComplex cuCmul_real(cuDoubleComplex a, double s) { return {a.x * s, a.y * s}; }

static constexpr cublasFillMode_t CUBLAS_FILL_MODE_LOWER = 0;
static constexpr cublasFillMode_t CUBLAS_FILL_MODE_UPPER = 1;
static constexpr cublasDiagType_t CUBLAS_DIAG_NON_UNIT = 0;
static constexpr cublasDiagType_t CUBLAS_DIAG_UNIT = 1;
static constexpr cublasSideMode_t CUBLAS_SIDE_LEFT = 0;
static constexpr cublasSideMode_t CUBLAS_SIDE_RIGHT = 1;

// Real context: carries stream binding and math mode so all operations on a
// handle run consistently.  stream==nullptr means use the default/null stream.
enum cublasMath_t { CUBLAS_DEFAULT_MATH = 0, CUBLAS_TENSOR_OP_MATH = 1, CUBLAS_PEDANTIC_MATH = 2 };
enum cublasPointerMode_t { CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1 };
enum cublasAtomicsMode_t { CUBLAS_ATOMICS_NOT_ALLOWED = 0, CUBLAS_ATOMICS_ALLOWED = 1 };
struct cublasContext {
    void*              stream      = nullptr;
    cublasMath_t       mathMode    = CUBLAS_DEFAULT_MATH;
    cublasPointerMode_t pointerMode = CUBLAS_POINTER_MODE_HOST;
    cublasAtomicsMode_t atomicsMode = CUBLAS_ATOMICS_ALLOWED;
    int                deviceId    = 0;
};

#if HAVE_CBLAS
static CBLAS_UPLO cublasToCblasUplo(cublasFillMode_t uplo) {
    return (uplo == CUBLAS_FILL_MODE_UPPER) ? CblasUpper : CblasLower;
}
static CBLAS_DIAG cublasToCblasDiag(cublasDiagType_t diag) {
    return (diag == CUBLAS_DIAG_NON_UNIT) ? CblasNonUnit : CblasUnit;
}
static CBLAS_SIDE cublasToCblasSide(cublasSideMode_t side) {
    return (side == CUBLAS_SIDE_LEFT) ? CblasLeft : CblasRight;
}
#endif

// ── Cache-blocked reference GEMM ─────────────────────────────────────────────
// C = alpha*op(A)*op(B) + beta*C  (row-major)
// Tile size chosen to fit two 64×64 float tiles (~32 KiB) within a typical
// 32–64 KiB L1 D-cache.  Falls back to a scalar loop for small matrices.
static constexpr int kTile = 64;

extern void refSgemm(bool tA, bool tB,
    int M, int N, int K,
    float alpha, const float* A, int lda,
                 const float* B, int ldb,
    float beta,        float* C, int ldc);

extern void refDgemm(bool tA, bool tB,
    int M, int N, int K,
    double alpha, const double* A, int lda,
                  const double* B, int ldb,
    double beta,        double* C, int ldc);

// ── Reference helpers for Level-2 BLAS ─────────────────────────────────────

static void refStrsv(bool upper, bool trans, bool unit,
    int n, const float* A, int lda, float* x, int incx)
{
    if (trans) {
        if (upper) {
            for (int i = 0; i < n; ++i) {
                float t = x[i*incx];
                for (int j = 0; j < i; ++j) t -= A[j*lda+i] * x[j*incx];
                if (!unit) t /= A[i*lda+i];
                x[i*incx] = t;
            }
        } else {
            for (int i = n-1; i >= 0; --i) {
                float t = x[i*incx];
                for (int j = i+1; j < n; ++j) t -= A[j*lda+i] * x[j*incx];
                if (!unit) t /= A[i*lda+i];
                x[i*incx] = t;
            }
        }
    } else {
        if (upper) {
            for (int i = n-1; i >= 0; --i) {
                float t = x[i*incx];
                for (int j = i+1; j < n; ++j) t -= A[i*lda+j] * x[j*incx];
                if (!unit) t /= A[i*lda+i];
                x[i*incx] = t;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                float t = x[i*incx];
                for (int j = 0; j < i; ++j) t -= A[i*lda+j] * x[j*incx];
                if (!unit) t /= A[i*lda+i];
                x[i*incx] = t;
            }
        }
    }
}

static void refDtrsv(bool upper, bool trans, bool unit,
    int n, const double* A, int lda, double* x, int incx)
{
    if (trans) {
        if (upper) {
            for (int i = 0; i < n; ++i) {
                double t = x[i*incx];
                for (int j = 0; j < i; ++j) t -= A[j*lda+i] * x[j*incx];
                if (!unit) t /= A[i*lda+i];
                x[i*incx] = t;
            }
        } else {
            for (int i = n-1; i >= 0; --i) {
                double t = x[i*incx];
                for (int j = i+1; j < n; ++j) t -= A[j*lda+i] * x[j*incx];
                if (!unit) t /= A[i*lda+i];
                x[i*incx] = t;
            }
        }
    } else {
        if (upper) {
            for (int i = n-1; i >= 0; --i) {
                double t = x[i*incx];
                for (int j = i+1; j < n; ++j) t -= A[i*lda+j] * x[j*incx];
                if (!unit) t /= A[i*lda+i];
                x[i*incx] = t;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                double t = x[i*incx];
                for (int j = 0; j < i; ++j) t -= A[i*lda+j] * x[j*incx];
                if (!unit) t /= A[i*lda+i];
                x[i*incx] = t;
            }
        }
    }
}

static void refStrmv(bool upper, bool trans, bool unit,
    int n, const float* A, int lda, float* x, int incx)
{
    std::vector<float> tmp(n);
    for (int i = 0; i < n; ++i) tmp[i] = x[i*incx];
    if (trans) {
        if (upper) {
            for (int i = 0; i < n; ++i) {
                float t = unit ? tmp[i] : A[i*lda+i] * tmp[i];
                for (int j = 0; j < i; ++j) t += A[j*lda+i] * tmp[j];
                x[i*incx] = t;
            }
        } else {
            for (int i = n-1; i >= 0; --i) {
                float t = unit ? tmp[i] : A[i*lda+i] * tmp[i];
                for (int j = i+1; j < n; ++j) t += A[j*lda+i] * tmp[j];
                x[i*incx] = t;
            }
        }
    } else {
        if (upper) {
            for (int i = n-1; i >= 0; --i) {
                float t = unit ? tmp[i] : A[i*lda+i] * tmp[i];
                for (int j = i+1; j < n; ++j) t += A[i*lda+j] * tmp[j];
                x[i*incx] = t;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                float t = unit ? tmp[i] : A[i*lda+i] * tmp[i];
                for (int j = 0; j < i; ++j) t += A[i*lda+j] * tmp[j];
                x[i*incx] = t;
            }
        }
    }
}

static void refDtrmv(bool upper, bool trans, bool unit,
    int n, const double* A, int lda, double* x, int incx)
{
    std::vector<double> tmp(n);
    for (int i = 0; i < n; ++i) tmp[i] = x[i*incx];
    if (trans) {
        if (upper) {
            for (int i = 0; i < n; ++i) {
                double t = unit ? tmp[i] : A[i*lda+i] * tmp[i];
                for (int j = 0; j < i; ++j) t += A[j*lda+i] * tmp[j];
                x[i*incx] = t;
            }
        } else {
            for (int i = n-1; i >= 0; --i) {
                double t = unit ? tmp[i] : A[i*lda+i] * tmp[i];
                for (int j = i+1; j < n; ++j) t += A[j*lda+i] * tmp[j];
                x[i*incx] = t;
            }
        }
    } else {
        if (upper) {
            for (int i = n-1; i >= 0; --i) {
                double t = unit ? tmp[i] : A[i*lda+i] * tmp[i];
                for (int j = i+1; j < n; ++j) t += A[i*lda+j] * tmp[j];
                x[i*incx] = t;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                double t = unit ? tmp[i] : A[i*lda+i] * tmp[i];
                for (int j = 0; j < i; ++j) t += A[i*lda+j] * tmp[j];
                x[i*incx] = t;
            }
        }
    }
}

// ── Level-2 BLAS (P2.3) ─────────────────────────────────────────────────────

// ── Level-3 BLAS (P2.4) ─────────────────────────────────────────────────────

// ── Reference helpers for Level-3 BLAS ────────────────────────────────────

static void refStrsm(bool left, bool upper, bool trans, bool unit,
    int m, int n, float alpha, const float* A, int lda, float* B, int ldb)
{
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            B[i*ldb+j] *= alpha;
    if (left) {
        if (!trans) {
            if (upper) {
                for (int j = 0; j < n; ++j)
                    for (int i = m-1; i >= 0; --i) {
                        float t = B[i*ldb+j];
                        for (int k = i+1; k < m; ++k) t -= A[i*lda+k] * B[k*ldb+j];
                        if (!unit) t /= A[i*lda+i];
                        B[i*ldb+j] = t;
                    }
            } else {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i) {
                        float t = B[i*ldb+j];
                        for (int k = 0; k < i; ++k) t -= A[i*lda+k] * B[k*ldb+j];
                        if (!unit) t /= A[i*lda+i];
                        B[i*ldb+j] = t;
                    }
            }
        } else {
            if (upper) {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i) {
                        float t = B[i*ldb+j];
                        for (int k = 0; k < i; ++k) t -= A[k*lda+i] * B[k*ldb+j];
                        if (!unit) t /= A[i*lda+i];
                        B[i*ldb+j] = t;
                    }
            } else {
                for (int j = 0; j < n; ++j)
                    for (int i = m-1; i >= 0; --i) {
                        float t = B[i*ldb+j];
                        for (int k = i+1; k < m; ++k) t -= A[k*lda+i] * B[k*ldb+j];
                        if (!unit) t /= A[i*lda+i];
                        B[i*ldb+j] = t;
                    }
            }
        }
    } else {
        if (!trans) {
            if (upper) {
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j) {
                        float t = B[i*ldb+j];
                        for (int k = 0; k < j; ++k) t -= B[i*ldb+k] * A[k*lda+j];
                        if (!unit) t /= A[j*lda+j];
                        B[i*ldb+j] = t;
                    }
            } else {
                for (int i = 0; i < m; ++i)
                    for (int j = n-1; j >= 0; --j) {
                        float t = B[i*ldb+j];
                        for (int k = j+1; k < n; ++k) t -= B[i*ldb+k] * A[k*lda+j];
                        if (!unit) t /= A[j*lda+j];
                        B[i*ldb+j] = t;
                    }
            }
        } else {
            if (upper) {
                for (int i = 0; i < m; ++i)
                    for (int j = n-1; j >= 0; --j) {
                        float t = B[i*ldb+j];
                        for (int k = j+1; k < n; ++k) t -= B[i*ldb+k] * A[j*lda+k];
                        if (!unit) t /= A[j*lda+j];
                        B[i*ldb+j] = t;
                    }
            } else {
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j) {
                        float t = B[i*ldb+j];
                        for (int k = 0; k < j; ++k) t -= B[i*ldb+k] * A[j*lda+k];
                        if (!unit) t /= A[j*lda+j];
                        B[i*ldb+j] = t;
                    }
            }
        }
    }
}

static void refDtrsm(bool left, bool upper, bool trans, bool unit,
    int m, int n, double alpha, const double* A, int lda, double* B, int ldb)
{
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            B[i*ldb+j] *= alpha;
    if (left) {
        if (!trans) {
            if (upper) {
                for (int j = 0; j < n; ++j)
                    for (int i = m-1; i >= 0; --i) {
                        double t = B[i*ldb+j];
                        for (int k = i+1; k < m; ++k) t -= A[i*lda+k] * B[k*ldb+j];
                        if (!unit) t /= A[i*lda+i];
                        B[i*ldb+j] = t;
                    }
            } else {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i) {
                        double t = B[i*ldb+j];
                        for (int k = 0; k < i; ++k) t -= A[i*lda+k] * B[k*ldb+j];
                        if (!unit) t /= A[i*lda+i];
                        B[i*ldb+j] = t;
                    }
            }
        } else {
            if (upper) {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i) {
                        double t = B[i*ldb+j];
                        for (int k = 0; k < i; ++k) t -= A[k*lda+i] * B[k*ldb+j];
                        if (!unit) t /= A[i*lda+i];
                        B[i*ldb+j] = t;
                    }
            } else {
                for (int j = 0; j < n; ++j)
                    for (int i = m-1; i >= 0; --i) {
                        double t = B[i*ldb+j];
                        for (int k = i+1; k < m; ++k) t -= A[k*lda+i] * B[k*ldb+j];
                        if (!unit) t /= A[i*lda+i];
                        B[i*ldb+j] = t;
                    }
            }
        }
    } else {
        if (!trans) {
            if (upper) {
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j) {
                        double t = B[i*ldb+j];
                        for (int k = 0; k < j; ++k) t -= B[i*ldb+k] * A[k*lda+j];
                        if (!unit) t /= A[j*lda+j];
                        B[i*ldb+j] = t;
                    }
            } else {
                for (int i = 0; i < m; ++i)
                    for (int j = n-1; j >= 0; --j) {
                        double t = B[i*ldb+j];
                        for (int k = j+1; k < n; ++k) t -= B[i*ldb+k] * A[k*lda+j];
                        if (!unit) t /= A[j*lda+j];
                        B[i*ldb+j] = t;
                    }
            }
        } else {
            if (upper) {
                for (int i = 0; i < m; ++i)
                    for (int j = n-1; j >= 0; --j) {
                        double t = B[i*ldb+j];
                        for (int k = j+1; k < n; ++k) t -= B[i*ldb+k] * A[j*lda+k];
                        if (!unit) t /= A[j*lda+j];
                        B[i*ldb+j] = t;
                    }
            } else {
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j) {
                        double t = B[i*ldb+j];
                        for (int k = 0; k < j; ++k) t -= B[i*ldb+k] * A[j*lda+k];
                        if (!unit) t /= A[j*lda+j];
                        B[i*ldb+j] = t;
                    }
            }
        }
    }
}

static inline float symAf(const float* A, int lda, int i, int k, bool upper) {
    return upper ? ((k >= i) ? A[i*lda+k] : A[k*lda+i])
                 : ((k <= i) ? A[i*lda+k] : A[k*lda+i]);
}
static inline double symAd(const double* A, int lda, int i, int k, bool upper) {
    return upper ? ((k >= i) ? A[i*lda+k] : A[k*lda+i])
                 : ((k <= i) ? A[i*lda+k] : A[k*lda+i]);
}

static void refSsymm(bool left, bool upper, int m, int n,
    float alpha, const float* A, int lda, const float* B, int ldb,
    float beta, float* C, int ldc)
{
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            C[i*ldc+j] = beta * C[i*ldc+j];
    if (left) {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                float acc = 0;
                for (int k = 0; k < m; ++k) acc += symAf(A, lda, i, k, upper) * B[k*ldb+j];
                C[i*ldc+j] += alpha * acc;
            }
    } else {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                float acc = 0;
                for (int k = 0; k < n; ++k) acc += symAf(A, lda, j, k, upper) * B[i*ldb+k];
                C[i*ldc+j] += alpha * acc;
            }
    }
}

static void refDsymm(bool left, bool upper, int m, int n,
    double alpha, const double* A, int lda, const double* B, int ldb,
    double beta, double* C, int ldc)
{
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            C[i*ldc+j] = beta * C[i*ldc+j];
    if (left) {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                double acc = 0;
                for (int k = 0; k < m; ++k) acc += symAd(A, lda, i, k, upper) * B[k*ldb+j];
                C[i*ldc+j] += alpha * acc;
            }
    } else {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                double acc = 0;
                for (int k = 0; k < n; ++k) acc += symAd(A, lda, j, k, upper) * B[i*ldb+k];
                C[i*ldc+j] += alpha * acc;
            }
    }
}

