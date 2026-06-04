/**
 * VGRE built-in LAPACK fallback — portable CPU implementations.
 *
 * Compiled only when VGRE_LAPACK_FALLBACK is defined (no system LAPACK).
 * Provides all LAPACK routines used by cusolver_core.cpp:
 *   dpotrf/spotrf  — Cholesky factorization
 *   dgetrf/sgetrf  — LU factorization (partial pivoting)
 *   dgetrs/sgetrs  — triangular solve using LU
 *   dgeqrf/sgeqrf  — QR factorization (Householder)
 *   dormqr/sormqr  — apply Q from QR
 *   dgesvd/sgesvd  — SVD (one-sided Jacobi)
 *   dsyevd/ssyevd  — symmetric eigenvalues (Jacobi iteration)
 *   dgelsd/sgelsd  — least-squares via truncated SVD
 */

#ifdef VGRE_LAPACK_FALLBACK

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

// ── Template kernel implementations (C++ linkage, called by C wrappers) ──────

template<typename T>
static void vgre_potrf(const char *uplo, int N, T *A, int LDA, int *info) {
    *info = 0;
    bool lower = (*uplo == 'L' || *uplo == 'l');
    if (lower) {
        for (int j = 0; j < N; ++j) {
            T ajj = A[j + j * LDA];
            for (int k = 0; k < j; ++k) ajj -= A[j + k * LDA] * A[j + k * LDA];
            if (ajj <= T(0)) { *info = j + 1; return; }
            ajj = std::sqrt(ajj);
            A[j + j * LDA] = ajj;
            T inv = T(1) / ajj;
            for (int i = j + 1; i < N; ++i) {
                T s = A[i + j * LDA];
                for (int k = 0; k < j; ++k) s -= A[i + k * LDA] * A[j + k * LDA];
                A[i + j * LDA] = s * inv;
            }
        }
    } else {
        for (int j = 0; j < N; ++j) {
            T ajj = A[j + j * LDA];
            for (int k = 0; k < j; ++k) ajj -= A[k + j * LDA] * A[k + j * LDA];
            if (ajj <= T(0)) { *info = j + 1; return; }
            ajj = std::sqrt(ajj);
            A[j + j * LDA] = ajj;
            T inv = T(1) / ajj;
            for (int i = j + 1; i < N; ++i) {
                T s = A[j + i * LDA];
                for (int k = 0; k < j; ++k) s -= A[k + j * LDA] * A[k + i * LDA];
                A[j + i * LDA] = s * inv;
            }
        }
    }
}

// dpotrs_ — solve A*x = b using Cholesky factors from dpotrf.
// uplo='L': A = L*L', forward/backward substitution with L.
// uplo='U': A = U'*U, forward/backward substitution with U.
template<typename T>
static void vgre_potrs(const char *uplo, int N, int NRHS, const T *A, int LDA,
                       T *B, int LDB, int *info) {
    *info = 0;
    bool lower = (*uplo == 'L' || *uplo == 'l');
    if (lower) {
        // Solve L * y = B (forward)
        for (int j = 0; j < NRHS; ++j)
            for (int k = 0; k < N; ++k) {
                B[k + j * LDB] /= A[k + k * LDA];
                for (int i = k + 1; i < N; ++i)
                    B[i + j * LDB] -= A[i + k * LDA] * B[k + j * LDB];
            }
        // Solve L' * x = y (backward)
        for (int j = 0; j < NRHS; ++j)
            for (int k = N - 1; k >= 0; --k) {
                B[k + j * LDB] /= A[k + k * LDA];
                for (int i = 0; i < k; ++i)
                    B[i + j * LDB] -= A[k + i * LDA] * B[k + j * LDB];
            }
    } else {
        // Solve U' * y = B (forward)
        for (int j = 0; j < NRHS; ++j)
            for (int k = 0; k < N; ++k) {
                B[k + j * LDB] /= A[k + k * LDA];
                for (int i = k + 1; i < N; ++i)
                    B[i + j * LDB] -= A[k + i * LDA] * B[k + j * LDB];
            }
        // Solve U * x = y (backward)
        for (int j = 0; j < NRHS; ++j)
            for (int k = N - 1; k >= 0; --k) {
                B[k + j * LDB] /= A[k + k * LDA];
                for (int i = 0; i < k; ++i)
                    B[i + j * LDB] -= A[i + k * LDA] * B[k + j * LDB];
            }
    }
}

template<typename T>
static void vgre_getrf(int M, int N, T *A, int LDA, int *ipiv, int *info) {
    *info = 0;
    int K = (M < N) ? M : N;
    for (int k = 0; k < K; ++k) {
        int piv = k;
        T maxv = std::abs(A[k + k * LDA]);
        for (int i = k + 1; i < M; ++i) {
            T v = std::abs(A[i + k * LDA]);
            if (v > maxv) { maxv = v; piv = i; }
        }
        ipiv[k] = piv + 1;
        if (maxv == T(0)) { *info = k + 1; return; }
        if (piv != k)
            for (int j = 0; j < N; ++j)
                std::swap(A[k + j * LDA], A[piv + j * LDA]);
        T diag = A[k + k * LDA];
        for (int i = k + 1; i < M; ++i) A[i + k * LDA] /= diag;
        for (int j = k + 1; j < N; ++j)
            for (int i = k + 1; i < M; ++i)
                A[i + j * LDA] -= A[i + k * LDA] * A[k + j * LDA];
    }
}

template<typename T>
static void vgre_getrs(const char *trans, int N, int NRHS,
                       const T *A, int LDA, const int *ipiv,
                       T *B, int LDB, int *info) {
    *info = 0;
    bool notrans = (*trans == 'N' || *trans == 'n');
    if (notrans) {
        for (int k = 0; k < N; ++k) {
            int piv = ipiv[k] - 1;
            if (piv != k)
                for (int j = 0; j < NRHS; ++j)
                    std::swap(B[k + j * LDB], B[piv + j * LDB]);
        }
        for (int j = 0; j < NRHS; ++j)
            for (int k = 0; k < N; ++k)
                for (int i = k + 1; i < N; ++i)
                    B[i + j * LDB] -= A[i + k * LDA] * B[k + j * LDB];
        for (int j = 0; j < NRHS; ++j)
            for (int k = N - 1; k >= 0; --k) {
                B[k + j * LDB] /= A[k + k * LDA];
                for (int i = 0; i < k; ++i)
                    B[i + j * LDB] -= A[i + k * LDA] * B[k + j * LDB];
            }
    } else {
        for (int j = 0; j < NRHS; ++j)
            for (int k = 0; k < N; ++k) {
                for (int i = 0; i < k; ++i)
                    B[k + j * LDB] -= A[i + k * LDA] * B[i + j * LDB];
                B[k + j * LDB] /= A[k + k * LDA];
            }
        for (int j = 0; j < NRHS; ++j)
            for (int k = N - 1; k >= 0; --k)
                for (int i = k + 1; i < N; ++i)
                    B[k + j * LDB] -= A[i + k * LDA] * B[i + j * LDB];
        for (int k = N - 1; k >= 0; --k) {
            int piv = ipiv[k] - 1;
            if (piv != k)
                for (int j = 0; j < NRHS; ++j)
                    std::swap(B[k + j * LDB], B[piv + j * LDB]);
        }
    }
}

// Householder reflector: modifies x[0..n-1], returns tau.
// After call: x[0] = -sign(x_orig[0])*norm(x), x[1..] = v[1..] (v[0]=1 implicit).
template<typename T>
static T vgre_house(T *x, int n) {
    T sigma = T(0);
    for (int i = 1; i < n; ++i) sigma += x[i] * x[i];
    if (sigma == T(0) && x[0] >= T(0)) return T(0);
    T norm = std::sqrt(x[0] * x[0] + sigma);
    if (x[0] > T(0)) norm = -norm;
    T v0 = x[0] - norm;
    T tau = -v0 / norm;
    x[0] = norm;
    T inv = T(1) / v0;
    for (int i = 1; i < n; ++i) x[i] *= inv;
    return tau;
}

template<typename T>
static void vgre_geqrf(int M, int N, T *A, int LDA, T *tau, int *info) {
    *info = 0;
    int K = (M < N) ? M : N;
    std::vector<T> v(M);
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < M - k; ++i) v[i] = A[(k + i) + k * LDA];
        T t = vgre_house<T>(v.data(), M - k);
        tau[k] = t;
        A[k + k * LDA] = v[0];
        for (int i = 1; i < M - k; ++i) A[(k + i) + k * LDA] = v[i];
        if (t == T(0)) continue;
        // v[0] = 1 (implicit), apply H = I - tau*v*v' to trailing columns
        v[0] = T(1);
        for (int j = k + 1; j < N; ++j) {
            T w = T(0);
            for (int i = 0; i < M - k; ++i) w += v[i] * A[(k + i) + j * LDA];
            for (int i = 0; i < M - k; ++i) A[(k + i) + j * LDA] -= t * v[i] * w;
        }
    }
}

template<typename T>
static void vgre_ormqr(const char *side, const char *trans,
                       int M, int N, int K,
                       const T *A, int LDA, const T *tau,
                       T *C, int LDC, int *info) {
    *info = 0;
    bool left  = (*side  == 'L' || *side  == 'l');
    bool nottr = (*trans == 'N' || *trans == 'n');
    int len_v = left ? M : N;
    std::vector<T> v(len_v);
    int start = nottr ? 0 : K - 1;
    int end   = nottr ? K : -1;
    int step  = nottr ? 1 : -1;
    for (int i = start; i != end; i += step) {
        int vlen = len_v - i;
        v[0] = T(1);
        for (int j = 1; j < vlen; ++j) v[j] = A[(i + j) + i * LDA];
        T t = tau[i];
        if (t == T(0)) continue;
        if (left) {
            for (int j = 0; j < N; ++j) {
                T w = T(0);
                for (int r = 0; r < vlen; ++r) w += v[r] * C[(i + r) + j * LDC];
                for (int r = 0; r < vlen; ++r) C[(i + r) + j * LDC] -= t * v[r] * w;
            }
        } else {
            for (int r = 0; r < M; ++r) {
                T w = T(0);
                for (int c = 0; c < vlen; ++c) w += C[r + (i + c) * LDC] * v[c];
                for (int c = 0; c < vlen; ++c) C[r + (i + c) * LDC] -= t * w * v[c];
            }
        }
    }
}

// Jacobi SVD (one-sided): A (M×N) → columns of A become u_i * sigma_i; V accumulates right rotations.
template<typename T>
static void vgre_gesvd(const char *jobu, const char *jobvt,
                       int M, int N, T *A, int LDA,
                       T *S, T *U, int LDU, T *VT, int LDVT, int *info) {
    *info = 0;
    bool want_u  = (*jobu  == 'S' || *jobu  == 's' || *jobu  == 'A' || *jobu  == 'a');
    bool want_vt = (*jobvt == 'S' || *jobvt == 's' || *jobvt == 'A' || *jobvt == 'a');
    int K = (M < N) ? M : N;

    // Working copy of A (column-major, size M×N stored as M stride)
    std::vector<T> Aw(M * N);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < M; ++i)
            Aw[i + j * M] = A[i + j * LDA];

    // V accumulates right rotations (N×N)
    std::vector<T> Vm(N * N, T(0));
    for (int i = 0; i < N; ++i) Vm[i + i * N] = T(1);

    const int MAX_SWEEPS = 30;
    const T eps = std::numeric_limits<T>::epsilon();
    for (int sw = 0; sw < MAX_SWEEPS; ++sw) {
        T off = T(0);
        for (int p = 0; p < K - 1; ++p)
            for (int q = p + 1; q < K; ++q) {
                T g = T(0);
                for (int i = 0; i < M; ++i) g += Aw[i + p * M] * Aw[i + q * M];
                off += g * g;
            }
        if (off < eps * eps) break;
        for (int p = 0; p < K - 1; ++p) {
            for (int q = p + 1; q < K; ++q) {
                T alpha = T(0), beta = T(0), gamma = T(0);
                for (int i = 0; i < M; ++i) {
                    T ap = Aw[i + p * M], aq = Aw[i + q * M];
                    alpha += ap * ap; beta += aq * aq; gamma += ap * aq;
                }
                if (std::abs(gamma) < eps * (std::sqrt(alpha) * std::sqrt(beta) + eps))
                    continue;
                // One-sided Jacobi SVD rotation: zero gamma = col_p'*col_q.
                // Correct root: t = -sign(ζ)/(|ζ|+√(1+ζ²)) where ζ=(β-α)/(2γ).
                // This gives tan(2θ) = -1/ζ = -2γ/(β-α), zeroing γ_new.
                T zeta = (beta - alpha) / (T(2) * gamma);
                T t = (zeta >= T(0))
                    ? T(-1) / (std::abs(zeta) + std::sqrt(T(1) + zeta * zeta))
                    : T(1)  / (std::abs(zeta) + std::sqrt(T(1) + zeta * zeta));
                T c = T(1) / std::sqrt(T(1) + t * t), s = c * t;
                for (int i = 0; i < M; ++i) {
                    T ap = Aw[i + p * M], aq = Aw[i + q * M];
                    Aw[i + p * M] =  c * ap + s * aq;
                    Aw[i + q * M] = -s * ap + c * aq;
                }
                for (int i = 0; i < N; ++i) {
                    T vp = Vm[i + p * N], vq = Vm[i + q * N];
                    Vm[i + p * N] =  c * vp + s * vq;
                    Vm[i + q * N] = -s * vp + c * vq;
                }
            }
        }
    }

    // Compute singular values = column norms of Aw; U cols = normalized Aw cols
    std::vector<T> sv(K);
    for (int j = 0; j < K; ++j) {
        T norm = T(0);
        for (int i = 0; i < M; ++i) norm += Aw[i + j * M] * Aw[i + j * M];
        sv[j] = std::sqrt(norm);
    }

    // Sort descending
    std::vector<int> idx(K);
    for (int i = 0; i < K; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return sv[a] > sv[b]; });

    for (int i = 0; i < K; ++i) S[i] = sv[idx[i]];

    if (want_u) {
        std::vector<T> Utmp(M * K);
        for (int j = 0; j < K; ++j) {
            T norm = sv[idx[j]];
            T inv = (norm > T(0)) ? T(1) / norm : T(0);
            for (int i = 0; i < M; ++i) Utmp[i + j * M] = Aw[i + idx[j] * M] * inv;
        }
        for (int j = 0; j < K; ++j)
            for (int i = 0; i < M; ++i)
                U[i + j * LDU] = Utmp[i + j * M];
    }
    if (want_vt) {
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < N; ++j)
                VT[i + j * LDVT] = Vm[j + idx[i] * N];
    }
}

// Jacobi symmetric eigensolver
template<typename T>
static void vgre_syevd(const char *jobz, int N, T *A, int LDA, T *W, int *info) {
    *info = 0;
    bool want_v = (*jobz == 'V' || *jobz == 'v');

    std::vector<T> Mc(N * N);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            Mc[i + j * N] = A[i + j * LDA];
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            Mc[j + i * N] = Mc[i + j * N];

    std::vector<T> Vc(N * N, T(0));
    for (int i = 0; i < N; ++i) Vc[i + i * N] = T(1);

    const int MAX_ITER = 100 * N * N;
    const T EPS = std::numeric_limits<T>::epsilon() * T(4);
    for (int it = 0; it < MAX_ITER; ++it) {
        T maxv = T(0);
        int p = 0, q = 1;
        for (int i = 0; i < N - 1; ++i)
            for (int j = i + 1; j < N; ++j) {
                T v = std::abs(Mc[i + j * N]);
                if (v > maxv) { maxv = v; p = i; q = j; }
            }
        if (maxv < EPS) break;
        T theta = (Mc[q + q * N] - Mc[p + p * N]) / (T(2) * Mc[p + q * N]);
        T t = (theta >= T(0))
            ? T(1) / (std::abs(theta) + std::sqrt(T(1) + theta * theta))
            : T(-1)/ (std::abs(theta) + std::sqrt(T(1) + theta * theta));
        T c = T(1) / std::sqrt(T(1) + t * t), s = t * c;
        T Mpp = Mc[p + p * N], Mqq = Mc[q + q * N], Mpq = Mc[p + q * N];
        Mc[p + p * N] = c*c*Mpp - T(2)*s*c*Mpq + s*s*Mqq;
        Mc[q + q * N] = s*s*Mpp + T(2)*s*c*Mpq + c*c*Mqq;
        Mc[p + q * N] = Mc[q + p * N] = T(0);
        for (int r = 0; r < N; ++r) {
            if (r == p || r == q) continue;
            T Mrp = Mc[r + p * N], Mrq = Mc[r + q * N];
            Mc[r + p * N] = Mc[p + r * N] = c * Mrp - s * Mrq;
            Mc[r + q * N] = Mc[q + r * N] = s * Mrp + c * Mrq;
        }
        if (want_v)
            for (int r = 0; r < N; ++r) {
                T Vrp = Vc[r + p * N], Vrq = Vc[r + q * N];
                Vc[r + p * N] =  c * Vrp - s * Vrq;
                Vc[r + q * N] =  s * Vrp + c * Vrq;
            }
    }

    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){
        return Mc[a + a * N] < Mc[b + b * N]; });
    for (int i = 0; i < N; ++i) W[i] = Mc[idx[i] + idx[i] * N];
    if (want_v) {
        std::vector<T> tmp(N * N);
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                tmp[i + j * N] = Vc[i + idx[j] * N];
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i)
                A[i + j * LDA] = tmp[i + j * N];
    }
}

template<typename T>
static void vgre_gelsd(int M, int N, int NRHS, T *A, int LDA,
                       T *B, int LDB, T *S, T rcond,
                       int *rank_, int *info) {
    *info = 0;
    int K = (M < N) ? M : N;
    std::vector<T> Ac(M * N);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < M; ++i)
            Ac[i + j * M] = A[i + j * LDA];

    std::vector<T> Um(M * K, T(0)), VTm(K * N, T(0)), sv(K);
    const char jobu = 'S', jobvt = 'S';
    vgre_gesvd<T>(&jobu, &jobvt, M, N, Ac.data(), M,
                  sv.data(), Um.data(), M, VTm.data(), K, info);
    if (*info != 0) return;
    if (S) for (int i = 0; i < K; ++i) S[i] = sv[i];

    T thresh = (rcond < T(0)) ? std::numeric_limits<T>::epsilon() * sv[0] : rcond * sv[0];
    int rnk = 0;
    for (int i = 0; i < K; ++i) if (sv[i] > thresh) ++rnk;
    *rank_ = rnk;

    // x = V * S^+ * U' * b
    std::vector<T> tmp(K * NRHS, T(0));
    for (int j = 0; j < NRHS; ++j)
        for (int i = 0; i < K; ++i)
            for (int r = 0; r < M; ++r)
                tmp[i + j * K] += Um[r + i * M] * B[r + j * LDB];
    for (int i = 0; i < rnk; ++i)
        for (int j = 0; j < NRHS; ++j)
            tmp[i + j * K] /= sv[i];
    for (int i = rnk; i < K; ++i)
        for (int j = 0; j < NRHS; ++j)
            tmp[i + j * K] = T(0);
    for (int j = 0; j < NRHS; ++j)
        for (int i = 0; i < N; ++i) {
            T sum = T(0);
            for (int r = 0; r < K; ++r) sum += VTm[r + i * K] * tmp[r + j * K];
            B[i + j * LDB] = sum;
        }
    for (int j = 0; j < NRHS; ++j)
        for (int i = N; i < M; ++i)
            B[i + j * LDB] = T(0);
}

// ── C-linkage wrappers (Fortran naming: all args by pointer) ─────────────────

extern "C" {

void dpotrf_(const char *uplo, const int *n, double *A, const int *lda, int *info) {
    vgre_potrf<double>(uplo, *n, A, *lda, info);
}
void spotrf_(const char *uplo, const int *n, float *A, const int *lda, int *info) {
    vgre_potrf<float>(uplo, *n, A, *lda, info);
}
void dpotrs_(const char *uplo, const int *n, const int *nrhs, const double *A,
             const int *lda, double *B, const int *ldb, int *info) {
    vgre_potrs<double>(uplo, *n, *nrhs, A, *lda, B, *ldb, info);
}
void spotrs_(const char *uplo, const int *n, const int *nrhs, const float *A,
             const int *lda, float *B, const int *ldb, int *info) {
    vgre_potrs<float>(uplo, *n, *nrhs, A, *lda, B, *ldb, info);
}

void dgetrf_(const int *m, const int *n, double *A, const int *lda, int *ipiv, int *info) {
    vgre_getrf<double>(*m, *n, A, *lda, ipiv, info);
}
void sgetrf_(const int *m, const int *n, float *A, const int *lda, int *ipiv, int *info) {
    vgre_getrf<float>(*m, *n, A, *lda, ipiv, info);
}

void dgetrs_(const char *trans, const int *n, const int *nrhs, const double *A,
             const int *lda, const int *ipiv, double *B, const int *ldb, int *info) {
    vgre_getrs<double>(trans, *n, *nrhs, A, *lda, ipiv, B, *ldb, info);
}
void sgetrs_(const char *trans, const int *n, const int *nrhs, const float *A,
             const int *lda, const int *ipiv, float *B, const int *ldb, int *info) {
    vgre_getrs<float>(trans, *n, *nrhs, A, *lda, ipiv, B, *ldb, info);
}

// Minimum workspace sizes returned for workspace queries (lwork=-1).
// Our implementations use std::vector internally so the provided work buffer
// is unused — but callers rely on work[0] being a valid size after a query.
static inline int vgre_geqrf_lwork(int m, int n) { return std::max(1, std::min(m,n) * 64); }
static inline int vgre_ormqr_lwork(int m, int n) { return std::max(1, (m + n) * 32); }
static inline int vgre_gesvd_lwork(int m, int n) { return std::max(1, (m + n) * 64); }
static inline int vgre_syevd_lwork(int n)         { return std::max(1, n * n + 6 * n); }
static inline int vgre_gelsd_lwork(int m, int n)  { return std::max(1, (m + n) * 64); }

void dgeqrf_(const int *m, const int *n, double *A, const int *lda,
             double *tau, double *work, const int *lwork, int *info) {
    if (work && *lwork < 0) { work[0] = static_cast<double>(vgre_geqrf_lwork(*m,*n)); *info = 0; return; }
    vgre_geqrf<double>(*m, *n, A, *lda, tau, info);
}
void sgeqrf_(const int *m, const int *n, float *A, const int *lda,
             float *tau, float *work, const int *lwork, int *info) {
    if (work && *lwork < 0) { work[0] = static_cast<float>(vgre_geqrf_lwork(*m,*n)); *info = 0; return; }
    vgre_geqrf<float>(*m, *n, A, *lda, tau, info);
}

void dormqr_(const char *side, const char *trans, const int *m, const int *n,
             const int *k, const double *A, const int *lda, const double *tau,
             double *C, const int *ldc, double *work, const int *lwork, int *info) {
    if (work && *lwork < 0) { work[0] = static_cast<double>(vgre_ormqr_lwork(*m,*n)); *info = 0; return; }
    vgre_ormqr<double>(side, trans, *m, *n, *k, A, *lda, tau, C, *ldc, info);
}
void sormqr_(const char *side, const char *trans, const int *m, const int *n,
             const int *k, const float *A, const int *lda, const float *tau,
             float *C, const int *ldc, float *work, const int *lwork, int *info) {
    if (work && *lwork < 0) { work[0] = static_cast<float>(vgre_ormqr_lwork(*m,*n)); *info = 0; return; }
    vgre_ormqr<float>(side, trans, *m, *n, *k, A, *lda, tau, C, *ldc, info);
}

void dgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n,
             double *A, const int *lda, double *S, double *U, const int *ldu,
             double *VT, const int *ldvt, double *work, const int *lwork, int *info) {
    if (work && *lwork < 0) { work[0] = static_cast<double>(vgre_gesvd_lwork(*m,*n)); *info = 0; return; }
    int LDU  = U  ? *ldu  : *m;
    int LDVT = VT ? *ldvt : std::min(*m, *n);
    vgre_gesvd<double>(jobu, jobvt, *m, *n, A, *lda, S, U, LDU, VT, LDVT, info);
}
void sgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n,
             float *A, const int *lda, float *S, float *U, const int *ldu,
             float *VT, const int *ldvt, float *work, const int *lwork, int *info) {
    if (work && *lwork < 0) { work[0] = static_cast<float>(vgre_gesvd_lwork(*m,*n)); *info = 0; return; }
    int LDU  = U  ? *ldu  : *m;
    int LDVT = VT ? *ldvt : std::min(*m, *n);
    vgre_gesvd<float>(jobu, jobvt, *m, *n, A, *lda, S, U, LDU, VT, LDVT, info);
}

void dsyevd_(const char *jobz, const char *uplo, const int *n, double *A,
             const int *lda, double *W, double *work, const int *lwork,
             int *, const int *, int *info) {
    (void)uplo;
    if (work && *lwork < 0) { work[0] = static_cast<double>(vgre_syevd_lwork(*n)); *info = 0; return; }
    vgre_syevd<double>(jobz, *n, A, *lda, W, info);
}
void ssyevd_(const char *jobz, const char *uplo, const int *n, float *A,
             const int *lda, float *W, float *work, const int *lwork,
             int *, const int *, int *info) {
    (void)uplo;
    if (work && *lwork < 0) { work[0] = static_cast<float>(vgre_syevd_lwork(*n)); *info = 0; return; }
    vgre_syevd<float>(jobz, *n, A, *lda, W, info);
}

void dgelsd_(const int *m, const int *n, const int *nrhs, double *A, const int *lda,
             double *B, const int *ldb, double *S, const double *rcond, int *rank_,
             double *work, const int *lwork, int *, int *info) {
    if (work && *lwork < 0) { work[0] = static_cast<double>(vgre_gelsd_lwork(*m,*n)); *info = 0; return; }
    vgre_gelsd<double>(*m, *n, *nrhs, A, *lda, B, *ldb, S, *rcond, rank_, info);
}
void sgelsd_(const int *m, const int *n, const int *nrhs, float *A, const int *lda,
             float *B, const int *ldb, float *S, const float *rcond, int *rank_,
             float *work, const int *lwork, int *, int *info) {
    if (work && *lwork < 0) { work[0] = static_cast<float>(vgre_gelsd_lwork(*m,*n)); *info = 0; return; }
    vgre_gelsd<float>(*m, *n, *nrhs, A, *lda, B, *ldb, S, *rcond, rank_, info);
}

// ── Complex (c/z) LAPACK stubs ────────────────────────────────────────────────
// VGRE represents complex matrices as flat float*/double* arrays; these stubs
// delegate to the corresponding real (s/d) implementations.

void cpotrf_(const char *uplo, const int *n, float *a, const int *lda, int *info) {
    vgre_potrf<float>(uplo, *n, a, *lda, info);
}
void zpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info) {
    vgre_potrf<double>(uplo, *n, a, *lda, info);
}
void cpotrs_(const char *uplo, const int *n, const int *nrhs,
             const float *A, const int *lda, float *B, const int *ldb, int *info) {
    vgre_potrs<float>(uplo, *n, *nrhs, A, *lda, B, *ldb, info);
}
void zpotrs_(const char *uplo, const int *n, const int *nrhs,
             const double *A, const int *lda, double *B, const int *ldb, int *info) {
    vgre_potrs<double>(uplo, *n, *nrhs, A, *lda, B, *ldb, info);
}
void cgetrf_(const int *m, const int *n, float *a, const int *lda, int *ipiv, int *info) {
    vgre_getrf<float>(*m, *n, a, *lda, ipiv, info);
}
void zgetrf_(const int *m, const int *n, double *a, const int *lda, int *ipiv, int *info) {
    vgre_getrf<double>(*m, *n, a, *lda, ipiv, info);
}
void cgetrs_(const char *trans, const int *n, const int *nrhs,
             const float *A, const int *lda, const int *ipiv,
             float *B, const int *ldb, int *info) {
    vgre_getrs<float>(trans, *n, *nrhs, A, *lda, ipiv, B, *ldb, info);
}
void zgetrs_(const char *trans, const int *n, const int *nrhs,
             const double *A, const int *lda, const int *ipiv,
             double *B, const int *ldb, int *info) {
    vgre_getrs<double>(trans, *n, *nrhs, A, *lda, ipiv, B, *ldb, info);
}
// cgesvd_/zgesvd_ have an extra rwork parameter; ignore it and delegate to real SVD.
void cgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n,
             float *A, const int *lda, float *S, float *U, const int *ldu,
             float *VT, const int *ldvt, float *work, const int *lwork,
             float * /*rwork*/, int *info) {
    if (work && *lwork < 0) { work[0] = static_cast<float>(vgre_gesvd_lwork(*m,*n)); *info = 0; return; }
    int LDU  = U  ? *ldu  : *m;
    int LDVT = VT ? *ldvt : std::min(*m, *n);
    vgre_gesvd<float>(jobu, jobvt, *m, *n, A, *lda, S, U, LDU, VT, LDVT, info);
}
void zgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n,
             double *A, const int *lda, double *S, double *U, const int *ldu,
             double *VT, const int *ldvt, double *work, const int *lwork,
             double * /*rwork*/, int *info) {
    if (work && *lwork < 0) { work[0] = static_cast<double>(vgre_gesvd_lwork(*m,*n)); *info = 0; return; }
    int LDU  = U  ? *ldu  : *m;
    int LDVT = VT ? *ldvt : std::min(*m, *n);
    vgre_gesvd<double>(jobu, jobvt, *m, *n, A, *lda, S, U, LDU, VT, LDVT, info);
}
// cheevd_/zheevd_ add rwork/iwork params; ignore them and delegate to real syevd.
void cheevd_(const char *jobz, const char *uplo, const int *n,
             float *A, const int *lda, float *W,
             float *work, const int *lwork,
             float * /*rwork*/, const int * /*lrwork*/,
             int * /*iwork*/, const int * /*liwork*/, int *info) {
    (void)uplo;
    if (work && *lwork < 0) { work[0] = static_cast<float>(vgre_syevd_lwork(*n)); *info = 0; return; }
    vgre_syevd<float>(jobz, *n, A, *lda, W, info);
}
void zheevd_(const char *jobz, const char *uplo, const int *n,
             double *A, const int *lda, double *W,
             double *work, const int *lwork,
             double * /*rwork*/, const int * /*lrwork*/,
             int * /*iwork*/, const int * /*liwork*/, int *info) {
    (void)uplo;
    if (work && *lwork < 0) { work[0] = static_cast<double>(vgre_syevd_lwork(*n)); *info = 0; return; }
    vgre_syevd<double>(jobz, *n, A, *lda, W, info);
}

} // extern "C"

#endif // VGRE_LAPACK_FALLBACK
