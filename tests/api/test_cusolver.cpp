// cuSOLVER test — validates LU (getrf/getrs), QR+ormqr, gelsd (least-squares),
// potrf (Cholesky), gesvd (SVD), syevd (eigenvalues).

#include "vgre/api/cusolver_shim.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

static int g_pass = 0, g_total = 0;

static void check(const char *name, bool ok) {
    ++g_total;
    if (ok) { ++g_pass; std::cout << "PASS [" << g_total << "] " << name << "\n"; }
    else    { std::cerr << "FAIL [" << g_total << "] " << name << "\n"; }
}

// ── Test: handle lifecycle ───────────────────────────────────────────────────
static void test_handle() {
    cusolverDnHandle_t h;
    bool ok = (cusolverDnCreate(&h) == CUSOLVER_STATUS_SUCCESS);
    ok &= (cusolverDnDestroy(h) == CUSOLVER_STATUS_SUCCESS);
    ok &= (cusolverDnCreate(nullptr) == CUSOLVER_STATUS_INVALID_VALUE);
    check("Handle create/destroy", ok);
}

// ── Test: LU factorization + solve (getrf + getrs) ──────────────────────────
// Solve A*x = b where A = [[2,1],[5,3]], b = [4,7]
// 2x + y = 4, 5x + 3y = 7 => x=5, y=-6
static void test_getrf_getrs() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    const int N = 2, NRHS = 1;
    // Column-major: A = [[2,1],[5,3]] => col0=[2,5], col1=[1,3]
    double A[4] = {2.0, 5.0, 1.0, 3.0};
    double B[2] = {4.0, 7.0};
    int ipiv[2], info = 0;

    int lwork;
    cusolverDnDgetrf_bufferSize(h, N, N, A, N, &lwork);
    std::vector<double> work(lwork);

    cusolverDnDgetrf(h, N, N, A, N, work.data(), ipiv, &info);
    bool ok = (info == 0);

    cusolverDnDgetrs(h, 0 /*no-trans*/, N, NRHS, A, N, ipiv, B, N, &info);
    ok &= (info == 0);
    ok &= (std::fabs(B[0] - 5.0) < 1e-10 && std::fabs(B[1] - (-6.0)) < 1e-10);

    cusolverDnDestroy(h);
    check("getrf+getrs solve Ax=b", ok);
}

// ── Test: LU factorization (float) ──────────────────────────────────────────
static void test_getrf_float() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    const int N = 3;
    // Identity matrix — LU should be trivial
    float A[9] = {1,0,0, 0,1,0, 0,0,1};
    int ipiv[3], info = 0;
    int lwork;
    cusolverDnSgetrf_bufferSize(h, N, N, A, N, &lwork);
    std::vector<float> work(lwork);
    cusolverDnSgetrf(h, N, N, A, N, work.data(), ipiv, &info);

    bool ok = (info == 0);
    // Diagonal should still be 1
    ok &= (std::fabs(A[0] - 1.0f) < 1e-6f && std::fabs(A[4] - 1.0f) < 1e-6f && std::fabs(A[8] - 1.0f) < 1e-6f);

    cusolverDnDestroy(h);
    check("getrf float (identity)", ok);
}

// ── Test: Cholesky (potrf) ──────────────────────────────────────────────────
static void test_potrf() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    // SPD matrix: A = [[4,2],[2,3]] (column-major)
    double A[4] = {4.0, 2.0, 2.0, 3.0};
    int info = 0;
    int lwork;
    cusolverDnDpotrf_bufferSize(h, 'L', 2, A, 2, &lwork);
    std::vector<double> work(lwork);
    cusolverDnDpotrf(h, 'L', 2, A, 2, work.data(), lwork, &info);

    bool ok = (info == 0);
    // Lower-triangular L: L[0,0]=2, L[1,0]=1, L[1,1]=sqrt(2)
    ok &= (std::fabs(A[0] - 2.0) < 1e-10);
    ok &= (std::fabs(A[1] - 1.0) < 1e-10);
    ok &= (std::fabs(A[3] - std::sqrt(2.0)) < 1e-10);

    cusolverDnDestroy(h);
    check("potrf Cholesky (2x2 SPD)", ok);
}

// ── Test: QR factorization (geqrf) ──────────────────────────────────────────
static void test_geqrf() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    const int M = 3, N = 2;
    // Column-major 3x2: [[1,4],[2,5],[3,6]]
    double A[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double tau[2];
    int info = 0, lwork;

    cusolverDnDgeqrf_bufferSize(h, M, N, A, M, &lwork);
    std::vector<double> work(lwork);
    cusolverDnDgeqrf(h, M, N, A, M, tau, work.data(), lwork, &info);

    bool ok = (info == 0);
    // R[0,0] should be -sqrt(14) ≈ -3.7417
    ok &= (std::fabs(std::fabs(A[0]) - std::sqrt(14.0)) < 1e-8);

    cusolverDnDestroy(h);
    check("geqrf QR (3x2)", ok);
}

// ── Test: ormqr (apply Q from QR) ───────────────────────────────────────────
static void test_ormqr() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    const int M = 3, N = 2;
    double A[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double A_orig[6]; memcpy(A_orig, A, sizeof(A));
    double tau[2];
    int info = 0, lwork;

    // First do QR
    cusolverDnDgeqrf_bufferSize(h, M, N, A, M, &lwork);
    std::vector<double> work(lwork);
    cusolverDnDgeqrf(h, M, N, A, M, tau, work.data(), lwork, &info);
    bool ok = (info == 0);

    // Apply Q to identity => should get Q columns
    double C[9] = {1,0,0, 0,1,0, 0,0,1}; // 3x3 identity
    int ormqr_lwork;
    cusolverDnDormqr_bufferSize(h, 0/*left*/, 0/*no-trans*/, M, M, N, A, M, tau, C, M, &ormqr_lwork);
    std::vector<double> work2(ormqr_lwork);
    cusolverDnDormqr(h, 0, 0, M, M, N, A, M, tau, C, M, work2.data(), ormqr_lwork, &info);
    ok &= (info == 0);

    // Q should be orthogonal: Q^T * Q ≈ I
    // Check Q[:,0] . Q[:,1] ≈ 0
    double dot01 = C[0]*C[3] + C[1]*C[4] + C[2]*C[5];
    ok &= (std::fabs(dot01) < 1e-10);
    // Check ||Q[:,0]|| ≈ 1
    double norm0 = std::sqrt(C[0]*C[0] + C[1]*C[1] + C[2]*C[2]);
    ok &= (std::fabs(norm0 - 1.0) < 1e-10);

    cusolverDnDestroy(h);
    check("ormqr (apply Q, orthogonality)", ok);
}

// ── Test: SVD (gesvd) ───────────────────────────────────────────────────────
static void test_gesvd() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    const int M = 3, N = 2;
    // Column-major: [[1,4],[2,5],[3,6]]
    double A[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double S[2], U[9], VT[4];
    int info = 0, lwork;

    cusolverDnDgesvd_bufferSize(h, M, N, &lwork);
    std::vector<double> work(lwork);
    cusolverDnDgesvd(h, 'A', 'A', M, N, A, M, S, U, M, VT, N, work.data(), lwork, nullptr, &info);

    bool ok = (info == 0);
    // Singular values should be positive and S[0] > S[1]
    ok &= (S[0] > S[1] && S[1] > 0);
    // Known: S[0] ≈ 9.508, S[1] ≈ 0.773
    ok &= (std::fabs(S[0] - 9.508) < 0.01);

    cusolverDnDestroy(h);
    check("gesvd SVD (3x2)", ok);
}

// ── Test: syevd (eigenvalue decomposition) ──────────────────────────────────
static void test_syevd() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    const int N = 2;
    // Symmetric: [[2,1],[1,2]] => eigenvalues 1, 3
    double A[4] = {2.0, 1.0, 1.0, 2.0};
    double W[2];
    int info = 0, lwork;

    cusolverDnDsyevd_bufferSize(h, 'V', 'L', N, A, N, W, &lwork);
    std::vector<double> work(lwork);
    cusolverDnDsyevd(h, 'V', 'L', N, A, N, W, work.data(), lwork, &info);

    bool ok = (info == 0);
    // Eigenvalues in ascending order: 1, 3
    ok &= (std::fabs(W[0] - 1.0) < 1e-10 && std::fabs(W[1] - 3.0) < 1e-10);

    cusolverDnDestroy(h);
    check("syevd eigenvalues (2x2 symmetric)", ok);
}

// ── Test: gelsd (least-squares) ──────────────────────────────────────────────
// Overdetermined system: A=[1;1;1; 1;2;3] (3x2 col-major), b=[1;2;3]
// Least-squares solution: x = [0, 1] (approx)
static void test_gelsd() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    const int M = 3, N = 2, NRHS = 1;
    double A[6] = {1.0, 1.0, 1.0, 1.0, 2.0, 3.0}; // column-major
    double B[3] = {1.0, 2.0, 3.0};
    double S[2];
    int rank_out = 0, info = 0, lwork;
    double rcond = -1.0;

    cusolverDnDgelsd_bufferSize(h, M, N, NRHS, A, M, B, M, S, &rcond, &rank_out, &lwork);
    std::vector<double> work(lwork);
    cusolverDnDgelsd(h, M, N, NRHS, A, M, B, M, S, &rcond, &rank_out, work.data(), lwork, nullptr, &info);

    bool ok = (info == 0);
    // Solution: B[0..1] ≈ [0, 1] (intercept ≈ 0, slope ≈ 1)
    ok &= (std::fabs(B[0]) < 0.1 && std::fabs(B[1] - 1.0) < 0.1);
    ok &= (rank_out == 2); // full rank

    cusolverDnDestroy(h);
    check("gelsd least-squares (3x2 overdetermined)", ok);
}

// ── Test: potrf failure on non-positive-definite matrix ─────────────────────
// A = [[-1,0],[0,1]] is not positive-definite; Cholesky must fail at j=0
// (A[0,0] = -1 ≤ 0), so devInfo must equal 1 (LAPACK convention: j+1).
// Invariant: spotrf_ sets info = j+1 when A[j,j] ≤ 0 after subtracting sum of
// squares of previous column entries; here j=0 so info = 1.
static void test_potrf_not_pd() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    float A[4] = {-1.0f, 0.0f, 0.0f, 1.0f}; // col-major [[-1,0],[0,1]]
    int info = 0;
    int lwork = 0;
    cusolverDnSpotrf_bufferSize(h, 'L', 2, A, 2, &lwork);
    std::vector<float> work(std::max(lwork, 1));
    cusolverDnSpotrf(h, 'L', 2, A, 2, work.data(), lwork, &info);

    // info must be 1: j=0 fails because A[0,0]=-1 ≤ 0
    bool ok = (info == 1);

    cusolverDnDestroy(h);
    check("potrf non-PD: info=1 for A[0,0]=-1", ok);
}

// ── Test: potrf upper-triangular (UPLO='U') ──────────────────────────────────
// A = [[4,2],[2,3]] stored upper-triangular in col-major: col0=[4,X], col1=[2,3]
// (lower triangle unused). Cholesky U factor: U[0,0]=2, U[0,1]=1, U[1,1]=sqrt(2).
// Invariant (upper Banachiewicz): U[j,j] = sqrt(A[j,j] - Σ_{k<j} U[k,j]²),
//   U[j,i] = (A[j,i] - Σ_{k<j} U[k,j]*U[k,i]) / U[j,j]  for i > j.
static void test_potrf_upper() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    // col-major, upper triangle: A[0,0]=4, A[0,1]=2, A[1,1]=3; lower unused
    double A[4] = {4.0, 0.0, 2.0, 3.0};
    int info = 0;
    int lwork = 0;
    cusolverDnDpotrf_bufferSize(h, 'U', 2, A, 2, &lwork);
    std::vector<double> work(std::max(lwork, 1));
    cusolverDnDpotrf(h, 'U', 2, A, 2, work.data(), lwork, &info);

    bool ok = (info == 0);
    // U[0,0] = sqrt(4) = 2  (A[0] in col-major)
    ok &= (std::fabs(A[0] - 2.0) < 1e-10);
    // U[0,1] = A[0,1] / U[0,0] = 2/2 = 1  (A[2] in col-major: row0,col1)
    ok &= (std::fabs(A[2] - 1.0) < 1e-10);
    // U[1,1] = sqrt(3 - U[0,1]²) = sqrt(2)  (A[3] in col-major)
    ok &= (std::fabs(A[3] - std::sqrt(2.0)) < 1e-10);

    cusolverDnDestroy(h);
    check("potrf upper-triangular: U for [[4,2],[2,3]]", ok);
}

// ── Test: null/invalid parameter rejection ──────────────────────────────────
static void test_invalid_params() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    bool ok = true;
    ok &= (cusolverDnDgetrf(h, 0, 2, nullptr, 2, nullptr, nullptr, nullptr) == CUSOLVER_STATUS_INVALID_VALUE);
    ok &= (cusolverDnDgetrs(h, 0, 2, 1, nullptr, 2, nullptr, nullptr, 2, nullptr) == CUSOLVER_STATUS_INVALID_VALUE);
    ok &= (cusolverDnDgelsd(h, 0, 2, 1, nullptr, 2, nullptr, 2, nullptr, nullptr, nullptr,
                             nullptr, 0, nullptr, nullptr) == CUSOLVER_STATUS_INVALID_VALUE);

    cusolverDnDestroy(h);
    check("Invalid parameter rejection", ok);
}

int main() {
    std::cout << "=== Test: cuSOLVER (LAPACK delegation) ===\n";

    test_handle();
    test_getrf_getrs();
    test_getrf_float();
    test_potrf();
    test_potrf_not_pd();
    test_potrf_upper();
    test_geqrf();
    test_ormqr();
    test_gesvd();
    test_syevd();
    test_gelsd();
    test_invalid_params();

    std::cout << "\n" << g_pass << "/" << g_total << " tests passed.\n";
    return (g_pass == g_total) ? 0 : 1;
}
