// cuSOLVER complex (C/Z prefix) test — validates:
//   1. cusolverDnCgetrf / cusolverDnCgetrs  (LU solve, complex float)
//   2. cusolverDnCpotrf / cusolverDnCpotrs  (Cholesky, complex float)
//   3. cusolverDnCgesvd                     (SVD, complex float)
//   4. cusolverDnCheevd                     (Hermitian eigenvalue, complex float)

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

// ── Test 1: Complex LU solve (Cgetrf + Cgetrs) ──────────────────────────────
// Solve A*x = b where
//   A = [[1+0i, 2+0i],
//        [0+1i, 0-1i]]  (column-major: col0=[1,0i], col1=[2,-1i])
//   b = [3+0i, 1+1i]
// Expected (verify A*x = b after solve).
static void test_Cgetrf_Cgetrs() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    const int N = 2, NRHS = 1;
    // Column-major complex float: each element = {re, im}
    // A col-major: col0 = [(1,0),(0,1)], col1 = [(2,0),(0,-1)]
    // Storage: [re(A[0,0]), im(A[0,0]), re(A[1,0]), im(A[1,0]), re(A[0,1]), im(A[0,1]), re(A[1,1]), im(A[1,1])]
    float A[8] = {
        1.f, 0.f,   // A[0,0] = 1+0i
        0.f, 1.f,   // A[1,0] = 0+1i
        2.f, 0.f,   // A[0,1] = 2+0i
        0.f,-1.f    // A[1,1] = 0-1i
    };
    float B[4] = {
        3.f, 0.f,   // b[0] = 3+0i
        1.f, 1.f    // b[1] = 1+1i
    };
    // Save original A for verification
    float A_orig[8];
    memcpy(A_orig, A, sizeof(A));

    int ipiv[2], info = 0;
    int lwork = 0;
    cusolverDnCgetrf_bufferSize(h, N, N, A, N, &lwork);

    std::vector<float> workspace(lwork > 0 ? lwork * 2 : 2, 0.f);
    cusolverDnCgetrf(h, N, N, A, N, workspace.data(), ipiv, &info);
    bool ok = (info == 0);

    cusolverDnCgetrs(h, 0 /*no-trans*/, N, NRHS, A, N, ipiv, B, N, &info);
    ok &= (info == 0);

    // Verify: A_orig * x = b_orig
    // A_orig * x[0] + A_orig * x[1] etc. in complex arithmetic
    // x[0] = B[0] + i*B[1], x[1] = B[2] + i*B[3]
    float xr0 = B[0], xi0 = B[1];
    float xr1 = B[2], xi1 = B[3];

    // A_orig col-major: A[j,i] at A_orig[(i*N+j)*2]
    // r0 = A[0,0]*x[0] + A[0,1]*x[1]
    float r0_re = (A_orig[0]*xr0 - A_orig[1]*xi0) + (A_orig[4]*xr1 - A_orig[5]*xi1);
    float r0_im = (A_orig[0]*xi0 + A_orig[1]*xr0) + (A_orig[4]*xi1 + A_orig[5]*xr1);
    float r1_re = (A_orig[2]*xr0 - A_orig[3]*xi0) + (A_orig[6]*xr1 - A_orig[7]*xi1);
    float r1_im = (A_orig[2]*xi0 + A_orig[3]*xr0) + (A_orig[6]*xi1 + A_orig[7]*xr1);

    ok &= (std::fabs(r0_re - 3.f) < 1e-5f && std::fabs(r0_im) < 1e-5f);
    ok &= (std::fabs(r1_re - 1.f) < 1e-5f && std::fabs(r1_im - 1.f) < 1e-5f);

    cusolverDnDestroy(h);
    check("Cgetrf+Cgetrs complex LU solve", ok);
}

// ── Test 2: Complex Cholesky (Cpotrf + Cpotrs) ──────────────────────────────
// A = [[4+0i, 1+0i], [1+0i, 3+0i]] (Hermitian positive-definite, real for simplicity)
// Solve A*x = b = [1,1]
static void test_Cpotrf_Cpotrs() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    const int N = 2, NRHS = 1;
    // Column-major complex float (real values, im=0)
    // col0=[(4,0),(1,0)], col1=[(1,0),(3,0)]
    float A[8] = {
        4.f, 0.f,   // A[0,0]
        1.f, 0.f,   // A[1,0]
        1.f, 0.f,   // A[0,1]
        3.f, 0.f    // A[1,1]
    };
    float B[4] = {1.f, 0.f, 1.f, 0.f};  // b = [1+0i, 1+0i]

    int lwork = 0, info = 0;
    cusolverDnCpotrf_bufferSize(h, 'L', N, A, N, &lwork);
    std::vector<float> workspace(lwork > 0 ? lwork * 2 : 2, 0.f);
    cusolverDnCpotrf(h, 'L', N, A, N, workspace.data(), lwork, &info);
    bool ok = (info == 0);

    cusolverDnCpotrs(h, 'L', N, NRHS, A, N, B, N, &info);
    ok &= (info == 0);

    // Verify: A*x = b_orig
    // A = [[4,1],[1,3]], b = [1,1], so 4x0+x1=1, x0+3x1=1 => x0=2/11, x1=3/11
    float xr0 = B[0], xr1 = B[2];
    float expected0 = 2.f / 11.f, expected1 = 3.f / 11.f;
    ok &= (std::fabs(xr0 - expected0) < 1e-4f);
    ok &= (std::fabs(xr1 - expected1) < 1e-4f);
    // Imaginary parts should be (near) zero
    ok &= (std::fabs(B[1]) < 1e-5f && std::fabs(B[3]) < 1e-5f);

    cusolverDnDestroy(h);
    check("Cpotrf+Cpotrs complex Cholesky solve", ok);
}

// ── Test 3: Complex SVD (Cgesvd) ────────────────────────────────────────────
// 2x2 identity-like complex matrix [[1+0i,0],[0,1+0i]] — singular values = [1,1]
static void test_Cgesvd() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    const int M = 2, N = 2;
    // Identity matrix in complex float column-major
    float A[8] = {
        1.f, 0.f,  0.f, 0.f,   // col0: [(1,0),(0,0)]
        0.f, 0.f,  1.f, 0.f    // col1: [(0,0),(1,0)]
    };
    float S[2] = {0.f};
    float U[8] = {0.f};
    float VT[8] = {0.f};

    int lwork = 0, info = 0;
    cusolverDnCgesvd_bufferSize(h, M, N, &lwork);
    std::vector<float> work(lwork > 0 ? lwork * 2 : 10, 0.f);

    cusolverDnCgesvd(h, 'A', 'A', M, N, A, M, S, U, M, VT, N,
                     work.data(), lwork, nullptr, &info);
    bool ok = (info == 0);

    // Singular values should be real, non-negative, and sorted descending
    ok &= (S[0] >= 0.f && S[1] >= 0.f);
    ok &= (S[0] >= S[1] - 1e-5f);
    // For identity, both singular values should be ~1
    ok &= (std::fabs(S[0] - 1.f) < 1e-4f);
    ok &= (std::fabs(S[1] - 1.f) < 1e-4f);

    cusolverDnDestroy(h);
    check("Cgesvd complex SVD singular values", ok);
}

// ── Test 4: Complex Hermitian eigenvalue (Cheevd) ───────────────────────────
// A = [[2+0i, 1+0i],[1+0i, 2+0i]] (real symmetric) — eigenvalues = 1 and 3
static void test_Cheevd() {
    cusolverDnHandle_t h;
    cusolverDnCreate(&h);

    const int N = 2;
    // Column-major complex float
    // [[2,1],[1,2]]: col0=[(2,0),(1,0)], col1=[(1,0),(2,0)]
    float A[8] = {
        2.f, 0.f,   // A[0,0]
        1.f, 0.f,   // A[1,0]
        1.f, 0.f,   // A[0,1]
        2.f, 0.f    // A[1,1]
    };
    float W[2] = {0.f};

    int lwork = 0, info = 0;
    cusolverDnCheevd_bufferSize(h, 'V', 'L', N, A, N, W, &lwork);
    std::vector<float> work(lwork > 0 ? lwork * 2 : 20, 0.f);

    cusolverDnCheevd(h, 'V', 'L', N, A, N, W, work.data(), lwork, &info);
    bool ok = (info == 0);

    // Eigenvalues should be 1 and 3 (ascending order from LAPACK)
    ok &= (std::fabs(W[0] - 1.f) < 1e-4f);
    ok &= (std::fabs(W[1] - 3.f) < 1e-4f);

    cusolverDnDestroy(h);
    check("Cheevd complex Hermitian eigenvalues [1,3]", ok);
}

int main() {
    std::cout << "=== cuSOLVER Complex (C/Z prefix) Tests ===\n";
    test_Cgetrf_Cgetrs();
    test_Cpotrf_Cpotrs();
    test_Cgesvd();
    test_Cheevd();
    std::cout << "\n" << g_pass << "/" << g_total << " tests passed.\n";
    return (g_pass == g_total) ? 0 : 1;
}
