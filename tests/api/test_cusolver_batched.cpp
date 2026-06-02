/**
 * Test: cuSolver batched APIs — potrfBatched and getrsBatched correctness
 *
 * cusolverDnSpotrfBatched: Cholesky factorization of a batch of SPD matrices.
 * cusolverDnSgetrsBatched: solve a batch of linear systems using LU factors
 *                           from getrf (devIpivArray is laid out as batchSize*n).
 */

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

static bool approx(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

// ── Handle lifecycle ──────────────────────────────────────────────────────────
static void test_handle() {
    cusolverDnHandle_t h = nullptr;
    bool ok = (cusolverDnCreate(&h) == CUSOLVER_STATUS_SUCCESS);
    ok &= (h != nullptr);
    ok &= (cusolverDnDestroy(h) == CUSOLVER_STATUS_SUCCESS);
    ok &= (cusolverDnCreate(nullptr) == CUSOLVER_STATUS_INVALID_VALUE);
    check("handle create/destroy / null rejection", ok);
}

// ── potrfBatched: two 2×2 SPD matrices ───────────────────────────────────────
// A0 = I₂ → Chol(A0) = I₂
// A1 = [[4,2],[2,3]] (col-major: [4,2,2,3]) → L = [[2,0],[1,√2]]
static void test_potrf_batched() {
    cusolverDnHandle_t h = nullptr;
    cusolverDnCreate(&h);

    const int N = 2, BATCH = 2;
    float A0[4] = {1.f, 0.f, 0.f, 1.f};           // I
    float A1[4] = {4.f, 2.f, 2.f, 3.f};            // col-major [[4,2],[2,3]]
    float *Aarray[BATCH] = {A0, A1};
    int info[BATCH] = {-1, -1};

    bool ok = (cusolverDnSpotrfBatched(h, 'L', N, Aarray, N, info, BATCH)
               == CUSOLVER_STATUS_SUCCESS);
    ok &= (info[0] == 0 && info[1] == 0);

    if (ok) {
        // A0: Chol(I) = I → L[0,0]=1, L[1,0]=0, L[1,1]=1
        ok &= approx(A0[0], 1.f) && approx(A0[1], 0.f) && approx(A0[3], 1.f);
        // A1: L[0,0]=2, L[1,0]=1, L[1,1]=√2
        ok &= approx(A1[0], 2.f) && approx(A1[1], 1.f);
        ok &= approx(A1[3], std::sqrt(2.f));
        if (!ok) {
            std::cerr << "  A0=[" << A0[0] << "," << A0[1] << "," << A0[2] << "," << A0[3] << "]\n";
            std::cerr << "  A1=[" << A1[0] << "," << A1[1] << "," << A1[2] << "," << A1[3] << "]\n";
        }
    }

    cusolverDnDestroy(h);
    check("potrfBatched: I and SPD [[4,2],[2,3]]", ok);
}

// ── potrfBatched: diagonal matrix ────────────────────────────────────────────
// diag(4,9) → Chol = diag(2,3)
static void test_potrf_batched_diagonal() {
    cusolverDnHandle_t h = nullptr;
    cusolverDnCreate(&h);

    const int N = 2, BATCH = 1;
    // col-major diag(4,9): col0=[4,0], col1=[0,9]
    float A[4] = {4.f, 0.f, 0.f, 9.f};
    float *Aarray[BATCH] = {A};
    int info[BATCH] = {-1};

    bool ok = (cusolverDnSpotrfBatched(h, 'L', N, Aarray, N, info, BATCH)
               == CUSOLVER_STATUS_SUCCESS);
    ok &= (info[0] == 0);
    if (ok) {
        ok &= approx(A[0], 2.f) && approx(A[1], 0.f) && approx(A[3], 3.f);
        if (!ok) std::cerr << "  A=[" << A[0] << "," << A[1] << "," << A[2] << "," << A[3] << "]\n";
    }

    cusolverDnDestroy(h);
    check("potrfBatched: diag(4,9) → Chol = diag(2,3)", ok);
}

// ── potrfBatched: null-input rejection ───────────────────────────────────────
static void test_potrf_batched_null() {
    cusolverDnHandle_t h = nullptr;
    cusolverDnCreate(&h);

    const int N = 2, BATCH = 1;
    float A[4] = {1.f, 0.f, 0.f, 1.f};
    float *Aarray[1] = {A};
    int info[1] = {0};

    // null Aarray
    bool ok = (cusolverDnSpotrfBatched(h, 'L', N, nullptr, N, info, BATCH)
               != CUSOLVER_STATUS_SUCCESS);
    // null infoArray
    ok &= (cusolverDnSpotrfBatched(h, 'L', N, Aarray, N, nullptr, BATCH)
           != CUSOLVER_STATUS_SUCCESS);
    // zero n
    ok &= (cusolverDnSpotrfBatched(h, 'L', 0, Aarray, N, info, BATCH)
           != CUSOLVER_STATUS_SUCCESS);
    // zero batchSize
    ok &= (cusolverDnSpotrfBatched(h, 'L', N, Aarray, N, info, 0)
           != CUSOLVER_STATUS_SUCCESS);

    cusolverDnDestroy(h);
    check("potrfBatched: rejects null Aarray/info and zero n/batch", ok);
}

// ── getrsBatched: getrf then getrsBatched ─────────────────────────────────────
// Solve A*x = b where A = [[2,4],[1,3]], b = [10,6] → x = [3,1]
// and   A = [[1,0],[0,1]] (identity), b = [5,7]     → x = [5,7]
//
// We run getrf on each A first to get the LU factors + pivots, then call
// getrsBatched with those factors and a flat pivot array (BATCH*N elements).
static void test_getrs_batched() {
    cusolverDnHandle_t h = nullptr;
    cusolverDnCreate(&h);

    const int N = 2, NRHS = 1, BATCH = 2;

    // System 0: A0 = [[2,4],[1,3]] col-major: [2,1,4,3], b0 = [10,6] → x=[3,1]
    float A0[4] = {2.f, 1.f, 4.f, 3.f};
    float B0[2] = {10.f, 6.f};

    // System 1: A1 = I, b1 = [5,7] → x=[5,7]
    float A1[4] = {1.f, 0.f, 0.f, 1.f};
    float B1[2] = {5.f, 7.f};

    // Factorize each with getrf to get LU factors and pivots
    int ipiv0[N], ipiv1[N], info0 = 0, info1 = 0;
    {
        int lwork = 0;
        cusolverDnSgetrf_bufferSize(h, N, N, A0, N, &lwork);
        std::vector<float> work(std::max(lwork, 1));
        cusolverDnSgetrf(h, N, N, A0, N, work.data(), ipiv0, &info0);
    }
    {
        int lwork = 0;
        cusolverDnSgetrf_bufferSize(h, N, N, A1, N, &lwork);
        std::vector<float> work(std::max(lwork, 1));
        cusolverDnSgetrf(h, N, N, A1, N, work.data(), ipiv1, &info1);
    }

    bool ok = (info0 == 0 && info1 == 0);

    if (ok) {
        // Build flat pivot array: BATCH * N elements (batch 0 first, then batch 1)
        int ipivAll[BATCH * N];
        ipivAll[0] = ipiv0[0]; ipivAll[1] = ipiv0[1];
        ipivAll[2] = ipiv1[0]; ipivAll[3] = ipiv1[1];

        const float *Aarray[BATCH] = {A0, A1};
        float       *Barray[BATCH] = {B0, B1};
        int info = -1;

        ok = (cusolverDnSgetrsBatched(h, 0 /*no-trans*/, N, NRHS,
                                       Aarray, N, ipivAll, Barray, N,
                                       &info, BATCH)
              == CUSOLVER_STATUS_SUCCESS);
        ok &= (info == 0);

        if (ok) {
            // System 0: x = [3, 1]
            ok &= approx(B0[0], 3.f) && approx(B0[1], 1.f);
            // System 1: x = [5, 7]
            ok &= approx(B1[0], 5.f) && approx(B1[1], 7.f);
            if (!ok) {
                std::cerr << "  B0=[" << B0[0] << "," << B0[1] << "]"
                          << " B1=[" << B1[0] << "," << B1[1] << "]\n";
            }
        }
    }

    cusolverDnDestroy(h);
    check("getrsBatched: 2-system batch via getrf+getrsBatched", ok);
}

// ── getrsBatched: null-input rejection ───────────────────────────────────────
static void test_getrs_batched_null() {
    cusolverDnHandle_t h = nullptr;
    cusolverDnCreate(&h);

    const int N = 2, NRHS = 1, BATCH = 1;
    float A[4] = {1.f, 0.f, 0.f, 1.f};
    float B[2] = {1.f, 2.f};
    const float *Aarray[1] = {A};
    float       *Barray[1] = {B};
    int ipiv[N] = {1, 2};
    int info = 0;

    // null Aarray
    bool ok = (cusolverDnSgetrsBatched(h, 0, N, NRHS,
                                        nullptr, N, ipiv, Barray, N, &info, BATCH)
               != CUSOLVER_STATUS_SUCCESS);
    // null Barray
    ok &= (cusolverDnSgetrsBatched(h, 0, N, NRHS,
                                    Aarray, N, ipiv, nullptr, N, &info, BATCH)
           != CUSOLVER_STATUS_SUCCESS);
    // null devIpivArray
    ok &= (cusolverDnSgetrsBatched(h, 0, N, NRHS,
                                    Aarray, N, nullptr, Barray, N, &info, BATCH)
           != CUSOLVER_STATUS_SUCCESS);
    // null info
    ok &= (cusolverDnSgetrsBatched(h, 0, N, NRHS,
                                    Aarray, N, ipiv, Barray, N, nullptr, BATCH)
           != CUSOLVER_STATUS_SUCCESS);
    // zero batchSize
    ok &= (cusolverDnSgetrsBatched(h, 0, N, NRHS,
                                    Aarray, N, ipiv, Barray, N, &info, 0)
           != CUSOLVER_STATUS_SUCCESS);

    cusolverDnDestroy(h);
    check("getrsBatched: rejects null arrays/info and zero batch", ok);
}

// ── getrsBatched: single system correctness (3×3) ────────────────────────────
// A = [[1,0,0],[2,1,0],[3,2,1]] (lower triangular), b = [1,4,10] → x = [1,2,3]
// col-major: col0=[1,2,3], col1=[0,1,2], col2=[0,0,1]
static void test_getrs_batched_3x3() {
    cusolverDnHandle_t h = nullptr;
    cusolverDnCreate(&h);

    const int N = 3, NRHS = 1, BATCH = 1;
    float A[9] = {1.f, 2.f, 3.f,   0.f, 1.f, 2.f,   0.f, 0.f, 1.f};
    float B[3] = {1.f, 4.f, 10.f};

    int ipiv[N], info0 = 0;
    {
        int lwork = 0;
        cusolverDnSgetrf_bufferSize(h, N, N, A, N, &lwork);
        std::vector<float> work(std::max(lwork, 1));
        cusolverDnSgetrf(h, N, N, A, N, work.data(), ipiv, &info0);
    }

    bool ok = (info0 == 0);
    if (ok) {
        const float *Aarray[BATCH] = {A};
        float       *Barray[BATCH] = {B};
        int info = -1;
        ok = (cusolverDnSgetrsBatched(h, 0, N, NRHS,
                                       Aarray, N, ipiv, Barray, N,
                                       &info, BATCH)
              == CUSOLVER_STATUS_SUCCESS);
        ok &= (info == 0);
        if (ok) {
            ok &= approx(B[0], 1.f) && approx(B[1], 2.f) && approx(B[2], 3.f);
            if (!ok)
                std::cerr << "  x=[" << B[0] << "," << B[1] << "," << B[2] << "]\n";
        }
    }

    cusolverDnDestroy(h);
    check("getrsBatched: 3×3 single system x=[1,2,3]", ok);
}

// ── CpotrfBatched: complex float batch Cholesky ──────────────────────────────
// Batch of two 2×2 Hermitian PD matrices (real-valued for simplicity):
//   A0 = I₂  → Chol(A0) = I₂
//   A1 = [[4,1],[1,3]] col-major → L with L[0,0]=2, L[1,0]=0.5, L[1,1]=sqrt(11)/2
// Storage: complex float = pairs {re, im} interleaved; all imaginary parts zero.
// Invariant (Banachiewicz lower): L[j,j] = sqrt(A[j,j] - Σ_{k<j} |L[j,k]|²),
//   L[i,j] = (A[i,j] - Σ_{k<j} L[i,k]*conj(L[j,k])) / L[j,j].
static void test_Cpotrf_batched() {
    cusolverDnHandle_t h = nullptr;
    cusolverDnCreate(&h);

    const int N = 2, BATCH = 2;
    // Complex float: each element = {re, im}; col-major 2×2 = 4 elements = 8 floats
    // A0 = identity: col0=[(1,0),(0,0)], col1=[(0,0),(1,0)]
    float A0[8] = {1.f,0.f,  0.f,0.f,  0.f,0.f,  1.f,0.f};
    // A1 = [[4,1],[1,3]]: col0=[(4,0),(1,0)], col1=[(1,0),(3,0)]
    float A1[8] = {4.f,0.f,  1.f,0.f,  1.f,0.f,  3.f,0.f};
    float *Aarray[BATCH] = {A0, A1};
    int info[BATCH] = {-1, -1};

    bool ok = (cusolverDnCpotrfBatched(h, 'L', N, Aarray, N, info, BATCH)
               == CUSOLVER_STATUS_SUCCESS);
    ok &= (info[0] == 0 && info[1] == 0);

    if (ok) {
        // A0 Chol = I: L[0,0]=1, L[1,0]=0, L[1,1]=1
        ok &= approx(A0[0], 1.f);  // re(L[0,0])
        ok &= approx(A0[2], 0.f);  // re(L[1,0])
        ok &= approx(A0[6], 1.f);  // re(L[1,1])
        // A1 Chol: L[0,0]=2, L[1,0]=0.5, L[1,1]=sqrt(11)/2 ≈ 1.6583
        ok &= approx(A1[0], 2.f);
        ok &= approx(A1[2], 0.5f);
        ok &= approx(A1[6], std::sqrt(11.f) / 2.f);
        if (!ok) {
            std::cerr << "  A0=[" << A0[0] << "," << A0[2] << "," << A0[6] << "]\n";
            std::cerr << "  A1=[" << A1[0] << "," << A1[2] << "," << A1[6] << "]\n";
        }
    }

    cusolverDnDestroy(h);
    check("CpotrfBatched: I and HPD [[4,1],[1,3]] complex float", ok);
}

// ── ZpotrfBatched: complex double batch Cholesky ─────────────────────────────
// Same two matrices as above but with complex double precision.
// Invariant: same Banachiewicz formula as CpotrfBatched, higher precision.
static void test_Zpotrf_batched() {
    cusolverDnHandle_t h = nullptr;
    cusolverDnCreate(&h);

    const int N = 2, BATCH = 2;
    // Complex double: each element = {re, im}; 4 elements per matrix = 8 doubles
    double A0[8] = {1.0,0.0,  0.0,0.0,  0.0,0.0,  1.0,0.0};
    double A1[8] = {4.0,0.0,  1.0,0.0,  1.0,0.0,  3.0,0.0};
    double *Aarray[BATCH] = {A0, A1};
    int info[BATCH] = {-1, -1};

    bool ok = (cusolverDnZpotrfBatched(h, 'L', N, Aarray, N, info, BATCH)
               == CUSOLVER_STATUS_SUCCESS);
    ok &= (info[0] == 0 && info[1] == 0);

    if (ok) {
        // A0: L[0,0]=1, L[1,0]=0, L[1,1]=1
        ok &= (std::fabs(A0[0] - 1.0) < 1e-9);
        ok &= (std::fabs(A0[2] - 0.0) < 1e-9);
        ok &= (std::fabs(A0[6] - 1.0) < 1e-9);
        // A1: L[0,0]=2, L[1,0]=0.5, L[1,1]=sqrt(11)/2
        ok &= (std::fabs(A1[0] - 2.0) < 1e-9);
        ok &= (std::fabs(A1[2] - 0.5) < 1e-9);
        ok &= (std::fabs(A1[6] - std::sqrt(11.0) / 2.0) < 1e-9);
        if (!ok) {
            std::cerr << "  A0=[" << A0[0] << "," << A0[2] << "," << A0[6] << "]\n";
            std::cerr << "  A1=[" << A1[0] << "," << A1[2] << "," << A1[6] << "]\n";
        }
    }

    cusolverDnDestroy(h);
    check("ZpotrfBatched: I and HPD [[4,1],[1,3]] complex double", ok);
}

// ── potrfBatched: mixed PD/non-PD batch ──────────────────────────────────────
// Verifies per-batch infoArray: A0 PD (success), A1 not PD (fail at j=0).
// A0 = diag(4,4) → infoArray[0] = 0
// A1 = [[-1,0],[0,1]] col-major → infoArray[1] = 1 (A[0,0]=-1 ≤ 0, j+1=1)
// Invariant: Banachiewicz sets info=j+1 when diagonal entry ≤ 0 after subtracting
// sum of squares of previous entries in the same column.
static void test_potrf_batched_mixed_pd() {
    cusolverDnHandle_t h = nullptr;
    cusolverDnCreate(&h);

    const int N = 2, BATCH = 2;
    float A0[4] = {4.f, 0.f, 0.f, 4.f};   // diag(4,4), PD
    float A1[4] = {-1.f, 0.f, 0.f, 1.f};  // [[-1,0],[0,1]], not PD
    float *Aarray[BATCH] = {A0, A1};
    int info[BATCH] = {-1, -1};

    bool ok = (cusolverDnSpotrfBatched(h, 'L', N, Aarray, N, info, BATCH)
               == CUSOLVER_STATUS_SUCCESS);
    // A0 succeeded: info[0] = 0
    ok &= (info[0] == 0);
    // A1 failed at j=0 (A[0,0]=-1 ≤ 0): info[1] = 1
    ok &= (info[1] == 1);

    if (!ok) {
        std::cerr << "  info[0]=" << info[0] << " info[1]=" << info[1] << "\n";
    }

    cusolverDnDestroy(h);
    check("potrfBatched mixed PD/non-PD: info[0]=0 info[1]=1", ok);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main()
{
    std::cout << "=== Test: cuSolver Batched APIs (potrfBatched / getrsBatched) ===\n";

    test_handle();
    test_potrf_batched();
    test_potrf_batched_diagonal();
    test_potrf_batched_null();
    test_getrs_batched();
    test_getrs_batched_null();
    test_getrs_batched_3x3();
    test_Cpotrf_batched();
    test_Zpotrf_batched();
    test_potrf_batched_mixed_pd();

    std::cout << "\nResult: " << g_pass << "/" << g_total << " passed\n";
    return (g_pass == g_total) ? 0 : 1;
}
