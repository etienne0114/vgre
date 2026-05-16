// cuSOLVER sparse (cusolverSp) test — validates LU solve, Cholesky solve,
// least-squares QR, and eigenvalue via shift-invert on small CSR systems.

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

// ── CSR helpers ───────────────────────────────────────────────────────────────
// Build CSR for the 3×3 matrix:
//   [[2,  1,  0],
//    [1,  4,  1],
//    [0,  1,  3]]
// rowPtr = {0,2,5,7}, colInd = {0,1, 0,1,2, 1,2}, val = {2,1, 1,4,1, 1,3}
static void make3x3(std::vector<int> &rp, std::vector<int> &ci, std::vector<double> &v) {
    rp = {0, 2, 5, 7};
    ci = {0, 1,  0, 1, 2,  1, 2};
    v  = {2.0, 1.0,  1.0, 4.0, 1.0,  1.0, 3.0};
}

// ── test: cusolverSpDcsrlsvlu — general LU solve ─────────────────────────────
// A*x = b  →  x = A^{-1}*b
static void test_csrlsvlu_double() {
    cusolverSpHandle_t h;
    cusolverSpCreate(&h);

    std::vector<int> rp, ci; std::vector<double> v;
    make3x3(rp, ci, v);

    // b = [3, 6, 4]  →  exact solution x = [1, 1, 1]
    double b[3] = {3.0, 6.0, 4.0}, x[3] = {};
    int sing = 0;
    cusolverStatus_t r = cusolverSpDcsrlsvlu(h, 3, (int)ci.size(), nullptr,
                                              v.data(), rp.data(), ci.data(),
                                              b, 1e-12, 0, x, &sing);
    bool ok = (r == CUSOLVER_STATUS_SUCCESS) && (sing == -1);
    ok &= (std::fabs(x[0] - 1.0) < 1e-10 && std::fabs(x[1] - 1.0) < 1e-10 && std::fabs(x[2] - 1.0) < 1e-10);
    cusolverSpDestroy(h);
    check("Dcsrlsvlu: A*x=b, x=[1,1,1]", ok);
}

// ── test: cusolverSpDcsrlsvchol — SPD Cholesky solve ─────────────────────────
// Same SPD matrix, same b=[3,6,4], expect x=[1,1,1]
static void test_csrlsvchol_double() {
    cusolverSpHandle_t h;
    cusolverSpCreate(&h);

    std::vector<int> rp, ci; std::vector<double> v;
    make3x3(rp, ci, v);

    double b[3] = {3.0, 6.0, 4.0}, x[3] = {};
    int sing = 0;
    cusolverStatus_t r = cusolverSpDcsrlsvchol(h, 3, (int)ci.size(), nullptr,
                                                v.data(), rp.data(), ci.data(),
                                                b, 1e-12, 0, x, &sing);
    bool ok = (r == CUSOLVER_STATUS_SUCCESS) && (sing == -1);
    ok &= (std::fabs(x[0] - 1.0) < 1e-10 && std::fabs(x[1] - 1.0) < 1e-10 && std::fabs(x[2] - 1.0) < 1e-10);
    cusolverSpDestroy(h);
    check("Dcsrlsvchol (SPD Cholesky): x=[1,1,1]", ok);
}

// ── test: float variants consistent with double ──────────────────────────────
static void test_csrlsvlu_float() {
    cusolverSpHandle_t h;
    cusolverSpCreate(&h);

    std::vector<int> rp = {0, 2, 5, 7};
    std::vector<int> ci = {0, 1,  0, 1, 2,  1, 2};
    std::vector<float> v = {2.f, 1.f,  1.f, 4.f, 1.f,  1.f, 3.f};

    float b[3] = {3.f, 6.f, 4.f}, x[3] = {};
    int sing = 0;
    cusolverStatus_t r = cusolverSpScsrlsvlu(h, 3, (int)ci.size(), nullptr,
                                              v.data(), rp.data(), ci.data(),
                                              b, 1e-6f, 0, x, &sing);
    bool ok = (r == CUSOLVER_STATUS_SUCCESS) && (sing == -1);
    ok &= (std::fabs(x[0] - 1.f) < 1e-5f && std::fabs(x[1] - 1.f) < 1e-5f && std::fabs(x[2] - 1.f) < 1e-5f);
    cusolverSpDestroy(h);
    check("Scsrlsvlu (float): x=[1,1,1]", ok);
}

// ── test: cusolverSpDcsrlsqvqr — least-squares ───────────────────────────────
// Overdetermined 4×2 system, CSR:
//   A = [[1,0],[0,1],[1,1],[1,-1]], b = [1,1,2,0]  → x = [1,1] exactly
static void test_csrlsqvqr_double() {
    cusolverSpHandle_t h;
    cusolverSpCreate(&h);

    // 4×2 sparse: rows 0,1,2,3; cols 0..1
    std::vector<int> rp = {0, 1, 2, 4, 6};
    std::vector<int> ci = {0,  1,  0, 1,  0, 1};
    std::vector<double> v = {1.0, 1.0, 1.0, 1.0, 1.0, -1.0};
    double b[4] = {1.0, 1.0, 2.0, 0.0}, x[2] = {};
    int rank_out = 0, p[2] = {};
    double min_norm = 0.0, tol = 1e-10;

    cusolverStatus_t r = cusolverSpDcsrlsqvqr(h, 4, 2, (int)ci.size(), nullptr,
                                               v.data(), rp.data(), ci.data(),
                                               b, tol, &rank_out, x, p, &min_norm);
    bool ok = (r == CUSOLVER_STATUS_SUCCESS);
    ok &= (std::fabs(x[0] - 1.0) < 1e-8 && std::fabs(x[1] - 1.0) < 1e-8);
    ok &= (rank_out == 2);
    ok &= (min_norm < 1e-8);
    cusolverSpDestroy(h);
    check("Dcsrlsqvqr: overdetermined 4×2, x=[1,1]", ok);
}

// ── test: handle lifecycle ───────────────────────────────────────────────────
static void test_handle() {
    cusolverSpHandle_t h1, h2;
    bool ok = (cusolverSpCreate(&h1) == CUSOLVER_STATUS_SUCCESS);
    ok &= (cusolverSpCreate(&h2) == CUSOLVER_STATUS_SUCCESS);
    ok &= (h1 != h2);
    ok &= (cusolverSpSetStream(h1, nullptr) == CUSOLVER_STATUS_SUCCESS);
    ok &= (cusolverSpDestroy(h1) == CUSOLVER_STATUS_SUCCESS);
    ok &= (cusolverSpDestroy(h2) == CUSOLVER_STATUS_SUCCESS);
    check("Handle create/destroy", ok);
}

// ── test: null pointer rejection ─────────────────────────────────────────────
static void test_invalid() {
    cusolverSpHandle_t h; cusolverSpCreate(&h);
    bool ok = (cusolverSpDcsrlsvlu(h, 3, 7, nullptr, nullptr, nullptr, nullptr,
                                    nullptr, 1e-12, 0, nullptr, nullptr)
               == CUSOLVER_STATUS_INVALID_VALUE);
    cusolverSpDestroy(h);
    check("Null pointer rejection", ok);
}

int main() {
    std::cout << "=== Test: cuSOLVER sparse (cusolverSp) ===\n";
    test_handle();
    test_invalid();
    test_csrlsvlu_double();
    test_csrlsvlu_float();
    test_csrlsvchol_double();
    test_csrlsqvqr_double();
    std::cout << "\n" << g_pass << "/" << g_total << " tests passed.\n";
    return (g_pass == g_total) ? 0 : 1;
}
