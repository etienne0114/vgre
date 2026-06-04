// Tests for QUEUE-57: cuSOLVER sygvd generalized eigenvalue A*x = lambda*B*x.
// For A=diag(3,2,1,4), B=I: eigenvalues = {1,2,3,4}.
#include "vgre/api/cusolver_shim.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

static void test_sygvd_diag_float() {
    cusolverDnHandle_t h; cusolverDnCreate(&h);
    const int n = 4;
    // Column-major 4×4 diagonal matrices
    std::vector<float> A = {3,0,0,0, 0,2,0,0, 0,0,1,0, 0,0,0,4};
    std::vector<float> B = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    std::vector<float> W(n, 0.f);
    int Lwork = 0;
    auto rs = cusolverDnSsygvd_bufferSize(h, 1, 'V', 'L', n, A.data(), n, B.data(), n, W.data(), &Lwork);
    assert(rs == CUSOLVER_STATUS_SUCCESS && Lwork > 0);
    std::vector<float> work(static_cast<size_t>(Lwork), 0.f);
    int info = -1;
    auto r = cusolverDnSsygvd(h, 1, 'V', 'L', n, A.data(), n, B.data(), n, W.data(),
                               work.data(), Lwork, &info);
    assert(r == CUSOLVER_STATUS_SUCCESS);
    assert(info == 0);
    // eigenvalues should be {1,2,3,4} (syevd returns in ascending order)
    std::sort(W.begin(), W.end());
    float expected[] = {1.f, 2.f, 3.f, 4.f};
    for (int i = 0; i < n; ++i)
        assert(std::fabs(W[i] - expected[i]) < 5e-4f);
    cusolverDnDestroy(h);
    printf("[PASS] QUEUE-57 Ssygvd itype=1 eigenvalues {1,2,3,4}\n");
}

int main() {
    test_sygvd_diag_float();
    printf("All QUEUE-57 sygvd tests passed.\n");
    return 0;
}
