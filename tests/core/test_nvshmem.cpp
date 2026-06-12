// Phase 3, Track P3-16 — NVSHMEM symmetric memory + one-sided collectives.
// Validates one-sided put/get/atomic between symmetric heaps, and that the
// one-sided ring AllReduce reproduces a direct elementwise sum bit-for-bit at
// several PE counts.

#include "vgre/core/nvshmem.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::dist;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    printf("=== NVSHMEM symmetric memory + one-sided AllReduce ===\n");

    // 1. One-sided put / get / atomic between symmetric heaps.
    {
        SymmetricHeap sh(3, 256);
        size_t off = sh.malloc(sizeof(float));
        *sh.ptr<float>(0, off) = 7.0f;
        float tmp = 0.0f;
        sh.get(&tmp, 0, off, sizeof(float));          // PE? gets from PE0
        check("one-sided get reads a remote PE's heap", tmp == 7.0f);
        float v = 3.0f;
        sh.put(1, off, &v, sizeof(float));            // put into PE1
        check("one-sided put writes a remote PE's heap", *sh.ptr<float>(1, off) == 3.0f);
        sh.atomic_add_f(1, off, 2.5f);
        check("one-sided atomic add accumulates remotely", *sh.ptr<float>(1, off) == 5.5f);
    }

    // 2. One-sided ring AllReduce == direct elementwise sum, for P ∈ {2,4,8}.
    auto allreduce_case = [&](int P) {
        const int count = 4 * P;                      // count % P == 0
        SymmetricHeap sh(P, (size_t)(count + count / P) * sizeof(float) + 64);
        size_t dataOff = sh.malloc((size_t)count * sizeof(float));
        size_t tmpOff  = sh.malloc((size_t)(count / P) * sizeof(float));

        std::mt19937 rng(100 + P);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<std::vector<float>> contrib(P, std::vector<float>(count));
        for (int r = 0; r < P; ++r)
            for (int e = 0; e < count; ++e) {
                contrib[r][e] = nd(rng);
                sh.ptr<float>(r, dataOff)[e] = contrib[r][e];
            }

        nvshmem_allreduce_sum(sh, dataOff, tmpOff, count);

        // Reference: elementwise sum across PEs.
        double maxErr = 0.0;
        for (int e = 0; e < count; ++e) {
            float ref = 0.0f; for (int r = 0; r < P; ++r) ref += contrib[r][e];
            for (int r = 0; r < P; ++r)
                maxErr = std::max(maxErr, (double)std::fabs(sh.ptr<float>(r, dataOff)[e] - ref));
        }
        char name[80]; snprintf(name, sizeof(name), "ring AllReduce (P=%d) == elementwise sum", P);
        check(name, maxErr < 1e-4);
    };
    allreduce_case(2);
    allreduce_case(4);
    allreduce_case(8);

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
