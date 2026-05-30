// cuFFT Xt API tests:
// 1. cufftXtMalloc + cufftXtFree
// 2. cufftXtMemcpy HOST_TO_DEVICE -> DEVICE_TO_HOST round-trip
// 3. cufftXtExecDescriptor: 4-element C2C FFT, verify DC component == sum of inputs

#include "vgre/api/cufft_shim.h"

#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <vector>

using cfloat = std::complex<float>;

static int g_pass = 0, g_total = 0;

static void check(const char *name, bool ok) {
    ++g_total;
    if (ok) { ++g_pass; std::cout << "PASS [" << g_total << "] " << name << "\n"; }
    else    { std::cerr << "FAIL [" << g_total << "] " << name << "\n"; }
}

// ── Test 1: cufftXtMalloc + cufftXtFree ──────────────────────────────────────
static void test_xt_malloc_free() {
    const int N = 16;
    cufftHandle plan;
    cufftResult_t r = cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    check("cufftPlan1d for Xt malloc", r == CUFFT_SUCCESS);

    cudaLibXtDesc *desc = nullptr;
    r = cufftXtMalloc(plan, &desc, CUFFT_XT_FORMAT_INPUT);
    check("cufftXtMalloc returns SUCCESS", r == CUFFT_SUCCESS);
    check("cufftXtMalloc: descriptor not null", desc != nullptr);
    if (desc) {
        check("cufftXtMalloc: nGPUs == 1", desc->nGPUs == 1);
        check("cufftXtMalloc: GPU[0] == 0",  desc->GPUs[0] == 0);
        check("cufftXtMalloc: data[0] not null", desc->data[0] != nullptr);
        check("cufftXtMalloc: size[0] > 0", desc->size[0] > 0);

        r = cufftXtFree(desc);
        check("cufftXtFree returns SUCCESS", r == CUFFT_SUCCESS);
    }

    cufftDestroy(plan);
}

// ── Test 2: cufftXtMemcpy round-trip HOST_TO_DEVICE -> DEVICE_TO_HOST ────────
static void test_xt_memcpy_roundtrip() {
    const int N = 8;
    cufftHandle plan;
    cufftResult_t r = cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    check("cufftPlan1d for Xt memcpy", r == CUFFT_SUCCESS);

    // Fill host source buffer
    std::vector<cfloat> src(N), dst(N, {0.f, 0.f});
    for (int i = 0; i < N; ++i) src[i] = {(float)i, (float)(N - i)};

    // Allocate Xt descriptor
    cudaLibXtDesc *desc = nullptr;
    r = cufftXtMalloc(plan, &desc, CUFFT_XT_FORMAT_INPUT);
    check("cufftXtMalloc for memcpy test", r == CUFFT_SUCCESS);

    if (desc) {
        // HOST -> DEVICE (Xt descriptor)
        r = cufftXtMemcpy(plan, desc, src.data(), CUFFT_COPY_HOST_TO_DEVICE);
        check("cufftXtMemcpy HOST_TO_DEVICE returns SUCCESS", r == CUFFT_SUCCESS);

        // DEVICE (Xt descriptor) -> HOST
        r = cufftXtMemcpy(plan, dst.data(), desc, CUFFT_COPY_DEVICE_TO_HOST);
        check("cufftXtMemcpy DEVICE_TO_HOST returns SUCCESS", r == CUFFT_SUCCESS);

        // Verify round-trip
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            if (src[i].real() != dst[i].real() || src[i].imag() != dst[i].imag()) {
                ok = false; break;
            }
        }
        check("cufftXtMemcpy round-trip data integrity", ok);

        cufftXtFree(desc);
    }

    cufftDestroy(plan);
}

// ── Test 3: cufftXtExecDescriptor C2C FFT: DC == sum of inputs ───────────────
static void test_xt_exec_c2c_dc() {
    const int N = 4;
    cufftHandle plan;
    cufftResult_t r = cufftPlan1d(&plan, N, CUFFT_C2C, 1);
    check("cufftPlan1d for XtExec C2C", r == CUFFT_SUCCESS);

    // Input: {1+0j, 2+0j, 3+0j, 4+0j}
    // Expected DC bin (index 0) = sum = 10+0j
    std::vector<cfloat> h_in = {{1.f,0.f},{2.f,0.f},{3.f,0.f},{4.f,0.f}};
    std::vector<cfloat> h_out(N, {0.f,0.f});

    float expectedDC_re = 0.f;
    for (auto &v : h_in) expectedDC_re += v.real();

    // Allocate input and output Xt descriptors
    cudaLibXtDesc *inDesc  = nullptr;
    cudaLibXtDesc *outDesc = nullptr;
    r = cufftXtMalloc(plan, &inDesc,  CUFFT_XT_FORMAT_INPUT);
    check("cufftXtMalloc for input desc", r == CUFFT_SUCCESS);
    r = cufftXtMalloc(plan, &outDesc, CUFFT_XT_FORMAT_OUTPUT);
    check("cufftXtMalloc for output desc", r == CUFFT_SUCCESS);

    if (inDesc && outDesc) {
        // Copy input data in
        r = cufftXtMemcpy(plan, inDesc, h_in.data(), CUFFT_COPY_HOST_TO_DEVICE);
        check("cufftXtMemcpy input to desc", r == CUFFT_SUCCESS);

        // Execute forward FFT via Xt descriptor
        r = cufftXtExecDescriptor(plan, inDesc, outDesc, CUFFT_FORWARD);
        check("cufftXtExecDescriptor forward", r == CUFFT_SUCCESS);

        // Copy result out
        r = cufftXtMemcpy(plan, h_out.data(), outDesc, CUFFT_COPY_DEVICE_TO_HOST);
        check("cufftXtMemcpy output from desc", r == CUFFT_SUCCESS);

        // DC bin should equal sum of inputs
        float dcErr = std::fabs(h_out[0].real() - expectedDC_re);
        check("Xt C2C FFT DC bin == sum of inputs", dcErr < 0.01f);

        cufftXtFree(inDesc);
        cufftXtFree(outDesc);
    }

    cufftDestroy(plan);
}

// ── Test 4: cufftXtGetSizeMany returns 0 workSize ────────────────────────────
static void test_xt_get_size_many() {
    cufftHandle plan;
    cufftResult_t r = cufftPlan1d(&plan, 16, CUFFT_C2C, 1);
    check("cufftPlan1d for XtGetSizeMany", r == CUFFT_SUCCESS);

    size_t ws = 999;
    int gpu = 0;
    r = cufftXtGetSizeMany(plan, &ws,
                           CUFFT_XT_FORMAT_INPUT, CUFFT_XT_FORMAT_OUTPUT,
                           CUFFT_XT_FORMAT_INPLACE,
                           4 /*CUDA_C_32F*/, 1, &gpu);
    check("cufftXtGetSizeMany returns SUCCESS", r == CUFFT_SUCCESS);
    check("cufftXtGetSizeMany workSize == 0", ws == 0);

    cufftDestroy(plan);
}

// ── Test 5: cufftXtSetCallback / cufftXtClearCallback ────────────────────────
static void test_xt_callback() {
    cufftHandle plan;
    cufftResult_t r = cufftPlan1d(&plan, 16, CUFFT_C2C, 1);
    check("cufftPlan1d for Xt callback", r == CUFFT_SUCCESS);

    void *dummy = nullptr;
    r = cufftXtSetCallback(plan, &dummy, CUFFT_CB_LD_COMPLEX, nullptr);
    check("cufftXtSetCallback returns SUCCESS", r == CUFFT_SUCCESS);

    r = cufftXtClearCallback(plan, CUFFT_CB_LD_COMPLEX);
    check("cufftXtClearCallback returns SUCCESS", r == CUFFT_SUCCESS);

    cufftDestroy(plan);
}

int main() {
    test_xt_malloc_free();
    test_xt_memcpy_roundtrip();
    test_xt_exec_c2c_dc();
    test_xt_get_size_many();
    test_xt_callback();

    std::cout << "\n" << g_pass << "/" << g_total << " tests passed\n";
    return (g_pass == g_total) ? 0 : 1;
}
