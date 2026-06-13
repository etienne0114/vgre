// HIP/ROCm compatibility — a real HIP program runs on VGRE's CUDA runtime.

#include "vgre/hip/hip_runtime.h"

#include <cstdio>
#include <cstring>

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    std::printf("=== HIP/ROCm compatibility layer ===\n");

    int dc = 0;
    check("hipGetDeviceCount returns >=1 device", hipGetDeviceCount(&dc) == hipSuccess && dc >= 1);

    const size_t N = 1024;
    void* dptr = nullptr;
    check("hipMalloc succeeds", hipMalloc(&dptr, N) == hipSuccess && dptr != nullptr);

    check("hipMemset device buffer", hipMemset(dptr, 0, N) == hipSuccess);

    unsigned char host[N], back[N];
    for (size_t i = 0; i < N; ++i) host[i] = (unsigned char)(i * 31 + 7);
    std::memset(back, 0, N);

    check("hipMemcpy HostToDevice", hipMemcpy(dptr, host, N, hipMemcpyHostToDevice) == hipSuccess);
    check("hipMemcpy DeviceToHost", hipMemcpy(back, dptr, N, hipMemcpyDeviceToHost) == hipSuccess);
    check("round-trip data matches through HIP API", std::memcmp(host, back, N) == 0);

    check("hipDeviceSynchronize", hipDeviceSynchronize() == hipSuccess);
    check("hipFree", hipFree(dptr) == hipSuccess);

    check("hipGetErrorString(hipSuccess) non-null", hipGetErrorString(hipSuccess) != nullptr);
    check("hipGetLastError clears to success", hipGetLastError() == hipSuccess);

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
