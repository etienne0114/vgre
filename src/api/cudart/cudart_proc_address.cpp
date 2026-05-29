// CUDA runtime and driver versioned symbol resolution.
//
// cudaGetProcAddress (CUDA 12.4+): resolves any CUDA runtime or driver symbol
// by name at the requested API version. Frameworks compiled with CUDA 12.4+
// call this instead of statically linking every function.
//
// cuGetProcAddress (driver-level equivalent): same but for driver API symbols.
//
// Implementation: delegate to the dynamic linker (dlsym on POSIX, GetProcAddress
// on Windows) so any symbol exported by the VGRE shared library is resolved
// automatically — no manual symbol table needed.

#include "vgre/api/cuda_interceptor.h"
#include "vgre/common/logger.h"
#include <cstring>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

// cudaDriverEntryPointQueryResult values
enum cudaDriverEntryPointQueryResult {
    cudaDriverEntryPointSuccess           = 0,
    cudaDriverEntryPointSymbolNotFound    = 1,
    cudaDriverEntryPointVersionNotSufficent = 2
};

using cudaError_t = vgre::api::cudaError_t;
static constexpr cudaError_t cudaSuccess            = vgre::api::cudaSuccess;
static constexpr cudaError_t cudaErrorSymbolNotFound = static_cast<cudaError_t>(500);

// Resolve a symbol from the currently-loaded VGRE library.
static void* vgre_resolve_symbol(const char* symbol) {
#if defined(_WIN32)
    HMODULE self = GetModuleHandleA(NULL);
    if (!self) self = GetModuleHandleA("vgre_cudart.dll");
    void* p = self ? reinterpret_cast<void*>(GetProcAddress(self, symbol)) : nullptr;
    if (!p) {
        // Also try the driver library name
        HMODULE drv = GetModuleHandleA("nvcuda.dll");
        if (!drv) drv = GetModuleHandleA("vgre_cudart.dll");
        if (drv) p = reinterpret_cast<void*>(GetProcAddress(drv, symbol));
    }
    return p;
#else
    // RTLD_DEFAULT searches all loaded shared objects in link order.
    void* p = dlsym(RTLD_DEFAULT, symbol);
    return p;
#endif
}

extern "C" {

// ── cudaGetProcAddress (CUDA 12.4+) ───────────────────────────────────────────
cudaError_t cudaGetProcAddress(const char* symbol,
                                void** pfn,
                                int    cudaVersion,
                                uint64_t flags,
                                cudaDriverEntryPointQueryResult* driverStatus) {
    (void)cudaVersion;
    (void)flags;
    if (!symbol || !pfn) return static_cast<cudaError_t>(1 /*cudaErrorInvalidValue*/);

    *pfn = vgre_resolve_symbol(symbol);

    if (!*pfn) {
        VGRE_LOG_WARN("cudaGetProcAddress",
            std::string("Symbol not found: ") + symbol);
        if (driverStatus) *driverStatus = cudaDriverEntryPointSymbolNotFound;
        return cudaErrorSymbolNotFound;
    }
    if (driverStatus) *driverStatus = cudaDriverEntryPointSuccess;
    return cudaSuccess;
}

// ── cuGetProcAddress (CUDA Driver API, 12.4+) ─────────────────────────────────
// Same mechanics; returns CUresult (int).
int cuGetProcAddress(const char* symbol,
                      void** pfn,
                      int    cudaVersion,
                      uint64_t flags) {
    (void)cudaVersion;
    (void)flags;
    if (!symbol || !pfn) return 1; // CUDA_ERROR_INVALID_VALUE

    *pfn = vgre_resolve_symbol(symbol);

    if (!*pfn) {
        VGRE_LOG_WARN("cuGetProcAddress",
            std::string("Symbol not found: ") + symbol);
        return 500; // CUDA_ERROR_NOT_FOUND (approx)
    }
    return 0; // CUDA_SUCCESS
}

// ── cuGetProcAddress_v2 (CUDA 12.4 extended form with status out-param) ────────
int cuGetProcAddress_v2(const char* symbol,
                         void** pfn,
                         int    cudaVersion,
                         uint64_t flags,
                         int* driverStatus) {
    (void)cudaVersion;
    (void)flags;
    if (!symbol || !pfn) return 1;

    *pfn = vgre_resolve_symbol(symbol);

    if (!*pfn) {
        if (driverStatus) *driverStatus = 1; // symbol not found
        return 500;
    }
    if (driverStatus) *driverStatus = 0; // success
    return 0;
}

} // extern "C"
