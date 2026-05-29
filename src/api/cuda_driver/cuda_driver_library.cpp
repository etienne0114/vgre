// CUDA Driver Library API (CUDA 12.0+).
//
// cuLibraryLoadData / cuLibraryLoadFromFile / cuLibraryGetKernel /
// cuKernelGetFunction / cuLibraryGetGlobal / cuLibraryUnload
//
// CUDA 12.0 changed how the runtime loads device code: instead of
// cuModuleLoad + cuModuleGetFunction it uses the cuLibrary* API.
// Any framework compiled with the CUDA 12.0+ toolkit will call these
// entry points.  VGRE maps each CUlibrary to a CUmodule internally, so
// existing JIT/PTX infrastructure is fully reused.

#include "cuda_driver_internal.h"
#include <mutex>
#include <string>
#include <unordered_map>

// Forward declarations from cuda_driver_module.cpp (same library).
extern "C" CUresult cuModuleLoad(CUmodule* module, const char* fname);
extern "C" CUresult cuModuleLoadData(CUmodule* module, const void* image);
extern "C" CUresult cuModuleUnload(CUmodule module);
extern "C" CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name);
extern "C" CUresult cuModuleGetGlobal(CUdeviceptr* dptr, size_t* bytes,
                                       CUmodule hmod, const char* name);

// ── Internal type definitions ─────────────────────────────────────────────────

struct CUkernel_st {
    CUfunction func;
    std::string name;
};

struct CUlibrary_st {
    CUmodule module = nullptr;
    std::mutex mu;
    std::unordered_map<std::string, CUkernel_st*> kernels;

    ~CUlibrary_st() {
        for (auto& [name, k] : kernels) delete k;
    }
};

using CUlibrary       = CUlibrary_st*;
using CUkernel        = CUkernel_st*;
using CUlibraryOption = int;

// JIT option types (opaque; we accept but ignore them — VGRE JIT is triggered
// by the module-load path regardless of jit option arrays).
using CUjit_option = int;

extern "C" {

// ── cuLibraryLoadData ─────────────────────────────────────────────────────────
// Load device code from an in-memory image (PTX text or cubin blob).
CUresult cuLibraryLoadData(CUlibrary* library,
                            const void* code,
                            CUjit_option* /*jitOptions*/,
                            void** /*jitOptionsValues*/,
                            unsigned int /*numJitOptions*/,
                            CUlibraryOption* /*libraryOptions*/,
                            void** /*libraryOptionValues*/,
                            unsigned int /*numLibraryOptions*/) {
    if (!library || !code) return CUDA_ERROR_INVALID_VALUE;
    auto* lib = new CUlibrary_st();
    CUresult res = cuModuleLoadData(&lib->module, code);
    if (res != CUDA_SUCCESS) { delete lib; return res; }
    *library = lib;
    return CUDA_SUCCESS;
}

// ── cuLibraryLoadFromFile ─────────────────────────────────────────────────────
CUresult cuLibraryLoadFromFile(CUlibrary* library,
                                const char* fileName,
                                CUjit_option* /*jitOptions*/,
                                void** /*jitOptionsValues*/,
                                unsigned int /*numJitOptions*/,
                                CUlibraryOption* /*libraryOptions*/,
                                void** /*libraryOptionValues*/,
                                unsigned int /*numLibraryOptions*/) {
    if (!library || !fileName) return CUDA_ERROR_INVALID_VALUE;
    auto* lib = new CUlibrary_st();
    CUresult res = cuModuleLoad(&lib->module, fileName);
    if (res != CUDA_SUCCESS) { delete lib; return res; }
    *library = lib;
    return CUDA_SUCCESS;
}

// ── cuLibraryGetKernel ────────────────────────────────────────────────────────
// Obtain (or cache) a kernel handle for a named function in the library.
CUresult cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name) {
    if (!pKernel || !library || !name) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(library->mu);
    auto it = library->kernels.find(name);
    if (it != library->kernels.end()) { *pKernel = it->second; return CUDA_SUCCESS; }
    auto* k = new CUkernel_st();
    k->name = name;
    CUresult res = cuModuleGetFunction(&k->func, library->module, name);
    if (res != CUDA_SUCCESS) { delete k; return res; }
    library->kernels[name] = k;
    *pKernel = k;
    return CUDA_SUCCESS;
}

// ── cuKernelGetFunction ───────────────────────────────────────────────────────
// Unwrap the CUkernel handle to a CUfunction usable with cuLaunchKernel.
CUresult cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel) {
    if (!pFunc || !kernel) return CUDA_ERROR_INVALID_VALUE;
    *pFunc = kernel->func;
    return CUDA_SUCCESS;
}

// ── cuLibraryGetGlobal ────────────────────────────────────────────────────────
CUresult cuLibraryGetGlobal(CUdeviceptr* dptr, size_t* bytes,
                             CUlibrary library, const char* name) {
    if (!library || !name) return CUDA_ERROR_INVALID_VALUE;
    return cuModuleGetGlobal(dptr, bytes, library->module, name);
}

// ── cuLibraryGetManaged ───────────────────────────────────────────────────────
// Managed variables live in the same address space in VGRE; delegate to global.
CUresult cuLibraryGetManaged(CUdeviceptr* dptr, size_t* bytes,
                              CUlibrary library, const char* name) {
    return cuLibraryGetGlobal(dptr, bytes, library, name);
}

// ── cuLibraryGetUnifiedFunction ───────────────────────────────────────────────
// CUDA 12.1 variant for unified virtual addressing; same as GetKernel in VGRE.
CUresult cuLibraryGetUnifiedFunction(void** fptr, CUlibrary library, const char* symbol) {
    if (!fptr || !library || !symbol) return CUDA_ERROR_INVALID_VALUE;
    CUkernel k = nullptr;
    CUresult res = cuLibraryGetKernel(&k, library, symbol);
    if (res != CUDA_SUCCESS) return res;
    *fptr = reinterpret_cast<void*>(k);
    return CUDA_SUCCESS;
}

// ── cuLibraryUnload ───────────────────────────────────────────────────────────
CUresult cuLibraryUnload(CUlibrary library) {
    if (!library) return CUDA_ERROR_INVALID_VALUE;
    cuModuleUnload(library->module);
    delete library;
    return CUDA_SUCCESS;
}

// ── cuKernelGetAttribute ──────────────────────────────────────────────────────
// Returns kernel attributes (mirrors cuFuncGetAttribute).
extern "C" CUresult cuFuncGetAttribute(int* pi, int attrib, CUfunction hfunc);

CUresult cuKernelGetAttribute(int* pi, int attrib, CUkernel kernel, CUdevice /*dev*/) {
    if (!pi || !kernel) return CUDA_ERROR_INVALID_VALUE;
    return cuFuncGetAttribute(pi, attrib, kernel->func);
}

// ── cuKernelSetAttribute ──────────────────────────────────────────────────────
extern "C" CUresult cuFuncSetAttribute(CUfunction hfunc, int attrib, int val);

CUresult cuKernelSetAttribute(int attrib, int val, CUkernel kernel, CUdevice /*dev*/) {
    if (!kernel) return CUDA_ERROR_INVALID_VALUE;
    return cuFuncSetAttribute(kernel->func, attrib, val);
}

} // extern "C"
