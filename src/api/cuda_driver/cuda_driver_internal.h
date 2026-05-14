// CUDA Driver API internal shared types and helpers.
#pragma once

#include "vgre/api/cuda_interceptor.h"
#include "vgre/common/elf_reader.h"
#include "vgre/core/runtime_engine.h"
#include <cstring>

extern "C" {

// Basic CUDA Driver API types
using CUdevice = int;
using CUcontext = void*;
static thread_local CUcontext g_current_ctx = nullptr;
using CUstream = vgre::api::cudaStream_t;
using CUevent = vgre::core::Event*;
using CUmodule = vgre::api::CUmodule;
using CUfunction = vgre::api::CUfunction;
using CUdeviceptr = void*;
using CUresult = int;
using CUarray = void*;

// CUDA memory types
enum CUmemorytype {
  CU_MEMORYTYPE_HOST = 1,
  CU_MEMORYTYPE_DEVICE = 2,
  CU_MEMORYTYPE_ARRAY = 3,
  CU_MEMORYTYPE_UNIFIED = 4
};

// CUresult values (minimal)
static constexpr CUresult CUDA_SUCCESS = 0;
static constexpr CUresult CUDA_ERROR_INVALID_VALUE = 1;
static constexpr CUresult CUDA_ERROR_NOT_INITIALIZED = 3;
static constexpr CUresult CUDA_ERROR_INVALID_DEVICE = 101;
static constexpr CUresult CUDA_ERROR_NOT_SUPPORTED = 801;
static constexpr CUresult CUDA_ERROR_UNKNOWN = 999;

static CUresult toCU(vgre::api::cudaError_t err) {
  switch (err) {
  case vgre::api::cudaSuccess:
    return CUDA_SUCCESS;
  case vgre::api::cudaErrorInvalidValue:
    return CUDA_ERROR_INVALID_VALUE;
  case vgre::api::cudaErrorInvalidDevice:
    return CUDA_ERROR_INVALID_DEVICE;
  case vgre::api::cudaErrorNotInitialized:
    return CUDA_ERROR_NOT_INITIALIZED;
  default:
    return CUDA_ERROR_UNKNOWN;
  }
}

// ── Extern module helpers (shared across module and texture files) ───────────

extern "C" void *vgre_lookup_texture_ref(void *handle, const char *name);

// ── Texture/Surface types (shared across module and texture files) ───────────

#include "vgre/core/texture_manager.h"

struct CUtexref_st {
  vgre::core::TextureDescriptor desc;
  void *devPtr = nullptr;
  size_t size = 0;
  vgre::core::TextureId texId = 0;
};

using CUtexref = CUtexref_st*;
using CUsurfref = void*;

// CUDA array formats
static constexpr int CU_AD_FORMAT_UNSIGNED_INT8  = 0x01;
static constexpr int CU_AD_FORMAT_UNSIGNED_INT16 = 0x02;
static constexpr int CU_AD_FORMAT_UNSIGNED_INT32 = 0x03;
static constexpr int CU_AD_FORMAT_SIGNED_INT8    = 0x08;
static constexpr int CU_AD_FORMAT_SIGNED_INT16   = 0x09;
static constexpr int CU_AD_FORMAT_SIGNED_INT32   = 0x0a;
static constexpr int CU_AD_FORMAT_FLOAT          = 0x20;
static constexpr int CU_AD_FORMAT_HALF           = 0x10;

// ── Graph node parameter structs (driver API) ────────────────────────────────

struct CUDA_KERNEL_NODE_PARAMS {
    CUfunction func;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    void **kernelParams;
    void **extra;
};

struct CUDA_MEMSET_NODE_PARAMS {
    CUdeviceptr dst;
    size_t pitch;
    unsigned int value;
    unsigned int elementSize;
    size_t width;
    size_t height;
};

} // extern "C"
