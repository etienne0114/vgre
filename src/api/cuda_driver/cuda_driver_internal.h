// CUDA Driver API internal shared types and helpers.
#pragma once

#include "vgre/api/cuda_interceptor.h"
#include "vgre/common/elf_reader.h"
#include "vgre/core/runtime_engine.h"
// VgreTMADescriptor is a C++ type (contains templates indirectly) — include
// outside any extern "C" block so templates are compiled with C++ linkage.
#include "vgre/compiler/wmma_emulation.h"
#include <cstring>

// CUtensorMap is layout-identical to VgreTMADescriptor (128 bytes).
// PTX emulation code casts TMA descriptor pointers directly to VgreTMADescriptor*.
using CUtensorMap = VgreTMADescriptor;

// TMA data-type codes (from CUDA Driver API)
enum CUtensorMapDataType {
    CU_TENSOR_MAP_DATA_TYPE_UINT8    = 0,
    CU_TENSOR_MAP_DATA_TYPE_UINT16   = 1,
    CU_TENSOR_MAP_DATA_TYPE_UINT32   = 2,
    CU_TENSOR_MAP_DATA_TYPE_INT32    = 3,
    CU_TENSOR_MAP_DATA_TYPE_UINT64   = 4,
    CU_TENSOR_MAP_DATA_TYPE_INT64    = 5,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16  = 6,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32  = 7,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT64  = 8,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ  = 10,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32     = 11,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ = 12,
};

// TMA interleave / swizzle / l2promotion / oobfill (opaque to CPU emulation)
enum CUtensorMapInterleave   { CU_TENSOR_MAP_INTERLEAVE_NONE = 0 };
enum CUtensorMapSwizzle      { CU_TENSOR_MAP_SWIZZLE_NONE = 0 };
enum CUtensorMapL2promotion  { CU_TENSOR_MAP_L2_PROMOTION_NONE = 0 };
enum CUtensorMapFloatOOBfill { CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0 };

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
using CUlinkState = void*;  // opaque handle to CUlinkStateImpl

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
static constexpr CUresult CUDA_ERROR_OUT_OF_MEMORY = 2;
static constexpr CUresult CUDA_ERROR_INVALID_DEVICE = 101;
static constexpr CUresult CUDA_ERROR_FILE_NOT_FOUND = 301;
static constexpr CUresult CUDA_ERROR_INVALID_PTX = 218;
static constexpr CUresult CUDA_ERROR_NO_BINARY_FOR_GPU = 209;
static constexpr CUresult CUDA_ERROR_INVALID_IMAGE = 200;
static constexpr CUresult CUDA_ERROR_NOT_SUPPORTED = 801;
static constexpr CUresult CUDA_ERROR_UNKNOWN = 999;

[[maybe_unused]] static CUresult toCU(vgre::api::cudaError_t err) {
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
  unsigned int numChannels = 1;
  // Per-dimension address modes (dim 0, 1, 2). Default CLAMP.
  vgre::core::TextureAddressMode addressMode[3] = {
    vgre::core::TextureAddressMode::CLAMP,
    vgre::core::TextureAddressMode::CLAMP,
    vgre::core::TextureAddressMode::CLAMP
  };
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
static constexpr int CU_AD_FORMAT_UNORM_INT8     = 0x04;
static constexpr int CU_AD_FORMAT_UNORM_INT16    = 0x05;
static constexpr int CU_AD_FORMAT_SNORM_INT8     = 0x0b;
static constexpr int CU_AD_FORMAT_SNORM_INT16    = 0x0c;
// Packed X1/X2/X4 variants
static constexpr int CU_AD_FORMAT_UNORM_INT8X1  = 0x0d;
static constexpr int CU_AD_FORMAT_UNORM_INT8X2  = 0x0e;
static constexpr int CU_AD_FORMAT_UNORM_INT8X4  = 0x0f;
static constexpr int CU_AD_FORMAT_UNORM_INT16X1 = 0x11;
static constexpr int CU_AD_FORMAT_UNORM_INT16X2 = 0x12;
static constexpr int CU_AD_FORMAT_UNORM_INT16X4 = 0x13;
static constexpr int CU_AD_FORMAT_SNORM_INT8X1  = 0x14;
static constexpr int CU_AD_FORMAT_SNORM_INT8X2  = 0x15;
static constexpr int CU_AD_FORMAT_SNORM_INT8X4  = 0x16;
static constexpr int CU_AD_FORMAT_SNORM_INT16X1 = 0x17;
static constexpr int CU_AD_FORMAT_SNORM_INT16X2 = 0x18;
static constexpr int CU_AD_FORMAT_SNORM_INT16X4 = 0x19;

// CUDA array descriptor (used by cuTexRefSetAddress2D)
struct CUDA_ARRAY_DESCRIPTOR {
    size_t Width;
    size_t Height;
    int Format;
    unsigned int NumChannels;
};

// ── External memory mipmapped array descriptor ──────────────────────────────
struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC {
    unsigned long long offset;
    CUDA_ARRAY_DESCRIPTOR arrayDesc;
    unsigned int numLevels;
    unsigned int flags;
    unsigned int reserved[16];
};

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
