// CUDA Driver API — Tensor Memory Accelerator (TMA) descriptor encoding.
//
// cuTensorMapEncodeTiled  — build a tiled-access descriptor from global tensor layout.
// cuTensorMapEncodeIm2col — build an im2col-access descriptor for convolution patches.
//
// Both functions populate a VgreTMADescriptor (= CUtensorMap) which the PTX
// emulation layer reads directly via vgre_tma_load_*_b / vgre_tma_load_im2col_2d.

#include "cuda_driver_internal.h"
#include <cstring>

extern "C" {

// ── Byte size per element by data type ────────────────────────────────────────
static uint32_t tmaElemBytes(CUtensorMapDataType dt)
{
    switch (dt) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8:   return 1;
    case CU_TENSOR_MAP_DATA_TYPE_UINT16:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: return 2;
    case CU_TENSOR_MAP_DATA_TYPE_UINT32:
    case CU_TENSOR_MAP_DATA_TYPE_INT32:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ: return 4;
    case CU_TENSOR_MAP_DATA_TYPE_UINT64:
    case CU_TENSOR_MAP_DATA_TYPE_INT64:
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:  return 8;
    default: return 4;
    }
}

// ── cuTensorMapEncodeTiled ────────────────────────────────────────────────────
//
// Encodes a tiled-access TMA descriptor.
//
// globalDim[rank]    — size of each tensor dimension (dim[0] is innermost)
// globalStrides[rank-1] — byte strides from dim[1] outward; stride[0] is
//                         the byte stride for the second dimension.
//                         (NVIDIA convention: strides for dim 1..rank-1)
// boxDim[rank]       — tile box dimensions (innermost first)
// elementStrides[rank] — per-element strides in elements (usually all 1)
//
// The descriptor matches VgreTMADescriptor layout so PTX code can cast directly.

CUresult cuTensorMapEncodeTiled(
    CUtensorMap*            tensorMap,
    CUtensorMapDataType     tensorDataType,
    unsigned int            tensorRank,
    void*                   globalAddress,
    const unsigned long long* globalDim,
    const unsigned long long* globalStrides,
    const unsigned int*     boxDim,
    const unsigned int*     elementStrides,
    CUtensorMapInterleave   /*interleave*/,
    CUtensorMapSwizzle      /*swizzle*/,
    CUtensorMapL2promotion  /*l2Promotion*/,
    CUtensorMapFloatOOBfill /*oobFill*/)
{
    if (!tensorMap || !globalAddress || !globalDim || !globalStrides || !boxDim)
        return CUDA_ERROR_INVALID_VALUE;
    if (tensorRank < 1 || tensorRank > 5)
        return CUDA_ERROR_INVALID_VALUE;

    memset(tensorMap, 0, sizeof(VgreTMADescriptor));

    (void)elementStrides; // per-dim element strides not modelled in CPU emulation
    tensorMap->baseAddr  = globalAddress;
    tensorMap->elemBytes = tmaElemBytes(tensorDataType);
    tensorMap->rank      = tensorRank;
    tensorMap->tag       = 0; // tiled

    // Tensor dimensions (innermost = index 0)
    for (unsigned i = 0; i < tensorRank; ++i)
        tensorMap->dim[i] = static_cast<uint32_t>(globalDim[i]);

    // Byte strides for dimensions 1..rank-1 (globalStrides[i] is stride of dim[i+1])
    // stride[0] = byte stride along dim-1 (row stride for 2D)
    for (unsigned i = 0; i < tensorRank - 1; ++i)
        tensorMap->stride[i] = static_cast<uint32_t>(globalStrides[i]);

    // Box (tile) dimensions
    for (unsigned i = 0; i < tensorRank; ++i)
        tensorMap->boxDim[i] = boxDim[i];

    return CUDA_SUCCESS;
}

// ── cuTensorMapEncodeIm2col ───────────────────────────────────────────────────
//
// Encodes an im2col-access TMA descriptor for convolution patch gathering.
//
// pixelBoxLowerCorner[rank-2] — lower corner of the filter window per spatial dim
// pixelBoxUpperCorner[rank-2] — upper corner of the filter window per spatial dim
// channelsPerPixel            — channels in the innermost tensor dimension
// pixelsPerColumn             — number of output pixels stacked per column
//
// For CPU emulation we store the Im2col parameters in the extended descriptor
// fields; vgre_tma_load_im2col_2d reads them when tag == 1.

CUresult cuTensorMapEncodeIm2col(
    CUtensorMap*            tensorMap,
    CUtensorMapDataType     tensorDataType,
    unsigned int            tensorRank,
    void*                   globalAddress,
    const unsigned long long* globalDim,
    const unsigned long long* globalStrides,
    const int*              pixelBoxLowerCorner,
    const int*              pixelBoxUpperCorner,
    unsigned int            channelsPerPixel,
    unsigned int            pixelsPerColumn,
    const unsigned int*     elementStrides,
    CUtensorMapInterleave   /*interleave*/,
    CUtensorMapSwizzle      /*swizzle*/,
    CUtensorMapL2promotion  /*l2Promotion*/,
    CUtensorMapFloatOOBfill /*oobFill*/)
{
    if (!tensorMap || !globalAddress || !globalDim || !globalStrides ||
        !pixelBoxLowerCorner || !pixelBoxUpperCorner)
        return CUDA_ERROR_INVALID_VALUE;
    if (tensorRank < 3 || tensorRank > 5)
        return CUDA_ERROR_INVALID_VALUE;

    (void)elementStrides; // per-dim element strides not modelled in CPU emulation
    memset(tensorMap, 0, sizeof(VgreTMADescriptor));

    tensorMap->baseAddr  = globalAddress;
    tensorMap->elemBytes = tmaElemBytes(tensorDataType);
    tensorMap->rank      = tensorRank;
    tensorMap->tag       = 1; // im2col

    // Tensor dimensions
    for (unsigned i = 0; i < tensorRank; ++i)
        tensorMap->dim[i] = static_cast<uint32_t>(globalDim[i]);

    // Byte strides for dimensions 1..rank-1
    for (unsigned i = 0; i < tensorRank - 1; ++i)
        tensorMap->stride[i] = static_cast<uint32_t>(globalStrides[i]);

    // boxDim[0] = channelsPerPixel (innermost, = tile width in channel dim)
    tensorMap->boxDim[0] = channelsPerPixel;
    tensorMap->boxDim[1] = pixelsPerColumn;

    // Im2col pixel box corners (spatial dims: rank-2 values)
    unsigned nSpatial = tensorRank - 2; // e.g. 2 for NCHW → HW filter window
    for (unsigned i = 0; i < nSpatial && i < 5; ++i) {
        tensorMap->im2colLower[i] = pixelBoxLowerCorner[i];
        tensorMap->im2colUpper[i] = pixelBoxUpperCorner[i];
    }

    tensorMap->channelsPerPixel = channelsPerPixel;
    tensorMap->pixelsPerColumn  = pixelsPerColumn;

    return CUDA_SUCCESS;
}

// ── cuTensorMapReplaceAddress ─────────────────────────────────────────────────
// Allows updating the base address of a descriptor in-place (e.g. for
// double-buffering) without re-encoding the full descriptor.
CUresult cuTensorMapReplaceAddress(CUtensorMap* tensorMap, void* globalAddress)
{
    if (!tensorMap || !globalAddress)
        return CUDA_ERROR_INVALID_VALUE;
    tensorMap->baseAddr = globalAddress;
    return CUDA_SUCCESS;
}

} // extern "C"
