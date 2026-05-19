// CUDA Driver API — memset variants (D8/D16/D32, 2D)

#include "cuda_driver_internal.h"
#include <cstring>

extern "C" {

CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  if (!dstDevice || N == 0) return CUDA_ERROR_INVALID_VALUE;
  memset(reinterpret_cast<void*>(dstDevice), static_cast<int>(uc), N);
  return CUDA_SUCCESS;
}

CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
  if (!dstDevice || N == 0) return CUDA_ERROR_INVALID_VALUE;
  auto *ptr = reinterpret_cast<unsigned short*>(dstDevice);
  for (size_t i = 0; i < N; ++i) ptr[i] = us;
  return CUDA_SUCCESS;
}

CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  if (!dstDevice || N == 0) return CUDA_ERROR_INVALID_VALUE;
  auto *ptr = reinterpret_cast<unsigned int*>(dstDevice);
  for (size_t i = 0; i < N; ++i) ptr[i] = ui;
  return CUDA_SUCCESS;
}

CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch,
                      unsigned char uc, size_t Width, size_t Height) {
  if (!dstDevice || Width == 0 || Height == 0) return CUDA_ERROR_INVALID_VALUE;
  auto *row = reinterpret_cast<unsigned char*>(dstDevice);
  for (size_t y = 0; y < Height; ++y) {
    memset(row + y * dstPitch, static_cast<int>(uc), Width);
  }
  return CUDA_SUCCESS;
}

CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch,
                       unsigned short us, size_t Width, size_t Height) {
  if (!dstDevice || Width == 0 || Height == 0) return CUDA_ERROR_INVALID_VALUE;
  for (size_t y = 0; y < Height; ++y) {
    auto *row = reinterpret_cast<unsigned short*>(
        static_cast<char*>(reinterpret_cast<void*>(dstDevice)) + y * dstPitch);
    for (size_t x = 0; x < Width; ++x) row[x] = us;
  }
  return CUDA_SUCCESS;
}

CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch,
                       unsigned int ui, size_t Width, size_t Height) {
  if (!dstDevice || Width == 0 || Height == 0) return CUDA_ERROR_INVALID_VALUE;
  for (size_t y = 0; y < Height; ++y) {
    auto *row = reinterpret_cast<unsigned int*>(
        static_cast<char*>(reinterpret_cast<void*>(dstDevice)) + y * dstPitch);
    for (size_t x = 0; x < Width; ++x) row[x] = ui;
  }
  return CUDA_SUCCESS;
}

} // extern "C"
