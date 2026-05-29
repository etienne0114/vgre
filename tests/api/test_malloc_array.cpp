#include <iostream>
#include <cstring>

// Standard CUDART API — resolved via LD_PRELOAD when libvgre_cudart.so is loaded
extern "C" {
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorInvalidValue 1

typedef unsigned long long cudaArray_t;

struct cudaChannelFormatDesc {
  int x, y, z, w;
  int f;
};
struct cudaExtent {
  size_t width, height, depth;
};

cudaError_t cudaMallocArray(void **array, const struct cudaChannelFormatDesc *desc,
                              size_t width, size_t height);
cudaError_t cudaMalloc3DArray(void **array, const struct cudaChannelFormatDesc *desc,
                                struct cudaExtent extent);
cudaError_t cudaFreeArray(void *array);

// CUDA 11.6+ array memory requirement query
struct cudaArrayMemoryRequirements {
    size_t       size;
    size_t       alignment;
    unsigned int reserved[4];
};
cudaError_t cudaArrayGetMemoryRequirements(
    struct cudaArrayMemoryRequirements* memRequirements, void* array, int device);

// CUDA 11.1+ sparse-texture tile query
struct cudaArraySparseProperties {
    struct { unsigned int width; unsigned int height; unsigned int depth; } tileExtent;
    unsigned int       miptailFirstLevel;
    unsigned long long miptailSize;
    unsigned int       flags;
    unsigned int       reserved[4];
};
cudaError_t cudaArrayGetSparseProperties(
    struct cudaArraySparseProperties* sparseProperties, void* array);
cudaError_t cudaMipmappedArrayGetSparseProperties(
    struct cudaArraySparseProperties* sparseProperties, void* mipArray);
}

int main() {
  std::cout << "--- Testing cudaMallocArray / cudaMalloc3DArray / cudaFreeArray ---\n";

  // 1. Test cudaMallocArray (1D, float32)
  std::cout << "Testing cudaMallocArray (1D float32)...\n";
  cudaChannelFormatDesc desc = {32, 0, 0, 0, 0}; // 32-bit float
  void *arr1D = nullptr;
  cudaError_t err = cudaMallocArray(&arr1D, &desc, 16, 0);
  if (err != cudaSuccess || !arr1D) {
    std::cerr << "FAIL: cudaMallocArray 1D failed (err=" << err << ", arr=" << arr1D << ")\n";
    return 1;
  }
  std::cout << "  -> 1D array created: " << arr1D << "\n";

  // 2. Test cudaMallocArray (2D, uint8)
  std::cout << "Testing cudaMallocArray (2D uint8)...\n";
  cudaChannelFormatDesc desc8 = {8, 0, 0, 0, 1}; // 8-bit unsigned
  void *arr2D = nullptr;
  err = cudaMallocArray(&arr2D, &desc8, 8, 8);
  if (err != cudaSuccess || !arr2D) {
    std::cerr << "FAIL: cudaMallocArray 2D failed (err=" << err << ", arr=" << arr2D << ")\n";
    return 1;
  }
  std::cout << "  -> 2D array created: " << arr2D << "\n";

  // 3. Test cudaMalloc3DArray
  std::cout << "Testing cudaMalloc3DArray...\n";
  cudaChannelFormatDesc desc16 = {16, 0, 0, 0, 2}; // 16-bit unsigned
  void *arr3D = nullptr;
  cudaExtent ext = {4, 4, 4};
  err = cudaMalloc3DArray(&arr3D, &desc16, ext);
  if (err != cudaSuccess || !arr3D) {
    std::cerr << "FAIL: cudaMalloc3DArray failed (err=" << err << ", arr=" << arr3D << ")\n";
    return 1;
  }
  std::cout << "  -> 3D array created: " << arr3D << "\n";

  // 4. Free arrays
  std::cout << "Testing cudaFreeArray...\n";
  err = cudaFreeArray(arr1D);
  if (err != cudaSuccess) {
    std::cerr << "FAIL: cudaFreeArray(arr1D) failed (err=" << err << ")\n";
    return 1;
  }
  err = cudaFreeArray(arr2D);
  if (err != cudaSuccess) {
    std::cerr << "FAIL: cudaFreeArray(arr2D) failed (err=" << err << ")\n";
    return 1;
  }
  err = cudaFreeArray(arr3D);
  if (err != cudaSuccess) {
    std::cerr << "FAIL: cudaFreeArray(arr3D) failed (err=" << err << ")\n";
    return 1;
  }
  std::cout << "  -> All arrays freed successfully\n";

  // 5. Test null parameter rejection
  std::cout << "Testing null parameter rejection...\n";
  err = cudaMallocArray(nullptr, &desc, 4, 4);
  if (err != cudaErrorInvalidValue) {
    std::cerr << "FAIL: Expected cudaErrorInvalidValue for null array ptr\n";
    return 1;
  }
  err = cudaMallocArray(&arr1D, nullptr, 4, 4);
  if (err != cudaErrorInvalidValue) {
    std::cerr << "FAIL: Expected cudaErrorInvalidValue for null desc\n";
    return 1;
  }
  err = cudaMallocArray(&arr1D, &desc, 0, 4);
  if (err != cudaErrorInvalidValue) {
    std::cerr << "FAIL: Expected cudaErrorInvalidValue for width=0\n";
    return 1;
  }
  err = cudaMalloc3DArray(&arr3D, &desc16, {0, 4, 4});
  if (err != cudaErrorInvalidValue) {
    std::cerr << "FAIL: Expected cudaErrorInvalidValue for extent.width=0\n";
    return 1;
  }
  std::cout << "  -> Null parameters correctly rejected\n";

  // 6. Test zero-element-size descriptor rejection
  std::cout << "Testing zero-bit descriptor rejection...\n";
  cudaChannelFormatDesc descZero = {0, 0, 0, 0, 0};
  err = cudaMallocArray(&arr1D, &descZero, 4, 4);
  if (err != cudaErrorInvalidValue) {
    std::cerr << "FAIL: Expected cudaErrorInvalidValue for zero-bit desc\n";
    return 1;
  }
  std::cout << "  -> Zero-bit descriptor correctly rejected\n";

  std::cout << "[PASS] cudaMallocArray / cudaMalloc3DArray / cudaFreeArray work correctly!\n";

  // ── cudaArrayGetMemoryRequirements ──────────────────────────────────────────
  std::cout << "Testing cudaArrayGetMemoryRequirements...\n";
  void *arr_mr = nullptr;
  err = cudaMallocArray(&arr_mr, &desc, 64, 64);
  if (err != cudaSuccess || !arr_mr) {
    std::cerr << "FAIL: cudaMallocArray for memreq test failed (err=" << err << ")\n";
    return 1;
  }
  cudaArrayMemoryRequirements memReq{};
  err = cudaArrayGetMemoryRequirements(&memReq, arr_mr, 0);
  if (err != cudaSuccess) {
    std::cerr << "FAIL: cudaArrayGetMemoryRequirements failed (err=" << err << ")\n";
    return 1;
  }
  if (memReq.size == 0) {
    std::cerr << "FAIL: cudaArrayGetMemoryRequirements returned size=0\n";
    return 1;
  }
  if (memReq.alignment == 0) {
    std::cerr << "FAIL: cudaArrayGetMemoryRequirements returned alignment=0\n";
    return 1;
  }
  std::cout << "  -> size=" << memReq.size << " alignment=" << memReq.alignment << "\n";

  // Null output pointer must be rejected
  err = cudaArrayGetMemoryRequirements(nullptr, arr_mr, 0);
  if (err != cudaErrorInvalidValue) {
    std::cerr << "FAIL: Expected cudaErrorInvalidValue for null memRequirements\n";
    return 1;
  }
  cudaFreeArray(arr_mr);
  std::cout << "[PASS] cudaArrayGetMemoryRequirements works correctly!\n";

  // ── cudaArrayGetSparseProperties ────────────────────────────────────────────
  std::cout << "Testing cudaArrayGetSparseProperties...\n";
  void *arr_sp = nullptr;
  err = cudaMallocArray(&arr_sp, &desc, 16, 16);
  if (err != cudaSuccess || !arr_sp) {
    std::cerr << "FAIL: cudaMallocArray for sparse test failed (err=" << err << ")\n";
    return 1;
  }
  cudaArraySparseProperties sp{};
  err = cudaArrayGetSparseProperties(&sp, arr_sp);
  if (err != cudaSuccess) {
    std::cerr << "FAIL: cudaArrayGetSparseProperties failed (err=" << err << ")\n";
    return 1;
  }
  if (sp.tileExtent.width == 0 || sp.miptailSize == 0) {
    std::cerr << "FAIL: cudaArrayGetSparseProperties returned zero tile width or miptailSize\n";
    return 1;
  }
  std::cout << "  -> tile=" << sp.tileExtent.width << "x" << sp.tileExtent.height
            << " miptailSize=" << sp.miptailSize << "\n";
  err = cudaArrayGetSparseProperties(nullptr, arr_sp);
  if (err != cudaErrorInvalidValue) {
    std::cerr << "FAIL: Expected cudaErrorInvalidValue for null sparseProperties\n";
    return 1;
  }
  cudaFreeArray(arr_sp);
  std::cout << "[PASS] cudaArrayGetSparseProperties works correctly!\n";

  std::cout << "[PASS] All cudaArray tests passed!\n";
  return 0;
}
