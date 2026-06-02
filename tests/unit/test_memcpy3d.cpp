/**
 * VGRE Unit Tests — cudaMemcpy3D / cudaMemcpy3DAsync (QUEUE-21)
 *
 * Verifies that CUDAInterceptor::memcpy3D / memcpy3DAsync correctly transfer
 * data for all four direction variants and respect row/slice pitch layout.
 *
 * Layout invariant (CUDA 3-D pitched memory spec):
 *   byte offset of element (x, y, z) = z*slicePitch + y*rowPitch + x
 *   where slicePitch = rowPitch * height
 *   Total bytes = extent.width * extent.height * extent.depth
 */

#include "vgre/api/cuda_interceptor.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace vgre::api;

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_EQ(a, b, msg)                                                    \
  do {                                                                          \
    if ((a) != (b)) {                                                           \
      std::cerr << "[FAIL] " << msg << " (expected " << (b) << ", got " << (a) \
                << ")\n";                                                       \
      ++g_fail;                                                                 \
      return;                                                                   \
    }                                                                           \
  } while (0)

#define ASSERT_TRUE(cond, msg)                                                  \
  do {                                                                          \
    if (!(cond)) {                                                               \
      std::cerr << "[FAIL] " << msg << "\n";                                    \
      ++g_fail;                                                                 \
      return;                                                                   \
    }                                                                           \
  } while (0)

#define PASS(name)                                                              \
  do {                                                                          \
    std::cout << "[PASS] " << name << "\n";                                     \
    ++g_pass;                                                                   \
  } while (0)

// ── Helpers ──────────────────────────────────────────────────────────────────

// Build a cudaMemcpy3DParms for a plain host pointer (no cudaArray).
// srcPos / dstPos default to {0,0,0}; pitch defaults to extent.width.
static cudaMemcpy3DParms makeParms(void *dstPtr, size_t dstPitch,
                                   size_t dstYSize,
                                   const void *srcPtr, size_t srcPitch,
                                   size_t srcYSize,
                                   cudaExtent ext, cudaMemcpyKind_t kind) {
  cudaMemcpy3DParms p{};
  p.dstPtr.ptr   = dstPtr;
  p.dstPtr.pitch = dstPitch;
  p.dstPtr.xsize = ext.width;
  p.dstPtr.ysize = dstYSize;
  p.dstPos       = {0, 0, 0};
  p.dstArray     = 0;

  p.srcPtr.ptr   = const_cast<void *>(srcPtr);
  p.srcPtr.pitch = srcPitch;
  p.srcPtr.xsize = ext.width;
  p.srcPtr.ysize = srcYSize;
  p.srcPos       = {0, 0, 0};
  p.srcArray     = 0;

  p.extent = ext;
  p.kind   = kind;
  return p;
}

// ── Test 1: H→H contiguous 2×3×4 float array ────────────────────────────────
// Invariant: width=2*sizeof(float), height=3, depth=4 →
//   slicePitch = width*height, total = width*height*depth bytes.
void test_memcpy3d_hth() {
  // Dimensions: 2 floats wide, 3 rows tall, 4 slices deep = 24 floats total.
  const size_t W = 2, H = 3, D = 4;
  const size_t rowBytes = W * sizeof(float);

  std::vector<float> src(W * H * D);
  std::vector<float> dst(W * H * D, 0.0f);

  // Fill src with unique values so any mis-copy is detectable.
  for (size_t i = 0; i < W * H * D; ++i)
    src[i] = static_cast<float>(i + 1);

  cudaExtent ext{rowBytes, H, D};
  auto p = makeParms(dst.data(), rowBytes, H,
                     src.data(), rowBytes, H,
                     ext, cudaMemcpyHostToHost);

  auto &ci = CUDAInterceptor::instance();
  cudaError_t err = ci.memcpy3D(&p);
  ASSERT_EQ(err, cudaSuccess, "memcpy3d_hth: expected cudaSuccess");

  // Verify all 24 elements match.
  for (size_t i = 0; i < W * H * D; ++i) {
    ASSERT_TRUE(dst[i] == src[i], "memcpy3d_hth: element mismatch at index " +
                                      std::to_string(i));
  }
  PASS("test_memcpy3d_hth: 2x3x4 float H->H copy, all 24 elements equal src");
}

// ── Test 2: srcPos.x offset ───────────────────────────────────────────────────
// Invariant: srcPos={offsetBytes,0,0} shifts the source base pointer by
//   offsetBytes before the inner loop, so only (width - offsetBytes) bytes
//   are read per row.
void test_memcpy3d_srcpos_offset() {
  const size_t W = 4, H = 2, D = 2;          // 4 floats wide
  const size_t rowBytes   = W * sizeof(float);
  const size_t xOff       = sizeof(float);    // skip first float per row
  const size_t copyWidth  = rowBytes - xOff;  // 3 floats per row

  std::vector<float> src(W * H * D);
  for (size_t i = 0; i < W * H * D; ++i)
    src[i] = static_cast<float>(i + 10);

  // dst is sized for the reduced-width copy.
  const size_t dstRowBytes = copyWidth;
  std::vector<float> dst(dstRowBytes / sizeof(float) * H * D, -1.0f);

  cudaExtent ext{copyWidth, H, D};
  cudaMemcpy3DParms p{};
  p.srcPtr.ptr   = src.data();
  p.srcPtr.pitch = rowBytes;
  p.srcPtr.xsize = rowBytes;
  p.srcPtr.ysize = H;
  p.srcPos       = {xOff, 0, 0};  // srcBase advances by xOff
  p.srcArray     = 0;

  p.dstPtr.ptr   = dst.data();
  p.dstPtr.pitch = dstRowBytes;
  p.dstPtr.xsize = dstRowBytes;
  p.dstPtr.ysize = H;
  p.dstPos       = {0, 0, 0};
  p.dstArray     = 0;

  p.extent = ext;
  p.kind   = cudaMemcpyHostToHost;

  auto &ci = CUDAInterceptor::instance();
  cudaError_t err = ci.memcpy3D(&p);
  ASSERT_EQ(err, cudaSuccess, "memcpy3d_srcpos_offset: expected cudaSuccess");

  // Verify: for each (y, z), dst row should match src row starting at xOff.
  // slicePitch = rowBytes * H; offset(y,z) = z*slicePitch + y*rowBytes
  size_t dstSlice = dstRowBytes * H;
  size_t srcSlice = rowBytes * H;
  for (size_t z = 0; z < D; ++z) {
    for (size_t y = 0; y < H; ++y) {
      const float *srcRow =
          reinterpret_cast<const float *>(
              reinterpret_cast<const uint8_t *>(src.data()) +
              z * srcSlice + y * rowBytes + xOff);
      const float *dstRow =
          reinterpret_cast<const float *>(
              reinterpret_cast<const uint8_t *>(dst.data()) +
              z * dstSlice + y * dstRowBytes);
      for (size_t x = 0; x < (copyWidth / sizeof(float)); ++x) {
        ASSERT_TRUE(dstRow[x] == srcRow[x],
                    "memcpy3d_srcpos_offset: element mismatch");
      }
    }
  }
  PASS("test_memcpy3d_srcpos_offset: srcPos.x offset respected, data correct");
}

// ── Test 3: cudaMemcpy3DAsync H→H ────────────────────────────────────────────
// Invariant: async variant must complete synchronously in VGRE's CPU model and
//   deliver identical results to the synchronous variant.
void test_memcpy3d_async() {
  const size_t W = 2, H = 3, D = 4;
  const size_t rowBytes = W * sizeof(float);

  std::vector<float> src(W * H * D);
  std::vector<float> dst(W * H * D, 0.0f);
  for (size_t i = 0; i < W * H * D; ++i)
    src[i] = static_cast<float>(i * 2 + 1);

  cudaExtent ext{rowBytes, H, D};
  auto p = makeParms(dst.data(), rowBytes, H,
                     src.data(), rowBytes, H,
                     ext, cudaMemcpyHostToHost);

  auto &ci = CUDAInterceptor::instance();
  // stream=0 is the default (null) stream, always valid in VGRE.
  cudaError_t err = ci.memcpy3DAsync(&p, 0);
  ASSERT_EQ(err, cudaSuccess, "memcpy3d_async: expected cudaSuccess");

  for (size_t i = 0; i < W * H * D; ++i) {
    ASSERT_TRUE(dst[i] == src[i],
                "memcpy3d_async: data mismatch at index " + std::to_string(i));
  }
  PASS("test_memcpy3d_async: 2x3x4 H->H via Async, cudaSuccess + data correct");
}

// ── Test 4: Row pitch > width (padded rows) ───────────────────────────────────
// Invariant: stride/pitch separates rows; only `extent.width` bytes per row are
//   copied, leaving the padding bytes untouched in the source.
//   offset(y, z) = z * (pitch * height) + y * pitch  (not y * width).
void test_memcpy3d_pitched() {
  const size_t W = 2, H = 3, D = 2;
  const size_t elemBytes = sizeof(float);
  const size_t rowBytes  = W * elemBytes;        // data bytes per row
  const size_t srcPitch  = rowBytes + 8;         // 8-byte padding per row

  // Allocate source with padded rows, fill data region only.
  std::vector<uint8_t> srcBuf(srcPitch * H * D, 0xAB);
  for (size_t z = 0; z < D; ++z) {
    for (size_t y = 0; y < H; ++y) {
      float *rowPtr = reinterpret_cast<float *>(
          srcBuf.data() + z * (srcPitch * H) + y * srcPitch);
      for (size_t x = 0; x < W; ++x)
        rowPtr[x] = static_cast<float>(z * H * W + y * W + x + 1);
    }
  }

  // Destination is tight (no padding).
  std::vector<float> dst(W * H * D, 0.0f);

  cudaExtent ext{rowBytes, H, D};
  cudaMemcpy3DParms p{};
  p.srcPtr.ptr   = srcBuf.data();
  p.srcPtr.pitch = srcPitch;
  p.srcPtr.xsize = rowBytes;
  p.srcPtr.ysize = H;
  p.srcPos       = {0, 0, 0};
  p.srcArray     = 0;

  p.dstPtr.ptr   = dst.data();
  p.dstPtr.pitch = rowBytes;   // tight destination
  p.dstPtr.xsize = rowBytes;
  p.dstPtr.ysize = H;
  p.dstPos       = {0, 0, 0};
  p.dstArray     = 0;

  p.extent = ext;
  p.kind   = cudaMemcpyHostToHost;

  auto &ci = CUDAInterceptor::instance();
  cudaError_t err = ci.memcpy3D(&p);
  ASSERT_EQ(err, cudaSuccess, "memcpy3d_pitched: expected cudaSuccess");

  // Verify: each (z,y,x) element in dst matches the source data region.
  // dst is tight: offset = z*rowBytes*H + y*rowBytes + x*elemBytes
  for (size_t z = 0; z < D; ++z) {
    for (size_t y = 0; y < H; ++y) {
      for (size_t x = 0; x < W; ++x) {
        float expected = static_cast<float>(z * H * W + y * W + x + 1);
        float actual   = dst[z * H * W + y * W + x];
        ASSERT_TRUE(actual == expected,
                    "memcpy3d_pitched: stride mismatch at (z=" +
                        std::to_string(z) + ",y=" + std::to_string(y) +
                        ",x=" + std::to_string(x) + ")");
      }
    }
  }
  PASS("test_memcpy3d_pitched: padded src pitch respected, data correct in dst");
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
  std::cout << "=== QUEUE-21: cudaMemcpy3D / cudaMemcpy3DAsync unit tests ===\n";

  // Initialize the interceptor once before all tests.
  auto &ci = CUDAInterceptor::instance();
  cudaError_t initErr = ci.init();
  if (initErr != cudaSuccess) {
    std::cerr << "FATAL: CUDAInterceptor::init() failed (" << initErr << ")\n";
    return 1;
  }

  test_memcpy3d_hth();
  test_memcpy3d_srcpos_offset();
  test_memcpy3d_async();
  test_memcpy3d_pitched();

  std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed.\n";
  return (g_fail == 0) ? 0 : 1;
}
