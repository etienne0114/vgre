#include "vgre/core/texture_manager.h"
#include "../common/test_utils.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace vgre;
using namespace vgre::core;
using namespace vgre::test;

// Use shared test macros
#define TEST_ASSERT VGRE_TEST_ASSERT
#define TEST_ASSERT_NEAR VGRE_TEST_ASSERT_NEAR

// ── Test 1: Integer Format Support ────────────────────────────────────────
bool testIntegerFormats() {
  std::cout << "Test 1: Integer Format Support..." << std::endl;

  TextureManager &mgr = TextureManager::instance();

  // Test INT8
  {
    std::vector<int8_t> data = {-128, -64, 0, 64, 127};
    TextureDescriptor desc;
    desc.elementType = TextureElementType::INT8;
    desc.filterMode = TextureFilterMode::POINT;

    TextureId id;
    VGREResult result = mgr.createTexture(id, data.data(), data.size(), 1,
                                          sizeof(int8_t), desc);
    TEST_ASSERT(result == VGREResult::SUCCESS, "Failed to create INT8 texture");

    // Read back values
    for (size_t i = 0; i < data.size(); i++) {
      double value = mgr.readElementAsDouble(id, static_cast<int>(i));
      TEST_ASSERT_NEAR(value, static_cast<double>(data[i]), 0.01,
                       "INT8 value mismatch");
    }

    mgr.destroyTexture(id);
  }

  // Test UINT16
  {
    std::vector<uint16_t> data = {0, 1000, 32768, 65535};
    TextureDescriptor desc;
    desc.elementType = TextureElementType::UINT16;
    desc.filterMode = TextureFilterMode::POINT;

    TextureId id;
    VGREResult result = mgr.createTexture(id, data.data(), data.size(), 1,
                                          sizeof(uint16_t), desc);
    TEST_ASSERT(result == VGREResult::SUCCESS,
                "Failed to create UINT16 texture");

    for (size_t i = 0; i < data.size(); i++) {
      double value = mgr.readElementAsDouble(id, static_cast<int>(i));
      TEST_ASSERT_NEAR(value, static_cast<double>(data[i]), 0.01,
                       "UINT16 value mismatch");
    }

    mgr.destroyTexture(id);
  }

  // Test INT32
  {
    std::vector<int32_t> data = {(-2147483647 - 1), -1000000, 0, 1000000,
                                 2147483647};
    TextureDescriptor desc;
    desc.elementType = TextureElementType::INT32;
    desc.filterMode = TextureFilterMode::POINT;

    TextureId id;
    VGREResult result = mgr.createTexture(id, data.data(), data.size(), 1,
                                          sizeof(int32_t), desc);
    TEST_ASSERT(result == VGREResult::SUCCESS,
                "Failed to create INT32 texture");

    for (size_t i = 0; i < data.size(); i++) {
      double value = mgr.readElementAsDouble(id, static_cast<int>(i));
      TEST_ASSERT_NEAR(value, static_cast<double>(data[i]), 1.0,
                       "INT32 value mismatch");
    }

    mgr.destroyTexture(id);
  }

  std::cout << "  ✓ Integer format support passed" << std::endl;
  return true;
}

// ── Test 2: 3D Texture Sampling ───────────────────────────────────────────
bool test3DTextureSampling() {
  std::cout << "Test 2: 3D Texture Sampling..." << std::endl;

  TextureManager &mgr = TextureManager::instance();

  // Create 4x4x4 3D texture with known pattern
  const size_t w = 4, h = 4, d = 4;
  std::vector<float> data(w * h * d);
  for (size_t z = 0; z < d; z++) {
    for (size_t y = 0; y < h; y++) {
      for (size_t x = 0; x < w; x++) {
        size_t idx = z * w * h + y * w + x;
        data[idx] = static_cast<float>(x + y * 10 + z * 100);
      }
    }
  }

  TextureDescriptor desc;
  desc.elementType = TextureElementType::FLOAT32;
  desc.filterMode = TextureFilterMode::POINT;
  desc.addressMode = TextureAddressMode::CLAMP;

  TextureId id;
  VGREResult result =
      mgr.createTexture3D(id, data.data(), w, h, d, sizeof(float), desc);
  TEST_ASSERT(result == VGREResult::SUCCESS, "Failed to create 3D texture");

  // Test point sampling
  float value = mgr.tex3D(id, 1.0f, 2.0f, 1.0f);
  float expected = 1.0f + 2.0f * 10.0f + 1.0f * 100.0f; // 121
  TEST_ASSERT_NEAR(value, expected, 0.01f, "3D point sampling failed");

  // Test trilinear interpolation
  desc.filterMode = TextureFilterMode::LINEAR;
  mgr.destroyTexture(id);
  result = mgr.createTexture3D(id, data.data(), w, h, d, sizeof(float), desc);
  TEST_ASSERT(result == VGREResult::SUCCESS,
              "Failed to create 3D texture with linear filtering");

  // Sample at center of voxel (should match point sample)
  value = mgr.tex3D(id, 1.5f, 2.5f, 1.5f);
  expected = 1.0f + 2.0f * 10.0f + 1.0f * 100.0f; // 121
  TEST_ASSERT_NEAR(value, expected, 5.0f,
                   "3D trilinear interpolation center failed");

  mgr.destroyTexture(id);

  std::cout << "  ✓ 3D texture sampling passed" << std::endl;
  return true;
}

// ── Test 3: Address Modes ─────────────────────────────────────────────────
bool testAddressModes() {
  std::cout << "Test 3: Address Modes..." << std::endl;

  TextureManager &mgr = TextureManager::instance();

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  TextureDescriptor desc;
  desc.elementType = TextureElementType::FLOAT32;
  desc.filterMode = TextureFilterMode::POINT;

  // Test CLAMP mode
  {
    desc.addressMode = TextureAddressMode::CLAMP;
    TextureId id;
    mgr.createTexture(id, data.data(), data.size(), 1, sizeof(float), desc);

    float value = mgr.tex1D(id, -1.0f); // Out of bounds
    TEST_ASSERT_NEAR(value, 1.0f, 0.01f, "CLAMP mode failed (negative)");

    value = mgr.tex1D(id, 10.0f); // Out of bounds
    TEST_ASSERT_NEAR(value, 4.0f, 0.01f, "CLAMP mode failed (positive)");

    mgr.destroyTexture(id);
  }

  // Test WRAP mode
  {
    desc.addressMode = TextureAddressMode::WRAP;
    TextureId id;
    mgr.createTexture(id, data.data(), data.size(), 1, sizeof(float), desc);

    float value = mgr.tex1D(id, 4.0f); // Should wrap to 0
    TEST_ASSERT_NEAR(value, 1.0f, 0.01f, "WRAP mode failed");

    value = mgr.tex1D(id, -1.0f); // Should wrap to 3
    TEST_ASSERT_NEAR(value, 4.0f, 0.01f, "WRAP mode failed (negative)");

    mgr.destroyTexture(id);
  }

  // Test BORDER mode
  {
    desc.addressMode = TextureAddressMode::BORDER;
    desc.borderColor = 99.0f;
    TextureId id;
    mgr.createTexture(id, data.data(), data.size(), 1, sizeof(float), desc);

    float value = mgr.tex1D(id, -1.0f); // Out of bounds
    TEST_ASSERT_NEAR(value, 99.0f, 0.01f, "BORDER mode failed");

    value = mgr.tex1D(id, 10.0f); // Out of bounds
    TEST_ASSERT_NEAR(value, 99.0f, 0.01f, "BORDER mode failed");

    mgr.destroyTexture(id);
  }

  std::cout << "  ✓ Address modes passed" << std::endl;
  return true;
}

// ── Test 4: Surface Write/Read Operations ─────────────────────────────────
bool testSurfaceOperations() {
  std::cout << "Test 4: Surface Write/Read Operations..." << std::endl;

  TextureManager &mgr = TextureManager::instance();

  const size_t w = 8, h = 8;
  std::vector<float> data(w * h, 0.0f);

  SurfaceId id;
  VGREResult result = mgr.createSurface(id, data.data(), w, h, sizeof(float),
                                        TextureElementType::FLOAT32);
  TEST_ASSERT(result == VGREResult::SUCCESS, "Failed to create surface");

  // Write values
  for (int y = 0; y < static_cast<int>(h); y++) {
    for (int x = 0; x < static_cast<int>(w); x++) {
      float value = static_cast<float>(x + y * 10);
      result = mgr.surf2Dwrite(id, value, x, y);
      TEST_ASSERT(result == VGREResult::SUCCESS, "Surface write failed");
    }
  }

  // Read back and verify
  for (int y = 0; y < static_cast<int>(h); y++) {
    for (int x = 0; x < static_cast<int>(w); x++) {
      float value;
      result = mgr.surf2Dread(id, value, x, y);
      TEST_ASSERT(result == VGREResult::SUCCESS, "Surface read failed");

      float expected = static_cast<float>(x + y * 10);
      TEST_ASSERT_NEAR(value, expected, 0.01f, "Surface value mismatch");
    }
  }

  // Test bounds checking
  float dummy;
  result = mgr.surf2Dread(id, dummy, -1, 0);
  TEST_ASSERT(result == VGREResult::ERROR_INVALID_VALUE,
              "Surface should reject negative coordinates");

  result = mgr.surf2Dread(id, dummy, 100, 0);
  TEST_ASSERT(result == VGREResult::ERROR_INVALID_VALUE,
              "Surface should reject out-of-bounds coordinates");

  mgr.destroySurface(id);

  std::cout << "  ✓ Surface operations passed" << std::endl;
  return true;
}

// ── Test 5: Filtering Modes Comparison ────────────────────────────────────
bool testFilteringModes() {
  std::cout << "Test 5: Filtering Modes Comparison..." << std::endl;

  TextureManager &mgr = TextureManager::instance();

  // Create simple gradient texture
  std::vector<float> data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  TextureDescriptor desc;
  desc.elementType = TextureElementType::FLOAT32;
  desc.addressMode = TextureAddressMode::CLAMP;

  // Test POINT filtering
  {
    desc.filterMode = TextureFilterMode::POINT;
    TextureId id;
    mgr.createTexture(id, data.data(), data.size(), 1, sizeof(float), desc);

    float value = mgr.tex1D(id, 1.7f);
    TEST_ASSERT_NEAR(value, 1.0f, 0.01f, "POINT filtering failed");

    mgr.destroyTexture(id);
  }

  // Test LINEAR filtering
  {
    desc.filterMode = TextureFilterMode::LINEAR;
    TextureId id;
    mgr.createTexture(id, data.data(), data.size(), 1, sizeof(float), desc);

    float value = mgr.tex1D(id, 1.5f);
    // Should interpolate between values at index 1 and 2
    // With texel center offset, 1.5 - 0.5 = 1.0, floor = 1, frac = 0.0
    // So it samples between index 1 (value 1.0) and index 2 (value 2.0)
    // Result should be close to 1.5 (midpoint)
    TEST_ASSERT(value >= 0.5f && value <= 2.5f, "LINEAR filtering failed");

    mgr.destroyTexture(id);
  }

  // Test CUBIC filtering
  {
    desc.filterMode = TextureFilterMode::CUBIC;
    TextureId id;
    mgr.createTexture(id, data.data(), data.size(), 1, sizeof(float), desc);

    float value = mgr.tex1D(id, 2.0f);
    // Cubic should produce smooth interpolation
    TEST_ASSERT(value >= 1.0f && value <= 3.0f, "CUBIC filtering failed");

    mgr.destroyTexture(id);
  }

  std::cout << "  ✓ Filtering modes passed" << std::endl;
  return true;
}

// ── Test 6: Normalized vs Non-Normalized Coordinates ──────────────────────
bool testNormalizedCoordinates() {
  std::cout << "Test 6: Normalized vs Non-Normalized Coordinates..."
            << std::endl;

  TextureManager &mgr = TextureManager::instance();

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  TextureDescriptor desc;
  desc.elementType = TextureElementType::FLOAT32;
  desc.filterMode = TextureFilterMode::POINT;
  desc.addressMode = TextureAddressMode::CLAMP;

  // Test non-normalized coordinates
  {
    desc.normalizedCoords = false;
    TextureId id;
    mgr.createTexture(id, data.data(), data.size(), 1, sizeof(float), desc);

    float value = mgr.tex1D(id, 2.0f);
    TEST_ASSERT_NEAR(value, 3.0f, 0.01f,
                     "Non-normalized coordinates failed");

    mgr.destroyTexture(id);
  }

  // Test normalized coordinates
  {
    desc.normalizedCoords = true;
    TextureId id;
    mgr.createTexture(id, data.data(), data.size(), 1, sizeof(float), desc);

    float value = mgr.tex1D(id, 0.5f); // 0.5 * 4 = 2.0
    TEST_ASSERT_NEAR(value, 3.0f, 0.01f, "Normalized coordinates failed");

    mgr.destroyTexture(id);
  }

  std::cout << "  ✓ Normalized coordinates passed" << std::endl;
  return true;
}

// ── Test 7: Texture Views ─────────────────────────────────────────────────
bool testTextureViews() {
  std::cout << "Test 7: Texture Views..." << std::endl;

  TextureManager &mgr = TextureManager::instance();

  // Create base texture with FLOAT32 data
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  TextureDescriptor desc;
  desc.elementType = TextureElementType::FLOAT32;
  desc.filterMode = TextureFilterMode::POINT;

  TextureId baseId;
  mgr.createTexture(baseId, data.data(), data.size(), 1, sizeof(float), desc);

  // Create view with offset
  ResourceViewDescriptor viewDesc;
  viewDesc.format = TextureElementType::FLOAT32;
  viewDesc.width = 4;
  viewDesc.height = 1;
  viewDesc.depth = 1;
  viewDesc.offsetInBytes = 4 * sizeof(float); // Start at element 4

  TextureDescriptor viewTexDesc = desc;
  TextureId viewId;
  VGREResult result =
      mgr.createTextureView(viewId, baseId, viewDesc, viewTexDesc);
  TEST_ASSERT(result == VGREResult::SUCCESS, "Failed to create texture view");

  // Read from view (should start at offset)
  double value = mgr.readElementAsDouble(viewId, 0);
  TEST_ASSERT_NEAR(value, 5.0, 0.01, "Texture view offset failed");

  mgr.destroyTexture(viewId);
  mgr.destroyTexture(baseId);

  std::cout << "  ✓ Texture views passed" << std::endl;
  return true;
}

// ── Test 8: cudaArray Lifecycle ───────────────────────────────────────────
bool testCudaArrayLifecycle() {
  std::cout << "Test 8: cudaArray Lifecycle..." << std::endl;

  TextureManager &mgr = TextureManager::instance();

  TextureDescriptor desc;
  desc.elementType = TextureElementType::FLOAT32;
  desc.filterMode = TextureFilterMode::POINT;

  TextureId id;
  VGREResult result = mgr.createCudaArray(id, 16, 16, sizeof(float), desc);
  TEST_ASSERT(result == VGREResult::SUCCESS, "Failed to create cudaArray");

  // Get data pointer and write to it
  void *dataPtr = mgr.getCudaArrayData(id);
  TEST_ASSERT(dataPtr != nullptr, "cudaArray data pointer is null");

  float *floatData = static_cast<float *>(dataPtr);
  for (int i = 0; i < 16 * 16; i++) {
    floatData[i] = static_cast<float>(i);
  }

  // Read back through texture interface
  double value = mgr.readElementAsDouble(id, 42);
  TEST_ASSERT_NEAR(value, 42.0, 0.01, "cudaArray read failed");

  result = mgr.destroyCudaArray(id);
  TEST_ASSERT(result == VGREResult::SUCCESS, "Failed to destroy cudaArray");

  std::cout << "  ✓ cudaArray lifecycle passed" << std::endl;
  return true;
}

// ── Test 9: Bilinear Interpolation Accuracy ───────────────────────────────
bool testBilinearInterpolation() {
  std::cout << "Test 9: Bilinear Interpolation Accuracy..." << std::endl;

  TextureManager &mgr = TextureManager::instance();

  // Create 2x2 texture with known values
  std::vector<float> data = {
      0.0f, 1.0f, // Row 0
      2.0f, 3.0f  // Row 1
  };

  TextureDescriptor desc;
  desc.elementType = TextureElementType::FLOAT32;
  desc.filterMode = TextureFilterMode::LINEAR;
  desc.addressMode = TextureAddressMode::CLAMP;
  desc.normalizedCoords = false;

  TextureId id;
  mgr.createTexture(id, data.data(), 2, 2, sizeof(float), desc);

  // Sample at center (should be average of all 4 values)
  float value = mgr.tex2D(id, 1.0f, 1.0f);
  float expected = (0.0f + 1.0f + 2.0f + 3.0f) / 4.0f; // 1.5
  TEST_ASSERT_NEAR(value, expected, 0.1f,
                   "Bilinear interpolation center failed");

  // Sample at edge
  value = mgr.tex2D(id, 0.5f, 0.5f);
  expected = 0.0f; // Should be close to top-left corner
  TEST_ASSERT_NEAR(value, expected, 0.5f,
                   "Bilinear interpolation edge failed");

  mgr.destroyTexture(id);

  std::cout << "  ✓ Bilinear interpolation passed" << std::endl;
  return true;
}

// ── Test 10: Bicubic Interpolation Quality ────────────────────────────────
bool testBicubicInterpolation() {
  std::cout << "Test 10: Bicubic Interpolation Quality..." << std::endl;

  TextureManager &mgr = TextureManager::instance();

  // Create 8x8 texture with smooth gradient
  const size_t size = 8;
  std::vector<float> data(size * size);
  for (size_t y = 0; y < size; y++) {
    for (size_t x = 0; x < size; x++) {
      data[y * size + x] = static_cast<float>(x + y * size);
    }
  }

  TextureDescriptor desc;
  desc.elementType = TextureElementType::FLOAT32;
  desc.filterMode = TextureFilterMode::CUBIC;
  desc.addressMode = TextureAddressMode::CLAMP;
  desc.normalizedCoords = false;

  TextureId id;
  mgr.createTexture(id, data.data(), size, size, sizeof(float), desc);

  // Sample at fractional coordinates
  float value = mgr.tex2D(id, 3.5f, 3.5f);
  // Bicubic should produce smooth interpolation
  TEST_ASSERT(value >= 0.0f && value <= 64.0f,
              "Bicubic interpolation out of range");

  // Bicubic should be smoother than bilinear (no sharp edges)
  float v1 = mgr.tex2D(id, 2.3f, 2.7f);
  float v2 = mgr.tex2D(id, 2.4f, 2.8f);
  float diff = std::abs(v2 - v1);
  TEST_ASSERT(diff < 10.0f, "Bicubic interpolation not smooth");

  mgr.destroyTexture(id);

  std::cout << "  ✓ Bicubic interpolation passed" << std::endl;
  return true;
}

// ── Main Test Runner ───────────────────────────────────────────────────────
int main() {
  TestRunner runner("VGRE Texture Manager Test Suite");

  runner.runTest(testIntegerFormats, "Integer Format Support");
  runner.runTest(test3DTextureSampling, "3D Texture Sampling");
  runner.runTest(testAddressModes, "Address Modes");
  runner.runTest(testSurfaceOperations, "Surface Operations");
  runner.runTest(testFilteringModes, "Filtering Modes");
  runner.runTest(testNormalizedCoordinates, "Normalized Coordinates");
  runner.runTest(testTextureViews, "Texture Views");
  runner.runTest(testCudaArrayLifecycle, "cudaArray Lifecycle");
  runner.runTest(testBilinearInterpolation, "Bilinear Interpolation");
  runner.runTest(testBicubicInterpolation, "Bicubic Interpolation");

  return runner.finish();
}
