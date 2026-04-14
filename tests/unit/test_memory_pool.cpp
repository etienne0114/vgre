/**
 * @file test_memory_pool.cpp
 * @brief Unit tests for VGRE Memory Pool APIs (cudaMallocAsync / cudaFreeAsync).
 *
 * Tests the stream-ordered pool allocation system in MemoryManager,
 * including pool creation, block reuse, capacity limits, and cleanup.
 */

#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

#include <cstdio>
#include <cstring>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                                                             \
  do {                                                                         \
    printf("  Testing: %-50s ", name);                                         \
  } while (0)

#define PASS()                                                                 \
  do {                                                                         \
    printf("[PASS]\n");                                                        \
    tests_passed++;                                                            \
  } while (0)

#define FAIL(msg)                                                              \
  do {                                                                         \
    printf("[FAIL] %s\n", msg);                                                \
    tests_failed++;                                                            \
  } while (0)

#define ASSERT_EQ(a, b, msg)                                                   \
  do {                                                                         \
    if ((a) != (b)) {                                                          \
      FAIL(msg);                                                               \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define ASSERT_NE(a, b, msg)                                                   \
  do {                                                                         \
    if ((a) == (b)) {                                                          \
      FAIL(msg);                                                               \
      return;                                                                  \
    }                                                                          \
  } while (0)

using namespace vgre;
using namespace vgre::core;

// ── Test 1: Pool creation and destruction ──────────────────────────────────
static void test_pool_create_destroy() {
  TEST("Pool create and destroy");
  auto &mm = RuntimeEngine::instance().getMemoryManager();

  PoolHandle pool = 0;
  auto r = mm.createPool(pool, 256);
  ASSERT_EQ(r, VGREResult::SUCCESS, "createPool should succeed");
  ASSERT_NE(pool, 0ULL, "pool handle should be non-zero");

  r = mm.destroyPool(pool);
  ASSERT_EQ(r, VGREResult::SUCCESS, "destroyPool should succeed");

  // Destroying again should fail
  r = mm.destroyPool(pool);
  ASSERT_EQ(r, VGREResult::ERR_INVALID_VALUE,
            "double-destroy should fail");
  PASS();
}

// ── Test 2: allocateFromPool / freeToPool round-trip ───────────────────────
static void test_pool_alloc_free() {
  TEST("Pool alloc and free round-trip");
  auto &mm = RuntimeEngine::instance().getMemoryManager();

  PoolHandle pool = 0;
  auto r = mm.createPool(pool, 128);
  ASSERT_EQ(r, VGREResult::SUCCESS, "createPool should succeed");

  MemoryHandle handle = nullptr;
  r = mm.allocateFromPool(pool, 1024, handle);
  ASSERT_EQ(r, VGREResult::SUCCESS, "allocateFromPool should succeed");
  ASSERT_NE(handle, nullptr, "handle should be non-null");

  // Write to the allocation to verify it's real memory
  std::memset(handle, 0xAB, 1024);

  // Verify first byte
  uint8_t *data = static_cast<uint8_t *>(handle);
  ASSERT_EQ(data[0], 0xAB, "written data should be readable");
  ASSERT_EQ(data[1023], 0xAB, "written data should be readable at end");

  r = mm.freeToPool(pool, handle);
  ASSERT_EQ(r, VGREResult::SUCCESS, "freeToPool should succeed");

  r = mm.destroyPool(pool);
  ASSERT_EQ(r, VGREResult::SUCCESS, "destroyPool should succeed");
  PASS();
}

// ── Test 3: Block reuse after free ─────────────────────────────────────────
static void test_pool_block_reuse() {
  TEST("Pool block reuse (alloc->free->alloc reuses block)");
  auto &mm = RuntimeEngine::instance().getMemoryManager();

  PoolHandle pool = 0;
  auto r = mm.createPool(pool, 256);
  ASSERT_EQ(r, VGREResult::SUCCESS, "createPool should succeed");

  // First allocation
  MemoryHandle handle1 = nullptr;
  r = mm.allocateFromPool(pool, 256, handle1);
  ASSERT_EQ(r, VGREResult::SUCCESS, "first alloc should succeed");
  ASSERT_NE(handle1, nullptr, "first handle should be non-null");

  // Free it
  r = mm.freeToPool(pool, handle1);
  ASSERT_EQ(r, VGREResult::SUCCESS, "free should succeed");

  // Second allocation of same size — should reuse the freed block
  MemoryHandle handle2 = nullptr;
  r = mm.allocateFromPool(pool, 256, handle2);
  ASSERT_EQ(r, VGREResult::SUCCESS, "second alloc should succeed");
  ASSERT_NE(handle2, nullptr, "second handle should be non-null");

  // The reused block should be the same pointer
  ASSERT_EQ(handle1, handle2, "reused block should have same pointer");

  r = mm.freeToPool(pool, handle2);
  ASSERT_EQ(r, VGREResult::SUCCESS, "cleanup free should succeed");

  r = mm.destroyPool(pool);
  ASSERT_EQ(r, VGREResult::SUCCESS, "destroyPool should succeed");
  PASS();
}

// ── Test 4: Multiple allocations ───────────────────────────────────────────
static void test_pool_multiple_allocs() {
  TEST("Pool multiple allocations");
  auto &mm = RuntimeEngine::instance().getMemoryManager();

  PoolHandle pool = 0;
  auto r = mm.createPool(pool, 64);
  ASSERT_EQ(r, VGREResult::SUCCESS, "createPool should succeed");

  std::vector<MemoryHandle> handles;
  const int NUM_ALLOCS = 10;

  for (int i = 0; i < NUM_ALLOCS; ++i) {
    MemoryHandle h = nullptr;
    r = mm.allocateFromPool(pool, 512, h);
    ASSERT_EQ(r, VGREResult::SUCCESS, "alloc should succeed");
    ASSERT_NE(h, nullptr, "handle should be non-null");

    // Write unique pattern to verify isolation
    std::memset(h, i + 1, 512);
    handles.push_back(h);
  }

  // Verify all allocations are distinct
  for (int i = 0; i < NUM_ALLOCS; ++i) {
    for (int j = i + 1; j < NUM_ALLOCS; ++j) {
      ASSERT_NE(handles[i], handles[j],
                "different allocations should have different pointers");
    }
  }

  // Verify data integrity
  for (int i = 0; i < NUM_ALLOCS; ++i) {
    uint8_t *data = static_cast<uint8_t *>(handles[i]);
    ASSERT_EQ(data[0], static_cast<uint8_t>(i + 1),
              "data should be intact");
  }

  // Free all
  for (auto &h : handles) {
    r = mm.freeToPool(pool, h);
    ASSERT_EQ(r, VGREResult::SUCCESS, "free should succeed");
  }

  r = mm.destroyPool(pool);
  ASSERT_EQ(r, VGREResult::SUCCESS, "destroyPool should succeed");
  PASS();
}

// ── Test 5: Invalid pool handle ────────────────────────────────────────────
static void test_pool_invalid_handle() {
  TEST("Pool invalid handle errors");
  auto &mm = RuntimeEngine::instance().getMemoryManager();

  // Allocate from non-existent pool
  MemoryHandle h = nullptr;
  auto r = mm.allocateFromPool(99999, 256, h);
  ASSERT_EQ(r, VGREResult::ERR_INVALID_VALUE,
            "alloc from invalid pool should fail");

  // Free to non-existent pool
  r = mm.freeToPool(99999, h);
  ASSERT_EQ(r, VGREResult::ERR_INVALID_VALUE,
            "free to invalid pool should fail");

  // Destroy non-existent pool
  r = mm.destroyPool(99999);
  ASSERT_EQ(r, VGREResult::ERR_INVALID_VALUE,
            "destroy invalid pool should fail");
  PASS();
}

// ── Test 6: Block size enforcement ─────────────────────────────────────────
static void test_pool_block_size() {
  TEST("Pool block size enforcement (minimum 64)");
  auto &mm = RuntimeEngine::instance().getMemoryManager();

  PoolHandle pool = 0;
  // Request block size < 64 — should be clamped to 64
  auto r = mm.createPool(pool, 1);
  ASSERT_EQ(r, VGREResult::SUCCESS, "createPool with small blockSize should succeed");

  // Allocate 10 bytes — should be rounded up to block size (64)
  MemoryHandle h = nullptr;
  r = mm.allocateFromPool(pool, 10, h);
  ASSERT_EQ(r, VGREResult::SUCCESS, "small alloc should succeed");
  ASSERT_NE(h, nullptr, "handle should be non-null");

  // We can write the full block size since it's rounded up
  std::memset(h, 0, 64);

  r = mm.freeToPool(pool, h);
  ASSERT_EQ(r, VGREResult::SUCCESS, "free should succeed");

  r = mm.destroyPool(pool);
  ASSERT_EQ(r, VGREResult::SUCCESS, "destroyPool should succeed");
  PASS();
}

// ── Test 7: Pool stats tracking ────────────────────────────────────────────
static void test_pool_stats() {
  TEST("Pool statistics tracking");
  auto &mm = RuntimeEngine::instance().getMemoryManager();

  PoolHandle pool = 0;
  auto r = mm.createPool(pool, 256);
  ASSERT_EQ(r, VGREResult::SUCCESS, "createPool should succeed");

  // Allocate 3 blocks
  MemoryHandle h1 = nullptr, h2 = nullptr, h3 = nullptr;
  mm.allocateFromPool(pool, 256, h1);
  mm.allocateFromPool(pool, 256, h2);
  mm.allocateFromPool(pool, 256, h3);

  // Free 2 blocks
  mm.freeToPool(pool, h1);
  mm.freeToPool(pool, h2);

  // Check stats via getPools()
  const auto &pools = mm.getPools();
  auto it = pools.find(pool);
  ASSERT_NE(it == pools.end(), true, "pool should be in pools map");

  ASSERT_EQ(it->second.allocCount, 3UL, "allocCount should be 3");
  ASSERT_EQ(it->second.freeCount, 2UL, "freeCount should be 2");

  // Cleanup
  mm.freeToPool(pool, h3);
  r = mm.destroyPool(pool);
  ASSERT_EQ(r, VGREResult::SUCCESS, "destroyPool should succeed");
  PASS();
}

int main() {
  printf("=== VGRE Memory Pool Test Suite ===\n\n");

  // Initialize the runtime engine
  auto r = vgre::core::RuntimeEngine::instance().initialize();
  if (r != vgre::VGREResult::SUCCESS) {
    printf("FATAL: Failed to initialize RuntimeEngine\n");
    return 1;
  }

  test_pool_create_destroy();
  test_pool_alloc_free();
  test_pool_block_reuse();
  test_pool_multiple_allocs();
  test_pool_invalid_handle();
  test_pool_block_size();
  test_pool_stats();

  printf("\n=== Results: %d/%d passed ===\n", tests_passed,
         tests_passed + tests_failed);

  vgre::core::RuntimeEngine::instance().shutdown();

  return tests_failed > 0 ? 1 : 0;
}
