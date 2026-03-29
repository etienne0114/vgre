/**
 * VGRE Unit Tests — Input Validation
 * Tests the input validation utilities for security hardening
 */
#include "vgre/common/input_validation.h"
#include "../common/test_utils.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <cstring>

using namespace vgre;
using namespace vgre::common;
using namespace vgre::test;

// Forward declarations
bool test_allocation_size_validation();
bool test_memcpy_params_validation();
bool test_packet_size_validation();
bool test_string_length_validation();
bool test_grid_dim_validation();
bool test_block_dim_validation();
bool test_safe_arithmetic();
bool test_memory_offset_validation();
bool test_packet_type_validation();
bool test_sequence_number_validation();
bool test_crc32_checksum();
bool test_texture_coords_validation();
bool test_mipmap_level_validation();
bool test_surface_bounds_validation();

bool test_allocation_size_validation() {
    std::cout << "Test: Allocation Size Validation..." << std::endl;
    
    // Valid sizes
    VGRE_TEST_ASSERT(InputValidator::validateAllocationSize(1024) == VGREResult::SUCCESS, "1KB should be valid");
    VGRE_TEST_ASSERT(InputValidator::validateAllocationSize(1024 * 1024) == VGREResult::SUCCESS, "1MB should be valid");
    VGRE_TEST_ASSERT(InputValidator::validateAllocationSize(1024 * 1024 * 1024) == VGREResult::SUCCESS, "1GB should be valid");
    
    // Invalid: zero
    VGRE_TEST_ASSERT(InputValidator::validateAllocationSize(0) == VGREResult::ERROR_INVALID_VALUE, "Zero should be invalid");
    
    // Invalid: too large
    VGRE_TEST_ASSERT(InputValidator::validateAllocationSize(InputValidator::MAX_ALLOCATION_SIZE + 1) == VGREResult::ERROR_INVALID_VALUE, "Too large should be invalid");
    
    std::cout << "  ✓ Allocation size validation passed" << std::endl;
    return true;
}

bool test_memcpy_params_validation() {
    std::cout << "\nTest: Memcpy Parameters Validation..." << std::endl;
    
    char src[1024];
    char dst[1024];
    
    // Valid parameters
    VGRE_TEST_ASSERT(InputValidator::validateMemcpyParams(dst, src, 1024) == VGREResult::SUCCESS, "Valid params should succeed");
    
    // Invalid: NULL destination
    VGRE_TEST_ASSERT(InputValidator::validateMemcpyParams(nullptr, src, 1024) == VGREResult::ERROR_INVALID_VALUE, "NULL dst should be invalid");
    
    // Invalid: NULL source
    VGRE_TEST_ASSERT(InputValidator::validateMemcpyParams(dst, nullptr, 1024) == VGREResult::ERROR_INVALID_VALUE, "NULL src should be invalid");
    
    // Invalid: zero count
    VGRE_TEST_ASSERT(InputValidator::validateMemcpyParams(dst, src, 0) == VGREResult::ERROR_INVALID_VALUE, "Zero count should be invalid");
    
    // Invalid: overlapping regions
    VGRE_TEST_ASSERT(InputValidator::validateMemcpyParams(dst, dst + 512, 1024) == VGREResult::ERROR_INVALID_VALUE, "Overlapping regions should be invalid");
    
    std::cout << "  ✓ Memcpy parameters validation passed" << std::endl;
    return true;
}

bool test_packet_size_validation() {
    std::cout << "\nTest: Packet Size Validation..." << std::endl;
    
    // Valid sizes
    VGRE_TEST_ASSERT(InputValidator::validatePacketSize(1024) == VGREResult::SUCCESS, "1KB packet should be valid");
    VGRE_TEST_ASSERT(InputValidator::validatePacketSize(1024 * 1024) == VGREResult::SUCCESS, "1MB packet should be valid");
    
    // Invalid: zero
    VGRE_TEST_ASSERT(InputValidator::validatePacketSize(0) == VGREResult::ERROR_INVALID_VALUE, "Zero should be invalid");
    
    // Invalid: too large
    VGRE_TEST_ASSERT(InputValidator::validatePacketSize(InputValidator::MAX_PACKET_SIZE + 1) == VGREResult::ERROR_INVALID_VALUE, "Too large should be invalid");
    
    std::cout << "  ✓ Packet size validation passed" << std::endl;
    return true;
}

bool test_string_length_validation() {
    std::cout << "\nTest: String Length Validation..." << std::endl;
    
    // Valid strings
    const char* short_str = "Hello";
    VGRE_TEST_ASSERT(InputValidator::validateStringLength(short_str) == VGREResult::SUCCESS, "Valid string should succeed");
    
    // Invalid: NULL
    VGRE_TEST_ASSERT(InputValidator::validateStringLength(nullptr) == VGREResult::ERROR_INVALID_VALUE, "NULL should be invalid");
    
    std::cout << "  ✓ String length validation passed" << std::endl;
    return true;
}

bool test_grid_dim_validation() {
    std::cout << "\nTest: Grid Dimension Validation..." << std::endl;
    
    // Valid dimensions
    VGRE_TEST_ASSERT(InputValidator::validateGridDim(1, 1, 1) == VGREResult::SUCCESS, "1x1x1 should be valid");
    VGRE_TEST_ASSERT(InputValidator::validateGridDim(256, 256, 1) == VGREResult::SUCCESS, "256x256x1 should be valid");
    VGRE_TEST_ASSERT(InputValidator::validateGridDim(65535, 65535, 65535) == VGREResult::SUCCESS, "Max dims should be valid");
    
    // Invalid: zero
    VGRE_TEST_ASSERT(InputValidator::validateGridDim(0, 1, 1) == VGREResult::ERROR_INVALID_VALUE, "Zero x should be invalid");
    VGRE_TEST_ASSERT(InputValidator::validateGridDim(1, 0, 1) == VGREResult::ERROR_INVALID_VALUE, "Zero y should be invalid");
    VGRE_TEST_ASSERT(InputValidator::validateGridDim(1, 1, 0) == VGREResult::ERROR_INVALID_VALUE, "Zero z should be invalid");
    
    // Invalid: too large
    VGRE_TEST_ASSERT(InputValidator::validateGridDim(65536, 1, 1) == VGREResult::ERROR_INVALID_VALUE, "Too large should be invalid");
    
    std::cout << "  ✓ Grid dimension validation passed" << std::endl;
    return true;
}

bool test_block_dim_validation() {
    std::cout << "\nTest: Block Dimension Validation..." << std::endl;
    
    // Valid dimensions
    VGRE_TEST_ASSERT(InputValidator::validateBlockDim(1, 1, 1) == VGREResult::SUCCESS, "1x1x1 should be valid");
    VGRE_TEST_ASSERT(InputValidator::validateBlockDim(256, 1, 1) == VGREResult::SUCCESS, "256x1x1 should be valid");
    VGRE_TEST_ASSERT(InputValidator::validateBlockDim(32, 32, 1) == VGREResult::SUCCESS, "32x32x1 should be valid");
    
    // Invalid: zero
    VGRE_TEST_ASSERT(InputValidator::validateBlockDim(0, 1, 1) == VGREResult::ERROR_INVALID_VALUE, "Zero should be invalid");
    
    // Invalid: too large
    VGRE_TEST_ASSERT(InputValidator::validateBlockDim(2048, 1, 1) == VGREResult::ERROR_INVALID_VALUE, "Too large should be invalid");
    
    // Invalid: total threads too large
    VGRE_TEST_ASSERT(InputValidator::validateBlockDim(1024, 1024, 1) == VGREResult::ERROR_INVALID_VALUE, "Too many threads should be invalid");
    
    std::cout << "  ✓ Block dimension validation passed" << std::endl;
    return true;
}

bool test_safe_arithmetic() {
    std::cout << "\nTest: Safe Arithmetic Operations..." << std::endl;
    
    size_t result;
    
    // Valid multiplication
    VGRE_TEST_ASSERT(InputValidator::safeMul(100, 200, result) == true, "Valid multiplication should succeed");
    VGRE_TEST_ASSERT(result == 20000, "Result should be 20000");
    
    // Overflow multiplication
    size_t max = std::numeric_limits<size_t>::max();
    VGRE_TEST_ASSERT(InputValidator::safeMul(max, 2, result) == false, "Overflow multiplication should fail");
    
    // Valid addition
    VGRE_TEST_ASSERT(InputValidator::safeAdd(100, 200, result) == true, "Valid addition should succeed");
    VGRE_TEST_ASSERT(result == 300, "Result should be 300");
    
    // Overflow addition
    VGRE_TEST_ASSERT(InputValidator::safeAdd(max, 1, result) == false, "Overflow addition should fail");
    
    std::cout << "  ✓ Safe arithmetic operations passed" << std::endl;
    return true;
}

bool test_memory_offset_validation() {
    std::cout << "\nTest: Memory Offset Validation..." << std::endl;
    
    // Valid offset
    VGRE_TEST_ASSERT(InputValidator::validateMemoryOffset(1024, 0, 512) == VGREResult::SUCCESS, "Valid offset should succeed");
    VGRE_TEST_ASSERT(InputValidator::validateMemoryOffset(1024, 512, 512) == VGREResult::SUCCESS, "Valid offset at end should succeed");
    
    // Invalid: offset + count > base
    VGRE_TEST_ASSERT(InputValidator::validateMemoryOffset(1024, 512, 1024) == VGREResult::ERROR_INVALID_VALUE, "Out of bounds should be invalid");
    
    // Invalid: overflow
    size_t max = std::numeric_limits<size_t>::max();
    VGRE_TEST_ASSERT(InputValidator::validateMemoryOffset(max, max, 1) == VGREResult::ERROR_INVALID_VALUE, "Overflow should be invalid");
    
    std::cout << "  ✓ Memory offset validation passed" << std::endl;
    return true;
}

int main() {
    TestRunner runner("VGRE Input Validation Test Suite");
    
    // Original tests
    runner.runTest(test_allocation_size_validation, "Allocation Size Validation");
    runner.runTest(test_memcpy_params_validation, "Memcpy Parameters Validation");
    runner.runTest(test_packet_size_validation, "Packet Size Validation");
    runner.runTest(test_string_length_validation, "String Length Validation");
    runner.runTest(test_grid_dim_validation, "Grid Dimension Validation");
    runner.runTest(test_block_dim_validation, "Block Dimension Validation");
    runner.runTest(test_safe_arithmetic, "Safe Arithmetic Operations");
    runner.runTest(test_memory_offset_validation, "Memory Offset Validation");
    
    // Advanced validation tests
    runner.runTest(test_packet_type_validation, "Packet Type Validation");
    runner.runTest(test_sequence_number_validation, "Sequence Number Validation");
    runner.runTest(test_crc32_checksum, "CRC32 Checksum");
    runner.runTest(test_texture_coords_validation, "Texture Coordinates Validation");
    runner.runTest(test_mipmap_level_validation, "Mipmap Level Validation");
    runner.runTest(test_surface_bounds_validation, "Surface Bounds Validation");
    
    return runner.finish();
}


// ========== ADVANCED VALIDATION TESTS ==========

bool test_packet_type_validation() {
    std::cout << "\nTest: Packet Type Validation..." << std::endl;
    
    // Valid packet types
    VGRE_TEST_ASSERT(InputValidator::validatePacketType(0, 10) == VGREResult::SUCCESS,
                     "Valid packet type 0 should pass");
    VGRE_TEST_ASSERT(InputValidator::validatePacketType(5, 10) == VGREResult::SUCCESS,
                     "Valid packet type 5 should pass");
    VGRE_TEST_ASSERT(InputValidator::validatePacketType(10, 10) == VGREResult::SUCCESS,
                     "Valid packet type 10 (max) should pass");
    
    // Invalid packet types
    VGRE_TEST_ASSERT(InputValidator::validatePacketType(11, 10) == VGREResult::ERROR_INVALID_VALUE,
                     "Invalid packet type 11 should fail");
    VGRE_TEST_ASSERT(InputValidator::validatePacketType(100, 10) == VGREResult::ERROR_INVALID_VALUE,
                     "Invalid packet type 100 should fail");
    
    std::cout << "✓ Packet type validation passed" << std::endl;
    return true;
}

bool test_sequence_number_validation() {
    std::cout << "\nTest: Sequence Number Validation..." << std::endl;
    
    // Valid sequence numbers (within window)
    VGRE_TEST_ASSERT(InputValidator::validateSequenceNumber(100, 100, 10) == VGREResult::SUCCESS,
                     "Exact sequence number should pass");
    VGRE_TEST_ASSERT(InputValidator::validateSequenceNumber(105, 100, 10) == VGREResult::SUCCESS,
                     "Sequence number within window should pass");
    VGRE_TEST_ASSERT(InputValidator::validateSequenceNumber(110, 100, 10) == VGREResult::SUCCESS,
                     "Sequence number at window edge should pass");
    
    // Invalid sequence numbers (outside window)
    VGRE_TEST_ASSERT(InputValidator::validateSequenceNumber(111, 100, 10) == VGREResult::ERROR_INVALID_VALUE,
                     "Sequence number outside window should fail");
    VGRE_TEST_ASSERT(InputValidator::validateSequenceNumber(200, 100, 10) == VGREResult::ERROR_INVALID_VALUE,
                     "Sequence number far outside window should fail");
    
    // Test wraparound
    VGRE_TEST_ASSERT(InputValidator::validateSequenceNumber(5, UINT32_MAX - 5, 20) == VGREResult::SUCCESS,
                     "Sequence number wraparound should pass");
    
    std::cout << "✓ Sequence number validation passed" << std::endl;
    return true;
}

bool test_crc32_checksum() {
    std::cout << "\nTest: CRC32 Checksum..." << std::endl;
    
    // Test known CRC32 values
    const char* test_data = "Hello, World!";
    uint32_t crc = InputValidator::calculateCRC32(test_data, strlen(test_data));
    
    std::cout << "  CRC32 of 'Hello, World!': 0x" << std::hex << crc << std::dec << std::endl;
    
    // Verify checksum validation
    VGRE_TEST_ASSERT(InputValidator::validatePacketChecksum(test_data, strlen(test_data), crc) == VGREResult::SUCCESS,
                     "Valid checksum should pass");
    VGRE_TEST_ASSERT(InputValidator::validatePacketChecksum(test_data, strlen(test_data), crc + 1) == VGREResult::ERROR_INVALID_VALUE,
                     "Invalid checksum should fail");
    
    // Test empty data
    uint32_t empty_crc = InputValidator::calculateCRC32("", 0);
    VGRE_TEST_ASSERT(empty_crc == 0, "Empty data CRC32 should be 0");
    
    std::cout << "✓ CRC32 checksum passed" << std::endl;
    return true;
}

bool test_texture_coords_validation() {
    std::cout << "\nTest: Texture Coordinates Validation..." << std::endl;
    
    // Valid normalized coordinates
    VGRE_TEST_ASSERT(InputValidator::validateTextureCoords(0.5f, 0.5f, 0.5f, 256, 256, 1, true) == VGREResult::SUCCESS,
                     "Valid normalized coords should pass");
    VGRE_TEST_ASSERT(InputValidator::validateTextureCoords(0.0f, 0.0f, 0.0f, 256, 256, 1, true) == VGREResult::SUCCESS,
                     "Normalized coords at origin should pass");
    VGRE_TEST_ASSERT(InputValidator::validateTextureCoords(1.0f, 1.0f, 1.0f, 256, 256, 1, true) == VGREResult::SUCCESS,
                     "Normalized coords at max should pass");
    
    // Invalid normalized coordinates
    VGRE_TEST_ASSERT(InputValidator::validateTextureCoords(2.0f, 0.5f, 0.5f, 256, 256, 1, true) == VGREResult::ERROR_INVALID_VALUE,
                     "Normalized coords > 1.5 should fail");
    VGRE_TEST_ASSERT(InputValidator::validateTextureCoords(-1.0f, 0.5f, 0.5f, 256, 256, 1, true) == VGREResult::ERROR_INVALID_VALUE,
                     "Normalized coords < -0.5 should fail");
    
    // Valid non-normalized coordinates
    VGRE_TEST_ASSERT(InputValidator::validateTextureCoords(128.0f, 128.0f, 0.0f, 256, 256, 1, false) == VGREResult::SUCCESS,
                     "Valid non-normalized coords should pass");
    VGRE_TEST_ASSERT(InputValidator::validateTextureCoords(0.0f, 0.0f, 0.0f, 256, 256, 1, false) == VGREResult::SUCCESS,
                     "Non-normalized coords at origin should pass");
    
    // Invalid non-normalized coordinates
    VGRE_TEST_ASSERT(InputValidator::validateTextureCoords(300.0f, 128.0f, 0.0f, 256, 256, 1, false) == VGREResult::ERROR_INVALID_VALUE,
                     "Non-normalized coords out of bounds should fail");
    
    // Test NaN and infinity
    VGRE_TEST_ASSERT(InputValidator::validateTextureCoords(NAN, 0.5f, 0.5f, 256, 256, 1, true) == VGREResult::ERROR_INVALID_VALUE,
                     "NaN coords should fail");
    VGRE_TEST_ASSERT(InputValidator::validateTextureCoords(INFINITY, 0.5f, 0.5f, 256, 256, 1, true) == VGREResult::ERROR_INVALID_VALUE,
                     "Infinity coords should fail");
    
    std::cout << "✓ Texture coordinates validation passed" << std::endl;
    return true;
}

bool test_mipmap_level_validation() {
    std::cout << "\nTest: Mipmap Level Validation..." << std::endl;
    
    // Valid mipmap levels
    VGRE_TEST_ASSERT(InputValidator::validateMipmapLevel(0, 10) == VGREResult::SUCCESS,
                     "Mipmap level 0 should pass");
    VGRE_TEST_ASSERT(InputValidator::validateMipmapLevel(5, 10) == VGREResult::SUCCESS,
                     "Mipmap level 5 should pass");
    VGRE_TEST_ASSERT(InputValidator::validateMipmapLevel(10, 10) == VGREResult::SUCCESS,
                     "Mipmap level 10 (max) should pass");
    
    // Invalid mipmap levels
    VGRE_TEST_ASSERT(InputValidator::validateMipmapLevel(11, 10) == VGREResult::ERROR_INVALID_VALUE,
                     "Mipmap level 11 should fail");
    VGRE_TEST_ASSERT(InputValidator::validateMipmapLevel(100, 10) == VGREResult::ERROR_INVALID_VALUE,
                     "Mipmap level 100 should fail");
    
    std::cout << "✓ Mipmap level validation passed" << std::endl;
    return true;
}

bool test_surface_bounds_validation() {
    std::cout << "\nTest: Surface Bounds Validation..." << std::endl;
    
    // Valid surface coordinates
    VGRE_TEST_ASSERT(InputValidator::validateSurfaceBounds(0, 0, 256, 256) == VGREResult::SUCCESS,
                     "Surface coords at origin should pass");
    VGRE_TEST_ASSERT(InputValidator::validateSurfaceBounds(128, 128, 256, 256) == VGREResult::SUCCESS,
                     "Surface coords in middle should pass");
    VGRE_TEST_ASSERT(InputValidator::validateSurfaceBounds(255, 255, 256, 256) == VGREResult::SUCCESS,
                     "Surface coords at max should pass");
    
    // Invalid surface coordinates
    VGRE_TEST_ASSERT(InputValidator::validateSurfaceBounds(256, 128, 256, 256) == VGREResult::ERROR_INVALID_VALUE,
                     "Surface X out of bounds should fail");
    VGRE_TEST_ASSERT(InputValidator::validateSurfaceBounds(128, 256, 256, 256) == VGREResult::ERROR_INVALID_VALUE,
                     "Surface Y out of bounds should fail");
    VGRE_TEST_ASSERT(InputValidator::validateSurfaceBounds(300, 300, 256, 256) == VGREResult::ERROR_INVALID_VALUE,
                     "Surface coords far out of bounds should fail");
    
    std::cout << "✓ Surface bounds validation passed" << std::endl;
    return true;
}
