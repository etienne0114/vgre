#ifndef VGRE_COMMON_INPUT_VALIDATION_H
#define VGRE_COMMON_INPUT_VALIDATION_H

#include "vgre/common/error_codes.h"
#include <cstddef>
#include <cstdint>
#include <string>

namespace vgre {
namespace common {

/**
 * @brief Input validation utilities for security hardening
 * 
 * Provides comprehensive validation for:
 * - Memory sizes and offsets
 * - Pointer validity
 * - String lengths
 * - Packet sizes
 * - Parameter ranges
 */
class InputValidator {
public:
    // Maximum safe allocation size (16 GB)
    static constexpr size_t MAX_ALLOCATION_SIZE = 16ULL * 1024 * 1024 * 1024;
    
    // Maximum packet size (256 MB)
    static constexpr size_t MAX_PACKET_SIZE = 256ULL * 1024 * 1024;
    
    // Maximum string length (1 MB)
    static constexpr size_t MAX_STRING_LENGTH = 1024 * 1024;
    
    // Maximum grid dimension
    static constexpr uint32_t MAX_GRID_DIM = 65535;
    
    // Maximum block dimension
    static constexpr uint32_t MAX_BLOCK_DIM = 1024;
    
    /**
     * @brief Validate memory allocation size
     * @param size Size to allocate
     * @return SUCCESS if valid, ERROR_INVALID_VALUE if too large
     */
    static VGREResult validateAllocationSize(size_t size);
    
    /**
     * @brief Validate memory copy parameters
     * @param dst Destination pointer
     * @param src Source pointer
     * @param count Number of bytes to copy
     * @return SUCCESS if valid, ERROR_INVALID_VALUE if invalid
     */
    static VGREResult validateMemcpyParams(const void* dst, const void* src, size_t count);
    
    /**
     * @brief Validate memory offset doesn't overflow
     * @param base Base size
     * @param offset Offset to add
     * @param count Number of bytes
     * @return SUCCESS if valid, ERROR_INVALID_VALUE if overflow
     */
    static VGREResult validateMemoryOffset(size_t base, size_t offset, size_t count);
    
    /**
     * @brief Validate packet size
     * @param size Packet size
     * @return SUCCESS if valid, ERROR_INVALID_VALUE if too large
     */
    static VGREResult validatePacketSize(size_t size);
    
    /**
     * @brief Validate string length
     * @param str String pointer
     * @param maxLen Maximum allowed length
     * @return SUCCESS if valid, ERROR_INVALID_VALUE if too long or NULL
     */
    static VGREResult validateStringLength(const char* str, size_t maxLen = MAX_STRING_LENGTH);
    
    /**
     * @brief Validate grid dimensions
     * @param x Grid X dimension
     * @param y Grid Y dimension
     * @param z Grid Z dimension
     * @return SUCCESS if valid, ERROR_INVALID_VALUE if out of range
     */
    static VGREResult validateGridDim(uint32_t x, uint32_t y, uint32_t z);
    
    /**
     * @brief Validate block dimensions
     * @param x Block X dimension
     * @param y Block Y dimension
     * @param z Block Z dimension
     * @return SUCCESS if valid, ERROR_INVALID_VALUE if out of range
     */
    static VGREResult validateBlockDim(uint32_t x, uint32_t y, uint32_t z);
    
    /**
     * @brief Validate pointer is not NULL
     * @param ptr Pointer to validate
     * @return SUCCESS if valid, ERROR_INVALID_VALUE if NULL
     */
    static VGREResult validatePointer(const void* ptr);
    
    /**
     * @brief Validate device ID
     * @param deviceId Device ID
     * @param maxDevices Maximum number of devices
     * @return SUCCESS if valid, ERROR_INVALID_DEVICE if out of range
     */
    static VGREResult validateDeviceId(int deviceId, int maxDevices);
    
    /**
     * @brief Check for integer overflow in multiplication
     * @param a First operand
     * @param b Second operand
     * @param result Output result
     * @return true if no overflow, false if overflow
     */
    static bool safeMul(size_t a, size_t b, size_t& result);
    
    /**
     * @brief Check for integer overflow in addition
     * @param a First operand
     * @param b Second operand
     * @param result Output result
     * @return true if no overflow, false if overflow
     */
    static bool safeAdd(size_t a, size_t b, size_t& result);
    
    // ========== ADVANCED VALIDATION (Production-Ready) ==========
    
    /**
     * @brief Validate packet type is within valid range
     * @param type Packet type value
     * @param max_type Maximum valid packet type
     * @return SUCCESS if valid, ERROR_INVALID_VALUE if out of range
     */
    static VGREResult validatePacketType(uint16_t type, uint16_t max_type);
    
    /**
     * @brief Validate sequence number is within expected window
     * @param seq Received sequence number
     * @param expected Expected sequence number
     * @param window_size Size of acceptable window
     * @return SUCCESS if valid, ERROR_INVALID_VALUE if out of window
     */
    static VGREResult validateSequenceNumber(uint32_t seq, uint32_t expected, uint32_t window_size = 100);
    
    /**
     * @brief Validate packet checksum
     * @param data Packet data
     * @param size Data size
     * @param expected_checksum Expected CRC32 checksum
     * @return SUCCESS if valid, ERROR_INVALID_VALUE if mismatch
     */
    static VGREResult validatePacketChecksum(const void* data, size_t size, uint32_t expected_checksum);
    
    /**
     * @brief Validate texture coordinates are within bounds
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     * @param width Texture width
     * @param height Texture height
     * @param depth Texture depth
     * @param normalized Whether coordinates are normalized [0,1]
     * @return SUCCESS if valid, ERROR_INVALID_VALUE if out of bounds
     */
    static VGREResult validateTextureCoords(float x, float y, float z,
                                           uint32_t width, uint32_t height, uint32_t depth,
                                           bool normalized);
    
    /**
     * @brief Validate mipmap level
     * @param level Requested mipmap level
     * @param max_level Maximum available level
     * @return SUCCESS if valid, ERROR_INVALID_VALUE if out of range
     */
    static VGREResult validateMipmapLevel(uint32_t level, uint32_t max_level);
    
    /**
     * @brief Validate surface read/write bounds
     * @param x X coordinate
     * @param y Y coordinate
     * @param width Surface width
     * @param height Surface height
     * @return SUCCESS if valid, ERROR_INVALID_VALUE if out of bounds
     */
    static VGREResult validateSurfaceBounds(uint32_t x, uint32_t y,
                                           uint32_t width, uint32_t height);
    
    /**
     * @brief Calculate CRC32 checksum
     * @param data Data to checksum
     * @param size Data size
     * @return CRC32 checksum value
     */
    static uint32_t calculateCRC32(const void* data, size_t size);
};

} // namespace common
} // namespace vgre

#endif // VGRE_COMMON_INPUT_VALIDATION_H
