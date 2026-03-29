#include "vgre/common/input_validation.h"
#include "vgre/common/logger.h"
#include <cstring>
#include <limits>
#include <cmath>

namespace vgre {
namespace common {

VGREResult InputValidator::validateAllocationSize(size_t size) {
    if (size == 0) {
        VGRE_LOG_WARN("InputValidator", "Allocation size is zero");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    if (size > MAX_ALLOCATION_SIZE) {
        VGRE_LOG_ERROR("InputValidator", 
                      "Allocation size " + std::to_string(size) + 
                      " exceeds maximum " + std::to_string(MAX_ALLOCATION_SIZE));
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    return VGREResult::SUCCESS;
}

VGREResult InputValidator::validateMemcpyParams(const void* dst, const void* src, size_t count) {
    if (!dst) {
        VGRE_LOG_ERROR("InputValidator", "Destination pointer is NULL");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    if (!src) {
        VGRE_LOG_ERROR("InputValidator", "Source pointer is NULL");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    if (count == 0) {
        VGRE_LOG_WARN("InputValidator", "Copy count is zero");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    if (count > MAX_ALLOCATION_SIZE) {
        VGRE_LOG_ERROR("InputValidator", 
                      "Copy count " + std::to_string(count) + 
                      " exceeds maximum " + std::to_string(MAX_ALLOCATION_SIZE));
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    // Check for pointer overlap (undefined behavior)
    uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);
    uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
    
    if (dst_addr < src_addr) {
        if (dst_addr + count > src_addr) {
            VGRE_LOG_ERROR("InputValidator", "Overlapping memory regions detected");
            return VGREResult::ERROR_INVALID_VALUE;
        }
    } else if (src_addr < dst_addr) {
        if (src_addr + count > dst_addr) {
            VGRE_LOG_ERROR("InputValidator", "Overlapping memory regions detected");
            return VGREResult::ERROR_INVALID_VALUE;
        }
    }
    
    return VGREResult::SUCCESS;
}

VGREResult InputValidator::validateMemoryOffset(size_t base, size_t offset, size_t count) {
    size_t total;
    
    // Check offset + count doesn't overflow
    if (!safeAdd(offset, count, total)) {
        VGRE_LOG_ERROR("InputValidator", "Memory offset + count overflow");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    // Check total doesn't exceed base
    if (total > base) {
        VGRE_LOG_ERROR("InputValidator", 
                      "Memory access out of bounds: offset=" + std::to_string(offset) +
                      " count=" + std::to_string(count) + 
                      " base=" + std::to_string(base));
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    return VGREResult::SUCCESS;
}

VGREResult InputValidator::validatePacketSize(size_t size) {
    if (size == 0) {
        VGRE_LOG_WARN("InputValidator", "Packet size is zero");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    if (size > MAX_PACKET_SIZE) {
        VGRE_LOG_ERROR("InputValidator", 
                      "Packet size " + std::to_string(size) + 
                      " exceeds maximum " + std::to_string(MAX_PACKET_SIZE));
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    return VGREResult::SUCCESS;
}

VGREResult InputValidator::validateStringLength(const char* str, size_t maxLen) {
    if (!str) {
        VGRE_LOG_ERROR("InputValidator", "String pointer is NULL");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    // Use strnlen to safely check length
    size_t len = strnlen(str, maxLen + 1);
    
    if (len > maxLen) {
        VGRE_LOG_ERROR("InputValidator", 
                      "String length exceeds maximum " + std::to_string(maxLen));
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    return VGREResult::SUCCESS;
}

VGREResult InputValidator::validateGridDim(uint32_t x, uint32_t y, uint32_t z) {
    if (x == 0 || y == 0 || z == 0) {
        VGRE_LOG_ERROR("InputValidator", "Grid dimension cannot be zero");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    if (x > MAX_GRID_DIM || y > MAX_GRID_DIM || z > MAX_GRID_DIM) {
        VGRE_LOG_ERROR("InputValidator", 
                      "Grid dimension exceeds maximum " + std::to_string(MAX_GRID_DIM) +
                      " (got " + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    return VGREResult::SUCCESS;
}

VGREResult InputValidator::validateBlockDim(uint32_t x, uint32_t y, uint32_t z) {
    if (x == 0 || y == 0 || z == 0) {
        VGRE_LOG_ERROR("InputValidator", "Block dimension cannot be zero");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    if (x > MAX_BLOCK_DIM || y > MAX_BLOCK_DIM || z > MAX_BLOCK_DIM) {
        VGRE_LOG_ERROR("InputValidator", 
                      "Block dimension exceeds maximum " + std::to_string(MAX_BLOCK_DIM) +
                      " (got " + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    // Check total threads per block
    size_t total;
    if (!safeMul(x, y, total) || !safeMul(total, z, total)) {
        VGRE_LOG_ERROR("InputValidator", "Block dimension multiplication overflow");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    if (total > MAX_BLOCK_DIM) {
        VGRE_LOG_ERROR("InputValidator", 
                      "Total threads per block " + std::to_string(total) + 
                      " exceeds maximum " + std::to_string(MAX_BLOCK_DIM));
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    return VGREResult::SUCCESS;
}

VGREResult InputValidator::validatePointer(const void* ptr) {
    if (!ptr) {
        VGRE_LOG_ERROR("InputValidator", "Pointer is NULL");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    return VGREResult::SUCCESS;
}

VGREResult InputValidator::validateDeviceId(int deviceId, int maxDevices) {
    if (deviceId < 0) {
        VGRE_LOG_ERROR("InputValidator", "Device ID is negative: " + std::to_string(deviceId));
        return VGREResult::ERROR_INVALID_DEVICE;
    }
    
    if (deviceId >= maxDevices) {
        VGRE_LOG_ERROR("InputValidator", 
                      "Device ID " + std::to_string(deviceId) + 
                      " exceeds maximum " + std::to_string(maxDevices - 1));
        return VGREResult::ERROR_INVALID_DEVICE;
    }
    
    return VGREResult::SUCCESS;
}

bool InputValidator::safeMul(size_t a, size_t b, size_t& result) {
    if (a == 0 || b == 0) {
        result = 0;
        return true;
    }
    
    // Check for overflow: if a * b > SIZE_MAX, then a > SIZE_MAX / b
    if (a > std::numeric_limits<size_t>::max() / b) {
        return false;
    }
    
    result = a * b;
    return true;
}

bool InputValidator::safeAdd(size_t a, size_t b, size_t& result) {
    // Check for overflow: if a + b > SIZE_MAX, then a > SIZE_MAX - b
    if (a > std::numeric_limits<size_t>::max() - b) {
        return false;
    }
    
    result = a + b;
    return true;
}

// ========== ADVANCED VALIDATION IMPLEMENTATIONS ==========

VGREResult InputValidator::validatePacketType(uint16_t type, uint16_t max_type) {
    if (type > max_type) {
        VGRE_LOG_ERROR("InputValidator", 
                      "Invalid packet type " + std::to_string(type) + 
                      " (max: " + std::to_string(max_type) + ")");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    return VGREResult::SUCCESS;
}

VGREResult InputValidator::validateSequenceNumber(uint32_t seq, uint32_t expected, uint32_t window_size) {
    // Allow sequence numbers within a window (handles reordering and wraparound)
    uint32_t diff;
    
    if (seq >= expected) {
        diff = seq - expected;
    } else {
        // Handle wraparound
        diff = (UINT32_MAX - expected) + seq + 1;
    }
    
    if (diff > window_size) {
        VGRE_LOG_WARN("InputValidator", 
                     "Sequence number " + std::to_string(seq) + 
                     " outside expected window (expected: " + std::to_string(expected) + 
                     ", window: " + std::to_string(window_size) + ")");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    return VGREResult::SUCCESS;
}

uint32_t InputValidator::calculateCRC32(const void* data, size_t size) {
    // CRC32 polynomial (IEEE 802.3)
    static const uint32_t polynomial = 0xEDB88320;
    
    // Build CRC table (cached)
    static uint32_t crc_table[256];
    static bool table_initialized = false;
    
    if (!table_initialized) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t crc = i;
            for (int j = 0; j < 8; ++j) {
                if (crc & 1) {
                    crc = (crc >> 1) ^ polynomial;
                } else {
                    crc >>= 1;
                }
            }
            crc_table[i] = crc;
        }
        table_initialized = true;
    }
    
    // Calculate CRC32
    uint32_t crc = 0xFFFFFFFF;
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    
    for (size_t i = 0; i < size; ++i) {
        uint8_t index = (crc ^ bytes[i]) & 0xFF;
        crc = (crc >> 8) ^ crc_table[index];
    }
    
    return ~crc;
}

VGREResult InputValidator::validatePacketChecksum(const void* data, size_t size, uint32_t expected_checksum) {
    uint32_t calculated = calculateCRC32(data, size);
    
    if (calculated != expected_checksum) {
        VGRE_LOG_ERROR("InputValidator", 
                      "Checksum mismatch: expected " + std::to_string(expected_checksum) + 
                      ", got " + std::to_string(calculated));
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    return VGREResult::SUCCESS;
}

VGREResult InputValidator::validateTextureCoords(float x, float y, float z,
                                                 uint32_t width, uint32_t height, uint32_t depth,
                                                 bool normalized) {
    // Check for NaN or infinity
    if (std::isnan(x) || std::isnan(y) || std::isnan(z) ||
        std::isinf(x) || std::isinf(y) || std::isinf(z)) {
        VGRE_LOG_ERROR("InputValidator", "Texture coordinates contain NaN or infinity");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    if (normalized) {
        // Normalized coordinates should be in [0, 1] range
        // Allow slight overflow for filtering (up to 1.5)
        if (x < -0.5f || x > 1.5f || y < -0.5f || y > 1.5f || z < -0.5f || z > 1.5f) {
            VGRE_LOG_WARN("InputValidator", 
                         "Normalized texture coordinates out of range: (" + 
                         std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")");
            return VGREResult::ERROR_INVALID_VALUE;
        }
    } else {
        // Non-normalized coordinates should be within texture dimensions
        // Allow slight overflow for filtering
        if (x < -1.0f || x > static_cast<float>(width) + 1.0f ||
            y < -1.0f || y > static_cast<float>(height) + 1.0f ||
            z < -1.0f || z > static_cast<float>(depth) + 1.0f) {
            VGRE_LOG_WARN("InputValidator", 
                         "Texture coordinates out of bounds: (" + 
                         std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + 
                         ") for texture size (" + std::to_string(width) + ", " + 
                         std::to_string(height) + ", " + std::to_string(depth) + ")");
            return VGREResult::ERROR_INVALID_VALUE;
        }
    }
    
    return VGREResult::SUCCESS;
}

VGREResult InputValidator::validateMipmapLevel(uint32_t level, uint32_t max_level) {
    if (level > max_level) {
        VGRE_LOG_ERROR("InputValidator", 
                      "Mipmap level " + std::to_string(level) + 
                      " exceeds maximum " + std::to_string(max_level));
        return VGREResult::ERROR_INVALID_VALUE;
    }
    return VGREResult::SUCCESS;
}

VGREResult InputValidator::validateSurfaceBounds(uint32_t x, uint32_t y,
                                                uint32_t width, uint32_t height) {
    if (x >= width) {
        VGRE_LOG_ERROR("InputValidator", 
                      "Surface X coordinate " + std::to_string(x) + 
                      " out of bounds (width: " + std::to_string(width) + ")");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    if (y >= height) {
        VGRE_LOG_ERROR("InputValidator", 
                      "Surface Y coordinate " + std::to_string(y) + 
                      " out of bounds (height: " + std::to_string(height) + ")");
        return VGREResult::ERROR_INVALID_VALUE;
    }
    
    return VGREResult::SUCCESS;
}

} // namespace common
} // namespace vgre

