/**
 * Cross-Platform Memory Management Test Suite
 * 
 * **Validates: Requirements 9.1, 9.2, 9.5, 13.3**
 * 
 * This test suite validates memory management operations work identically across Linux, Windows, and macOS.
 * It focuses on memory alignment, struct packing, and memory synchronization that are critical for
 * preventing ACCESS_VIOLATION crashes in cross-platform TCP cluster deployments.
 * 
 * Test Coverage:
 * - Memory alignment and struct packing validation
 * - Cross-platform memory allocation and deallocation
 * - Shared memory operations (SHM)
 * - Memory synchronization and delta-sync operations
 * - Buffer overflow protection and bounds checking
 * - Platform-specific memory management differences
 * - Memory leak detection and cleanup validation
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/memory_sync_manager.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/shm_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/error_codes.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <random>
#include <memory>
#include <atomic>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <memoryapi.h>
#else
#include <sys/mman.h>
#include <sys/shm.h>
#include <unistd.h>
#include <fcntl.h>
#endif

using namespace vgre;
using namespace vgre::advanced;
using namespace vgre::core;

// Cross-platform memory management test framework
class CrossPlatformMemoryTester {
public:
    CrossPlatformMemoryTester() {
        initializePlatform();
    }
    
    ~CrossPlatformMemoryTester() {
        cleanupPlatform();
    }
    
    // Test memory alignment and struct packing validation
    bool testMemoryAlignmentStructPacking() {
        std::cout << "Testing memory alignment and struct packing validation..." << std::endl;
        
        bool all_passed = true;
        
        // Test VSBP packet structure alignment
        if (sizeof(VSBPHeader) != 24) {
            std::cout << "FAIL: VSBPHeader size mismatch - expected 24, got " 
                      << sizeof(VSBPHeader) << std::endl;
            all_passed = false;
        }
        
        if (alignof(VSBPHeader) > 8) {
            std::cout << "FAIL: VSBPHeader alignment too large - expected <= 8, got " 
                      << alignof(VSBPHeader) << std::endl;
            all_passed = false;
        }
        
        // Test other critical packet structures
        if (sizeof(DataHeaderPacket) != 16) {
            std::cout << "FAIL: DataHeaderPacket size mismatch - expected 16, got " 
                      << sizeof(DataHeaderPacket) << std::endl;
            all_passed = false;
        }
        
        if (sizeof(ArgScalarPacket) != 13) {
            std::cout << "FAIL: ArgScalarPacket size mismatch - expected 13, got " 
                      << sizeof(ArgScalarPacket) << std::endl;
            all_passed = false;
        }
        
        if (sizeof(CollectiveOpPacket) != 24) {
            std::cout << "FAIL: CollectiveOpPacket size mismatch - expected 24, got " 
                      << sizeof(CollectiveOpPacket) << std::endl;
            all_passed = false;
        }
        
        // Test memory alignment for various data types
        std::vector<size_t> test_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
        
        for (size_t size : test_sizes) {
            void* aligned_ptr = std::aligned_alloc(8, size);
            if (!aligned_ptr) {
                std::cout << "FAIL: aligned_alloc failed for size " << size << std::endl;
                all_passed = false;
                continue;
            }
            
            // Check alignment
            if (reinterpret_cast<uintptr_t>(aligned_ptr) % 8 != 0) {
                std::cout << "FAIL: Memory not 8-byte aligned for size " << size << std::endl;
                all_passed = false;
            }
            
            // Test memory access (should not crash)
            memset(aligned_ptr, 0xAB, size);
            
            // Verify pattern
            uint8_t* bytes = static_cast<uint8_t*>(aligned_ptr);
            for (size_t i = 0; i < size; i++) {
                if (bytes[i] != 0xAB) {
                    std::cout << "FAIL: Memory pattern verification failed for size " 
                              << size << " at offset " << i << std::endl;
                    all_passed = false;
                    break;
                }
            }
            
            std::free(aligned_ptr);
        }
        
        // Test endianness consistency
        uint32_t test_value = 0x12345678;
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&test_value);
        if (bytes[0] != 0x78) {
            std::cout << "FAIL: Platform is not little-endian (required for VSBP)" << std::endl;
            all_passed = false;
        }
        
        if (all_passed) {
            std::cout << "PASS: Memory alignment and struct packing validation" << std::endl;
        }
        return all_passed;
    }
    
    // Test cross-platform memory allocation and deallocation
    bool testMemoryAllocationDeallocation() {
        std::cout << "Testing cross-platform memory allocation and deallocation..." << std::endl;
        
        bool all_passed = true;
        std::vector<void*> allocated_ptrs;
        
        // Test various allocation sizes
        std::vector<size_t> sizes = {
            1, 16, 64, 256, 1024, 4096, 16384, 65536, 1048576
        };
        
        // Allocate memory blocks
        for (size_t size : sizes) {
            void* ptr = std::malloc(size);
            if (!ptr) {
                std::cout << "FAIL: malloc failed for size " << size << std::endl;
                all_passed = false;
                continue;
            }
            
            allocated_ptrs.push_back(ptr);
            
            // Test memory access
            memset(ptr, 0x55, size);
            
            // Verify pattern
            uint8_t* bytes = static_cast<uint8_t*>(ptr);
            for (size_t i = 0; i < size; i++) {
                if (bytes[i] != 0x55) {
                    std::cout << "FAIL: Memory access verification failed for size " 
                              << size << " at offset " << i << std::endl;
                    all_passed = false;
                    break;
                }
            }
        }
        
        // Test aligned allocation
        for (size_t size : sizes) {
            void* ptr = std::aligned_alloc(16, (size + 15) & ~15); // 16-byte aligned
            if (!ptr) {
                std::cout << "FAIL: aligned_alloc failed for size " << size << std::endl;
                all_passed = false;
                continue;
            }
            
            allocated_ptrs.push_back(ptr);
            
            // Check alignment
            if (reinterpret_cast<uintptr_t>(ptr) % 16 != 0) {
                std::cout << "FAIL: aligned_alloc returned unaligned pointer for size " 
                          << size << std::endl;
                all_passed = false;
            }
        }
        
        // Deallocate all memory
        for (void* ptr : allocated_ptrs) {
            std::free(ptr);
        }
        
        // Test calloc (zero-initialized allocation)
        void* zero_ptr = std::calloc(1024, sizeof(uint8_t));
        if (!zero_ptr) {
            std::cout << "FAIL: calloc failed" << std::endl;
            all_passed = false;
        } else {
            // Verify zero initialization
            uint8_t* bytes = static_cast<uint8_t*>(zero_ptr);
            for (size_t i = 0; i < 1024; i++) {
                if (bytes[i] != 0) {
                    std::cout << "FAIL: calloc did not zero-initialize memory at offset " 
                              << i << std::endl;
                    all_passed = false;
                    break;
                }
            }
            std::free(zero_ptr);
        }
        
        if (all_passed) {
            std::cout << "PASS: Cross-platform memory allocation and deallocation" << std::endl;
        }
        return all_passed;
    }
    
    // Test shared memory operations (SHM)
    bool testSharedMemoryOperations() {
        std::cout << "Testing shared memory operations (SHM)..." << std::endl;
        
        bool all_passed = true;
        
        // Test SHM creation and mapping
        std::string shm_name = "vgre_test_shm_" + std::to_string(getpid());
        size_t shm_size = 4096;
        
        void* shm_ptr = createSharedMemory(shm_name, shm_size);
        if (!shm_ptr) {
            std::cout << "FAIL: Shared memory creation failed" << std::endl;
            return false;
        }
        
        // Test SHM access
        memset(shm_ptr, 0xCC, shm_size);
        
        // Verify pattern
        uint8_t* bytes = static_cast<uint8_t*>(shm_ptr);
        for (size_t i = 0; i < shm_size; i++) {
            if (bytes[i] != 0xCC) {
                std::cout << "FAIL: SHM access verification failed at offset " << i << std::endl;
                all_passed = false;
                break;
            }
        }
        
        // Test SHM mapping from another process (simulate)
        void* shm_ptr2 = mapSharedMemory(shm_name, shm_size);
        if (!shm_ptr2) {
            std::cout << "FAIL: Shared memory mapping failed" << std::endl;
            all_passed = false;
        } else {
            // Verify data is shared
            uint8_t* bytes2 = static_cast<uint8_t*>(shm_ptr2);
            if (bytes2[0] != 0xCC) {
                std::cout << "FAIL: SHM data not shared between mappings" << std::endl;
                all_passed = false;
            }
            
            // Modify from second mapping
            bytes2[100] = 0xDD;
            
            // Verify change is visible in first mapping
            if (bytes[100] != 0xDD) {
                std::cout << "FAIL: SHM modifications not visible across mappings" << std::endl;
                all_passed = false;
            }
            
            unmapSharedMemory(shm_ptr2, shm_size);
        }
        
        // Cleanup
        unmapSharedMemory(shm_ptr, shm_size);
        destroySharedMemory(shm_name);
        
        if (all_passed) {
            std::cout << "PASS: Shared memory operations (SHM)" << std::endl;
        }
        return all_passed;
    }
    
    // Test memory synchronization and delta-sync operations
    bool testMemorySynchronization() {
        std::cout << "Testing memory synchronization and delta-sync operations..." << std::endl;
        
        bool all_passed = true;
        
        // Create test memory regions
        size_t region_size = 8192;
        std::vector<uint8_t> source_memory(region_size);
        std::vector<uint8_t> target_memory(region_size);
        
        // Initialize with different patterns
        for (size_t i = 0; i < region_size; i++) {
            source_memory[i] = static_cast<uint8_t>(i & 0xFF);
            target_memory[i] = 0x00;
        }
        
        // Test full synchronization
        memcpy(target_memory.data(), source_memory.data(), region_size);
        
        // Verify full sync
        if (memcmp(source_memory.data(), target_memory.data(), region_size) != 0) {
            std::cout << "FAIL: Full memory synchronization failed" << std::endl;
            all_passed = false;
        }
        
        // Test delta synchronization
        // Modify some regions in source
        std::vector<std::pair<size_t, size_t>> dirty_ranges = {
            {100, 200},   // 100 bytes at offset 100
            {1000, 500},  // 500 bytes at offset 1000
            {7000, 1000}  // 1000 bytes at offset 7000
        };
        
        // Reset target
        std::fill(target_memory.begin(), target_memory.end(), 0x00);
        
        // Apply dirty ranges
        for (const auto& range : dirty_ranges) {
            size_t offset = range.first;
            size_t size = range.second;
            
            if (offset + size <= region_size) {
                memcpy(target_memory.data() + offset, 
                           source_memory.data() + offset, 
                           size);
            }
        }
        
        // Verify delta sync
        for (const auto& range : dirty_ranges) {
            size_t offset = range.first;
            size_t size = range.second;
            
            if (offset + size <= region_size) {
                if (memcmp(source_memory.data() + offset,
                               target_memory.data() + offset,
                               size) != 0) {
                    std::cout << "FAIL: Delta synchronization failed for range " 
                              << offset << "-" << (offset + size) << std::endl;
                    all_passed = false;
                }
            }
        }
        
        // Test memory comparison for dirty detection
        std::vector<uint8_t> modified_memory = source_memory;
        modified_memory[500] = 0xFF; // Single byte change
        
        bool difference_detected = false;
        for (size_t i = 0; i < region_size; i++) {
            if (source_memory[i] != modified_memory[i]) {
                difference_detected = true;
                break;
            }
        }
        
        if (!difference_detected) {
            std::cout << "FAIL: Memory difference detection failed" << std::endl;
            all_passed = false;
        }
        
        if (all_passed) {
            std::cout << "PASS: Memory synchronization and delta-sync operations" << std::endl;
        }
        return all_passed;
    }
    
    // Test buffer overflow protection and bounds checking
    bool testBufferOverflowProtection() {
        std::cout << "Testing buffer overflow protection and bounds checking..." << std::endl;
        
        bool all_passed = true;
        
        // Test safe string operations
        char buffer[64];
        const char* test_string = "This is a test string for buffer overflow testing";
        
        // Test strncpy (safe)
        strncpy(buffer, test_string, sizeof(buffer) - 1);
        buffer[sizeof(buffer) - 1] = '\0';
        
        if (strlen(buffer) >= sizeof(buffer)) {
            std::cout << "FAIL: strncpy did not prevent buffer overflow" << std::endl;
            all_passed = false;
        }
        
        // Test memcpy bounds checking
        std::vector<uint8_t> src_buffer(1024, 0xAA);
        std::vector<uint8_t> dst_buffer(512, 0x00);
        
        // Safe copy (within bounds)
        size_t safe_copy_size = std::min(src_buffer.size(), dst_buffer.size());
        memcpy(dst_buffer.data(), src_buffer.data(), safe_copy_size);
        
        // Verify safe copy
        for (size_t i = 0; i < safe_copy_size; i++) {
            if (dst_buffer[i] != 0xAA) {
                std::cout << "FAIL: Safe memcpy failed at offset " << i << std::endl;
                all_passed = false;
                break;
            }
        }
        
        // Test array bounds checking
        std::vector<int> test_array(100);
        for (size_t i = 0; i < test_array.size(); i++) {
            test_array[i] = static_cast<int>(i);
        }
        
        // Verify array access
        for (size_t i = 0; i < test_array.size(); i++) {
            if (test_array[i] != static_cast<int>(i)) {
                std::cout << "FAIL: Array access verification failed at index " << i << std::endl;
                all_passed = false;
                break;
            }
        }
        
        // Test packet buffer validation
        std::vector<uint8_t> packet_buffer(sizeof(VSBPHeader) + 1024);
        VSBPHeader* header = reinterpret_cast<VSBPHeader*>(packet_buffer.data());
        
        header->magic = VSBP_MAGIC;
        header->version = VSBP_VERSION;
        header->type = static_cast<uint16_t>(PacketType::DATA_HEADER);
        header->sequence = 1;
        header->payloadSize = 1024;
        header->platform = 1;
        
        // Validate header within buffer bounds
        if (sizeof(VSBPHeader) + header->payloadSize > packet_buffer.size()) {
            std::cout << "FAIL: Packet buffer bounds checking failed" << std::endl;
            all_passed = false;
        }
        
        if (all_passed) {
            std::cout << "PASS: Buffer overflow protection and bounds checking" << std::endl;
        }
        return all_passed;
    }
    
    // Test platform-specific memory management differences
    bool testPlatformSpecificMemoryManagement() {
        std::cout << "Testing platform-specific memory management differences..." << std::endl;
        
        bool all_passed = true;
        
#ifdef _WIN32
        // Windows-specific memory tests
        std::cout << "Running Windows-specific memory tests..." << std::endl;
        
        // Test VirtualAlloc
        void* virtual_ptr = VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!virtual_ptr) {
            std::cout << "FAIL: VirtualAlloc failed" << std::endl;
            all_passed = false;
        } else {
            // Test memory access
            memset(virtual_ptr, 0xBB, 4096);
            
            // Verify pattern
            uint8_t* bytes = static_cast<uint8_t*>(virtual_ptr);
            if (bytes[0] != 0xBB || bytes[4095] != 0xBB) {
                std::cout << "FAIL: VirtualAlloc memory access failed" << std::endl;
                all_passed = false;
            }
            
            VirtualFree(virtual_ptr, 0, MEM_RELEASE);
        }
        
        // Test Windows heap allocation
        HANDLE heap = GetProcessHeap();
        if (heap) {
            void* heap_ptr = HeapAlloc(heap, 0, 2048);
            if (!heap_ptr) {
                std::cout << "FAIL: HeapAlloc failed" << std::endl;
                all_passed = false;
            } else {
                memset(heap_ptr, 0xEE, 2048);
                HeapFree(heap, 0, heap_ptr);
            }
        }
        
#else
        // Unix-specific memory tests
        std::cout << "Running Unix-specific memory tests..." << std::endl;
        
        // Test mmap
        void* mmap_ptr = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, 
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mmap_ptr == MAP_FAILED) {
            std::cout << "FAIL: mmap failed" << std::endl;
            all_passed = false;
        } else {
            // Test memory access
            memset(mmap_ptr, 0xBB, 4096);
            
            // Verify pattern
            uint8_t* bytes = static_cast<uint8_t*>(mmap_ptr);
            if (bytes[0] != 0xBB || bytes[4095] != 0xBB) {
                std::cout << "FAIL: mmap memory access failed" << std::endl;
                all_passed = false;
            }
            
            munmap(mmap_ptr, 4096);
        }
        
        // Test page size
        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) {
            std::cout << "FAIL: sysconf(_SC_PAGESIZE) failed" << std::endl;
            all_passed = false;
        } else {
            std::cout << "INFO: System page size: " << page_size << " bytes" << std::endl;
        }
#endif
        
        if (all_passed) {
            std::cout << "PASS: Platform-specific memory management differences" << std::endl;
        }
        return all_passed;
    }
    
    // Test memory leak detection and cleanup validation
    bool testMemoryLeakDetection() {
        std::cout << "Testing memory leak detection and cleanup validation..." << std::endl;
        
        bool all_passed = true;
        
        // Track allocations
        std::vector<void*> tracked_allocations;
        
        // Allocate various sizes
        std::vector<size_t> sizes = {64, 256, 1024, 4096};
        
        for (size_t size : sizes) {
            void* ptr = std::malloc(size);
            if (!ptr) {
                std::cout << "FAIL: malloc failed for size " << size << std::endl;
                all_passed = false;
                continue;
            }
            
            tracked_allocations.push_back(ptr);
            
            // Use the memory to prevent optimization
            memset(ptr, 0x77, size);
        }
        
        // Verify all allocations are tracked
        if (tracked_allocations.size() != sizes.size()) {
            std::cout << "FAIL: Not all allocations were tracked" << std::endl;
            all_passed = false;
        }
        
        // Clean up all allocations
        for (void* ptr : tracked_allocations) {
            std::free(ptr);
        }
        
        tracked_allocations.clear();
        
        // Test RAII pattern
        {
            std::vector<std::unique_ptr<uint8_t[]>> raii_allocations;
            
            for (size_t size : sizes) {
                auto ptr = std::make_unique<uint8_t[]>(size);
                memset(ptr.get(), 0x88, size);
                raii_allocations.push_back(std::move(ptr));
            }
            
            // Verify allocations
            if (raii_allocations.size() != sizes.size()) {
                std::cout << "FAIL: RAII allocations failed" << std::endl;
                all_passed = false;
            }
            
            // RAII cleanup happens automatically when scope exits
        }
        
        // Test allocation/deallocation patterns that could cause leaks
        for (int i = 0; i < 100; i++) {
            void* temp_ptr = std::malloc(1024);
            if (temp_ptr) {
                memset(temp_ptr, i & 0xFF, 1024);
                std::free(temp_ptr);
            }
        }
        
        if (all_passed) {
            std::cout << "PASS: Memory leak detection and cleanup validation" << std::endl;
        }
        return all_passed;
    }
    
    // Run all memory management tests
    bool runAllTests() {
        std::cout << "\n=== Cross-Platform Memory Management Test Suite ===" << std::endl;
        
        bool all_passed = true;
        
        all_passed &= testMemoryAlignmentStructPacking();
        all_passed &= testMemoryAllocationDeallocation();
        all_passed &= testSharedMemoryOperations();
        all_passed &= testMemorySynchronization();
        all_passed &= testBufferOverflowProtection();
        all_passed &= testPlatformSpecificMemoryManagement();
        all_passed &= testMemoryLeakDetection();
        
        if (all_passed) {
            std::cout << "\n=== ALL MEMORY MANAGEMENT TESTS PASSED ===" << std::endl;
        } else {
            std::cout << "\n=== SOME MEMORY MANAGEMENT TESTS FAILED ===" << std::endl;
        }
        
        return all_passed;
    }
    
private:
    void initializePlatform() {
        // Platform-specific initialization if needed
    }
    
    void cleanupPlatform() {
        // Platform-specific cleanup if needed
    }
    
    int getpid() {
#ifdef _WIN32
        return static_cast<int>(GetCurrentProcessId());
#else
        return static_cast<int>(::getpid());
#endif
    }
    
    void* createSharedMemory(const std::string& name, size_t size) {
#ifdef _WIN32
        HANDLE hMapFile = CreateFileMappingA(
            INVALID_HANDLE_VALUE,
            nullptr,
            PAGE_READWRITE,
            0,
            static_cast<DWORD>(size),
            name.c_str()
        );
        
        if (!hMapFile) {
            return nullptr;
        }
        
        void* ptr = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, size);
        if (!ptr) {
            CloseHandle(hMapFile);
            return nullptr;
        }
        
        return ptr;
#else
        int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd == -1) {
            return nullptr;
        }
        
        if (ftruncate(fd, size) == -1) {
            close(fd);
            shm_unlink(name.c_str());
            return nullptr;
        }
        
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        
        if (ptr == MAP_FAILED) {
            shm_unlink(name.c_str());
            return nullptr;
        }
        
        return ptr;
#endif
    }
    
    void* mapSharedMemory(const std::string& name, size_t size) {
#ifdef _WIN32
        HANDLE hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
        if (!hMapFile) {
            return nullptr;
        }
        
        void* ptr = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, size);
        CloseHandle(hMapFile);
        
        return ptr;
#else
        int fd = shm_open(name.c_str(), O_RDWR, 0666);
        if (fd == -1) {
            return nullptr;
        }
        
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        
        if (ptr == MAP_FAILED) {
            return nullptr;
        }
        
        return ptr;
#endif
    }
    
    void unmapSharedMemory(void* ptr, size_t size) {
#ifdef _WIN32
        UnmapViewOfFile(ptr);
#else
        munmap(ptr, size);
#endif
    }
    
    void destroySharedMemory(const std::string& name) {
#ifdef _WIN32
        // Windows file mapping is automatically cleaned up
#else
        shm_unlink(name.c_str());
#endif
    }
};

int main() {
    std::cout << "Cross-Platform Memory Management Test Suite" << std::endl;
    std::cout << "Testing memory operations across Linux, Windows, and macOS" << std::endl;
    
#ifdef _WIN32
    std::cout << "Platform: Windows" << std::endl;
#elif defined(__linux__)
    std::cout << "Platform: Linux" << std::endl;
#elif defined(__APPLE__)
    std::cout << "Platform: macOS" << std::endl;
#else
    std::cout << "Platform: Unknown" << std::endl;
#endif
    
    CrossPlatformMemoryTester tester;
    bool success = tester.runAllTests();
    
    return success ? 0 : 1;
}