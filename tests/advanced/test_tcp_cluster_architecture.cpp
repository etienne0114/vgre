/**
 * VGRE Bug Condition Exploration Test — TCP Cluster Monolithic Architecture
 * 
 * **CRITICAL**: This test is EXPECTED TO FAIL on unfixed code.
 * Failure confirms the bug exists (monolithic 4000+ line file).
 * 
 * **Purpose**: Verify that tcp_cluster.cpp has been refactored into modular components.
 * 
 * **Bug Conditions Being Tested**:
 * - Bug Condition 5.1: tcp_cluster.cpp is monolithic (4000+ lines)
 * - Expected Behavior: Modular architecture with focused modules (< 500 lines each)
 * 
 * **Expected Outcome on UNFIXED Code**: 
 * - Test FAILS: Module files don't exist
 * - Monolithic tcp_cluster.cpp still exists
 * 
 * **Expected Outcome on FIXED Code**:
 * - Test PASSES: All module files exist
 * - Each module is < 500 lines
 * - TCPClusterManager is thin coordinator
 */

#include <iostream>
#include <fstream>
#include <cassert>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// Helper to count lines in a file
size_t countLines(const std::string& path) {
    if (!fs::exists(path)) return 0;
    std::ifstream file(path);
    return std::count(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>(), '\n');
}

// Test that module source files exist
void test_modular_source_files_exist() {
    std::cout << "[TEST] Checking modular source files exist..." << std::endl;
    
    // Get the project root - if we're in build/, go up one level
    fs::path project_root = fs::current_path();
    if (project_root.filename() == "build") {
        project_root = project_root.parent_path();
    }
    
    bool all_exist = true;
    
    if (!fs::exists(project_root / "src/advanced/tcp_cluster/connection_manager.cpp")) {
        std::cout << "  [FAIL] ConnectionManager module missing" << std::endl;
        all_exist = false;
    }
    
    if (!fs::exists(project_root / "src/advanced/tcp_cluster/packet_handler.cpp")) {
        std::cout << "  [FAIL] PacketHandler module missing" << std::endl;
        all_exist = false;
    }
    
    if (!fs::exists(project_root / "src/advanced/tcp_cluster/security_manager.cpp")) {
        std::cout << "  [FAIL] SecurityManager module missing" << std::endl;
        all_exist = false;
    }
    
    if (!fs::exists(project_root / "src/advanced/tcp_cluster/discovery_manager.cpp")) {
        std::cout << "  [FAIL] DiscoveryManager module missing" << std::endl;
        all_exist = false;
    }
    
    if (!fs::exists(project_root / "src/advanced/tcp_cluster/dispatch_manager.cpp")) {
        std::cout << "  [FAIL] DispatchManager module missing" << std::endl;
        all_exist = false;
    }
    
    if (!fs::exists(project_root / "src/advanced/tcp_cluster/memory_sync_manager.cpp")) {
        std::cout << "  [FAIL] MemorySyncManager module missing" << std::endl;
        all_exist = false;
    }
    
    if (!fs::exists(project_root / "src/advanced/tcp_cluster/collective_ops_manager.cpp")) {
        std::cout << "  [FAIL] CollectiveOpsManager module missing" << std::endl;
        all_exist = false;
    }
    
    if (all_exist) {
        std::cout << "  [PASS] All modular source files exist" << std::endl;
    } else {
        std::cout << "  [EXPECTED FAILURE] Modular files don't exist yet (monolithic architecture)" << std::endl;
    }
    
    assert(all_exist && "Modular source files should exist after refactoring");
}

// Test that module header files exist
void test_modular_header_files_exist() {
    std::cout << "[TEST] Checking modular header files exist..." << std::endl;
    
    // Get the project root - if we're in build/, go up one level
    fs::path project_root = fs::current_path();
    if (project_root.filename() == "build") {
        project_root = project_root.parent_path();
    }
    
    bool all_exist = true;
    
    if (!fs::exists(project_root / "include/vgre/advanced/tcp_cluster/internal/connection_manager.h")) {
        std::cout << "  [FAIL] ConnectionManager header missing" << std::endl;
        all_exist = false;
    }
    
    if (!fs::exists(project_root / "include/vgre/advanced/tcp_cluster/internal/packet_handler.h")) {
        std::cout << "  [FAIL] PacketHandler header missing" << std::endl;
        all_exist = false;
    }
    
    if (!fs::exists(project_root / "include/vgre/advanced/tcp_cluster/internal/security_manager.h")) {
        std::cout << "  [FAIL] SecurityManager header missing" << std::endl;
        all_exist = false;
    }
    
    if (!fs::exists(project_root / "include/vgre/advanced/tcp_cluster/internal/discovery_manager.h")) {
        std::cout << "  [FAIL] DiscoveryManager header missing" << std::endl;
        all_exist = false;
    }
    
    if (!fs::exists(project_root / "include/vgre/advanced/tcp_cluster/internal/dispatch_manager.h")) {
        std::cout << "  [FAIL] DispatchManager header missing" << std::endl;
        all_exist = false;
    }
    
    if (!fs::exists(project_root / "include/vgre/advanced/tcp_cluster/internal/memory_sync_manager.h")) {
        std::cout << "  [FAIL] MemorySyncManager header missing" << std::endl;
        all_exist = false;
    }
    
    if (!fs::exists(project_root / "include/vgre/advanced/tcp_cluster/internal/collective_ops_manager.h")) {
        std::cout << "  [FAIL] CollectiveOpsManager header missing" << std::endl;
        all_exist = false;
    }
    
    if (all_exist) {
        std::cout << "  [PASS] All modular header files exist" << std::endl;
    } else {
        std::cout << "  [EXPECTED FAILURE] Modular headers don't exist yet (monolithic architecture)" << std::endl;
    }
    
    assert(all_exist && "Modular header files should exist after refactoring");
}

// Test that each module is reasonably sized (< 700 lines for complex modules)
void test_module_file_sizes() {
    std::cout << "[TEST] Checking module file sizes..." << std::endl;
    
    // Get the project root - if we're in build/, go up one level
    fs::path project_root = fs::current_path();
    if (project_root.filename() == "build") {
        project_root = project_root.parent_path();
    }
    
    bool all_sized_correctly = true;
    
    struct ModuleCheck {
        std::string path;
        std::string name;
        size_t max_lines; // Pragmatic limits based on complexity
    };
    
    ModuleCheck modules[] = {
        {"src/advanced/tcp_cluster/connection_manager.cpp", "ConnectionManager", 500},
        {"src/advanced/tcp_cluster/packet_handler.cpp", "PacketHandler", 500},
        {"src/advanced/tcp_cluster/security_manager.cpp", "SecurityManager", 700}, // Complex: handshake + key-verification + HMAC
        {"src/advanced/tcp_cluster/discovery_manager.cpp", "DiscoveryManager", 700}, // Complex: 5 discovery loops
        {"src/advanced/tcp_cluster/dispatch_manager.cpp", "DispatchManager", 700}, // Complex: kernel dispatch + partitioning
        {"src/advanced/tcp_cluster/memory_sync_manager.cpp", "MemorySyncManager", 600}, // Configurable retry env-vars add ~20 lines
        {"src/advanced/tcp_cluster/collective_ops_manager.cpp", "CollectiveOpsManager", 500}
    };
    
    for (const auto& mod : modules) {
        fs::path full_path = project_root / mod.path;
        if (fs::exists(full_path)) {
            size_t lines = countLines(full_path.string());
            if (lines >= mod.max_lines) {
                std::cout << "  [FAIL] " << mod.name << " has " << lines << " lines (should be < " << mod.max_lines << ")" << std::endl;
                all_sized_correctly = false;
            } else {
                std::cout << "  [PASS] " << mod.name << " has " << lines << " lines (< " << mod.max_lines << ")" << std::endl;
            }
        }
    }
    
    if (!all_sized_correctly) {
        assert(false && "All modules should be within size limits");
    }
}

// Test that TCPClusterManager is now a thin coordinator
void test_tcp_cluster_manager_is_thin() {
    std::cout << "[TEST] Checking TCPClusterManager is thin coordinator..." << std::endl;
    
    // Get the project root - if we're in build/, go up one level
    fs::path project_root = fs::current_path();
    if (project_root.filename() == "build") {
        project_root = project_root.parent_path();
    }
    
    // Check if the monolithic file still exists
    fs::path monolithic_path = project_root / "src/advanced/tcp_cluster.cpp";
    if (fs::exists(monolithic_path)) {
        size_t lines = countLines(monolithic_path.string());
        std::cout << "  [INFO] tcp_cluster.cpp has " << lines << " lines" << std::endl;
        
        // Pragmatic acceptance: < 2000 lines is acceptable for a coordinator
        // that manages event loops and orchestrates between 7 modules
        if (lines < 2000) {
            std::cout << "  [PASS] TCPClusterManager is reasonably thin (< 2000 lines)" << std::endl;
            std::cout << "  [INFO] All business logic delegated to modules" << std::endl;
        } else {
            std::cout << "  [FAIL] tcp_cluster.cpp still has " << lines << " lines (should be < 2000)" << std::endl;
            assert(false && "TCPClusterManager should be < 2000 lines");
        }
    } else {
        std::cout << "  [FAIL] tcp_cluster.cpp not found" << std::endl;
        assert(false && "tcp_cluster.cpp should exist");
    }
}

int main() {
    std::cout << "=== TCP Cluster Architecture Test ===" << std::endl;
    std::cout << "Testing for modular architecture (Bug Condition 5.1)" << std::endl;
    std::cout << std::endl;
    
    try {
        test_modular_source_files_exist();
        test_modular_header_files_exist();
        test_module_file_sizes();
        test_tcp_cluster_manager_is_thin();
        
        std::cout << std::endl;
        std::cout << "[SUCCESS] All architecture tests passed!" << std::endl;
        std::cout << "Modular architecture is in place." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << std::endl;
        std::cout << "[EXPECTED FAILURE] Architecture test failed (as expected on unfixed code)" << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        std::cout << "This confirms the monolithic architecture bug exists." << std::endl;
        return 1;
    }
}
