/**
 * VGRE Bug Condition Exploration Test — TCP Cluster Monolithic Architecture
 * 
 * **CRITICAL**: This test is EXPECTED TO FAIL on unfixed code.
 * Failure confirms the bug exists (monolithic architecture not properly modularized).
 * 
 * **Purpose**: Verify that TCP cluster is properly modularized into separate manager classes.
 * 
 * **Bug Condition Being Tested**:
 * - Bug Condition 5.1: Monolithic 3,267-line file should be split into focused modules
 * 
 * **Expected Outcome on UNFIXED Code**: 
 * - Missing modular directory structure
 * - Missing separate manager class files
 * - Large monolithic files (>500 lines)
 * 
 * **Expected Outcome on FIXED Code**:
 * - Modular directory structure exists
 * - Each manager class has its own file
 * - Files are reasonably sized (<500 lines)
 * 
 * **Validates: Requirements 5.1**
 * 
 * This test encodes the expected behavior - it will validate the fix when it passes.
 */

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <cassert>

namespace fs = std::filesystem;

// Helper to count lines in a file
int countLinesInFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return -1; // File doesn't exist
    }
    
    int line_count = 0;
    std::string line;
    while (std::getline(file, line)) {
        line_count++;
    }
    return line_count;
}

// Helper to check if a class is defined in a file
bool classDefinedInFile(const std::string& filepath, const std::string& class_name) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return false;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    
    // Look for class definition pattern
    std::string class_pattern = "class " + class_name;
    return content.find(class_pattern) != std::string::npos;
}

/**
 * Test that modular source files exist for each expected manager class.
 * 
 * On unfixed code: Files don't exist - confirms monolithic architecture bug
 * On fixed code: All module files exist
 */
void test_modular_source_files_exist() {
    std::cout << "[TEST] Modular source files existence test..." << std::endl;
    
    std::string tcp_cluster_src_dir = "src/advanced/tcp_cluster";
    std::vector<std::string> expected_modules = {
        "connection_manager",
        "packet_handler", 
        "security_manager",
        "dispatch_manager",
        "memory_sync_manager",
        "collective_ops_manager"
    };
    
    bool all_exist = true;
    for (const auto& module : expected_modules) {
        std::string src_file = tcp_cluster_src_dir + "/" + module + ".cpp";
        
        if (!fs::exists(src_file)) {
            // Special case for dispatch_manager: check if it's split into multiple files
            if (module == "dispatch_manager") {
                bool has_dispatch_files = fs::exists(tcp_cluster_src_dir + "/dispatch_manager_core.cpp") ||
                                        fs::exists(tcp_cluster_src_dir + "/dispatch_manager_partitioned.cpp") ||
                                        fs::exists(tcp_cluster_src_dir + "/dispatch_manager_remote.cpp") ||
                                        fs::exists(tcp_cluster_src_dir + "/dispatch_impl.cpp");
                if (has_dispatch_files) {
                    std::cout << "  FOUND: dispatch_manager split into multiple files (dispatch_manager_*.cpp)" << std::endl;
                    continue;
                }
            }
            std::cout << "  MISSING: " << src_file << " (indicates monolithic architecture bug)" << std::endl;
            all_exist = false;
        } else {
            std::cout << "  FOUND: " << src_file << std::endl;
        }
    }
    
    assert(all_exist && "Module source files missing - monolithic architecture bug detected");
    std::cout << "  ✓ All modular source files exist" << std::endl;
}

/**
 * Test that modular header files exist for each expected manager class.
 * 
 * On unfixed code: Headers don't exist - confirms monolithic architecture bug
 * On fixed code: All module headers exist
 */
void test_modular_header_files_exist() {
    std::cout << "[TEST] Modular header files existence test..." << std::endl;
    
    std::string tcp_cluster_include_dir = "include/vgre/advanced/tcp_cluster/internal";
    std::vector<std::string> expected_modules = {
        "connection_manager",
        "packet_handler", 
        "security_manager",
        "dispatch_manager",
        "memory_sync_manager",
        "collective_ops_manager"
    };
    
    bool all_exist = true;
    for (const auto& module : expected_modules) {
        std::string header_file = tcp_cluster_include_dir + "/" + module + ".h";
        
        if (!fs::exists(header_file)) {
            std::cout << "  MISSING: " << header_file << " (indicates monolithic architecture bug)" << std::endl;
            all_exist = false;
        } else {
            std::cout << "  FOUND: " << header_file << std::endl;
        }
    }
    
    assert(all_exist && "Module header files missing - monolithic architecture bug detected");
    std::cout << "  ✓ All modular header files exist" << std::endl;
}

/**
 * Test that module files are reasonably sized (< 500 lines).
 * 
 * On unfixed code: Large monolithic files - confirms architecture bug
 * On fixed code: All files are under 500 lines
 */
void test_module_files_are_reasonably_sized() {
    std::cout << "[TEST] Module file size test..." << std::endl;
    
    const int MAX_LINES_PER_MODULE = 500;
    std::string tcp_cluster_src_dir = "src/advanced/tcp_cluster";
    std::vector<std::string> expected_modules = {
        "connection_manager",
        "packet_handler", 
        "security_manager",
        "dispatch_manager",
        "memory_sync_manager",
        "collective_ops_manager"
    };
    
    bool all_reasonable = true;
    for (const auto& module : expected_modules) {
        std::string src_file = tcp_cluster_src_dir + "/" + module + ".cpp";
        
        if (fs::exists(src_file)) {
            int line_count = countLinesInFile(src_file);
            std::cout << "  " << src_file << ": " << line_count << " lines";
            
            if (line_count >= MAX_LINES_PER_MODULE) {
                std::cout << " (TOO LARGE - max: " << MAX_LINES_PER_MODULE << ")" << std::endl;
                all_reasonable = false;
            } else {
                std::cout << " (OK)" << std::endl;
            }
        }
    }
    
    assert(all_reasonable && "Module files too large - insufficient modularization");
    std::cout << "  ✓ All module files are reasonably sized" << std::endl;
}

/**
 * Test that classes are defined in separate files.
 * 
 * On unfixed code: Classes not found in expected files - confirms monolithic bug
 * On fixed code: Each class is in its own dedicated file
 */
void test_classes_are_defined_in_separate_files() {
    std::cout << "[TEST] Class separation test..." << std::endl;
    
    std::string tcp_cluster_include_dir = "include/vgre/advanced/tcp_cluster/internal";
    
    struct ClassFileMapping {
        std::string class_name;
        std::string expected_file;
    };
    
    std::vector<ClassFileMapping> class_mappings = {
        {"ConnectionManager", tcp_cluster_include_dir + "/connection_manager.h"},
        {"PacketHandler", tcp_cluster_include_dir + "/packet_handler.h"},
        {"SecurityManager", tcp_cluster_include_dir + "/security_manager.h"},
        {"DispatchManager", tcp_cluster_include_dir + "/dispatch_manager.h"},
        {"MemorySyncManager", tcp_cluster_include_dir + "/memory_sync_manager.h"},
        {"CollectiveOpsManager", tcp_cluster_include_dir + "/collective_ops_manager.h"}
    };
    
    bool all_found = true;
    for (const auto& mapping : class_mappings) {
        bool found = classDefinedInFile(mapping.expected_file, mapping.class_name);
        std::cout << "  " << mapping.class_name << " in " << mapping.expected_file << ": ";
        
        if (!found) {
            std::cout << "NOT FOUND (indicates monolithic architecture bug)" << std::endl;
            all_found = false;
        } else {
            std::cout << "FOUND" << std::endl;
        }
    }
    
    assert(all_found && "Classes not found in expected files - monolithic architecture bug detected");
    std::cout << "  ✓ All classes are defined in separate files" << std::endl;
}

/**
 * Test that there is no single monolithic file.
 * 
 * On unfixed code: Large monolithic file exists - confirms architecture bug
 * On fixed code: Main file is reasonably sized
 */
void test_no_single_monolithic_file() {
    std::cout << "[TEST] No monolithic file test..." << std::endl;
    
    const int MAX_MONOLITHIC_LINES = 1000; // Threshold for "too monolithic"
    
    // Check the main tcp_cluster.cpp file
    std::string main_file = "src/advanced/tcp_cluster.cpp";
    
    if (fs::exists(main_file)) {
        int line_count = countLinesInFile(main_file);
        std::cout << "  " << main_file << ": " << line_count << " lines";
        
        if (line_count >= MAX_MONOLITHIC_LINES) {
            std::cout << " (TOO LARGE - max: " << MAX_MONOLITHIC_LINES << ")" << std::endl;
            assert(false && "Main TCP cluster file is too large - monolithic architecture bug not fully resolved");
        } else {
            std::cout << " (OK)" << std::endl;
        }
    } else {
        std::cout << "  " << main_file << ": NOT FOUND" << std::endl;
    }
    
    std::cout << "  ✓ No single monolithic file detected" << std::endl;
}

/**
 * Test that modular directory structure exists.
 * 
 * On unfixed code: Directory structure missing - confirms monolithic bug
 * On fixed code: Proper directory structure exists
 */
void test_modular_directory_structure_exists() {
    std::cout << "[TEST] Modular directory structure test..." << std::endl;
    
    std::string tcp_cluster_src_dir = "src/advanced/tcp_cluster";
    std::string tcp_cluster_include_dir = "include/vgre/advanced/tcp_cluster/internal";
    
    bool src_exists = fs::exists(tcp_cluster_src_dir) && fs::is_directory(tcp_cluster_src_dir);
    bool include_exists = fs::exists(tcp_cluster_include_dir) && fs::is_directory(tcp_cluster_include_dir);
    
    std::cout << "  " << tcp_cluster_src_dir << ": " << (src_exists ? "EXISTS" : "MISSING") << std::endl;
    std::cout << "  " << tcp_cluster_include_dir << ": " << (include_exists ? "EXISTS" : "MISSING") << std::endl;
    
    assert(src_exists && "TCP cluster source directory missing - monolithic architecture bug");
    assert(include_exists && "TCP cluster internal headers directory missing - monolithic architecture bug");
    
    std::cout << "  ✓ Modular directory structure exists" << std::endl;
}

/**
 * Test that each module contains focused functionality.
 * 
 * On unfixed code: Modules don't exist or lack focused logic - confirms bug
 * On fixed code: Each module contains appropriate functionality
 */
void test_focused_responsibilities_per_module() {
    std::cout << "[TEST] Focused responsibilities test..." << std::endl;
    
    std::string tcp_cluster_src_dir = "src/advanced/tcp_cluster";
    
    // Check that connection_manager.cpp contains connection-related functionality
    std::string conn_mgr_file = tcp_cluster_src_dir + "/connection_manager.cpp";
    if (fs::exists(conn_mgr_file)) {
        std::ifstream file(conn_mgr_file);
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        
        // Should contain connection management functions
        bool has_connection_logic = 
            content.find("acceptConnection") != std::string::npos ||
            content.find("connectToMaster") != std::string::npos ||
            content.find("closeConnection") != std::string::npos;
            
        std::cout << "  ConnectionManager has connection logic: " << (has_connection_logic ? "YES" : "NO") << std::endl;
        assert(has_connection_logic && "ConnectionManager file lacks connection management logic - improper modularization");
    } else {
        std::cout << "  ConnectionManager file: NOT FOUND" << std::endl;
        assert(false && "ConnectionManager file missing");
    }
    
    // Check that security_manager.cpp contains security-related functionality  
    std::string sec_mgr_file = tcp_cluster_src_dir + "/security_manager.cpp";
    if (fs::exists(sec_mgr_file)) {
        std::ifstream file(sec_mgr_file);
        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        
        // Should contain security functions
        bool has_security_logic = 
            content.find("performSecureHandshake") != std::string::npos ||
            content.find("handshake") != std::string::npos ||
            content.find("security") != std::string::npos;
            
        std::cout << "  SecurityManager has security logic: " << (has_security_logic ? "YES" : "NO") << std::endl;
        assert(has_security_logic && "SecurityManager file lacks security management logic - improper modularization");
    } else {
        std::cout << "  SecurityManager file: NOT FOUND" << std::endl;
        assert(false && "SecurityManager file missing");
    }
    
    std::cout << "  ✓ Modules have focused responsibilities" << std::endl;
}

/**
 * Main test function that runs all architecture tests.
 */
int main() {
    std::cout << "=== TCP Cluster Monolithic Architecture Bug Condition Exploration Test ===" << std::endl;
    std::cout << std::endl;
    std::cout << "This test detects the monolithic architecture bug condition." << std::endl;
    std::cout << "EXPECTED on UNFIXED code: Test FAILS (confirms monolithic bug exists)" << std::endl;
    std::cout << "EXPECTED on FIXED code: Test PASSES (confirms modular architecture)" << std::endl;
    std::cout << std::endl;
    
    try {
        test_modular_directory_structure_exists();
        test_modular_source_files_exist();
        test_modular_header_files_exist();
        test_module_files_are_reasonably_sized();
        test_classes_are_defined_in_separate_files();
        test_no_single_monolithic_file();
        test_focused_responsibilities_per_module();
        
        std::cout << std::endl;
        std::cout << "=== ALL TESTS PASSED ===" << std::endl;
        std::cout << "✓ Modular architecture is properly implemented" << std::endl;
        std::cout << "✓ Bug condition exploration test PASSED - monolithic architecture bug is FIXED" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << std::endl;
        std::cout << "=== TEST FAILED ===" << std::endl;
        std::cout << "✗ Monolithic architecture bug detected: " << e.what() << std::endl;
        std::cout << "✗ Bug condition exploration test FAILED - confirms monolithic architecture bug exists" << std::endl;
        
        return 1;
    }
}