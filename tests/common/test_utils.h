#ifndef VGRE_TEST_UTILS_H
#define VGRE_TEST_UTILS_H

#include <iostream>
#include <string>
#include <functional>
#include <cmath>

namespace vgre {
namespace test {

// Test assertion macros
#define VGRE_TEST_ASSERT(condition, message)                                   \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "FAIL: " << message << std::endl;                          \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define VGRE_TEST_ASSERT_NEAR(actual, expected, epsilon, message)              \
  do {                                                                         \
    if (std::abs((actual) - (expected)) > (epsilon)) {                         \
      std::cerr << "FAIL: " << message << " (expected " << (expected)          \
                << ", got " << (actual) << ")" << std::endl;                   \
      return false;                                                            \
    }                                                                          \
  } while (0)

// Test runner class for consistent test execution
class TestRunner {
public:
    explicit TestRunner(const std::string& suiteName) : suiteName_(suiteName) {
        std::cout << "\n=== " << suiteName_ << " ===\n" << std::endl;
    }

    // Run a single test function
    void runTest(bool (*testFunc)(), const char* name) {
        total_++;
        std::cout << "Test: " << name << "..." << std::endl;
        if (testFunc()) {
            passed_++;
            std::cout << "  ✓ " << name << " passed" << std::endl;
        } else {
            std::cerr << "  ✗ " << name << " FAILED" << std::endl;
        }
    }

    // Finish test suite and return exit code
    int finish() {
        std::cout << "\n=== Test Results ===" << std::endl;
        std::cout << "Passed: " << passed_ << "/" << total_ << std::endl;

        if (passed_ == total_) {
            std::cout << "\n✓ All Tests Passed ✓\n" << std::endl;
            return 0;
        } else {
            std::cout << "\n✗ Some Tests Failed ✗\n" << std::endl;
            return 1;
        }
    }

    // Get statistics
    int getPassed() const { return passed_; }
    int getTotal() const { return total_; }

private:
    std::string suiteName_;
    int passed_ = 0;
    int total_ = 0;
};

} // namespace test
} // namespace vgre

#endif // VGRE_TEST_UTILS_H
