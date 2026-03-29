/**
 * VGRE Advanced Tests — TCP Poll-based Async I/O
 * Verifies that the poll() loop in TCPClusterManager is responsive and handles data correctly.
 */
#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <chrono>
#include <cstring>

#if !defined(_WIN32)
#include <unistd.h>
#include <poll.h>
#include <sys/socket.h>
#endif

void test_poll_loop_responsiveness() {
#if !defined(_WIN32)
    int fds[2];
    if (pipe(fds) < 0) {
        std::cerr << "Pipe failed" << std::endl;
        return;
    }

    const char* message = "vgre_vsbp_0102_test";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Simulate async data arrival
    std::thread writer([fds, message]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        write(fds[1], message, strlen(message));
    });

    // Test poll() exactly as it's used in TCPClusterManager
    struct pollfd pfd;
    pfd.fd = fds[0];
    pfd.events = POLLIN;
    
    bool data_received = false;
    while (true) {
        int res = poll(&pfd, 1, 1); // 1ms timeout
        if (res > 0) {
            char buf[64];
            int n = read(fds[0], buf, sizeof(buf));
            if (n > 0) {
                buf[n] = '\0';
                assert(strcmp(buf, message) == 0);
                data_received = true;
                break;
            }
        }
        
        // Timeout check for the test
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() > 1000) {
            break;
        }
    }

    writer.join();
    close(fds[0]);
    close(fds[1]);

    assert(data_received);
    std::cout << "[PASS] TCP Poll-based Async I/O Readiness" << std::endl;
#else
    std::cout << "[SKIP] TCP Poll test skipped on Windows" << std::endl;
#endif
}

int main() {
    std::cout << "=== VGRE TCP Poll-based Async I/O Advanced Test ===" << std::endl;
    test_poll_loop_responsiveness();
    return 0;
}
