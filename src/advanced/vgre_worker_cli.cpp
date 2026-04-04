#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include "vgre/common/error_codes.h"
#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdlib>

static std::atomic<bool> g_stop_requested(false);

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        g_stop_requested.store(true);
    }
}

void print_usage(const char* prog) {
    std::cout << "VGRE Remote Worker Utility\n";
    std::cout << "Usage: " << prog << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --port <p>          Port to listen on (default: 9090)\n";
    std::cout << "  --auth-token <t>    Authentication token for secure cluster access\n";
    std::cout << "  --master <ip>       IP address of the master node (default: auto-discovery)\n";
    std::cout << "  --help              Print this help message\n";
}

int main(int argc, char** argv) {
    int port = 9090;
    std::string auth_token;
    std::string master_ip = "auto";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--auth-token" && i + 1 < argc) {
            auth_token = argv[++i];
        } else if (arg == "--master" && i + 1 < argc) {
            master_ip = argv[++i];
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    vgre::Logger::instance().setLevel(vgre::LogLevel::INFO);
    
    if (!auth_token.empty()) {
        vgre::Logger::instance().log(vgre::LogLevel::INFO, "Worker", "Using provided auth token.");
        setenv("VGRE_TCP_AUTH_TOKEN", auth_token.c_str(), 1);
    }

    vgre::advanced::TCPClusterManager& cluster = vgre::advanced::TCPClusterManager::instance();
    
    vgre::Logger::instance().log(vgre::LogLevel::INFO, "Worker", "Initializing VGRE Remote Worker on port " + std::to_string(port));
    
    // Initialize as Worker (is_master = false)
    if (cluster.initialize(false, master_ip, port) != vgre::VGREResult::SUCCESS) {
        vgre::Logger::instance().log(vgre::LogLevel::ERROR, "Worker", "Failed to initialize TCP Cluster worker.");
        return 1;
    }

    vgre::Logger::instance().log(vgre::LogLevel::INFO, "Worker", "Worker node is active and waiting for master connection...");
    
    while (!g_stop_requested.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        
        // Safety check: if cluster was disabled externally, stop.
        if (!cluster.isEnabled()) {
            vgre::Logger::instance().log(vgre::LogLevel::WARN, "Worker", "Cluster disabled; shutting down.");
            break;
        }
    }

    vgre::Logger::instance().log(vgre::LogLevel::INFO, "Worker", "Shutting down worker...");
    cluster.shutdown();

    return 0;
}
