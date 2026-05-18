#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/advanced/gpu_passthrough.h"
#include "vgre/common/logger.h"
#include "vgre/common/error_codes.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/platform.h"
#ifdef VGRE_HAS_OPENCL_BACKEND
#include "vgre/runtime/igpu_opencl_executor.h"
#endif
#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <cstdlib>
#include "vgre/common/os_backend.h"

static std::atomic<bool>& getStopRequested() {
    static std::atomic<bool> v{false};
    return v;
}

static std::condition_variable mainCv;
static std::mutex mainMtx;

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        getStopRequested().store(true);
        mainCv.notify_all();
    }
}

void print_usage(const char* prog) {
    std::cout << "VGRE Remote Worker Utility\n";
    std::cout << "Usage: " << prog << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --port <p>          Port to listen on (default: 7777)\n";
    std::cout << "  --threads <n>       Number of worker threads (default: auto = CPU cores)\n";
    std::cout << "  --auth-token <t>    Authentication token for secure cluster access\n";
    std::cout << "  --master <ip>       IP address of the master node (default: auto-discovery)\n";
    std::cout << "  --no-gpu            Disable GPU dispatch (force CPU-only execution)\n";
    std::cout << "  --help              Print this help message\n";
}

int main(int argc, char** argv) {
    int  port       = 7777;
    int  threads    = 0;       // 0 = auto-detect from CPU cores
    bool enable_gpu = true;
    std::string auth_token;
    std::string master_ip = "auto";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--auth-token" && i + 1 < argc) {
            auth_token = argv[++i];
        } else if (arg == "--master" && i + 1 < argc) {
            master_ip = argv[++i];
        } else if (arg == "--no-gpu") {
            enable_gpu = false;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Apply thread count to the runtime config before init
    if (threads > 0) {
        vgre_set_config("VGRE_WORKER_THREADS", std::to_string(threads).c_str());
        std::cout << "[Worker] Thread pool: " << threads << " threads\n";
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#if !defined(_WIN32)
    std::signal(SIGPIPE, SIG_IGN); // suppress SIGPIPE on broken pipes (Linux + macOS)
#endif

    vgre::Logger::instance().setLevel(vgre::LogLevel::INFO);
    
    if (!auth_token.empty()) {
        // Hash the token for logging — SHA256, same algorithm as the C++ engine
        // startup log and handshake failure messages so all output is consistent.
        uint8_t digest[vgre::advanced::crypto::kSHA256DigestLen];
        vgre::advanced::crypto::sha256(
            reinterpret_cast<const uint8_t*>(auth_token.data()),
            auth_token.size(), digest);
        std::string token_hash_hex;
        for (size_t k = 0; k < vgre::advanced::crypto::kSHA256DigestLen; ++k) {
            char buf[3]; snprintf(buf, sizeof(buf), "%02x", digest[k]);
            token_hash_hex += buf;
        }
        vgre::Logger::instance().log(vgre::LogLevel::INFO, "Worker",
            "Using provided auth token (SHA256: " + token_hash_hex.substr(0, 16) + "...)");
        // Use thread-safe config store instead of setenv to avoid glibc heap corruption race
        vgre_set_config("VGRE_TCP_AUTH_TOKEN", auth_token.c_str());
    }

    vgre::advanced::TCPClusterManager& cluster = vgre::advanced::TCPClusterManager::instance();
    
    // ── GPU capability probe ──────────────────────────────────────────────────
    // Probe BEFORE cluster init so the capability packet sent during handshake
    // carries accurate GPU information. Results are logged to help operators
    // verify which compute backends are active on this worker node.
    if (enable_gpu) {
        // Discrete NVIDIA GPU via libcuda / nvcuda.dll (runtime dlopen)
        auto& gp = vgre::advanced::GPUPassthrough::instance();
        if (gp.initialize() && gp.isAvailable()) {
            const auto& devs = gp.getDevices();
            std::cout << "[Worker] Discrete GPU: " << (devs.empty() ? "?" : devs[0].name)
                      << " (" << devs.size() << " device(s) via GPU passthrough)" << std::endl;
        } else {
            std::cout << "[Worker] Discrete GPU: not found (libcuda not present)" << std::endl;
        }
#ifdef VGRE_HAS_OPENCL_BACKEND
        // Integrated GPU via OpenCL
        auto& igpu = vgre::runtime::IGPUOpenCLExecutor::instance();
        if (igpu.initialize() == vgre::VGREResult::SUCCESS) {
            std::cout << "[Worker] Integrated GPU: " << igpu.getDeviceName()
                      << " (OpenCL, " << std::fixed
                      << igpu.getEstimatedGFLOPS() << " GFLOPS est.)" << std::endl;
        } else {
            std::cout << "[Worker] Integrated GPU: not available (no OpenCL device)" << std::endl;
        }
#else
        std::cout << "[Worker] Integrated GPU: disabled (compiled without VGRE_HAS_OPENCL_BACKEND)" << std::endl;
#endif
    } else {
        std::cout << "[Worker] GPU dispatch disabled (--no-gpu)" << std::endl;
    }

    vgre::Logger::instance().log(vgre::LogLevel::INFO, "Worker", "Initializing VGRE Remote Worker on port " + std::to_string(port));
    std::cout << "[Worker] Startup phase 1/2: Initializing networking..." << std::endl;
    std::cout.flush();

    // Initialize as Worker (is_master = false)
    vgre::VGREResult initRes = cluster.initialize(false, master_ip, port);
    if (initRes != vgre::VGREResult::SUCCESS) {
        vgre::Logger::instance().log(vgre::LogLevel::ERR, "Worker", "Failed to initialize TCP Cluster worker: " + std::to_string(static_cast<int>(initRes)));
        std::cerr << "[Worker] FATAL: Initialization failed. Check if port " << port << " is available." << std::endl;
        return 1;
    }

    vgre::Logger::instance().log(vgre::LogLevel::INFO, "Worker", "Worker node is active and scanning for master (UDP Discovery)...");
    std::cout << "[Worker] Startup phase 2/2: Entering discovery loop, scanning local subnet..." << std::endl;
    std::cout.flush();
    
    int reconnectCounter = 0;
    while (!getStopRequested().load()) {
        std::unique_lock<std::mutex> lock(mainMtx);
        mainCv.wait_for(lock, std::chrono::milliseconds(200), [] { return getStopRequested().load(); });
        if (getStopRequested().load()) break;
        
        if (!cluster.isEnabled()) {
            if (master_ip == "auto") {
                // If in standby mode, don't exit! Try to restart the cluster manager.
                vgre::Logger::instance().log(vgre::LogLevel::WARN, "Worker", "Cluster engine went offline unexpectedly; attempting to re-enable (attempt #" + std::to_string(++reconnectCounter) + ")...");
                std::cout << "[Worker] Cluster engine offline. Retrying..." << std::endl;
                std::cout.flush();
                mainCv.wait_for(lock, std::chrono::seconds(2), [] { return getStopRequested().load(); });
                if (getStopRequested().load()) break;
                cluster.initialize(false, master_ip, port);
            } else {
                // In direct mode, we exit upon failure as per previous design.
                vgre::Logger::instance().log(vgre::LogLevel::WARN, "Worker", "Direct connection lost; shutting down.");
                break;
            }
        } else {
            reconnectCounter = 0; // reset on success
        }
    }

    vgre::Logger::instance().log(vgre::LogLevel::INFO, "Worker", "Shutting down worker...");
    std::cout << "[Worker] Shutdown signal received. Cleaning up..." << std::endl;
    std::cout.flush();
    cluster.shutdown();

    return 0;
}
