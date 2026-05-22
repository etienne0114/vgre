#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/tcp_cluster_defaults.h"
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
#include <cstring>
#include <cstdio>
#include "vgre/common/os_backend.h"

#if defined(_WIN32)
// Win32 raw console write — guaranteed before any C++ stream init.
static void vgreDiag(const char* msg) {
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    DWORD n = (DWORD)strlen(msg);
    DWORD written = 0;
    if (!WriteConsoleA(h, msg, n, &written, NULL))
        WriteFile(h, msg, n, &written, NULL);
}

// Persistent file log — survives even when console/pipe handles are invalid.
static void vgreFileLog(const char* msg) {
    char path[MAX_PATH];
    DWORD len = GetTempPathA(MAX_PATH, path);
    if (len == 0 || len >= MAX_PATH) return;
    strncat(path, "\\vgre_worker_startup.log", MAX_PATH - len - 1);
    path[MAX_PATH - 1] = '\0';
    FILE* f = nullptr;
    if (fopen_s(&f, path, "a") == 0 && f) {
        SYSTEMTIME st;
        GetLocalTime(&st);
        fprintf(f, "%04d-%02d-%02d %02d:%02d:%02d.%03d  %s\n",
                st.wYear, st.wMonth, st.wDay,
                st.wHour, st.wMinute, st.wSecond, st.wMilliseconds,
                msg);
        fclose(f);
    }
}

static LONG WINAPI vgreWorkerExceptionFilter(EXCEPTION_POINTERS* ep) {
    char buf[512];
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    // Use Win32 directly so it works even after C++ infrastructure fails.
    vgreDiag("\r\n[VGRE CRASH] Unhandled exception — see code below.\r\n");
    snprintf(buf, sizeof(buf),
        "[VGRE CRASH] code=0x%08lX  STATUS_ACCESS_VIOLATION=0xC0000005\r\n"
        "[VGRE CRASH] addr=%p\r\n",
        (unsigned long)code, ep->ExceptionRecord->ExceptionAddress);
    vgreDiag(buf);
    vgreDiag("[VGRE CRASH] Check the last [VGRE-DIAG] line above for location.\r\n");
    vgreFileLog(buf);
    vgreFileLog("[VGRE CRASH] Check %TEMP%\\vgre_worker_startup.log for startup trace.");
    return EXCEPTION_CONTINUE_SEARCH;
}
#define VGRE_DIAG(msg) do { vgreDiag("[VGRE-DIAG] " msg "\r\n"); vgreFileLog(msg); } while(0)
#else
#define VGRE_DIAG(msg) ((void)0)
#endif

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
    std::cout <<
        "VGRE Remote Worker\n"
        "\n"
        "Usage:\n"
        "  " << prog << " [options]\n"
        "\n"
        "Options:\n"
        "  --port <N>                   TCP port (default: " << vgre::advanced::kDefaultClusterPort << ")\n"
        "  --threads <N>                Worker thread count (default: auto = CPU cores)\n"
        "  --master <IP>                Master IP for direct LAN connect\n"
        "  --master-address <HOST:PORT> Master address — hostname, IPv4, or [::1]:port\n"
        "                               Required for WAN / internet clusters.\n"
        "                               Also reads from VGRE_CLUSTER_MASTER_ADDRESS env var.\n"
        "  --auth-token <TOKEN>         Auth token (prefer VGRE_TCP_AUTH_TOKEN_FILE)\n"
        "  --no-gpu                     Disable GPU dispatch (CPU-only execution)\n"
        "  --help                       Print this help\n"
        "\n"
        "Connection modes:\n"
        "  No --master*       — LAN UDP broadcast auto-discovery (default)\n"
        "  --master <IP>      — direct connect to LAN master, auto-discover as fallback\n"
        "  --master-address   — direct connect to WAN/hostname master\n"
        "\n"
        "Recommended: use 'vgre-start --worker' which sets up PATH and token automatically.\n"
        "\n"
        "Examples:\n"
        "  vgre-worker                                  # LAN auto-discover\n"
        "  vgre-worker --master 192.168.1.10            # LAN explicit master\n"
        "  vgre-worker --master-address 78.45.12.99:7777  # WAN master\n"
        "  vgre-worker --master-address myhost.example.com:7777  # hostname\n";
}

// Split "host:port" or "[::1]:port" into components.
// Returns false on parse failure (address used unchanged, port unchanged).
static bool splitHostPort(const std::string& addr, std::string& host, int& port) {
    if (addr.empty()) return false;
    if (addr.front() == '[') {
        size_t close = addr.find(']');
        if (close == std::string::npos) return false;
        host = addr.substr(1, close - 1);
        size_t colon = addr.find(':', close + 1);
        if (colon == std::string::npos) return false;
        try { port = std::stoi(addr.substr(colon + 1)); } catch (...) { return false; }
    } else {
        size_t colon = addr.rfind(':');
        if (colon == std::string::npos) { host = addr; return true; }
        host = addr.substr(0, colon);
        try { port = std::stoi(addr.substr(colon + 1)); } catch (...) { return false; }
    }
    return !host.empty() && port > 0 && port < 65536;
}

int main(int argc, char** argv) {
    VGRE_DIAG("main() entered");
#if defined(_WIN32)
    SetUnhandledExceptionFilter(vgreWorkerExceptionFilter);
#endif
    VGRE_DIAG("step 1/6: arg parse");
    int         port       = vgre::advanced::kDefaultClusterPort;
    int         threads    = 0;    // 0 = auto-detect from CPU cores
    bool        enable_gpu = true;
    std::string auth_token;
    std::string master_host;       // empty = use UDP auto-discovery
    bool        explicit_master = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--port" && i + 1 < argc) {
            try { port = std::stoi(argv[++i]); }
            catch (...) {
                std::cerr << "[Worker] Invalid port: " << argv[i] << "\n";
                return 1;
            }
        } else if (arg == "--threads" && i + 1 < argc) {
            try { threads = std::stoi(argv[++i]); }
            catch (...) {
                std::cerr << "[Worker] Invalid thread count: " << argv[i] << "\n";
                return 1;
            }
        } else if (arg == "--auth-token" && i + 1 < argc) {
            auth_token = argv[++i];

        } else if (arg == "--master" && i + 1 < argc) {
            // Legacy: just an IP, no port.
            master_host    = argv[++i];
            explicit_master = true;

        } else if (arg == "--master-address" && i + 1 < argc) {
            // New: HOST:PORT — parses hostname, IPv4 literal, or [::1]:port.
            std::string raw = argv[++i];
            int parsed_port  = port;
            std::string parsed_host;
            if (splitHostPort(raw, parsed_host, parsed_port)) {
                master_host = parsed_host;
                port        = parsed_port;
            } else {
                master_host = raw;   // pass as-is; getaddrinfo will resolve it
            }
            explicit_master = true;

        } else if (arg == "--no-gpu") {
            enable_gpu = false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "[Worker] Unknown option: " << arg << "\n"
                      << "         Run: vgre-worker --help\n";
            return 1;
        } else {
            // Positional argument — most likely user typed a subcommand by mistake.
            std::cerr << "[Worker] Unexpected argument: '" << arg << "'\n"
                      << "         vgre-worker does not take subcommands.\n"
                      << "         To start a worker node, just run:  vgre-worker\n"
                      << "         Or use the high-level launcher:     vgre-start --worker\n"
                      << "         See all options:                     vgre-worker --help\n";
            return 1;
        }
    }

    VGRE_DIAG("step 2/6: env-var config read");
    // Environment variable fallback for master address (set by vgre-start/vgre-discover).
    // Only applies when no --master* flag was given on the command line.
    if (!explicit_master) {
        const char* envAddr = vgre_get_config("VGRE_CLUSTER_MASTER_ADDRESS");
        if (envAddr && envAddr[0]) {
            std::string parsed_host;
            int parsed_port = port;
            if (splitHostPort(std::string(envAddr), parsed_host, parsed_port)) {
                master_host = parsed_host;
                port        = parsed_port;
            } else {
                master_host = std::string(envAddr);
            }
            explicit_master = true;
        }
    }

    // Thread count
    if (threads > 0) {
        vgre_set_config("VGRE_WORKER_THREADS", std::to_string(threads).c_str());
        std::cout << "[Worker] Thread pool: " << threads << " threads\n";
    }

    VGRE_DIAG("step 3/6: signal handlers");
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
#if !defined(_WIN32)
    std::signal(SIGPIPE, SIG_IGN);
#endif

    VGRE_DIAG("step 4/6: logger init");
    vgre::Logger::instance().setLevel(vgre::LogLevel::INFO);
    VGRE_DIAG("step 4/6: logger OK");

    // Auth token from flag
    if (!auth_token.empty()) {
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
        vgre_set_config("VGRE_TCP_AUTH_TOKEN", auth_token.c_str());
    }

    VGRE_DIAG("step 5/6: compute backends probe");
    // ── Compute backend probe ─────────────────────────────────────────────────
    // Print and flush BEFORE initializing the TCP cluster manager so the
    // console always shows something even if networking init crashes on Windows.
    std::cout << "[Worker] Compute backends:\n";
    std::cout.flush();

    if (enable_gpu) {
        // ── NVIDIA GPU passthrough (optional) ─────────────────────────────────
        auto& gp = vgre::advanced::GPUPassthrough::instance();
        if (gp.initialize() && gp.isAvailable()) {
            const auto& devs = gp.getDevices();
            for (size_t di = 0; di < devs.size(); ++di) {
                std::cout << "[Worker]   NVIDIA CUDA   ACTIVE  " << devs[di].name
                          << "  (device " << di << ", passthrough)\n";
                std::cout.flush();
            }
        } else {
            std::cout << "[Worker]   NVIDIA CUDA   n/a     no NVIDIA GPU detected\n";
            std::cout.flush();
        }

        // ── Integrated / discrete non-NVIDIA GPU via OpenCL (optional) ────────
#ifdef VGRE_HAS_OPENCL_BACKEND
        auto& igpu = vgre::runtime::IGPUOpenCLExecutor::instance();
        if (igpu.initialize() == vgre::VGREResult::SUCCESS) {
            std::cout << "[Worker]   iGPU OpenCL   ACTIVE  " << igpu.getDeviceName()
                      << "  (" << std::fixed << igpu.getEstimatedGFLOPS() << " GFLOPS est.)\n";
            std::cout.flush();
        } else {
            std::cout << "[Worker]   iGPU OpenCL   n/a     no OpenCL device found\n";
            std::cout.flush();
        }
#else
        std::cout << "[Worker]   iGPU OpenCL   n/a     not compiled in (VGRE_HAS_OPENCL_BACKEND off)\n";
        std::cout.flush();
#endif
    } else {
        std::cout << "[Worker]   NVIDIA CUDA   off     --no-gpu\n";
        std::cout.flush();
        std::cout << "[Worker]   iGPU OpenCL   off     --no-gpu\n";
        std::cout.flush();
    }

    // CPU LLVM JIT is always active — it is the primary CUDA emulation engine.
    std::cout << "[Worker]   CPU LLVM JIT  ACTIVE  primary CUDA emulation engine\n";
    std::cout.flush();

    VGRE_DIAG("step 6/6: TCP cluster init (Winsock)");
    // Initialize the TCP cluster manager (Winsock / networking).
    // Done here — after flushing compute-backend output — so the console always
    // shows something useful if this step crashes on Windows.
    vgre::advanced::TCPClusterManager& cluster = vgre::advanced::TCPClusterManager::instance();

    vgre::Logger::instance().log(vgre::LogLevel::INFO, "Worker",
        "Initializing VGRE Remote Worker on port " + std::to_string(port));
    std::cout << "[Worker] Startup phase 1/2: Initializing networking...\n";
    std::cout.flush();

    // Initialize as worker.  Empty master_host → pure UDP auto-discovery.
    vgre::VGREResult initRes = cluster.initialize(false, master_host, port);
    if (initRes != vgre::VGREResult::SUCCESS) {
        vgre::Logger::instance().log(vgre::LogLevel::ERR, "Worker",
            "Failed to initialize TCP Cluster: " + std::to_string(static_cast<int>(initRes)));
        std::cerr << "[Worker] FATAL: Initialization failed. Check if port " << port << " is available.\n";
        return 1;
    }

    if (explicit_master) {
        std::cout << "[Worker] Startup phase 2/2: Connecting to master at "
                  << master_host << ":" << port << " ...\n";
    } else {
        std::cout << "[Worker] Startup phase 2/2: Scanning local subnet for master (UDP discovery)...\n";
    }
    std::cout.flush();

    // ── Main loop — keep worker running; handle reconnection ─────────────────
    int reconnectCounter = 0;
    while (!getStopRequested().load()) {
        std::unique_lock<std::mutex> lock(mainMtx);
        mainCv.wait_for(lock, std::chrono::milliseconds(200),
                        [] { return getStopRequested().load(); });
        if (getStopRequested().load()) break;

        if (!cluster.isEnabled()) {
            if (!explicit_master) {
                // Auto-discovery mode: keep scanning, never give up.
                vgre::Logger::instance().log(vgre::LogLevel::WARN, "Worker",
                    "Cluster engine went offline; restarting (attempt #" +
                    std::to_string(++reconnectCounter) + ")...");
                std::cout << "[Worker] Cluster engine offline. Retrying...\n";
                std::cout.flush();
                mainCv.wait_for(lock, std::chrono::seconds(2),
                                [] { return getStopRequested().load(); });
                if (getStopRequested().load()) break;
                cluster.initialize(false, master_host, port);
            } else {
                // Explicit master: the proactive reconnect loop inside the cluster
                // manager handles retries with exponential backoff.  Only exit when
                // explicitly stopped.
                mainCv.wait_for(lock, std::chrono::seconds(5),
                                [] { return getStopRequested().load(); });
            }
        } else {
            reconnectCounter = 0;
        }
    }

    vgre::Logger::instance().log(vgre::LogLevel::INFO, "Worker", "Shutting down worker...");
    std::cout << "[Worker] Shutdown signal received. Cleaning up...\n";
    std::cout.flush();
    cluster.shutdown();
    return 0;
}
