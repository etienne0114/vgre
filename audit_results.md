# Deep Codebase Audit Results

## ./include/vgre/advanced/grpc_transport.h
### stubs_and_mocks
- Line 46: `// VGREGRPCClient provides a C++ wrapper around the generated stub so that`
- Line 82: `void*       stubPtr_ = nullptr;   // VGRECluster::Stub* stored opaquely`

## ./include/vgre/advanced/hardware_token_manager.h
### basic_logic
- Line 27: `* - Fallback: TPM 2.0 NV storage (if available)`
- Line 29: `* This replaces the insecure environment variable fallback.`
- Line 43: `FALLBACK_ENCRYPTED  // Encrypted file (last resort)`
- Line 140: `VGREResult initFallbackEncrypted();`
- Line 147: `VGREResult storeFallbackEncrypted(const std::string& service, const std::string& token);`
- Line 154: `VGREResult getFallbackEncrypted(const std::string& service, std::string& outToken);`
- Line 161: `VGREResult deleteFallbackEncrypted(const std::string& service);`
- Line 163: `// Authenticated encryption for fallback storage (PBKDF2 + AES-256-CTR + HMAC-SHA256)`
- Line 171: `std::string fallback_path_;`

## ./include/vgre/advanced/tcp_cluster.h
### windows_hardcoding
- Line 2: `// sockets.h must come first on Windows: it includes <winsock2.h> which`
- Line 3: `// must precede <windows.h>, and also defines ERROR_* macros via winerror.h.`

## ./include/vgre/advanced/tcp_cluster/internal/configuration_manager.h
### basic_logic
- Line 54: `bool allow_auth_fallback = true;`

## ./include/vgre/advanced/tcp_cluster/internal/interfaces.h
### stubs_and_mocks
- Line 18: `* manager, allowing for mock implementations in unit tests and fake`
- Line 55: `* manager, allowing for mock implementations that can simulate different memory`
- Line 84: `* manager, allowing for mock implementations that can simulate different`

## ./include/vgre/advanced/tcp_cluster/internal/windows_socket_manager.h
### windows_hardcoding
- Line 17: `#include <winsock2.h>`

## ./include/vgre/advanced/tcp_cluster_protocol.h
### basic_logic
- Line 288: `// Worker echoes back BandwidthAckPacket so master can measure round-trip.`
- Line 290: `struct BandwidthAckPacket {`
- Line 296: `// Cross-platform validation for BandwidthAckPacket`
- Line 297: `static_assert(detail::validate_struct_size<BandwidthAckPacket>(16),`
- Line 298: `"BandwidthAckPacket must be exactly 16 bytes across all platforms");`

## ./include/vgre/api/cublaslt_shim.h
### basic_logic
- Line 156: `// ── Matmul preference (heuristic) ────────────────────────────────────────────`
- Line 170: `struct cublasLtMatmulHeuristicResult_t {`
- Line 188: `// ── Algorithm heuristic ──────────────────────────────────────────────────────`
- Line 189: `// Singular form: fills heuristicResultsArray[0..requestedAlgoCount-1].`
- Line 190: `cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,`
- Line 198: `cublasLtMatmulHeuristicResult_t *heuristicResultsArray,`
- Line 202: `cublasStatus_t cublasLtMatmulAlgoGetHeuristics(cublasLtHandle_t lightHandle,`
- Line 210: `cublasLtMatmulHeuristicResult_t *heuristicResultsArray,`

## ./include/vgre/api/cuda_interceptor.h
### stubs_and_mocks
- Line 35: `constexpr cudaError_t cudaErrorStubLibrary = 38;`

## ./include/vgre/common/error_codes.h
### windows_hardcoding
- Line 33: `// On Windows, <winerror.h> (pulled in via <winsock2.h>) defines many`

## ./include/vgre/common/retry.h
### simulation_delays
- Line 36: `std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));`

## ./include/vgre/common/sockets.h
### simulation_delays
- Line 406: `std::this_thread::sleep_for(std::chrono::milliseconds(1));`
### linux_hardcoding
- Line 17: `#include <sys/socket.h>`
- Line 18: `#include <sys/types.h>`
- Line 19: `#include <unistd.h>`
### windows_hardcoding
- Line 5: `#include <winsock2.h>`
### basic_logic
- Line 346: `// Fallback: IPv4-only.`

## ./include/vgre/common/system_utils.h
### stubs_and_mocks
- Line 56: `static int dummy = 0;`
- Line 57: `if (dladdr((void*)&dummy, &info) && info.dli_fname) {`
### windows_hardcoding
- Line 12: `#include <windows.h>`
### basic_logic
- Line 97: `// 2. Fallback: Search upwards from CWD`
- Line 106: `// 3. Platform-specific absolute fallbacks`
- Line 157: `// Fallback to hoping it's in the system PATH`

## ./include/vgre/common/types.h
### basic_logic
- Line 119: `// set by the heuristic syntax parser.  Used to distinguish "kernel has`

## ./include/vgre/compiler/cpu_cuda_env.h
### basic_logic
- Line 65: `// ── Cooperative Groups & CUB Fallback ───────────────────────────────────────`
- Line 70: `#include "cuda_device_libs/cub_fallback.h"`

## ./include/vgre/compiler/cuda_device_libs/cub_fallback.h
### stubs_and_mocks
- Line 217: `// ── CachingDeviceAllocator stub ────────────────────────────────────────────────`
- Line 218: `// Many CUB-using kernels instantiate this; provide a minimal stub.`
### basic_logic
- Line 1: `#ifndef VGRE_COMPILER_CUDA_DEVICE_LIBS_CUB_FALLBACK_H`
- Line 2: `#define VGRE_COMPILER_CUDA_DEVICE_LIBS_CUB_FALLBACK_H`
- Line 228: `#endif // VGRE_COMPILER_CUDA_DEVICE_LIBS_CUB_FALLBACK_H`

## ./include/vgre/compiler/kernel_parser.h
### basic_logic
- Line 96: `// Cache for token-based fallback results (keyed by kernel source SHA)`
- Line 97: `std::unordered_map<std::string, KernelIR> fallbackCache_;`

## ./include/vgre/compiler/wmma_emulation.h
### basic_logic
- Line 4: `// Tensor Core Emulation via scalar FP32 fallback.`
- Line 231: `// ── Tier 3: scalar fallback ───────────────────────────────────────────────────`
- Line 252: `//   3. Scalar fallback (any hardware, any tile size)`

## ./include/vgre/core/memory_manager.h
### windows_hardcoding
- Line 22: `#include <windows.h>`

## ./include/vgre/core/runtime_engine.h
### basic_logic
- Line 79: `// Look up KernelIR by compiled function pointer (used for iGPU fallback).`

## ./include/vgre/core/scheduler.h
### basic_logic
- Line 176: `static bool enqueueWithWorkerFallback(int workerIdx,`
- Line 185: `std::priority_queue<WorkItem> queue_; // Global ready queue (work-stealing fallback)`

## ./src/advanced/adaptive_execution_engine.cpp
### simulation_delays
- Line 442: `std::this_thread::sleep_for(std::chrono::seconds(5));`
### linux_hardcoding
- Line 24: `#include <unistd.h>`
### windows_hardcoding
- Line 34: `#include <windows.h>`
### basic_logic
- Line 359: `// Fallback: optional helper writes temperature to /tmp/.vgre_cpu_temp`

## ./src/advanced/adaptive_execution_engine_record.cpp
### linux_hardcoding
- Line 24: `#include <unistd.h>`
### windows_hardcoding
- Line 34: `#include <windows.h>`

## ./src/advanced/adaptive_execution_engine_tune.cpp
### linux_hardcoding
- Line 24: `#include <unistd.h>`
### windows_hardcoding
- Line 34: `#include <windows.h>`
### basic_logic
- Line 300: `// Fallback: use RDTSC to estimate instruction count via assumed IPC.`

## ./src/advanced/gpu_passthrough.cpp
### windows_hardcoding
- Line 9: `#  include <windows.h>`

## ./src/advanced/grpc_transport.cpp
### stubs_and_mocks
- Line 4: `// as empty stubs so the rest of the codebase can reference them unconditionally.`
- Line 255: `auto* stub = vgre::cluster::VGRECluster::NewStub(channel).release();`
- Line 256: `stubPtr_ = stub;`
- Line 260: `delete reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_);`
- Line 263: `bool VGREGRPCClient::isConnected() const { return stubPtr_ != nullptr; }`
- Line 270: `auto st = reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_)`
- Line 280: `auto st = reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_)`
- Line 293: `auto st = reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_)`
- Line 305: `auto st = reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_)`
- Line 328: `auto st = reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_)`
- Line 338: `auto st = reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_)`
- Line 347: `auto st = reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_)`
- Line 359: `// ── Stub implementations when gRPC is disabled ───────────────────────────────`

## ./src/advanced/hybrid_compute_manager.cpp
### linux_hardcoding
- Line 28: `#include <unistd.h>`
- Line 31: `#include <sys/socket.h>`
- Line 32: `#include <unistd.h>`
### windows_hardcoding
- Line 23: `#include <winsock2.h>`
- Line 25: `#include <windows.h>`
### basic_logic
- Line 197: `// Fallback: PCI class code — 0x030200 = 3D Controller (APU pattern)`

## ./src/advanced/hybrid_compute_manager_remote.cpp
### linux_hardcoding
- Line 28: `#include <unistd.h>`
- Line 31: `#include <sys/socket.h>`
- Line 32: `#include <unistd.h>`
### windows_hardcoding
- Line 23: `#include <winsock2.h>`
- Line 25: `#include <windows.h>`

## ./src/advanced/hybrid_compute_manager_workload.cpp
### simulation_delays
- Line 321: `std::this_thread::sleep_for(std::chrono::milliseconds(100));`
### linux_hardcoding
- Line 28: `#include <unistd.h>`
- Line 31: `#include <sys/socket.h>`
- Line 32: `#include <unistd.h>`
### windows_hardcoding
- Line 23: `#include <winsock2.h>`
- Line 25: `#include <windows.h>`
### basic_logic
- Line 108: `// Graceful fallback to CPU when iGPU path unavailable`

## ./src/advanced/ipc_manager.cpp
### simulation_delays
- Line 83: `std::this_thread::yield();`
### linux_hardcoding
- Line 16: `#include <unistd.h>`
### windows_hardcoding
- Line 12: `#include <windows.h>`

## ./src/advanced/memory_compression.cpp
### basic_logic
- Line 154: `// Legacy fallback if signature fails (direct LZ4 decompress attempt)`

## ./src/advanced/mps_control.cpp
### linux_hardcoding
- Line 29: `#include <sys/socket.h>`
- Line 31: `#include <unistd.h>`
### windows_hardcoding
- Line 27: `#include <windows.h>`

## ./src/advanced/rdma_transport.cpp
### simulation_delays
- Line 379: `// low on fast NICs), then yield (avoids starving other threads on`
- Line 385: `__builtin_ia32_pause();`
- Line 387: `asm volatile("yield" ::: "memory");`
- Line 392: `std::this_thread::yield();`
### basic_logic
- Line 78: `VGRE_LOG_INFO("RDMATransport", "No RDMA devices found — TCP fallback active");`
- Line 91: `VGRE_LOG_WARN("RDMATransport", "Failed to open RDMA device — TCP fallback active");`
- Line 99: `VGRE_LOG_WARN("RDMATransport", "ibv_alloc_pd failed — TCP fallback active");`
- Line 111: `VGRE_LOG_WARN("RDMATransport", "ibv_create_cq failed — TCP fallback active");`
- Line 467: `// Fallback implementations when RDMA support is not compiled in.`

## ./src/advanced/resource_ledger.cpp
### linux_hardcoding
- Line 19: `#include <sys/types.h>`
- Line 20: `#include <unistd.h>`
### windows_hardcoding
- Line 14: `#include <windows.h>`

## ./src/advanced/runtime_profiler.cpp
### linux_hardcoding
- Line 27: `#  include <sys/socket.h>`
- Line 29: `#  include <unistd.h>`
### windows_hardcoding
- Line 20: `#  include <winsock2.h>`

## ./src/advanced/secure_channel.cpp
### linux_hardcoding
- Line 25: `#include <unistd.h>`
- Line 32: `#include <unistd.h>`

## ./src/advanced/secure_channel_crypto.cpp
### simulation_delays
- Line 375: `// Running 4 independent pipelines keeps all AES execution slots busy, yielding`
### linux_hardcoding
- Line 22: `#include <unistd.h>`
- Line 29: `#include <unistd.h>`
### basic_logic
- Line 337: `// Fallback: /dev/urandom for kernels < 3.17`
- Line 642: `// Software fallback (portable — all platforms, all ISAs)`

## ./src/advanced/tcp_cluster/client_loop.cpp
### simulation_delays
- Line 67: `std::this_thread::sleep_for(std::chrono::milliseconds(100));`
- Line 139: `std::this_thread::sleep_for(std::chrono::milliseconds(1));`
- Line 391: `else { std::this_thread::sleep_for(std::chrono::milliseconds(10)); break; }`
- Line 404: `else { std::this_thread::sleep_for(std::chrono::milliseconds(10)); break; }`
- Line 514: `std::this_thread::sleep_for(std::chrono::milliseconds(1000));`

## ./src/advanced/tcp_cluster/client_packet_dispatch.cpp
### basic_logic
- Line 298: `BandwidthAckPacket ack{};`

## ./src/advanced/tcp_cluster/collective_ops_manager.cpp
### windows_hardcoding
- Line 14: `#include <windows.h>`

## ./src/advanced/tcp_cluster/configuration_manager_backup.cpp
### linux_hardcoding
- Line 20: `#include <unistd.h>`
### windows_hardcoding
- Line 16: `#include <windows.h>`

## ./src/advanced/tcp_cluster/configuration_manager_core.cpp
### basic_logic
- Line 120: `// Production: auth fallback is not supported.`
- Line 223: `config.connection_retry_delay_ms = 2000; config.allow_auth_fallback = false;`
- Line 233: `config.connection_retry_delay_ms = 1500; config.allow_auth_fallback = false;`
- Line 243: `config.connection_retry_delay_ms = 1000; config.allow_auth_fallback = false;`
- Line 278: `emit("allow_auth_fallback", old_config.allow_auth_fallback ? "true" : "false", new_config.allow_auth`

## ./src/advanced/tcp_cluster/configuration_manager_documentation.cpp
### linux_hardcoding
- Line 11: `#include <unistd.h>`
### basic_logic
- Line 27: `j << "  \"allow_auth_fallback\": " << (config.allow_auth_fallback ? "true" : "false") << ",\n";`
- Line 60: `y << "allow_auth_fallback: " << (c.allow_auth_fallback ? "true" : "false") << "\n";`
- Line 92: `// Production: auth fallback is not supported.`
- Line 136: `md << "| `allow_auth_fallback` | " << (config.allow_auth_fallback ? "true" : "false") << " | Allow a`

## ./src/advanced/tcp_cluster/configuration_manager_file_io.cpp
### windows_hardcoding
- Line 10: `#include <windows.h>`
### basic_logic
- Line 131: `pos = 0; if (findKey("allow_auth_fallback")) config.allow_auth_fallback = parseBool();`
- Line 195: `else if (key == "allow_auth_fallback")      config.allow_auth_fallback = (value == "true" || value =`

## ./src/advanced/tcp_cluster/configuration_manager_monitoring.cpp
### simulation_delays
- Line 94: `std::this_thread::sleep_for(std::chrono::milliseconds(check_interval_ms));`
- Line 97: `std::this_thread::sleep_for(std::chrono::milliseconds(check_interval_ms * 2)); // Back off on error`
### basic_logic
- Line 54: `new_config.allow_auth_fallback = env_config.allow_auth_fallback;`

## ./src/advanced/tcp_cluster/configuration_manager_validation.cpp
### linux_hardcoding
- Line 20: `#include <unistd.h>`
### windows_hardcoding
- Line 18: `#include <windows.h>`
### basic_logic
- Line 157: `if (config.allow_auth_fallback && config.auth_token.empty() && config.auth_token_file.empty()) {`
- Line 158: `result.warnings.push_back("Auth fallback requested but is not supported in production builds");`

## ./src/advanced/tcp_cluster/discovery_loops_proactive.cpp
### simulation_delays
- Line 289: `std::this_thread::sleep_for(std::chrono::milliseconds(50));`
### linux_hardcoding
- Line 30: `#include <sys/socket.h>`
### windows_hardcoding
- Line 25: `#include <winsock2.h>`

## ./src/advanced/tcp_cluster/discovery_loops_udp.cpp
### simulation_delays
- Line 106: `std::this_thread::sleep_for(std::chrono::seconds(2));`
- Line 123: `std::this_thread::sleep_for(std::chrono::milliseconds(100));`
- Line 219: `std::this_thread::sleep_for(std::chrono::milliseconds(250));`
### linux_hardcoding
- Line 33: `#include <sys/socket.h>`
### windows_hardcoding
- Line 28: `#include <winsock2.h>`

## ./src/advanced/tcp_cluster/discovery_manager.cpp
### simulation_delays
- Line 181: `std::this_thread::sleep_for(std::chrono::seconds(2));`
- Line 314: `std::this_thread::sleep_for(std::chrono::milliseconds(200));`

## ./src/advanced/tcp_cluster/dispatch_manager_remote.cpp
### basic_logic
- Line 159: `// 3. CPU JIT fallback (always available)`

## ./src/advanced/tcp_cluster/interfaces.cpp
### linux_hardcoding
- Line 27: `#include <sys/socket.h>`
- Line 28: `#include <unistd.h>`
### windows_hardcoding
- Line 25: `#include <winsock2.h>`

## ./src/advanced/tcp_cluster/memory_sync_manager.cpp
### simulation_delays
- Line 188: `std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));`
### basic_logic
- Line 121: `// Full sync (first time or delta sync fallback)`
- Line 288: `VGRE_LOG_WARN("TCPCluster", "RDMA failed — TCP fallback for this delta range");`
- Line 291: `// TCP Fallback`
- Line 360: `VGRE_LOG_WARN("TCPCluster", "RDMA failed mid-stream — TCP fallback for full sync");`

## ./src/advanced/tcp_cluster/mesh_topology_impl.cpp
### basic_logic
- Line 88: `// Fallback if packet_handler is not initialized`

## ./src/advanced/tcp_cluster/packet_handler.cpp
### basic_logic
- Line 86: `// Fallback to direct send without encryption`

## ./src/advanced/tcp_cluster/security_manager.cpp
### basic_logic
- Line 52: `// Production policy: strict authentication only. No runtime fallback modes.`
- Line 153: `return std::string("Authentication mode: ") + (strict_auth_mode_ ? "STRICT" : "FALLBACK") + " (" + (`

## ./src/advanced/tcp_cluster/server_loop_connection_handling.cpp
### windows_hardcoding
- Line 17: `#include <winsock2.h>`

## ./src/advanced/tcp_cluster/server_packet_dispatch.cpp
### basic_logic
- Line 421: `if (hdr.payloadSize < sizeof(BandwidthAckPacket)) {`
- Line 425: `BandwidthAckPacket ack;`
- Line 426: `std::memcpy(&ack, payload, sizeof(BandwidthAckPacket));`

## ./src/advanced/tcp_cluster/tcp_cluster_init.cpp
### basic_logic
- Line 99: `discovery_manager_->startWorkerDiscovery(); // Start UDP discovery as fallback`

## ./src/advanced/tcp_cluster/tcp_cluster_manager.cpp
### basic_logic
- Line 262: `// Fallback: any active worker (GPU-unaware dispatch is still valid)`

## ./src/advanced/tcp_cluster/tcp_cluster_restart.cpp
### simulation_delays
- Line 30: `std::this_thread::sleep_for(std::chrono::milliseconds(1000));`
- Line 62: `if (delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));`

## ./src/advanced/tcp_cluster/tcp_cluster_transport.cpp
### basic_logic
- Line 169: `// Direct send fallback (for non-queued scenarios like handshake packets)`
- Line 179: `// Fallback to direct send without encryption`
- Line 236: `// Fallback to direct send without encryption`

## ./src/advanced/tcp_cluster/windows_socket_manager_errors.cpp
### windows_hardcoding
- Line 11: `#include <winsock2.h>`

## ./src/advanced/tcp_cluster/windows_socket_manager_lifecycle.cpp
### simulation_delays
- Line 61: `std::this_thread::sleep_for(std::chrono::milliseconds(500));`
### windows_hardcoding
- Line 13: `#include <winsock2.h>`
### basic_logic
- Line 68: `int fallback_result = WSAStartup(MAKEWORD(2, 0), &wsa_data_);`
- Line 69: `if (fallback_result == 0) {`

## ./src/advanced/tcp_cluster/windows_socket_manager_recovery.cpp
### simulation_delays
- Line 29: `std::this_thread::sleep_for(std::chrono::milliseconds(100 * (1 << (attempt - 2))));`
### windows_hardcoding
- Line 13: `#include <winsock2.h>`

## ./src/advanced/token/hardware_token_manager.cpp
### basic_logic
- Line 15: `fallback_path_("")`
- Line 57: `if (std::string(name) == "file"      && initFallbackEncrypted() == VGREResult::SUCCESS) { backend_ =`
- Line 112: `if (initFallbackEncrypted() == VGREResult::SUCCESS) {`
- Line 113: `backend_ = BackendType::FALLBACK_ENCRYPTED;`
- Line 115: `VGRE_LOG_WARN("HardwareTokenManager", "Using encrypted file fallback - not recommended for productio`
- Line 146: `case BackendType::FALLBACK_ENCRYPTED:`
- Line 147: `return storeFallbackEncrypted(service, token);`
- Line 162: `const BackendType fallbackOrder[] = {`
- Line 165: `BackendType::FALLBACK_ENCRYPTED`
- Line 167: `for (BackendType candidate : fallbackOrder) {`
- Line 173: `case BackendType::FALLBACK_ENCRYPTED: initRes = initFallbackEncrypted(); break;`
- Line 212: `case BackendType::FALLBACK_ENCRYPTED:`
- Line 213: `return getFallbackEncrypted(service, outToken);`
- Line 225: `const BackendType fallbackOrder[] = {`
- Line 228: `BackendType::FALLBACK_ENCRYPTED`
- Line 230: `for (BackendType candidate : fallbackOrder) {`
- Line 236: `case BackendType::FALLBACK_ENCRYPTED: initRes = initFallbackEncrypted(); break;`
- Line 275: `case BackendType::FALLBACK_ENCRYPTED:`
- Line 276: `return deleteFallbackEncrypted(service);`
- Line 288: `const BackendType fallbackOrder[] = {`
- Line 291: `BackendType::FALLBACK_ENCRYPTED`
- Line 293: `for (BackendType candidate : fallbackOrder) {`
- Line 299: `case BackendType::FALLBACK_ENCRYPTED: initRes = initFallbackEncrypted(); break;`
- Line 327: `case BackendType::FALLBACK_ENCRYPTED: return "Encrypted File (Fallback)";`

## ./src/advanced/token/token_manager_fallback.cpp
### linux_hardcoding
- Line 15: `#include <unistd.h>`
### windows_hardcoding
- Line 20: `#include <windows.h>`
### basic_logic
- Line 26: `VGREResult HardwareTokenManager::initFallbackEncrypted() {`
- Line 27: `const char* overridePath = vgre_get_config("VGRE_TOKEN_FALLBACK_PATH");`
- Line 29: `fallback_path_ = overridePath;`
- Line 34: `fallback_path_ = std::string(appdata) + "\\vgre\\tokens.enc";`
- Line 36: `fallback_path_ = ".\\vgre\\tokens.enc";`
- Line 41: `fallback_path_ = std::string(home) + "/.vgre/tokens.enc";`
- Line 43: `// CI/sandbox fallback when HOME is unset.`
- Line 44: `fallback_path_ = ".vgre/tokens.enc";`
- Line 49: `std::string dir = fallback_path_.substr(0, fallback_path_.find_last_of("/\\"));`
- Line 63: `VGREResult HardwareTokenManager::storeFallbackEncrypted(const std::string& service, const std::strin`
- Line 65: `std::ifstream infile(fallback_path_);`
- Line 81: `std::ofstream outfile(fallback_path_);`
- Line 93: `chmod(fallback_path_.c_str(), 0600);`
- Line 99: `VGREResult HardwareTokenManager::getFallbackEncrypted(const std::string& service, std::string& outTo`
- Line 100: `std::ifstream infile(fallback_path_);`
- Line 123: `VGREResult HardwareTokenManager::deleteFallbackEncrypted(const std::string& service) {`
- Line 125: `std::ifstream infile(fallback_path_);`
- Line 150: `std::ofstream outfile(fallback_path_);`
- Line 211: `std::string identity = "vgre_fallback_kdf";`
- Line 228: `std::string saltInput = "vgre_fallback_v2:" + identity;`
- Line 235: `reinterpret_cast<const uint8_t*>("vgre_fallback_kdf"), 17,`

## ./src/advanced/token/token_manager_linux.cpp
### linux_hardcoding
- Line 5: `#include <sys/types.h>`
- Line 7: `#include <unistd.h>`

## ./src/advanced/token/token_manager_win32.cpp
### windows_hardcoding
- Line 5: `#include <windows.h>`

## ./src/advanced/vgre_worker_cli.cpp
### simulation_delays
- Line 153: `std::this_thread::sleep_for(std::chrono::milliseconds(200));`
- Line 161: `std::this_thread::sleep_for(std::chrono::seconds(2));`
### windows_hardcoding
- Line 19: `#include <windows.h>`

## ./src/advanced/vgre_workload_engine.cpp
### simulation_delays
- Line 165: `std::this_thread::sleep_until(nextTarget);`

## ./src/advanced/websocket_transport.cpp
### linux_hardcoding
- Line 24: `#include <unistd.h>`
- Line 27: `#include <sys/socket.h>`
### windows_hardcoding
- Line 21: `#include <winsock2.h>`
### basic_logic
- Line 293: `return true; // fallback to plain`

## ./src/api/cublaslt/cublaslt_core.cpp
### basic_logic
- Line 1: `// cuBLASLt emulation shim — descriptors, layout management, algorithm heuristics.`
- Line 267: `// ── Algorithm heuristics ─────────────────────────────────────────────────────`
- Line 269: `cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t /*handle*/,`
- Line 277: `cublasLtMatmulHeuristicResult_t *heuristicResultsArray,`
- Line 279: `if (!heuristicResultsArray || !returnAlgoCount || requestedAlgoCount <= 0)`
- Line 305: `heuristicResultsArray[0].algo          = 0;`
- Line 306: `heuristicResultsArray[0].workspaceSize = 0;`
- Line 307: `heuristicResultsArray[0].state         = CUBLAS_STATUS_SUCCESS;`
- Line 308: `heuristicResultsArray[0].wavesCount    = 1.0f;`
- Line 314: `cublasStatus_t cublasLtMatmulAlgoGetHeuristics(cublasLtHandle_t handle,`
- Line 322: `cublasLtMatmulHeuristicResult_t *heuristicResultsArray,`
- Line 324: `return cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, Adesc, Bdesc,`
- Line 326: `heuristicResultsArray, returnAlgoCount);`

## ./src/api/cuda_driver/cuda_driver_external.cpp
### linux_hardcoding
- Line 18: `#include <unistd.h>`

## ./src/api/cuda_driver/cuda_driver_module.cpp
### stubs_and_mocks
- Line 191: `// VGRE does not have a CUDA linker; we provide stubs that collect PTX data`
### basic_logic
- Line 27: `if (len == 0) len = 8 * 1024 * 1024; // Fallback to conservative estimate`

## ./src/api/cuda_driver/cuda_driver_occupancy.cpp
### basic_logic
- Line 7: `// VGRE emulates a single virtual GPU.  Occupancy is a heuristic based on`
- Line 26: `// Heuristic: each "SM" can hold ~2048 threads (Ampere-like).`
- Line 31: `// Shared memory limitation heuristic: 48 KB per block default.`

## ./src/api/cuda_driver/cuda_driver_texture.cpp
### basic_logic
- Line 310: `// Authoritative fallback: return a fresh one if it doesn't exist in the module`

## ./src/api/cuda_external_semaphore.cpp
### linux_hardcoding
- Line 24: `#include <unistd.h>`
### windows_hardcoding
- Line 28: `#include <windows.h>`

## ./src/api/cuda_interceptor.cpp
### stubs_and_mocks
- Line 666: `case cudaErrorStubLibrary:`
- Line 667: `return "CUDA runtime is a stub library";`
- Line 799: `case cudaErrorStubLibrary:`
- Line 800: `return "cudaErrorStubLibrary";`

## ./src/api/cuda_virtual_memory.cpp
### linux_hardcoding
- Line 25: `#include <unistd.h>`
### windows_hardcoding
- Line 29: `#include <windows.h>`

## ./src/api/cudart/cudart_cooperative.cpp
### stubs_and_mocks
- Line 147: `// Path: host stub pointer → device name → KernelId → KernelIR`

## ./src/api/cudart/cudart_shim.cpp
### basic_logic
- Line 67: `// Fallback: Scan for PTX signatures (.version, .target).`
- Line 154: `// Fallback: Scan for global variables if it's raw PTX`
- Line 630: `// Pool was never created — fallback to regular free`
- Line 636: `// Fallback: regular free`

## ./src/api/cudart/cudart_shim_graph_nodes.cpp
### stubs_and_mocks
- Line 4: `* P1.10  cudaGraphAddKernelNode        — resolves host stub → KernelId`

## ./src/api/cudnn/cudnn_internal.h
### stubs_and_mocks
- Line 46: `// ── cuDNN type stubs (no cudnn.h needed) ─────────────────────────────────────`

## ./src/api/cusolver/cusolver_core.cpp
### stubs_and_mocks
- Line 186: `std::vector<float> dummy_a(static_cast<size_t>(ld_a) * n);`
- Line 187: `std::vector<float> dummy_b(static_cast<size_t>(ld_b) * std::max(1, nrhs));`
- Line 188: `std::vector<float> dummy_s(std::min(m, n));`
- Line 189: `sgelsd_(&m, &n, &nrhs, dummy_a.data(), &ld_a, dummy_b.data(), &ld_b,`
- Line 190: `dummy_s.data(), &rcond_tmp, &rank_tmp, &work_query, &query_lwork, &iwork_query, &info);`
- Line 204: `std::vector<double> dummy_a(static_cast<size_t>(ld_a) * n);`
- Line 205: `std::vector<double> dummy_b(static_cast<size_t>(ld_b) * std::max(1, nrhs));`
- Line 206: `std::vector<double> dummy_s(std::min(m, n));`
- Line 207: `dgelsd_(&m, &n, &nrhs, dummy_a.data(), &ld_a, dummy_b.data(), &ld_b,`
- Line 208: `dummy_s.data(), &rcond_tmp, &rank_tmp, &work_query, &query_lwork, &iwork_query, &info);`
- Line 363: `std::vector<float> dummy_a(static_cast<size_t>(m) * n), dummy_s(std::min(m,n));`
- Line 364: `std::vector<float> dummy_u(static_cast<size_t>(m) * m), dummy_vt(static_cast<size_t>(n) * n);`
- Line 366: `sgesvd_(&jobu, &jobvt, &m, &n, dummy_a.data(), &m, dummy_s.data(),`
- Line 367: `dummy_u.data(), &m, dummy_vt.data(), &n, &work_query, &query_lwork, &info);`
- Line 375: `std::vector<double> dummy_a(static_cast<size_t>(m) * n), dummy_s(std::min(m,n));`
- Line 376: `std::vector<double> dummy_u(static_cast<size_t>(m) * m), dummy_vt(static_cast<size_t>(n) * n);`
- Line 378: `dgesvd_(&jobu, &jobvt, &m, &n, dummy_a.data(), &m, dummy_s.data(),`
- Line 379: `dummy_u.data(), &m, dummy_vt.data(), &n, &work_query, &query_lwork, &info);`

## ./src/api/cusolver/lapack_fallback.cpp
### basic_logic
- Line 2: `* VGRE built-in LAPACK fallback — portable CPU implementations.`
- Line 4: `* Compiled only when VGRE_LAPACK_FALLBACK is defined (no system LAPACK).`
- Line 16: `#ifdef VGRE_LAPACK_FALLBACK`
- Line 566: `#endif // VGRE_LAPACK_FALLBACK`

## ./src/api/nccl/nccl_communicator.cpp
### linux_hardcoding
- Line 5: `#include <unistd.h>`
### basic_logic
- Line 30: `// Fallback: fill from /dev/urandom`

## ./src/api/nccl/nccl_core.cpp
### linux_hardcoding
- Line 8: `#  include <unistd.h>`
### windows_hardcoding
- Line 5: `#  include <windows.h>`
### basic_logic
- Line 121: `// Byte-level fallback for int8/uint8 sum`

## ./src/api/opencl_adapter.cpp
### linux_hardcoding
- Line 18: `#include <unistd.h>`
### windows_hardcoding
- Line 16: `#include <windows.h>`

## ./src/compiler/clang_kernel_analysis.cpp
### linux_hardcoding
- Line 18: `#include <unistd.h>`

## ./src/compiler/clang_kernel_parser.cpp
### stubs_and_mocks
- Line 26: `// ── Minimal CUDA stub for AST-only analysis ─────────────────────────────────`
- Line 32: `// stub provides only the declarations needed to parse typical CUDA kernel`
- Line 34: `static constexpr const char kAstAnalysisStub[] = R"VGRE_STUB(`
- Line 36: `// This stub provides only what is needed to parse kernel AST without OOM.`
- Line 112: `// Minimal half-precision stubs (avoid heavy cuda_fp16.h)`
- Line 116: `// Cooperative groups stub — provides just enough for AST parsing.`
- Line 159: `)VGRE_STUB";`
- Line 570: `// Use the minimal analysis stub (NOT the full cpu_cuda_env.h) so that the`
- Line 572: `std::string sourceWithHeader = std::string(kAstAnalysisStub) + source;`
- Line 768: `std::string sourceWithHeader = std::string(kAstAnalysisStub) + source;`
### linux_hardcoding
- Line 18: `#include <unistd.h>`
### basic_logic
- Line 42: `#define VGRE_COMPILER_CUDA_DEVICE_LIBS_CUB_FALLBACK_H`
- Line 674: `// Fallback: If requested name fails, try finding ANY global kernel`
- Line 923: `// Fallback if no bytes calculated`

## ./src/compiler/kernel_cache.cpp
### linux_hardcoding
- Line 15: `#include <unistd.h>`
### windows_hardcoding
- Line 12: `#include <windows.h>`

## ./src/compiler/kernel_parser.cpp
### stubs_and_mocks
- Line 607: `KernelIR dummyIR;`
- Line 609: `// Create a dummy kernel that uses the struct to trigger AST analysis`
- Line 613: `VGREResult result = clangParser.parse("__vgre_struct_test__", testSource, dummyIR);`
- Line 617: `for (size_t i = 0; i < dummyIR.argTypes.size(); i++) {`
- Line 618: `if (dummyIR.argTypes[i] == ArgType::STRUCT && dummyIR.argSizes[i] > 0) {`
- Line 621: `"' size: " + std::to_string(dummyIR.argSizes[i]) + " bytes");`
- Line 622: `return dummyIR.argSizes[i];`
### basic_logic
- Line 534: `// This replaces the old heuristic token-counting approach`
- Line 550: `// in production. Reject the kernel rather than using unreliable heuristics.`

## ./src/compiler/llvm_translation_codegen.cpp
### simulation_delays
- Line 808: `VGRE_LOG_DEBUG("LLVMTranslationEngine", "JIT Instruction Recalibration yielded 0 (empty kernel or un`

## ./src/core/event.cpp
### basic_logic
- Line 34: `task(); // Execute synchronously as fallback`

## ./src/core/graph_optimizer.cpp
### basic_logic
- Line 194: `// Fallback to conservative limit if properties lookup fails.`

## ./src/core/memory/memory_manager.cpp
### simulation_delays
- Line 144: `// short yields to let signal handlers run`
- Line 145: `std::this_thread::yield();`
- Line 149: `std::this_thread::sleep_for(milliseconds(sleepMs));`
- Line 810: `std::this_thread::sleep_for(1ms);`
### linux_hardcoding
- Line 23: `#include <unistd.h>`
### windows_hardcoding
- Line 19: `#include <windows.h>`
### basic_logic
- Line 258: `if (si->si_code != SEGV_ACCERR) goto fallback;`
- Line 266: `goto fallback;`
- Line 295: `// migration heuristics. t_currentDevice is a thread-local set by the`
- Line 323: `// Fallback: enqueue for background drainer`
- Line 338: `fallback:`

## ./src/core/memory/memory_manager_copy.cpp
### simulation_delays
- Line 301: `// We no longer artificially throttle transfers with sleep_for.`

## ./src/core/memory/memory_manager_managed.cpp
### stubs_and_mocks
- Line 127: `char dummy = p[i];`
- Line 128: `(void)dummy;`
- Line 131: `char dummyLast = p[count - 1];`
- Line 132: `(void)dummyLast;`
### basic_logic
- Line 221: `nodes = 4; // Fallback for complex sparse masks`

## ./src/core/memory/pool_allocator.cpp
### windows_hardcoding
- Line 8: `#include <windows.h>`

## ./src/core/memory/uvm_migration.cpp
### simulation_delays
- Line 67: `std::this_thread::sleep_for(kInterval);`
- Line 207: `std::this_thread::sleep_for(1ms);`

## ./src/core/runtime_engine.cpp
### windows_hardcoding
- Line 31: `#include <winsock2.h>`

## ./src/core/runtime_engine_graph.cpp
### basic_logic
- Line 164: `// Fallback: single body exec (behaves like IF with integer condition)`

## ./src/core/runtime_engine_launch.cpp
### simulation_delays
- Line 553: `// avoids the yield()-spin that wastes CPU between thread creation and launch.`
### basic_logic
- Line 172: `// Authoritative fallback based on type`
- Line 302: `// calculations don't divide by zero; do NOT apply the 30% heuristic.`
- Line 328: `(fpi > 0.0 ? " (ratio=" + std::to_string(fpi) + ")" : " (fallback)"));`

## ./src/core/scheduler.cpp
### basic_logic
- Line 12: `bool Scheduler::enqueueWithWorkerFallback(int workerIdx,`

## ./src/core/scheduler_numa.cpp
### linux_hardcoding
- Line 10: `#include <pthread.h>`
- Line 16: `#include <pthread.h>`
### windows_hardcoding
- Line 13: `#include <windows.h>`

## ./src/core/scheduler_tasks.cpp
### basic_logic
- Line 108: `Scheduler::enqueueWithWorkerFallback(t_workerIdx, std::move(item), workerDeques_, queue_, mutex_);`
- Line 138: `Scheduler::enqueueWithWorkerFallback(t_workerIdx, std::move(item), workerDeques_, queue_, mutex_);`
- Line 186: `Scheduler::enqueueWithWorkerFallback(t_workerIdx, std::move(item), workerDeques_, queue_, mutex_);`

## ./src/core/scheduler_worker.cpp
### linux_hardcoding
- Line 8: `#include <pthread.h>`
- Line 13: `#include <pthread.h>`
### windows_hardcoding
- Line 11: `#include <windows.h>`
### basic_logic
- Line 56: `// 3. Fallback: wait on global / NUMA-local priority queue`

## ./src/core/shm_manager.cpp
### linux_hardcoding
- Line 13: `#include <unistd.h>`
### windows_hardcoding
- Line 8: `#include <windows.h>`

## ./src/core/virtual_gpu_device.cpp
### linux_hardcoding
- Line 16: `#include <sys/types.h>`
- Line 19: `#include <sys/types.h>`
- Line 20: `#include <unistd.h>`
### windows_hardcoding
- Line 12: `#include <windows.h>`
### basic_logic
- Line 195: `// This avoids non-authoritative "multiplier" heuristics while protecting host stability.`
- Line 239: `// Fallback synthetic PCI on Linux if no VGA class found (e.g. headless server)`
- Line 265: `// Fallback: read nominal CPU MHz from registry (always present on Windows).`
- Line 359: `// Fallback for non-gcc compilers on x86`

## ./src/runtime/block_worker_pool.cpp
### simulation_delays
- Line 116: `std::this_thread::yield();`
- Line 154: `// Hybrid wait: short spin for fast-completing kernels, then yield to condvar.`
- Line 163: `__builtin_ia32_pause();`
- Line 224: `__builtin_ia32_pause();`
### linux_hardcoding
- Line 12: `#include <unistd.h>`
- Line 13: `#include <pthread.h>`
### windows_hardcoding
- Line 16: `#include <windows.h>`

## ./src/runtime/cdp_executor.cpp
### stubs_and_mocks
- Line 120: `// ── C interface called from device-side CDP stubs ─────────────────────────────`
- Line 129: `// kernelFn: function pointer as returned by __device_stub__xxx (ignored in VGRE;`
### basic_logic
- Line 80: `// Fallback: over-allocate and align manually`

## ./src/runtime/cpu_parallel_executor.cpp
### basic_logic
- Line 160: `std::string ompSuffix = " (OpenMP NOT available — single-threaded fallback)";`
- Line 302: `// ── No-OpenMP scalar fallback ─────────────────────────────────────────`

## ./src/runtime/igpu_opencl_executor.cpp
### basic_logic
- Line 98: `// Previously searched for CPU fallback here.`
- Line 246: `// Warp Shuffles (Mapping to cl_intel_subgroups if available, or local memory fallback)`
- Line 253: `// Software warp-shuffle fallback using __local memory.`
- Line 344: `// No longer need the atomicAdd_f hack if we use 'overloadable'`

## ./src/runtime/vector_engine.cpp
### linux_hardcoding
- Line 16: `#include <unistd.h>`

## ./src/runtime/vector_engine_double.cpp
### linux_hardcoding
- Line 16: `#include <unistd.h>`

## ./src/runtime/vector_engine_float.cpp
### linux_hardcoding
- Line 16: `#include <unistd.h>`

