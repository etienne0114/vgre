// VGRE MPS (Multi-Process Service) — Unix domain socket server + client.
// Allows multiple processes to share one VGRE RuntimeEngine instance,
// mirroring the functionality of NVIDIA's CUDA MPS control daemon.
//
// Server: listens on a Unix socket; each accepted connection gets its own
//         handler thread that dispatches MALLOC/FREE/MEMCPY/LAUNCH/SYNC to
//         the process-global RuntimeEngine.
//
// Client: a thin shim that converts CUDA API calls to MPS wire messages.
//         Activated when VGRE_MPS_PIPE env-var is set.

#include "vgre/advanced/mps_control.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/common/types.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

#include <atomic>
#include <cstring>
#if !defined(_WIN32)
#include <pwd.h>    // getpwuid
#include <unistd.h> // getuid
#endif
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <sstream>
#include <llvm/Support/JSON.h>

#include "vgre/common/os_backend.h"
#if defined(_WIN32)
#  include <aclapi.h>
#endif
#if !defined(_WIN32)
#include <sys/un.h>  // sockaddr_un — Unix domain sockets
#endif

namespace vgre {
namespace mps {

// ── Shared memory allocation registry ────────────────────────────────────────
// In VGRE's single-address-space model, "device memory" is heap-allocated.
// The MPS server keeps a map from opaque uint64 handle → real pointer so that
// MEMCPY and FREE know where to write/free.
namespace {

static std::mutex& getAllocMutex() {
    static std::mutex m;
    return m;
}
static std::unordered_map<uint64_t, void*>& getAllocMap() {
    static std::unordered_map<uint64_t, void*> m;
    return m;
}

static uint64_t mpsAlloc(uint64_t bytes) {
    vgre::MemoryHandle h = nullptr;
    if (vgre::core::MemoryManager::instance().allocate(bytes, h) != vgre::VGREResult::SUCCESS) {
        return 0;
    }
    uint64_t handle = reinterpret_cast<uint64_t>(h);
    std::lock_guard<std::mutex> lk(getAllocMutex());
    getAllocMap()[handle] = h;
    return handle;
}

static bool mpsFree(uint64_t handle) {
    std::lock_guard<std::mutex> lk(getAllocMutex());
    auto it = getAllocMap().find(handle);
    if (it == getAllocMap().end()) return false;
    vgre::core::MemoryManager::instance().free(it->second);
    getAllocMap().erase(it);
    return true;
}

static void* mpsResolve(uint64_t handle) {
    std::lock_guard<std::mutex> lk(getAllocMutex());
    auto it = getAllocMap().find(handle);
    return it != getAllocMap().end() ? it->second : nullptr;
}

// ── Per-client resource quotas (Track P) ─────────────────────────────────────
// The policy is loaded once from VGRE_MPS_POLICY_FILE; each connected client
// (identified by its slot id) gets a ClientQuota that tracks live device memory
// and bounds in-flight / concurrent kernel launches.

static const VgreMPSPolicy& mpsPolicy() {
    static const VgreMPSPolicy pol = [] {
        const char* path = vgre_get_config("VGRE_MPS_POLICY_FILE");
        return loadMPSPolicy(path ? path : "");
    }();
    return pol;
}

struct ClientQuota {
    std::mutex mtx;
    std::condition_variable cv;
    uint64_t allocatedBytes = 0;  // live device memory held by this client
    uint32_t inFlight       = 0;  // launches accepted but not yet completed
    uint32_t running        = 0;  // launches currently executing (concurrency)
};

static std::mutex& quotaMutex() { static std::mutex m; return m; }
static std::unordered_map<uint32_t, std::shared_ptr<ClientQuota>>& quotaMap() {
    static std::unordered_map<uint32_t, std::shared_ptr<ClientQuota>> m;
    return m;
}
static std::shared_ptr<ClientQuota> clientQuota(uint32_t slot) {
    std::lock_guard<std::mutex> lk(quotaMutex());
    auto& q = quotaMap()[slot];
    if (!q) q = std::make_shared<ClientQuota>();
    return q;
}
static void releaseClientQuota(uint32_t slot) {
    std::lock_guard<std::mutex> lk(quotaMutex());
    quotaMap().erase(slot);
}

// devPtr handle → (owning slot, byte count) so FREE refunds the right client.
static std::mutex& allocOwnerMutex() { static std::mutex m; return m; }
static std::unordered_map<uint64_t, std::pair<uint32_t,uint64_t>>& allocOwnerMap() {
    static std::unordered_map<uint64_t, std::pair<uint32_t,uint64_t>> m;
    return m;
}

// Concurrency cap derived from maxThreadFraction and the host thread count.
static uint32_t mpsMaxConcurrent() {
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    float frac = mpsPolicy().maxThreadFraction;
    if (frac <= 0.0f) frac = 1.0f;          // 0/negative → treat as unlimited
    if (frac > 1.0f)  frac = 1.0f;
    uint32_t n = static_cast<uint32_t>(std::floor(frac * static_cast<float>(hw)));
    return n < 1 ? 1u : n;
}

// Helper: write exactly len bytes to fd; returns false on error.
// Uses platform-safe writes (no SIGPIPE on Unix, no partial writes on Windows).
static bool writeAll(mps_handle_t fd, const void* buf, size_t len) {
#if !defined(_WIN32)
    // Suppress SIGPIPE for this process on first write.  Safe to call multiple times.
    static bool sigpipe_ignored = []() {
        struct sigaction sa{};
        sa.sa_handler = SIG_IGN;
        sigaction(SIGPIPE, &sa, nullptr);
        return true;
    }();
    (void)sigpipe_ignored;
#endif
    const uint8_t* p = reinterpret_cast<const uint8_t*>(buf);
    size_t sent = 0;
    while (sent < len) {
#if defined(_WIN32)
        DWORD written = 0;
        BOOL ok = WriteFile(reinterpret_cast<HANDLE>(fd), p + sent,
                            static_cast<DWORD>(len - sent), &written, nullptr);
        if (!ok || written == 0) return false;
        sent += static_cast<size_t>(written);
#else
        ssize_t n = write(fd, p + sent, len - sent);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
#endif
    }
    return true;
}

// Helper: read exactly len bytes from fd; returns false on error/EOF.
static bool readAll(mps_handle_t fd, void* buf, size_t len) {
    uint8_t* p = reinterpret_cast<uint8_t*>(buf);
    size_t got = 0;
    while (got < len) {
#if defined(_WIN32)
        DWORD rd = 0;
        BOOL ok = ReadFile(reinterpret_cast<HANDLE>(fd), p + got,
                           static_cast<DWORD>(len - got), &rd, nullptr);
        if (!ok || rd == 0) return false;
        got += static_cast<size_t>(rd);
#else
        ssize_t n = read(fd, p + got, len - got);
        if (n <= 0) return false;
        got += static_cast<size_t>(n);
#endif
    }
    return true;
}

} // namespace

// ── MPS policy file loader (Track P) ──────────────────────────────────────────
// JSON shape (all fields optional):
//   { "maxMemoryBytes": 1073741824, "maxThreadFraction": 0.5, "maxQueueDepth": 64 }
VgreMPSPolicy loadMPSPolicy(const std::string& path) {
    VgreMPSPolicy pol;  // defaults
    if (path.empty()) return pol;

    std::ifstream f(path);
    if (!f) {
        VGRE_LOG_WARN("MPS", "VGRE_MPS_POLICY_FILE='" + path +
                              "' could not be opened — using default (unlimited) quotas.");
        return pol;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();

    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(content);
    if (!parsed) {
        llvm::consumeError(parsed.takeError());
        VGRE_LOG_WARN("MPS", "VGRE_MPS_POLICY_FILE='" + path +
                              "' is not valid JSON — using default quotas.");
        return pol;
    }
    const llvm::json::Object* obj = parsed->getAsObject();
    if (!obj) {
        VGRE_LOG_WARN("MPS", "VGRE_MPS_POLICY_FILE='" + path +
                              "' is not a JSON object — using default quotas.");
        return pol;
    }
    if (auto v = obj->getInteger("maxMemoryBytes"); v && *v >= 0)
        pol.maxMemoryBytes = static_cast<uint64_t>(*v);
    if (auto v = obj->getNumber("maxThreadFraction"); v && *v > 0.0)
        pol.maxThreadFraction = static_cast<float>(*v);
    if (auto v = obj->getInteger("maxQueueDepth"); v && *v > 0)
        pol.maxQueueDepth = static_cast<uint32_t>(*v);

    VGRE_LOG_INFO("MPS", "Loaded MPS policy: maxMemoryBytes=" +
        std::to_string(pol.maxMemoryBytes) + " maxThreadFraction=" +
        std::to_string(pol.maxThreadFraction) + " maxQueueDepth=" +
        std::to_string(pol.maxQueueDepth));
    return pol;
}

// ─────────────────────────────────────────────────────────────────────────────
//  MPSServer
// ─────────────────────────────────────────────────────────────────────────────

MPSServer::MPSServer(const std::string& socketPath, int maxClients)
    : socketPath_(socketPath), maxClients_(maxClients) {
    if (socketPath_.empty()) {
        // Prefer an env override, then XDG_RUNTIME_DIR (Linux), then TMPDIR, then /tmp.
        const char* env = ::getenv("VGRE_MPS_SOCKET_PATH");
        if (env && env[0]) {
            socketPath_ = env;
        } else {
#if defined(_WIN32)
            socketPath_ = "\\\\.\\pipe\\vgre_mps";
#else
            const char* runtimeDir = ::getenv("XDG_RUNTIME_DIR");
            if (runtimeDir && runtimeDir[0]) {
                socketPath_ = std::string(runtimeDir) + "/vgre_mps.sock";
            } else {
                socketPath_ = vgre::os::get_temp_dir() + "/vgre_mps.sock";
            }
#endif
        }
    }
}

MPSServer::~MPSServer() {
    stop();
}

#if defined(_WIN32)
// Create one secure named-pipe instance with a CREATOR_OWNER + SYSTEM DACL and
// PIPE_UNLIMITED_INSTANCES so every concurrent client gets its own instance with
// the same ACL. Called for the initial listen instance (start) and for each
// subsequent per-client instance (acceptLoop) — previously acceptLoop re-created
// the pipe with a NULL DACL, silently dropping the security descriptor.
mps_handle_t MPSServer::createSecurePipeInstance(const std::string& pipeName) {
    PSECURITY_DESCRIPTOR pSD = nullptr;
    PACL pACL = nullptr;
    PSID pOwnerSid = nullptr, pSystemSid = nullptr;
    EXPLICIT_ACCESS_W ea[2] = {};
    SID_IDENTIFIER_AUTHORITY creatorAuth = SECURITY_CREATOR_SID_AUTHORITY;
    SID_IDENTIFIER_AUTHORITY ntAuth     = SECURITY_NT_AUTHORITY;
    AllocateAndInitializeSid(&creatorAuth, 1, SECURITY_CREATOR_OWNER_RID,
                             0,0,0,0,0,0,0, &pOwnerSid);
    AllocateAndInitializeSid(&ntAuth, 1, SECURITY_LOCAL_SYSTEM_RID,
                             0,0,0,0,0,0,0, &pSystemSid);
    ea[0].grfAccessPermissions = GENERIC_ALL;
    ea[0].grfAccessMode        = SET_ACCESS;
    ea[0].grfInheritance       = NO_INHERITANCE;
    ea[0].Trustee.TrusteeForm  = TRUSTEE_IS_SID;
    ea[0].Trustee.TrusteeType  = TRUSTEE_IS_WELL_KNOWN_GROUP;
    ea[0].Trustee.ptstrName    = (LPWSTR)pOwnerSid;
    ea[1].grfAccessPermissions = GENERIC_ALL;
    ea[1].grfAccessMode        = SET_ACCESS;
    ea[1].grfInheritance       = NO_INHERITANCE;
    ea[1].Trustee.TrusteeForm  = TRUSTEE_IS_SID;
    ea[1].Trustee.TrusteeType  = TRUSTEE_IS_WELL_KNOWN_GROUP;
    ea[1].Trustee.ptstrName    = (LPWSTR)pSystemSid;
    SetEntriesInAclW(2, ea, nullptr, &pACL);
    pSD = LocalAlloc(LPTR, SECURITY_DESCRIPTOR_MIN_LENGTH);
    InitializeSecurityDescriptor(pSD, SECURITY_DESCRIPTOR_REVISION);
    SetSecurityDescriptorDacl(pSD, TRUE, pACL, FALSE);
    SECURITY_ATTRIBUTES sa = {};
    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = pSD;
    sa.bInheritHandle = FALSE;

    HANDLE h = CreateNamedPipeA(
        pipeName.c_str(),
        PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
        PIPE_UNLIMITED_INSTANCES,
        65536, 65536,
        0, &sa);

    if (pOwnerSid) FreeSid(pOwnerSid);
    if (pSystemSid) FreeSid(pSystemSid);
    if (pACL) LocalFree(pACL);
    if (pSD) LocalFree(pSD);

    return (h == INVALID_HANDLE_VALUE) ? MPS_INVALID_HANDLE
                                       : reinterpret_cast<mps_handle_t>(h);
}
#endif

bool MPSServer::start() {
#if defined(_WIN32)
    // Windows: use named pipe instead of Unix domain socket.
    std::string pipeName;
    if (socketPath_.find("\\\\.\\pipe\\") == 0 || socketPath_.find("\\\\?\\pipe\\") == 0) {
        // Already a properly-formed named pipe path.
        pipeName = socketPath_;
    } else if (!socketPath_.empty() && socketPath_[0] == '/') {
        // POSIX-style path (e.g. /tmp/vgre_mps.sock, /run/user/.../vgre_mps.sock):
        // derive pipe name from the basename.
        auto sep = socketPath_.rfind('/');
        std::string base = (sep != std::string::npos) ? socketPath_.substr(sep + 1) : socketPath_;
        if (base.empty()) base = "vgre_mps";
        // Strip .sock suffix for a cleaner pipe name.
        if (base.size() > 5 && base.substr(base.size() - 5) == ".sock")
            base = base.substr(0, base.size() - 5);
        pipeName = "\\\\.\\pipe\\" + base;
    } else {
        pipeName = "\\\\.\\pipe\\" + socketPath_;
    }

    listenFd_ = createSecurePipeInstance(pipeName);
    if (listenFd_ == MPS_INVALID_HANDLE) {
        VGRE_LOG_ERROR("MPS", "CreateNamedPipe failed: " + std::to_string(GetLastError()));
        return false;
    }
    socketPath_ = pipeName; // store resolved pipe name

    running_.store(true);
    auto* t = new std::thread([this]{ acceptLoop(); });
    acceptThread_ = t;
    VGRE_LOG_INFO("MPS", "MPS daemon listening on named pipe " + pipeName);
    return true;
#elif defined(__linux__) || defined(__APPLE__)
    // Security: relocate socket from /tmp/ to user-owned $HOME/.vgre/ directory
    // to prevent local privilege escalation via socket hijacking.
    // Use getpwuid(getuid()) as authoritative home directory instead of $HOME
    // so that the path cannot be spoofed by the calling environment.
    std::string secureSocketPath = socketPath_;
    // Relocate from any world-writable temp directory (/tmp, /var/tmp) to $HOME/.vgre/
    // to prevent local privilege escalation via socket hijacking.
    // Paths under XDG_RUNTIME_DIR, TMPDIR (macOS per-user), or $HOME are already safe.
    auto isWorldWritableTmp = [](const std::string& p) {
        return p.find("/tmp/") == 0 || p.find("/var/tmp/") == 0;
    };
    if (isWorldWritableTmp(socketPath_)) {
        const char* homeDir = nullptr;
#if !defined(_WIN32)
        struct passwd *pw = getpwuid(getuid());
        if (pw && pw->pw_dir && pw->pw_dir[0] == '/') {
            homeDir = pw->pw_dir;
        } else {
            // Validated fallback: accept $HOME only if it starts with '/'
            // (rejects relative paths and shell substitutions like ~)
            const char* envHome = getenv("HOME");
            if (envHome && envHome[0] == '/') homeDir = envHome;
        }
#endif
        if (homeDir && homeDir[0] != '\0') {
            std::string vgreDir = std::string(homeDir) + "/.vgre";
            // Create .vgre directory with owner-only permissions if it doesn't exist
            struct stat st;
            if (stat(vgreDir.c_str(), &st) != 0) {
                mkdir(vgreDir.c_str(), 0700);
                chmod(vgreDir.c_str(), 0700);
            }
            // Replace /tmp/vgre_mps with $HOME/.vgre/vgre_mps
            std::string socketName = socketPath_.substr(socketPath_.find_last_of('/') + 1);
            secureSocketPath = vgreDir + "/" + socketName;
        }
    }

    listenFd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listenFd_ < 0) {
        VGRE_LOG_ERROR("MPS", "socket() failed: " + std::string(strerror(errno)));
        return false;
    }

    ::unlink(secureSocketPath.c_str()); // remove stale socket

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, secureSocketPath.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(listenFd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        VGRE_LOG_ERROR("MPS", "bind() failed: " + std::string(strerror(errno)));
        close(listenFd_); listenFd_ = -1;
        return false;
    }

    // Prevent local privilege escalation via the socket — owner access only.
    chmod(secureSocketPath.c_str(), 0600);
    socketPath_ = secureSocketPath; // update stored path

    if (listen(listenFd_, maxClients_) < 0) {
        VGRE_LOG_ERROR("MPS", "listen() failed: " + std::string(strerror(errno)));
        close(listenFd_); listenFd_ = -1;
        return false;
    }

    running_.store(true);
    // Store accept loop thread as void* to avoid including <thread> in the header.
    auto* t = new std::thread([this]{ acceptLoop(); });
    acceptThread_ = t;
    VGRE_LOG_INFO("MPS", "MPS daemon listening on " + socketPath_);
    return true;
#else
    VGRE_LOG_WARN("MPS", "MPS server not supported on this platform");
    return false;
#endif
}

void MPSServer::stop() {
    if (!running_.load()) return;
    running_.store(false);
#if defined(_WIN32)
    if (listenFd_ != MPS_INVALID_HANDLE) {
        CloseHandle(listenFd_);
        listenFd_ = MPS_INVALID_HANDLE;
    }
    if (acceptThread_) {
        auto* t = reinterpret_cast<std::thread*>(acceptThread_);
        if (t->joinable()) t->join();
        delete t;
        acceptThread_ = nullptr;
    }
#elif defined(__linux__) || defined(__APPLE__)
    if (listenFd_ != MPS_INVALID_HANDLE) { close(listenFd_); listenFd_ = MPS_INVALID_HANDLE; }
    if (acceptThread_) {
        auto* t = reinterpret_cast<std::thread*>(acceptThread_);
        if (t->joinable()) t->join();
        delete t;
        acceptThread_ = nullptr;
    }
    ::unlink(socketPath_.c_str());
#endif
    VGRE_LOG_INFO("MPS", "MPS daemon stopped");
}

void MPSServer::acceptLoop() {
    static std::atomic<uint32_t> slotCounter{0};
#if defined(_WIN32)
    while (running_.load()) {
        HANDLE hPipe = reinterpret_cast<HANDLE>(listenFd_);
        BOOL connected = ConnectNamedPipe(hPipe, nullptr);
        if (!connected && GetLastError() != ERROR_PIPE_CONNECTED) {
            if (!running_.load()) break;
            Sleep(50);
            continue;
        }
        if (!running_.load()) {
            DisconnectNamedPipe(hPipe);
            break;
        }
        // On Windows the "listen fd" IS the client pipe after ConnectNamedPipe.
        // Hand off the connected instance to a worker and create a fresh, equally
        // secured instance for the next client (per-client pipe instances, the
        // POSIX per-connection model). createSecurePipeInstance re-applies the
        // full DACL and uses PIPE_UNLIMITED_INSTANCES so concurrent clients never
        // contend on one instance.
        uint32_t slot = slotCounter.fetch_add(1, std::memory_order_relaxed);
        VGRE_LOG_INFO("MPS", "Client connected: slot=" + std::to_string(slot));
        mps_handle_t clientFd = listenFd_;
        listenFd_ = createSecurePipeInstance(socketPath_);
        if (listenFd_ == MPS_INVALID_HANDLE) {
            VGRE_LOG_ERROR("MPS", "CreateNamedPipe for next client failed");
            running_.store(false);
            // Still service the already-connected client before exiting the loop.
            std::thread([this, clientFd, slot]{ handleClient(clientFd, slot); }).detach();
            break;
        }
        std::thread([this, clientFd, slot]{ handleClient(clientFd, slot); }).detach();
    }
#elif defined(__linux__) || defined(__APPLE__)
    while (running_.load()) {
        struct pollfd pfd{ static_cast<int>(listenFd_), POLLIN, 0 };
        int r = poll(&pfd, 1, 200); // 200 ms timeout so we can check running_
        if (r <= 0) continue;
        if (!(pfd.revents & POLLIN)) continue;

        int clientFd = accept(listenFd_, nullptr, nullptr);
        if (clientFd < 0) continue;
        if (!running_.load()) {
#if defined(_WIN32)
            // nothing to close — the fd was already handed off
#else
            close(clientFd);
#endif
            break;
        }

        uint32_t slot = slotCounter.fetch_add(1, std::memory_order_relaxed);
        VGRE_LOG_INFO("MPS", "Client connected: slot=" + std::to_string(slot));
        std::thread([this, clientFd, slot]{ handleClient(clientFd, slot); }).detach();
    }
#endif
}

void MPSServer::handleClient(mps_handle_t fd, uint32_t slotId) {
    // Send CONNECT_ACK
    {
        MPSHeader hdr{ static_cast<uint32_t>(MPSMsgType::CONNECT_ACK),
                       sizeof(MPSConnectAck) };
        MPSConnectAck ack{ slotId, static_cast<uint32_t>(maxClients_) };
        writeAll(fd, &hdr, sizeof(hdr));
        writeAll(fd, &ack, sizeof(ack));
    }

    while (running_.load()) {
        MPSHeader hdr{};
        if (!readAll(fd, &hdr, sizeof(hdr))) break;

        switch (static_cast<MPSMsgType>(hdr.type)) {

        case MPSMsgType::MALLOC: {
            MPSMallocReq req{};
            if (!readAll(fd, &req, sizeof(req))) break;

            // Track P: enforce the per-client memory quota before allocating.
            auto q = clientQuota(slotId);
            uint64_t handle = 0;
            {
                std::lock_guard<std::mutex> lk(q->mtx);
                if (q->allocatedBytes + req.size > mpsPolicy().maxMemoryBytes) {
                    VGRE_LOG_WARN("MPS", "slot=" + std::to_string(slotId) +
                        " MALLOC of " + std::to_string(req.size) +
                        " denied: would exceed maxMemoryBytes (" +
                        std::to_string(q->allocatedBytes) + "/" +
                        std::to_string(mpsPolicy().maxMemoryBytes) + ")");
                } else {
                    handle = mpsAlloc(req.size);
                    if (handle) {
                        q->allocatedBytes += req.size;
                        std::lock_guard<std::mutex> ol(allocOwnerMutex());
                        allocOwnerMap()[handle] = {slotId, req.size};
                    }
                }
            }
            MPSHeader rh{ static_cast<uint32_t>(MPSMsgType::MALLOC_RESP),
                          sizeof(MPSMallocResp) };
            MPSMallocResp rsp{ handle ? 0u : 1u, handle };
            writeAll(fd, &rh, sizeof(rh));
            writeAll(fd, &rsp, sizeof(rsp));
            break;
        }

        case MPSMsgType::FREE: {
            MPSFreeReq req{};
            if (!readAll(fd, &req, sizeof(req))) break;
            bool ok = mpsFree(req.devPtr);
            if (ok) {
                // Track P: refund the freed bytes to the owning client's quota.
                std::pair<uint32_t,uint64_t> owner{0,0};
                bool found = false;
                {
                    std::lock_guard<std::mutex> ol(allocOwnerMutex());
                    auto it = allocOwnerMap().find(req.devPtr);
                    if (it != allocOwnerMap().end()) {
                        owner = it->second; found = true;
                        allocOwnerMap().erase(it);
                    }
                }
                if (found) {
                    auto q = clientQuota(owner.first);
                    std::lock_guard<std::mutex> lk(q->mtx);
                    q->allocatedBytes = (q->allocatedBytes >= owner.second)
                                        ? q->allocatedBytes - owner.second : 0;
                }
            }
            MPSHeader rh{ static_cast<uint32_t>(MPSMsgType::FREE_RESP),
                          sizeof(MPSFreeResp) };
            MPSFreeResp rsp{ ok ? 0u : 1u };
            writeAll(fd, &rh, sizeof(rh));
            writeAll(fd, &rsp, sizeof(rsp));
            break;
        }

        case MPSMsgType::MEMCPY_H2D: {
            MPSMemcpyReq req{};
            if (!readAll(fd, &req, sizeof(req))) break;
            std::vector<uint8_t> buf(static_cast<size_t>(req.bytes));
            if (!readAll(fd, buf.data(), static_cast<uint32_t>(req.bytes))) break;
            void* dst = mpsResolve(req.devPtr);
            if (dst) memcpy(dst, buf.data(), req.bytes);
            MPSHeader rh{ static_cast<uint32_t>(MPSMsgType::MEMCPY_RESP), 4 };
            uint32_t st = dst ? 0u : 1u;
            writeAll(fd, &rh, sizeof(rh));
            writeAll(fd, &st, sizeof(st));
            break;
        }

        case MPSMsgType::MEMCPY_D2H: {
            MPSMemcpyReq req{};
            if (!readAll(fd, &req, sizeof(req))) break;
            void* src = mpsResolve(req.devPtr);
            uint32_t payloadLen = src ? static_cast<uint32_t>(req.bytes) : 0;
            MPSHeader rh{ static_cast<uint32_t>(MPSMsgType::MEMCPY_RESP),
                          static_cast<uint32_t>(sizeof(uint32_t) + payloadLen) };
            uint32_t st = src ? 0u : 1u;
            writeAll(fd, &rh, sizeof(rh));
            writeAll(fd, &st, sizeof(st));
            if (src && payloadLen > 0)
                writeAll(fd, src, payloadLen);
            break;
        }

        case MPSMsgType::LAUNCH_KERNEL: {
            MPSLaunchReq req{};
            if (hdr.payloadLen < sizeof(req)) {
                VGRE_LOG_ERROR("MPS", "LAUNCH_KERNEL payload too small");
                break;
            }
            if (!readAll(fd, &req, sizeof(req))) break;

            uint32_t remainingPayload = hdr.payloadLen - sizeof(MPSLaunchReq);
            std::vector<uint8_t> payload(remainingPayload);
            if (remainingPayload > 0 && !readAll(fd, payload.data(), remainingPayload)) break;

            // Find null-terminated kernel name within payload bounds.
            const char* kernelName = nullptr;
            size_t nameLen = 0;
            if (remainingPayload > 0) {
                kernelName = reinterpret_cast<const char*>(payload.data());
                const char* end = reinterpret_cast<const char*>(
                    memchr(payload.data(), '\0', remainingPayload));
                if (!end) {
                    VGRE_LOG_ERROR("MPS", "LAUNCH_KERNEL: kernel name not null-terminated");
                    break;
                }
                nameLen = static_cast<size_t>(end - kernelName) + 1;
            } else {
                VGRE_LOG_ERROR("MPS", "LAUNCH_KERNEL: missing kernel name");
                break;
            }

            // Parse arguments with bounds checking.
            // Argument type encoding: 0=scalar, 1=device pointer, 2=host pointer, 3=array
            std::vector<void*> args;
            std::vector<std::vector<uint8_t>> argDataBuffers;
            uint8_t* pArgs = payload.data() + nameLen;
            size_t bytesLeft = remainingPayload - nameLen;
            bool parseOk = true;
            for (uint32_t i = 0; i < req.numArgs && parseOk; ++i) {
                if (bytesLeft < 5) { parseOk = false; break; }
                uint8_t type = *pArgs++;
                --bytesLeft;
                if (bytesLeft < 4) { parseOk = false; break; }
                uint32_t size = *reinterpret_cast<uint32_t*>(pArgs);
                pArgs += 4; bytesLeft -= 4;
                if (bytesLeft < size) { parseOk = false; break; }
                
                // Handle different argument types
                std::vector<uint8_t> argBuf(size);
                if (type == 1 || type == 2) {
                    // Pointer type: store the pointer value directly
                    if (size == sizeof(void*)) {
                        memcpy(argBuf.data(), pArgs, size);
                    } else {
                        // Invalid pointer size
                        parseOk = false;
                        break;
                    }
                } else if (type == 3) {
                    // Array type: store array data
                    memcpy(argBuf.data(), pArgs, size);
                } else {
                    // Scalar type (type == 0): store scalar value
                    memcpy(argBuf.data(), pArgs, size);
                }
                
                pArgs += size; bytesLeft -= size;
                argDataBuffers.push_back(std::move(argBuf));
                args.push_back(argDataBuffers.back().data());
            }
            if (!parseOk) {
                VGRE_LOG_ERROR("MPS", "LAUNCH_KERNEL: argument parse error (malformed payload)");
                break;
            }

            // Track P: enforce per-client queue depth, then throttle the number
            // of concurrent launches to the client's thread-fraction share.
            auto q = clientQuota(slotId);
            {
                std::unique_lock<std::mutex> lk(q->mtx);
                if (q->inFlight >= mpsPolicy().maxQueueDepth) {
                    lk.unlock();
                    VGRE_LOG_WARN("MPS", "slot=" + std::to_string(slotId) +
                        " LAUNCH denied: queue depth " + std::to_string(q->inFlight) +
                        " >= maxQueueDepth " + std::to_string(mpsPolicy().maxQueueDepth));
                    MPSHeader rh{ static_cast<uint32_t>(MPSMsgType::LAUNCH_RESP),
                                  sizeof(MPSLaunchResp) };
                    MPSLaunchResp rsp{ 1u };  // launch rejected (queue full)
                    writeAll(fd, &rh, sizeof(rh));
                    writeAll(fd, &rsp, sizeof(rsp));
                    break;
                }
                ++q->inFlight;
                // Throttle execution: wait until this client's concurrent run
                // count is below its core-share cap (maxThreadFraction).
                const uint32_t maxConc = mpsMaxConcurrent();
                q->cv.wait(lk, [&]{ return q->running < maxConc; });
                ++q->running;
            }

            auto& engine = vgre::core::RuntimeEngine::instance();
            vgre::VGREResult rc = engine.launchKernel(kernelName, "",
                {req.gridX, req.gridY, req.gridZ},
                {req.blockX, req.blockY, req.blockZ},
                args.empty() ? nullptr : args.data(),
                req.sharedMemBytes);

            {
                std::lock_guard<std::mutex> lk(q->mtx);
                if (q->running)  --q->running;
                if (q->inFlight) --q->inFlight;
            }
            q->cv.notify_all();

            VGRE_LOG_DEBUG("MPS", "Launched kernel '" + std::string(kernelName) +
                           "' via MPS. Result=" + std::to_string(static_cast<int>(rc)));

            MPSHeader rh{ static_cast<uint32_t>(MPSMsgType::LAUNCH_RESP), sizeof(MPSLaunchResp) };
            MPSLaunchResp rsp{ rc == vgre::VGREResult::SUCCESS ? 0u : 1u };
            writeAll(fd, &rh, sizeof(rh));
            writeAll(fd, &rsp, sizeof(rsp));
            break;
        }

        case MPSMsgType::SYNC: {
            MPSHeader rh{ static_cast<uint32_t>(MPSMsgType::SYNC_RESP),
                          sizeof(MPSSyncResp) };
            MPSSyncResp rsp{ 0 };
            writeAll(fd, &rh, sizeof(rh));
            writeAll(fd, &rsp, sizeof(rsp));
            break;
        }

        case MPSMsgType::DISCONNECT:
            VGRE_LOG_INFO("MPS", "Client disconnected: slot=" + std::to_string(slotId));
            releaseClientQuota(slotId);  // Track P: drop this client's quota state
#if defined(_WIN32)
            CloseHandle(fd);
#else
            close(fd);
#endif
            return;

        default:
            VGRE_LOG_WARN("MPS", "Unknown MPS msg type: " + std::to_string(hdr.type));
            if (hdr.payloadLen > 0) {
                std::vector<uint8_t> tmp(hdr.payloadLen);
                readAll(fd, tmp.data(), hdr.payloadLen);
            }
            break;
        }
    }
    releaseClientQuota(slotId);  // Track P: client loop ended — drop quota state
#if defined(_WIN32)
    CloseHandle(fd);
#else
    close(fd);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
//  MPSClient
// ─────────────────────────────────────────────────────────────────────────────

static MPSClient*& getMpsClient() {
    static MPSClient* client = nullptr;
    return client;
}
static std::mutex& getMpsClientMutex() {
    static std::mutex m;
    return m;
}

MPSClient* MPSClient::instance() {
    std::lock_guard<std::mutex> lk(getMpsClientMutex());
    if (getMpsClient()) return getMpsClient();
    const char* pipe = vgre_get_config("VGRE_MPS_PIPE");
    if (!pipe) return nullptr;
    getMpsClient() = new MPSClient();
    if (!getMpsClient()->connect(pipe)) {
        delete getMpsClient(); getMpsClient() = nullptr;
        VGRE_LOG_WARN("MPS", "Could not connect to MPS daemon at " + std::string(pipe));
    }
    return getMpsClient();
}

void MPSClient::shutdown() {
    std::lock_guard<std::mutex> lk(getMpsClientMutex());
    if (getMpsClient()) {
        getMpsClient()->disconnect();
        delete getMpsClient();
        getMpsClient() = nullptr;
    }
}

bool MPSClient::connect(const std::string& socketPath) {
#if defined(_WIN32)
    std::string pipeName = socketPath;
    if (socketPath.find("\\\\.\\pipe\\") != 0) {
        pipeName = "\\\\.\\pipe\\vgre_mps";
    }
    mps_handle_t hPipe = CreateFileA(pipeName.c_str(), GENERIC_READ | GENERIC_WRITE,
                                     0, nullptr, OPEN_EXISTING, 0, nullptr);
    if (hPipe == INVALID_HANDLE_VALUE) {
        VGRE_LOG_WARN("MPS", "Could not open named pipe: " + pipeName);
        return false;
    }
    fd_ = hPipe;

    // Send CONNECT header (no payload)
    MPSHeader ch{ static_cast<uint32_t>(MPSMsgType::CONNECT), 0 };
    if (!writeAll(fd_, &ch, sizeof(ch))) {
        CloseHandle(hPipe); fd_ = MPS_INVALID_HANDLE; return false;
    }

    MPSHeader rh{};
    MPSConnectAck ack{};
    if (!readAll(fd_, &rh, sizeof(rh)) ||
        rh.type != static_cast<uint32_t>(MPSMsgType::CONNECT_ACK) ||
        !readAll(fd_, &ack, sizeof(ack))) {
        CloseHandle(hPipe); fd_ = MPS_INVALID_HANDLE; return false;
    }
    slotId_ = ack.slotId;
    VGRE_LOG_INFO("MPS", "Connected to MPS daemon, slot=" + std::to_string(slotId_));
    return true;
#elif defined(__linux__) || defined(__APPLE__)
    fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd_ < 0) return false;

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socketPath.c_str(), sizeof(addr.sun_path) - 1);

    if (::connect(fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(fd_); fd_ = MPS_INVALID_HANDLE;
        return false;
    }

    // Send CONNECT header (no payload)
    MPSHeader ch{ static_cast<uint32_t>(MPSMsgType::CONNECT), 0 };
    if (!writeAll(fd_, &ch, sizeof(ch))) { close(fd_); fd_ = MPS_INVALID_HANDLE; return false; }

    // Receive CONNECT_ACK
    MPSHeader rh{};
    MPSConnectAck ack{};
    if (!readAll(fd_, &rh, sizeof(rh)) ||
        rh.type != static_cast<uint32_t>(MPSMsgType::CONNECT_ACK) ||
        !readAll(fd_, &ack, sizeof(ack))) {
        close(fd_); fd_ = MPS_INVALID_HANDLE; return false;
    }
    slotId_ = ack.slotId;
    VGRE_LOG_INFO("MPS", "Connected to MPS daemon, slot=" + std::to_string(slotId_));
    return true;
#else
    return false;
#endif
}

void MPSClient::disconnect() {
    if (fd_ == MPS_INVALID_HANDLE) return;
    MPSHeader h{ static_cast<uint32_t>(MPSMsgType::DISCONNECT), 0 };
    writeAll(fd_, &h, sizeof(h));
#if defined(_WIN32)
    CloseHandle(fd_);
#else
    close(fd_);
#endif
    fd_ = MPS_INVALID_HANDLE;
}

bool MPSClient::sendMsg(MPSMsgType type, const void* payload, uint32_t len) {
    MPSHeader h{ static_cast<uint32_t>(type), len };
    return writeAll(fd_, &h, sizeof(h)) && (len == 0 || writeAll(fd_, payload, len));
}

bool MPSClient::recvHeader(MPSHeader& hdr) {
    return readAll(fd_, &hdr, sizeof(hdr));
}

bool MPSClient::recvBytes(void* buf, uint32_t len) {
    return readAll(fd_, buf, len);
}

uint64_t MPSClient::malloc(uint64_t bytes) {
    if (fd_ == MPS_INVALID_HANDLE) return 0;
    MPSMallocReq req{ bytes };
    if (!sendMsg(MPSMsgType::MALLOC, &req, sizeof(req))) return 0;
    MPSHeader rh{};
    MPSMallocResp rsp{};
    if (!recvHeader(rh) || !recvBytes(&rsp, sizeof(rsp))) return 0;
    return rsp.status == 0 ? rsp.devPtr : 0;
}

bool MPSClient::free(uint64_t devPtr) {
    if (fd_ == MPS_INVALID_HANDLE) return false;
    MPSFreeReq req{ devPtr };
    if (!sendMsg(MPSMsgType::FREE, &req, sizeof(req))) return false;
    MPSHeader rh{};
    MPSFreeResp rsp{};
    return recvHeader(rh) && recvBytes(&rsp, sizeof(rsp)) && rsp.status == 0;
}

bool MPSClient::memcpyH2D(uint64_t devPtr, const void* hostSrc, uint64_t bytes) {
    if (fd_ == MPS_INVALID_HANDLE) return false;
    MPSMemcpyReq req{ devPtr, bytes };
    MPSHeader h{ static_cast<uint32_t>(MPSMsgType::MEMCPY_H2D),
                 static_cast<uint32_t>(sizeof(req) + bytes) };
    if (!writeAll(fd_, &h, sizeof(h))) return false;
    if (!writeAll(fd_, &req, sizeof(req))) return false;
    if (!writeAll(fd_, hostSrc, static_cast<size_t>(bytes))) return false;
    MPSHeader rh{};
    uint32_t st{};
    return recvHeader(rh) && recvBytes(&st, sizeof(st)) && st == 0;
}

bool MPSClient::memcpyD2H(void* hostDst, uint64_t devPtr, uint64_t bytes) {
    if (fd_ == MPS_INVALID_HANDLE) return false;
    MPSMemcpyReq req{ devPtr, bytes };
    if (!sendMsg(MPSMsgType::MEMCPY_D2H, &req, sizeof(req))) return false;
    MPSHeader rh{};
    if (!recvHeader(rh)) return false;
    uint32_t st{};
    if (!recvBytes(&st, sizeof(st))) return false;
    if (st != 0) return false;
    uint32_t dataLen = rh.payloadLen - sizeof(uint32_t);
    if (dataLen > 0 && !recvBytes(hostDst, dataLen)) return false;
    return true;
}

bool MPSClient::launchKernel(const char* name,
                              uint32_t gx, uint32_t gy, uint32_t gz,
                              uint32_t bx, uint32_t by, uint32_t bz,
                              uint64_t sharedMem,
                              void** args, int numArgs) {
    if (fd_ == MPS_INVALID_HANDLE) return false;
    
    // 1. Prepare argument payload
    // In CUDA, we don't have sizes, but VGRE's launchKernel (the one called on server)
    // takes void** and uses KernelIR. 
    // For the wire protocol, we'll assume 8-byte arguments (standard for pointers/scalars).
    std::vector<uint8_t> argPayload;
    for (int i = 0; i < numArgs; ++i) {
        argPayload.push_back(0); // type (ArgType::POINTER/STRUCT default)
        uint32_t sz = 8;
        uint8_t* psz = reinterpret_cast<uint8_t*>(&sz);
        argPayload.insert(argPayload.end(), psz, psz + 4);
        uint8_t* pval = reinterpret_cast<uint8_t*>(args[i]);
        argPayload.insert(argPayload.end(), pval, pval + 8);
    }

    MPSLaunchReq req{ gx, gy, gz, bx, by, bz, sharedMem, static_cast<uint32_t>(numArgs) };
    uint32_t nameLen = static_cast<uint32_t>(strlen(name)) + 1; // include null-terminator
    
    uint32_t totalPayloadLen = sizeof(req) + nameLen + static_cast<uint32_t>(argPayload.size());
    MPSHeader h{ static_cast<uint32_t>(MPSMsgType::LAUNCH_KERNEL), totalPayloadLen };
    
    if (!writeAll(fd_, &h, sizeof(h))) return false;
    if (!writeAll(fd_, &req, sizeof(req))) return false;
    if (!writeAll(fd_, name, nameLen)) return false;
    if (!argPayload.empty() && !writeAll(fd_, argPayload.data(), argPayload.size())) return false;
    
    MPSHeader rh{};
    MPSLaunchResp rsp{};
    return recvHeader(rh) && recvBytes(&rsp, sizeof(rsp)) && rsp.status == 0;
}

bool MPSClient::sync() {
    if (fd_ == MPS_INVALID_HANDLE) return false;
    MPSHeader h{ static_cast<uint32_t>(MPSMsgType::SYNC), 0 };
    if (!writeAll(fd_, &h, sizeof(h))) return false;
    MPSHeader rh{};
    MPSSyncResp rsp{};
    return recvHeader(rh) && recvBytes(&rsp, sizeof(rsp)) && rsp.status == 0;
}

} // namespace mps
} // namespace vgre
