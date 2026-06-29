// Self-contained, cross-platform TCP all-reduce for the lightweight (LLVM-free)
// libvgre_nn.
//
// The full libvgre routes vgre_cluster_all_reduce through TCPClusterManager, but
// that pulls in the core runtime + api + LLVM — too heavy for the lightweight ML
// library. So this is a real, from-scratch collective over plain TCP sockets,
// with no project dependencies: it genuinely sum-reduces a float buffer across
// `world_size` nodes so libvgre_nn can train data-/tensor-parallel on a CPU
// cluster while staying LLVM/BLAS/CUDA-free.
//
// Topology: a star. rank 0 binds+listens and accepts (world-1) workers; workers
// connect to rank 0. Each all-reduce: workers send their buffer to rank 0, rank
// 0 sums all contributions, then broadcasts the result back. Connections are
// established once (lazily) and reused. Configuration mirrors the standard
// distributed-launch convention via environment variables:
//   VGRE_NN_WORLD_SIZE  (default 1 → single node, all-reduce is identity)
//   VGRE_NN_RANK        (0..world-1; rank 0 is the reducer)
//   VGRE_NN_MASTER_HOST (default 127.0.0.1)
//   VGRE_NN_MASTER_PORT (default 29500)

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
using sock_t = SOCKET;
static const sock_t kBadSock = INVALID_SOCKET;
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <unistd.h>
using sock_t = int;
static const sock_t kBadSock = -1;
#endif

namespace {

int env_int(const char* k, int def) {
    const char* v = std::getenv(k);
    if (!v || !*v) return def;
    return std::atoi(v);
}

void close_sock(sock_t s) {
    if (s == kBadSock) return;
#if defined(_WIN32)
    closesocket(s);
#else
    ::close(s);
#endif
}

bool send_all(sock_t s, const char* buf, size_t n) {
    size_t off = 0;
    while (off < n) {
#if defined(_WIN32)
        int r = ::send(s, buf + off, (int)(n - off), 0);
#else
        ssize_t r = ::send(s, buf + off, n - off, 0);
#endif
        if (r <= 0) return false;
        off += (size_t)r;
    }
    return true;
}

bool recv_all(sock_t s, char* buf, size_t n) {
    size_t off = 0;
    while (off < n) {
#if defined(_WIN32)
        int r = ::recv(s, buf + off, (int)(n - off), 0);
#else
        ssize_t r = ::recv(s, buf + off, n - off, 0);
#endif
        if (r <= 0) return false;
        off += (size_t)r;
    }
    return true;
}

// Star topology, established once and reused across all-reduce calls.
struct ClusterState {
    std::mutex mu;
    bool inited = false;
    bool ok = false;            // multi-node connections established
    int  world = 1;
    int  rank = 0;
    std::vector<sock_t> workers; // rank 0: one socket per worker (index = rank-1)
    sock_t master = kBadSock;    // workers: socket to rank 0

    ~ClusterState() {
        for (sock_t s : workers) close_sock(s);
        close_sock(master);
#if defined(_WIN32)
        if (inited && world > 1) WSACleanup();
#endif
    }

    void init() {
        if (inited) return;
        inited = true;
        world = env_int("VGRE_NN_WORLD_SIZE", 1);
        rank  = env_int("VGRE_NN_RANK", 0);
        if (world <= 1) return;                 // single node: identity, nothing to set up

#if defined(_WIN32)
        WSADATA wsa; if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return;
#endif
        const std::string host = std::getenv("VGRE_NN_MASTER_HOST")
                                     ? std::getenv("VGRE_NN_MASTER_HOST") : "127.0.0.1";
        const int port = env_int("VGRE_NN_MASTER_PORT", 29500);

        if (rank == 0) {
            sock_t srv = ::socket(AF_INET, SOCK_STREAM, 0);
            if (srv == kBadSock) return;
            int yes = 1;
            ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char*)&yes, sizeof(yes));
            sockaddr_in addr{}; addr.sin_family = AF_INET;
            addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons((uint16_t)port);
            if (::bind(srv, (sockaddr*)&addr, sizeof(addr)) != 0 || ::listen(srv, world) != 0) {
                close_sock(srv); return;
            }
            workers.assign(world - 1, kBadSock);
            for (int i = 0; i < world - 1; ++i) {
                sock_t c = ::accept(srv, nullptr, nullptr);
                if (c == kBadSock) { close_sock(srv); return; }
                int one = 1; ::setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
                // The worker announces its rank so buffers map deterministically.
                int wrank = 0;
                if (!recv_all(c, (char*)&wrank, sizeof(wrank)) || wrank < 1 || wrank >= world) {
                    close_sock(c); close_sock(srv); return;
                }
                workers[wrank - 1] = c;
            }
            close_sock(srv);
            ok = true;
        } else {
            // Worker: connect to rank 0 (retry briefly — the master may start late).
            sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_port = htons((uint16_t)port);
            ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
            for (int attempt = 0; attempt < 200; ++attempt) {
                master = ::socket(AF_INET, SOCK_STREAM, 0);
                if (master != kBadSock && ::connect(master, (sockaddr*)&addr, sizeof(addr)) == 0) break;
                close_sock(master); master = kBadSock;
#if defined(_WIN32)
                Sleep(50);
#else
                usleep(50000);
#endif
            }
            if (master == kBadSock) return;
            int one = 1; ::setsockopt(master, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
            if (!send_all(master, (const char*)&rank, sizeof(rank))) { close_sock(master); master = kBadSock; return; }
            ok = true;
        }
    }
};

ClusterState& state() { static ClusterState s; return s; }

}  // namespace

// Sum-reduce `count` floats across all nodes, in place (identical on every node).
extern "C" int vgre_cluster_all_reduce(void* ptr, size_t count, int datatype) {
    if (!ptr || count == 0) return -2;          // VGRE_ERROR_INVALID_VALUE
    if (datatype != 3 /*FLOAT32*/) return -2;   // this transport reduces f32
    ClusterState& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    st.init();
    if (st.world <= 1) return 0;                 // single node: identity (correct, not a stub)
    if (!st.ok) return -6;                       // VGRE_ERROR_IO (setup failed)

    float* data = static_cast<float*>(ptr);
    const size_t bytes = count * sizeof(float);

    if (st.rank == 0) {
        std::vector<float> tmp(count);
        for (sock_t w : st.workers) {            // sum every worker's contribution
            if (!recv_all(w, (char*)tmp.data(), bytes)) return -6;
            for (size_t i = 0; i < count; ++i) data[i] += tmp[i];
        }
        for (sock_t w : st.workers)              // broadcast the reduced result
            if (!send_all(w, (const char*)data, bytes)) return -6;
    } else {
        if (!send_all(st.master, (const char*)data, bytes)) return -6;
        if (!recv_all(st.master, (char*)data, bytes)) return -6;
    }
    return 0;  // VGRE_SUCCESS
}

extern "C" int vgre_cluster_world_size(void) {
    ClusterState& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    st.init();
    return st.world < 1 ? 1 : st.world;
}
