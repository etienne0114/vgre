/**
 * VGRE TCP Cluster Manager - Shutdown Implementation
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <thread>

namespace vgre {
namespace advanced {

// Force-close a socket with SHUT_RDWR to immediately unblock any pending
// recv/accept/send, then close it.
static void forceCloseSocket(vgre::common::vgre_socket_t &fd) {
    if (fd == vgre::common::VGRE_INVALID_SOCKET) return;
#ifndef _WIN32
    ::shutdown(fd, SHUT_RDWR);
#else
    ::shutdown(fd, SD_BOTH);
#endif
    vgre::common::vgre_close_socket(fd);
    fd = vgre::common::VGRE_INVALID_SOCKET;
}

// joinWithTimeout: join the thread directly.
//
// A prior version transferred ownership into a spawned "joiner" thread and
// waited on it with a timeout, to avoid std::async's future destructor
// blocking. But spawning any new OS thread here is itself unsafe: shutdown()
// runs from RuntimeEngine's atexit handler (registered so callers who never
// call vgre_shutdown() explicitly still get clean teardown), i.e. during
// process-exit-time static teardown — exactly the context Windows documents
// CreateThread as unsafe in, and where a spawned joiner previously crashed
// with STATUS_STACK_BUFFER_OVERRUN (observed as the identical crash already
// fixed in Scheduler::~Scheduler and MemoryManager::stopMigrationThread/
// stopPendingDrainer). By the time this runs, sockets are already force-closed
// and every CV already notified (see call sites below), so each thread's own
// wait predicate is satisfied and it returns within ~50 ms — a direct join
// is both safe and prompt; no timeout wrapper is needed.
static void joinWithTimeout(std::thread& t, const char* name, int /*timeoutSec*/ = 5) {
    if (!t.joinable()) return;
    fprintf(stderr, "DEBUG [TCPCluster] Joining %s\n", name);
    t.join();
    fprintf(stderr, "DEBUG [TCPCluster] %s joined cleanly\n", name);
}

void TCPClusterManager::shutdown() {
  bool wasEnabled = enabled_.exchange(false);

  // A shutdown returns the manager to its pre-initialization state: runtime
  // security is disabled (a subsequent initialize() re-enables it from the auth
  // token). Without this, the process-wide singleton keeps security_enabled_
  // latched across an init→shutdown cycle on hosts that have an auth token.
  security_enabled_.store(false, std::memory_order_release);

  // Always notify CVs and join threads regardless of wasEnabled.
  // A thread blocked on JIT inside data_processor_thread_ may still be running
  // even after clientLoop set enabled_=false on a disconnect. Skipping join
  // here would leave that thread running past the object's lifetime → UB.
  if (!wasEnabled) {
    // Discovery threads may still be running from a prior init/shutdown
    // cycle; stop them now so ~DiscoveryManager() doesn't std::terminate().
    if (discovery_manager_) {
      discovery_manager_->stopAll();
    }
    shutdown_cv_.notify_all();
    staging_cv_.notify_all();
    remote_results_cv_.notify_all();
    barrier_cv_.notify_all();
    joinWithTimeout(data_processor_thread_, "data_processor_thread_");
    joinWithTimeout(client_loop_thread_,    "client_loop_thread_");
    joinWithTimeout(cluster_thread_,        "cluster_thread_");
    joinWithTimeout(monitoring_thread_,     "monitoring_thread_");
    return;
  }

  VGRE_LOG_INFO("TCPCluster", "Shutting down cluster...");

  // 1. Force-close all sockets to unblock poll/accept/recv immediately.
  {
    std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
    for (auto &c : clients_)
        if (c) forceCloseSocket(c->socket_fd);
    forceCloseSocket(server_fd_);
  }

  {
    std::lock_guard<std::mutex> lock(client_mutex_);
    forceCloseSocket(client_fd_);
  }

  // 2. Stop discovery UDP threads.
  if (discovery_manager_) {
    fprintf(stderr, "DEBUG [TCPCluster] Calling discovery_manager_->stopAll()\n");
    discovery_manager_->stopAll();
  }

  // 3. Notify all condition variables so blocked threads wake immediately.
  fprintf(stderr, "DEBUG [TCPCluster] Notifying CVs...\n");
  shutdown_cv_.notify_all();
  staging_cv_.notify_all();
  remote_results_cv_.notify_all();
  barrier_cv_.notify_all();

  // 4. Join cluster threads with 5-second timeout each.
  //    After socket close + CV notify, threads check enabled_ (now false)
  //    and exit within one poll cycle (~50 ms).
  joinWithTimeout(data_processor_thread_, "data_processor_thread_");
  joinWithTimeout(client_loop_thread_,    "client_loop_thread_");
  joinWithTimeout(cluster_thread_,        "cluster_thread_");
  joinWithTimeout(monitoring_thread_,     "monitoring_thread_");

  // 5. Join auth threads — sockets are already closed above.
  {
    std::lock_guard<std::mutex> lk(server_auth_mutex_);
    for (auto &e : server_auth_threads_)
        if (e.t.joinable())
            joinWithTimeout(e.t, "server_auth_thread");
    server_auth_threads_.clear();
  }

  if (is_master_) clients_.clear();
  VGRE_LOG_INFO("TCPCluster", "Shutdown completed.");
}

} // namespace advanced
} // namespace vgre
