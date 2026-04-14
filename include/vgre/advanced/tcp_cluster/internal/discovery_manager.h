/**
 * VGRE Discovery Manager — UDP-based Master/Worker Discovery
 * 
 * Responsibilities:
 * - UDP master announcements (broadcast master presence)
 * - UDP worker discovery (scan for master broadcasts)
 * - UDP worker announcements (broadcast worker presence)
 * - UDP master discovery of workers (scan for worker broadcasts)
 * - Proactive connection attempts to known worker addresses
 * 
 * Design: Uses facade pattern to access parent state through controlled interface
 * rather than direct member access, reducing coupling.
 */

#ifndef VGRE_DISCOVERY_MANAGER_H
#define VGRE_DISCOVERY_MANAGER_H

#include "vgre/advanced/tcp_cluster.h"  // Need full TCPClusterManager definition for friend access
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <functional>
#include <memory>
#include "vgre/common/sockets.h"

namespace vgre {
namespace advanced {

// TCPClusterManager is fully defined via tcp_cluster.h include above
// (required for MSVC friend class access to work correctly)

/**
 * DiscoveryManager - Handles UDP-based discovery and proactive connections
 * 
 * NOTE: This is a PARTIAL extraction for Phase 5. Due to deep coupling with
 * TCPClusterManager state (enabled_, is_master_, clients_, etc.), this module
 * maintains a parent pointer and accesses state through the parent.
 * 
 * A full extraction would require:
 * 1. Dependency injection of all state
 * 2. Callback interfaces for connection establishment
 * 3. Significant refactoring of TCPClusterManager
 * 
 * Current approach: Extract discovery LOGIC while maintaining parent access.
 * This reduces file size and improves organization without breaking functionality.
 */
class DiscoveryManager {
public:
    /**
     * Constructor
     * @param parent Pointer to parent TCPClusterManager (non-owning)
     */
    explicit DiscoveryManager(TCPClusterManager* parent);
    
    /**
     * Destructor - stops all discovery threads
     */
    ~DiscoveryManager();

    // Disable copy and move
    DiscoveryManager(const DiscoveryManager&) = delete;
    DiscoveryManager& operator=(const DiscoveryManager&) = delete;
    DiscoveryManager(DiscoveryManager&&) = delete;
    DiscoveryManager& operator=(DiscoveryManager&&) = delete;

    /**
     * Start master UDP announcer (broadcasts master presence)
     * Called when cluster is initialized as master
     */
    void startMasterAnnouncer();

    /**
     * Start master UDP discovery of workers (scans for worker broadcasts)
     * Called when cluster is initialized as master
     */
    void startMasterWorkerDiscovery();

    /**
     * Start worker UDP discovery (scans for master broadcasts)
     * Called when cluster is initialized as worker
     */
    void startWorkerDiscovery();

    /**
     * Start worker UDP announcer (broadcasts worker presence)
     * Called when cluster is initialized as worker
     */
    void startWorkerAnnouncer();

    /**
     * Start proactive connection loop (master connects to known workers)
     * Called when cluster is initialized as master
     */
    void startProactiveConnections();

    /**
     * Stop all discovery threads
     * Called during shutdown
     */
    void stopAll();

private:
    // Parent cluster manager (non-owning pointer)
    // Used to access: enabled_, is_master_, port_, security_enabled_,
    // clients_, proactive_worker_addresses_, and connection methods
    TCPClusterManager* parent_;

    // Discovery threads
    std::thread udp_announcer_thread_;
    std::thread master_discovery_thread_;
    std::thread worker_discovery_thread_;
    std::thread worker_announcer_thread_;
    std::thread proactive_thread_;

    // Stop flag for proactive connection loop
    std::atomic<bool> stop_proactive_{false};

    // Async handshake threads spawned by proactiveConnectionLoop.
    // Tracked here so stopAll() can join them before returning, preventing
    // the threads from accessing a destroyed TCPClusterManager via parent_.
    std::mutex auth_threads_mutex_;
    std::vector<std::thread> auth_threads_;

    // Thread loop methods (moved from TCPClusterManager)
    void udpAnnouncerLoop();
    void udpMasterDiscoveryLoop();
    void udpDiscoveryLoop();
    void udpWorkerAnnouncerLoop();
    void proactiveConnectionLoop();
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_DISCOVERY_MANAGER_H
