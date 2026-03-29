#ifndef VGRE_ADVANCED_RESOURCE_LEDGER_H
#define VGRE_ADVANCED_RESOURCE_LEDGER_H

#include "vgre/common/error_codes.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace vgre {
namespace advanced {

// ── Credit direction ──────────────────────────────────────────────────────
enum class CreditDirection : uint8_t {
  DEBIT = 0,   // master consumed compute from remote
  CREDIT = 1   // remote provided compute to master
};

// ── Single ledger entry ───────────────────────────────────────────────────
struct CreditEntry {
  std::string node_address;
  double compute_unit_seconds = 0.0; // execution_time_seconds × cpu_cores
  uint64_t timestamp = 0;
  uint64_t kernel_id = 0;
  CreditDirection direction = CreditDirection::DEBIT;
};

// ── Per-node balance summary ──────────────────────────────────────────────
struct NodeBalance {
  std::string address;
  double total_credits = 0.0;   // compute received by this node
  double total_debits = 0.0;    // compute consumed from this node
  double balance = 0.0;         // credits - debits
  uint64_t last_activity = 0;
  int transaction_count = 0;
};

// ── Resource Ledger ───────────────────────────────────────────────────────
// Tracks compute resource usage across the distributed network.
// Each unit is measured in compute-unit-seconds (CUS):
//   CUS = wall_clock_seconds × cpu_cores_used
//
// Persisted in JSON format for cross-session accounting.
class ResourceLedger {
public:
  ResourceLedger();
  ~ResourceLedger();

  /**
   * @brief Records compute usage for billing.
   *
   * @param nodeAddress IP address of the remote node.
   * @param executionTimeSec Wall-clock execution time in seconds.
   * @param cpuCores Number of CPU cores used.
   * @param kernelId ID of the executed kernel.
   * @param direction DEBIT if master consumed, CREDIT if worker provided.
   */
  VGREResult recordCompute(const std::string &nodeAddress,
                            double executionTimeSec, int cpuCores,
                            uint64_t kernelId,
                            CreditDirection direction);

  /**
   * @brief Gets the balance for a specific node.
   */
  VGREResult getBalance(const std::string &nodeAddress,
                         NodeBalance &outBalance) const;

  /**
   * @brief Gets balances for all nodes.
   */
  std::vector<NodeBalance> getAllBalances() const;

  /**
   * @brief Resets all ledger entries and balances.
   */
  void reset();

  /**
   * @brief Persists the ledger to disk (JSON format).
   */
  VGREResult persist() const;

  /**
   * @brief Loads the ledger from disk.
   */
  VGREResult load();

  /**
   * @brief Gets the ledger file path.
   */
  std::string getLedgerPath() const;

  /**
   * @brief Gets total entries in the transaction log.
   */
  size_t getEntryCount() const;

  // Singleton
  static ResourceLedger &instance();

private:
  // Internal JSON serialization (no external dependency)
  std::string toJSON() const;
  VGREResult fromJSON(const std::string &json);

  // Compute storage path
  std::string computeLedgerPath() const;

  std::vector<CreditEntry> entries_;
  static constexpr size_t kMaxEntries = 10000;
  mutable std::recursive_mutex mutex_;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_RESOURCE_LEDGER_H
