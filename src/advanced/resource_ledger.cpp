#include "vgre/advanced/resource_ledger.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#include <shlobj.h>
#include <windows.h>
#include <direct.h>
#else
#include <pwd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace vgre {
namespace advanced {

ResourceLedger::ResourceLedger() {
  // Auto-load existing ledger on construction
  VGREResult r = load();
  if (r == VGREResult::SUCCESS) {
    VGRE_LOG_INFO("ResourceLedger",
                  "Loaded " + std::to_string(entries_.size()) +
                      " entries from " + getLedgerPath());
  } else {
    VGRE_LOG_DEBUG("ResourceLedger",
                   "No existing ledger found — starting fresh");
  }
}

ResourceLedger::~ResourceLedger() {
  // Auto-save on destruction
  if (!entries_.empty()) {
    persist();
  }
}

VGREResult ResourceLedger::recordCompute(const std::string &nodeAddress,
                                          double executionTimeSec,
                                          int cpuCores, uint64_t kernelId,
                                          CreditDirection direction) {
  if (nodeAddress.empty() || executionTimeSec < 0.0 || cpuCores <= 0) {
    return VGREResult::ERROR_INVALID_VALUE;
  }

  double computeUnitSeconds = executionTimeSec * static_cast<double>(cpuCores);

  uint64_t now = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());

  CreditEntry entry;
  entry.node_address = nodeAddress;
  entry.compute_unit_seconds = computeUnitSeconds;
  entry.timestamp = now;
  entry.kernel_id = kernelId;
  entry.direction = direction;

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  entries_.push_back(entry);

  // Rolling history — trim to max entries
  if (entries_.size() > kMaxEntries) {
    entries_.erase(entries_.begin(),
                   entries_.begin() +
                       static_cast<long>(entries_.size() - kMaxEntries));
  }

  VGRE_LOG_DEBUG("ResourceLedger",
                 std::string(direction == CreditDirection::CREDIT ? "CREDIT"
                                                                  : "DEBIT") +
                     " " + std::to_string(computeUnitSeconds) +
                     " CUS from " + nodeAddress + " (kernel=" +
                     std::to_string(kernelId) + ")");

  return VGREResult::SUCCESS;
}

VGREResult ResourceLedger::getBalance(const std::string &nodeAddress,
                                       NodeBalance &outBalance) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  outBalance = {};
  outBalance.address = nodeAddress;

  bool found = false;
  for (const auto &entry : entries_) {
    if (entry.node_address == nodeAddress) {
      found = true;
      if (entry.direction == CreditDirection::CREDIT) {
        outBalance.total_credits += entry.compute_unit_seconds;
      } else {
        outBalance.total_debits += entry.compute_unit_seconds;
      }
      outBalance.transaction_count++;
      if (entry.timestamp > outBalance.last_activity) {
        outBalance.last_activity = entry.timestamp;
      }
    }
  }

  outBalance.balance = outBalance.total_credits - outBalance.total_debits;

  if (!found) {
    return VGREResult::ERROR_INVALID_VALUE;
  }

  return VGREResult::SUCCESS;
}

std::vector<NodeBalance> ResourceLedger::getAllBalances() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Aggregate per-node
  std::vector<std::string> uniqueAddresses;
  for (const auto &entry : entries_) {
    if (std::find(uniqueAddresses.begin(), uniqueAddresses.end(),
                  entry.node_address) == uniqueAddresses.end()) {
      uniqueAddresses.push_back(entry.node_address);
    }
  }

  std::vector<NodeBalance> result;
  result.reserve(uniqueAddresses.size());

  for (const auto &addr : uniqueAddresses) {
    NodeBalance bal;
    bal.address = addr;

    for (const auto &entry : entries_) {
      if (entry.node_address == addr) {
        if (entry.direction == CreditDirection::CREDIT) {
          bal.total_credits += entry.compute_unit_seconds;
        } else {
          bal.total_debits += entry.compute_unit_seconds;
        }
        bal.transaction_count++;
        if (entry.timestamp > bal.last_activity) {
          bal.last_activity = entry.timestamp;
        }
      }
    }

    bal.balance = bal.total_credits - bal.total_debits;
    result.push_back(bal);
  }

  return result;
}

void ResourceLedger::reset() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  entries_.clear();
  VGRE_LOG_INFO("ResourceLedger", "Ledger reset — all entries cleared");
}

size_t ResourceLedger::getEntryCount() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return entries_.size();
}

// ── Persistence ─────────────────────────────────────────────────────────

std::string ResourceLedger::computeLedgerPath() const {
#if defined(_WIN32)
  char path[MAX_PATH];
  if (SUCCEEDED(SHGetFolderPathA(NULL, CSIDL_APPDATA, NULL, 0, path))) {
    std::string base(path);
    base += "\\VGRE";
    _mkdir(base.c_str());
    return base + "\\ledger.json";
  }
  return "vgre_ledger.json";
#else
  const char *home = std::getenv("HOME");
  if (!home) {
    struct passwd *pw = getpwuid(getuid());
    if (pw)
      home = pw->pw_dir;
  }
  if (home) {
    std::string base = std::string(home) + "/.vgre";
    mkdir(base.c_str(), 0700);
    return base + "/ledger.json";
  }
  return "/tmp/vgre_ledger.json";
#endif
}

std::string ResourceLedger::getLedgerPath() const {
  return computeLedgerPath();
}

// ── JSON Serialization (Manual — no external dependency) ──────────────────

namespace {
// Simple JSON string escaping
std::string jsonEscape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out += c;
    }
  }
  return out;
}
} // namespace

std::string ResourceLedger::toJSON() const {
  std::ostringstream oss;
  oss << std::fixed;
  oss << "{\n";
  oss << "  \"version\": 1,\n";
  oss << "  \"entry_count\": " << entries_.size() << ",\n";
  oss << "  \"entries\": [\n";

  for (size_t i = 0; i < entries_.size(); ++i) {
    const auto &e = entries_[i];
    oss << "    {\n";
    oss << "      \"address\": \"" << jsonEscape(e.node_address) << "\",\n";
    oss << "      \"cus\": " << e.compute_unit_seconds << ",\n";
    oss << "      \"timestamp\": " << e.timestamp << ",\n";
    oss << "      \"kernel_id\": " << e.kernel_id << ",\n";
    oss << "      \"direction\": "
        << (e.direction == CreditDirection::CREDIT ? "1" : "0") << "\n";
    oss << "    }" << (i + 1 < entries_.size() ? "," : "") << "\n";
  }

  oss << "  ]\n";
  oss << "}\n";
  return oss.str();
}

VGREResult ResourceLedger::fromJSON(const std::string &json) {
  // Minimal JSON parser for the known ledger format.
  // This is NOT a general-purpose JSON parser — it handles only the
  // exact format produced by toJSON() above.

  entries_.clear();

  // Find entries array
  size_t entriesPos = json.find("\"entries\"");
  if (entriesPos == std::string::npos) {
    return VGREResult::ERROR_INVALID_VALUE;
  }

  // Find array start
  size_t arrayStart = json.find('[', entriesPos);
  if (arrayStart == std::string::npos) {
    return VGREResult::ERROR_INVALID_VALUE;
  }

  // Parse each entry
  size_t pos = arrayStart;
  while (pos < json.size()) {
    size_t objStart = json.find('{', pos);
    if (objStart == std::string::npos)
      break;

    size_t objEnd = json.find('}', objStart);
    if (objEnd == std::string::npos)
      break;

    std::string obj = json.substr(objStart, objEnd - objStart + 1);

    CreditEntry entry;

    // Parse address
    size_t addrPos = obj.find("\"address\"");
    if (addrPos != std::string::npos) {
      size_t quote1 = obj.find('"', addrPos + 9);
      if (quote1 != std::string::npos) {
        quote1++; // skip opening quote
        size_t quote2 = obj.find('"', quote1);
        if (quote2 != std::string::npos) {
          entry.node_address = obj.substr(quote1, quote2 - quote1);
        }
      }
    }

    // Parse cus
    size_t cusPos = obj.find("\"cus\"");
    if (cusPos != std::string::npos) {
      size_t colonPos = obj.find(':', cusPos);
      if (colonPos != std::string::npos) {
        entry.compute_unit_seconds = std::stod(obj.substr(colonPos + 1));
      }
    }

    // Parse timestamp
    size_t tsPos = obj.find("\"timestamp\"");
    if (tsPos != std::string::npos) {
      size_t colonPos = obj.find(':', tsPos);
      if (colonPos != std::string::npos) {
        entry.timestamp = std::stoull(obj.substr(colonPos + 1));
      }
    }

    // Parse kernel_id
    size_t kidPos = obj.find("\"kernel_id\"");
    if (kidPos != std::string::npos) {
      size_t colonPos = obj.find(':', kidPos);
      if (colonPos != std::string::npos) {
        entry.kernel_id = std::stoull(obj.substr(colonPos + 1));
      }
    }

    // Parse direction
    size_t dirPos = obj.find("\"direction\"");
    if (dirPos != std::string::npos) {
      size_t colonPos = obj.find(':', dirPos);
      if (colonPos != std::string::npos) {
        int dir = std::stoi(obj.substr(colonPos + 1));
        entry.direction =
            (dir == 1) ? CreditDirection::CREDIT : CreditDirection::DEBIT;
      }
    }

    if (!entry.node_address.empty()) {
      entries_.push_back(entry);
    }

    pos = objEnd + 1;
  }

  return VGREResult::SUCCESS;
}

VGREResult ResourceLedger::persist() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  std::string path = computeLedgerPath();
  std::string json = toJSON();

  std::ofstream file(path, std::ios::out | std::ios::trunc);
  if (!file.is_open()) {
    VGRE_LOG_ERROR("ResourceLedger",
                   "Failed to open ledger file for writing: " + path);
    return VGREResult::ERROR_IO;
  }

  file << json;
  file.close();

  VGRE_LOG_DEBUG("ResourceLedger",
                 "Persisted " + std::to_string(entries_.size()) +
                     " entries to " + path);
  return VGREResult::SUCCESS;
}

VGREResult ResourceLedger::load() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  std::string path = computeLedgerPath();
  std::ifstream file(path);
  if (!file.is_open()) {
    return VGREResult::ERROR_IO;
  }

  std::ostringstream oss;
  oss << file.rdbuf();
  file.close();

  return fromJSON(oss.str());
}

// ── Singleton ──────────────────────────────────────────────────────────────

ResourceLedger &ResourceLedger::instance() {
  static ResourceLedger inst;
  return inst;
}

} // namespace advanced
} // namespace vgre
