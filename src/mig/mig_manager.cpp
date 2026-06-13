// VGRE MIG — Multi-Instance GPU partitioning (impl).  See mig_manager.h.

#include "vgre/mig/mig_manager.h"

#include "vgre/advanced/secure_channel.h" // random_bytes for UUIDs
#include "vgre/common/logger.h"

#include <cstdlib>
#include <cstring>
#include <sstream>

namespace vgre {
namespace mig {

namespace crypto = vgre::advanced::crypto;

namespace {
// Standard A100/H100 compute-slice → memory-slice mapping.
int memSlicesFor(int computeSlices) {
    switch (computeSlices) {
        case 1: return 1;
        case 2: return 2;
        case 3: return 4;
        case 4: return 4;
        case 7: return 8;
        default: return 0;
    }
}
std::string randomHex(size_t bytes) {
    std::vector<uint8_t> b(bytes);
    crypto::random_bytes(b.data(), b.size());
    static const char* h = "0123456789abcdef";
    std::string s;
    for (uint8_t x : b) { s.push_back(h[x >> 4]); s.push_back(h[x & 0xF]); }
    return s;
}
} // namespace

bool parseProfile(const std::string& s, MigProfile& out) {
    // Accept "Ng" or "Ng.Mgb"; the compute-slice count N is authoritative.
    if (s.size() < 2) return false;
    size_t gpos = s.find('g');
    if (gpos == std::string::npos || gpos == 0) return false;
    int n = 0;
    for (size_t i = 0; i < gpos; ++i) {
        if (s[i] < '0' || s[i] > '9') return false;
        n = n * 10 + (s[i] - '0');
    }
    int m = memSlicesFor(n);
    if (m == 0) return false;
    out.computeSlices = n;
    out.memSlices = m;
    out.name = std::to_string(n) + "g." + std::to_string(m) + "memslice";
    return true;
}

MigManager& MigManager::instance() {
    static MigManager m;
    return m;
}

bool MigManager::placeContiguous_locked(int count, std::vector<bool>& slots, int& outStart) {
    int run = 0;
    for (int i = 0; i < (int)slots.size(); ++i) {
        if (!slots[i]) {
            if (run == 0) outStart = i;
            if (++run == count) {
                for (int j = outStart; j < outStart + count; ++j) slots[j] = true;
                return true;
            }
        } else {
            run = 0;
        }
    }
    return false;
}

void MigManager::configureDevice(uint64_t totalMemBytes, int computeSlices, int memSlices) {
    std::lock_guard<std::mutex> lk(mu_);
    if (configured_) return;
    totalMem_ = totalMemBytes;
    computeSlices_ = computeSlices > 0 ? computeSlices : 7;
    memSlices_ = memSlices > 0 ? memSlices : 8;
    computeUsed_.assign(computeSlices_, false);
    memUsed_.assign(memSlices_, false);
    configured_ = true;
    seedFromEnv_locked();
}

bool MigManager::configured() const {
    std::lock_guard<std::mutex> lk(mu_);
    return configured_;
}

void MigManager::seedFromEnv_locked() {
    // VGRE_MIG_PROFILE="3g.20gb,2g.10gb,2g.10gb" auto-creates instances.
    const char* prof = std::getenv("VGRE_MIG_PROFILE");
    if (prof && *prof) {
        std::stringstream ss(prof);
        std::string item;
        std::vector<std::string> created;
        while (std::getline(ss, item, ',')) {
            if (item.empty()) continue;
            MigProfile p;
            if (!parseProfile(item, p)) continue;
            int cs, ms;
            if (!placeContiguous_locked(p.computeSlices, computeUsed_, cs)) continue;
            if (!placeContiguous_locked(p.memSlices, memUsed_, ms)) {
                for (int j = cs; j < cs + p.computeSlices; ++j) computeUsed_[j] = false; // rollback
                continue;
            }
            GpuInstance gi;
            gi.id = nextGiId_++;
            gi.uuid = "MIG-GPU-" + randomHex(8) + "/" + std::to_string(gi.id);
            gi.profile = p;
            gi.computeStart = cs; gi.computeCount = p.computeSlices;
            gi.memStart = ms; gi.memCount = p.memSlices;
            gi.memBudgetBytes = (totalMem_ / memSlices_) * p.memSlices;
            instances_.push_back(std::move(gi));
            created.push_back(instances_.back().uuid);
        }
        // VGRE_MIG_DEVICE selects the active instance: a UUID or an index.
        if (const char* dev = std::getenv("VGRE_MIG_DEVICE")) {
            std::string d = dev;
            for (auto& gi : instances_) if (gi.uuid == d) { active_ = gi.uuid; break; }
            if (active_.empty()) {
                char* endp = nullptr;
                long idx = std::strtol(d.c_str(), &endp, 10);
                if (endp && *endp == '\0' && idx >= 0 && idx < (long)instances_.size())
                    active_ = instances_[idx].uuid;
            }
        }
        if (!created.empty())
            VGRE_LOG_INFO("MIG", "Seeded " + std::to_string(created.size()) +
                                     " MIG instance(s) from VGRE_MIG_PROFILE");
    }
}

GpuInstance* MigManager::find_locked(const std::string& uuid) {
    for (auto& gi : instances_) if (gi.uuid == uuid) return &gi;
    return nullptr;
}

vgre::VGREResult MigManager::createGpuInstance(const std::string& profile, std::string& outUuid) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!configured_) return vgre::VGREResult::ERR_NOT_INITIALIZED;
    MigProfile p;
    if (!parseProfile(profile, p)) return vgre::VGREResult::ERR_INVALID_VALUE;
    int cs, ms;
    if (!placeContiguous_locked(p.computeSlices, computeUsed_, cs))
        return vgre::VGREResult::ERR_OUT_OF_MEMORY; // no compute capacity/placement
    if (!placeContiguous_locked(p.memSlices, memUsed_, ms)) {
        for (int j = cs; j < cs + p.computeSlices; ++j) computeUsed_[j] = false;
        return vgre::VGREResult::ERR_OUT_OF_MEMORY;
    }
    GpuInstance gi;
    gi.id = nextGiId_++;
    gi.uuid = "MIG-GPU-" + randomHex(8) + "/" + std::to_string(gi.id);
    gi.profile = p;
    gi.computeStart = cs; gi.computeCount = p.computeSlices;
    gi.memStart = ms; gi.memCount = p.memSlices;
    gi.memBudgetBytes = (totalMem_ / memSlices_) * p.memSlices;
    outUuid = gi.uuid;
    instances_.push_back(std::move(gi));
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult MigManager::destroyGpuInstance(const std::string& uuid) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto it = instances_.begin(); it != instances_.end(); ++it) {
        if (it->uuid == uuid) {
            for (int j = it->computeStart; j < it->computeStart + it->computeCount; ++j) computeUsed_[j] = false;
            for (int j = it->memStart; j < it->memStart + it->memCount; ++j) memUsed_[j] = false;
            if (active_ == uuid) active_.clear();
            instances_.erase(it);
            return vgre::VGREResult::SUCCESS;
        }
    }
    return vgre::VGREResult::ERR_NOT_FOUND;
}

std::vector<GpuInstance> MigManager::listInstances() const {
    std::lock_guard<std::mutex> lk(mu_);
    return instances_;
}

bool MigManager::getInstance(const std::string& uuid, GpuInstance& out) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& gi : instances_) if (gi.uuid == uuid) { out = gi; return true; }
    return false;
}

vgre::VGREResult MigManager::createComputeInstance(const std::string& giUuid, int computeSlices,
                                                   std::string& outUuid) {
    std::lock_guard<std::mutex> lk(mu_);
    GpuInstance* gi = find_locked(giUuid);
    if (!gi) return vgre::VGREResult::ERR_NOT_FOUND;
    if (computeSlices <= 0 || gi->computeSlicesUsedByCIs + computeSlices > gi->computeCount)
        return vgre::VGREResult::ERR_OUT_OF_MEMORY; // exceeds the GI's compute slices
    ComputeInstance ci;
    ci.id = nextCiId_++;
    ci.computeSlices = computeSlices;
    ci.uuid = giUuid + "/" + std::to_string(ci.id);
    gi->computeSlicesUsedByCIs += computeSlices;
    outUuid = ci.uuid;
    gi->computeInstances.push_back(std::move(ci));
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult MigManager::instanceAllocate(const std::string& uuid, uint64_t bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    GpuInstance* gi = find_locked(uuid);
    if (!gi) return vgre::VGREResult::ERR_NOT_FOUND;
    if (gi->memUsedBytes + bytes > gi->memBudgetBytes)
        return vgre::VGREResult::ERR_OUT_OF_MEMORY; // tenant budget exceeded — isolation
    gi->memUsedBytes += bytes;
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult MigManager::instanceFree(const std::string& uuid, uint64_t bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    GpuInstance* gi = find_locked(uuid);
    if (!gi) return vgre::VGREResult::ERR_NOT_FOUND;
    gi->memUsedBytes = (bytes >= gi->memUsedBytes) ? 0 : gi->memUsedBytes - bytes;
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult MigManager::setActiveInstance(const std::string& uuid) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!find_locked(uuid)) return vgre::VGREResult::ERR_NOT_FOUND;
    active_ = uuid;
    return vgre::VGREResult::SUCCESS;
}

void MigManager::clearActiveInstance() {
    std::lock_guard<std::mutex> lk(mu_);
    active_.clear();
}

std::string MigManager::activeInstance() const {
    std::lock_guard<std::mutex> lk(mu_);
    return active_;
}

uint64_t MigManager::effectiveDeviceMemory(uint64_t physicalTotal) {
    std::lock_guard<std::mutex> lk(mu_);
    if (active_.empty()) return physicalTotal;
    for (const auto& gi : instances_) if (gi.uuid == active_) return gi.memBudgetBytes;
    return physicalTotal;
}

uint64_t MigManager::effectiveUsedMemory(uint64_t physicalUsed) {
    std::lock_guard<std::mutex> lk(mu_);
    if (active_.empty()) return physicalUsed;
    for (const auto& gi : instances_) if (gi.uuid == active_)
        return physicalUsed < gi.memBudgetBytes ? physicalUsed : gi.memBudgetBytes;
    return physicalUsed;
}

void MigManager::reset() {
    std::lock_guard<std::mutex> lk(mu_);
    instances_.clear();
    active_.clear();
    if (configured_) {
        computeUsed_.assign(computeSlices_, false);
        memUsed_.assign(memSlices_, false);
    }
    nextGiId_ = 0;
    nextCiId_ = 0;
}

int MigManager::freeComputeSlices() const {
    std::lock_guard<std::mutex> lk(mu_);
    int n = 0; for (bool b : computeUsed_) if (!b) ++n; return n;
}
int MigManager::freeMemorySlices() const {
    std::lock_guard<std::mutex> lk(mu_);
    int n = 0; for (bool b : memUsed_) if (!b) ++n; return n;
}

} // namespace mig
} // namespace vgre
