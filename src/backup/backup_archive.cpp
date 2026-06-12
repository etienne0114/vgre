// VGRE Backup / DR — content-addressed snapshot archive (impl).
// See include/vgre/backup/backup_archive.h.

#include "vgre/backup/backup_archive.h"

#include "vgre/advanced/secure_channel.h" // crypto: sha256, aes256_gcm, random_bytes
#include "vgre/common/logger.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>

namespace fs = std::filesystem;

namespace vgre {
namespace backup {

namespace crypto = vgre::advanced::crypto;

namespace {

void putU32(std::string& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(char((v >> (8 * i)) & 0xFF)); }
void putU64(std::string& b, uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back(char((v >> (8 * i)) & 0xFF)); }
void putStr(std::string& b, const std::string& s) { putU32(b, uint32_t(s.size())); b.append(s); }
void putRaw(std::string& b, const uint8_t* p, size_t n) { b.append(reinterpret_cast<const char*>(p), n); }

struct Reader {
    const char* p; const char* end; bool ok = true;
    explicit Reader(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}
    uint8_t u8() { if (p + 1 > end) { ok = false; return 0; } return uint8_t(*p++); }
    uint32_t u32() { if (p + 4 > end) { ok = false; return 0; } uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= uint32_t(uint8_t(p[i])) << (8 * i); p += 4; return v; }
    uint64_t u64() { if (p + 8 > end) { ok = false; return 0; } uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= uint64_t(uint8_t(p[i])) << (8 * i); p += 8; return v; }
    std::string str() { uint32_t n = u32(); if (!ok || p + n > end) { ok = false; return {}; } std::string s(p, p + n); p += n; return s; }
    bool raw(uint8_t* out, size_t n) { if (p + n > end) { ok = false; return false; } std::memcpy(out, p, n); p += n; return true; }
};

constexpr char kManifestMagic[8] = {'V', 'G', 'R', 'E', 'B', 'K', 'P', '1'};

uint64_t nowNs() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}
std::string hex(const uint8_t* d, size_t n) {
    static const char* h = "0123456789abcdef";
    std::string s; s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) { s.push_back(h[d[i] >> 4]); s.push_back(h[d[i] & 0xF]); }
    return s;
}
bool readFile(const fs::path& p, std::string& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    out.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return true;
}
bool writeFileAtomic(const fs::path& p, const std::string& data) {
    fs::path tmp = p; tmp += ".tmp";
    { std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
      if (!f) return false;
      f.write(data.data(), std::streamsize(data.size()));
      f.flush();
      if (!f) return false; }
    std::error_code ec;
    fs::rename(tmp, p, ec);
    if (ec) { fs::remove(tmp, ec); return false; }
    return true;
}

struct Entry { std::string name; std::array<uint8_t, 32> hash{}; uint64_t size = 0; };

} // namespace

struct BackupArchive::Impl {
    fs::path dir;
    bool encrypted = false;
    std::array<uint8_t, 32> key{};
    std::vector<SnapshotInfo> snapshots; // loaded manifests (header only)
    mutable std::mutex mu;

    fs::path objectsDir() const { return dir / "objects"; }
    fs::path snapshotsDir() const { return dir / "snapshots"; }
    fs::path objectPath(const std::array<uint8_t, 32>& h) const {
        return objectsDir() / hex(h.data(), 32);
    }

    // Write object content-addressed; dedups if already present.
    bool writeObject(const std::array<uint8_t, 32>& hash, const std::string& plain) const {
        fs::path op = objectPath(hash);
        std::error_code ec;
        if (fs::exists(op, ec)) return true; // dedup
        std::string onDisk;
        if (encrypted) {
            onDisk.resize(12 + 16 + plain.size());
            auto* iv = reinterpret_cast<uint8_t*>(&onDisk[0]);
            auto* tag = reinterpret_cast<uint8_t*>(&onDisk[12]);
            auto* ct = reinterpret_cast<uint8_t*>(&onDisk[28]);
            crypto::random_bytes(iv, 12);
            crypto::aes256_gcm_encrypt(key.data(), iv, hash.data(), 32,
                                       reinterpret_cast<const uint8_t*>(plain.data()), plain.size(),
                                       ct, tag);
        } else {
            onDisk = plain;
        }
        return writeFileAtomic(op, onDisk);
    }

    // Read + (decrypt) + verify an object against its expected hash.
    bool readObject(const std::array<uint8_t, 32>& hash, std::string& plain) const {
        std::string onDisk;
        if (!readFile(objectPath(hash), onDisk)) return false;
        if (encrypted) {
            if (onDisk.size() < 28) return false;
            const auto* iv = reinterpret_cast<const uint8_t*>(&onDisk[0]);
            const auto* tag = reinterpret_cast<const uint8_t*>(&onDisk[12]);
            const auto* ct = reinterpret_cast<const uint8_t*>(&onDisk[28]);
            size_t ctLen = onDisk.size() - 28;
            plain.assign(ctLen, '\0');
            if (!crypto::aes256_gcm_decrypt(key.data(), iv, hash.data(), 32, ct, ctLen,
                                            reinterpret_cast<uint8_t*>(&plain[0]), tag))
                return false;
        } else {
            plain = std::move(onDisk);
        }
        std::array<uint8_t, 32> calc{};
        crypto::sha256(reinterpret_cast<const uint8_t*>(plain.data()), plain.size(), calc.data());
        return crypto::secure_compare(calc.data(), hash.data(), 32);
    }

    std::string serializeManifest(const std::string& label, uint64_t ts,
                                  const std::array<uint8_t, 32>& prev,
                                  const std::vector<Entry>& entries) const {
        std::string b(kManifestMagic, sizeof(kManifestMagic));
        putU64(b, ts);
        b.push_back(encrypted ? 1 : 0);
        putStr(b, label);
        putRaw(b, prev.data(), 32);
        putU32(b, uint32_t(entries.size()));
        for (const auto& e : entries) {
            putStr(b, e.name);
            putRaw(b, e.hash.data(), 32);
            putU64(b, e.size);
        }
        return b;
    }

    bool parseManifest(const std::string& data, SnapshotInfo& info, std::vector<Entry>& entries) const {
        if (data.size() < sizeof(kManifestMagic) ||
            std::memcmp(data.data(), kManifestMagic, sizeof(kManifestMagic)) != 0)
            return false;
        std::string body(data.begin() + sizeof(kManifestMagic), data.end());
        Reader rd(body);
        info.timestamp_ns = rd.u64();
        rd.u8(); // encrypted flag (informational)
        info.label = rd.str();
        rd.raw(info.prevHash.data(), 32);
        uint32_t n = rd.u32();
        info.totalBytes = 0;
        for (uint32_t i = 0; i < n && rd.ok; ++i) {
            Entry e;
            e.name = rd.str();
            rd.raw(e.hash.data(), 32);
            e.size = rd.u64();
            info.totalBytes += e.size;
            entries.push_back(std::move(e));
        }
        if (!rd.ok) return false;
        info.objectCount = n;
        crypto::sha256(reinterpret_cast<const uint8_t*>(data.data()), data.size(),
                       info.manifestHash.data());
        info.id = hex(info.manifestHash.data(), 32);
        return true;
    }

    SnapshotInfo* latest() {
        SnapshotInfo* best = nullptr;
        for (auto& s : snapshots)
            if (!best || s.timestamp_ns > best->timestamp_ns) best = &s;
        return best;
    }

    bool loadManifestEntries(const std::string& id, SnapshotInfo& info,
                             std::vector<Entry>& entries) const {
        std::string data;
        if (!readFile(snapshotsDir() / id, data)) return false;
        return parseManifest(data, info, entries);
    }
};

BackupArchive::BackupArchive() : impl_(new Impl()) {}
BackupArchive::~BackupArchive() = default;

vgre::VGREResult BackupArchive::open(const std::string& dir, const uint8_t* encKey,
                                     std::unique_ptr<BackupArchive>& out) {
    std::unique_ptr<BackupArchive> a(new BackupArchive());
    Impl& im = *a->impl_;
    im.dir = dir;
    im.encrypted = encKey != nullptr;
    if (encKey) std::memcpy(im.key.data(), encKey, 32);

    std::error_code ec;
    fs::create_directories(im.objectsDir(), ec);
    fs::create_directories(im.snapshotsDir(), ec);
    if (ec) return vgre::VGREResult::ERR_IO;

    // Load existing snapshot manifests (headers).
    for (const auto& de : fs::directory_iterator(im.snapshotsDir(), ec)) {
        if (ec) break;
        if (!de.is_regular_file()) continue;
        if (de.path().extension() == ".tmp") continue;
        std::string data;
        if (!readFile(de.path(), data)) continue;
        SnapshotInfo info;
        std::vector<Entry> entries;
        if (im.parseManifest(data, info, entries)) im.snapshots.push_back(std::move(info));
    }
    out = std::move(a);
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult BackupArchive::createSnapshot(const std::string& label,
                                               const std::map<std::string, std::string>& blobs,
                                               std::string& outId) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::vector<Entry> entries; // std::map already ordered by name → deterministic
    for (const auto& kv : blobs) {
        Entry e;
        e.name = kv.first;
        e.size = kv.second.size();
        crypto::sha256(reinterpret_cast<const uint8_t*>(kv.second.data()), kv.second.size(),
                       e.hash.data());
        if (!impl_->writeObject(e.hash, kv.second)) return vgre::VGREResult::ERR_IO;
        entries.push_back(std::move(e));
    }
    std::array<uint8_t, 32> prev{};
    if (SnapshotInfo* l = impl_->latest()) prev = l->manifestHash;

    uint64_t ts = nowNs();
    std::string manifest = impl_->serializeManifest(label, ts, prev, entries);
    SnapshotInfo info;
    std::vector<Entry> parsed;
    if (!impl_->parseManifest(manifest, info, parsed)) return vgre::VGREResult::ERR_IO;
    if (!writeFileAtomic(impl_->snapshotsDir() / info.id, manifest)) return vgre::VGREResult::ERR_IO;
    impl_->snapshots.push_back(info);
    outId = info.id;
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult BackupArchive::createSnapshotFromFiles(
    const std::string& label, const std::map<std::string, std::string>& nameToPath,
    std::string& outId) {
    std::map<std::string, std::string> blobs;
    for (const auto& kv : nameToPath) {
        std::string data;
        if (!readFile(kv.second, data)) return vgre::VGREResult::ERR_IO;
        blobs[kv.first] = std::move(data);
    }
    return createSnapshot(label, blobs, outId);
}

vgre::VGREResult BackupArchive::restoreSnapshot(const std::string& id,
                                                std::map<std::string, std::string>& outBlobs) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    SnapshotInfo info;
    std::vector<Entry> entries;
    if (!impl_->loadManifestEntries(id, info, entries)) return vgre::VGREResult::ERR_NOT_FOUND;
    for (const auto& e : entries) {
        std::string plain;
        if (!impl_->readObject(e.hash, plain)) return vgre::VGREResult::ERR_CRYPTO; // missing/corrupt/tampered
        outBlobs[e.name] = std::move(plain);
    }
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult BackupArchive::restoreSnapshotToFiles(
    const std::string& id, const std::map<std::string, std::string>& nameToPath) {
    std::map<std::string, std::string> blobs;
    vgre::VGREResult r = restoreSnapshot(id, blobs);
    if (r != vgre::VGREResult::SUCCESS) return r;
    for (const auto& kv : nameToPath) {
        auto it = blobs.find(kv.first);
        if (it == blobs.end()) return vgre::VGREResult::ERR_NOT_FOUND;
        if (!writeFileAtomic(kv.second, it->second)) return vgre::VGREResult::ERR_IO;
    }
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult BackupArchive::verifySnapshot(const std::string& id) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    SnapshotInfo info;
    std::vector<Entry> entries;
    if (!impl_->loadManifestEntries(id, info, entries)) return vgre::VGREResult::ERR_NOT_FOUND;
    for (const auto& e : entries) {
        std::string plain;
        if (!impl_->readObject(e.hash, plain)) return vgre::VGREResult::ERR_CRYPTO;
    }
    return vgre::VGREResult::SUCCESS;
}

std::vector<SnapshotInfo> BackupArchive::listSnapshots() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::vector<SnapshotInfo> v = impl_->snapshots;
    std::sort(v.begin(), v.end(),
              [](const SnapshotInfo& a, const SnapshotInfo& b) { return a.timestamp_ns < b.timestamp_ns; });
    return v;
}

bool BackupArchive::getSnapshot(const std::string& id, SnapshotInfo& out) const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    for (const auto& s : impl_->snapshots)
        if (s.id == id) { out = s; return true; }
    return false;
}

vgre::VGREResult BackupArchive::pruneSnapshots(size_t keepLast, size_t& outRemoved) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    outRemoved = 0;
    std::sort(impl_->snapshots.begin(), impl_->snapshots.end(),
              [](const SnapshotInfo& a, const SnapshotInfo& b) { return a.timestamp_ns > b.timestamp_ns; });
    if (impl_->snapshots.size() <= keepLast) return vgre::VGREResult::SUCCESS;

    std::vector<SnapshotInfo> keep(impl_->snapshots.begin(), impl_->snapshots.begin() + keepLast);
    std::vector<SnapshotInfo> drop(impl_->snapshots.begin() + keepLast, impl_->snapshots.end());

    std::error_code ec;
    for (const auto& s : drop) {
        fs::remove(impl_->snapshotsDir() / s.id, ec);
        ++outRemoved;
    }
    impl_->snapshots = keep;

    // GC: delete object files no longer referenced by any surviving manifest.
    std::set<std::string> referenced;
    for (const auto& s : keep) {
        SnapshotInfo info; std::vector<Entry> entries;
        if (impl_->loadManifestEntries(s.id, info, entries))
            for (const auto& e : entries) referenced.insert(hex(e.hash.data(), 32));
    }
    for (const auto& de : fs::directory_iterator(impl_->objectsDir(), ec)) {
        if (ec) break;
        if (!de.is_regular_file()) continue;
        if (de.path().extension() == ".tmp") { fs::remove(de.path(), ec); continue; }
        if (!referenced.count(de.path().filename().string()))
            fs::remove(de.path(), ec);
    }
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult BackupArchive::exportSnapshot(const std::string& id, const std::string& destDir) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    SnapshotInfo info;
    std::vector<Entry> entries;
    if (!impl_->loadManifestEntries(id, info, entries)) return vgre::VGREResult::ERR_NOT_FOUND;

    std::error_code ec;
    fs::path dst(destDir);
    fs::create_directories(dst / "objects", ec);
    fs::create_directories(dst / "snapshots", ec);
    if (ec) return vgre::VGREResult::ERR_IO;

    for (const auto& e : entries) {
        std::string objName = hex(e.hash.data(), 32);
        fs::path src = impl_->objectsDir() / objName;
        fs::path dstObj = dst / "objects" / objName;
        if (fs::exists(dstObj, ec)) continue;
        std::string data;
        if (!readFile(src, data) || !writeFileAtomic(dstObj, data)) return vgre::VGREResult::ERR_IO;
    }
    std::string manifest;
    if (!readFile(impl_->snapshotsDir() / id, manifest)) return vgre::VGREResult::ERR_IO;
    if (!writeFileAtomic(dst / "snapshots" / id, manifest)) return vgre::VGREResult::ERR_IO;
    return vgre::VGREResult::SUCCESS;
}

size_t BackupArchive::snapshotCount() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->snapshots.size();
}

} // namespace backup
} // namespace vgre
