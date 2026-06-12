// VGRE Backup / Disaster Recovery — content-addressed snapshot archive
// ---------------------------------------------------------------------------
// docs/missingFeatures.md §9.1 (Enterprise Backup & Disaster Recovery).  A
// self-contained, cross-platform backup engine for checkpoints / model weights
// / configuration / security state:
//   * Content-addressed object store (objects keyed by SHA-256) → automatic
//     deduplication, so successive snapshots that share data are incremental in
//     space without any explicit diffing.
//   * Point-in-time snapshots with a hash-chained manifest lineage (prevHash),
//     giving tamper-evident history and recovery to any prior snapshot.
//   * Optional AES-256-GCM encryption of every object at rest (DR-at-rest).
//   * Restore *validation*: every restore (and verifySnapshot()) recomputes the
//     SHA-256 of each object and checks it against the manifest — the automated
//     restore-testing §9.1 asks for.
//   * Replication primitive: exportSnapshot() copies a snapshot + its objects to
//     another archive directory (e.g. a cross-region mount); the remote transfer
//     itself is the only out-of-process part.

#ifndef VGRE_BACKUP_BACKUP_ARCHIVE_H
#define VGRE_BACKUP_BACKUP_ARCHIVE_H

#include "vgre/common/error_codes.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace vgre {
namespace backup {

struct SnapshotInfo {
    std::string id;            // hex(manifestHash)
    std::string label;
    uint64_t    timestamp_ns = 0;
    uint64_t    totalBytes = 0;   // sum of entry sizes (logical)
    uint32_t    objectCount = 0;
    std::array<uint8_t, 32> manifestHash{};
    std::array<uint8_t, 32> prevHash{};   // previous snapshot (lineage), zero for first
};

class BackupArchive {
public:
    ~BackupArchive();

    // Open/create an archive rooted at `dir`.  If `encKey` is non-null (32
    // bytes) every stored object is AES-256-GCM-sealed; the same key is required
    // to read.  Pass nullptr for plaintext archives.
    static vgre::VGREResult open(const std::string& dir, const uint8_t* encKey,
                                 std::unique_ptr<BackupArchive>& out);

    // Snapshot a set of named in-memory blobs.  Returns the snapshot id.
    vgre::VGREResult createSnapshot(const std::string& label,
                                    const std::map<std::string, std::string>& blobs,
                                    std::string& outId);

    // Snapshot a set of named files (name → source path).
    vgre::VGREResult createSnapshotFromFiles(const std::string& label,
                                             const std::map<std::string, std::string>& nameToPath,
                                             std::string& outId);

    // Restore a snapshot's blobs, verifying every object's checksum.
    vgre::VGREResult restoreSnapshot(const std::string& id,
                                     std::map<std::string, std::string>& outBlobs);

    // Restore a snapshot's entries to files (name → destination path).
    vgre::VGREResult restoreSnapshotToFiles(const std::string& id,
                                            const std::map<std::string, std::string>& nameToPath);

    // Integrity check: recompute every object hash; SUCCESS iff all match.
    vgre::VGREResult verifySnapshot(const std::string& id);

    std::vector<SnapshotInfo> listSnapshots() const;
    bool getSnapshot(const std::string& id, SnapshotInfo& out) const;

    // Keep the newest `keepLast` snapshots; delete older manifests and garbage-
    // collect now-unreferenced objects.  Returns count of snapshots removed.
    vgre::VGREResult pruneSnapshots(size_t keepLast, size_t& outRemoved);

    // Replicate one snapshot (manifest + referenced objects) into `destDir`,
    // which becomes/extends a valid archive (must use the same encryption key).
    vgre::VGREResult exportSnapshot(const std::string& id, const std::string& destDir);

    size_t snapshotCount() const;

private:
    BackupArchive();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace backup
} // namespace vgre

#endif // VGRE_BACKUP_BACKUP_ARCHIVE_H
