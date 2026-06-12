// Stable C ABI for the backup / DR archive.  See backup_c_api.h.

#include "vgre/backup/backup_c_api.h"
#include "vgre/backup/backup_archive.h"

#include <cstring>
#include <map>
#include <memory>
#include <string>

namespace {

int hexVal(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
// Returns true and fills key[32] if hexKey is exactly 64 hex chars; false if
// hexKey is null/empty (→ plaintext archive); -1-style error not distinguished.
bool parseKey(const char* hexKey, unsigned char key[32], bool& haveKey) {
    haveKey = false;
    if (!hexKey || !*hexKey) return true; // no key → plaintext
    size_t n = std::strlen(hexKey);
    if (n != 64) return false;
    for (int i = 0; i < 32; ++i) {
        int hi = hexVal(hexKey[2 * i]), lo = hexVal(hexKey[2 * i + 1]);
        if (hi < 0 || lo < 0) return false;
        key[i] = static_cast<unsigned char>((hi << 4) | lo);
    }
    haveKey = true;
    return true;
}

bool openArchive(const char* dir, const char* hexKey,
                 std::unique_ptr<vgre::backup::BackupArchive>& out) {
    unsigned char key[32];
    bool haveKey = false;
    if (!parseKey(hexKey, key, haveKey)) return false;
    return vgre::backup::BackupArchive::open(dir, haveKey ? key : nullptr, out) ==
           vgre::VGREResult::SUCCESS;
}

} // namespace

extern "C" {

int vgre_backup_snapshot_files(const char* dir, const char* hexKey32, const char* label,
                               const char* const* names, const char* const* paths,
                               size_t count, char* outId, size_t outIdCap) {
    if (!dir || (count && (!names || !paths))) return 1;
    std::unique_ptr<vgre::backup::BackupArchive> ar;
    if (!openArchive(dir, hexKey32, ar)) return 2;
    std::map<std::string, std::string> nameToPath;
    for (size_t i = 0; i < count; ++i) {
        if (!names[i] || !paths[i]) return 1;
        nameToPath[names[i]] = paths[i];
    }
    std::string id;
    if (ar->createSnapshotFromFiles(label ? label : "", nameToPath, id) !=
        vgre::VGREResult::SUCCESS)
        return 3;
    if (outId && outIdCap) {
        std::strncpy(outId, id.c_str(), outIdCap - 1);
        outId[outIdCap - 1] = '\0';
    }
    return 0;
}

int vgre_backup_restore_files(const char* dir, const char* hexKey32, const char* snapshotId,
                              const char* const* names, const char* const* paths, size_t count) {
    if (!dir || !snapshotId || (count && (!names || !paths))) return 1;
    std::unique_ptr<vgre::backup::BackupArchive> ar;
    if (!openArchive(dir, hexKey32, ar)) return 2;
    std::map<std::string, std::string> nameToPath;
    for (size_t i = 0; i < count; ++i) {
        if (!names[i] || !paths[i]) return 1;
        nameToPath[names[i]] = paths[i];
    }
    return ar->restoreSnapshotToFiles(snapshotId, nameToPath) == vgre::VGREResult::SUCCESS ? 0 : 3;
}

int vgre_backup_verify(const char* dir, const char* hexKey32, const char* snapshotId) {
    if (!dir || !snapshotId) return 1;
    std::unique_ptr<vgre::backup::BackupArchive> ar;
    if (!openArchive(dir, hexKey32, ar)) return 2;
    return ar->verifySnapshot(snapshotId) == vgre::VGREResult::SUCCESS ? 0 : 3;
}

int vgre_backup_prune(const char* dir, const char* hexKey32, size_t keepLast, size_t* outRemoved) {
    if (!dir) return 1;
    std::unique_ptr<vgre::backup::BackupArchive> ar;
    if (!openArchive(dir, hexKey32, ar)) return 2;
    size_t removed = 0;
    vgre::VGREResult r = ar->pruneSnapshots(keepLast, removed);
    if (outRemoved) *outRemoved = removed;
    return r == vgre::VGREResult::SUCCESS ? 0 : 3;
}

} // extern "C"
