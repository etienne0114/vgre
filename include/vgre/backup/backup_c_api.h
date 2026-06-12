// Stable C ABI for the backup / DR archive — the shipped operator-facing surface
// for snapshotting and restoring runtime/config/checkpoint files.
#ifndef VGRE_BACKUP_C_API_H
#define VGRE_BACKUP_C_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Snapshot a set of files into archive `dir`. `names`/`paths` are parallel
// arrays of length `count`; each file at paths[i] is stored under logical name
// names[i]. If `hexKey32` is non-NULL it must be 64 hex chars (32 bytes) and the
// archive is AES-256-GCM encrypted. On success writes the snapshot id (hex, up
// to 65 bytes incl. NUL) into `outId`. Returns 0 on success, non-zero on error.
int vgre_backup_snapshot_files(const char* dir, const char* hexKey32,
                               const char* label, const char* const* names,
                               const char* const* paths, size_t count,
                               char* outId, size_t outIdCap);

// Restore a snapshot's entries to files. `names`/`paths` parallel arrays select
// which entries to restore and where. Every object's checksum is verified.
// Returns 0 on success.
int vgre_backup_restore_files(const char* dir, const char* hexKey32,
                              const char* snapshotId, const char* const* names,
                              const char* const* paths, size_t count);

// Verify a snapshot's integrity (recompute all object hashes). Returns 0 if intact.
int vgre_backup_verify(const char* dir, const char* hexKey32, const char* snapshotId);

// Keep only the newest `keepLast` snapshots; GC unreferenced objects.
// Writes the number removed to *outRemoved if non-NULL. Returns 0 on success.
int vgre_backup_prune(const char* dir, const char* hexKey32, size_t keepLast,
                      size_t* outRemoved);

#ifdef __cplusplus
}
#endif

#endif // VGRE_BACKUP_C_API_H
