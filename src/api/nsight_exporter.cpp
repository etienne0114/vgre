/**
 * @file nsight_exporter.cpp
 * @brief Track Q — export the VGRE telemetry timeline to NVIDIA Nsight Systems'
 *        SQLite-based .nsys-rep format so traces open in familiar tooling.
 *
 * The .nsys-rep container is a SQLite database.  We emit the minimum viable
 * schema documented by Nsight Systems:
 *   - StringIds(id, value)             : interned kernel / API names
 *   - KernelLaunches(...)              : kernel name id, start/end ns, grid/block
 *   - MemcpyOps(...)                   : direction, size, start/end ns
 *   - NvtxEvents(...)                  : domain, range name, start/end ns
 *
 * SQLite is linked from the system library; only the small, ABI-stable subset of
 * the C API used here is declared locally (no -dev header required).
 */

#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/advanced/runtime_profiler.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#if !defined(VGRE_HAS_SQLITE)
// SQLite unavailable at build time: provide a clear, non-crashing no-op so the
// API symbol still exists and reports the missing capability.
extern "C" VGRE_EXPORT int vgre_export_nsight_trace(const char *output_path) {
    (void)output_path;
    VGRE_LOG_ERROR("nsight_exporter",
                   "Nsight .nsys-rep export unavailable: VGRE was built without "
                   "SQLite. Install libsqlite3 and rebuild.");
    return VGRE_ERROR_NOT_SUPPORTED;
}
#else

// ── Minimal SQLite C API (ABI-stable; avoids a build-time -dev dependency) ────
extern "C" {
typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;
int  sqlite3_open(const char *filename, sqlite3 **ppDb);
int  sqlite3_close(sqlite3 *db);
int  sqlite3_exec(sqlite3 *, const char *sql,
                  int (*cb)(void *, int, char **, char **), void *, char **errmsg);
int  sqlite3_prepare_v2(sqlite3 *db, const char *zSql, int nByte,
                        sqlite3_stmt **ppStmt, const char **pzTail);
int  sqlite3_bind_int64(sqlite3_stmt *, int, long long);
int  sqlite3_bind_text(sqlite3_stmt *, int, const char *, int,
                       void (*)(void *));
int  sqlite3_step(sqlite3_stmt *);
int  sqlite3_reset(sqlite3_stmt *);
int  sqlite3_finalize(sqlite3_stmt *);
const char *sqlite3_errmsg(sqlite3 *);
}

#define SQLITE_OK    0
#define SQLITE_ROW   100
#define SQLITE_DONE  101
// SQLITE_TRANSIENT tells SQLite to copy the bound string.
#define SQLITE_TRANSIENT_FN reinterpret_cast<void (*)(void *)>(-1)

namespace {

bool execOrLog(sqlite3 *db, const char *sql) {
    char *err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        VGRE_LOG_ERROR("nsight_exporter",
                       std::string("SQL failed: ") + (err ? err : "?"));
        return false;
    }
    return true;
}

uint64_t toNs(const std::chrono::steady_clock::time_point &tp) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            tp.time_since_epoch()).count());
}

} // namespace

extern "C" VGRE_EXPORT int vgre_export_nsight_trace(const char *output_path) {
    if (!output_path) return VGRE_ERROR_INVALID_VALUE;

    auto &prof = vgre::advanced::RuntimeProfiler::instance();
    std::vector<vgre::advanced::ProfileEvent> events = prof.getAllEvents();

    sqlite3 *db = nullptr;
    if (sqlite3_open(output_path, &db) != SQLITE_OK || !db) {
        VGRE_LOG_ERROR("nsight_exporter",
                       std::string("Cannot open .nsys-rep for write: ") + output_path);
        if (db) sqlite3_close(db);
        return VGRE_ERROR_IO;
    }

    // Schema (Nsight Systems minimum viable subset).
    const char *schema =
        "PRAGMA journal_mode=MEMORY;"
        "BEGIN;"
        "CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);"
        "CREATE TABLE KernelLaunches("
        " nameId INTEGER, start INTEGER, end INTEGER,"
        " gridX INTEGER, gridY INTEGER, gridZ INTEGER,"
        " blockX INTEGER, blockY INTEGER, blockZ INTEGER);"
        "CREATE TABLE MemcpyOps("
        " direction INTEGER, bytes INTEGER, start INTEGER, end INTEGER);"
        "CREATE TABLE NvtxEvents("
        " domain TEXT, rangeName TEXT, start INTEGER, end INTEGER);";
    if (!execOrLog(db, schema)) { sqlite3_close(db); return VGRE_ERROR_IO; }

    uint64_t base_ns = events.empty() ? 0 : toNs(events[0].timestamp);

    // Intern names into StringIds.
    std::unordered_map<std::string, int64_t> stringIds;
    sqlite3_stmt *insStr = nullptr;
    sqlite3_prepare_v2(db, "INSERT INTO StringIds(id,value) VALUES(?,?);", -1, &insStr, nullptr);
    auto internString = [&](const std::string &s) -> int64_t {
        auto it = stringIds.find(s);
        if (it != stringIds.end()) return it->second;
        int64_t id = static_cast<int64_t>(stringIds.size());
        stringIds[s] = id;
        sqlite3_reset(insStr);
        sqlite3_bind_int64(insStr, 1, id);
        sqlite3_bind_text(insStr, 2, s.c_str(), -1, SQLITE_TRANSIENT_FN);
        sqlite3_step(insStr);
        return id;
    };

    sqlite3_stmt *insKernel = nullptr;
    sqlite3_prepare_v2(db,
        "INSERT INTO KernelLaunches(nameId,start,end,gridX,gridY,gridZ,blockX,blockY,blockZ)"
        " VALUES(?,?,?,?,?,?,?,?,?);", -1, &insKernel, nullptr);
    sqlite3_stmt *insMemcpy = nullptr;
    sqlite3_prepare_v2(db,
        "INSERT INTO MemcpyOps(direction,bytes,start,end) VALUES(?,?,?,?);",
        -1, &insMemcpy, nullptr);

    size_t nKernels = 0, nMemcpys = 0;
    for (const auto &ev : events) {
        uint64_t start = toNs(ev.timestamp) - base_ns;
        uint64_t end   = start + static_cast<uint64_t>(ev.durationMs * 1e6);

        if (ev.gridDim.x > 0) {
            int64_t nameId = internString(ev.kernelName.empty() ? "kernel" : ev.kernelName);
            sqlite3_reset(insKernel);
            sqlite3_bind_int64(insKernel, 1, nameId);
            sqlite3_bind_int64(insKernel, 2, static_cast<long long>(start));
            sqlite3_bind_int64(insKernel, 3, static_cast<long long>(end));
            sqlite3_bind_int64(insKernel, 4, ev.gridDim.x);
            sqlite3_bind_int64(insKernel, 5, ev.gridDim.y);
            sqlite3_bind_int64(insKernel, 6, ev.gridDim.z);
            sqlite3_bind_int64(insKernel, 7, ev.blockDim.x);
            sqlite3_bind_int64(insKernel, 8, ev.blockDim.y);
            sqlite3_bind_int64(insKernel, 9, ev.blockDim.z);
            sqlite3_step(insKernel);
            ++nKernels;
        } else if (ev.memoryBytes > 0) {
            // Direction is not tracked per-event; derive a hint from the name
            // (0=unknown, 1=H2D, 2=D2H, 3=D2D) so the column is still meaningful.
            int64_t dir = 0;
            const std::string &n = ev.kernelName;
            if      (n.find("H2D") != std::string::npos) dir = 1;
            else if (n.find("D2H") != std::string::npos) dir = 2;
            else if (n.find("D2D") != std::string::npos) dir = 3;
            sqlite3_reset(insMemcpy);
            sqlite3_bind_int64(insMemcpy, 1, dir);
            sqlite3_bind_int64(insMemcpy, 2, static_cast<long long>(ev.memoryBytes));
            sqlite3_bind_int64(insMemcpy, 3, static_cast<long long>(start));
            sqlite3_bind_int64(insMemcpy, 4, static_cast<long long>(end));
            sqlite3_step(insMemcpy);
            ++nMemcpys;
        }
    }

    sqlite3_finalize(insStr);
    sqlite3_finalize(insKernel);
    sqlite3_finalize(insMemcpy);
    execOrLog(db, "COMMIT;");
    sqlite3_close(db);

    VGRE_LOG_INFO("nsight_exporter",
                  "Exported " + std::to_string(nKernels) + " kernels, " +
                  std::to_string(nMemcpys) + " memcpys to Nsight trace: " +
                  output_path);
    return VGRE_SUCCESS;
}

#endif // VGRE_HAS_SQLITE
