// Track Q — Nsight Systems .nsys-rep export test.
//
// Enables profiling, launches a kernel (recorded by RuntimeProfiler), exports an
// .nsys-rep, then opens the resulting SQLite database and verifies the schema
// and that the kernel launch was recorded with a resolvable name.

#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"
#include "vgre/advanced/runtime_profiler.h"
#include "vgre/api/vgre_c_api.h"

#include <cstdio>
#include <cstring>
#include <string>

// Minimal SQLite read-back API (ABI-stable).
extern "C" {
typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;
int sqlite3_open(const char *, sqlite3 **);
int sqlite3_close(sqlite3 *);
int sqlite3_prepare_v2(sqlite3 *, const char *, int, sqlite3_stmt **, const char **);
int sqlite3_step(sqlite3_stmt *);
long long sqlite3_column_int64(sqlite3_stmt *, int);
const unsigned char *sqlite3_column_text(sqlite3_stmt *, int);
int sqlite3_finalize(sqlite3_stmt *);
}
#define SQLITE_ROW 100

using namespace vgre;

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static long long scalarQuery(sqlite3 *db, const char *sql, std::string *firstText = nullptr) {
    sqlite3_stmt *st = nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &st, nullptr) != 0 || !st) return -1;
    long long v = -1;
    if (sqlite3_step(st) == SQLITE_ROW) {
        v = sqlite3_column_int64(st, 0);
        if (firstText) {
            const unsigned char *t = sqlite3_column_text(st, 1);
            if (t) *firstText = reinterpret_cast<const char *>(t);
        }
    }
    sqlite3_finalize(st);
    return v;
}

int main() {
    printf("=== Nsight .nsys-rep Export Tests (Track Q) ===\n");

    advanced::RuntimeProfiler::instance().setEnabled(true);

    core::RuntimeEngine engine;
    if (engine.initialize() != VGREResult::SUCCESS) { printf("  FAIL  init\n"); return 1; }

    auto &mm = engine.getMemoryManager();
    MemoryHandle devOut = nullptr;
    mm.allocate(64 * sizeof(float), devOut);

    const char *src = R"(
        __global__ void nsight_probe(float* out) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            out[i] = (float)i;
        }
    )";
    void *args[] = {&devOut};
    dim3 grid(2), block(32);
    VGREResult r = engine.launchKernel("nsight_probe", src, grid, block, args, 0);
    check("kernel launch", r == VGREResult::SUCCESS);
    engine.synchronize();

    const char *path = "/tmp/vgre_nsight_test.nsys-rep";
    std::remove(path);
    int rc = vgre_export_nsight_trace(path);
    check("export returns success", rc == VGRE_SUCCESS);

    sqlite3 *db = nullptr;
    bool opened = (sqlite3_open(path, &db) == 0 && db);
    check("exported .nsys-rep opens as SQLite", opened);

    if (opened) {
        // Schema tables exist.
        long long tbls = scalarQuery(db,
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND "
            "name IN ('StringIds','KernelLaunches','MemcpyOps','NvtxEvents');");
        check("all four Nsight tables present", tbls == 4);

        // The kernel launch was recorded.
        long long kcount = scalarQuery(db, "SELECT COUNT(*) FROM KernelLaunches;");
        check("at least one KernelLaunch recorded", kcount >= 1);

        // Grid/block dims and a resolvable name for the launch.
        std::string name;
        long long gx = scalarQuery(db,
            "SELECT k.gridX, s.value FROM KernelLaunches k "
            "JOIN StringIds s ON s.id = k.nameId LIMIT 1;", &name);
        check("kernel gridX recorded (==2)", gx == 2);
        check("kernel name resolved via StringIds",
              name.find("nsight_probe") != std::string::npos);

        // start <= end ordering.
        long long ordered = scalarQuery(db,
            "SELECT COUNT(*) FROM KernelLaunches WHERE end >= start;");
        check("kernel start<=end", ordered == kcount);

        sqlite3_close(db);
    }

    mm.free(devOut);
    engine.shutdown();
    std::remove(path);

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
