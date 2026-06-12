// Backup / DR archive — snapshot/restore, content-addressed dedup, point-in-time
// lineage, restore integrity validation, corruption detection, encryption at
// rest, pruning + GC, and replication export.

#include "vgre/backup/backup_archive.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

#ifdef _WIN32
#include <process.h>
#define VGRE_GETPID _getpid
#else
#include <unistd.h>
#define VGRE_GETPID getpid
#endif

namespace fs = std::filesystem;
using namespace vgre::backup;
using vgre::VGREResult;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static size_t countFiles(const fs::path& dir) {
    size_t n = 0;
    std::error_code ec;
    for (const auto& de : fs::directory_iterator(dir, ec))
        if (de.is_regular_file() && de.path().extension() != ".tmp") ++n;
    return n;
}

int main() {
    std::printf("=== Backup / DR archive ===\n");
    fs::path root = fs::temp_directory_path() / ("vgre_backup_" + std::to_string(VGRE_GETPID()));
    fs::remove_all(root);
    std::string dir = (root / "archive").string();

    std::array<uint8_t, 32> key{};
    for (int i = 0; i < 32; ++i) key[i] = uint8_t(0x70 + i);

    // ── 1. snapshot + restore with integrity ─────────────────────────────
    std::string s1, s2;
    {
        std::unique_ptr<BackupArchive> ar;
        check("open() encrypted archive", BackupArchive::open(dir, key.data(), ar) == VGREResult::SUCCESS);

        std::map<std::string, std::string> blobs = {
            {"weights.bin", std::string(4096, 'W')},
            {"config.json", "{\"lr\":0.001}"},
        };
        check("createSnapshot v1", ar->createSnapshot("epoch-1", blobs, s1) == VGREResult::SUCCESS);

        std::map<std::string, std::string> out;
        check("restoreSnapshot v1", ar->restoreSnapshot(s1, out) == VGREResult::SUCCESS);
        check("restored weights match", out["weights.bin"] == std::string(4096, 'W'));
        check("restored config matches", out["config.json"] == "{\"lr\":0.001}");

        // ── 2. content-addressed dedup: v2 shares weights, changes config ─
        blobs["config.json"] = "{\"lr\":0.0005}";
        check("createSnapshot v2", ar->createSnapshot("epoch-2", blobs, s2) == VGREResult::SUCCESS);
        check("two snapshots", ar->snapshotCount() == 2);
        // weights.bin identical across v1,v2 → only 3 unique objects total
        // (weights, config-v1, config-v2).
        check("identical blobs deduplicated", countFiles(fs::path(dir) / "objects") == 3);

        // ── 3. point-in-time lineage ─────────────────────────────────────
        SnapshotInfo i1, i2;
        ar->getSnapshot(s1, i1);
        ar->getSnapshot(s2, i2);
        std::array<uint8_t, 32> zero{};
        check("v1 has no predecessor", i1.prevHash == zero);
        check("v2 chains to v1", i2.prevHash == i1.manifestHash);
    }

    // ── 4. PII/data sealed at rest (encrypted) ───────────────────────────
    {
        bool clear = false;
        for (const auto& de : fs::directory_iterator(fs::path(dir) / "objects")) {
            std::ifstream f(de.path(), std::ios::binary);
            std::string raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            if (raw.find(std::string(64, 'W')) != std::string::npos) clear = true;
        }
        check("object bytes are encrypted at rest", !clear);
    }

    // ── 5. reopen persists; verify + restore validation ──────────────────
    {
        std::unique_ptr<BackupArchive> ar;
        check("reopen archive", BackupArchive::open(dir, key.data(), ar) == VGREResult::SUCCESS);
        check("snapshots persisted", ar->snapshotCount() == 2);
        check("verifySnapshot intact", ar->verifySnapshot(s2) == VGREResult::SUCCESS);

        // ── 6. wrong key cannot restore ──────────────────────────────────
        std::array<uint8_t, 32> wrong = key; wrong[0] ^= 0xFF;
        std::unique_ptr<BackupArchive> bad;
        BackupArchive::open(dir, wrong.data(), bad);
        std::map<std::string, std::string> out;
        check("wrong key fails restore (auth tag)", bad->restoreSnapshot(s2, out) == VGREResult::ERR_CRYPTO);
    }

    // ── 7. corruption detection ──────────────────────────────────────────
    {
        // Corrupt one object file.
        fs::path victim;
        for (const auto& de : fs::directory_iterator(fs::path(dir) / "objects")) { victim = de.path(); break; }
        { std::fstream f(victim, std::ios::in | std::ios::out | std::ios::binary);
          f.seekp(30); char c = 'X'; f.write(&c, 1); }
        std::unique_ptr<BackupArchive> ar;
        BackupArchive::open(dir, key.data(), ar);
        bool anyFail = ar->verifySnapshot(s1) != VGREResult::SUCCESS ||
                       ar->verifySnapshot(s2) != VGREResult::SUCCESS;
        check("corruption detected by verify", anyFail);
    }

    // ── 8. replication export → independent archive restores identically ─
    {
        fs::remove_all(root);
        std::string dir2 = (root / "a2").string();
        std::string dirRep = (root / "replica").string();
        std::string sid;
        std::map<std::string, std::string> blobs = {{"k", "v-payload-1234"}};
        { std::unique_ptr<BackupArchive> ar; BackupArchive::open(dir2, key.data(), ar);
          ar->createSnapshot("snap", blobs, sid);
          check("exportSnapshot to replica", ar->exportSnapshot(sid, dirRep) == VGREResult::SUCCESS); }
        std::unique_ptr<BackupArchive> rep;
        check("open replica archive", BackupArchive::open(dirRep, key.data(), rep) == VGREResult::SUCCESS);
        std::map<std::string, std::string> out;
        check("replica restores identical data",
              rep->restoreSnapshot(sid, out) == VGREResult::SUCCESS && out["k"] == "v-payload-1234");
    }

    // ── 9. prune + GC ────────────────────────────────────────────────────
    {
        fs::remove_all(root);
        std::string d = (root / "pr").string();
        std::unique_ptr<BackupArchive> ar;
        BackupArchive::open(d, nullptr, ar); // plaintext archive this time
        std::string a, b, c;
        ar->createSnapshot("a", {{"f", "AAAA"}}, a);
        ar->createSnapshot("b", {{"f", "BBBB"}}, b);
        ar->createSnapshot("c", {{"f", "CCCC"}}, c);
        size_t removed = 0;
        check("prune keepLast=1", ar->pruneSnapshots(1, removed) == VGREResult::SUCCESS && removed == 2);
        check("only newest snapshot remains", ar->snapshotCount() == 1);
        check("GC reclaimed old objects", countFiles(fs::path(d) / "objects") == 1);
        std::map<std::string, std::string> out;
        check("surviving snapshot still restores", ar->restoreSnapshot(c, out) == VGREResult::SUCCESS && out["f"] == "CCCC");
    }

    fs::remove_all(root);
    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
