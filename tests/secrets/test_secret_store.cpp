// Sealed secret store — create/read, rotation history, access policy,
// pruning, at-rest confidentiality, tamper detection, and persistence.

#include "vgre/secrets/secret_store.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#ifdef _WIN32
#include <process.h>
#define VGRE_GETPID _getpid
#else
#include <unistd.h>
#define VGRE_GETPID getpid
#endif

using namespace vgre::secrets;
using vgre::VGREResult;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static std::string tmpPath(const char* leaf) {
    const char* dir = std::getenv("TMPDIR");
    std::string base = dir && *dir ? dir : "/tmp";
    return base + "/vgre_secrets_" + std::to_string(VGRE_GETPID()) + "_" + leaf;
}
static void removeAll(const std::string& p) {
    std::remove(p.c_str());
    std::remove((p + ".mk").c_str());
}

int main() {
    std::printf("=== Secret store (sealed, versioned, policy-guarded) ===\n");

    std::array<uint8_t, 32> master{};
    for (int i = 0; i < 32; ++i) master[i] = uint8_t(0x40 + i);

    std::string path = tmpPath("store.bin");
    removeAll(path);

    // ── 1. Create + read ─────────────────────────────────────────────────
    {
        std::unique_ptr<SecretStore> st;
        check("open() new store", SecretStore::open(path, master.data(), st) == VGREResult::SUCCESS);
        check("put() creates a secret", st->put("alice", "db/password", "s3cr3t-v1") == VGREResult::SUCCESS);
        check("put() twice rejects", st->put("alice", "db/password", "x") == VGREResult::ERR_ALREADY_EXISTS);

        std::string v;
        check("owner get() returns plaintext", st->get("alice", "db/password", v) == VGREResult::SUCCESS && v == "s3cr3t-v1");

        // ── 2. Access policy ─────────────────────────────────────────────
        check("non-owner get() is denied", st->get("mallory", "db/password", v) == VGREResult::ERR_AUTH_FAILED);
        AccessPolicy pol;
        pol.grant("bob", SecretAction::READ);
        check("owner sets policy", st->setPolicy("alice", "db/password", pol) == VGREResult::SUCCESS);
        check("granted principal can read", st->get("bob", "db/password", v) == VGREResult::SUCCESS && v == "s3cr3t-v1");
        uint32_t dummy = 0;
        check("granted reader cannot rotate", st->rotate("bob", "db/password", "no", dummy) == VGREResult::ERR_AUTH_FAILED);

        // ── 3. Rotation (zero-downtime: old version still readable) ───────
        uint32_t nv = 0;
        check("rotate() bumps version", st->rotate("alice", "db/password", "s3cr3t-v2", nv) == VGREResult::SUCCESS && nv == 2);
        check("current get() returns new value", st->get("alice", "db/password", v) == VGREResult::SUCCESS && v == "s3cr3t-v2");
        check("old version still readable during grace", st->getVersion("alice", "db/password", 1, v) == VGREResult::SUCCESS && v == "s3cr3t-v1");

        st->rotate("alice", "db/password", "s3cr3t-v3", nv);
        // ── 4. Prune old versions ────────────────────────────────────────
        size_t pruned = 0;
        check("prune keepLast=1 drops the two old versions", st->pruneOldVersions("alice", "db/password", 1, pruned) == VGREResult::SUCCESS && pruned == 2);
        check("pruned old version no longer present", st->getVersion("alice", "db/password", 1, v) == VGREResult::ERR_NOT_FOUND);
        check("current version survives prune", st->get("alice", "db/password", v) == VGREResult::SUCCESS && v == "s3cr3t-v3");
    }

    // ── 5. At-rest confidentiality ───────────────────────────────────────
    {
        std::ifstream f(path, std::ios::binary);
        std::string raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        check("secret values never stored in cleartext",
              raw.find("s3cr3t-v3") == std::string::npos && raw.find("s3cr3t-v1") == std::string::npos);
    }

    // ── 6. Persistence across reopen ─────────────────────────────────────
    {
        std::unique_ptr<SecretStore> st;
        check("reopen() loads + verifies", SecretStore::open(path, master.data(), st) == VGREResult::SUCCESS);
        std::string v;
        check("secret survives reopen", st->get("alice", "db/password", v) == VGREResult::SUCCESS && v == "s3cr3t-v3");
        check("count persisted", st->count() == 1);

        // ── 7. delete ────────────────────────────────────────────────────
        check("remove() succeeds", st->remove("alice", "db/password") == VGREResult::SUCCESS);
        check("removed secret is gone", st->get("alice", "db/password", v) == VGREResult::ERR_NOT_FOUND);
    }

    // ── 8. Tamper detection (flip a body byte) + wrong key ───────────────
    {
        removeAll(path);
        { std::unique_ptr<SecretStore> st; SecretStore::open(path, master.data(), st);
          st->put("alice", "k", "value-here"); }
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(0, std::ios::end);
        long sz = static_cast<long>(f.tellg());
        long pos = 8 + (sz - 8 - 32) / 2; // inside the body region
        f.seekg(pos); char c; f.read(&c, 1);
        char x = static_cast<char>(c ^ 0x01);
        f.seekp(pos); f.write(&x, 1); f.flush(); f.close();
        std::unique_ptr<SecretStore> st;
        check("open() detects tampered store (ERR_CRYPTO)", SecretStore::open(path, master.data(), st) == VGREResult::ERR_CRYPTO);

        std::array<uint8_t, 32> wrong = master; wrong[0] ^= 0xFF;
        removeAll(path);
        { std::unique_ptr<SecretStore> st2; SecretStore::open(path, master.data(), st2); st2->put("a","b","c"); }
        std::unique_ptr<SecretStore> st3;
        check("wrong master key rejects store", SecretStore::open(path, wrong.data(), st3) == VGREResult::ERR_CRYPTO);
    }

    removeAll(path);
    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
