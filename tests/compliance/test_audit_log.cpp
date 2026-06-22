// Tamper-evident audit log — integrity chain, on-disk tamper detection,
// GDPR crypto-erasure + deletion certificate, and cross-restart persistence.

#include "vgre/compliance/audit_log.h"

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

using namespace vgre::compliance;
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
    return base + "/vgre_audit_" + std::to_string(VGRE_GETPID()) + "_" + leaf;
}

static void removeAll(const std::string& p) {
    std::remove(p.c_str());
    std::remove((p + ".keys").c_str());
    std::remove((p + ".mk").c_str());
}

int main() {
    std::printf("=== Compliance audit log (crypto-integrity + GDPR erasure) ===\n");

    std::array<uint8_t, 32> chainKey{}, piiKey{};
    for (int i = 0; i < 32; ++i) { chainKey[i] = uint8_t(0xA0 + i); piiKey[i] = uint8_t(0x10 + i); }

    std::string path = tmpPath("main.log");
    removeAll(path);

    // ── 1. Append + chain integrity ──────────────────────────────────────
    std::array<uint8_t, 32> headAfter3{};
    {
        std::unique_ptr<AuditLog> log;
        check("open() creates a new log", AuditLog::open(path, chainKey.data(), piiKey.data(), log) == VGREResult::SUCCESS);

        uint64_t s0 = 999, s1 = 999, s2 = 999;
        log->emit("alice", "auth.login", "cluster:node-1", AuditOutcome::Ok, AuditSeverity::Info);
        log->append([]{ AuditRecord r; r.actor="bob"; r.action="kernel.load"; r.resource="device:0"; return r; }(), s1);
        // a PII-bearing record (crypto-shreddable)
        AuditRecord pr; pr.actor="svc"; pr.action="user.export"; pr.resource="db";
        pr.subjectId="subj-42"; pr.piiPlaintext="email=carol@example.com;ssn=123-45-6789";
        log->append(std::move(pr), s2);
        (void)s0;

        check("three records appended", log->count() == 3);
        check("seqs are monotonic from 0", s1 == 1 && s2 == 2);

        uint64_t bad = 0;
        check("verify() reports an intact chain", log->verify(bad) == VGREResult::SUCCESS && bad == UINT64_MAX);

        // PII is decryptable while the subject key exists.
        AuditQuery q; q.actionEquals = "user.export";
        auto rows = log->query(q);
        check("PII decrypts to plaintext while key present",
              rows.size() == 1 && !rows[0].redacted &&
              rows[0].piiPlaintext.find("carol@example.com") != std::string::npos);

        headAfter3 = log->head();
    }

    // ── 2. PII never stored in cleartext on disk ─────────────────────────
    {
        std::ifstream f(path, std::ios::binary);
        std::string raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        check("raw file does NOT contain PII plaintext",
              raw.find("carol@example.com") == std::string::npos &&
              raw.find("123-45-6789") == std::string::npos);
    }

    // ── 3. Reopen verifies persisted chain; head matches ─────────────────
    {
        std::unique_ptr<AuditLog> log;
        check("reopen() loads + verifies existing chain",
              AuditLog::open(path, chainKey.data(), piiKey.data(), log) == VGREResult::SUCCESS);
        check("count persisted across reopen", log->count() == 3);
        check("head MAC stable across reopen", log->head() == headAfter3);

        // ── 4. GDPR crypto-erasure ───────────────────────────────────────
        DeletionCertificate cert;
        check("forget(subject) succeeds", log->forget("subj-42", cert) == VGREResult::SUCCESS);
        check("certificate covers the one PII record", cert.affectedSeqs.size() == 1 && cert.affectedSeqs[0] == 2);
        check("deletion certificate verifies", log->verifyDeletionCertificate(cert));
        check("tombstone appended (now 4 records)", log->count() == 4);

        AuditQuery q; q.actionEquals = "user.export";
        auto rows = log->query(q);
        check("after erasure PII is redacted + unrecoverable",
              rows.size() == 1 && rows[0].redacted && rows[0].piiPlaintext.empty());

        uint64_t bad = 0;
        check("chain still verifies after erasure", log->verify(bad) == VGREResult::SUCCESS);

        // Forged certificate must fail.
        DeletionCertificate forged = cert;
        forged.proof[0] ^= 0xFF;
        check("tampered certificate is rejected", !log->verifyDeletionCertificate(forged));
    }

    // ── 5. On-disk tamper detection ──────────────────────────────────────
    {
        // Flip one byte in the body (past the 8-byte magic) and confirm verify() trips.
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(0, std::ios::end);
        long sz = static_cast<long>(f.tellg());
        long pos = 8 + (sz - 8) / 2; // somewhere in the record region
        f.seekg(pos);
        char c; f.read(&c, 1);
        char flipped = static_cast<char>(c ^ 0x01);
        f.seekp(pos); f.write(&flipped, 1); f.flush(); f.close();

        std::unique_ptr<AuditLog> log;
        VGREResult r = AuditLog::open(path, chainKey.data(), piiKey.data(), log);
        check("open() detects on-disk tampering (ERR_CRYPTO)", r == VGREResult::ERR_CRYPTO);
    }

    // ── 6. Wrong key cannot validate the chain ───────────────────────────
    {
        std::array<uint8_t, 32> wrong = chainKey; wrong[0] ^= 0xFF;
        // rebuild a clean log first
        removeAll(path);
        { std::unique_ptr<AuditLog> log; AuditLog::open(path, chainKey.data(), piiKey.data(), log);
          log->emit("a","x","y",AuditOutcome::Ok,AuditSeverity::Info); }
        std::unique_ptr<AuditLog> log;
        VGREResult r = AuditLog::open(path, wrong.data(), piiKey.data(), log);
        check("wrong chain key rejects the log", r == VGREResult::ERR_CRYPTO);
    }

    removeAll(path);
    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
