// VGRE Compliance — Cryptographically-Integrity-Protected Audit Log (impl).
// See include/vgre/compliance/audit_log.h for the integrity / crypto-erasure model.

#include "vgre/compliance/audit_log.h"

#include "vgre/advanced/secure_channel.h" // vgre::advanced::crypto primitives
#include "vgre/common/logger.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>

#ifndef _WIN32
#include <sys/stat.h> // chmod for the 0600 sidecar key file
#endif

namespace vgre {
namespace compliance {

namespace crypto = vgre::advanced::crypto;

// ── canonical little-endian byte (de)serialization ────────────────────────
namespace {

void putU8(std::string& b, uint8_t v) { b.push_back(static_cast<char>(v)); }
void putU32(std::string& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}
void putU64(std::string& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}
void putStr(std::string& b, const std::string& s) {
    putU32(b, static_cast<uint32_t>(s.size()));
    b.append(s);
}

struct Reader {
    const char* p;
    const char* end;
    bool ok = true;
    explicit Reader(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}
    uint8_t u8() {
        if (p + 1 > end) { ok = false; return 0; }
        return static_cast<uint8_t>(*p++);
    }
    uint32_t u32() {
        if (p + 4 > end) { ok = false; return 0; }
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(static_cast<uint8_t>(p[i])) << (8 * i);
        p += 4;
        return v;
    }
    uint64_t u64() {
        if (p + 8 > end) { ok = false; return 0; }
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(static_cast<uint8_t>(p[i])) << (8 * i);
        p += 8;
        return v;
    }
    std::string str() {
        uint32_t n = u32();
        if (!ok || p + n > end) { ok = false; return {}; }
        std::string s(p, p + n);
        p += n;
        return s;
    }
};

constexpr char kLogMagic[8]  = {'V', 'G', 'R', 'E', 'A', 'U', 'D', '1'};
constexpr char kKeysMagic[8] = {'V', 'G', 'R', 'E', 'A', 'K', 'T', '1'};
constexpr char kGenesisLabel[] = "VGRE-AUDIT-GENESIS-v1";

uint64_t nowNs() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

int hexVal(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
// Parse up to `n` bytes of hex into out; returns count decoded.
size_t hexDecode(const char* s, uint8_t* out, size_t n) {
    size_t i = 0;
    for (; i < n && s[2 * i] && s[2 * i + 1]; ++i) {
        int hi = hexVal(s[2 * i]), lo = hexVal(s[2 * i + 1]);
        if (hi < 0 || lo < 0) break;
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return i;
}

} // namespace

// ── Impl ──────────────────────────────────────────────────────────────────
struct AuditLog::Impl {
    bool opened = false;
    std::string path;
    std::array<uint8_t, 32> chainKey{};
    std::array<uint8_t, 32> piiMasterKey{};

    struct Stored {
        AuditRecord rec;            // public fields (piiPlaintext stays empty on disk)
        std::string piiBlob;        // iv(12)+tag(16)+ciphertext, or empty
        std::array<uint8_t, 32> mac{};
    };
    std::vector<Stored> records;
    std::map<std::string, std::array<uint8_t, 32>> subjectKeys; // subjectId -> sealing key
    std::array<uint8_t, 32> genesisMac{};
    std::array<uint8_t, 32> headMac{};
    uint64_t nextSeq = 0;
    std::ofstream out;
    mutable std::mutex mu;

    // canonical(record) — the exact bytes that are MAC-chained and persisted.
    std::string canonical(const AuditRecord& r, const std::string& piiBlob) const {
        std::string b;
        putU64(b, r.seq);
        putU64(b, r.timestamp_ns);
        putU8(b, static_cast<uint8_t>(r.outcome));
        putU8(b, static_cast<uint8_t>(r.severity));
        putU8(b, r.redacted ? 1 : 0);
        putStr(b, r.actor);
        putStr(b, r.action);
        putStr(b, r.resource);
        putStr(b, r.subjectId);
        putU32(b, static_cast<uint32_t>(r.attributes.size())); // std::map => sorted by key
        for (const auto& kv : r.attributes) {
            putStr(b, kv.first);
            putStr(b, kv.second);
        }
        putStr(b, piiBlob);
        return b;
    }

    static bool parseCanonical(const std::string& c, AuditRecord& r, std::string& piiBlob) {
        Reader rd(c);
        r.seq = rd.u64();
        r.timestamp_ns = rd.u64();
        r.outcome = static_cast<AuditOutcome>(rd.u8());
        r.severity = static_cast<AuditSeverity>(rd.u8());
        r.redacted = rd.u8() != 0;
        r.actor = rd.str();
        r.action = rd.str();
        r.resource = rd.str();
        r.subjectId = rd.str();
        uint32_t n = rd.u32();
        for (uint32_t i = 0; i < n && rd.ok; ++i) {
            std::string k = rd.str();
            std::string v = rd.str();
            r.attributes[k] = v;
        }
        piiBlob = rd.str();
        return rd.ok && rd.p == rd.end;
    }

    std::array<uint8_t, 32> chainStep(const std::array<uint8_t, 32>& prev,
                                      const std::string& canon) const {
        std::string msg;
        msg.append(reinterpret_cast<const char*>(prev.data()), prev.size());
        msg.append(canon);
        std::array<uint8_t, 32> mac{};
        crypto::hmac_sha256(chainKey.data(), chainKey.size(),
                            reinterpret_cast<const uint8_t*>(msg.data()), msg.size(),
                            mac.data());
        return mac;
    }

    // Wrap/unwrap a subject key under the PII master key (AES-256-GCM,
    // aad = subjectId).  Wrapped blob = iv(12) || tag(16) || ct(32).
    std::string wrapKey(const std::string& subjectId, const std::array<uint8_t, 32>& k) const {
        std::string blob;
        blob.resize(12 + 16 + 32);
        auto* iv = reinterpret_cast<uint8_t*>(&blob[0]);
        auto* tag = reinterpret_cast<uint8_t*>(&blob[12]);
        auto* ct = reinterpret_cast<uint8_t*>(&blob[28]);
        crypto::random_bytes(iv, 12);
        crypto::aes256_gcm_encrypt(piiMasterKey.data(), iv,
                                   reinterpret_cast<const uint8_t*>(subjectId.data()),
                                   subjectId.size(), k.data(), 32, ct, tag);
        return blob;
    }
    bool unwrapKey(const std::string& subjectId, const std::string& blob,
                   std::array<uint8_t, 32>& out) const {
        if (blob.size() != 12 + 16 + 32) return false;
        const auto* iv = reinterpret_cast<const uint8_t*>(&blob[0]);
        const auto* tag = reinterpret_cast<const uint8_t*>(&blob[12]);
        const auto* ct = reinterpret_cast<const uint8_t*>(&blob[28]);
        return crypto::aes256_gcm_decrypt(piiMasterKey.data(), iv,
                                          reinterpret_cast<const uint8_t*>(subjectId.data()),
                                          subjectId.size(), ct, 32, out.data(), tag);
    }

    void persistKeys() const {
        std::ofstream f(path + ".keys", std::ios::binary | std::ios::trunc);
        if (!f) return;
        f.write(kKeysMagic, sizeof(kKeysMagic));
        std::string body;
        putU32(body, static_cast<uint32_t>(subjectKeys.size()));
        for (const auto& kv : subjectKeys) {
            putStr(body, kv.first);
            putStr(body, wrapKey(kv.first, kv.second));
        }
        f.write(body.data(), static_cast<std::streamsize>(body.size()));
        f.flush();
#ifndef _WIN32
        ::chmod((path + ".keys").c_str(), 0600);
#endif
    }

    bool loadKeys() {
        std::ifstream f(path + ".keys", std::ios::binary);
        if (!f) return true; // no key table yet — fine
        std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (all.size() < sizeof(kKeysMagic) ||
            std::memcmp(all.data(), kKeysMagic, sizeof(kKeysMagic)) != 0)
            return false;
        std::string body(all.begin() + sizeof(kKeysMagic), all.end());
        Reader rd(body);
        uint32_t n = rd.u32();
        for (uint32_t i = 0; i < n && rd.ok; ++i) {
            std::string sid = rd.str();
            std::string blob = rd.str();
            std::array<uint8_t, 32> k{};
            if (!rd.ok || !unwrapKey(sid, blob, k)) return false; // tampered master/table
            subjectKeys[sid] = k;
        }
        return rd.ok;
    }

    std::array<uint8_t, 32>& getOrCreateSubjectKey(const std::string& sid) {
        auto it = subjectKeys.find(sid);
        if (it != subjectKeys.end()) return it->second;
        std::array<uint8_t, 32> k{};
        crypto::random_bytes(k.data(), k.size());
        auto& slot = (subjectKeys[sid] = k);
        persistKeys();
        return slot;
    }

    void writeRecordToDisk(const std::string& canon, const std::array<uint8_t, 32>& mac) {
        std::string frame;
        putU32(frame, static_cast<uint32_t>(canon.size()));
        frame.append(canon);
        frame.append(reinterpret_cast<const char*>(mac.data()), mac.size());
        out.write(frame.data(), static_cast<std::streamsize>(frame.size()));
        out.flush();
    }
};

AuditLog::AuditLog() : impl_(new Impl()) {}
AuditLog::~AuditLog() = default;

vgre::VGREResult AuditLog::open(const std::string& path, const uint8_t chainKey[32],
                                const uint8_t piiMasterKey[32],
                                std::unique_ptr<AuditLog>& out) {
    std::unique_ptr<AuditLog> log(new AuditLog());
    Impl& im = *log->impl_;
    im.path = path;
    std::memcpy(im.chainKey.data(), chainKey, 32);
    std::memcpy(im.piiMasterKey.data(), piiMasterKey, 32);
    crypto::hmac_sha256(im.chainKey.data(), im.chainKey.size(),
                        reinterpret_cast<const uint8_t*>(kGenesisLabel),
                        sizeof(kGenesisLabel) - 1, im.genesisMac.data());
    im.headMac = im.genesisMac;

    if (!im.loadKeys()) return vgre::VGREResult::ERR_CRYPTO;

    // Load + verify any existing chain.
    std::ifstream in(path, std::ios::binary);
    if (in) {
        std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (!all.empty()) {
            if (all.size() < sizeof(kLogMagic) ||
                std::memcmp(all.data(), kLogMagic, sizeof(kLogMagic)) != 0)
                return vgre::VGREResult::ERR_IO;
            std::string body(all.begin() + sizeof(kLogMagic), all.end());
            Reader rd(body);
            std::array<uint8_t, 32> prev = im.genesisMac;
            while (rd.p < rd.end) {
                uint32_t len = rd.u32();
                if (!rd.ok || rd.p + len + 32 > rd.end) return vgre::VGREResult::ERR_IO;
                std::string canon(rd.p, rd.p + len);
                rd.p += len;
                std::array<uint8_t, 32> storedMac{};
                std::memcpy(storedMac.data(), rd.p, 32);
                rd.p += 32;
                std::array<uint8_t, 32> calc = im.chainStep(prev, canon);
                if (!crypto::secure_compare(calc.data(), storedMac.data(), 32))
                    return vgre::VGREResult::ERR_CRYPTO; // tampered on disk
                Impl::Stored s;
                if (!Impl::parseCanonical(canon, s.rec, s.piiBlob))
                    return vgre::VGREResult::ERR_IO;
                s.mac = storedMac;
                im.nextSeq = s.rec.seq + 1;
                im.records.push_back(std::move(s));
                prev = storedMac;
            }
            im.headMac = prev;
        }
    }

    // Open for append (create with magic if new).
    bool exists = false;
    { std::ifstream probe(path, std::ios::binary); exists = probe.good() && probe.peek() != EOF; }
    im.out.open(path, std::ios::binary | std::ios::app);
    if (!im.out) return vgre::VGREResult::ERR_IO;
    if (!exists) {
        im.out.write(kLogMagic, sizeof(kLogMagic));
        im.out.flush();
    }
    im.opened = true;
    out = std::move(log);
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult AuditLog::append(AuditRecord rec, uint64_t& outSeq) {
    if (!impl_ || !impl_->opened) { outSeq = 0; return vgre::VGREResult::SUCCESS; }
    std::lock_guard<std::mutex> lk(impl_->mu);
    rec.seq = impl_->nextSeq;
    rec.timestamp_ns = nowNs();
    rec.mac = {};

    std::string piiBlob;
    if (!rec.subjectId.empty() && !rec.piiPlaintext.empty()) {
        const auto& k = impl_->getOrCreateSubjectKey(rec.subjectId);
        piiBlob.resize(12 + 16 + rec.piiPlaintext.size());
        auto* iv = reinterpret_cast<uint8_t*>(&piiBlob[0]);
        auto* tag = reinterpret_cast<uint8_t*>(&piiBlob[12]);
        auto* ct = reinterpret_cast<uint8_t*>(&piiBlob[28]);
        crypto::random_bytes(iv, 12);
        crypto::aes256_gcm_encrypt(k.data(), iv,
                                   reinterpret_cast<const uint8_t*>(rec.subjectId.data()),
                                   rec.subjectId.size(),
                                   reinterpret_cast<const uint8_t*>(rec.piiPlaintext.data()),
                                   rec.piiPlaintext.size(), ct, tag);
    }
    rec.piiPlaintext.clear(); // plaintext never persisted

    std::string canon = impl_->canonical(rec, piiBlob);
    std::array<uint8_t, 32> mac = impl_->chainStep(impl_->headMac, canon);
    impl_->writeRecordToDisk(canon, mac);

    rec.mac = mac;
    impl_->headMac = mac;
    impl_->nextSeq++;
    Impl::Stored s;
    s.rec = rec;
    s.piiBlob = std::move(piiBlob);
    s.mac = mac;
    impl_->records.push_back(std::move(s));
    outSeq = rec.seq;
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult AuditLog::emit(const std::string& actor, const std::string& action,
                                const std::string& resource, AuditOutcome outcome,
                                AuditSeverity severity,
                                std::map<std::string, std::string> attributes) {
    AuditRecord r;
    r.actor = actor;
    r.action = action;
    r.resource = resource;
    r.outcome = outcome;
    r.severity = severity;
    r.attributes = std::move(attributes);
    uint64_t seq = 0;
    return append(std::move(r), seq);
}

vgre::VGREResult AuditLog::verify(uint64_t& firstBadSeq) const {
    firstBadSeq = UINT64_MAX;
    if (!impl_ || !impl_->opened) return vgre::VGREResult::SUCCESS;
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::ifstream in(impl_->path, std::ios::binary);
    if (!in) return vgre::VGREResult::ERR_IO;
    std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (all.empty()) return vgre::VGREResult::SUCCESS;
    if (all.size() < sizeof(kLogMagic) ||
        std::memcmp(all.data(), kLogMagic, sizeof(kLogMagic)) != 0)
        return vgre::VGREResult::ERR_IO;
    std::string body(all.begin() + sizeof(kLogMagic), all.end());
    Reader rd(body);
    std::array<uint8_t, 32> prev = impl_->genesisMac;
    while (rd.p < rd.end) {
        uint32_t len = rd.u32();
        if (!rd.ok || rd.p + len + 32 > rd.end) return vgre::VGREResult::ERR_IO;
        std::string canon(rd.p, rd.p + len);
        rd.p += len;
        std::array<uint8_t, 32> storedMac{};
        std::memcpy(storedMac.data(), rd.p, 32);
        rd.p += 32;
        std::array<uint8_t, 32> calc = impl_->chainStep(prev, canon);
        if (!crypto::secure_compare(calc.data(), storedMac.data(), 32)) {
            AuditRecord r;
            std::string blob;
            Impl::parseCanonical(canon, r, blob);
            firstBadSeq = r.seq;
            return vgre::VGREResult::ERR_CRYPTO;
        }
        prev = storedMac;
    }
    return vgre::VGREResult::SUCCESS;
}

std::vector<AuditRecord> AuditLog::query(const AuditQuery& q) const {
    std::vector<AuditRecord> result;
    if (!impl_ || !impl_->opened) return result;
    std::lock_guard<std::mutex> lk(impl_->mu);
    for (const auto& s : impl_->records) {
        const AuditRecord& r = s.rec;
        if (!q.actorEquals.empty() && r.actor != q.actorEquals) continue;
        if (!q.actionEquals.empty() && r.action != q.actionEquals) continue;
        if (!q.resourceEquals.empty() && r.resource != q.resourceEquals) continue;
        if (r.timestamp_ns < q.sinceNs || r.timestamp_ns > q.untilNs) continue;
        if (static_cast<uint8_t>(r.severity) < static_cast<uint8_t>(q.minSeverity)) continue;
        AuditRecord out = r;
        if (!s.piiBlob.empty() && !r.subjectId.empty()) {
            auto it = impl_->subjectKeys.find(r.subjectId);
            bool decrypted = false;
            if (it != impl_->subjectKeys.end() && s.piiBlob.size() > 28) {
                const auto* iv = reinterpret_cast<const uint8_t*>(&s.piiBlob[0]);
                const auto* tag = reinterpret_cast<const uint8_t*>(&s.piiBlob[12]);
                const auto* ct = reinterpret_cast<const uint8_t*>(&s.piiBlob[28]);
                size_t ctLen = s.piiBlob.size() - 28;
                std::string pt(ctLen, '\0');
                if (crypto::aes256_gcm_decrypt(it->second.data(), iv,
                                               reinterpret_cast<const uint8_t*>(r.subjectId.data()),
                                               r.subjectId.size(), ct, ctLen,
                                               reinterpret_cast<uint8_t*>(&pt[0]), tag)) {
                    out.piiPlaintext = std::move(pt);
                    decrypted = true;
                }
            }
            out.redacted = !decrypted; // key destroyed (crypto-erased) or unreadable
        }
        result.push_back(std::move(out));
        if (q.limit && result.size() >= q.limit) break;
    }
    return result;
}

vgre::VGREResult AuditLog::forget(const std::string& subjectId, DeletionCertificate& outCert) {
    if (!impl_ || !impl_->opened) return vgre::VGREResult::ERR_NOT_INITIALIZED;
    std::vector<uint64_t> affected;
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        auto it = impl_->subjectKeys.find(subjectId);
        if (it == impl_->subjectKeys.end()) return vgre::VGREResult::ERR_NOT_FOUND;
        for (const auto& s : impl_->records)
            if (s.rec.subjectId == subjectId && !s.piiBlob.empty())
                affected.push_back(s.rec.seq);
        // Crypto-shred: overwrite then erase the key so the sealed PII is gone.
        crypto::random_bytes(it->second.data(), it->second.size());
        std::memset(it->second.data(), 0, it->second.size());
        impl_->subjectKeys.erase(it);
        impl_->persistKeys();
    }

    std::array<uint8_t, 32> sidHash{};
    crypto::sha256(reinterpret_cast<const uint8_t*>(subjectId.data()), subjectId.size(),
                   sidHash.data());
    std::string sidHex;
    static const char* hx = "0123456789abcdef";
    for (uint8_t b : sidHash) { sidHex.push_back(hx[b >> 4]); sidHex.push_back(hx[b & 0xF]); }

    AuditRecord tomb;
    tomb.actor = "system:compliance";
    tomb.action = "gdpr.erasure";
    tomb.resource = "subject:" + sidHex;
    tomb.outcome = AuditOutcome::SUCCESS;
    tomb.severity = AuditSeverity::NOTICE;
    tomb.attributes["method"] = "crypto-shred-aes256gcm";
    tomb.attributes["affected_records"] = std::to_string(affected.size());
    uint64_t tombSeq = 0;
    vgre::VGREResult r = append(std::move(tomb), tombSeq);
    if (r != vgre::VGREResult::SUCCESS) return r;

    outCert.subjectIdHash = sidHash;
    outCert.affectedSeqs = affected;
    outCert.timestamp_ns = nowNs();
    outCert.tombstoneSeq = tombSeq;

    // proof = HMAC(chainKey, sidHash || seqs(le64...) || ts || tombSeq)
    std::string body;
    body.append(reinterpret_cast<const char*>(sidHash.data()), sidHash.size());
    for (uint64_t s : affected) putU64(body, s);
    putU64(body, outCert.timestamp_ns);
    putU64(body, outCert.tombstoneSeq);
    crypto::hmac_sha256(impl_->chainKey.data(), impl_->chainKey.size(),
                        reinterpret_cast<const uint8_t*>(body.data()), body.size(),
                        outCert.proof.data());
    return vgre::VGREResult::SUCCESS;
}

bool AuditLog::verifyDeletionCertificate(const DeletionCertificate& cert) const {
    if (!impl_ || !impl_->opened) return false;
    std::string body;
    body.append(reinterpret_cast<const char*>(cert.subjectIdHash.data()), cert.subjectIdHash.size());
    for (uint64_t s : cert.affectedSeqs) putU64(body, s);
    putU64(body, cert.timestamp_ns);
    putU64(body, cert.tombstoneSeq);
    std::array<uint8_t, 32> calc{};
    crypto::hmac_sha256(impl_->chainKey.data(), impl_->chainKey.size(),
                        reinterpret_cast<const uint8_t*>(body.data()), body.size(), calc.data());
    return crypto::secure_compare(calc.data(), cert.proof.data(), 32);
}

uint64_t AuditLog::count() const {
    if (!impl_ || !impl_->opened) return 0;
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->records.size();
}

std::array<uint8_t, 32> AuditLog::head() const {
    if (!impl_ || !impl_->opened) return {};
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->headMac;
}

bool AuditLog::enabled() const { return impl_ && impl_->opened; }

// ── process-wide instance ─────────────────────────────────────────────────
AuditLog& AuditLog::global() {
    static std::unique_ptr<AuditLog> g;
    static std::once_flag once;
    std::call_once(once, [] {
        const char* path = std::getenv("VGRE_AUDIT_LOG");
        if (!path || !*path) { g.reset(new AuditLog()); return; } // disabled no-op sink

        uint8_t keys[64];
        bool haveKeys = false;
        if (const char* hk = std::getenv("VGRE_AUDIT_KEY")) {
            if (hexDecode(hk, keys, 64) == 64) haveKeys = true;
        }
        if (!haveKeys) {
            // Persistent 0600 sidecar so the chain remains verifiable across restarts.
            std::string mk = std::string(path) + ".mk";
            std::ifstream f(mk, std::ios::binary);
            if (f) {
                std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                if (b.size() >= 64) { std::memcpy(keys, b.data(), 64); haveKeys = true; }
            }
            if (!haveKeys) {
                crypto::random_bytes(keys, 64);
                std::ofstream wf(mk, std::ios::binary | std::ios::trunc);
                if (wf) { wf.write(reinterpret_cast<char*>(keys), 64); wf.flush(); }
#ifndef _WIN32
                ::chmod(mk.c_str(), 0600);
#endif
                haveKeys = true;
            }
        }
        std::unique_ptr<AuditLog> opened;
        if (AuditLog::open(path, keys, keys + 32, opened) == vgre::VGREResult::SUCCESS) {
            g = std::move(opened);
            VGRE_LOG_INFO("Compliance", std::string("Audit log active: ") + path);
        } else {
            VGRE_LOG_ERROR("Compliance", std::string("Failed to open audit log: ") + path);
            g.reset(new AuditLog());
        }
        std::memset(keys, 0, sizeof(keys));
    });
    return *g;
}

} // namespace compliance
} // namespace vgre
