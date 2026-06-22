// VGRE Secrets — sealed, versioned secret store (impl).
// See include/vgre/secrets/secret_store.h.

#include "vgre/secrets/secret_store.h"

#include "vgre/advanced/secure_channel.h"        // crypto primitives
#include "vgre/advanced/hardware_token_manager.h" // master-key root (OS/TPM)
#include "vgre/compliance/audit_log.h"            // every access is audited
#include "vgre/common/logger.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>

#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace vgre {
namespace secrets {

namespace crypto = vgre::advanced::crypto;

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
    const char* p; const char* end; bool ok = true;
    explicit Reader(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}
    uint8_t u8() { if (p + 1 > end) { ok = false; return 0; } return static_cast<uint8_t>(*p++); }
    uint32_t u32() {
        if (p + 4 > end) { ok = false; return 0; }
        uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= uint32_t(uint8_t(p[i])) << (8 * i);
        p += 4; return v;
    }
    uint64_t u64() {
        if (p + 8 > end) { ok = false; return 0; }
        uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= uint64_t(uint8_t(p[i])) << (8 * i);
        p += 8; return v;
    }
    std::string str() {
        uint32_t n = u32();
        if (!ok || p + n > end) { ok = false; return {}; }
        std::string s(p, p + n); p += n; return s;
    }
};

constexpr char kMagic[8] = {'V', 'G', 'R', 'E', 'S', 'E', 'C', '1'};

uint64_t nowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}
int hexVal(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}
size_t hexDecode(const char* s, uint8_t* out, size_t n) {
    size_t i = 0;
    for (; i < n && s[2 * i] && s[2 * i + 1]; ++i) {
        int hi = hexVal(s[2 * i]), lo = hexVal(s[2 * i + 1]);
        if (hi < 0 || lo < 0) break;
        out[i] = uint8_t((hi << 4) | lo);
    }
    return i;
}
const char* actionName(SecretAction a) {
    switch (a) {
        case SecretAction::READ:   return "secret.read";
        case SecretAction::WRITE:  return "secret.write";
        case SecretAction::ROTATE: return "secret.rotate";
        case SecretAction::Remove: return "secret.delete";
        case SecretAction::LIST:   return "secret.list";
        case SecretAction::ADMIN:  return "secret.admin";
    }
    return "secret.unknown";
}

struct Version { uint32_t num; uint64_t created_ns; std::string sealed; }; // sealed = iv12+tag16+ct
struct Secret {
    std::string name, owner;
    uint64_t created_ns = 0, rotated_ns = 0;
    uint32_t currentVersion = 0;
    AccessPolicy policy;
    std::vector<Version> versions;
};

} // namespace

struct SecretStore::Impl {
    bool opened = false;
    std::string path;
    std::array<uint8_t, 32> masterKey{};
    std::array<uint8_t, 32> macKey{};
    std::map<std::string, Secret> secrets;
    mutable std::mutex mu;

    void deriveMacKey() {
        const char label[] = "VGRE-SECRETS-FILE-MAC-v1";
        crypto::hmac_sha256(masterKey.data(), masterKey.size(),
                            reinterpret_cast<const uint8_t*>(label), sizeof(label) - 1,
                            macKey.data());
    }

    std::string seal(const std::string& name, uint32_t ver, const std::string& plain) const {
        std::string aad = name + ":" + std::to_string(ver);
        std::string blob;
        blob.resize(12 + 16 + plain.size());
        auto* iv = reinterpret_cast<uint8_t*>(&blob[0]);
        auto* tag = reinterpret_cast<uint8_t*>(&blob[12]);
        auto* ct = reinterpret_cast<uint8_t*>(&blob[28]);
        crypto::random_bytes(iv, 12);
        crypto::aes256_gcm_encrypt(masterKey.data(), iv,
                                   reinterpret_cast<const uint8_t*>(aad.data()), aad.size(),
                                   reinterpret_cast<const uint8_t*>(plain.data()), plain.size(),
                                   ct, tag);
        return blob;
    }
    bool unseal(const std::string& name, uint32_t ver, const std::string& blob,
                std::string& out) const {
        if (blob.size() < 28) return false;
        std::string aad = name + ":" + std::to_string(ver);
        const auto* iv = reinterpret_cast<const uint8_t*>(&blob[0]);
        const auto* tag = reinterpret_cast<const uint8_t*>(&blob[12]);
        const auto* ct = reinterpret_cast<const uint8_t*>(&blob[28]);
        size_t ctLen = blob.size() - 28;
        out.assign(ctLen, '\0');
        return crypto::aes256_gcm_decrypt(masterKey.data(), iv,
                                          reinterpret_cast<const uint8_t*>(aad.data()), aad.size(),
                                          ct, ctLen, reinterpret_cast<uint8_t*>(&out[0]), tag);
    }

    std::string serializeBody() const {
        std::string b;
        putU32(b, static_cast<uint32_t>(secrets.size()));
        for (const auto& kv : secrets) {
            const Secret& s = kv.second;
            putStr(b, s.name);
            putStr(b, s.owner);
            putU64(b, s.created_ns);
            putU64(b, s.rotated_ns);
            putU32(b, s.currentVersion);
            putU32(b, static_cast<uint32_t>(s.policy.grants.size()));
            for (const auto& g : s.policy.grants) { putStr(b, g.first); putU32(b, g.second); }
            putU32(b, static_cast<uint32_t>(s.versions.size()));
            for (const auto& v : s.versions) {
                putU32(b, v.num); putU64(b, v.created_ns); putStr(b, v.sealed);
            }
        }
        return b;
    }

    void persist() const {
        std::string body = serializeBody();
        std::string mac;
        std::array<uint8_t, 32> macv{};
        std::string macInput(kMagic, sizeof(kMagic));
        macInput += body;
        crypto::hmac_sha256(macKey.data(), macKey.size(),
                            reinterpret_cast<const uint8_t*>(macInput.data()), macInput.size(),
                            macv.data());
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) return;
        f.write(kMagic, sizeof(kMagic));
        f.write(body.data(), static_cast<std::streamsize>(body.size()));
        f.write(reinterpret_cast<const char*>(macv.data()), macv.size());
        f.flush();
#ifndef _WIN32
        ::chmod(path.c_str(), 0600);
#endif
    }

    vgre::VGREResult load() {
        std::ifstream f(path, std::ios::binary);
        if (!f) return vgre::VGREResult::SUCCESS; // new store
        std::string all((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (all.empty()) return vgre::VGREResult::SUCCESS;
        if (all.size() < sizeof(kMagic) + 32 ||
            std::memcmp(all.data(), kMagic, sizeof(kMagic)) != 0)
            return vgre::VGREResult::ERR_IO;
        // Split body / footer MAC and verify the whole-file integrity.
        std::string body(all.begin() + sizeof(kMagic), all.end() - 32);
        std::array<uint8_t, 32> stored{};
        std::memcpy(stored.data(), all.data() + all.size() - 32, 32);
        std::array<uint8_t, 32> calc{};
        std::string macInput(kMagic, sizeof(kMagic));
        macInput += body;
        crypto::hmac_sha256(macKey.data(), macKey.size(),
                            reinterpret_cast<const uint8_t*>(macInput.data()), macInput.size(),
                            calc.data());
        if (!crypto::secure_compare(calc.data(), stored.data(), 32))
            return vgre::VGREResult::ERR_CRYPTO; // tampered or wrong master key

        Reader rd(body);
        uint32_t n = rd.u32();
        for (uint32_t i = 0; i < n && rd.ok; ++i) {
            Secret s;
            s.name = rd.str();
            s.owner = rd.str();
            s.created_ns = rd.u64();
            s.rotated_ns = rd.u64();
            s.currentVersion = rd.u32();
            uint32_t gc = rd.u32();
            for (uint32_t g = 0; g < gc && rd.ok; ++g) {
                std::string pr = rd.str();
                uint32_t mask = rd.u32();
                s.policy.grants[pr] = mask;
            }
            uint32_t vc = rd.u32();
            for (uint32_t v = 0; v < vc && rd.ok; ++v) {
                Version ver;
                ver.num = rd.u32();
                ver.created_ns = rd.u64();
                ver.sealed = rd.str();
                s.versions.push_back(std::move(ver));
            }
            if (!rd.ok) return vgre::VGREResult::ERR_IO;
            secrets[s.name] = std::move(s);
        }
        return rd.ok ? vgre::VGREResult::SUCCESS : vgre::VGREResult::ERR_IO;
    }

    bool authorize(const Secret& s, const std::string& principal, SecretAction a) const {
        return principal == s.owner || s.policy.allows(principal, a);
    }
    void audit(const std::string& principal, SecretAction a, const std::string& name,
               bool allowed) const {
        vgre::compliance::AuditLog::global().emit(
            principal, actionName(a), "secret:" + name,
            allowed ? vgre::compliance::AuditOutcome::Ok
                    : vgre::compliance::AuditOutcome::Denied,
            allowed ? vgre::compliance::AuditSeverity::Notice
                    : vgre::compliance::AuditSeverity::Warn);
    }
};

SecretStore::SecretStore() : impl_(new Impl()) {}
SecretStore::~SecretStore() = default;

vgre::VGREResult SecretStore::open(const std::string& path, const uint8_t masterKey[32],
                                   std::unique_ptr<SecretStore>& out) {
    std::unique_ptr<SecretStore> st(new SecretStore());
    Impl& im = *st->impl_;
    im.path = path;
    std::memcpy(im.masterKey.data(), masterKey, 32);
    im.deriveMacKey();
    vgre::VGREResult r = im.load();
    if (r != vgre::VGREResult::SUCCESS) return r;
    im.opened = true;
    out = std::move(st);
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult SecretStore::openDefault(const std::string& path,
                                          std::unique_ptr<SecretStore>& out) {
    uint8_t master[32];
    bool haveMaster = false;

    auto& tm = vgre::advanced::HardwareTokenManager::instance();
    if (tm.initialize() == vgre::VGREResult::SUCCESS) {
        std::string token;
        if (tm.getToken("vgre-secrets-master", token) != vgre::VGREResult::SUCCESS ||
            token.empty()) {
            token = vgre::advanced::HardwareTokenManager::generateToken(48);
            tm.storeToken("vgre-secrets-master", token); // best effort (OS/TPM-backed)
        }
        if (!token.empty()) {
            const char salt[] = "VGRE-SECRETS-MK-v1";
            crypto::pbkdf2_sha256(reinterpret_cast<const uint8_t*>(token.data()), token.size(),
                                  reinterpret_cast<const uint8_t*>(salt), sizeof(salt) - 1,
                                  200000, master, 32);
            haveMaster = true;
        }
    }
    if (!haveMaster) {
        if (const char* hk = std::getenv("VGRE_SECRETS_KEY")) {
            if (hexDecode(hk, master, 32) == 32) haveMaster = true;
        }
    }
    if (!haveMaster) {
        std::string mk = path + ".mk";
        std::ifstream f(mk, std::ios::binary);
        if (f) {
            std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            if (b.size() >= 32) { std::memcpy(master, b.data(), 32); haveMaster = true; }
        }
        if (!haveMaster) {
            crypto::random_bytes(master, 32);
            std::ofstream wf(mk, std::ios::binary | std::ios::trunc);
            if (wf) { wf.write(reinterpret_cast<char*>(master), 32); wf.flush(); }
#ifndef _WIN32
            ::chmod(mk.c_str(), 0600);
#endif
            haveMaster = true;
        }
    }
    vgre::VGREResult r = open(path, master, out);
    std::memset(master, 0, sizeof(master));
    return r;
}

vgre::VGREResult SecretStore::put(const std::string& principal, const std::string& name,
                                  const std::string& value) {
    if (!impl_ || !impl_->opened) return vgre::VGREResult::ERR_NOT_INITIALIZED;
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (impl_->secrets.count(name)) {
        impl_->audit(principal, SecretAction::WRITE, name, false);
        return vgre::VGREResult::ERR_ALREADY_EXISTS;
    }
    Secret s;
    s.name = name;
    s.owner = principal;
    s.created_ns = s.rotated_ns = nowNs();
    s.currentVersion = 1;
    Version v;
    v.num = 1;
    v.created_ns = s.created_ns;
    v.sealed = impl_->seal(name, 1, value);
    s.versions.push_back(std::move(v));
    impl_->secrets[name] = std::move(s);
    impl_->persist();
    impl_->audit(principal, SecretAction::WRITE, name, true);
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult SecretStore::getVersion(const std::string& principal, const std::string& name,
                                         uint32_t version, std::string& outValue) const {
    if (!impl_ || !impl_->opened) return vgre::VGREResult::ERR_NOT_INITIALIZED;
    std::lock_guard<std::mutex> lk(impl_->mu);
    auto it = impl_->secrets.find(name);
    if (it == impl_->secrets.end()) { impl_->audit(principal, SecretAction::READ, name, false); return vgre::VGREResult::ERR_NOT_FOUND; }
    if (!impl_->authorize(it->second, principal, SecretAction::READ)) {
        impl_->audit(principal, SecretAction::READ, name, false);
        return vgre::VGREResult::ERR_AUTH_FAILED;
    }
    for (const auto& v : it->second.versions) {
        if (v.num == version) {
            if (!impl_->unseal(name, version, v.sealed, outValue)) return vgre::VGREResult::ERR_CRYPTO;
            impl_->audit(principal, SecretAction::READ, name, true);
            return vgre::VGREResult::SUCCESS;
        }
    }
    return vgre::VGREResult::ERR_NOT_FOUND;
}

vgre::VGREResult SecretStore::get(const std::string& principal, const std::string& name,
                                  std::string& outValue) const {
    if (!impl_ || !impl_->opened) return vgre::VGREResult::ERR_NOT_INITIALIZED;
    uint32_t cur;
    { std::lock_guard<std::mutex> lk(impl_->mu);
      auto it = impl_->secrets.find(name);
      if (it == impl_->secrets.end()) { impl_->audit(principal, SecretAction::READ, name, false); return vgre::VGREResult::ERR_NOT_FOUND; }
      cur = it->second.currentVersion; }
    return getVersion(principal, name, cur, outValue);
}

vgre::VGREResult SecretStore::rotate(const std::string& principal, const std::string& name,
                                     const std::string& newValue, uint32_t& newVersion) {
    if (!impl_ || !impl_->opened) return vgre::VGREResult::ERR_NOT_INITIALIZED;
    std::lock_guard<std::mutex> lk(impl_->mu);
    auto it = impl_->secrets.find(name);
    if (it == impl_->secrets.end()) { impl_->audit(principal, SecretAction::ROTATE, name, false); return vgre::VGREResult::ERR_NOT_FOUND; }
    Secret& s = it->second;
    if (!impl_->authorize(s, principal, SecretAction::ROTATE)) {
        impl_->audit(principal, SecretAction::ROTATE, name, false);
        return vgre::VGREResult::ERR_AUTH_FAILED;
    }
    newVersion = s.currentVersion + 1;
    Version v;
    v.num = newVersion;
    v.created_ns = nowNs();
    v.sealed = impl_->seal(name, newVersion, newValue);
    s.versions.push_back(std::move(v));
    s.currentVersion = newVersion;
    s.rotated_ns = v.created_ns;
    impl_->persist();
    impl_->audit(principal, SecretAction::ROTATE, name, true);
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult SecretStore::remove(const std::string& principal, const std::string& name) {
    if (!impl_ || !impl_->opened) return vgre::VGREResult::ERR_NOT_INITIALIZED;
    std::lock_guard<std::mutex> lk(impl_->mu);
    auto it = impl_->secrets.find(name);
    if (it == impl_->secrets.end()) { impl_->audit(principal, SecretAction::Remove, name, false); return vgre::VGREResult::ERR_NOT_FOUND; }
    if (!impl_->authorize(it->second, principal, SecretAction::Remove)) {
        impl_->audit(principal, SecretAction::Remove, name, false);
        return vgre::VGREResult::ERR_AUTH_FAILED;
    }
    for (auto& v : it->second.versions)
        std::fill(v.sealed.begin(), v.sealed.end(), '\0'); // scrub in memory
    impl_->secrets.erase(it);
    impl_->persist();
    impl_->audit(principal, SecretAction::Remove, name, true);
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult SecretStore::pruneOldVersions(const std::string& principal,
                                               const std::string& name, uint32_t keepLast,
                                               size_t& outPruned) {
    outPruned = 0;
    if (!impl_ || !impl_->opened) return vgre::VGREResult::ERR_NOT_INITIALIZED;
    std::lock_guard<std::mutex> lk(impl_->mu);
    auto it = impl_->secrets.find(name);
    if (it == impl_->secrets.end()) return vgre::VGREResult::ERR_NOT_FOUND;
    if (!impl_->authorize(it->second, principal, SecretAction::ADMIN)) {
        impl_->audit(principal, SecretAction::ADMIN, name, false);
        return vgre::VGREResult::ERR_AUTH_FAILED;
    }
    Secret& s = it->second;
    if (keepLast == 0) keepLast = 1; // never drop the current version
    std::sort(s.versions.begin(), s.versions.end(),
              [](const Version& a, const Version& b) { return a.num < b.num; });
    if (s.versions.size() > keepLast) {
        size_t drop = s.versions.size() - keepLast;
        for (size_t i = 0; i < drop; ++i)
            std::fill(s.versions[i].sealed.begin(), s.versions[i].sealed.end(), '\0');
        s.versions.erase(s.versions.begin(), s.versions.begin() + drop);
        outPruned = drop;
    }
    impl_->persist();
    impl_->audit(principal, SecretAction::ADMIN, name, true);
    return vgre::VGREResult::SUCCESS;
}

std::vector<SecretMetadata> SecretStore::list(const std::string& principal) const {
    std::vector<SecretMetadata> out;
    if (!impl_ || !impl_->opened) return out;
    std::lock_guard<std::mutex> lk(impl_->mu);
    for (const auto& kv : impl_->secrets) {
        const Secret& s = kv.second;
        if (!impl_->authorize(s, principal, SecretAction::LIST)) continue;
        SecretMetadata m;
        m.name = s.name;
        m.owner = s.owner;
        m.currentVersion = s.currentVersion;
        m.versionCount = static_cast<uint32_t>(s.versions.size());
        m.created_ns = s.created_ns;
        m.rotated_ns = s.rotated_ns;
        out.push_back(std::move(m));
    }
    impl_->audit(principal, SecretAction::LIST, "*", true);
    return out;
}

vgre::VGREResult SecretStore::setPolicy(const std::string& principal, const std::string& name,
                                        const AccessPolicy& policy) {
    if (!impl_ || !impl_->opened) return vgre::VGREResult::ERR_NOT_INITIALIZED;
    std::lock_guard<std::mutex> lk(impl_->mu);
    auto it = impl_->secrets.find(name);
    if (it == impl_->secrets.end()) return vgre::VGREResult::ERR_NOT_FOUND;
    if (!impl_->authorize(it->second, principal, SecretAction::ADMIN)) {
        impl_->audit(principal, SecretAction::ADMIN, name, false);
        return vgre::VGREResult::ERR_AUTH_FAILED;
    }
    it->second.policy = policy;
    impl_->persist();
    impl_->audit(principal, SecretAction::ADMIN, name, true);
    return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult SecretStore::getPolicy(const std::string& principal, const std::string& name,
                                        AccessPolicy& out) const {
    if (!impl_ || !impl_->opened) return vgre::VGREResult::ERR_NOT_INITIALIZED;
    std::lock_guard<std::mutex> lk(impl_->mu);
    auto it = impl_->secrets.find(name);
    if (it == impl_->secrets.end()) return vgre::VGREResult::ERR_NOT_FOUND;
    if (!impl_->authorize(it->second, principal, SecretAction::ADMIN))
        return vgre::VGREResult::ERR_AUTH_FAILED;
    out = it->second.policy;
    return vgre::VGREResult::SUCCESS;
}

size_t SecretStore::count() const {
    if (!impl_ || !impl_->opened) return 0;
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->secrets.size();
}

bool SecretStore::enabled() const { return impl_ && impl_->opened; }

SecretStore& SecretStore::global() {
    static std::unique_ptr<SecretStore> g;
    static std::once_flag once;
    std::call_once(once, [] {
        const char* path = std::getenv("VGRE_SECRETS_STORE");
        if (!path || !*path) { g.reset(new SecretStore()); return; }
        std::unique_ptr<SecretStore> opened;
        if (SecretStore::openDefault(path, opened) == vgre::VGREResult::SUCCESS) {
            g = std::move(opened);
            VGRE_LOG_INFO("Secrets", std::string("Secret store active: ") + path);
        } else {
            VGRE_LOG_ERROR("Secrets", std::string("Failed to open secret store: ") + path);
            g.reset(new SecretStore());
        }
    });
    return *g;
}

} // namespace secrets
} // namespace vgre
