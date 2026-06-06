#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <vector>
#include <cstring>
#include <algorithm>

#include "vgre/common/os_backend.h"
#if defined(__x86_64__)
#include <cpuid.h>
#endif

namespace vgre {
namespace advanced {

VGREResult HardwareTokenManager::initFallbackEncrypted() {
    const char* overridePath = vgre_get_config("VGRE_TOKEN_FALLBACK_PATH");
    if (overridePath && overridePath[0] != '\0') {
        fallback_path_ = overridePath;
    } else {
#if defined(_WIN32)
        const char* appdata = vgre_get_config("APPDATA");
        if (appdata && appdata[0] != '\0') {
            fallback_path_ = std::string(appdata) + "\\vgre\\tokens.enc";
        } else {
            fallback_path_ = ".\\vgre\\tokens.enc";
        }
#else
        const char* home = vgre_get_config("HOME");
        if (home && home[0] != '\0') {
            fallback_path_ = std::string(home) + "/.vgre/tokens.enc";
        } else {
            // CI/sandbox fallback when HOME is unset.
            fallback_path_ = ".vgre/tokens.enc";
        }
#endif
    }
    
    std::string dir = fallback_path_.substr(0, fallback_path_.find_last_of("/\\"));
#if defined(_WIN32)
    if (!dir.empty()) {
        CreateDirectoryA(dir.c_str(), NULL);
    }
#else
    if (!dir.empty()) {
        mkdir(dir.c_str(), 0700);
    }
#endif
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::storeFallbackEncrypted(const std::string& service, const std::string& token) {
    std::map<std::string, std::string> tokens;
    std::ifstream infile(fallback_path_);
    if (infile.is_open()) {
        std::string line;
        while (std::getline(infile, line)) {
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string encrypted = line.substr(pos + 1);
                tokens[key] = encrypted;
            }
        }
        infile.close();
    }
    
    tokens[service] = encryptToken(token);
    
    std::ofstream outfile(fallback_path_);
    if (!outfile.is_open()) {
        return VGREResult::ERR_IO;
    }
    
    for (const auto& [key, value] : tokens) {
        outfile << key << "=" << value << "\n";
    }
    
    outfile.close();
    
#if !defined(_WIN32)
    chmod(fallback_path_.c_str(), 0600);
#endif
    
    return VGREResult::SUCCESS;
}

VGREResult HardwareTokenManager::getFallbackEncrypted(const std::string& service, std::string& outToken) {
    std::ifstream infile(fallback_path_);
    if (!infile.is_open()) {
        return VGREResult::ERR_AUTH_FAILED;
    }
    
    std::string line;
    while (std::getline(infile, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            if (key == service) {
                std::string encrypted = line.substr(pos + 1);
                outToken = decryptToken(encrypted);
                infile.close();
                return VGREResult::SUCCESS;
            }
        }
    }
    
    infile.close();
    return VGREResult::ERR_AUTH_FAILED;
}

VGREResult HardwareTokenManager::deleteFallbackEncrypted(const std::string& service) {
    std::map<std::string, std::string> tokens;
    std::ifstream infile(fallback_path_);
    if (!infile.is_open()) {
        return VGREResult::ERR_AUTH_FAILED;
    }
    
    std::string line;
    bool found = false;
    while (std::getline(infile, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            if (key == service) {
                found = true;
                continue;
            }
            std::string encrypted = line.substr(pos + 1);
            tokens[key] = encrypted;
        }
    }
    infile.close();
    
    if (!found) {
        return VGREResult::ERR_AUTH_FAILED;
    }
    
    std::ofstream outfile(fallback_path_);
    if (!outfile.is_open()) {
        return VGREResult::ERR_IO;
    }
    
    for (const auto& [key, value] : tokens) {
        outfile << key << "=" << value << "\n";
    }
    
    outfile.close();
    return VGREResult::SUCCESS;
}

namespace {
static std::string toHex(const uint8_t* data, size_t len) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i)
        oss << std::setw(2) << static_cast<int>(data[i]);
    return oss.str();
}

static bool fromHex(const std::string& hex, std::vector<uint8_t>& out) {
    if (hex.size() % 2 != 0) return false;
    out.resize(hex.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) {
        char buf[3] = {hex[2*i], hex[2*i+1], '\0'};
        char* end;
        long v = std::strtol(buf, &end, 16);
        if (end != buf + 2) return false;
        out[i] = static_cast<uint8_t>(v);
    }
    return true;
}

static void sha256CtrEncrypt(const uint8_t key[32], const uint8_t nonce[16],
                              const uint8_t* input, uint8_t* output, size_t len) {
    size_t offset = 0;
    uint32_t counter = 0;
    while (offset < len) {
        uint8_t block[32 + 16 + 4];
        memcpy(block, key, 32);
        memcpy(block + 32, nonce, 16);
        block[48] = (counter >> 24) & 0xff;
        block[49] = (counter >> 16) & 0xff;
        block[50] = (counter >>  8) & 0xff;
        block[51] =  counter        & 0xff;
        uint8_t ks[32];
        crypto::sha256(block, sizeof(block), ks);
        memset(block, 0, sizeof(block));
        size_t n = std::min(len - offset, static_cast<size_t>(32));
        for (size_t i = 0; i < n; ++i)
            output[offset + i] = input[offset + i] ^ ks[i];
        memset(ks, 0, sizeof(ks));
        offset += n;
        ++counter;
    }
}
}

std::array<uint8_t, 32> HardwareTokenManager::getMachineKey(const uint8_t dynamicSalt[32]) {
    std::string identity = "vgre_fallback_kdf_v3";

#if defined(_WIN32)
    // Windows: computer name + hardware product ID
    char hostname[256] = {}; DWORD sz = sizeof(hostname);
    GetComputerNameA(hostname, &sz);
    identity += hostname;
    char username[256] = {}; sz = sizeof(username);
    GetUserNameA(username, &sz);
    identity += username;
#else
    // Linux / macOS: layer multiple hardware-unique sources for entropy

    // Source 1: /etc/machine-id — 128-bit systemd-generated UUID, persistent
    // across reboots. Available on all modern Linux distributions.
    {
        std::ifstream f("/etc/machine-id");
        if (f) {
            std::string mid; f >> mid;
            if (!mid.empty()) identity += "mid:" + mid;
        }
    }

    // Source 2: SMBIOS product UUID from kernel DMI interface.
    // Hardware-unique: tied to the physical motherboard/VM hypervisor identity.
    {
        std::ifstream f("/sys/class/dmi/id/product_uuid");
        if (f) {
            std::string uuid; f >> uuid;
            if (!uuid.empty()) identity += "dmi:" + uuid;
        }
    }

#if defined(__x86_64__)
    // Source 3: CPUID brand string (48 bytes) — processor model + stepping.
    // Not globally unique but adds CPU-specific entropy and migrates with the CPU.
    {
        char brand[49] = {};
        unsigned int eax, ebx, ecx, edx;
        for (unsigned int leaf = 0x80000002u; leaf <= 0x80000004u; ++leaf) {
            if (__get_cpuid(leaf, &eax, &ebx, &ecx, &edx)) {
                unsigned int idx = (leaf - 0x80000002u) * 16u;
                memcpy(&brand[idx + 0],  &eax, 4);
                memcpy(&brand[idx + 4],  &ebx, 4);
                memcpy(&brand[idx + 8],  &ecx, 4);
                memcpy(&brand[idx + 12], &edx, 4);
            }
        }
        brand[48] = '\0';
        if (brand[0] != '\0') identity += "cpu:" + std::string(brand);
    }
#endif // __x86_64__

    // Source 4: hostname + USER — backward-compatible fallback. Always present.
    // Ensures key derivation succeeds in minimal container environments.
    {
        char hostname[256] = {};
        gethostname(hostname, sizeof(hostname) - 1);
        identity += "host:" + std::string(hostname);
        const char* user = vgre_get_config("USER");
        if (user && *user) identity += "user:" + std::string(user);
    }
#endif // !_WIN32

    // Mix the dynamic salt with machine identity using HMAC-SHA256.
    // This binds the derived key to BOTH the per-token random salt
    // AND the physical machine, preventing precomputation attacks.
    uint8_t salt[32];
    crypto::hmac_sha256(dynamicSalt, 32,
                        reinterpret_cast<const uint8_t*>(identity.data()),
                        identity.size(), salt);

    // PBKDF2 iteration count: default 600k (NIST SP 800-132 2025 minimum).
    // Operators can override via VGRE_PBKDF2_ITERATIONS for performance tuning,
    // mirroring the same pattern used in SecureChannel::deriveSessionKey().
    uint32_t pbkdf2Iters = static_cast<uint32_t>(crypto::kPBKDF2Iterations);
    const char *iterEnv = vgre_get_config("VGRE_PBKDF2_ITERATIONS");
    if (iterEnv && iterEnv[0] != '\0') {
        try {
            long val = std::stol(iterEnv);
            if (val >= 10000 && val <= 10000000) {
                pbkdf2Iters = static_cast<uint32_t>(val);
            }
        } catch (...) {
        }
    }

    std::array<uint8_t, 32> key{};
    crypto::pbkdf2_sha256(
        reinterpret_cast<const uint8_t*>("vgre_fallback_kdf"), 17,
        salt, sizeof(salt),
        pbkdf2Iters,
        key.data(), key.size());

    memset(salt, 0, sizeof(salt));
    return key;
}

std::array<uint8_t, 32> HardwareTokenManager::getMachineKey() {
    // Legacy path: derive a deterministic salt from machine identity
    // for backward compatibility with 3-field encrypted tokens.
    std::string identity = "vgre_fallback_kdf_v3";

#if defined(_WIN32)
    char hostname[256] = {}; DWORD sz = sizeof(hostname);
    GetComputerNameA(hostname, &sz);
    identity += hostname;
    char username[256] = {}; sz = sizeof(username);
    GetUserNameA(username, &sz);
    identity += username;
#else
    {
        std::ifstream f("/etc/machine-id");
        if (f) { std::string mid; f >> mid; if (!mid.empty()) identity += "mid:" + mid; }
    }
    {
        std::ifstream f("/sys/class/dmi/id/product_uuid");
        if (f) { std::string uuid; f >> uuid; if (!uuid.empty()) identity += "dmi:" + uuid; }
    }
#if defined(__x86_64__)
    {
        char brand[49] = {};
        unsigned int eax, ebx, ecx, edx;
        for (unsigned int leaf = 0x80000002u; leaf <= 0x80000004u; ++leaf) {
            if (__get_cpuid(leaf, &eax, &ebx, &ecx, &edx)) {
                unsigned int idx = (leaf - 0x80000002u) * 16u;
                memcpy(&brand[idx + 0],  &eax, 4);
                memcpy(&brand[idx + 4],  &ebx, 4);
                memcpy(&brand[idx + 8],  &ecx, 4);
                memcpy(&brand[idx + 12], &edx, 4);
            }
        }
        brand[48] = '\0';
        if (brand[0] != '\0') identity += "cpu:" + std::string(brand);
    }
#endif
    {
        char hostname[256] = {};
        gethostname(hostname, sizeof(hostname) - 1);
        identity += "host:" + std::string(hostname);
        const char* user = vgre_get_config("USER");
        if (user && *user) identity += "user:" + std::string(user);
    }
#endif

    // Compute a deterministic salt from identity (legacy behavior)
    std::string saltInput = "vgre_fallback_v3_pbkdf2:" + identity;
    uint8_t legacySalt[32];
    crypto::sha256(reinterpret_cast<const uint8_t*>(saltInput.data()),
                   saltInput.size(), legacySalt);

    auto key = getMachineKey(legacySalt);
    memset(legacySalt, 0, sizeof(legacySalt));
    return key;
}

std::string HardwareTokenManager::encryptToken(const std::string& plaintext) {
    // Generate a cryptographically random 32-byte salt per token.
    // This prevents precomputation attacks even if machine identity is known.
    uint8_t salt[32];
    crypto::random_bytes(salt, sizeof(salt));

    auto key = getMachineKey(salt);

    uint8_t nonce[16];
    crypto::random_bytes(nonce, sizeof(nonce));

    std::vector<uint8_t> cipher(plaintext.size());
    sha256CtrEncrypt(key.data(), nonce,
                     reinterpret_cast<const uint8_t*>(plaintext.data()),
                     cipher.data(), plaintext.size());

    std::vector<uint8_t> macInput(sizeof(nonce) + cipher.size());
    memcpy(macInput.data(), nonce, sizeof(nonce));
    if (!cipher.empty())
        memcpy(macInput.data() + sizeof(nonce), cipher.data(), cipher.size());
    uint8_t mac[32];
    crypto::hmac_sha256(key.data(), 32, macInput.data(), macInput.size(), mac);

    std::fill(key.begin(), key.end(), 0);

    // 4-field format: salt:nonce:cipher:mac (new dynamic-salt format)
    return toHex(salt, sizeof(salt)) + ":" +
           toHex(nonce, sizeof(nonce)) + ":" +
           toHex(cipher.data(), cipher.size()) + ":" +
           toHex(mac, sizeof(mac));
}

std::string HardwareTokenManager::decryptToken(const std::string& ciphertext) {
    // Detect format: 3-field (legacy) vs 4-field (new dynamic-salt)
    size_t sep1 = ciphertext.find(':');
    if (sep1 == std::string::npos) return "";
    size_t sep2 = ciphertext.find(':', sep1 + 1);
    if (sep2 == std::string::npos) return "";
    size_t sep3 = ciphertext.find(':', sep2 + 1);

    std::vector<uint8_t> nonce, cipher, storedMac;
    std::array<uint8_t, 32> key{};

    if (sep3 != std::string::npos) {
        // 4-field format: salt:nonce:cipher:mac
        std::vector<uint8_t> salt;
        if (!fromHex(ciphertext.substr(0, sep1), salt)           || salt.size() != 32) return "";
        if (!fromHex(ciphertext.substr(sep1 + 1, sep2 - sep1 - 1), nonce) || nonce.size() != 16) return "";
        if (!fromHex(ciphertext.substr(sep2 + 1, sep3 - sep2 - 1), cipher)) return "";
        if (!fromHex(ciphertext.substr(sep3 + 1), storedMac)    || storedMac.size() != 32) return "";
        key = getMachineKey(salt.data());
    } else {
        // 3-field format (legacy): nonce:cipher:mac
        if (!fromHex(ciphertext.substr(0, sep1), nonce)         || nonce.size() != 16) return "";
        if (!fromHex(ciphertext.substr(sep1 + 1, sep2 - sep1 - 1), cipher)) return "";
        if (!fromHex(ciphertext.substr(sep2 + 1), storedMac)    || storedMac.size() != 32) return "";
        key = getMachineKey();
    }

    std::vector<uint8_t> macInput(nonce.size() + cipher.size());
    memcpy(macInput.data(), nonce.data(), nonce.size());
    if (!cipher.empty())
        memcpy(macInput.data() + nonce.size(), cipher.data(), cipher.size());

    uint8_t expectedMac[32];
    crypto::hmac_sha256(key.data(), 32, macInput.data(), macInput.size(), expectedMac);

    if (!crypto::secure_compare(expectedMac, storedMac.data(), 32)) {
        std::fill(key.begin(), key.end(), 0);
        return "";
    }

    std::vector<uint8_t> plain(cipher.size());
    if (!cipher.empty())
        sha256CtrEncrypt(key.data(), nonce.data(), cipher.data(), plain.data(), cipher.size());

    std::fill(key.begin(), key.end(), 0);
    return std::string(reinterpret_cast<const char*>(plain.data()), plain.size());
}

} // namespace advanced
} // namespace vgre
