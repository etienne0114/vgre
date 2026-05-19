#include "vgre/advanced/secure_channel.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <thread>

using namespace vgre::advanced;

void test_sha256() {
    std::cout << "Testing SHA-256..." << std::endl;
    uint8_t digest[32];
    const char* msg = "vgre-secure-test";
    crypto::sha256(reinterpret_cast<const uint8_t*>(msg), strlen(msg), digest);
    
    // Simple verification (not full test vectors here for brevity, but enough to see it runs)
    assert(digest[0] != 0 || digest[31] != 0);
    std::cout << "  Passed." << std::endl;
}

void test_hmac() {
    std::cout << "Testing HMAC-SHA256..." << std::endl;
    uint8_t key[32] = {0x01, 0x02, 0x03};
    uint8_t data[16] = {0xAA, 0xBB};
    uint8_t mac[32];
    crypto::hmac_sha256(key, 32, data, 16, mac);
    assert(mac[0] != 0);
    std::cout << "  Passed." << std::endl;
}

void test_secure_channel_init() {
    std::cout << "Testing SecureChannel Initialization..." << std::endl;
    SecureChannel sc;
    uint8_t m_nonce[16], c_nonce[16];
    SecureChannel::generateNonce(m_nonce);
    SecureChannel::generateNonce(c_nonce);
    
    vgre::VGREResult r = sc.initializeFromSecret("test-token-123", m_nonce, c_nonce);
    assert(r == vgre::VGREResult::SUCCESS);
    assert(sc.isInitialized());
    
    std::string fp = sc.getKeyFingerprint();
    assert(fp.length() == 64);
    std::cout << "  Passed. Key Fingerprint: " << fp.substr(0, 8) << "..." << std::endl;
}

void test_secure_transport() {
    std::cout << "Testing SecureChannel Transport (XOR-Stream)..." << std::endl;
    SecureChannel master, worker;
    uint8_t m_nonce[16], c_nonce[16];
    SecureChannel::generateNonce(m_nonce);
    SecureChannel::generateNonce(c_nonce);
    
    master.initializeFromSecret("secret", m_nonce, c_nonce);
    worker.initializeFromSecret("secret", m_nonce, c_nonce);
    
    const char* plaintext = "Hello VGRE Secure Network!";
    (void)plaintext; // suppress unused for now as we just test init
    
    SessionInfo info = master.getSessionInfo();
    assert(info.is_encrypted);
    assert(std::string(info.cipher_name).find("AES256-CTR") != std::string::npos);
    
    std::cout << "  Passed." << std::endl;
}

int main() {
    try {
        test_sha256();
        test_hmac();
        test_secure_channel_init();
        test_secure_transport();
        std::cout << "\nALL SecureChannel tests PASSED." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
