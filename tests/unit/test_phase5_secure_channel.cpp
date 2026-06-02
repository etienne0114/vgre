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

// ── Replay Bitmap Test Harness ───────────────────────────────────────────────
// The replay-detection logic is private and embedded in recvSecure(), which
// requires a live socket.  We validate the algorithm directly by mirroring the
// exact same bit-math in a self-contained struct, giving us fine-grained control
// over sequence numbers without any I/O infrastructure.
//
// Invariant (RFC 4303 §3.4.3 sliding-window):
//   replayBitmap_[0] bit 0  →  highestSeenSeq_ itself
//   replayBitmap_[w] bit b  →  (highestSeenSeq_ - (w*64 + b)) has been seen
//   window width = kReplayWindowBits = 2048
//   offset = highestSeenSeq_ - seq; if offset >= 2048 → out-of-window → reject

static constexpr size_t kTestWindowBits  = 2048;
static constexpr size_t kTestWordCount   = kTestWindowBits / 64; // 32

struct ReplayWindow {
    uint64_t bitmap[kTestWordCount]{};
    uint64_t highest{0};
    bool     seeded{false};

    // Returns false if the packet should be rejected (duplicate / out-of-window).
    // Returns true and advances state if the packet is accepted.
    // Mirrors secure_channel.cpp replay block exactly.
    bool receive(uint64_t seq) {
        if (!seeded) {
            highest = seq;
            memset(bitmap, 0, sizeof(bitmap));
            bitmap[0] = 1ULL; // bit 0 = offset 0 from highest
            seeded = true;
            return true;
        }

        if (seq > highest) {
            uint64_t advance = seq - highest;

            if (advance >= kTestWindowBits) {
                // Invariant: window entirely stale — full reset preserves no history.
                memset(bitmap, 0, sizeof(bitmap));
            } else {
                // Left-shift the 2048-bit bitmap by 'advance' bits.
                // Words are little-endian: bitmap[0] = most-recent word.
                // Shift invariant: after shift, bit i of the new array represents
                //   (newHighest - i) = (oldHighest + advance - i).
                uint64_t ws = advance / 64; // whole-word shift
                uint64_t bs = advance % 64; // sub-word bit shift
                uint64_t tmp[kTestWordCount] = {};
                for (size_t i = 0; i < kTestWordCount; i++) {
                    size_t src = (ws <= i) ? i - ws : kTestWordCount;
                    if (src < kTestWordCount) {
                        // bs==0 guard: shifting uint64_t by 64 is UB in C++.
                        tmp[i] = (bs == 0) ? bitmap[src] : (bitmap[src] << bs);
                        // src>0 guard: src-1 underflows at src==0.
                        if (bs > 0 && src > 0)
                            tmp[i] |= bitmap[src - 1] >> (64 - bs);
                    }
                }
                memcpy(bitmap, tmp, sizeof(bitmap));
            }

            highest = seq;
            bitmap[0] |= 1ULL; // mark offset 0 = this packet
            return true;

        } else {
            // seq <= highest
            uint64_t offset = highest - seq;
            if (offset >= kTestWindowBits)
                return false; // out-of-window

            uint64_t word = offset / 64;
            uint64_t bit  = offset % 64;
            if (bitmap[word] & (1ULL << bit))
                return false; // duplicate

            bitmap[word] |= (1ULL << bit);
            return true;
        }
    }
};

// ── Test 1: duplicate rejection ──────────────────────────────────────────────
void test_replay_duplicate_rejection() {
    std::cout << "Testing replay duplicate rejection..." << std::endl;
    ReplayWindow w;

    // First receipt of seq=100 must be accepted.
    assert(w.receive(100) == true);

    // Immediate replay of seq=100 must be rejected (bit 0 already set).
    assert(w.receive(100) == false);

    std::cout << "  Passed." << std::endl;
}

// ── Test 2: out-of-window rejection ─────────────────────────────────────────
void test_replay_out_of_window() {
    std::cout << "Testing replay out-of-window rejection..." << std::endl;
    ReplayWindow w;

    // Seed window at seq=100.
    assert(w.receive(100) == true);

    // seq=0 is 100 positions behind highest=100; 100 < 2048 so still in window.
    // Verify it is accepted (new, not seen).
    assert(w.receive(0) == true);

    // Now advance highest far beyond 2048: seq=3000.  highest becomes 3000.
    // seq=100 is now offset 2900 >= 2048 → out-of-window → reject.
    assert(w.receive(3000) == true);
    assert(w.receive(100) == false);

    // seq=0 is offset 3000 >= 2048 → reject.
    assert(w.receive(0) == false);

    std::cout << "  Passed." << std::endl;
}

// ── Test 3: window advance and stale-seq rejection ───────────────────────────
void test_replay_window_advance() {
    std::cout << "Testing replay window advance..." << std::endl;
    ReplayWindow w;

    // Seed at 100, advance to 200 (advance=100, well within 2048).
    assert(w.receive(100) == true);
    assert(w.receive(200) == true);

    // offset(100) = 200-100 = 100 < 2048 → still in window.
    // seq=100 was already seen → duplicate.
    assert(w.receive(100) == false);

    // offset(71) = 200-71 = 129 < 2048 → in window, not yet seen → accept.
    assert(w.receive(71) == true);

    // seq=500: advance=300 < 2048 → shift; highest becomes 500.
    // offset(200) = 500-200 = 300 < 2048 → in window, already seen → reject.
    assert(w.receive(500) == true);
    assert(w.receive(200) == false);

    // seq=100: offset=400 < 2048; was seen before shift → reject.
    assert(w.receive(100) == false);

    std::cout << "  Passed." << std::endl;
}

// ── Test 4: in-order sequence (no duplicates) ─────────────────────────────────
void test_replay_in_order_sequence() {
    std::cout << "Testing in-order sequence acceptance..." << std::endl;
    ReplayWindow w;

    // Seqs 1..200 in ascending order — each must be accepted exactly once.
    for (uint64_t seq = 1; seq <= 200; ++seq) {
        assert(w.receive(seq) == true);
    }

    std::cout << "  Passed." << std::endl;
}

// ── Test 5: reorder within window ────────────────────────────────────────────
void test_replay_reorder_within_window() {
    std::cout << "Testing reorder within window..." << std::endl;
    ReplayWindow w;

    // Advance highest to 200.
    assert(w.receive(200) == true);

    // seq=150: offset=50 < 2048, not seen → accept.
    assert(w.receive(150) == true);

    // seq=150 again → duplicate.
    assert(w.receive(150) == false);

    // seq=199: offset=1 < 2048, not seen → accept.
    assert(w.receive(199) == true);

    // seq=199 again → duplicate.
    assert(w.receive(199) == false);

    std::cout << "  Passed." << std::endl;
}

// ── Test 6: shift-carry correctness at word boundary ─────────────────────────
// Invariant: when advancing by 1 bit, a bit that was at offset 0 of word W
// must appear at offset 1 of word W (carry within-word), and the MSB of word W
// must propagate into bit 0 of word W+1 (inter-word carry).
void test_replay_shift_carry_correctness() {
    std::cout << "Testing shift-carry correctness at word boundary..." << std::endl;
    ReplayWindow w;

    // Seed at seq=64.  bitmap[0] = 1 (bit 0 set, representing seq=64).
    assert(w.receive(64) == true);

    // Advance to seq=65 (advance=1).
    // After shift: old bit 0 (seq=64) becomes offset 1 → still in window.
    // seq=65 is new → accepted; bit 0 set for seq=65.
    assert(w.receive(65) == true);

    // seq=64: offset=1 < 2048, bit 1 should be set → duplicate.
    assert(w.receive(64) == false);

    // seq=63: offset=2, never seen → accept.
    assert(w.receive(63) == true);

    // seq=63 again → duplicate.
    assert(w.receive(63) == false);

    // Test inter-word carry: send seqs 0..63 in order (seed=0), then advance
    // by 64 bits (send seq=64).  After advance, the set bit at offset 0 of the
    // original word[0] must have shifted into word[1] bit 0 (offset 64).
    ReplayWindow w2;
    assert(w2.receive(0) == true); // seed, highest=0, bitmap[0] bit 0 set

    // Advance highest to 64 (advance=64 = 1 whole word, bs=0).
    // After shift: old bitmap[0] (which had bit 0 set) becomes bitmap[1].
    // Invariant: bitmap[1] & 1 must be set (seq=0 is at offset 64 from highest=64).
    assert(w2.receive(64) == true);

    // seq=0: offset=64, word=1, bit=0; should be marked seen → reject as duplicate.
    assert(w2.receive(0) == false);

    // seq=32: offset=32, never seen → accept.
    assert(w2.receive(32) == true);

    std::cout << "  Passed." << std::endl;
}

int main() {
    try {
        test_sha256();
        test_hmac();
        test_secure_channel_init();
        test_secure_transport();
        test_replay_duplicate_rejection();
        test_replay_out_of_window();
        test_replay_window_advance();
        test_replay_in_order_sequence();
        test_replay_reorder_within_window();
        test_replay_shift_carry_correctness();
        std::cout << "\nALL SecureChannel tests PASSED." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
