// Distributed primitives — consistent hashing + distributed lock + leader election.

#include "vgre/advanced/distributed.h"

#include <cstdio>
#include <map>
#include <string>

using namespace vgre::advanced;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    std::printf("=== Distributed primitives (hash ring / lock / election) ===\n");

    // ── Consistent hashing: even-ish spread + minimal remap on node removal ─
    {
        ConsistentHashRing ring;
        ring.addNode("n1"); ring.addNode("n2"); ring.addNode("n3"); ring.addNode("n4");
        const int K = 4000;
        std::map<std::string, std::string> before;
        std::map<std::string, int> spread;
        for (int i = 0; i < K; ++i) {
            std::string key = "key-" + std::to_string(i);
            std::string n = ring.getNode(key);
            before[key] = n;
            spread[n]++;
        }
        // each of 4 nodes gets roughly K/4 (allow ±40%)
        bool balanced = true;
        for (auto& kv : spread) if (kv.second < K / 4 * 0.6 || kv.second > K / 4 * 1.4) balanced = false;
        check("hash ring spreads keys ~evenly across 4 nodes", balanced && spread.size() == 4);

        // remove one node → only its keys move (others keep their node)
        ring.removeNode("n3");
        int moved = 0, kept = 0;
        for (int i = 0; i < K; ++i) {
            std::string key = "key-" + std::to_string(i);
            std::string n = ring.getNode(key);
            if (before[key] == "n3") ++moved; // these had to move
            else if (n == before[key]) ++kept;
            else ++moved;
        }
        // keys that weren't on n3 should overwhelmingly keep their node
        check("removing a node remaps minimally (non-n3 keys stable)", kept > (K - spread["n3"]) * 0.95);

        check("replica set returns distinct nodes", ring.getNodes("hot-key", 2).size() == 2);
    }

    // ── Distributed lock: mutual exclusion + TTL expiry + renew/release ──
    {
        DistributedLock lk;
        uint64_t t = 1000, ttl = 500;
        uint64_t tokA = lk.acquire("res", "A", t, ttl);
        check("A acquires lock (non-zero token)", tokA != 0);
        check("B cannot acquire held lock", lk.acquire("res", "B", t, ttl) == 0);
        check("holder is A", lk.holder("res", t) == "A");

        // A renews → still A's past original expiry
        check("A renews before expiry", lk.renew("res", "A", t + 400, ttl));
        check("B still blocked after renew", lk.acquire("res", "B", t + 600, ttl) == 0);

        // after expiry without renew, B takes over
        check("B acquires after TTL expiry", lk.acquire("res", "B", t + 2000, ttl) != 0);
        check("holder now B", lk.holder("res", t + 2000) == "B");

        // release frees it
        check("B releases", lk.release("res", "B"));
        check("free after release", lk.holder("res", t + 2000).empty());
        check("A re-acquires released lock", lk.acquire("res", "A", t + 2001, ttl) != 0);
    }

    // ── Leader election: single leader, failover on lease expiry ─────────
    {
        LeaderElection le(/*leaseMs*/300);
        uint64_t t = 0;
        check("first candidate becomes leader", le.campaign("node-a", t));
        check("second candidate is NOT leader while lease held", !le.campaign("node-b", t + 100));
        check("leader is node-a", le.leader(t + 100) == "node-a");

        // node-a stops renewing → after lease expiry node-b wins
        check("node-b wins after lease expiry", le.campaign("node-b", t + 500));
        check("leader is now node-b", le.leader(t + 500) == "node-b");

        // node-b resigns → node-a can take over immediately
        le.resign("node-b");
        check("after resignation a new leader can be elected", le.campaign("node-a", t + 600));
    }

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
