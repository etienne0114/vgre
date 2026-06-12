// Phase 3, Track P3-10 — prefix caching (content-addressed KV-block sharing).
// Verifies: identical-prefix sequences share ONE physical block (memory saving);
// refcount lifetime (block freed only on last release); copy-on-write isolates a
// diverging writer without corrupting the shared block.

#include "vgre/core/prefix_cache.h"

#include <cstdio>
#include <vector>

using namespace vgre::serving;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    printf("=== Prefix caching (content-addressed KV sharing) ===\n");
    const int nBlocks = 8, fpb = 16;   // 16 floats/block
    PrefixCache pc(nBlocks, fpb);

    // A shared prefix block (e.g. a system-prompt KV block).
    std::vector<float> prefix(fpb);
    for (int i = 0; i < fpb; ++i) prefix[i] = (float)(i + 1);
    uint64_t hp = hash_block(prefix.data(), fpb);

    // 1. Four sequences with the same prefix share ONE block.
    int b0 = pc.getOrCreate(hp, prefix.data());
    int b1 = pc.getOrCreate(hp, prefix.data());
    int b2 = pc.getOrCreate(hp, prefix.data());
    int b3 = pc.getOrCreate(hp, prefix.data());
    check("identical-prefix sequences map to the same physical block",
          b0 == b1 && b1 == b2 && b2 == b3 && b0 >= 0);
    check("only one block consumed for the shared prefix (4 refs)",
          pc.freeBlocks() == nBlocks - 1 && pc.refcount(b0) == 4);

    // 2. A different prefix gets its own block.
    std::vector<float> other(fpb, 0.5f);
    int bo = pc.getOrCreate(hash_block(other.data(), fpb), other.data());
    check("a distinct prefix allocates a separate block",
          bo != b0 && pc.freeBlocks() == nBlocks - 2);

    // 3. Refcount lifetime: block freed only when the last reference releases.
    pc.release(b0); pc.release(b1); pc.release(b2);   // 4 → 1
    check("shared block survives while references remain", pc.freeBlocks() == nBlocks - 2);
    pc.release(b3);                                    // 1 → 0, freed
    check("shared block freed on the last release", pc.freeBlocks() == nBlocks - 1);

    // 4. Copy-on-write: two sharers; one writer diverges → its own copy, the
    //    other's block is untouched.
    int s0 = pc.getOrCreate(hp, prefix.data());
    int s1 = pc.getOrCreate(hp, prefix.data());        // shared, refcount 2
    check("two sharers before COW", s0 == s1 && pc.refcount(s0) == 2);
    int w = pc.makeWritable(s0);                        // COW for sequence 0
    pc.mutableData(w)[0] = 999.0f;                      // diverge
    bool isolated = (w != s1) && (pc.data(s1)[0] == 1.0f) && (pc.data(w)[0] == 999.0f);
    check("COW gives the writer a private block; the sharer is untouched", isolated);
    check("original block now has one remaining reference", pc.refcount(s1) == 1);

    // 5. Sole owner writes in place (no wasted copy).
    int only = pc.getOrCreate(hash_block(other.data(), fpb) ^ 0x55, other.data());
    int onlyW = pc.makeWritable(only);
    check("sole-owner makeWritable writes in place (no new block)", onlyW == only);

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
